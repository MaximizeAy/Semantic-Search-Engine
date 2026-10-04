# Semantic Product Search Engine — Pipeline Design

> Status: **draft for review**
> Last updated: 2026-10-04

## 1. Goal

Take raw e-commerce product data (**name**, **description**) and:

1. **Generate tags** — a clean, canonical, relevance-ranked set of tags per product.
2. **Power semantic search** — let a user find products with a free-text, typo-tolerant query.

The engine combines four techniques in one pipeline: **fuzzy string matching**,
**NLP semantics**, **classification**, and **content filtering**.

## 2. Two execution paths

The stages are shared but run at two different times. Keeping these separate is the
single most important architectural decision — it stops build-time concerns (tag
generation, indexing) from leaking into the latency-sensitive query path.

| Path | When | Input | Output |
|------|------|-------|--------|
| **Offline (indexing)** | Build time, batch | product name + description | canonical tags + product embedding, written to the vector store |
| **Online (query)** | Serve time, per request | raw search query | ranked list of matching products |

```
  OFFLINE  (index the catalog)                    ONLINE  (answer a query)
  ─────────────────────────────                   ─────────────────────────
  product name + description                      raw query ("wireles earbuds")
          │                                                │
          ▼                                                ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  1. NORMALIZE            TextNormalizer  (built)                   │ shared
  └──────────────────────────────────────────────────────────────────┘
          │                                                │
          ▼                                                ▼
  ┌────────────────────────────┐             ┌───────────────────────────────┐
  │ 2. LINGUISTICS (spaCy)     │             │ 2'. FUZZY-CORRECT query terms │
  │    tokens, lemmas, POS,    │             │     RapidFuzz vs. vocabulary  │
  │    noun-chunks, entities   │             │     "wireles" → "wireless"    │
  └────────────────────────────┘             └───────────────────────────────┘
          │                                                │
          ▼                                                ▼
  ┌────────────────────────────┐             ┌───────────────────────────────┐
  │ 3. CANDIDATE TAGS          │             │ 3'. EMBED query               │
  │    noun-chunks + entities  │             │     (same encoder as offline) │
  │    + keyword scoring       │             └───────────────────────────────┘
  └────────────────────────────┘                          │
          │                                                ▼
          ▼                                   ┌───────────────────────────────┐
  ┌────────────────────────────┐             │ 6'. VECTOR SEARCH (pgvector)  │
  │ 4. FUZZY CANONICALIZE      │             │     cosine over product index │
  │    candidates → taxonomy   │             │     → ranked products         │
  └────────────────────────────┘             └───────────────────────────────┘
          │                                                │
          ▼                                                ▼
  ┌────────────────────────────┐                      results
  │ 5. EMBED (sentence-BERT)   │
  └────────────────────────────┘
          │
          ▼
  ┌────────────────────────────┐
  │ 6. CLASSIFY / SCORE        │
  └────────────────────────────┘
          │
          ▼
  ┌────────────────────────────┐
  │ 7. CONTENT FILTER          │
  └────────────────────────────┘
          │
          ▼
  write tags + embedding ───────►  Supabase pgvector
```

## 3. Stages

Each stage has one interface so stages can be run, tested, and swapped independently:

```python
class Stage(Protocol):
    def process(self, record: dict) -> dict: ...
```

| # | Stage | Responsibility | Technique | Tooling | Status |
|---|-------|----------------|-----------|---------|--------|
| 1 | Normalize | unicode NFKC, whitespace collapse, casefold | — | `TextNormalizer` | **built** |
| 2 | Linguistics | tokenize, lemmatize, POS, noun-chunks, NER → raw tag candidates | NLP | spaCy `en_core_web_sm` | todo |
| 3 | Candidate tags | score & rank keyword candidates | NLP semantics | `yake` or `keybert` | todo |
| 4 | Fuzzy canonicalize | collapse typos/variants onto a controlled tag vocabulary | fuzzy matching | `rapidfuzz` | todo |
| 5 | Embed | semantic vectors for product text (+ candidate tags) | NLP semantics | `sentence-transformers` — `BAAI/bge-m3` (1024-dim, multilingual), batch-encoded on Kaggle GPU | todo |
| 6 | Classify / score | product category + tag relevance ranking | classification | zero-shot via label embeddings (no training to start) | todo |
| 7 | Content filter | denylist, profanity/PII, dedupe, threshold, cap N | content filtering | `better_profanity` + fuzzy/semantic dedupe | todo |
| 8 | Index / search | persist & query vectors | — | **Supabase pgvector** | todo |

### Notes per stage

- **Stage 1 — Normalize.** Already implemented and clean. Shared verbatim by both paths.
- **Stage 2 — Linguistics.** Candidate tags start as noun-chunks + named entities; this is
  the "N" in NLP. spaCy model loads once and is reused.
- **Stage 3 — Candidates.** Rank the raw candidates so we keep the salient ones. YAKE is
  unsupervised and dependency-light; KeyBERT reuses the stage-5 encoder if we want
  embedding-based keywording.
- **Stage 4 — Fuzzy canonicalize.** A **controlled tag vocabulary** is the anchor, built from
  **two sources combined** (decided):
  - **Imported taxonomy** — a curated e-commerce taxonomy backbone (e.g. Google Product
    Taxonomy) provides clean canonical terms.
  - **Data-discovered** — frequent noun-chunks / entities mined from the catalog itself,
    capturing real, slang, and local-product terms the taxonomy will not have.

  The two are merged: discovered terms that are fuzzy/semantic near-duplicates of a taxonomy
  term fold into it as synonyms; genuinely novel frequent terms are added as new entries flagged
  `discovered`. RapidFuzz then maps noisy candidates (`"wireles"`, `"blutooth"`) onto the
  canonical tag. This stage also corrects typos on the query path.
- **Stage 5 — Embed.** One encoder, used in **both** paths, so query and product vectors live in
  the same space. **`BAAI/bge-m3`** → 1024-dim, multilingual (handles slang / local-language
  terms), no query/passage prefix needed. Product embeddings are **batch-encoded on Kaggle GPU**
  (offline). The model id is a config value, but note the dimension is baked into the pgvector
  column, so a model swap is a schema migration. **Constraint:** the online query path must embed
  queries with this *same* model — so wherever queries are served needs the model loaded (trivial
  on Kaggle; a CPU single-query encode is acceptable for `bge-m3`, a 7B model would require an
  always-on GPU).
- **Stage 6 — Classify / score.** Category assignment uses **both approaches combined** (decided),
  so we keep the slang/local flavour *and* a clean canonical category:
  - **Fixed list (zero-shot)** — embed a known category label set once; assign each product its
    nearest label. This is the clean "depiction" of the product's category.
  - **Discovered** — slang / local category-like terms mined from the data are preserved as
    secondary category tags rather than discarded, so local products keep their own vocabulary.

  Tag relevance = cosine(tag, product). A supervised classifier can replace the zero-shot step
  later without changing the interface.
- **Stage 7 — Content filter.** The safety gate: drop banned/unsafe tags, strip near-duplicates
  (fuzzy + semantic), apply a confidence threshold, cap the tag count. Always the last step
  before anything is persisted or returned.

## 4. The `NoiseGenerator` is offline-only

`NoiseGenerator` (already built) is **not** part of the serving pipeline. It belongs to the
**eval harness**: it turns clean product terms into realistic typos so we can build a test set
that measures whether stage 2' (fuzzy-correct) and stage 5 (embedding robustness) actually hold
up against messy real-world queries. Keeping it out of `semantic_search/` and in `eval/` makes
that boundary explicit and prevents it being wired into inference by accident.

## 5. Data model (Supabase / pgvector)

Vectors live in Supabase Postgres with the `vector` extension. This doubles as the product
database, so there is one source of truth.

```sql
create extension if not exists vector;

-- Controlled tag vocabulary (stage 4 anchor + stage 7 allowlist).
-- Hybrid: imported taxonomy terms + data-discovered terms, with synonym folding.
create table tag_vocabulary (
    id            bigint generated always as identity primary key,
    tag           text not null unique,
    category      text,
    source        text not null default 'taxonomy',   -- 'taxonomy' | 'discovered'
    canonical_id  bigint references tag_vocabulary(id), -- null = canonical; set = synonym of
    created_at    timestamptz default now()
);

-- Products + their embedding
create table products (
    id                 bigint generated always as identity primary key,
    external_id        text unique,
    name               text not null,
    description        text,
    category           text,                -- stage 6 zero-shot (canonical "depiction")
    discovered_categories text[],           -- stage 6 slang / local category terms, preserved
    tags               text[],              -- canonical tags, stage 4 + 7 output
    embedding          vector(1024),        -- stage 5, BAAI/bge-m3
    indexed_at         timestamptz default now()
);

-- Approximate-nearest-neighbour index for the online query path
create index on products using hnsw (embedding vector_cosine_ops);
```

Online search is a single RPC (cosine distance, `<=>`):

```sql
create or replace function search_products(query_embedding vector(1024), match_count int)
returns table (id bigint, name text, category text, tags text[], score float)
language sql stable as $$
    select p.id, p.name, p.category, p.tags,
           1 - (p.embedding <=> query_embedding) as score
    from products p
    order by p.embedding <=> query_embedding
    limit match_count;
$$;
```

> Note: embedding dimension is tied to the stage-5 model (1024 for `bge-m3` — changing the model
> changes the column type and index). HNSW vs IVFFlat index can be tuned once we know catalog size.

## 6. Proposed repository structure

Graduating from a single notebook to a small package keeps each stage testable and reusable,
with the notebook kept as a demo / walkthrough surface.

```
semantic_search/
  __init__.py
  normalize.py      # TextNormalizer            (move existing here)
  linguistics.py    # spaCy tokenizer / extractor
  candidates.py     # keyword candidate generation
  fuzzy.py          # RapidFuzz canonicalization
  embed.py          # sentence-transformers wrapper
  classify.py       # category + tag scoring
  filters.py        # content filtering + dedupe
  store.py          # Supabase pgvector client (index + search)
  pipeline.py       # TagPipeline (offline 1→7) ; SearchPipeline (online query)
eval/
  noise.py          # NoiseGenerator            (move existing here)
  build_testset.py  # uses noise.py to synthesise a typo eval set
  test_robustness.py
semantic-product-search-engine.ipynb   # demo / walkthrough
requirements.txt
.env.example        # SUPABASE_URL, SUPABASE_KEY (never commit real keys)
```

The spine is `TagPipeline` — a list of composable `Stage`s, each `process(record) -> record`,
so any stage can be run, tested, or swapped alone.

## 7. Dependencies

```
spacy            # stage 2
yake             # stage 3   (or keybert)
rapidfuzz        # stage 4
sentence-transformers  # stage 5  (loads BAAI/bge-m3)
better_profanity # stage 7
supabase         # stage 8  (python client)
```

Plus `python -m spacy download en_core_web_sm`. The `bge-m3` weights (~2.2 GB) download on first
use — pre-fetched in the Kaggle environment so the GPU indexing job starts clean.

## 8. Evaluation strategy

- Use `NoiseGenerator` to synthesise a typo test set from clean catalog terms.
- **Fuzzy correction (2'/4):** % of noisy terms mapped back to the correct canonical tag.
- **Retrieval (search path):** recall@k / MRR — does the clean product still rank for a noisy query?
- **Tag quality:** precision of generated tags against a hand-labelled sample.

## 9. Roadmap

1. Package skeleton + move `TextNormalizer` / `NoiseGenerator` into place.
2. Stage 2–3: spaCy linguistics + candidate tags.
3. Stage 4: fuzzy canonicalization + seed the tag vocabulary.
4. Stage 5: embedding wrapper.
5. Stage 8: Supabase schema + store client (index + search RPC).
6. Stage 6–7: classification + content filter.
7. End-to-end `TagPipeline` and `SearchPipeline`, wired in the demo notebook.
8. Eval harness with the noise-generated test set.

## 10. Decisions & open questions

**Decided:**

- **Tag vocabulary** — *hybrid*: imported e-commerce taxonomy backbone **+** data-discovered terms
  (keeps slang / local-product vocabulary), merged with synonym folding. (§3 stage 4, §5)
- **Category set** — *hybrid*: fixed-list zero-shot for the canonical category depiction **+**
  discovered slang/local category terms preserved alongside it. (§3 stage 6, §5)
- **Embedding model** — `BAAI/bge-m3` (1024-dim, multilingual), batch-encoded on Kaggle GPU. (§3 stage 5)

**Still open:**

- **Query-time embedding host** — where the online query encode runs (Kaggle-only demo vs. a
  deployed serving host). Determines whether the `bge-m3` model must be available outside Kaggle.
- **Taxonomy source** — which e-commerce taxonomy to import as the backbone (Google Product
  Taxonomy is the default candidate).
- **Index tuning** — HNSW vs IVFFlat, and the distance threshold below which a query returns
  "no good match".
```

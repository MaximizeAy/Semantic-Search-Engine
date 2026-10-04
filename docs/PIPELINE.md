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
| 5 | Embed | semantic vectors for product text (+ candidate tags) | NLP semantics | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) | todo |
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
- **Stage 4 — Fuzzy canonicalize.** A **controlled tag vocabulary** (taxonomy) is the anchor:
  RapidFuzz maps noisy candidates (`"wireles"`, `"blutooth"`) onto canonical tags
  (`"wireless"`, `"bluetooth"`). This is also where the query path corrects typos.
- **Stage 5 — Embed.** One encoder, used in **both** paths, so query and product vectors live
  in the same space. `all-MiniLM-L6-v2` → 384-dim, fast, CPU-friendly. Model id is a config
  value so we can upgrade later without touching call sites.
- **Stage 6 — Classify / score.** Start training-free: embed category labels once, assign each
  product to its nearest label (zero-shot). Tag relevance = cosine(tag, product). A supervised
  classifier can replace this later without changing the interface.
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

-- Controlled tag vocabulary (stage 4 anchor + stage 7 allowlist)
create table tag_vocabulary (
    id          bigint generated always as identity primary key,
    tag         text not null unique,
    category    text,
    created_at  timestamptz default now()
);

-- Products + their embedding
create table products (
    id          bigint generated always as identity primary key,
    external_id text unique,
    name        text not null,
    description text,
    category    text,                       -- assigned in stage 6
    tags        text[],                     -- canonical tags, stage 4 + 7 output
    embedding   vector(384),                -- stage 5, all-MiniLM-L6-v2
    indexed_at  timestamptz default now()
);

-- Approximate-nearest-neighbour index for the online query path
create index on products using hnsw (embedding vector_cosine_ops);
```

Online search is a single RPC (cosine distance, `<=>`):

```sql
create or replace function search_products(query_embedding vector(384), match_count int)
returns table (id bigint, name text, category text, tags text[], score float)
language sql stable as $$
    select p.id, p.name, p.category, p.tags,
           1 - (p.embedding <=> query_embedding) as score
    from products p
    order by p.embedding <=> query_embedding
    limit match_count;
$$;
```

> Decisions to confirm: embedding dimension is tied to the stage-5 model
> (384 for MiniLM — changing the model changes the column type and index). HNSW vs IVFFlat
> index can be tuned once we know catalog size.

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
sentence-transformers  # stage 5
better_profanity # stage 7
supabase         # stage 8  (python client)
```

Plus `python -m spacy download en_core_web_sm`.

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

## 10. Open questions

- **Tag vocabulary source** — seed it from the catalog's own frequent noun-chunks, or import an
  existing e-commerce taxonomy?
- **Embedding model** — start with `all-MiniLM-L6-v2` (fast, 384-dim); revisit if quality is short.
- **Category set** — fixed known list (enables zero-shot), or discovered from data?
- **Index tuning** — HNSW vs IVFFlat, and the distance threshold for "no good match".
```

# Semantic Product Search Engine

Turns raw e-commerce product data (name + description) into **canonical tags**
and **semantic-search-ready embeddings**, combining fuzzy string matching, NLP
semantics, classification, and content filtering into one pipeline.

Full design: [`docs/PIPELINE.md`](docs/PIPELINE.md).

## Layout

```
semantic_search/      # the engine (one module per pipeline stage)
  normalize.py        # 1. unicode / whitespace / case
  linguistics.py      # 2. spaCy tokens, noun-chunks, entities
  candidates.py       # 3. keyword candidate scoring (YAKE)
  fuzzy.py            # 4. RapidFuzz canonicalization + query typo correction
  embed.py            # 5. sentence-transformers (BAAI/bge-m3, 1024-dim)
  classify.py         # 6. zero-shot category + tag relevance
  filters.py          # 7. content filter + dedupe
  store.py            # 8. Supabase pgvector client
  pipeline.py         # TagPipeline (offline) + SearchPipeline (online)
eval/                 # typo robustness harness (offline-only)
  noise.py            # synthetic typo generator
  build_testset.py
  test_robustness.py
db/schema.sql         # Supabase / pgvector schema — run in your project
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env            # then fill in your Supabase project details
```

Supabase (you can reuse an existing project — the engine lives in its own
`semantic_search` schema, isolated from your `public` tables):

1. Run [`db/schema.sql`](db/schema.sql) in the SQL editor.
2. Project Settings → API → **Exposed schemas**: add `semantic_search`.
3. Use the **service-role key** for indexing (bypasses RLS).

## Usage

Offline — generate tags + embedding for a product:

```python
from semantic_search.fuzzy import TagVocabulary
from semantic_search.pipeline import build_tag_pipeline

vocab = TagVocabulary.from_taxonomy(["wireless", "bluetooth", "headphones"])
categories = ["Electronics", "Apparel", "Home & Kitchen"]

pipeline = build_tag_pipeline(vocab, categories)
record = pipeline.run_text("Wireless Bluetooth Headphones",
                           "Over-ear, noise cancelling")
print(record["tags"], record["category"])
```

Online — search (needs a populated Supabase project):

```python
from semantic_search.store import SupabaseVectorStore
from semantic_search.pipeline import build_search_pipeline

store = SupabaseVectorStore()                    # reads SUPABASE_URL / SUPABASE_KEY
search = build_search_pipeline(store, vocab)
results = search.search("wireles earbuds", k=5)  # typo-tolerant
```

Embeddings are meant to be batch-encoded on **Kaggle GPU** for the catalog; a
single query encodes fine on CPU at serve time.

## Service (two usages)

The engine runs as an HTTP microservice (`api/main.py`) with two flows:

1. **Tag on create** — `POST /tag` when a product/service is created: generates
   tags + category + embedding, upserts the embedding into
   `semantic_search.products`, and writes tags back to `public.products.tags`.
2. **Realtime search** — `POST /search`: typo-tolerant semantic search over the index.

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000

# tag on create
curl -X POST localhost:8000/tag -H 'content-type: application/json' \
  -d '{"external_id":"<uuid>","name":"Wireless Earbuds","description":"noise cancelling"}'

# search
curl -X POST localhost:8000/search -H 'content-type: application/json' \
  -d '{"query":"wireles earbuds","k":5}'
```

The service loads `bge-m3` lazily on first request; hit `GET /health?warm=1` at
boot to preload. Run it on a host with the model available (the same place query
embedding happens).

## Keeping tags fresh (weekly rebuild)

`scripts/index_catalog.py` re-tags and re-indexes the catalog:

```bash
python -m scripts.index_catalog --rebuild            # full rebuild
python -m scripts.index_catalog --since 2026-10-01T00:00:00Z   # incremental
```

A scheduled **GitHub Action** (`.github/workflows/weekly-rebuild.yml`) runs it
weekly (needs `SUPABASE_URL` / `SUPABASE_KEY` repo secrets; active once merged to
`main`). Alternatively cron the command, or `POST /rebuild` on the service.

> What a rebuild refreshes: `bge-m3` is a fixed pretrained model — it does **not**
> learn over time. The rebuild re-generates **tags** (so new/changed listings and
> the growing discovered-term vocabulary are reflected) and **re-embeds** changed
> products. "Up to date" = catalog + vocabulary freshness, not model retraining.

## Status

Pipeline runs end-to-end (validated on Kaggle GPU). Service + weekly rebuild in
place. See `docs/PIPELINE.md` §9 for the roadmap and §10 for open decisions.

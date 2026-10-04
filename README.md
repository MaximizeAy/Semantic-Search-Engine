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

Create a dedicated Supabase project, enable `pgvector`, and run
[`db/schema.sql`](db/schema.sql) in its SQL editor.

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

## Status

Pipeline skeleton with working stages. See `docs/PIPELINE.md` §9 for the roadmap
and §10 for open decisions.

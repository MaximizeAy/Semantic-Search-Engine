# Deployment options

> Status: **decision pending** — written up for later; no deploy code built yet.
> Last updated: 2026-10-12

## The core constraint

The service has two kinds of work with very different runtime needs:

- **NLP tagging** (spaCy, YAKE, RapidFuzz) — Python-only, CPU, light.
- **Embedding** (`bge-m3`) — historically PyTorch + a 2.2 GB model, the heavy part.

**Plain Cloudflare Workers (the V8/edge runtime) cannot host this.** No PyTorch, no
2.2 GB model, tight memory/CPU limits, and the Python NLP libraries don't run there.
There is no slice of the current FastAPI app that "fits" on a Worker as-is.

## The unlock: Workers AI already hosts `bge-m3`

Cloudflare Workers AI serves `@cf/baai/bge-m3` — the *exact* model we chose,
multilingual, callable from a Worker binding or REST
(`/accounts/{id}/ai/run/@cf/baai/bge-m3`), ~$0.012 / M input tokens.

Since embedding is the only heavy step, delegating it to Workers AI means the Python
side no longer needs `torch` or the model — it shrinks to a light CPU service. And the
**search flow needs no Python at all**.

Sources:
- https://www.llmreference.com/model/bge-m3/cloudflare
- https://developers.cloudflare.com/changelog/2025-03-17-new-workers-ai-models/
- https://developers.cloudflare.com/containers/platform-details/limits/

## How the two flows decompose

```
SEARCH (flow 2)                         TAG ON CREATE (flow 1)
──────────────                          ──────────────────────
query                                   name + description
  │ normalize (trivial JS)                │ normalize + spaCy/YAKE/RapidFuzz  ← Python
  │ embed  ──► Workers AI bge-m3          │ embed ──► Workers AI bge-m3
  │ pgvector RPC (Supabase)               │ upsert + merge tags (Supabase)
  ▼                                       ▼
ranked products                         tags written back
```

Search = pure Worker. Tagging = light Python + Workers AI for the embed.

## Options

### Option 1 — Cloudflare-native (recommended if committed to Cloudflare)
- **Search**: a Cloudflare Worker (normalize → Workers AI embed → Supabase RPC). No Python.
- **Tagging**: a light Python container (spaCy/YAKE/RapidFuzz, **no torch, no model**) on
  Cloudflare Containers or any small host, calling Workers AI for the embedding.
- **All embedding (index + query) standardized on Workers AI `bge-m3`** so vectors share
  one space.
- Effort: **medium** — refactor `embed.py` to a Workers-AI backend; route the Kaggle/
  backfill indexing through Workers AI too; re-verify vectors line up; write the Worker.
- Pros: lightest to run, one vendor, no GPU/torch anywhere, cheap per-token.
- Cons: vendor lock to Workers AI; must re-standardize + re-test the embedding path;
  drops the local `sentence-transformers` path.

### Option 2 — Container host + Worker proxy (least effort, keeps current code)
- Deploy the current FastAPI + `bge-m3` **unchanged** to a container PaaS
  (Render / Railway / Fly.io / Cloud Run).
- Optionally keep a thin Cloudflare Worker that proxies `/tag` and `/search` so the public
  API stays on your Cloudflare domain.
- Effort: **low** — add a `Dockerfile` + start command (+ a ~20-line proxy Worker).
- Pros: zero code change; the reviewed/tested service runs as-is.
- Cons: needs a ~4 GB-RAM container off Cloudflare; a second platform to manage.

### Option 3 — Cloudflare Containers, current image
- Run the full `torch` + `bge-m3` image on Cloudflare Containers (public beta).
- Effort: **low–medium** — `Dockerfile` + `wrangler.toml` with a container binding.
- Pros: stays entirely on Cloudflare; no code change.
- Cons: large image, slow cold starts (loading 2.2 GB), instance memory caps
  (predefined up to ~12 GiB); beta. Verify current limits before relying on it.

## Recommendation

If you're committed to Cloudflare, **Option 1** is the best end state — nothing heavy to
run, search is a cheap Worker, tagging is a small container. The cost is a medium refactor
of the embedding layer and re-testing vector consistency.

If you want the shortest path to *something live right now*, **Option 2** ships the exact
service we already built and reviewed, with a Worker proxy so your public endpoint still
lives on Cloudflare.

## What I'd build, per option (when you decide)

| Option | Artifacts I'd add |
|--------|-------------------|
| 1 | Worker (search) + Workers-AI embed backend in `embed.py` + indexing switch + `Dockerfile` for the light tagging container + tests |
| 2 | `Dockerfile` + start command + minimal proxy Worker + deploy notes |
| 3 | `Dockerfile` + `wrangler.toml` (container binding) + deploy notes |

## Chosen path: $0 (Option 1, on-create dispatch) — BUILT

Decided and implemented. No always-on server; everything on free tiers.

```
SEARCH (real-time)                      TAGGING (on product-create)
──────────────────                      ───────────────────────────
Cloudflare Worker (free plan)           GitHub Actions (free for public repo)
  normalize (JS)                          repository_dispatch "product-created"
  embed → Workers AI bge-m3                 { external_id }
  Supabase search_products RPC             runs scripts/tag_one.py
  (anon key, security-definer)             embed → Workers AI bge-m3 (REST)
                                           upsert index + merge tags back
```

**Cost:** Workers (100k req/day free) + Workers AI (10k neurons/day free) +
GitHub Actions (free, public repo) + Supabase (existing free tier) = **$0** at
low volume. No Cloudflare Containers → no $5 Workers-Paid floor.

**What was built:**
- `semantic_search/embed.py` — `EMBEDDING_BACKEND=workers_ai` backend (REST, no
  torch), L2-normalized to match the sentence-transformers path.
- `workers/search/` — the search Worker (`wrangler.toml` + `src/index.js`), uses
  the **anon** key via the security-definer RPC.
- `.github/workflows/tag-on-create.yml` — `repository_dispatch` job running
  `scripts/tag_one.py` with `requirements-tagging.txt` (light, no torch).
- `scripts/tag_one.py` + `service.tag_by_id()` + `store.get_source_product()`.

**Setup (when going live):**
1. Deploy the Worker: `cd workers/search && wrangler secret put SUPABASE_URL` /
   `SUPABASE_KEY` (anon) → `wrangler deploy`.
2. Repo secrets for the Action: `SUPABASE_URL`, `SUPABASE_KEY` (service-role),
   `CF_ACCOUNT_ID`, `CF_AI_TOKEN`. Merge to `main` so dispatch is active.
3. From the marketplace backend (or a Supabase DB webhook) on product-create,
   POST a `repository_dispatch` of type `product-created` with
   `client_payload: { external_id }`.
4. Backfill once: `EMBEDDING_BACKEND=workers_ai python -m scripts.index_catalog`
   (spread over days if it would exceed 10k neurons/day).

**Key-safety:** Worker uses the **anon** key (search via security-definer RPC);
the Action uses the **service-role** key (writes). Never ship service-role to
the Worker.

## Open sub-decisions (for Option 1)
- Confirm Workers AI `bge-m3` output is L2-normalized / matches our indexing vectors
  (one quick empirical check before committing the catalog).
- Where the light tagging container runs (Cloudflare Containers vs a small always-on host).
- Whether tag-on-create is called synchronously by the marketplace backend or via a queue.

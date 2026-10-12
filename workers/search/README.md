# Search Worker

Realtime semantic search (flow 2) as a Cloudflare Worker — no Python, no model
to host. Runs on the **free** Workers plan (Workers AI has a free daily
allocation).

## Deploy

```bash
cd workers/search
npm i -g wrangler          # if needed
wrangler secret put SUPABASE_URL   # https://<ref>.supabase.co
wrangler secret put SUPABASE_KEY   # the ANON key (NOT service-role)
wrangler deploy
```

`wrangler.toml` already binds Workers AI (`AI`) and sets `SUPABASE_SCHEMA` /
`EMBED_MODEL`.

## Call

```bash
curl -X POST https://semantic-search.<your-subdomain>.workers.dev/search \
  -H 'content-type: application/json' \
  -d '{"query":"wireles earbuds","k":5}'
```

Returns `{ query, results: [{ external_id, name, category, tags, score }] }`.

## Notes

- Uses the **anon** key on purpose: search goes through the `search_products`
  security-definer RPC, so anon is sufficient and safe to expose at the edge.
  The service-role key is only used by the tagging job (writes), never here.
- Query embedding is L2-normalized to match the indexing vectors.
- Query-side fuzzy typo-correction (the Python `SearchPipeline` step) is omitted
  here for v1; `bge-m3` is fairly typo-robust. Revisit if recall on very noisy
  queries is poor.
- Requires `search_products` to return `external_id` (it does — see
  `db/schema.sql`) so results link back to `public.products`.

// Realtime semantic search as a Cloudflare Worker (flow 2).
//
// POST /search  { "query": "wireles earbuds", "k": 5 }
//   -> normalize -> Workers AI bge-m3 embed -> Supabase search_products RPC
//
// No Python, no model to host. Uses the ANON Supabase key (search goes through
// the security-definer RPC). Embedding is L2-normalized to match the indexing
// vectors produced by semantic_search.embed.

export default {
  async fetch(request, env) {
    if (request.method !== "POST") return json({ error: "POST only" }, 405);
    if (new URL(request.url).pathname !== "/search") return json({ error: "not found" }, 404);

    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "invalid JSON body" }, 400);
    }
    const query = (body.query ?? "").toString();
    const k = Math.min(Math.max(parseInt(body.k ?? 10, 10) || 10, 1), 100);
    if (!query.trim()) return json({ error: "query is required" }, 400);

    // normalize: NFKC, collapse whitespace, casefold (mirrors TextNormalizer)
    const norm = query.normalize("NFKC").replace(/\s+/g, " ").trim().toLowerCase();

    // embed via Workers AI
    let emb;
    try {
      emb = await env.AI.run(env.EMBED_MODEL || "@cf/baai/bge-m3", { text: [norm] });
    } catch (e) {
      return json({ error: "embedding failed", detail: String(e) }, 502);
    }
    const vec = emb?.data?.[0] ?? emb?.result?.data?.[0];
    if (!vec) return json({ error: "no embedding returned" }, 502);
    const literal = "[" + l2normalize(vec).join(",") + "]";

    // vector search via the pgvector RPC in the semantic_search schema
    const schema = env.SUPABASE_SCHEMA || "semantic_search";
    const resp = await fetch(`${env.SUPABASE_URL}/rest/v1/rpc/search_products`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        apikey: env.SUPABASE_KEY,
        Authorization: `Bearer ${env.SUPABASE_KEY}`,
        "Accept-Profile": schema,
        "Content-Profile": schema,
      },
      body: JSON.stringify({ query_embedding: literal, match_count: k }),
    });
    if (!resp.ok) {
      return json({ error: "search RPC failed", status: resp.status, detail: await resp.text() }, 502);
    }
    return json({ query, results: await resp.json() });
  },
};

function l2normalize(v) {
  const norm = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1;
  return v.map((x) => x / norm);
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

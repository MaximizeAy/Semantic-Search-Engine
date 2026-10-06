-- Semantic Product Search Engine — Supabase / pgvector schema
--
-- This installs into a DEDICATED `semantic_search` schema so it never collides
-- with your app's existing `public` tables (products, orders, etc.).
--
-- Setup (run once, in the Supabase SQL editor):
--   1. Run this whole file.
--   2. Project Settings -> API -> "Exposed schemas": add `semantic_search`.
--   3. Index/search from the engine using the SERVICE-ROLE key (bypasses RLS).
--
-- Embedding dimension is 1024 to match BAAI/bge-m3. Changing the embedding
-- model means changing vector(1024) here and reindexing.

create schema if not exists semantic_search;

-- pgvector (shared across the database; no-op if already enabled via dashboard).
create extension if not exists vector;

-- Let the Supabase API roles reach this schema.
grant usage on schema semantic_search to anon, authenticated, service_role;

-- Controlled tag vocabulary (stage 4 anchor + stage 7 allowlist).
-- Hybrid: imported taxonomy terms + data-discovered terms, with synonym folding.
create table if not exists semantic_search.tag_vocabulary (
    id            bigint generated always as identity primary key,
    tag           text not null unique,
    category      text,
    source        text not null default 'taxonomy',                  -- 'taxonomy' | 'discovered'
    canonical_id  bigint references semantic_search.tag_vocabulary(id), -- null = canonical; set = synonym of
    created_at    timestamptz default now()
);

-- Products + their embedding.
create table if not exists semantic_search.products (
    id                    bigint generated always as identity primary key,
    external_id           text unique,
    name                  text not null,
    description           text,
    category              text,          -- stage 6 zero-shot (canonical "depiction")
    discovered_categories text[],        -- stage 6 slang / local category terms, preserved
    tags                  text[],        -- canonical tags, stage 4 + 7 output
    embedding             vector(1024),  -- stage 5, BAAI/bge-m3
    indexed_at            timestamptz default now()
);

-- Approximate-nearest-neighbour index for the online query path.
create index if not exists products_embedding_hnsw
    on semantic_search.products using hnsw (embedding vector_cosine_ops);

-- RLS on: writes happen with the service-role key (bypasses RLS); reads go
-- through the search_products function below, which is security definer.
alter table semantic_search.tag_vocabulary enable row level security;
alter table semantic_search.products        enable row level security;

grant all on all tables    in schema semantic_search to service_role;
grant all on all sequences in schema semantic_search to service_role;

-- Online search RPC: cosine similarity (1 - distance). security definer so a
-- client (authenticated/anon) can search without a per-row RLS policy.
create or replace function semantic_search.search_products(
    query_embedding vector(1024),
    match_count int
)
returns table (id bigint, name text, category text, tags text[], score float)
language sql
stable
security definer
set search_path = semantic_search
as $$
    select p.id, p.name, p.category, p.tags,
           1 - (p.embedding <=> query_embedding) as score
    from semantic_search.products p
    order by p.embedding <=> query_embedding
    limit match_count;
$$;

grant execute on function semantic_search.search_products(vector, int)
    to anon, authenticated, service_role;

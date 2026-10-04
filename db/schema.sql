-- Semantic Product Search Engine — Supabase / pgvector schema
-- Run this in your Supabase project's SQL editor (see docs/PIPELINE.md §5).
--
-- Embedding dimension is 1024 to match BAAI/bge-m3. Changing the embedding
-- model means changing vector(1024) here and reindexing.

create extension if not exists vector;

-- Controlled tag vocabulary (stage 4 anchor + stage 7 allowlist).
-- Hybrid: imported taxonomy terms + data-discovered terms, with synonym folding.
create table if not exists tag_vocabulary (
    id            bigint generated always as identity primary key,
    tag           text not null unique,
    category      text,
    source        text not null default 'taxonomy',      -- 'taxonomy' | 'discovered'
    canonical_id  bigint references tag_vocabulary(id),   -- null = canonical; set = synonym of
    created_at    timestamptz default now()
);

-- Products + their embedding.
create table if not exists products (
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
    on products using hnsw (embedding vector_cosine_ops);

-- Online search RPC: cosine similarity (1 - distance).
create or replace function search_products(query_embedding vector(1024), match_count int)
returns table (id bigint, name text, category text, tags text[], score float)
language sql stable as $$
    select p.id, p.name, p.category, p.tags,
           1 - (p.embedding <=> query_embedding) as score
    from products p
    order by p.embedding <=> query_embedding
    limit match_count;
$$;

-- On-create dispatch: when a product is inserted into public.products, fire a
-- GitHub repository_dispatch so the tag-on-create Action tags it (near-real-time).
--
-- This is the glue between the marketplace (Supabase) and the tagging job
-- (GitHub Actions). Run it in the Supabase SQL editor AFTER db/schema.sql.
--
-- Prereqs:
--   1. pg_net extension (ships with Supabase).
--   2. A GitHub token with permission to create dispatch events on this repo
--      (classic PAT with `repo` scope, or a fine-grained token limited to this
--      repo with Contents: read/write), stored in Supabase Vault:
--
--        select vault.create_secret('<GITHUB_TOKEN>', 'github_dispatch_token');
--
-- Security: the token lives only in Vault; this function reads it server-side.
-- net.http_post is async (fire-and-forget), so it does not slow the insert.

create extension if not exists pg_net;

create or replace function semantic_search.notify_product_created()
returns trigger
language plpgsql
security definer
set search_path = public, extensions, vault
as $$
declare
    gh_token text;
begin
    select decrypted_secret into gh_token
    from vault.decrypted_secrets
    where name = 'github_dispatch_token';

    if gh_token is null then
        raise warning 'github_dispatch_token not set in Vault; skipping dispatch';
        return new;
    end if;

    perform net.http_post(
        url     := 'https://api.github.com/repos/MaximizeAy/Semantic-Search-Engine/dispatches',
        headers := jsonb_build_object(
            'Authorization', 'Bearer ' || gh_token,
            'Accept',        'application/vnd.github+json',
            'User-Agent',    'supabase-product-dispatch',
            'Content-Type',  'application/json'
        ),
        body    := jsonb_build_object(
            'event_type',     'product-created',
            'client_payload', jsonb_build_object('external_id', new.id::text)
        )
    );
    return new;
end;
$$;

-- Fire once per newly inserted product.
drop trigger if exists product_created_dispatch on public.products;
create trigger product_created_dispatch
    after insert on public.products
    for each row
    execute function semantic_search.notify_product_created();

-- Notes:
-- * Bulk imports will fire one dispatch per row. For a large backfill, either
--   disable this trigger first and run `scripts/index_catalog.py` instead, or
--   expect many Action runs.
-- * To also re-tag on edits, add an `after update of name, description` trigger
--   using the same function.

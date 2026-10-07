"""Backfill / rebuild the search index from the marketplace catalog.

Reads products from public.products, tags + embeds each, upserts the embedding
into semantic_search.products, and writes tags back to public.products.tags.

Usage:
    python -m scripts.index_catalog                 # index active products
    python -m scripts.index_catalog --rebuild       # full rebuild (same as above)
    python -m scripts.index_catalog --since 2026-10-01T00:00:00Z   # incremental
    python -m scripts.index_catalog --all           # include inactive listings
    python -m scripts.index_catalog --no-source-tags # don't touch public.products.tags

Requires SUPABASE_URL / SUPABASE_KEY (service-role key) in the environment.
"""

from __future__ import annotations

import argparse
import time

from semantic_search.service import TaggingService


def main() -> None:
    parser = argparse.ArgumentParser(description="Index / rebuild the catalog.")
    parser.add_argument("--rebuild", action="store_true", help="full rebuild (default behaviour)")
    parser.add_argument("--since", default=None, help="ISO timestamp; only reindex products changed after it")
    parser.add_argument("--all", action="store_true", help="include inactive listings")
    parser.add_argument("--no-source-tags", action="store_true", help="do not write tags back to public.products")
    args = parser.parse_args()

    service = TaggingService()
    started = time.time()
    count = service.rebuild(
        only_active=not args.all,
        since=args.since,
        write_source_tags=not args.no_source_tags,
    )
    elapsed = time.time() - started
    scope = f"since {args.since}" if args.since else "full catalog"
    print(f"Indexed {count} products ({scope}) in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

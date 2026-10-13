"""Tag a single product by id — the entry point for the on-create dispatch job.

Reads the product from public.products, runs the tagging pipeline (embedding via
whatever EMBEDDING_BACKEND is set), upserts the embedding into the search index,
and merges the generated tags back into public.products.tags.

Usage:
    python -m scripts.tag_one --external-id <uuid>

Env: SUPABASE_URL, SUPABASE_KEY (service-role). For the $0 path also set
EMBEDDING_BACKEND=workers_ai, CF_ACCOUNT_ID, CF_AI_TOKEN.
"""

from __future__ import annotations

import argparse
import sys

from semantic_search.service import TaggingService


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag one product by id.")
    parser.add_argument("--external-id", required=True, help="public.products.id (uuid)")
    args = parser.parse_args()

    rec = TaggingService().tag_by_id(args.external_id)
    if rec is None:
        print(f"No product found for {args.external_id}", file=sys.stderr)
        sys.exit(1)
    print(f"Tagged {args.external_id}: category={rec['category']} tags={rec['tags']}")


if __name__ == "__main__":
    main()

"""Stage 8 — Vector store (Supabase / pgvector).

Thin client over a Supabase project with the ``vector`` extension. Reads
credentials from the environment (``SUPABASE_URL`` / ``SUPABASE_KEY``) so no
secrets live in code. The matching schema is in ``db/schema.sql`` — run it in
your Supabase project before using this client.

The ``supabase`` package is imported lazily.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional

from .types import Record


class SupabaseVectorStore:
    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        schema: Optional[str] = None,
        table: str = "products",
        search_fn: str = "search_products",
        source_table: str = "products",
        source_schema: str = "public",
    ):
        self.url = url or os.environ.get("SUPABASE_URL")
        self.key = key or os.environ.get("SUPABASE_KEY")
        self.schema = schema or os.environ.get("SUPABASE_SCHEMA", "semantic_search")
        self.table = table
        self.search_fn = search_fn
        # The marketplace's own products table (read catalog, write tags back).
        self.source_table = source_table
        self.source_schema = source_schema
        self._client = None
        if not self.url or not self.key:
            raise RuntimeError(
                "Set SUPABASE_URL and SUPABASE_KEY (see .env.example) before "
                "using SupabaseVectorStore."
            )

    @property
    def client(self):
        if self._client is None:
            from supabase import create_client  # lazy

            self._client = create_client(self.url, self.key)
        return self._client

    def _db(self):
        """A client bound to the engine's schema (default: semantic_search)."""
        return self.client.schema(self.schema)

    def upsert(self, record: Record) -> None:
        """Write a processed product record to the products table."""
        row = {
            "external_id": record.get("external_id"),
            "name": record.get("name"),
            "description": record.get("description"),
            "category": record.get("category"),
            "discovered_categories": record.get("discovered_categories", []),
            "tags": record.get("tags", []),
            "embedding": record.get("embedding"),
        }
        self._db().table(self.table).upsert(
            row, on_conflict="external_id"
        ).execute()

    def search(self, query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """Cosine similarity search via the search_products RPC."""
        resp = self._db().rpc(
            self.search_fn,
            {"query_embedding": query_embedding, "match_count": k},
        ).execute()
        return resp.data or []

    # --- marketplace source table (public.products) -----------------------

    def write_source_tags(self, external_id: str, tags: List[str]) -> None:
        """Write generated tags back into the marketplace's products.tags column."""
        (
            self.client.schema(self.source_schema)
            .table(self.source_table)
            .update({"tags": tags})
            .eq("id", external_id)
            .execute()
        )

    def iter_source_products(
        self,
        only_active: bool = True,
        since: Optional[str] = None,
        batch: int = 500,
    ) -> Iterator[Dict[str, Any]]:
        """Yield products from the marketplace table for (re)indexing.

        ``since`` (ISO timestamp) limits to rows changed after it — used for
        incremental rebuilds. ``only_active`` skips inactive listings.
        """
        offset = 0
        while True:
            q = (
                self.client.schema(self.source_schema)
                .table(self.source_table)
                .select("id,name,description,updated_at")
            )
            if only_active:
                q = q.eq("is_active", True)
            if since:
                q = q.gte("updated_at", since)
            rows = (q.range(offset, offset + batch - 1).execute().data) or []
            for row in rows:
                yield row
            if len(rows) < batch:
                break
            offset += batch

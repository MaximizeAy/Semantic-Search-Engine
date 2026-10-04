"""Stage 8 — Vector store (Supabase / pgvector).

Thin client over a Supabase project with the ``vector`` extension. Reads
credentials from the environment (``SUPABASE_URL`` / ``SUPABASE_KEY``) so no
secrets live in code. The matching schema is in ``db/schema.sql`` — run it in
your Supabase project before using this client.

The ``supabase`` package is imported lazily.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .types import Record


class SupabaseVectorStore:
    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        table: str = "products",
        search_fn: str = "search_products",
    ):
        self.url = url or os.environ.get("SUPABASE_URL")
        self.key = key or os.environ.get("SUPABASE_KEY")
        self.table = table
        self.search_fn = search_fn
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
        self.client.table(self.table).upsert(
            row, on_conflict="external_id"
        ).execute()

    def search(self, query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """Cosine similarity search via the search_products RPC."""
        resp = self.client.rpc(
            self.search_fn,
            {"query_embedding": query_embedding, "match_count": k},
        ).execute()
        return resp.data or []

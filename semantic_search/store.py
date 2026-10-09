"""Stage 8 — Vector store (Supabase / pgvector).

Thin client over a Supabase project with the ``vector`` extension. Reads
credentials from the environment (``SUPABASE_URL`` / ``SUPABASE_KEY``) so no
secrets live in code. The matching schema is in ``db/schema.sql`` — run it in
your Supabase project before using this client.

The ``supabase`` package is imported lazily.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from .types import Record


def to_vector_literal(vec: List[float]) -> str:
    """pgvector input format: '[0.1,0.2,...]'.

    PostgREST does not reliably cast a JSON array into a vector column, so
    embeddings are sent as this string literal, which vector's input function
    parses directly.
    """
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        """Write a processed product record to the engine's products table."""
        embedding = record.get("embedding")
        row = {
            "external_id": record.get("external_id"),
            "name": record.get("name"),
            "description": record.get("description"),
            "category": record.get("category"),
            "discovered_categories": record.get("discovered_categories", []),
            "tags": record.get("tags", []),
            # vector column: send the pgvector string literal, not a JSON array
            "embedding": to_vector_literal(embedding) if embedding is not None else None,
            # refresh on every (re)index so incremental rebuilds can compare it
            "indexed_at": _now_iso(),
        }
        self._db().table(self.table).upsert(
            row, on_conflict="external_id"
        ).execute()

    def get_indexed(self, external_id: str) -> Optional[Dict[str, Any]]:
        """Return the engine's current {tags, indexed_at} for a product, or None."""
        rows = (
            self._db()
            .table(self.table)
            .select("tags,indexed_at")
            .eq("external_id", external_id)
            .limit(1)
            .execute()
            .data
        ) or []
        return rows[0] if rows else None

    def indexed_map(self) -> Dict[str, str]:
        """external_id -> indexed_at across the index (for incremental rebuilds)."""
        out: Dict[str, str] = {}
        offset = 0
        batch = 1000
        while True:
            rows = (
                self._db()
                .table(self.table)
                .select("external_id,indexed_at")
                .range(offset, offset + batch - 1)
                .execute()
                .data
            ) or []
            for r in rows:
                if r.get("external_id"):
                    out[r["external_id"]] = r.get("indexed_at")
            if len(rows) < batch:
                break
            offset += batch
        return out

    def search(self, query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """Cosine similarity search via the search_products RPC."""
        resp = self._db().rpc(
            self.search_fn,
            {"query_embedding": to_vector_literal(query_embedding), "match_count": k},
        ).execute()
        return resp.data or []

    # --- marketplace source table (public.products) -----------------------

    def get_source_tags(self, external_id: str) -> List[str]:
        """Current tags on the marketplace product row (seller-set + prior)."""
        rows = (
            self.client.schema(self.source_schema)
            .table(self.source_table)
            .select("tags")
            .eq("id", external_id)
            .limit(1)
            .execute()
            .data
        ) or []
        return (rows[0].get("tags") or []) if rows else []

    def write_source_tags(self, external_id: str, tags: List[str]) -> int:
        """Write tags into the marketplace's products.tags; return rows updated.

        A return of 0 means no product matched that id (e.g. tagged before the
        row existed) — the caller can warn rather than silently losing tags.
        """
        resp = (
            self.client.schema(self.source_schema)
            .table(self.source_table)
            .update({"tags": tags})
            .eq("id", external_id)
            .execute()
        )
        return len(resp.data or [])

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

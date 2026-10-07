"""Service layer — the two ways the engine is used.

1. **Tag on create** — ``tag_product()``: a seller creates a product/service,
   we generate tags + category + embedding, upsert the embedding into
   ``semantic_search.products`` (so it is searchable) and write the tags back
   into the marketplace's ``public.products.tags``.

2. **Realtime search** — ``search()``: a shopper query is normalized, typo-
   corrected, embedded, and matched against the vector index.

``rebuild()`` re-runs tagging across the catalog (see the weekly-rebuild note in
the README). The embedder, pipeline, and store are built once and shared.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .embed import Embedder
from .pipeline import build_search_pipeline, build_tag_pipeline
from .store import SupabaseVectorStore
from .types import Record


class TaggingService:
    def __init__(
        self,
        store: Optional[SupabaseVectorStore] = None,
        embedder: Optional[Embedder] = None,
    ):
        self.embedder = embedder or Embedder()
        self.store = store or SupabaseVectorStore()
        self.pipeline = build_tag_pipeline(embedder=self.embedder)
        self.search_pipeline = build_search_pipeline(self.store, embedder=self.embedder)

    # --- flow 1: tag on create ------------------------------------------
    def tag_product(
        self,
        name: str,
        description: str = "",
        external_id: Optional[str] = None,
        persist: bool = True,
        write_source_tags: bool = True,
    ) -> Record:
        """Generate tags + category + embedding for one product.

        With ``persist``, the embedding is upserted into the search index; with
        ``write_source_tags`` and an ``external_id``, the tags are also written
        back to the marketplace's products.tags.
        """
        record = self.pipeline.run_text(name, description, external_id)
        if persist:
            self.store.upsert(record)
            if write_source_tags and external_id:
                self.store.write_source_tags(external_id, record["tags"])
        return record

    # --- flow 2: realtime search ----------------------------------------
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        return self.search_pipeline.search(query, k)

    # --- maintenance: periodic rebuild ----------------------------------
    def rebuild(
        self,
        only_active: bool = True,
        since: Optional[str] = None,
        write_source_tags: bool = True,
    ) -> int:
        """Re-tag + re-index catalog products. Returns how many were processed.

        ``since`` (ISO timestamp) does an incremental rebuild of recently
        changed products; omit it for a full rebuild.
        """
        count = 0
        for row in self.store.iter_source_products(only_active=only_active, since=since):
            self.tag_product(
                row["name"],
                row.get("description") or "",
                external_id=row["id"],
                write_source_tags=write_source_tags,
            )
            count += 1
        return count

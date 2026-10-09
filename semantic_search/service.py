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

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .embed import Embedder
from .pipeline import build_search_pipeline, build_tag_pipeline
from .store import SupabaseVectorStore
from .types import Record

logger = logging.getLogger(__name__)

MAX_SOURCE_TAGS = 25


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    """Parse a Postgres/ISO timestamp string; None on failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def merge_tags(
    current: List[str],
    previous_generated: List[str],
    generated: List[str],
    cap: int = MAX_SOURCE_TAGS,
) -> List[str]:
    """Union seller tags with freshly generated ones (case-insensitive dedup).

    Seller-set tags are preserved; tags we generated on a prior run that are no
    longer generated are dropped (so the column does not accumulate stale tags);
    new generated tags are appended. Order: kept tags first, then new ones.
    """
    prev = {t.casefold() for t in previous_generated}
    gen = {t.casefold() for t in generated}
    result: List[str] = []
    seen: set[str] = set()
    for t in current:
        c = t.casefold()
        if c in seen:
            continue
        if c in prev and c not in gen:
            continue  # stale: we added it before, no longer generated
        result.append(t)
        seen.add(c)
    for t in generated:
        c = t.casefold()
        if c not in seen:
            result.append(t)
            seen.add(c)
    return result[:cap]


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

        # Persistence requires an external_id to link back to the product and to
        # avoid piling up NULL-keyed rows in the index.
        if persist and external_id:
            # read our previously-generated tags BEFORE the upsert overwrites them
            previous = (self.store.get_indexed(external_id) or {}).get("tags") or []
            self.store.upsert(record)
            if write_source_tags:
                self._merge_and_write_tags(external_id, record["tags"], previous)
        return record

    def _merge_and_write_tags(
        self, external_id: str, generated: List[str], previous: List[str]
    ) -> None:
        """Union generated tags with the product's existing tags and write back."""
        current = self.store.get_source_tags(external_id)
        merged = merge_tags(current, previous, generated)
        if merged == current:
            return  # no change — skip the write (and avoid churn)
        updated = self.store.write_source_tags(external_id, merged)
        if updated == 0:
            logger.warning(
                "tag write matched no product row (external_id=%s); "
                "was it tagged before the product existed?",
                external_id,
            )

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
        # Skip products already indexed since their last real change, so our own
        # tag-writes (which may bump updated_at) don't cause re-processing churn.
        indexed = self.store.indexed_map()
        count = 0
        for row in self.store.iter_source_products(only_active=only_active, since=since):
            updated_at = _parse_ts(row.get("updated_at"))
            indexed_at = _parse_ts(indexed.get(row["id"]))
            if indexed_at and updated_at and indexed_at >= updated_at:
                continue
            self.tag_product(
                row["name"],
                row.get("description") or "",
                external_id=row["id"],
                write_source_tags=write_source_tags,
            )
            count += 1
        return count

"""Stage 7 — Content filtering.

The safety gate before anything is persisted or returned:
  - drop banned / unsafe / profane tags (denylist + better_profanity),
  - strip near-duplicate tags (fuzzy),
  - apply a confidence threshold (from tag_scores, if present),
  - cap the number of tags.

Always the last stage.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Set

from .types import Record


class ContentFilter:
    def __init__(
        self,
        denylist: Optional[Iterable[str]] = None,
        min_score: float = 0.0,
        max_tags: int = 15,
        dedupe_threshold: float = 90.0,
        use_profanity: bool = True,
    ):
        self.denylist: Set[str] = {d.strip().lower() for d in (denylist or [])}
        self.min_score = min_score
        self.max_tags = max_tags
        self.dedupe_threshold = dedupe_threshold
        self.use_profanity = use_profanity
        self._profanity = None

    def _is_profane(self, term: str) -> bool:
        if not self.use_profanity:
            return False
        if self._profanity is None:
            try:
                from better_profanity import profanity  # lazy

                profanity.load_censor_words()
                self._profanity = profanity
            except ImportError:
                self.use_profanity = False
                return False
        return self._profanity.contains_profanity(term)

    def _blocked(self, term: str) -> bool:
        return term in self.denylist or self._is_profane(term)

    def _dedupe(self, tags: List[str]) -> List[str]:
        from rapidfuzz import fuzz  # lazy

        kept: List[str] = []
        for tag in tags:
            if all(fuzz.WRatio(tag, k) < self.dedupe_threshold for k in kept):
                kept.append(tag)
        return kept

    def apply(self, tags: List[str], scores: Optional[dict] = None) -> List[str]:
        scores = scores or {}
        out = [
            t
            for t in tags
            if not self._blocked(t) and scores.get(t, 1.0) >= self.min_score
        ]
        out = self._dedupe(out)
        return out[: self.max_tags]


class FilterStage:
    name = "filter"

    def __init__(self, content_filter: ContentFilter | None = None):
        self.content_filter = content_filter or ContentFilter()

    def process(self, record: Record) -> Record:
        record["tags"] = self.content_filter.apply(
            record.get("tags", []), record.get("tag_scores")
        )
        # Also clean the discovered/slang category terms.
        record["discovered_categories"] = self.content_filter.apply(
            record.get("discovered_categories", [])
        )
        return record

"""Stage 1 — Input normalization.

Unicode (NFKC), whitespace collapse, and casefolding. Shared verbatim by both
the offline indexing path and the online query path.

``TextNormalizer`` is carried over from the original notebook (with the
``strip_whitespace`` typo fixed); ``NormalizeStage`` adapts it to the pipeline.
"""

from __future__ import annotations

import re
import unicodedata

from .types import Record


class TextNormalizer:
    def __init__(
        self,
        unicode_normalize: bool = True,
        strip_whitespace: bool = True,
        normalize_case: bool = True,
    ):
        self.unicode_normalize = unicode_normalize
        self.strip_whitespace = strip_whitespace
        self.normalize_case = normalize_case
        self.whitespace_pattern = re.compile(r"[\s​-‍﻿  ]+")

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)  # Convert non string into string
        if self.unicode_normalize:
            text = unicodedata.normalize("NFKC", text)
        if self.strip_whitespace:
            text = self.whitespace_pattern.sub(" ", text)
            text = text.strip()
        if self.normalize_case:
            text = text.casefold()
        return text


class NormalizeStage:
    """Normalize name + description and build the combined text for embedding."""

    name = "normalize"

    def __init__(self, normalizer: TextNormalizer | None = None):
        self.normalizer = normalizer or TextNormalizer()

    def process(self, record: Record) -> Record:
        record["normalized_name"] = self.normalizer.normalize(record.get("name", ""))
        record["normalized_description"] = self.normalizer.normalize(
            record.get("description", "") or ""
        )
        # Combined text is what stage 5 embeds. Name first (higher signal).
        parts = [record["normalized_name"], record["normalized_description"]]
        record["text"] = ". ".join(p for p in parts if p).strip()
        return record

    def normalize_query(self, query: str) -> str:
        """Query-path helper: normalize a raw search query."""
        return self.normalizer.normalize(query)

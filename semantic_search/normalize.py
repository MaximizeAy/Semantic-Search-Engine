"""Stage 1 — Input normalization.

Unicode (NFKC), whitespace collapse, and casefolding. Shared verbatim by both
the offline indexing path and the online query path.

``TextNormalizer`` is carried over from the original notebook (with the
``strip_whitespace`` typo fixed); ``NormalizeStage`` adapts it to the pipeline.
"""

from __future__ import annotations

import html
import re
import unicodedata

from .types import Record

_SCRIPT_STYLE = re.compile(r"(?is)<(script|style)[^>]*>.*?</\1>")
_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|data:\S+")
_BASE64ISH = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")


class TextNormalizer:
    def __init__(
        self,
        unicode_normalize: bool = True,
        strip_whitespace: bool = True,
        normalize_case: bool = True,
        strip_html: bool = True,
    ):
        self.unicode_normalize = unicode_normalize
        self.strip_whitespace = strip_whitespace
        self.normalize_case = normalize_case
        self.strip_html = strip_html
        self.whitespace_pattern = re.compile(r"[\s​-‍﻿  ]+")

    def _clean_html(self, text: str) -> str:
        # unescape first so escaped markup (&lt;img&gt;) is stripped too
        text = html.unescape(text)
        text = _SCRIPT_STYLE.sub(" ", text)
        text = _TAG.sub(" ", text)
        text = _URL.sub(" ", text)
        text = _BASE64ISH.sub(" ", text)
        return text

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)  # Convert non string into string
        if self.strip_html:
            text = self._clean_html(text)
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

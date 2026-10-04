"""Stage 2 — Linguistics (spaCy).

Tokenize, lemmatize, tag POS, and extract noun-chunks + named entities. The
noun-chunks and entities are the first source of raw tag candidates.

spaCy is imported lazily and the model is loaded once, so importing this module
is free and the ~12 MB model is only pulled in when a ``LinguisticsStage`` is
actually constructed.
"""

from __future__ import annotations

from typing import List

from .types import Record

DEFAULT_MODEL = "en_core_web_sm"


class LinguisticExtractor:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model_name = model
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy  # lazy

            try:
                self._nlp = spacy.load(self.model_name)
            except OSError as exc:  # model not downloaded
                raise RuntimeError(
                    f"spaCy model '{self.model_name}' is not installed. "
                    f"Run: python -m spacy download {self.model_name}"
                ) from exc
        return self._nlp

    def candidates(self, text: str) -> List[str]:
        """Return deduplicated noun-chunk + entity spans as candidate terms."""
        doc = self.nlp(text)
        seen = set()
        out: List[str] = []
        for span in list(doc.noun_chunks) + list(doc.ents):
            term = span.text.strip().lower()
            if term and term not in seen:
                seen.add(term)
                out.append(term)
        return out


class LinguisticsStage:
    name = "linguistics"

    def __init__(self, extractor: LinguisticExtractor | None = None):
        self.extractor = extractor or LinguisticExtractor()

    def process(self, record: Record) -> Record:
        text = record.get("text") or ""
        for term in self.extractor.candidates(text):
            record["candidates"].append({"term": term, "score": None})
        return record

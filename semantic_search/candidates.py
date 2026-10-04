"""Stage 3 — Candidate tag scoring.

Rank the raw candidates so the salient ones survive. YAKE is unsupervised and
dependency-light; it scores keyword phrases (lower score = more relevant, which
we invert to a 0..1 relevance). Candidates already proposed by the linguistics
stage are merged in and kept even if YAKE doesn't surface them.
"""

from __future__ import annotations

from typing import Dict, List

from .types import Record


class KeywordCandidates:
    def __init__(self, language: str = "en", max_ngram: int = 3, top: int = 20):
        self.language = language
        self.max_ngram = max_ngram
        self.top = top
        self._extractor = None

    @property
    def extractor(self):
        if self._extractor is None:
            import yake  # lazy

            self._extractor = yake.KeywordExtractor(
                lan=self.language, n=self.max_ngram, top=self.top
            )
        return self._extractor

    def score(self, text: str) -> Dict[str, float]:
        """Map term -> relevance in 0..1 (higher = more relevant)."""
        raw = self.extractor.extract_keywords(text)  # [(term, yake_score), ...]
        if not raw:
            return {}
        # YAKE: lower is better. Invert and min-max normalise to 0..1.
        scores = [s for _, s in raw]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        return {term.lower(): 1.0 - (s - lo) / span for term, s in raw}


class CandidateStage:
    name = "candidates"

    def __init__(self, keyworder: KeywordCandidates | None = None):
        self.keyworder = keyworder or KeywordCandidates()

    def process(self, record: Record) -> Record:
        text = record.get("text") or ""
        scored = self.keyworder.score(text)

        merged: Dict[str, float] = {}
        # existing (linguistics) candidates first
        for c in record["candidates"]:
            merged[c["term"]] = scored.get(c["term"], c.get("score") or 0.5)
        # add any keyword-only candidates
        for term, s in scored.items():
            merged.setdefault(term, s)

        record["candidates"] = [
            {"term": t, "score": s}
            for t, s in sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
        ]
        return record

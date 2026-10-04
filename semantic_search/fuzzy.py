"""Stage 4 — Fuzzy canonicalization.

Map noisy candidate terms onto a controlled tag vocabulary with RapidFuzz.
The vocabulary is *hybrid* (see docs/PIPELINE.md):
  - an imported taxonomy backbone, plus
  - data-discovered terms (slang / local product words).

A candidate that fuzzy-matches a vocabulary term at or above ``threshold`` is
canonicalized to it; otherwise it is kept as a *discovered* term so local
vocabulary is never thrown away. This module also powers query-side typo
correction.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from .types import Record, candidate_terms


class TagVocabulary:
    """A controlled vocabulary of canonical tags with sources."""

    def __init__(self):
        # term -> source ('taxonomy' | 'discovered')
        self._terms: dict[str, str] = {}

    @classmethod
    def from_taxonomy(cls, terms: Iterable[str]) -> "TagVocabulary":
        v = cls()
        for t in terms:
            v.add(t, source="taxonomy")
        return v

    def add(self, term: str, source: str = "taxonomy") -> None:
        term = term.strip().lower()
        if term:
            self._terms.setdefault(term, source)

    def __contains__(self, term: str) -> bool:
        return term.strip().lower() in self._terms

    def __len__(self) -> int:
        return len(self._terms)

    @property
    def terms(self) -> List[str]:
        return list(self._terms)


class FuzzyCanonicalizer:
    def __init__(self, vocabulary: TagVocabulary, threshold: float = 85.0):
        self.vocabulary = vocabulary
        self.threshold = threshold

    def canonical(self, term: str) -> Optional[Tuple[str, float]]:
        """Best vocabulary match for ``term`` if it clears the threshold."""
        from rapidfuzz import fuzz, process  # lazy

        if not self.vocabulary.terms:
            return None
        match = process.extractOne(
            term.strip().lower(), self.vocabulary.terms, scorer=fuzz.WRatio
        )
        if match and match[1] >= self.threshold:
            return match[0], match[1]
        return None

    def correct_query(self, token: str) -> str:
        """Query-path typo correction: snap a token to the nearest tag, else keep."""
        hit = self.canonical(token)
        return hit[0] if hit else token


class FuzzyStage:
    name = "fuzzy"

    def __init__(self, canonicalizer: FuzzyCanonicalizer, learn_discovered: bool = True):
        self.canonicalizer = canonicalizer
        self.learn_discovered = learn_discovered

    def process(self, record: Record) -> Record:
        canonical: List[str] = []
        discovered: List[str] = []
        seen = set()
        for term in candidate_terms(record):
            hit = self.canonicalizer.canonical(term)
            if hit:
                tag = hit[0]
                if tag not in seen:
                    seen.add(tag)
                    canonical.append(tag)
            else:
                if term not in seen:
                    seen.add(term)
                    discovered.append(term)
                    if self.learn_discovered:
                        self.canonicalizer.vocabulary.add(term, source="discovered")
        record["tags"] = canonical
        record["discovered_tags"] = discovered
        return record

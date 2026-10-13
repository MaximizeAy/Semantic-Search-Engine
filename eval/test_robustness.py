"""Measure fuzzy-correction accuracy against a noisy test set.

Given a tag vocabulary and a list of clean terms, build noisy variants and
report the fraction that the FuzzyCanonicalizer snaps back to the correct term.

Run: python -m eval.test_robustness
"""

from __future__ import annotations

from typing import Iterable, Sequence

from semantic_search.fuzzy import FuzzyCanonicalizer, TagVocabulary

from .build_testset import build_testset


def correction_accuracy(
    clean_terms: Sequence[str],
    vocabulary: Iterable[str] | None = None,
    variants: int = 5,
    threshold: float = 85.0,
) -> float:
    """Fraction of noisy variants correctly canonicalized to their clean term."""
    vocab = TagVocabulary.from_taxonomy(vocabulary or clean_terms)
    canon = FuzzyCanonicalizer(vocab, threshold=threshold)

    rows = build_testset(clean_terms, variants=variants)
    hits = 0
    for row in rows:
        result = canon.canonical(row["noisy"])
        if result and result[0] == row["clean"]:
            hits += 1
    return hits / len(rows) if rows else 0.0


if __name__ == "__main__":
    terms = [
        "wireless",
        "bluetooth",
        "headphones",
        "stainless steel",
        "water bottle",
        "cotton",
        "organic",
    ]
    acc = correction_accuracy(terms, variants=10)
    print(f"Fuzzy correction accuracy: {acc:.1%}")

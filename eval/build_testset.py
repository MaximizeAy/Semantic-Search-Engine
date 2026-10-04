"""Build a typo test set from clean terms.

Each row pairs a clean canonical term with ``variants`` noisy versions of it,
so fuzzy-correction and retrieval can be scored against a known ground truth.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List

from .noise import NoiseGenerator


def build_testset(
    terms: Iterable[str],
    variants: int = 5,
    seed: int | None = 42,
    generator: NoiseGenerator | None = None,
) -> List[Dict[str, str]]:
    """Return rows of {"clean": term, "noisy": variant}.

    ``variants`` noisy versions are produced per clean term. A ``seed`` makes
    the set reproducible.
    """
    if seed is not None:
        random.seed(seed)
    gen = generator or NoiseGenerator()

    rows: List[Dict[str, str]] = []
    for term in terms:
        for _ in range(variants):
            rows.append({"clean": term, "noisy": gen.generate(term)})
    return rows


if __name__ == "__main__":  # quick demo
    sample = ["wireless bluetooth headphones", "stainless steel water bottle"]
    for row in build_testset(sample, variants=3):
        print(f"{row['noisy']!r:40} -> {row['clean']!r}")

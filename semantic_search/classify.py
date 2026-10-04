"""Stage 6 — Classification & tag scoring.

Hybrid category assignment (see docs/PIPELINE.md):
  - Fixed-list zero-shot: embed a known category label set once, assign each
    product its nearest label. This is the clean canonical "depiction".
  - Discovered: slang / local category-like terms are preserved from the
    discovered tags rather than discarded.

Also scores each canonical tag by cosine(tag, product) so stage 7 can threshold
and cap.
"""

from __future__ import annotations

from typing import List, Sequence

from .embed import Embedder
from .types import Record


def _cosine(a, b):
    import numpy as np  # lazy

    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(a @ b / denom)


class ZeroShotCategorizer:
    def __init__(self, embedder: Embedder, labels: Sequence[str]):
        self.embedder = embedder
        self.labels = list(labels)
        self._label_vectors = None

    @property
    def label_vectors(self):
        if self._label_vectors is None:
            self._label_vectors = self.embedder.encode(self.labels)
        return self._label_vectors

    def classify(self, embedding) -> tuple[str, float]:
        """Return (nearest_label, score) for a product embedding."""
        import numpy as np  # lazy

        scores = [_cosine(embedding, lv) for lv in self.label_vectors]
        i = int(np.argmax(scores))
        return self.labels[i], scores[i]


class ClassifyStage:
    name = "classify"

    def __init__(self, categorizer: ZeroShotCategorizer, embedder: Embedder):
        self.categorizer = categorizer
        self.embedder = embedder

    def process(self, record: Record) -> Record:
        emb = record.get("embedding")
        if emb is None:
            raise ValueError("ClassifyStage requires an embedding (run EmbedStage first)")

        label, _ = self.categorizer.classify(emb)
        record["category"] = label
        # Preserve discovered terms as secondary (slang/local) categories.
        record["discovered_categories"] = list(record.get("discovered_tags", []))

        # Score canonical tags by relevance to the product embedding.
        tags: List[str] = record.get("tags", [])
        if tags:
            tag_vectors = self.embedder.encode(tags)
            scored = sorted(
                ((t, _cosine(emb, tv)) for t, tv in zip(tags, tag_vectors)),
                key=lambda kv: kv[1],
                reverse=True,
            )
            record["tags"] = [t for t, _ in scored]
            record["tag_scores"] = {t: s for t, s in scored}
        return record

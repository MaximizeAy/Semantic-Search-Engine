"""Stage 5 — Embedding.

One encoder, shared by both paths so product and query vectors live in the same
space. Default model is ``BAAI/bge-m3`` (1024-dim, multilingual, no query/passage
prefix needed). Product embeddings are meant to be batch-encoded on Kaggle GPU;
a single query can be encoded on CPU at serve time.

The embedding dimension is baked into the pgvector column, so changing the model
is a schema migration (see docs/PIPELINE.md).
"""

from __future__ import annotations

import os
from typing import List, Sequence

from .types import Record

DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024  # bge-m3


class Embedder:
    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None):
        self.model_name = model
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> "list":
        """Encode a batch of texts into L2-normalized vectors (list of lists)."""
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def encode_one(self, text: str) -> List[float]:
        return self.encode([text])[0]


class EmbedStage:
    name = "embed"

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def process(self, record: Record) -> Record:
        record["embedding"] = self.embedder.encode_one(record.get("text") or "")
        return record

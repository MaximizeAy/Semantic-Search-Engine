"""Stage 5 — Embedding.

One encoder, shared by both paths so product and query vectors live in the same
space. Model is ``bge-m3`` (1024-dim, multilingual). Two interchangeable
backends — selected by the ``EMBEDDING_BACKEND`` env var:

- ``sentence_transformers`` (default): local/GPU ``BAAI/bge-m3`` via
  sentence-transformers. Good for Kaggle GPU batch work.
- ``workers_ai``: Cloudflare Workers AI ``@cf/baai/bge-m3`` over REST. No torch,
  no model download — used by the $0 deploy (search Worker + tagging Action) so
  everything embeds with the same hosted model.

Both backends L2-normalize output, so vectors are comparable across them. The
embedding dimension is baked into the pgvector column; changing the model is a
schema migration (see docs/PIPELINE.md).
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from typing import List, Sequence

from .types import Record

DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024  # bge-m3
DEFAULT_BACKEND = os.environ.get("EMBEDDING_BACKEND", "sentence_transformers")


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class SentenceTransformerBackend:
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

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> List[List[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]


class WorkersAIBackend:
    """Cloudflare Workers AI embeddings over REST (no torch, no model download).

    Needs CF_ACCOUNT_ID and CF_AI_TOKEN in the environment. Output is
    L2-normalized to match the sentence-transformers backend.
    """

    def __init__(
        self,
        account_id: str | None = None,
        token: str | None = None,
        model: str | None = None,
        timeout: int = 30,
    ):
        self.account_id = account_id or os.environ.get("CF_ACCOUNT_ID")
        self.token = token or os.environ.get("CF_AI_TOKEN")
        self.model = model or os.environ.get("WORKERS_AI_MODEL", "@cf/baai/bge-m3")
        self.timeout = timeout
        if not self.account_id or not self.token:
            raise RuntimeError(
                "Workers AI backend needs CF_ACCOUNT_ID and CF_AI_TOKEN in the "
                "environment."
            )

    @property
    def _url(self) -> str:
        return (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self.account_id}/ai/run/{self.model}"
        )

    def _run(self, texts: List[str]) -> List[List[float]]:
        body = json.dumps({"text": texts}).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload.get("success", True):
            raise RuntimeError(f"Workers AI error: {payload.get('errors')}")
        result = payload.get("result", payload)
        # bge embeddings come back as result.data: [[...], ...]
        data = result.get("data") if isinstance(result, dict) else result
        if not data:
            raise RuntimeError(f"Unexpected Workers AI response: {payload!r}")
        return [list(map(float, v)) for v in data]

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> List[List[float]]:
        texts = list(texts)
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(self._run(texts[i : i + batch_size]))
        return [_l2_normalize(v) for v in out]


def _make_backend(name: str, model: str, device: str | None):
    if name == "workers_ai":
        return WorkersAIBackend()
    return SentenceTransformerBackend(model=model, device=device)


class Embedder:
    """Facade over a backend; same interface regardless of where it runs."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        backend: str | None = None,
    ):
        self.model_name = model
        self.device = device
        self.backend_name = backend or DEFAULT_BACKEND
        self._backend = None

    @property
    def backend(self):
        if self._backend is None:
            self._backend = _make_backend(self.backend_name, self.model_name, self.device)
        return self._backend

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> List[List[float]]:
        return self.backend.encode(texts, batch_size=batch_size)

    def encode_one(self, text: str) -> List[float]:
        return self.encode([text])[0]


class EmbedStage:
    name = "embed"

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def process(self, record: Record) -> Record:
        record["embedding"] = self.embedder.encode_one(record.get("text") or "")
        return record

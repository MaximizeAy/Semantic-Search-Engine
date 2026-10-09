"""Test doubles that let the pipeline/service/store logic be exercised without
spaCy, YAKE, RapidFuzz, sentence-transformers, numpy, or a live Supabase.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, List


def has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


# --- stage doubles -------------------------------------------------------

class FakeExtractor:
    """Stand-in for linguistics.LinguisticExtractor."""

    def __init__(self, terms: List[str]):
        self._terms = terms

    def candidates(self, text: str) -> List[str]:
        return list(self._terms)


class FakeKeyworder:
    """Stand-in for candidates.KeywordCandidates."""

    def __init__(self, scores: Dict[str, float]):
        self._scores = scores

    def score(self, text: str) -> Dict[str, float]:
        return dict(self._scores)


class FakeVocabulary:
    def __init__(self):
        self.added: List[tuple] = []

    def add(self, term: str, source: str = "taxonomy") -> None:
        self.added.append((term, source))


class FakeCanonicalizer:
    """Stand-in for fuzzy.FuzzyCanonicalizer.

    ``mapping`` maps a candidate term -> canonical tag; anything missing is
    treated as a no-match (discovered).
    """

    def __init__(self, mapping: Dict[str, str]):
        self._mapping = mapping
        self.vocabulary = FakeVocabulary()

    def canonical(self, term: str):
        if term in self._mapping:
            return self._mapping[term], 100.0
        return None

    def correct_query(self, token: str) -> str:
        hit = self.canonical(token)
        return hit[0] if hit else token


class FakeEmbedder:
    """Deterministic stand-in for embed.Embedder (no model load)."""

    def __init__(self, vectors: Dict[str, List[float]] | None = None, dim: int = 4):
        self._vectors = vectors or {}
        self._dim = dim

    def _vec(self, text: str) -> List[float]:
        if text in self._vectors:
            return list(self._vectors[text])
        # deterministic pseudo-vector from the text
        h = sum(ord(c) for c in text) or 1
        return [((h >> i) & 1) + 0.1 for i in range(self._dim)]

    def encode(self, texts, batch_size: int = 32) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def encode_one(self, text: str) -> List[float]:
        return self._vec(text)


# --- service / store doubles --------------------------------------------

class FakePipeline:
    def __init__(self, result: Dict[str, Any]):
        self._result = result
        self.calls: List[tuple] = []

    def run_text(self, name, description="", external_id=None):
        self.calls.append((name, description, external_id))
        rec = dict(self._result)
        rec["external_id"] = external_id
        rec["name"] = name
        return rec


class FakeSearchPipeline:
    def __init__(self, results):
        self._results = results
        self.calls: List[tuple] = []

    def search(self, query, k=10):
        self.calls.append((query, k))
        return self._results


class FakeStore:
    def __init__(self, source_rows=None, indexed=None, source_tags=None, indexed_at_map=None):
        self.upserts: List[dict] = []
        self.tag_writes: List[tuple] = []
        self._source_rows = source_rows or []
        self._indexed = indexed or {}            # external_id -> {"tags", "indexed_at"}
        self._source_tags = source_tags or {}    # external_id -> [tags]
        self._indexed_at_map = indexed_at_map or {}
        self.write_result = 1                    # rows "updated" by write_source_tags

    def upsert(self, record):
        self.upserts.append(record)

    def get_indexed(self, external_id):
        return self._indexed.get(external_id)

    def indexed_map(self):
        return dict(self._indexed_at_map)

    def get_source_tags(self, external_id):
        return list(self._source_tags.get(external_id, []))

    def write_source_tags(self, external_id, tags):
        self.tag_writes.append((external_id, tags))
        return self.write_result

    def iter_source_products(self, only_active=True, since=None, batch=500):
        for row in self._source_rows:
            yield row


# --- fake supabase client (for store tests) ------------------------------

class _Resp:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name
        self.op = None
        self.payload = None
        self.on_conflict = None
        self.filters = []
        self._range = None

    def upsert(self, row, on_conflict=None):
        self.op = "upsert"
        self.payload = row
        self.on_conflict = on_conflict
        return self

    def update(self, row):
        self.op = "update"
        self.payload = row
        return self

    def select(self, cols):
        self.op = "select"
        self.payload = cols
        return self

    def limit(self, n):
        self._limit = n
        return self

    def eq(self, col, val):
        self.filters.append(("eq", col, val))
        return self

    def gte(self, col, val):
        self.filters.append(("gte", col, val))
        return self

    def range(self, a, b):
        self._range = (a, b)
        return self

    def execute(self):
        self.parent.parent.calls.append(self)  # record on the client
        if self.op == "select":
            rows = self.parent.parent.source_rows
            if self._range:
                a, b = self._range
                return _Resp(rows[a : b + 1])
            return _Resp(list(rows))
        return _Resp(None)


class _FakeSchema:
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name

    def table(self, name):
        return _FakeTable(self, name)

    def rpc(self, fn, params):
        self.parent.calls.append(("rpc", self.name, fn, params))
        return self

    def execute(self):
        return _Resp(self.parent.rpc_result)


class FakeSupabaseClient:
    def __init__(self, source_rows=None, rpc_result=None):
        self.calls: List[Any] = []
        self.source_rows = source_rows or []
        self.rpc_result = rpc_result or []

    def schema(self, name):
        return _FakeSchema(self, name)

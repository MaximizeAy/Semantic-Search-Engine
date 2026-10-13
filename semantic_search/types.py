"""Shared types for the pipeline.

A *record* is a plain ``dict`` that flows through the pipeline. Each stage reads
some keys and writes others, so the record accumulates fields as it progresses.
Using a dict (rather than a rigid dataclass) keeps stages loosely coupled and
easy to add to.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

Record = Dict[str, Any]


@runtime_checkable
class Stage(Protocol):
    """One step of the pipeline.

    Implementations transform a record in place (and return it). ``name`` is a
    short identifier used in logging and tests.
    """

    name: str

    def process(self, record: Record) -> Record:  # pragma: no cover - protocol
        ...


def new_record(
    name: str,
    description: str = "",
    external_id: Optional[str] = None,
) -> Record:
    """Create a fresh product record with the raw input fields populated.

    Downstream stages fill in the remaining keys:
        normalized_name, normalized_description, text   (normalize)
        candidates                                      (linguistics, candidates)
        tags, discovered_tags                           (fuzzy, filters)
        category, discovered_categories                 (classify)
        embedding                                       (embed)
    """
    return {
        "external_id": external_id,
        "name": name,
        "description": description,
        # populated downstream
        "normalized_name": None,
        "normalized_description": None,
        "text": None,
        "candidates": [],            # list[{"term": str, "score": float}]
        "tags": [],                  # canonical tags (final)
        "discovered_tags": [],       # novel/slang terms not in the taxonomy
        "category": None,            # zero-shot canonical category
        "discovered_categories": [], # slang/local category terms, preserved
        "embedding": None,           # list[float], len == embedding dim
    }


def candidate_terms(record: Record) -> List[str]:
    """Convenience: the plain term strings from ``record['candidates']``."""
    return [c["term"] for c in record.get("candidates", [])]

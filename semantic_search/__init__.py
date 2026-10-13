"""Semantic Product Search Engine.

A pipeline that turns raw e-commerce product data (name + description) into
canonical tags and semantic-search-ready embeddings.

See ``docs/PIPELINE.md`` for the full design. The package is organised as one
module per pipeline stage, each exposing a ``Stage`` with a uniform
``process(record) -> record`` interface so stages can be run, tested, and
swapped independently.

Heavy dependencies (spaCy, sentence-transformers, supabase) are imported lazily
inside the stages that need them, so importing this package is cheap and does
not require those libraries to be installed.
"""

from .types import Record, Stage, new_record

__all__ = ["Record", "Stage", "new_record"]

__version__ = "0.1.0"

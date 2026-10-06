"""The pipeline spine.

``TagPipeline`` runs the offline indexing path (stages 1-7) over a product
record. ``SearchPipeline`` runs the online query path (normalize -> fuzzy
correct -> embed -> vector search).

Use the ``build_*`` factories for a wired-up default, or compose stages
yourself for custom pipelines.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from .candidates import CandidateStage, KeywordCandidates
from .classify import ClassifyStage, ZeroShotCategorizer
from .embed import Embedder, EmbedStage
from .filters import ContentFilter, FilterStage
from .fuzzy import FuzzyCanonicalizer, FuzzyStage, TagVocabulary
from .linguistics import LinguisticsStage
from .normalize import NormalizeStage
from .types import Record, Stage, new_record


class TagPipeline:
    """Offline indexing path: raw product record -> tags + embedding."""

    def __init__(self, stages: Sequence[Stage]):
        self.stages: List[Stage] = list(stages)

    def run(self, record: Record) -> Record:
        for stage in self.stages:
            record = stage.process(record)
        return record

    def run_text(self, name: str, description: str = "", external_id=None) -> Record:
        return self.run(new_record(name, description, external_id))

    def run_batch(self, records: Iterable[Record]) -> List[Record]:
        return [self.run(r) for r in records]


class SearchPipeline:
    """Online query path: raw query -> ranked products."""

    def __init__(
        self,
        normalize: NormalizeStage,
        canonicalizer: FuzzyCanonicalizer,
        embedder: Embedder,
        store,
    ):
        self.normalize = normalize
        self.canonicalizer = canonicalizer
        self.embedder = embedder
        self.store = store

    def search(self, query: str, k: int = 10):
        text = self.normalize.normalize_query(query)
        corrected = " ".join(self.canonicalizer.correct_query(tok) for tok in text.split())
        vector = self.embedder.encode_one(corrected)
        return self.store.search(vector, k=k)


def build_tag_pipeline(
    vocabulary: Optional[TagVocabulary] = None,
    categories: Optional[Sequence[str]] = None,
    embedder: Optional[Embedder] = None,
    content_filter: Optional[ContentFilter] = None,
) -> TagPipeline:
    """Wire the default offline pipeline (stages 1-7) with a shared embedder.

    ``vocabulary`` and ``categories`` default to the seed taxonomy
    (semantic_search.taxonomy) when not supplied.
    """
    from . import taxonomy  # lazy to avoid import cost when custom values are passed

    vocabulary = vocabulary if vocabulary is not None else taxonomy.build_vocabulary()
    categories = categories if categories is not None else taxonomy.default_categories()
    embedder = embedder or Embedder()
    canonicalizer = FuzzyCanonicalizer(vocabulary)
    categorizer = ZeroShotCategorizer(embedder, categories)
    return TagPipeline(
        [
            NormalizeStage(),
            LinguisticsStage(),
            CandidateStage(KeywordCandidates()),
            FuzzyStage(canonicalizer),
            EmbedStage(embedder),
            ClassifyStage(categorizer, embedder),
            FilterStage(content_filter or ContentFilter()),
        ]
    )


def build_search_pipeline(
    store,
    vocabulary: Optional[TagVocabulary] = None,
    embedder: Optional[Embedder] = None,
) -> SearchPipeline:
    """Wire the default online query pipeline, sharing the embedder/vocabulary.

    ``vocabulary`` defaults to the seed taxonomy when not supplied.
    """
    from . import taxonomy  # lazy

    vocabulary = vocabulary if vocabulary is not None else taxonomy.build_vocabulary()
    embedder = embedder or Embedder()
    return SearchPipeline(
        normalize=NormalizeStage(),
        canonicalizer=FuzzyCanonicalizer(vocabulary),
        embedder=embedder,
        store=store,
    )

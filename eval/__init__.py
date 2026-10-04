"""Evaluation harness.

Offline-only tooling that measures the pipeline's robustness to messy input.
``NoiseGenerator`` synthesises realistic typos; the other modules build a test
set from clean terms and score fuzzy-correction / retrieval against it.

Nothing here is part of the serving pipeline (see docs/PIPELINE.md §4).
"""

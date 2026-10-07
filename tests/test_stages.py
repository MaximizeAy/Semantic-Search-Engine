"""Stage logic with injected fakes — no heavy libraries required."""

import unittest

from semantic_search.candidates import CandidateStage
from semantic_search.embed import EmbedStage
from semantic_search.fuzzy import FuzzyStage, TagVocabulary
from semantic_search.linguistics import LinguisticsStage
from semantic_search.types import candidate_terms, new_record

from ._fakes import FakeCanonicalizer, FakeEmbedder, FakeExtractor, FakeKeyworder


class TestLinguisticsStage(unittest.TestCase):
    def test_appends_candidates(self):
        stage = LinguisticsStage(FakeExtractor(["wireless headphones", "battery"]))
        rec = new_record("x")
        rec["text"] = "whatever"
        stage.process(rec)
        self.assertEqual(candidate_terms(rec), ["wireless headphones", "battery"])
        self.assertTrue(all(c["score"] is None for c in rec["candidates"]))


class TestCandidateStage(unittest.TestCase):
    def test_merges_and_sorts_by_score(self):
        rec = new_record("x")
        rec["text"] = "t"
        rec["candidates"] = [{"term": "headphones", "score": None}]
        stage = CandidateStage(FakeKeyworder({"headphones": 0.9, "wireless": 0.4}))
        stage.process(rec)
        terms = candidate_terms(rec)
        self.assertEqual(terms[0], "headphones")  # highest score first
        self.assertIn("wireless", terms)          # keyword-only candidate added

    def test_existing_without_score_gets_default(self):
        rec = new_record("x")
        rec["text"] = "t"
        rec["candidates"] = [{"term": "lamp", "score": None}]
        CandidateStage(FakeKeyworder({})).process(rec)
        self.assertEqual(rec["candidates"][0]["score"], 0.5)


class TestFuzzyStage(unittest.TestCase):
    def test_splits_canonical_and_discovered(self):
        canon = FakeCanonicalizer({"wireles": "wireless", "bt": "bluetooth"})
        rec = new_record("x")
        rec["candidates"] = [
            {"term": "wireles", "score": 1.0},
            {"term": "bt", "score": 0.9},
            {"term": "jollof rice", "score": 0.8},  # local term, no match
        ]
        FuzzyStage(canon).process(rec)
        self.assertEqual(rec["tags"], ["wireless", "bluetooth"])
        self.assertEqual(rec["discovered_tags"], ["jollof rice"])

    def test_learns_discovered_into_vocabulary(self):
        canon = FakeCanonicalizer({})
        rec = new_record("x")
        rec["candidates"] = [{"term": "ankara", "score": 1.0}]
        FuzzyStage(canon, learn_discovered=True).process(rec)
        self.assertIn(("ankara", "discovered"), canon.vocabulary.added)

    def test_dedupes_repeated_canonical(self):
        canon = FakeCanonicalizer({"a": "wireless", "b": "wireless"})
        rec = new_record("x")
        rec["candidates"] = [{"term": "a", "score": 1}, {"term": "b", "score": 1}]
        FuzzyStage(canon).process(rec)
        self.assertEqual(rec["tags"], ["wireless"])


class TestEmbedStage(unittest.TestCase):
    def test_sets_embedding(self):
        rec = new_record("x")
        rec["text"] = "wireless headphones"
        EmbedStage(FakeEmbedder(dim=4)).process(rec)
        self.assertEqual(len(rec["embedding"]), 4)


class TestTagVocabulary(unittest.TestCase):
    def test_from_taxonomy_and_contains(self):
        v = TagVocabulary.from_taxonomy(["Wireless", "bluetooth"])
        self.assertIn("wireless", v)      # lowercased
        self.assertIn("BLUETOOTH", v)     # membership is case-insensitive
        self.assertEqual(len(v), 2)

    def test_add_is_idempotent(self):
        v = TagVocabulary()
        v.add("cotton")
        v.add("cotton")
        self.assertEqual(len(v), 1)


if __name__ == "__main__":
    unittest.main()

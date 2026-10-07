"""Integration tests that exercise the real libraries.

Each is guarded so it runs where the dependency is installed (CI / Kaggle) and
skips cleanly where it is not.
"""

import unittest

from ._fakes import FakeEmbedder, has


@unittest.skipUnless(has("numpy"), "numpy not installed")
class TestClassify(unittest.TestCase):
    def test_zero_shot_and_tag_ranking(self):
        from semantic_search.classify import ClassifyStage, ZeroShotCategorizer
        from semantic_search.types import new_record

        emb = FakeEmbedder(vectors={
            "Electronics": [1.0, 0.0],
            "Apparel": [0.0, 1.0],
            "wireless": [1.0, 0.0],
            "cotton": [0.0, 1.0],
        })
        categorizer = ZeroShotCategorizer(emb, ["Electronics", "Apparel"])
        stage = ClassifyStage(categorizer, emb)

        rec = new_record("x")
        rec["embedding"] = [0.9, 0.1]          # closest to Electronics
        rec["tags"] = ["cotton", "wireless"]    # should re-rank wireless first
        rec["discovered_tags"] = ["ankara"]
        stage.process(rec)

        self.assertEqual(rec["category"], "Electronics")
        self.assertEqual(rec["tags"][0], "wireless")
        self.assertEqual(rec["discovered_categories"], ["ankara"])

    def test_classify_requires_embedding(self):
        from semantic_search.classify import ClassifyStage, ZeroShotCategorizer
        from semantic_search.types import new_record

        emb = FakeEmbedder()
        stage = ClassifyStage(ZeroShotCategorizer(emb, ["A"]), emb)
        with self.assertRaises(ValueError):
            stage.process(new_record("x"))  # no embedding set


@unittest.skipUnless(has("rapidfuzz"), "rapidfuzz not installed")
class TestFuzzy(unittest.TestCase):
    def test_corrects_typo_to_vocabulary(self):
        from semantic_search.fuzzy import FuzzyCanonicalizer, TagVocabulary

        canon = FuzzyCanonicalizer(TagVocabulary.from_taxonomy(["wireless", "bluetooth"]))
        self.assertEqual(canon.correct_query("wireles"), "wireless")
        self.assertEqual(canon.correct_query("blutooth"), "bluetooth")

    def test_unknown_term_is_left_alone(self):
        from semantic_search.fuzzy import FuzzyCanonicalizer, TagVocabulary

        canon = FuzzyCanonicalizer(TagVocabulary.from_taxonomy(["wireless"]), threshold=85.0)
        self.assertEqual(canon.correct_query("zzzxq"), "zzzxq")

    def test_correction_accuracy_on_noisy_testset(self):
        from eval.test_robustness import correction_accuracy

        terms = ["wireless", "bluetooth", "headphones", "stainless steel", "cotton"]
        acc = correction_accuracy(terms, variants=10)
        self.assertGreaterEqual(acc, 0.5)


@unittest.skipUnless(has("rapidfuzz"), "rapidfuzz needed for dedupe")
class TestContentFilter(unittest.TestCase):
    def test_denylist_threshold_dedupe_cap(self):
        from semantic_search.filters import ContentFilter

        cf = ContentFilter(denylist=["banned"], min_score=0.3, max_tags=3,
                           dedupe_threshold=90.0, use_profanity=False)
        tags = ["wireless", "wireles", "banned", "bluetooth", "cotton", "lowscore"]
        scores = {"wireless": 0.9, "wireles": 0.9, "bluetooth": 0.8,
                 "cotton": 0.7, "lowscore": 0.1}
        out = cf.apply(tags, scores)
        self.assertNotIn("banned", out)        # denylisted
        self.assertNotIn("lowscore", out)      # below min_score
        self.assertLessEqual(len(out), 3)      # capped
        self.assertNotIn("wireles", out)       # near-duplicate of 'wireless'


@unittest.skipUnless(has("yake"), "yake not installed")
class TestKeywordCandidates(unittest.TestCase):
    def test_scores_in_unit_range(self):
        from semantic_search.candidates import KeywordCandidates

        scores = KeywordCandidates().score("wireless bluetooth noise cancelling headphones")
        self.assertTrue(scores)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in scores.values()))


@unittest.skipUnless(has("spacy"), "spacy not installed")
class TestLinguistics(unittest.TestCase):
    def test_extracts_candidates(self):
        from semantic_search.linguistics import LinguisticExtractor

        try:
            cands = LinguisticExtractor().candidates("red cotton t-shirt for men")
        except RuntimeError as e:
            self.skipTest(str(e))  # model not downloaded
        self.assertTrue(cands)


if __name__ == "__main__":
    unittest.main()

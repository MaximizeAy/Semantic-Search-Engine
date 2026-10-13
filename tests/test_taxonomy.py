import unittest

from semantic_search import taxonomy


class TestTaxonomy(unittest.TestCase):
    def test_category_slugs(self):
        self.assertIn("electronics", taxonomy.CATEGORIES)
        self.assertIn("other", taxonomy.CATEGORIES)
        self.assertEqual(len(set(taxonomy.CATEGORIES)), len(taxonomy.CATEGORIES), "must be unique")

    def test_fallback_excluded_from_classification_labels(self):
        pairs = taxonomy.classification_labels()
        slugs = [s for s, _ in pairs]
        self.assertNotIn("other", slugs)                 # fallback is not classified against
        self.assertIn("electronics", slugs)
        self.assertTrue(all(lbl and lbl[0].isupper() for _, lbl in pairs))  # readable labels

    def test_default_categories_is_a_copy(self):
        n = len(taxonomy.CATEGORIES)
        c = taxonomy.default_categories()
        c.append("tampered")
        self.assertEqual(len(taxonomy.CATEGORIES), n, "default_categories must not alias the module list")

    def test_starter_tags_nonempty_and_lowercase(self):
        self.assertGreater(len(taxonomy.STARTER_TAGS), 50)
        self.assertTrue(all(t == t.lower() for t in taxonomy.STARTER_TAGS))

    def test_build_vocabulary(self):
        v = taxonomy.build_vocabulary()
        self.assertIn("wireless", v)
        self.assertEqual(len(v), len(set(taxonomy.STARTER_TAGS)))

    def test_build_vocabulary_with_extra(self):
        v = taxonomy.build_vocabulary(extra=["artisanal"])
        self.assertIn("artisanal", v)


if __name__ == "__main__":
    unittest.main()

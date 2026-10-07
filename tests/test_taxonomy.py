import unittest

from semantic_search import taxonomy


class TestTaxonomy(unittest.TestCase):
    def test_21_top_level_categories(self):
        self.assertEqual(len(taxonomy.CATEGORIES), 21)
        self.assertIn("Electronics", taxonomy.CATEGORIES)
        self.assertEqual(len(set(taxonomy.CATEGORIES)), 21, "categories must be unique")

    def test_default_categories_is_a_copy(self):
        c = taxonomy.default_categories()
        c.append("tampered")
        self.assertEqual(len(taxonomy.CATEGORIES), 21, "default_categories must not alias the module list")

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

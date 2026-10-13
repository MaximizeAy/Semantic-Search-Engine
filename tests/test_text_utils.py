import unittest

from semantic_search.text_utils import clean_terms, is_valid_term


class TestIsValidTerm(unittest.TestCase):
    def test_keeps_real_terms(self):
        for t in ["wireless", "stainless steel", "water bottle", "noise cancelling"]:
            self.assertTrue(is_valid_term(t), t)

    def test_rejects_junk(self):
        bad = [
            "",                      # empty
            "a",                     # too short
            "2",                     # pure number
            "you", "that", "them",   # stopwords
            '<img src="x">',         # markup
            "src=",                  # code remnant
            "ahr0chm6ly9mdxjhdgvkx2ltywdlx3mymdi1xzexxziw",  # long base64-ish (> max len)
            "aGVsbG8x+/wQ==1234zz",  # mixed base64 symbols/digits (low letter ratio)
            "a b c d e f",           # too many words
            "&amp;",                 # entity
        ]
        for t in bad:
            self.assertFalse(is_valid_term(t), t)

    def test_rejects_overlong(self):
        self.assertFalse(is_valid_term("x" * 50))


class TestCleanTerms(unittest.TestCase):
    def test_filters_and_dedupes(self):
        out = clean_terms(["Wireless", "wireless", "you", "<p>", "cotton"])
        self.assertEqual(out, ["Wireless", "cotton"])


if __name__ == "__main__":
    unittest.main()

import unittest

from semantic_search.normalize import NormalizeStage, TextNormalizer
from semantic_search.types import new_record


class TestTextNormalizer(unittest.TestCase):
    def setUp(self):
        self.n = TextNormalizer()

    def test_casefold(self):
        self.assertEqual(self.n.normalize("WiReLeSS"), "wireless")

    def test_collapses_all_whitespace_kinds(self):
        # regular, non-breaking (u00a0), zero-width (u200b)
        self.assertEqual(self.n.normalize("a  b c​d"), "a b c d")

    def test_strip(self):
        self.assertEqual(self.n.normalize("  hi  "), "hi")

    def test_unicode_nfkc(self):
        # full-width characters fold to ASCII under NFKC
        self.assertEqual(self.n.normalize("ＡＢＣ"), "abc")

    def test_non_string_coerced(self):
        self.assertEqual(self.n.normalize(123), "123")

    def test_flags_off(self):
        n = TextNormalizer(normalize_case=False)
        self.assertEqual(n.normalize("ABC"), "ABC")

    def test_strips_html_tags_and_entities(self):
        out = self.n.normalize("<p>Wireless <strong>Earbuds</strong></p> &amp; case")
        self.assertEqual(out, "wireless earbuds & case")
        self.assertNotIn("strong", out)
        self.assertNotIn("<", out)

    def test_strips_urls_and_base64(self):
        out = self.n.normalize(
            'Sneaker <img src="https://x.com/AAAAAAAAAAAAAAAAAAAAAAAAAAAA.png"> noir'
        )
        self.assertNotIn("http", out)
        self.assertNotIn("img", out)
        self.assertIn("sneaker", out)
        self.assertIn("noir", out)

    def test_strips_escaped_markup(self):
        out = self.n.normalize("Shoe &lt;script&gt;bad()&lt;/script&gt; leather")
        self.assertNotIn("script", out)
        self.assertIn("leather", out)

    def test_html_strip_can_be_disabled(self):
        n = TextNormalizer(strip_html=False)
        self.assertIn("<p>", n.normalize("<p>x</p>"))


class TestNormalizeStage(unittest.TestCase):
    def test_combined_text_and_fields(self):
        rec = NormalizeStage().process(new_record("  Wireless BT ", "Noise  Cancelling"))
        self.assertEqual(rec["normalized_name"], "wireless bt")
        self.assertEqual(rec["normalized_description"], "noise cancelling")
        self.assertEqual(rec["text"], "wireless bt. noise cancelling")

    def test_empty_description(self):
        rec = NormalizeStage().process(new_record("Lamp"))
        self.assertEqual(rec["text"], "lamp")

    def test_normalize_query_helper(self):
        self.assertEqual(NormalizeStage().normalize_query("  WiReLeS  "), "wireles")


if __name__ == "__main__":
    unittest.main()

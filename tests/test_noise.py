import random
import unittest

from eval.build_testset import build_testset
from eval.noise import NoiseGenerator


class TestNoiseGenerator(unittest.TestCase):
    def test_single_char_ops_are_safe(self):
        ng = NoiseGenerator()
        # should never raise on short/edge-case words
        for w in ["a", "ab", "abc", ""]:
            ng.apply_error(w, "deletion")
            ng.apply_error(w, "transposition")
            ng.apply_error(w, "duplication")
            ng.apply_error(w, "keyboard")

    def test_generate_preserves_word_count(self):
        ng = NoiseGenerator()
        text = "wireless bluetooth headphones"
        out = ng.generate(text)
        self.assertEqual(len(out.split()), len(text.split()))

    def test_empty_text(self):
        self.assertEqual(NoiseGenerator().generate(""), "")

    def test_duplicate_two_char_word_does_not_crash(self):
        # regression: randrange(1, len-1) raised on 2-char words
        out = NoiseGenerator().duplicate_character("ab")
        self.assertEqual(len(out), 3)

    def test_transpose_three_char_word_does_not_crash(self):
        # regression: randrange(1, len-2) raised on 3-char words
        out = NoiseGenerator().transpose_characters("abc")
        self.assertEqual(sorted(out), sorted("abc"))

    def test_generate_stress_short_words(self):
        # short words ('tv', 'hd', 'usb') must never crash generate()
        ng = NoiseGenerator()
        for _ in range(200):
            ng.generate("tv hd usb 4k led")

    def test_keyboard_substitution_uses_neighbors(self):
        ng = NoiseGenerator()
        out = ng.keyboard_substitution("s")
        # 's' has neighbours; single char len<2 guards in others, but keyboard
        # substitution operates on any char with neighbours
        self.assertIn(out, list(NoiseGenerator.KEYBOARD_NEIGHBORS["s"]))


class TestBuildTestset(unittest.TestCase):
    def test_row_count_and_clean_preserved(self):
        rows = build_testset(["wireless", "bluetooth"], variants=4)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(set(r.keys()) == {"clean", "noisy"} for r in rows))
        self.assertEqual({r["clean"] for r in rows}, {"wireless", "bluetooth"})

    def test_seed_is_reproducible(self):
        a = build_testset(["wireless headphones"], variants=5, seed=7)
        b = build_testset(["wireless headphones"], variants=5, seed=7)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()

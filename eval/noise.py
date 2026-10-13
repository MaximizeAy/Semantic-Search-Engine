"""Synthetic typo generator (eval-only).

Carried over from the original notebook. Turns clean product terms into
realistic noisy variants so we can build a test set that measures typo
robustness. NOT part of the serving pipeline.

Error types: character deletion, transposition, duplication, and
keyboard-neighbour substitution.
"""

import random


class NoiseGenerator:
    KEYBOARD_NEIGHBORS = {
        "q": "was",
        "w": "qase",
        "e": "wsdr",
        "r": "edft",
        "t": "rfgy",
        "y": "tugh",
        "u": "yhji",
        "i": "ujko",
        "o": "iklp",
        "p": "ol",
        "a": "qwsz",
        "s": "qweadzx",
        "d": "wersfxc",
        "f": "ertgdcv",
        "g": "rtyfhvb",
        "h": "tyugjbn",
        "j": "yuihkm",
        "k": "uiojln",
        "l": "opk",
        "z": "asx",
        "x": "zsdc",
        "c": "xdfv",
        "v": "cfgb",
        "b": "vghn",
        "n": "bhjm",
        "m": "njk",
    }

    def __init__(
        self,
        min_errors=1,
        max_errors=2,
        deletion_weight=0.35,
        transposition_weight=0.20,
        duplication_weight=0.10,
        keyboard_weight=0.35,
    ):
        self.min_errors = min_errors
        self.max_errors = max_errors

        self.error_types = [
            ("deletion", deletion_weight),
            ("transposition", transposition_weight),
            ("duplication", duplication_weight),
            ("keyboard", keyboard_weight),
        ]

    def delete_character(self, word):
        if len(word) <= 2:
            return word

        index = random.randrange(1, len(word) - 1)

        return word[:index] + word[index + 1:]

    def transpose_characters(self, word):
        if len(word) < 3:
            return word

        # max(2, ...) keeps a valid range for 3-char words (randrange(1, 1)
        # would otherwise raise); behaviour is unchanged for longer words.
        index = random.randrange(1, max(2, len(word) - 2))

        chars = list(word)

        chars[index], chars[index + 1] = (
            chars[index + 1],
            chars[index],
        )

        return "".join(chars)

    def duplicate_character(self, word):
        if len(word) < 2:
            return word

        # max(2, ...) keeps a valid range for 2-char words (randrange(1, 1)
        # would otherwise raise); behaviour is unchanged for longer words.
        index = random.randrange(1, max(2, len(word) - 1))

        return (
            word[:index]
            + word[index]
            + word[index:]
        )

    def keyboard_substitution(self, word):
        candidates = [
            i for i, char in enumerate(word)
            if char.lower() in self.KEYBOARD_NEIGHBORS
        ]

        if not candidates:
            return word

        index = random.choice(candidates)

        char = word[index].lower()

        replacement = random.choice(
            self.KEYBOARD_NEIGHBORS[char]
        )

        if word[index].isupper():
            replacement = replacement.upper()

        return (
            word[:index]
            + replacement
            + word[index + 1:]
        )

    def apply_error(self, word, error_type):
        if error_type == "deletion":
            return self.delete_character(word)

        if error_type == "transposition":
            return self.transpose_characters(word)

        if error_type == "duplication":
            return self.duplicate_character(word)

        if error_type == "keyboard":
            return self.keyboard_substitution(word)

        return word

    def generate(self, text):
        words = text.split()

        if not words:
            return text

        number_of_errors = random.randint(
            self.min_errors,
            min(self.max_errors, len(words))
        )

        selected_words = random.sample(
            range(len(words)),
            number_of_errors
        )

        for index in selected_words:

            error_type = random.choices(
                [x[0] for x in self.error_types],
                weights=[x[1] for x in self.error_types],
                k=1
            )[0]

            words[index] = self.apply_error(
                words[index],
                error_type
            )

        return " ".join(words)

"""Shared text hygiene for candidate terms / tags.

Defense-in-depth against junk that survives HTML stripping and NLP extraction:
code/markup remnants, URLs, pure numbers, base64 blobs, generic stopwords, and
overlong phrases. Keeps real multi-word product terms.
"""

from __future__ import annotations

import re
from typing import Iterable, List

# Small generic stopword set — single-word noun chunks that are never useful tags.
STOPWORDS = {
    "you", "your", "yours", "we", "us", "our", "they", "them", "their", "it",
    "its", "this", "that", "these", "those", "there", "here", "what", "which",
    "who", "whom", "the", "a", "an", "and", "or", "but", "if", "then", "so",
    "to", "of", "in", "on", "for", "with", "as", "at", "by", "from", "all",
    "any", "each", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "than", "too", "very", "can", "will", "just",
}

_HAS_MARKUP = re.compile(r'[<>/=\\"{}]|&[a-z]+;|http|src=|www\.')
_MAX_LEN = 40
_MAX_WORDS = 5


def is_valid_term(term: str) -> bool:
    t = (term or "").strip()
    if len(t) < 2 or len(t) > _MAX_LEN:
        return False
    if len(t.split()) > _MAX_WORDS:
        return False
    if _HAS_MARKUP.search(t):
        return False
    letters = sum(c.isalpha() for c in t)
    if letters == 0:                      # pure numbers / symbols
        return False
    if letters / len(t) < 0.5:            # base64-ish / mostly non-letters
        return False
    if t.casefold() in STOPWORDS:
        return False
    return True


def clean_terms(terms: Iterable[str]) -> List[str]:
    """Filter to valid terms, de-duplicated case-insensitively, order preserved."""
    out: List[str] = []
    seen = set()
    for t in terms:
        if is_valid_term(t):
            key = t.casefold()
            if key not in seen:
                seen.add(key)
                out.append(t)
    return out

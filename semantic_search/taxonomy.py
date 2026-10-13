"""Seed taxonomy and starter vocabulary.

- ``CATEGORIES`` — the marketplace's category slugs, the fixed label set for
  zero-shot classification (stage 6). ``other`` is the fallback for products
  that don't clearly fit, not a classified label.
- ``CATEGORY_LABELS`` — a readable phrase per slug used for the embedding
  comparison (slugs like ``health-beauty`` embed poorly; phrases don't). The
  stored value is always the slug.
- ``STARTER_TAGS`` — a curated starter tag vocabulary (materials, features,
  product types, colors, descriptors) so the fuzzy stage (stage 4) has an anchor
  on first run. Data-discovered terms are added to this at index time.

Grow them from your own catalog (see docs/PIPELINE.md §3-4).
"""

from __future__ import annotations

from typing import List, Tuple

from .fuzzy import TagVocabulary

# Marketplace category slugs. 'other' is the catch-all fallback.
CATEGORIES: List[str] = [
    "accessories",
    "automotive",
    "baby-products",
    "computing",
    "electronics",
    "fashion",
    "footwear",
    "groceries",
    "health-beauty",
    "home-office",
    "phones-tablets",
    "sporting-goods",
    "other",
]

FALLBACK_CATEGORY = "other"

# Default similarity below which a product falls back to FALLBACK_CATEGORY.
# Tune on real data: inspect scores after indexing and adjust.
DEFAULT_CATEGORY_THRESHOLD = 0.25

# Readable label per slug for the embedding comparison (excludes the fallback).
CATEGORY_LABELS = {
    "accessories": "Accessories",
    "automotive": "Automotive, car and vehicle parts",
    "baby-products": "Baby and toddler products",
    "computing": "Computing, laptops and PCs",
    "electronics": "Electronics and gadgets",
    "fashion": "Fashion, clothing and apparel",
    "footwear": "Footwear, shoes and sneakers",
    "groceries": "Groceries, food and drinks",
    "health-beauty": "Health, beauty and personal care",
    "home-office": "Home, office and furniture",
    "phones-tablets": "Phones, tablets and mobile devices",
    "sporting-goods": "Sporting goods and fitness",
}


def classification_labels() -> List[Tuple[str, str]]:
    """(slug, label) pairs to classify against — excludes the fallback."""
    return [(slug, CATEGORY_LABELS[slug]) for slug in CATEGORIES if slug in CATEGORY_LABELS]

# Curated starter tags — common e-commerce descriptors across categories.
STARTER_TAGS: List[str] = [
    # materials
    "cotton", "leather", "stainless steel", "wood", "bamboo", "plastic",
    "silicone", "ceramic", "glass", "wool", "polyester", "nylon", "aluminum",
    "denim", "linen", "rubber", "carbon fiber",
    # features
    "wireless", "bluetooth", "waterproof", "water resistant", "rechargeable",
    "portable", "foldable", "adjustable", "noise cancelling", "fast charging",
    "non-stick", "insulated", "breathable", "lightweight", "ergonomic",
    "reusable", "eco friendly", "organic", "handmade", "dishwasher safe",
    "wireless charging", "touchscreen", "cordless",
    # product types
    "headphones", "earbuds", "speaker", "charger", "cable", "laptop",
    "keyboard", "mouse", "monitor", "phone case", "water bottle", "mug",
    "backpack", "wallet", "t-shirt", "jeans", "sneakers", "jacket", "dress",
    "watch", "sunglasses", "blender", "kettle", "toaster", "lamp", "pillow",
    "blanket", "towel", "knife", "frying pan", "cookware", "toy", "puzzle",
    "book", "notebook", "pen", "desk", "chair", "stroller", "diapers",
    # colors
    "black", "white", "red", "blue", "green", "gray", "pink", "gold", "silver",
    "beige", "navy",
    # descriptors
    "premium", "vintage", "slim", "mini", "compact", "large", "smart", "led",
    "usb", "hd", "4k", "gaming", "unisex", "kids", "travel",
]


def build_vocabulary(extra: List[str] | None = None) -> TagVocabulary:
    """A TagVocabulary seeded with STARTER_TAGS (+ any extras you pass)."""
    vocab = TagVocabulary.from_taxonomy(STARTER_TAGS)
    for term in extra or []:
        vocab.add(term, source="taxonomy")
    return vocab


def default_categories() -> List[str]:
    """The fixed category label set (Google Product Taxonomy, top level)."""
    return list(CATEGORIES)

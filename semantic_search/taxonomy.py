"""Seed taxonomy and starter vocabulary.

- ``CATEGORIES`` — the 21 top-level categories of the Google Product Taxonomy,
  used as the fixed label set for zero-shot classification (stage 6).
- ``STARTER_TAGS`` — a curated starter tag vocabulary (materials, features,
  product types, colors, descriptors) so the fuzzy stage (stage 4) has an anchor
  on first run. Data-discovered terms are added to this at index time.

Both are intended as a backbone, not a final list. Grow them from your own
catalog (see docs/PIPELINE.md §3-4).
"""

from __future__ import annotations

from typing import List

from .fuzzy import TagVocabulary

# Google Product Taxonomy — top level (21 categories).
# Source: https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt
CATEGORIES: List[str] = [
    "Animals & Pet Supplies",
    "Apparel & Accessories",
    "Arts & Entertainment",
    "Baby & Toddler",
    "Business & Industrial",
    "Cameras & Optics",
    "Electronics",
    "Food, Beverages & Tobacco",
    "Furniture",
    "Hardware",
    "Health & Beauty",
    "Home & Garden",
    "Luggage & Bags",
    "Mature",
    "Media",
    "Office Supplies",
    "Religious & Ceremonial",
    "Software",
    "Sporting Goods",
    "Toys & Games",
    "Vehicles & Parts",
]

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

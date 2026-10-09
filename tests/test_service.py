"""TaggingService flows, with the pipeline and store faked out."""

import unittest

from semantic_search.service import TaggingService, merge_tags

from ._fakes import FakePipeline, FakeSearchPipeline, FakeStore, FakeEmbedder

RESULT = {
    "tags": ["wireless", "bluetooth"],
    "category": "Electronics",
    "discovered_tags": ["ankara"],
    "embedding": [0.1, 0.2, 0.3, 0.4],
}


def make_service(store):
    svc = TaggingService(store=store, embedder=FakeEmbedder())
    svc.pipeline = FakePipeline(RESULT)
    svc.search_pipeline = FakeSearchPipeline([{"id": 1, "score": 0.9}])
    return svc


class TestTagProduct(unittest.TestCase):
    def test_persists_and_writes_source_tags(self):
        store = FakeStore()
        svc = make_service(store)
        rec = svc.tag_product("Wireless Earbuds", "noise cancelling", external_id="uuid-1")
        self.assertEqual(rec["tags"], ["wireless", "bluetooth"])
        self.assertEqual(len(store.upserts), 1)
        self.assertEqual(store.tag_writes, [("uuid-1", ["wireless", "bluetooth"])])

    def test_no_external_id_skips_persistence(self):
        store = FakeStore()
        svc = make_service(store)
        rec = svc.tag_product("Lamp", "")             # no external_id to link back
        self.assertEqual(rec["tags"], ["wireless", "bluetooth"])  # still returns tags
        self.assertEqual(store.upserts, [])           # but nothing persisted (avoids NULL-keyed rows)
        self.assertEqual(store.tag_writes, [])

    def test_merge_preserves_seller_tags_and_drops_stale(self):
        store = FakeStore(
            indexed={"uuid-1": {"tags": ["oldgen"]}},          # we generated 'oldgen' last time
            source_tags={"uuid-1": ["seller1", "oldgen"]},     # current column
        )
        svc = make_service(store)
        svc.tag_product("x", "", external_id="uuid-1")
        # seller1 kept, oldgen dropped (no longer generated), new gen appended
        self.assertEqual(store.tag_writes, [("uuid-1", ["seller1", "wireless", "bluetooth"])])

    def test_skips_write_when_no_change(self):
        store = FakeStore(source_tags={"uuid-1": ["wireless", "bluetooth"]})
        svc = make_service(store)
        svc.tag_product("x", "", external_id="uuid-1")
        self.assertEqual(store.tag_writes, [])         # merged == current -> no write

    def test_persist_false_writes_nothing(self):
        store = FakeStore()
        svc = make_service(store)
        svc.tag_product("Lamp", "", external_id="uuid-9", persist=False)
        self.assertEqual(store.upserts, [])
        self.assertEqual(store.tag_writes, [])

    def test_write_source_tags_flag_off(self):
        store = FakeStore()
        svc = make_service(store)
        svc.tag_product("Lamp", "", external_id="uuid-2", write_source_tags=False)
        self.assertEqual(len(store.upserts), 1)
        self.assertEqual(store.tag_writes, [])


class TestSearch(unittest.TestCase):
    def test_delegates_to_search_pipeline(self):
        svc = make_service(FakeStore())
        out = svc.search("wireles earbuds", k=3)
        self.assertEqual(out, [{"id": 1, "score": 0.9}])
        self.assertEqual(svc.search_pipeline.calls, [("wireles earbuds", 3)])


class TestRebuild(unittest.TestCase):
    def test_iterates_catalog_and_counts(self):
        rows = [
            {"id": "u1", "name": "A", "description": "d"},
            {"id": "u2", "name": "B", "description": None},
        ]
        store = FakeStore(source_rows=rows)
        svc = make_service(store)
        n = svc.rebuild()
        self.assertEqual(n, 2)
        self.assertEqual(len(store.upserts), 2)
        self.assertEqual([w[0] for w in store.tag_writes], ["u1", "u2"])

    def test_skips_products_indexed_since_last_change(self):
        rows = [
            {"id": "u1", "name": "A", "updated_at": "2026-10-01T00:00:00+00:00"},  # stale -> skip
            {"id": "u2", "name": "B", "updated_at": "2026-10-05T00:00:00+00:00"},  # changed -> do
        ]
        store = FakeStore(
            source_rows=rows,
            indexed_at_map={
                "u1": "2026-10-02T00:00:00+00:00",  # indexed after its change
                "u2": "2026-10-03T00:00:00+00:00",  # indexed before its change
            },
        )
        svc = make_service(store)
        n = svc.rebuild()
        self.assertEqual(n, 1)
        self.assertEqual([w[0] for w in store.tag_writes], ["u2"])


class TestMergeTags(unittest.TestCase):
    def test_union_dedup_casefold(self):
        self.assertEqual(
            merge_tags(["Wireless"], [], ["wireless", "bluetooth"]),
            ["Wireless", "bluetooth"],  # case-insensitive dedup keeps the existing form
        )

    def test_drops_stale_generated_keeps_seller(self):
        self.assertEqual(
            merge_tags(["seller", "oldgen"], ["oldgen"], ["new"]),
            ["seller", "new"],
        )

    def test_cap(self):
        out = merge_tags([], [], [f"t{i}" for i in range(40)], cap=25)
        self.assertEqual(len(out), 25)


if __name__ == "__main__":
    unittest.main()

"""TaggingService flows, with the pipeline and store faked out."""

import unittest

from semantic_search.service import TaggingService

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

    def test_no_external_id_skips_source_write(self):
        store = FakeStore()
        svc = make_service(store)
        svc.tag_product("Lamp", "")
        self.assertEqual(len(store.upserts), 1)      # still indexed
        self.assertEqual(store.tag_writes, [])        # but nothing written to public.products

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


if __name__ == "__main__":
    unittest.main()

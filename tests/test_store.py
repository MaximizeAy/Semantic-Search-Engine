"""SupabaseVectorStore DB-interaction logic, with a fake supabase client.

This is the write path that /tag uses, so it is worth pinning down precisely
before pointing it at the real public.products table.
"""

import unittest

from semantic_search.store import SupabaseVectorStore

from ._fakes import FakeSupabaseClient


def make_store(**kw):
    store = SupabaseVectorStore(url="http://x", key="k", **kw)
    return store


def tables(client):
    return [c for c in client.calls if hasattr(c, "op")]


class TestUpsert(unittest.TestCase):
    def test_upsert_targets_engine_schema_with_expected_row(self):
        store = make_store()
        store._client = FakeSupabaseClient()
        store.upsert({
            "external_id": "u1", "name": "N", "description": "D",
            "category": "Electronics", "discovered_categories": ["x"],
            "tags": ["a", "b"], "embedding": [0.1, 0.2],
        })
        t = tables(store._client)[0]
        self.assertEqual(t.parent.name, "semantic_search")   # engine schema
        self.assertEqual(t.name, "products")
        self.assertEqual(t.op, "upsert")
        self.assertEqual(t.on_conflict, "external_id")
        self.assertEqual(set(t.payload), {
            "external_id", "name", "description", "category",
            "discovered_categories", "tags", "embedding",
        })


class TestWriteSourceTags(unittest.TestCase):
    def test_updates_public_products_tags_by_id(self):
        store = make_store()
        store._client = FakeSupabaseClient()
        store.write_source_tags("uuid-1", ["wireless", "bluetooth"])
        t = tables(store._client)[0]
        self.assertEqual(t.parent.name, "public")            # the marketplace schema
        self.assertEqual(t.name, "products")
        self.assertEqual(t.op, "update")
        self.assertEqual(t.payload, {"tags": ["wireless", "bluetooth"]})
        self.assertIn(("eq", "id", "uuid-1"), t.filters)     # scoped to the one row


class TestIterSourceProducts(unittest.TestCase):
    def test_reads_active_from_public(self):
        rows = [{"id": "u1", "name": "A"}, {"id": "u2", "name": "B"}]
        store = make_store()
        store._client = FakeSupabaseClient(source_rows=rows)
        out = list(store.iter_source_products())
        self.assertEqual(out, rows)
        t = tables(store._client)[0]
        self.assertEqual(t.parent.name, "public")
        self.assertEqual(t.op, "select")
        self.assertIn(("eq", "is_active", True), t.filters)

    def test_since_adds_gte_filter(self):
        store = make_store()
        store._client = FakeSupabaseClient(source_rows=[{"id": "u1"}])
        list(store.iter_source_products(since="2026-10-01T00:00:00Z", only_active=False))
        t = tables(store._client)[0]
        self.assertIn(("gte", "updated_at", "2026-10-01T00:00:00Z"), t.filters)
        self.assertFalse(any(f[0] == "eq" and f[1] == "is_active" for f in t.filters))


class TestSearchRpc(unittest.TestCase):
    def test_search_calls_rpc_in_engine_schema(self):
        store = make_store()
        store._client = FakeSupabaseClient(rpc_result=[{"id": 1, "score": 0.8}])
        out = store.search([0.1, 0.2], k=5)
        self.assertEqual(out, [{"id": 1, "score": 0.8}])
        rpc = [c for c in store._client.calls if isinstance(c, tuple) and c[0] == "rpc"][0]
        self.assertEqual(rpc[1], "semantic_search")
        self.assertEqual(rpc[2], "search_products")
        self.assertEqual(rpc[3], {"query_embedding": [0.1, 0.2], "match_count": 5})


class TestConstructorGuards(unittest.TestCase):
    def test_missing_credentials_raises(self):
        import os
        saved = {k: os.environ.pop(k, None) for k in ("SUPABASE_URL", "SUPABASE_KEY")}
        try:
            with self.assertRaises(RuntimeError):
                SupabaseVectorStore()
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()

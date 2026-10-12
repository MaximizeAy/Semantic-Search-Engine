"""Embedder backends — no torch, no network (urlopen is faked)."""

import io
import json
import os
import unittest
from unittest import mock

from semantic_search.embed import Embedder, WorkersAIBackend, _l2_normalize


class TestL2Normalize(unittest.TestCase):
    def test_unit_length(self):
        v = _l2_normalize([3.0, 4.0])
        self.assertAlmostEqual(sum(x * x for x in v), 1.0, places=6)

    def test_zero_vector_safe(self):
        self.assertEqual(_l2_normalize([0.0, 0.0]), [0.0, 0.0])


class TestWorkersAIBackend(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"CF_ACCOUNT_ID": "acct", "CF_AI_TOKEN": "tok"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def _fake_response(self, payload):
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    def test_encode_parses_and_normalizes(self):
        backend = WorkersAIBackend()
        payload = {"success": True, "result": {"data": [[3.0, 4.0]]}}
        with mock.patch("urllib.request.urlopen", return_value=self._fake_response(payload)):
            out = backend.encode(["hello"])
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(sum(x * x for x in out[0]), 1.0, places=6)

    def test_missing_credentials_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                WorkersAIBackend()

    def test_error_payload_raises(self):
        backend = WorkersAIBackend()
        payload = {"success": False, "errors": ["nope"]}
        with mock.patch("urllib.request.urlopen", return_value=self._fake_response(payload)):
            with self.assertRaises(RuntimeError):
                backend.encode(["x"])


class TestEmbedderBackendSelection(unittest.TestCase):
    def test_selects_workers_ai(self):
        with mock.patch.dict(os.environ, {"CF_ACCOUNT_ID": "a", "CF_AI_TOKEN": "t"}):
            e = Embedder(backend="workers_ai")
            self.assertIsInstance(e.backend, WorkersAIBackend)

    def test_default_backend_name(self):
        e = Embedder(backend="sentence_transformers")
        # don't instantiate (would import torch); just check the selected name
        self.assertEqual(e.backend_name, "sentence_transformers")


if __name__ == "__main__":
    unittest.main()

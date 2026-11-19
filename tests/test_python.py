import tempfile
import unittest
from pathlib import Path

import numpy as np

from vector_search import Index


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(98)
        self.rows = self.rng.normal(size=(179, 12)).astype("f4")
        self.queries = self.rows[::11, ::-1]
        self.index = Index(self.rows, tree=True)

    def test_numpy_oracle(self):
        index = Index(self.rows.astype("f8")[::-1])
        distances = ((self.rows[::-1].astype("f8") - self.queries[:, None]) ** 2).sum(axis=2)
        want = np.argsort(distances, axis=1, kind="stable")[:, :10]
        ids, scores = index.search(self.queries, threads=3)
        np.testing.assert_array_equal(ids, want)
        np.testing.assert_allclose(scores, np.take_along_axis(distances, want, axis=1))

    def test_approximate_scores_and_distinct_ids(self):
        ids, scores = self.index.search(self.queries, k=5, budget=23)
        for q, matches, ds in zip(self.queries, ids, scores):
            self.assertEqual(len(set(matches)), 5)
            self.assertTrue(np.all(ds[:-1] <= ds[1:]))
            np.testing.assert_allclose(ds, ((self.rows[matches].astype("f8") - q) ** 2).sum(axis=1))

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = str(Path(tmp) / "index")
            for tree in (False, True):
                idx = Index(self.rows, tree=tree)
                idx.save(p)
                restored = Index.load(p)
                self.assertEqual(restored.has_tree, tree)
                for budget in ([0, 32] if tree else [0]):
                    for a, b in zip(restored.search(self.queries, budget=budget),
                                    idx.search(self.queries, budget=budget)):
                        np.testing.assert_array_equal(a, b)
            raw = Path(p).read_bytes()
            for bad in [raw[:-1], b"bad", raw + b"x", raw[:8] + b"\xff" * 16 + raw[24:]]:
                Path(p).write_bytes(bad)
                with self.assertRaises(RuntimeError):
                    Index.load(p)

    def test_validation(self):
        for bad in [np.array([1]), [[np.nan]], [[np.inf]], np.empty((0, 7)), np.empty((2, 0))]:
            with self.assertRaises(ValueError):
                Index(bad)
        for bad in [self.queries[0], self.queries[:, :1], np.full((1, 12), np.inf)]:
            with self.assertRaises(ValueError):
                self.index.search(bad)
        for kwargs in [dict(k=0), dict(threads=0), dict(budget=2)]:
            with self.assertRaises(ValueError):
                self.index.search(self.queries, **kwargs)
        with self.assertRaises(ValueError):
            Index(self.rows).search(self.queries, budget=30)
        self.assertEqual(self.index.search(self.queries[:0])[0].shape, (0, 10))

    def test_hnsw_roundtrip_and_legacy_file(self):
        import struct

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "index.vsi"
            # Construct the previous writer's file independently.
            path.write_bytes(b"VSIDX01\0" + struct.pack("<QQQ", *self.rows.shape, 1)
                             + self.rows.astype("<f4").tobytes())
            legacy = Index.load(str(path))
            self.assertTrue(legacy.has_tree)
            self.assertFalse(legacy.has_hnsw)
            graph = Index(self.rows, True, hnsw=True, M=6, efConstruction=40)
            graph.save(str(path))
            restored = Index.load(str(path))
            self.assertTrue(restored.has_hnsw and restored.has_tree)
            for options in ({}, {"budget": 32}, {"efSearch": 20}):
                for a, b in zip(graph.search(self.queries, **options),
                                restored.search(self.queries, threads=3, **options)):
                    np.testing.assert_array_equal(a, b)
            raw = path.read_bytes()
            for bad in [raw[:-1], raw[:32] + struct.pack("<Q", 1) + raw[40:]]:
                path.write_bytes(bad)
                with self.assertRaises(RuntimeError):
                    Index.load(str(path))

    def test_graph_options_and_cardinality(self):
        for options in [dict(M=1), dict(M=257), dict(M=8, efConstruction=7)]:
            with self.assertRaises(ValueError):
                Index(self.rows, hnsw=True, **options)
        with self.assertRaises(ValueError):
            self.index.search(self.queries, efSearch=40)
        graph = Index(np.zeros((45, 7)), True, hnsw=True, M=2, efConstruction=4)
        q = np.zeros((1, 7))
        for opts in [dict(efSearch=9), dict(efSearch=20, budget=15)]:
            with self.assertRaises(ValueError):
                graph.search(q, **opts)
        ids, scores = graph.search(q, k=50, efSearch=45)
        np.testing.assert_array_equal(ids, np.arange(45)[None, :])
        np.testing.assert_array_equal(scores, np.zeros_like(scores))

    def test_trace_matches_search_and_records_first_discoveries(self):
        graph = Index(self.rows, hnsw=True, M=4, efConstruction=24)
        for ef in (10, 32, 100):
            ids, scores = graph.search(self.queries, k=10, efSearch=ef)
            for q, expected_ids, expected_scores in zip(self.queries, ids, scores):
                trace = graph.trace_hnsw(q, efSearch=ef)
                np.testing.assert_array_equal(trace["ids"], expected_ids)
                np.testing.assert_array_equal(trace["squared_distances"], expected_scores)
                examined = trace["examined"]
                self.assertEqual(len(set(examined)), len(examined))
                self.assertTrue(all(0 <= row < graph.size for row in examined))
                seen = {examined[0]}
                self.assertEqual([dst for _, dst in trace["discovery_edges"]], examined[1:])
                for src, dst in trace["discovery_edges"]:
                    self.assertIn(src, seen)
                    seen.add(dst)
                if not trace["exact_fallback"]:
                    self.assertTrue(set(trace["ids"]).issubset(seen))

    def test_trace_includes_rejected_scores_and_separates_recovery(self):
        pair = Index([[0], [5]], hnsw=True)
        trace = pair.trace_hnsw([0], k=1, efSearch=1)
        self.assertEqual(trace["ids"], [0])
        self.assertEqual(trace["examined"], [0, 1])
        self.assertEqual(trace["discovery_edges"], [(0, 1)])
        self.assertFalse(trace["exact_fallback"])

        duplicates = Index(np.zeros((45, 2)), hnsw=True, M=2, efConstruction=4)
        trace = duplicates.trace_hnsw([0, 0], k=45, efSearch=45)
        self.assertTrue(trace["exact_fallback"])
        self.assertLess(len(trace["examined"]), 45)
        self.assertEqual(trace["ids"], list(range(45)))

    def test_trace_rejects_invalid_requests(self):
        graph = Index(self.rows, hnsw=True)
        for query in [self.queries, [0], [np.nan] * graph.dimension]:
            with self.assertRaises(ValueError):
                graph.trace_hnsw(query)
        for options in [dict(k=0), dict(efSearch=9)]:
            with self.assertRaises(ValueError):
                graph.trace_hnsw(self.queries[0], **options)
        with self.assertRaises(ValueError):
            self.index.trace_hnsw(self.queries[0])

    def test_ties_and_ownership(self):
        rows = np.zeros((57, 3), dtype="f4")
        idx = Index(rows, tree=True)
        rows[:] = 9
        ids, distances = idx.search(np.zeros((1, 3)), budget=57, k=99)
        np.testing.assert_array_equal(ids[0], np.arange(57))
        np.testing.assert_array_equal(distances[0], np.zeros(57))


if __name__ == "__main__":
    unittest.main()

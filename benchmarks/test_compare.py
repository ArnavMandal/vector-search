"""Exercise a calibrated width sweep against an independent exhaustive reference."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

import faiss
import numpy as np
import vector_search

from compare import dataset, hit_counts


class BenchmarkTest(unittest.TestCase):
    def test_distinct_hits(self):
        reference = np.arange(10).reshape(1, 10)
        np.testing.assert_array_equal(hit_counts(np.zeros_like(reference), reference), [1])

    def test_calibration_and_held_out_scores(self):
        runner = Path(__file__).with_name("compare.py")
        # On this fixture the native graph crosses 95% at different widths on the two splits.
        target = 0.95
        cmd = [sys.executable, str(runner), "--rows", "500", "--dims", "12", "--queries", "19",
               "--calibration-queries", "13", "--batch-size", "8", "--threads", "1", "--repeats", "1",
               "--M", "4", "--efConstruction", "16", "--efSearch", "128", "32", "16", "16",
               "--recall-target", str(target), "--engines", "faiss-flat", "native-hnsw", "faiss-hnsw"]
        records = [json.loads(line) for line in subprocess.check_output(cmd, text=True).splitlines()]
        flat, *graphs = records
        self.assertEqual(flat["engine"], "faiss-flat")
        self.assertEqual(flat["recall_at_10"], 1.0)
        self.assertEqual(flat["index_core_mib"] * 2**20, 500 * 12 * 4)

        rows, queries = dataset(500, 12, 32, 42)
        delta = rows.astype("float64")[None, :, :] - queries[:, None, :]
        truth = np.argsort(np.sum(delta * delta, axis=2), axis=1)[:, :10]
        index = vector_search.Index(rows, hnsw=True, M=4, efConstruction=16)
        faiss.omp_set_num_threads(1)
        faiss_index = faiss.IndexHNSWFlat(12, 4)
        faiss_index.hnsw.efConstruction = 16
        faiss_index.add(rows)
        for engine in ("native-hnsw", "faiss-hnsw"):
            sweep = [r for r in graphs if r["engine"] == engine]
            self.assertEqual([r["efSearch"] for r in sweep], [16, 32, 128])
            eligible = [r["efSearch"] for r in sweep if r["calibration_recall_at_10"] >= target]
            self.assertEqual([r["efSearch"] for r in sweep if r["selected"]], eligible[:1])
            for r in sweep:
                if engine == "native-hnsw":
                    ids, _ = index.search(queries, k=10, efSearch=r["efSearch"], threads=1)
                else:
                    faiss_index.hnsw.efSearch = r["efSearch"]
                    _, ids = faiss_index.search(queries, 10)
                hits = [len(set(found) & set(ref)) for found, ref in zip(ids, truth)]
                self.assertAlmostEqual(r["recall_at_10"], sum(hits[:19]) / 190)
                self.assertAlmostEqual(r["calibration_recall_at_10"], sum(hits[19:]) / 130)
                self.assertEqual(r["test_hits_histogram"], np.bincount(hits[:19], minlength=11).tolist())


if __name__ == "__main__":
    unittest.main()

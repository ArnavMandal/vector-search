import argparse
import platform
import statistics
from time import perf_counter

import numpy as np

from vector_search import Index


def positive(s):
    n = int(s)
    if n <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return n


def main():
    p = argparse.ArgumentParser(description="Measure search cost against exact top-k neighbors")
    for name, default in [("rows", 12000), ("dim", 32), ("queries", 200), ("k", 10),
                          ("threads", 4), ("repeats", 5)]:
        p.add_argument("--" + name, type=positive, default=default)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budgets", type=positive, nargs="+", default=[64, 256, 1024])
    p.add_argument("--distribution", choices=["clustered", "gaussian"], default="clustered")
    a = p.parse_args()
    k = min(a.k, a.rows)
    if min(a.budgets) < k:
        p.error("every budget must be at least min(k, rows)")
    rng = np.random.default_rng(a.seed)
    centers = rng.normal(0, 2, (32, a.dim)).astype("f4")

    def sample(n):
        noise = rng.standard_normal((n, a.dim), dtype="f4")
        if a.distribution == "clustered":
            noise += centers[rng.integers(len(centers), size=n)]
        return noise

    rows, queries = sample(a.rows), sample(a.queries)
    print(f"platform={platform.system()}-{platform.machine()} python={platform.python_version()} "
          f"numpy={np.__version__}")
    print(f"distribution={a.distribution} rows={a.rows} dim={a.dim} queries={a.queries} "
          f"k={k} threads={a.threads} repeats={a.repeats} seed={a.seed}")
    print("mode       build_ms   index_MiB   p50_us   p95_us   batch_qps   recall@k")
    truth = None
    for tree, budgets in [(False, [0]), (True, a.budgets)]:
        start = perf_counter()
        index = Index(rows, tree=tree)
        build_ms = (perf_counter() - start) * 1000
        if truth is None:
            truth = index.search(queries, k=k)[0]
        for budget in budgets:
            ids, _ = index.search(queries, k=k, budget=budget, threads=a.threads)
            recall = sum(len(set(x) & set(y)) for x, y in zip(truth, ids)) / ids.size
            latencies = []
            for i in range(len(queries)):
                query = queries[i:i + 1]
                start = perf_counter()
                index.search(query, k=k, budget=budget)
                latencies.append((perf_counter() - start) * 1e6)
            times = []
            for _ in range(a.repeats):
                start = perf_counter()
                index.search(queries, k=k, budget=budget, threads=a.threads)
                times.append(perf_counter() - start)
            name = f"tree/{budget}" if tree else "exact"
            p50, p95 = np.percentile(latencies, [50, 95])
            print(f"{name:<10} {build_ms:8.2f} {index.storage_bytes / 2**20:11.3f} "
                  f"{p50:8.1f} {p95:8.1f} {len(queries)/statistics.median(times):11.0f} "
                  f"{recall:10.3f}")


if __name__ == "__main__":
    main()

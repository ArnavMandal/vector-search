"""Run each CPU index in a fresh process. Write one JSON record per measured index."""
import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
from time import perf_counter

import numpy as np

ENGINES = ("faiss-flat", "faiss-hnsw", "native-flat", "native-hnsw")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", nargs="+", type=int, default=[10000, 100000, 1000000])
    p.add_argument("--dims", nargs="+", type=int, default=[32, 128, 384])
    p.add_argument("--engines", nargs="+", choices=ENGINES, default=ENGINES)
    p.add_argument("--queries", type=int, default=64)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--M", type=int, default=4)
    p.add_argument("--efConstruction", type=int, default=16)
    p.add_argument("--efSearch", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", help="Append completed measurements in JSON Lines format")
    p.add_argument("--worker", choices=ENGINES, help=argparse.SUPPRESS)
    a = p.parse_args()
    if min(a.rows) < 10 or min(*a.dims, a.queries, a.threads, a.repeats) < 1:
        p.error("require at least 10 rows and positive dimensions, queries, threads and repeats")
    if not 2 <= a.M <= 256 or a.efConstruction < a.M or a.efSearch < 10 or a.seed < 0:
        p.error("require 2 <= M <= 256, efConstruction >= M, efSearch >= 10 and seed >= 0")
    return a


def dataset(n, d, count, seed):
    # Independent streams keep the corpus identical when only query count changes.
    streams = np.random.SeedSequence([seed, n, d]).spawn(3)
    centers = np.random.default_rng(streams[0]).normal(0, 2, (64, d)).astype("f4")

    def sample(rows, stream):
        rng = np.random.default_rng(stream)
        data = np.empty((rows, d), dtype="f4")
        for start in range(0, rows, 4096):
            block = data[start:start + 4096]
            rng.standard_normal(block.shape, dtype="f4", out=block)
            block += centers[rng.integers(len(centers), size=len(block))]
        return data

    return sample(n, streams[1]), sample(count, streams[2])


def measure(a):
    import faiss
    import psutil

    n, d, engine = a.rows[0], a.dims[0], a.worker
    faiss.omp_set_num_threads(1)  # single-thread insertion for repeatability
    native = engine.startswith("native")
    if native:
        import vector_search as vs
    rows, queries = dataset(n, d, a.queries, a.seed)
    process = psutil.Process()
    rss0 = process.memory_info().rss
    t = perf_counter()
    if native:
        options = dict(hnsw=True, M=a.M, efConstruction=a.efConstruction) if engine == "native-hnsw" else {}
        index = vs.Index(rows, **options)
    else:
        if engine == "faiss-hnsw":
            index = faiss.IndexHNSWFlat(d, a.M)
            index.hnsw.efConstruction = a.efConstruction
            index.hnsw.efSearch = a.efSearch
        else:
            index = faiss.IndexFlatL2(d)
        index.add(rows)
    build = perf_counter() - t
    rss1 = process.memory_info().rss
    # Drop the source only after sampling index memory, so both snapshots include it.
    del rows

    def search(q, threads):
        if native:
            ef = a.efSearch if engine == "native-hnsw" else 0
            return index.search(q, k=10, efSearch=ef, threads=threads)[0]
        return index.search(q, 10)[1]

    ids = search(queries, 1)  # warm up; also provides the recall sample
    latency = []
    for i in range(len(queries)):
        q = queries[i:i+1]
        t = perf_counter()
        search(q, 1)
        latency.append((perf_counter() - t) * 1e6)
    faiss.omp_set_num_threads(a.threads)
    search(queries, a.threads)
    durations = []
    for _ in range(a.repeats):
        t = perf_counter()
        search(queries, a.threads)
        durations.append(perf_counter() - t)
    p50, p95 = np.percentile(latency, [50, 95])
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, Darwin reports bytes.
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    record = dict(engine=engine, rows=n, dim=d, seed=a.seed, queries=a.queries, k=10,
                  M=a.M, efConstruction=a.efConstruction, efSearch=a.efSearch,
                  build_threads=1, batch_threads=a.threads, repeats=a.repeats, build_s=build,
                  p50_us=p50, p95_us=p95, batch_qps=len(queries)/statistics.median(durations),
                  build_rss_delta_mib=(rss1-rss0)/2**20, peak_process_mib=peak_bytes/2**20,
                  os=platform.platform(), arch=platform.machine(), python=platform.python_version(),
                  numpy=np.__version__, faiss=faiss.__version__, faiss_options=faiss.get_compile_options().strip(),
                  kernel=vs.distance_backend if native else None)
    return record, ids.tolist()


def main():
    a = parse_args()
    if a.worker:
        print(json.dumps(measure(a)))
        return
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    for n in a.rows:
        for d in a.dims:
            truth = None
            for engine in dict.fromkeys(("faiss-flat", *a.engines)):
                cmd = [sys.executable, __file__, "--worker", engine, "--rows", str(n), "--dims", str(d)]
                for opt in ("queries", "threads", "repeats", "M", "efConstruction", "efSearch", "seed"):
                    cmd.extend(["--" + opt, str(getattr(a, opt))])
                print(f"Measuring {engine}  {n:,} × {d}", file=sys.stderr, flush=True)
                record, ids = json.loads(subprocess.check_output(cmd, env=env))
                if truth is None:
                    truth = [set(row) for row in ids]
                if engine not in a.engines:
                    continue
                record["recall_at_10"] = sum(len(ref.intersection(row)) for ref, row in zip(truth, ids)) / (10 * len(ids))
                line = json.dumps(record, allow_nan=False)
                if a.output:
                    with open(a.output, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                print(line, flush=True)


if __name__ == "__main__":
    main()

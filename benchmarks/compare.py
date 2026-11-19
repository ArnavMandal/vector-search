"""Run each CPU index in a fresh process. Build once and measure each requested search width."""
import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
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
    p.add_argument("--efSearch", nargs="+", type=int, default=[32],
                   help="One or more HNSW query widths, measured on the same graph")
    p.add_argument("--calibration-queries", type=int, default=0,
                   help="Additional independent queries used only to select a search width")
    p.add_argument("--recall-target", type=float, default=0.95)
    p.add_argument("--batch-size", type=int, help="Queries per throughput call (default: all test queries)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", help="Append completed measurements in JSON Lines format")
    p.add_argument("--worker", choices=ENGINES, help=argparse.SUPPRESS)
    p.add_argument("--truth-file", type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    a.efSearch = sorted(set(a.efSearch))
    if a.batch_size is None:
        a.batch_size = a.queries
    if not 1 <= a.batch_size <= a.queries or a.calibration_queries < 0 or not 0 < a.recall_target <= 1:
        p.error("require a valid batch size, nonnegative calibration count, and recall target in (0, 1]")
    if min(a.rows) < 10 or min(*a.dims, a.queries, a.threads, a.repeats) < 1:
        p.error("require at least 10 rows and positive dimensions, queries, threads and repeats")
    if not 2 <= a.M <= 256 or a.efConstruction < a.M or min(a.efSearch) < 10 or a.seed < 0:
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


def hit_counts(ids, truth):
    """Count distinct matching row IDs independently for each query."""
    if ids.shape != truth.shape or ids.shape[1] != 10:
        raise ValueError("both results must contain ten IDs per query")
    return np.array([len(set(found).intersection(exact)) for found, exact in zip(ids, truth)])


def core_bytes(index, native, graph):
    if native:
        return index.storage_bytes
    size = index.ntotal * index.d * 4
    if graph:
        h = index.hnsw
        # FAISS stores int32 neighbors/levels and uint64 offsets.
        size += 4 * (h.neighbors.size() + h.levels.size()) + 8 * h.offsets.size()
    return size


def time_search(search, queries, batch_size, threads, repeats):
    import faiss

    faiss.omp_set_num_threads(1)
    batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    ids = np.concatenate([search(batch, 1) for batch in batches])
    latency = []
    for i in range(len(queries)):
        query = queries[i:i + 1]
        t = perf_counter()
        search(query, 1)
        latency.append((perf_counter() - t) * 1e6)
    faiss.omp_set_num_threads(threads)
    for batch in batches:
        search(batch, threads)
    durations = []
    for _ in range(repeats):
        t = perf_counter()
        for batch in batches:
            search(batch, threads)
        durations.append(perf_counter() - t)
    p50, p95 = np.percentile(latency, [50, 95])
    return ids, dict(p50_us=p50, p95_us=p95, batch_qps=len(queries) / statistics.median(durations))


def measure(a):
    import faiss
    import psutil

    n, d, engine = a.rows[0], a.dims[0], a.worker
    native = engine.startswith("native")
    graph = engine.endswith("hnsw")
    if native:
        import vector_search as vs
    faiss.omp_set_num_threads(1)
    rows, all_queries = dataset(n, d, a.queries + a.calibration_queries, a.seed)
    queries, calibration = all_queries[:a.queries], all_queries[a.queries:]
    process = psutil.Process()
    rss0 = process.memory_info().rss
    t = perf_counter()
    if native:
        options = dict(hnsw=True, M=a.M, efConstruction=a.efConstruction) if graph else {}
        index = vs.Index(rows, **options)
    else:
        if graph:
            index = faiss.IndexHNSWFlat(d, a.M)
            index.hnsw.efConstruction = a.efConstruction
        else:
            index = faiss.IndexFlatL2(d)
        index.add(rows)
    build = perf_counter() - t
    rss1 = process.memory_info().rss
    del rows
    memory = core_bytes(index, native, graph) / 2**20
    print(f"  built {engine} in {build:.2f}s; core index {memory:.1f} MiB", file=sys.stderr, flush=True)

    def at_width(ef):
        if native:
            return lambda q, threads: index.search(q, k=10, efSearch=ef if graph else 0, threads=threads)[0]
        if graph:
            index.hnsw.efSearch = ef
        return lambda q, threads: index.search(q, 10)[1]

    if engine == "faiss-flat":
        # Only the small top-10 reference array is exchanged between workers.
        truth = np.concatenate([index.search(all_queries[i:i+a.batch_size], 10)[1]
                                for i in range(0, len(all_queries), a.batch_size)])
        np.save(a.truth_file, truth, allow_pickle=False)
    else:
        truth = np.load(a.truth_file, allow_pickle=False)
    if truth.shape != (len(all_queries), 10):
        raise ValueError("reference dimensions differ from query dimensions")

    widths = a.efSearch if graph else [a.efSearch[0]]
    calibration_scores = {}
    selected = None
    if len(calibration) and graph:
        # Select before using the test queries. Keep the full width grid in the output.
        for ef in widths:
            ids = at_width(ef)(calibration, 1)
            calibration_scores[ef] = hit_counts(ids, truth[a.queries:]).mean() / 10
        selected = next((ef for ef in widths if calibration_scores[ef] >= a.recall_target), None)
        print(f"  first width at calibration target: {selected}", file=sys.stderr, flush=True)

    records = []
    selected_ids = None
    for ef in widths:
        ids, times = time_search(at_width(ef), queries, a.batch_size, a.threads, a.repeats)
        counts = hit_counts(ids, truth[:a.queries])
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = peak if sys.platform == "darwin" else peak * 1024
        record = dict(engine=engine, rows=n, dim=d, seed=a.seed, queries=a.queries, k=10,
                      M=a.M, efConstruction=a.efConstruction, efSearch=ef,
                      build_threads=1, batch_threads=a.threads, batch_size=a.batch_size,
                      repeats=a.repeats, build_s=build, **times, recall_at_10=counts.mean()/10,
                      build_rss_delta_mib=(rss1-rss0)/2**20, peak_process_mib=peak_bytes/2**20,
                      index_core_mib=memory, os=platform.platform(), arch=platform.machine(),
                      python=platform.python_version(), numpy=np.__version__, faiss=faiss.__version__,
                      faiss_options=faiss.get_compile_options().strip(), kernel=vs.distance_backend if native else None)
        if a.calibration_queries:
            record.update(calibration_queries=a.calibration_queries, recall_target=a.recall_target,
                          calibration_recall_at_10=calibration_scores.get(ef), selected=ef == selected,
                          test_hits_histogram=np.bincount(counts, minlength=11).tolist())
        if ef == selected:
            selected_ids = ids
        records.append(record)

    # Exclude diagnostic tracing from all timed regions and peak RSS readings.
    if native and graph and selected is not None:
        fallback = 0
        for q, expected in zip(queries, selected_ids):
            trace = index.trace_hnsw(q, k=10, efSearch=selected)
            if trace["ids"] != expected.tolist():
                raise ValueError("trace and production search returned different neighbors")
            fallback += trace["exact_fallback"]
        next(r for r in records if r["selected"])["test_exact_fallbacks"] = fallback
    return records


def main():
    a = parse_args()
    if a.worker:
        print(json.dumps(measure(a), allow_nan=False))
        return
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    with tempfile.TemporaryDirectory() as tmp:
        for n in a.rows:
            for d in a.dims:
                for engine in dict.fromkeys(("faiss-flat", *a.engines)):
                    cmd = [sys.executable, __file__, "--worker", engine, "--rows", str(n), "--dims", str(d),
                           "--truth-file", str(Path(tmp)/"truth.npy"), "--efSearch", *map(str, a.efSearch)]
                    for opt in ("queries", "threads", "repeats", "M", "efConstruction", "seed",
                                "calibration_queries", "recall_target", "batch_size"):
                        cmd.extend(["--" + opt.replace("_", "-"), str(getattr(a, opt))])
                    print(f"Measuring {engine}  {n:,} × {d}", file=sys.stderr, flush=True)
                    records = json.loads(subprocess.check_output(cmd, env=env))
                    if engine not in a.engines:
                        continue
                    for record in records:
                        line = json.dumps(record, allow_nan=False)
                        if a.output:
                            with open(a.output, "a", encoding="utf-8") as f:
                                f.write(line + "\n")
                        print(line, flush=True)


if __name__ == "__main__":
    main()

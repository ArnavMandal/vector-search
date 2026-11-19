# Native vector search and FAISS on CPU

## Reproduce the matrix

From the project directory, with Python 3.12 and a C++20 compiler, run

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r benchmarks/requirements.txt
make bindings PYTHON=python
PYTHONPATH=build python benchmarks/compare.py --output my-run.jsonl
```

The default matrix is the Cartesian product of 10K, 100K and 1M rows with 32, 128 and 384 dimensions. Each shape runs native exact, native HNSW, FAISS IndexFlatL2 and FAISS IndexHNSWFlat. Output appends one JSON record per completed measurement. Use a fresh filename for a new run. A smaller subset or a different graph setting can be selected with the same script.

```sh
PYTHONPATH=build python benchmarks/compare.py --rows 10000 --dims 128 --M 16 --efConstruction 100 --efSearch 64
```

## Protocol

- **Data** uses float32 Gaussian noise around 64 random centers. A seed of 42 plus the shape initializes separate streams for centers, corpus rows and 64 held-out queries. Every backend regenerates the identical data in bounded chunks. Times exclude generation and model inference.
- **Graph settings** for the scale matrix are M = 4, efConstruction = 16 and efSearch = 32. This deliberately small graph makes the memory and recall costs of sparse connections visible. The native API defaults at construction are M = 16 and efConstruction = 100.
- **Recall@10** compares row IDs with FAISS IndexFlatL2 exhaustive search on the same query batch. Flat FAISS scores its own reference at 1 by definition. An independent NumPy oracle and native scalar tests verify the engine separately.
- **Latency** times one warmed Python search call per query, including binding and result materialization. p50 and p95 summarize those 64 individual calls. **Throughput** uses the median of three timed batches of all 64 queries following a warm-up. Construction and queries both use one thread, and BLAS/OpenMP thread counts are constrained. The original benchmark supports a separate parallel-batch experiment.
- **Build time** starts with a populated, resident NumPy matrix and includes copying the rows, native validation where applicable and graph construction. Search timing excludes loading and building. These measurements represent one build per configuration.
- **Memory** in the table is peak process RSS, covering input, index, Python runtime and transient allocations over the worker's lifetime. Each index runs in its own process. The raw records also include the RSS change across construction with the input still alive. On this 8 GiB Mac, compression and paging can yield a negative change. Peak RSS and exact vector payload answer different questions.
- **Environment** was arm64 macOS 26.0.1, Apple Clang 17 with `-O3`, Python 3.12.11, NumPy 2.3.4 and faiss-cpu 1.16.0. Native runs selected NEON and FAISS used its CPU index classes. The largest shape was rerun for all four engines with compilation finished, after a competing build affected the first FAISS measurement. The table and raw data use that repeat for all four entries. Treat timings, particularly under memory pressure, as observations from this host. The data, parameters, versions, thread counts, dispatch information and unrounded results are recorded in [results.jsonl](results.jsonl).

## Results

### 10,000 vectors

| Dimensions | Engine | Recall@10 | p50 µs | p95 µs | Batch queries/s | Peak MiB | Build s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | native-flat | 1.000 | 78.3 | 81.9 | 12,996 | 65.4 | 0.0003 |
| 32 | faiss-flat | 1.000 | 31.4 | 49.4 | 36,919 | 65.5 | 0.0007 |
| 32 | native-hnsw | 0.631 | 8.4 | 11.8 | 141,802 | 65.7 | 0.0583 |
| 32 | faiss-hnsw | 0.614 | 8.3 | 13.6 | 160,284 | 65.7 | 0.0585 |
| 128 | native-flat | 1.000 | 272.7 | 286.5 | 3,684 | 74.1 | 0.0010 |
| 128 | faiss-flat | 1.000 | 96.8 | 108.4 | 9,598 | 74.0 | 0.0009 |
| 128 | native-hnsw | 0.497 | 12.0 | 16.7 | 94,245 | 74.1 | 0.0966 |
| 128 | faiss-hnsw | 0.503 | 10.2 | 15.2 | 127,670 | 74.5 | 0.0691 |
| 384 | native-flat | 1.000 | 881.4 | 895.6 | 1,135 | 102.5 | 0.0031 |
| 384 | faiss-flat | 1.000 | 302.1 | 352.5 | 3,391 | 102.3 | 0.0027 |
| 384 | native-hnsw | 0.438 | 22.0 | 31.9 | 46,777 | 102.7 | 0.2098 |
| 384 | faiss-hnsw | 0.412 | 12.3 | 16.9 | 99,501 | 102.5 | 0.1003 |

### 100,000 vectors

| Dimensions | Engine | Recall@10 | p50 µs | p95 µs | Batch queries/s | Peak MiB | Build s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | native-flat | 1.000 | 748.4 | 767.0 | 1,334 | 88.7 | 0.0026 |
| 32 | faiss-flat | 1.000 | 251.1 | 265.4 | 4,004 | 88.7 | 0.0016 |
| 32 | native-hnsw | 0.345 | 10.3 | 12.1 | 116,895 | 92.9 | 0.6768 |
| 32 | faiss-hnsw | 0.394 | 9.9 | 14.1 | 127,163 | 93.9 | 0.6527 |
| 128 | native-flat | 1.000 | 2,751.6 | 3,041.5 | 355 | 162.2 | 0.0105 |
| 128 | faiss-flat | 1.000 | 1,093.8 | 1,387.4 | 905 | 155.9 | 0.0126 |
| 128 | native-hnsw | 0.212 | 17.9 | 24.5 | 71,439 | 153.2 | 1.6516 |
| 128 | faiss-hnsw | 0.236 | 14.4 | 25.0 | 104,847 | 157.0 | 0.9332 |
| 384 | native-flat | 1.000 | 8,781.7 | 8,935.2 | 114 | 365.9 | 0.0349 |
| 384 | faiss-flat | 1.000 | 3,031.7 | 3,276.5 | 316 | 352.3 | 0.0594 |
| 384 | native-hnsw | 0.163 | 35.2 | 44.9 | 35,622 | 367.0 | 3.1029 |
| 384 | faiss-hnsw | 0.278 | 24.6 | 34.5 | 46,967 | 367.1 | 1.6799 |

### 1,000,000 vectors

| Dimensions | Engine | Recall@10 | p50 µs | p95 µs | Batch queries/s | Peak MiB | Build s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | native-flat | 1.000 | 7,462.7 | 7,595.3 | 134 | 308.5 | 0.0284 |
| 32 | faiss-flat | 1.000 | 2,543.5 | 2,838.5 | 365 | 308.3 | 0.0120 |
| 32 | native-hnsw | 0.147 | 13.3 | 17.8 | 98,912 | 359.4 | 10.4955 |
| 32 | faiss-hnsw | 0.180 | 15.9 | 33.0 | 101,467 | 351.7 | 12.2751 |
| 128 | native-flat | 1.000 | 27,023.1 | 27,261.7 | 37 | 1,041.0 | 0.0898 |
| 128 | faiss-flat | 1.000 | 9,947.6 | 16,150.8 | 63 | 1,007.9 | 0.3687 |
| 128 | native-hnsw | 0.102 | 26.7 | 36.4 | 56,020 | 1,079.4 | 21.5497 |
| 128 | faiss-hnsw | 0.094 | 21.3 | 31.4 | 79,126 | 1,060.4 | 21.2339 |
| 384 | native-flat | 1.000 | 88,326.8 | 105,912.2 | 11 | 2,828.4 | 1.1621 |
| 384 | faiss-flat | 1.000 | 32,710.1 | 62,385.2 | 32 | 2,000.3 | 1.1802 |
| 384 | native-hnsw | 0.086 | 62.5 | 88.0 | 22,109 | 2,132.8 | 61.0728 |
| 384 | faiss-hnsw | 0.084 | 40.4 | 67.6 | 30,840 | 2,866.7 | 35.2836 |

## What the measurements show

At 1M × 384, the native scan took 88.3 ms at p50 and recovered 100.0% of the reference IDs. The sparse native graph took 62.5 µs and recovered 8.6%. FAISS HNSW took 40.4 µs and recovered 8.4%. Both graphs made a large speed-for-recall tradeoff. These operating points would be poor choices for a service that needs most of its true top ten.

### Changing construction and search effort

The following reruns use the same 10K × 128 corpus and queries as the scale matrix. Here M is 16 and efConstruction is 100. Each setting uses a fresh deterministic build. Compare the two search widths within each engine, then compare width 32 with the sparse graph above. [Supplementary measurements](tuning.jsonl)

| Engine | efSearch | Recall@10 | p50 µs | Batch queries/s | Build s |
| --- | ---: | ---: | ---: | ---: | ---: |
| native-hnsw | 32 | 0.997 | 17.8 | 61,438 | 0.467 |
| faiss-hnsw | 32 | 0.997 | 13.1 | 95,085 | 0.336 |
| native-hnsw | 128 | 1.000 | 35.9 | 29,363 | 0.458 |
| faiss-hnsw | 128 | 0.998 | 39.4 | 27,811 | 0.334 |

On the same exact-search workload the portable kernel measured 693.7 µs at p50 and NEON measured 272.7 µs (2.54× scalar-to-NEON latency ratio). Both were release builds using the same compiler and data. This measures complete Python search calls. AVX2 passed correctness checks under Rosetta. Performance on physical x86 hardware requires a separate run.

To reproduce these follow-up measurements, run from the project directory with the benchmark environment active.

```sh
for ef in 32 128; do
  PYTHONPATH=build python benchmarks/compare.py --rows 10000 --dims 128 --M 16 --efConstruction 100 --efSearch "$ef" --engines native-hnsw faiss-hnsw
done
make BUILD=build/scalar-release CPPFLAGS=-DVS_FORCE_SCALAR PYTHON=python bindings
for dir in build build/scalar-release; do
  PYTHONPATH="$dir" python benchmarks/compare.py --rows 10000 --dims 128 --engines native-flat
done
```

# Vector search

A standalone C++20 nearest-neighbor library, a command-line client and a batched Python interface. The C++ engine owns a contiguous row-major float32 matrix, computes squared Euclidean distances in double precision and returns zero-based row IDs. Equal distances resolve to the smaller ID.

The engine also supports HNSW and hardware-dispatched distance kernels. Start with the [learning guide](LEARNING.md) for a worked search, SIMD examples, tuning exercises and a tour of the changes. The [FAISS comparison](benchmarks/REPORT.md) covers all nine requested workloads.

## Run the C++ milestone

From this directory, on macOS or Linux with Make and a C++20 compiler, run

```sh
make
./build/vector-search examples/points.txt '0 0'
make test
```

The CLI loads one whitespace-separated vector per nonblank line. Pass the query as one quoted argument and an optional positive `k` as the last argument. The default is 10. For the example above, the ten IDs in order are `1 2 3 4 5 6 7 8 9 10`. ID 1 has distance zero. IDs 2, 3 and 4 each have squared distance one. For smaller collections, the CLI returns every row.

The static library is `build/libvector_search.a`. Include `include/vector_search.hpp` to use it directly. Link with C++20 and thread support. Storage and search have no Python dependency.

## Use the Python interface

Use one Python environment to compile and import the extension.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'pybind11>=2.13.6' numpy
make bindings PYTHON=python
export PYTHONPATH="$PWD/build"
make test-python PYTHON=python
```

```python
import numpy as np
from vector_search import Index

rng = np.random.default_rng(42)
vectors = rng.standard_normal((1000, 32), dtype=np.float32)
queries = vectors[:8]
index = Index(vectors, tree=True)
ids, squared_distances = index.search(queries, k=10)
approx_ids, approx_distances = index.search(queries, k=10, budget=128, threads=4)

index.save("catalog.vsi")
restored = Index.load("catalog.vsi")
```

All search modes return arrays of shape `(query_count, min(k, index.size))`. Both inputs must be 2-D, with the same positive feature dimension. The index requires at least one row. Python converts numeric inputs to float32 and C++ rejects nonfinite coordinates. The immutable index owns its vectors. Each search copies the query batch once into native storage and releases the GIL during search. Workers split the batch across queries.

## Use HNSW

```python
from vector_search import Index, distance_backend

graph = Index(vectors, hnsw=True, M=16, efConstruction=100)
ids, distances = graph.search(queries, k=10, efSearch=64, threads=4)
print(distance_backend)  # neon, avx2 or scalar
```

In C++, construct `vs::Index graph(rows, dim, false, vs::HnswConfig{16, 100})` and call `graph.search(queries, 10, 0, 4, 64)`. An ordinary `search(queries)` still requests the exact scan. Set a positive `efSearch` for graph search or a positive `budget` for tree search. You can build both accelerators for the same vectors. Set `2 <= M <= 256`, `efConstruction >= M` and `efSearch >= min(k, size)`. The last parameter is per call, so concurrent readers can use different recall settings.

## Search text

With the extension on `PYTHONPATH`, install the demo's encoder dependencies and run

```sh
python -m pip install 'transformers>=4.51,<5' 'torch>=2.6'
python python/demo.py 'basketball shoes' 'something to grind espresso beans'
```

Python loads a pinned MiniLM checkpoint, averages the token representations using the attention mask, and normalizes each embedding to unit length. Squared Euclidean distance then gives the same ranking as cosine similarity. The first run downloads model weights. Use `--model-dir PATH` to run from a local copy. [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

The demo embeds the catalog in one batch and all queries in another. For “basketball shoes,” the first two matches are cushioned basketball footwear and high-top sneakers for indoor courts. For “something to grind espresso beans,” the burr grinder ranks first.

## How search works

**Exact scan.** C++ scores all rows and keeps a max-heap of the best `k`. For `n` rows of dimension `d`, the worst-case query time is O(n d + n log k) with O(k) result storage.

**Budgeted k-d tree.** Construction splits at the median along the widest coordinate until a leaf holds at most 16 points. A query visits branches in increasing order of their lower bound on squared distance. It prunes a branch once that bound exceeds the worst retained score. The budget caps the number of vector distances evaluated, including pivots. A positive budget requires `tree=True` and must cover the requested result count. A budget at least as large as the corpus yields the exact result. With both search controls at zero the engine runs the reference scan.

**Persistence.** The versioned binary snapshot contains the row count, dimension, construction mode and float32 vectors. Legacy exact and tree files retain their v1 format. Graph snapshots use v2 and store M and efConstruction. Loading rebuilds requested accelerators with a fixed graph seed, which adds construction time at startup. Saving writes a temporary sibling and atomically renames it into place. The format targets little-endian POSIX systems.

## Measure recall and speed

```sh
python python/benchmark.py
python python/benchmark.py --distribution gaussian
```

The default workload uses 12,000 vectors, 32 dimensions, 200 independent queries, k = 10 and seed 42. The clustered workload samples rows and queries around the same 32 centers. The Gaussian workload samples independent coordinates. Recall@10 is the mean fraction of exact top-10 IDs recovered per query.

Historical run of the original scalar engine on arm64 macOS with Apple Clang 17 at `-O3`, Python 3.12.11 and NumPy 2.3.4. Latencies measure individual warmed Python calls with one search thread. Throughput uses four threads and the median of five whole-batch timings. Memory counts allocated vector and tree array storage, excluding the Python process, result arrays and per-query scratch space.

| Clustered workload | p50 latency | p95 latency | Queries/sec | Index MiB | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact | 257.7 µs | 265.5 µs | 15,497 | 1.465 | 1.000 |
| Tree, budget 64 | 8.2 µs | 9.3 µs | 544,834 | 1.648 | 0.256 |
| Tree, budget 256 | 16.8 µs | 19.4 µs | 248,834 | 1.648 | 0.659 |
| Tree, budget 1,024 | 47.8 µs | 53.8 µs | 81,462 | 1.648 | 0.947 |

At budget 1,024 the same tree recovered 40.0% of the exact neighbors on the Gaussian workload. The difference illustrates how strongly this index depends on the geometry of the data. Both benchmark commands print construction time, latency percentiles, throughput, memory and recall for each setting.

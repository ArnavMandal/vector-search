# From an exact vector scan to HNSW

Follow one query through the code, then change one parameter and measure what moves. You can run the first example with only a C++ compiler. The later exercises use the Python environment in the README.

## 1. Establish a result you can calculate by hand

Suppose your query is (1, 1). Row 0 is (4, 5), row 1 is (1, 3), and row 2 is (3, 1). Subtract the query coordinate by coordinate, square each difference, and add. The squared distances are 25, 4 and 4. The correct top two are rows 1 and 2 in that order. Taking a square root would preserve this order and add work. The engine uses the row ID to break equal-score ties.

```sh
make
printf '4 5\n1 3\n3 1\n' > /tmp/vector-search-lesson.txt
./build/vector-search /tmp/vector-search-lesson.txt '1 1' 2
```

Find `Index::search_one` in [src/index.cpp](src/index.cpp). Picture the worst accepted neighbor at the top of a heap. Once the heap contains k rows, every new row needs only one comparison to decide whether it can enter. A better row displaces that worst neighbor. The final heap sort emits nearest first. The loop still reads the entire matrix. On a large corpus, the time to read those bytes can dominate the arithmetic.

**Check your understanding.** What changes if k becomes 1? The scan still evaluates three distances. Its heap holds one result, and row 1 wins the tie.

## 2. Use SIMD for the inner loop

The original loop took one difference, squared it, and added it to one running sum. SIMD applies a single instruction to several independent lanes. For example, if the coordinate differences are [1, 2, 3, 4], their squares are [1, 4, 9, 16] and the final horizontal sum is 30. For a five-dimensional row with a fifth difference of 2, a scalar tail adds 4 to make 34.

Vectors occupy adjacent float32 coordinates. The new kernels widen coordinates **before subtraction**, then square and accumulate in float64. This matters for inputs near opposite float32 extremes, where even subtraction could overflow in float32. A 256-bit AVX2 register holds four doubles. The x86 kernel uses two accumulators for eight coordinates per iteration. A 128-bit NEON register holds two doubles. The AArch64 kernel uses two accumulators for four coordinates per iteration. Independent sums reduce the dependency on the preceding addition. The kernels accept unaligned rows and handle leftover coordinates without reading past the row. See [src/distance.cpp](src/distance.cpp) and the [Arm intrinsics reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html).

The choice happens once per process. A baseline x86 binary checks CPU support before calling a function compiled with the AVX2 target attribute. AArch64 builds with NEON select the NEON kernel. Other supported compilers and targets can use portable C++. Every search still compares the resulting squared distances. Adding floats in a different order, or using fused multiply-add, can change the last bits of a score and the order of exceptionally close neighbors. Identical computed scores still resolve by ID.

**Try it.** Build into separate directories so Make tracks each configuration independently.

```sh
make BUILD=build/native test
make BUILD=build/scalar CPPFLAGS=-DVS_FORCE_SCALAR test
```

The accelerator test prints the kernel selected for each binary. Read [tests/accelerators.cpp](tests/accelerators.cpp) and look at dimensions 7, 8 and 9. An implementation that silently ignores the tail can pass an eight-dimensional test and fail the other two. To time the complete exact search, build bindings in both directories, then run the same benchmark with each directory on PYTHONPATH. A smaller time per distance helps every scan. Whether it doubles overall throughput depends on memory bandwidth, heap operations, Python call overhead and the dimensions tested.

On the measured 10K × 128 workload, changing only the compiled kernel reduced p50 exact-search latency from 693.7 µs to 272.7 µs on this ARM machine. That 2.54× ratio includes Python calls, scoring and heap work. You can reproduce it with the two release-build commands in the [benchmark report](benchmarks/REPORT.md).

## 3. Trace an HNSW query

HNSW stores row IDs in a hierarchy of proximity graphs. Every row joins level zero. Roughly one in M rows also joins level one, and roughly one in M of those joins level two. The sparse upper levels give a query long-range routes through the data. The dense base supports local exploration. Read the [original HNSW paper](https://arxiv.org/abs/1603.09320).

For a one-dimensional illustration, put the query at 9. Imagine entering an upper layer at 0 with a link to 8. Move to 8 because its squared distance is 1. Descend at 8. At the base, suppose 8 links to 6 and 10, and 10 links to 9.1. Expanding 10 uncovers a row with squared distance about 0.01. The key operation is deciding which unfinished route to expand while preserving several good alternatives.

Here is one possible trace with efSearch = 2. The entries are row coordinates. Both heaps compare their squared distance to 9.

| Action at the base | Frontier, nearest first | Best two retained |
| --- | --- | --- |
| Start at 8 | 8 | 8 |
| Expand 8 and discover 6, then 10 | 10, 6 | 8, 10 |
| Expand 10 and discover 9.1 | 9.1, 6 | 9.1 and one of the tied rows 8 or 10 |
| Expand 9.1, then inspect 6 | 6 | Stop, since 6 is farther than both retained rows |

Inside `explore`, the **frontier** is a min-heap of candidates to expand. A second heap keeps the best ef candidates seen so far, with the worst retained score on top. Each neighbor is scored on its first visit. Once that heap is full and even the best remaining frontier candidate is worse than its threshold, exploration stops. A close row can still lie behind a poor intermediate node, so this stopping rule permits misses. The returned distances are actual distances to the retrieved rows. Approximation concerns which rows were reached. Follow `descend` and `explore` in [src/hnsw.cpp](src/hnsw.cpp).

**Why the visited array has a counter.** Clearing a million flags for every short query would add work proportional to the whole corpus. Each traversal advances an epoch and marks IDs with that epoch. An old mark means the node is unvisited in this traversal. Separate search threads own their scratch arrays. Each active thread can retain about four bytes per visited-array slot up to the largest index it has queried. The index itself stays immutable.

## 4. Build and tune the graph

The constructor inserts rows in input order. It chooses levels with a fixed random seed, descends from an existing entry point, and searches candidate neighborhoods on shared layers. When selecting links, it favors candidates that add a direction of travel. For example, two near-duplicate neighbors east of a node may offer similar routes. A slightly farther western neighbor can be a better use of a limited slot. The code rejects a candidate if an already chosen neighbor is closer to that candidate than the inserting row is.

Each insertion also proposes reverse links. If a neighbor's adjacency list is full, pruning chooses which links survive. This can leave directed edges. An unusually small reachable component can yield fewer than k distinct rows. In that case the native API runs an exact scan to honor its result-count contract. That recovery can show up as a slow query. Use the exhaustive scan when a query requires a guarantee of global exactness.

| Control | When it applies | What changes | Cost to watch |
| --- | --- | --- | --- |
| M | Build | Number of selected outgoing neighbors and layer promotion probability. Lists hold up to 2M IDs at the base and M above. | Persistent graph bytes and distance checks per expanded node |
| efConstruction | Build | Number of candidate neighbors retained while searching each insertion layer | Construction time and the quality of routes available for future queries |
| efSearch | Query | Number of good candidates retained at the base | Latency and recall at fixed graph memory |

efSearch is a search width. A query can evaluate many more than efSearch distances while expanding the frontier. The older k-d tree's budget literally caps vector-distance evaluations. The tree uses bounds from coordinate splits. The graph follows proximity edges. SIMD accelerates distance calculations for both approaches as well as the exact baseline.

**Run a controlled experiment.** The query batch below is independent of the indexed rows. Self-queries can make retrieval look easier than a production workload.

```python
import numpy as np
from vector_search import Index

rng = np.random.default_rng(7)
rows = rng.standard_normal((10000, 32), dtype=np.float32)
queries = rng.standard_normal((100, 32), dtype=np.float32)
graph = Index(rows, hnsw=True, M=12, efConstruction=80)
truth, _ = graph.search(queries, k=10)
for ef in (10, 32, 100):
    found, _ = graph.search(queries, k=10, efSearch=ef)
    recall = sum(len(set(a) & set(b)) for a, b in zip(found, truth)) / truth.size
    print(ef, round(recall, 3))
```

Predict how the wider frontier changes recall, then run it. Now hold efSearch fixed and rebuild with M of 4 and 16. Record construction time, index.storage_bytes and recall. Increasing construction effort can improve the entire recall–latency curve, whereas changing query effort moves along a curve for an existing graph.

For a measured example, look at the 10K × 128 clustered corpus. With M = 16 and efConstruction = 100, the native graph recovered 638 of 640 reference IDs at efSearch = 32. Widening to 128 recovered the remaining two IDs and raised p50 from 17.8 µs to 35.9 µs. The reward for spending more time depends on where the application sits on that curve. [Measured settings and results](benchmarks/REPORT.md)

## 5. Read a benchmark without losing the question

**Recall@10** asks how many exact IDs were recovered. Over two queries with eight and five correct IDs, it is 13/20 = 0.65. It answers a different question from how close the returned distances are. If many rows sit nearly the same distance away, a missed ID can have a tiny distance penalty.

**p50 and p95 latency** describe typical and slower individual queries. With 64 samples the p95 lies near the slowest four observations and can move noticeably on a repeat. **Throughput** counts queries completed per second in a batch. Batch time divided by the count is an amortized cost. Actual single-query requests also pay call overhead and may follow a different library execution path. FAISS can use a matrix multiplication implementation for sufficiently large batches. [FAISS implementation notes](https://github.com/facebookresearch/faiss/wiki/Implementation-notes)

**Memory has several meanings.** Float32 vector payload is exactly N × d × 4 bytes. For 1M × 384 it is 1,536,000,000 bytes, about 1.43 GiB. The graph adds adjacency and offset arrays, and each search worker needs scratch space. A resident-memory measurement can be smaller than the logical payload under OS compression or paging. Peak process memory also includes the Python interpreter, source data, and temporary construction buffers. Compare corresponding columns across backends.

**Build time matters over the lifetime of an index.** If flat search takes 2 ms per query and graph search takes 0.2 ms at acceptable recall, a graph that costs 90 extra seconds to build needs about 50,000 serial queries to repay construction time. This is an illustrative calculation. Plug in measured numbers for your deployment, account for index reloads, and use batch measurements if your service batches requests.

The comparison uses one exact flat FAISS index as a reference and one FAISS HNSW index with full vectors. Both have the same L2 metric as the native engine. FAISS scores in float32 and the native engine accumulates in float64, so nearly tied rankings can differ even for the two exhaustive scans. The two HNSW builders can also produce different routes at identical M and ef settings. To select a backend, compare latency at a recall level your application actually needs. [FAISS index overview](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

## 6. Map the changes back to the original engine

| Before | Extension | Where to read or verify it |
| --- | --- | --- |
| Each visited row ran an inline scalar L2 loop | A shared kernel serves exact, tree and graph scoring | [Distance implementation](src/distance.cpp), [kernel tests](tests/accelerators.cpp) |
| Flat scan and budgeted k-d tree | Optional native HNSW with compact 32-bit link slots and a fixed build seed | [Graph implementation](src/hnsw.cpp), [index dispatch](src/index.cpp) |
| Python batched searches accepted k, budget and threads | Keyword controls add graph construction and per-call efSearch | [Python bindings](python/bindings.cpp) |
| Snapshots contained the vector matrix and a tree flag | Graph snapshots add construction settings. Loading rebuilds accelerators. v1 files remain readable and writable for flat and tree indexes. | [Storage](src/storage.cpp), [round-trip and legacy tests](tests/test_python.py) |
| A benchmark compared the tree with a native exact scan | A second runner compares two native modes with two FAISS modes across nine workloads | [FAISS runner](benchmarks/compare.py), [results and reproduction](benchmarks/REPORT.md) |

When you reload a large graph snapshot, budget for another graph build. The file stores vectors and construction settings. Row order and the fixed seed reproduce the graph in the tested build. Different floating-point kernels can resolve a near tie differently during insertion, which can alter the links chosen across machines.

For a final exercise, select a row in the report where a graph loses recall you care about. Hold the corpus and query seed fixed, increase efSearch, and measure again. If query time rises without enough recall recovery, rebuild with higher M or efConstruction and repeat. That sequence distinguishes a narrow query search from a graph whose routes are too sparse for the workload.

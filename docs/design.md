# Design doc: mini vector search engine

## Problem
Approximate nearest-neighbor search over 128-d SIFT vectors with three requirements, in order:
(1) a fast static index, (2) attribute filters (label and numeric range), (3) inserts while queries run.
Success = a clear recall@10-vs-QPS curve within a stated factor of FAISS HNSW, plus an honest map of
where each filtering strategy works and breaks, plus tail latency under mixed load.

## Benchmark methodology
Dataset SIFT1M (1M x 128, 10k queries, exact ground truth). Metrics defined in README. Single-threaded
QPS for Phase 1 so it is comparable with FAISS (`faiss.omp_set_num_threads(1)`, one query per call).
Filtered ground truth is computed exactly over the matching set. Phase 3 reports p99 query latency and
insert throughput for a fixed query-thread count, versus the same index queried with no inserts.
Hardware: see `results/machine.txt` from each run.

## Phase 1: HNSW
- Level-0 lists in one flat array (count + 2M ids per node); upper levels in per-node arrays.
- Neighbor-selection heuristic (Algorithm 4); no "keep pruned" extension.
- Visited set: thread-local tag array with an epoch counter (no per-query clearing).
- Vectors are copied into the index so streaming inserts work; this makes index_mb comparable with FAISS.
- Level assignment is a hash of (seed, id), so builds are reproducible and ids equal dataset rows even
  when built in parallel.
- Search prefetches neighbor vectors before the distance loop.
Open knobs to try, judged only by the curve: explicit AVX2 distance, reordering nodes for locality,
storing vectors inline with neighbor lists, M and efConstruction sweeps.

## Phase 2: filtered search
Strategies: **pre-filter** (matching ids from an inverted list / value-sorted array, exact scan),
**post-filter** (HNSW with ef results, then drop non-matches), **in-graph** (navigate through all
nodes, admit only matching nodes into the result set).
Expected failure modes, to confirm or refute with the SIFT1M data:
- post-filter: returns fewer than k results when ef x selectivity < k, so recall collapses at 1%.
- in-graph: result set fills slowly at low selectivity, so the search degenerates toward a full graph
  traversal (cost grows roughly as 1/selectivity); `--max-visit` caps it at a recall cost.
- pre-filter: cost is linear in the matching set, so it wins at 1% and loses at 50%.
- correlated attributes (`--attr-mode correlated`): matching points form a spatial region. This is the
  realistic hard case: in-graph search can start far from the region and post-filter wastes more work.
Limitation: two attribute types, one filter per query, synthetic attributes.

## Phase 3: concurrent updates
Fine-grained locking: one spinlock per node protects that node's neighbor lists. A thread holds at most
one node lock at a time (no deadlock possible). Readers copy a list under the lock and release it before
computing distances. Entry point and its level are one atomic word. A global mutex is taken only by the
~1/M inserts that raise the top level. With locking off (static builds) none of this code runs.
Not done (future work): epoch-based reclamation / lock-free reads, deletes, CANDOR-Bench workloads.
`bench_stream` uses its own mixed workload (initial fraction pre-built, rest inserted while query threads
run, optional fixed insert rate). Framing: an engineering and benchmark study of insertion cost and tail
latency, not a new algorithm; EMA (May 2026) already covers filtered search with dynamic updates.
Spinlocks assume threads <= cores; with more threads than cores, a preempted lock holder stalls spinners,
so report thread counts and cores with every result.

## What failed / bugs found while building (keep this section honest and growing)
- First parallel build used insertion order as the id, which silently broke recall against ground truth.
  Fix: explicit external ids (row numbers) and a hash-based level.
- Reading the entry point and max level as two separate variables let a query see a new level with the
  old entry node and index past that node's upper-level array. Fix: pack them into one atomic.
- In the first version of `bench_stream`, query threads captured stack variables by reference after the
  function returned (3 separate dangling captures), giving garbage timings or hangs. Fix: capture by value;
  found by a run that never terminated.
- Add here: every optimization that did not move the curve, with the numbers.

## Results
(to fill from SIFT1M runs: Phase 1 table vs FAISS and the gap explanation; Phase 2 table at 1/10/50%
for both attribute modes; Phase 3 table of p99 and insert throughput vs the static baseline.)

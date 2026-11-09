# vecsearch: mini vector search engine in C++

HNSW written from scratch, benchmarked against FAISS, extended with filtered search and
concurrent streaming updates. Every claim in this repo is backed by a CSV a script produced.

| Phase | What | Entry point |
|---|---|---|
| 0 | Benchmark harness + brute-force baseline (recall@10 vs QPS) | `bench`, `scripts/run_all.sh` |
| 1 | HNSW baseline vs FAISS HNSW (curve, build time, memory) | `include/vs/hnsw.hpp` |
| 2 | Filtered search: pre-filter vs post-filter vs in-graph at 1/10/50% selectivity | `bench_filtered`, `scripts/run_filtered.sh` |
| 3 | Concurrent inserts during queries (per-node locks): p99 and insert throughput | `bench_stream`, `scripts/run_stream.sh` |

## Quick start (Linux, g++ 11+, CMake 3.16+, Python 3 with numpy/pandas/matplotlib/faiss-cpu)
```bash
./scripts/download_sift1m.sh data              # or add a SIFT1M dataset on Kaggle
./scripts/run_all.sh data/sift                 # Phase 1  -> results/phase1_curve.png
./scripts/run_filtered.sh data/sift 1000 4     # Phase 2  -> results/phase2_filtered.png
./scripts/run_stream.sh data/sift 2 10         # Phase 3  -> results/phase3_stream.png
```
No SIFT1M handy? `python3 scripts/make_synthetic.py data/synth 20000 64` then pass `data/synth`.
Synthetic Gaussian data is much harder for graph indexes than SIFT, so its recall is lower;
use it only to check that the pipeline runs.

## Layout
```
include/vs/   index.hpp (common interface)  bruteforce.hpp  hnsw.hpp  filtered.hpp  io.hpp  distance.hpp
bench/        bench.cpp (Phase 1)  bench_filtered.cpp (Phase 2)  bench_stream.cpp (Phase 3)
scripts/      run_*.sh, faiss_baseline.py, plot*.py, download_sift1m.sh, make_synthetic.py
tests/        test_hnsw.cpp  (recall, filter correctness, parallel build, concurrent insert+query)
docs/design.md
```

## Metrics (identical definitions for our code and FAISS)
- recall@k: fraction of the exact top-k returned, averaged over queries (filtered: exact top-k among matching points)
- QPS: queries / wall time of the fastest of `--repeat` runs; single thread unless stated
- p50/p99: per-query latency from that same run
- build_s, index_mb: wall time and resident-memory delta around the build (includes the copied vectors)

## Reproducibility rules
Record `results/machine.txt` (written by every run script) next to the numbers. Pin thread counts.
On shared machines (Kaggle) p99 is noisy: repeat runs and report the spread.

## Results
Filled in from real SIFT1M runs (see docs/design.md, "Results").

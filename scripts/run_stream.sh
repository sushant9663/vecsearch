#!/usr/bin/env bash
# Phase 3: static baseline vs mixed insert+query load, sweeping insert threads and insert rate.
#   ./scripts/run_stream.sh [dataset_dir] [query_threads] [baseline_secs]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_paths.sh "${1:-data/sift}"; QT="${2:-2}"; BS="${3:-10}"
cmake -B build >/dev/null && cmake --build build -j >/dev/null
rm -f results/stream.csv
for IT in 1 2; do   # max insert throughput, 1 and 2 insert threads
  ./build/bench_stream --base $B --query $Q --gt $G --insert-threads $IT --query-threads $QT \
    --baseline-secs $BS --label unthrottled --out results/stream.csv
done
for R in 500 2000 5000; do   # fixed insert rate (ops/s) -> p99 vs write pressure
  ./build/bench_stream --base $B --query $Q --gt $G --insert-threads 1 --query-threads $QT \
    --insert-rate $R --baseline-secs $BS --label rate$R --out results/stream.csv
done
python3 scripts/plot_stream.py results/stream.csv results/phase3_stream.png
echo "done: results/phase3_stream.png"

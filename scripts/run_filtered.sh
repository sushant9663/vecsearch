#!/usr/bin/env bash
# Phase 2: pre/post/in-graph at 1%, 10%, 50% selectivity, independent and correlated attributes.
#   ./scripts/run_filtered.sh [dataset_dir] [max_queries] [threads]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_paths.sh "${1:-data/sift}"; MQ="${2:-1000}"; T="${3:-1}"
cmake -B build >/dev/null && cmake --build build -j >/dev/null
for MODE in random correlated; do
  ./build/bench_filtered --base $B --query $Q --attr-mode $MODE --max-queries $MQ --threads $T --out results/filtered_$MODE.csv
done
python3 scripts/plot_filtered.py results/filtered_random.csv results/filtered_correlated.csv results/phase2_filtered.png
echo "done: results/phase2_filtered.png"

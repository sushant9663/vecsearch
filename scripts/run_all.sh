#!/usr/bin/env bash
# Phase 1: build, then brute force + FAISS HNSW + our HNSW (unlocked and locked), plot.
#   ./scripts/run_all.sh [dataset_dir] [max_queries]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_paths.sh "${1:-data/sift}"; MQ="${2:--1}"
cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build -j >/dev/null
ctest --test-dir build --output-on-failure
SW=16,32,64,128,256,512
./build/bench --base $B --query $Q --gt $G --index bruteforce --max-queries $MQ --repeat 2 --out results/bruteforce.csv
python3 scripts/faiss_baseline.py --base $B --query $Q --gt $G --index hnsw --sweep $SW --max-queries $MQ --out results/faiss_hnsw.csv >/dev/null
./build/bench --base $B --query $Q --gt $G --index hnsw --sweep $SW --max-queries $MQ --out results/hnsw.csv
./build/bench --base $B --query $Q --gt $G --index hnsw --threadsafe 1 --sweep $SW --max-queries $MQ --out results/hnsw_locked.csv
python3 scripts/plot.py results/phase1_curve.png results/bruteforce.csv results/faiss_hnsw.csv results/hnsw.csv results/hnsw_locked.csv
echo "done: results/phase1_curve.png"

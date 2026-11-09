#!/usr/bin/env python3
"""FAISS baseline in the same CSV format as the C++ bench.
  faiss_baseline.py --base B --query Q --gt G --index {flat,hnsw} [--sweep 16,32,64,128] [--M 16] [--efc 200]
Single-query, single-thread by default so it is comparable with `bench --threads 1`."""
import argparse, time
import numpy as np
import faiss

def read_vecs(path, dtype):
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    return raw.reshape(-1, d + 1)[:, 1:].copy().view(dtype)

def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return -1.0

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--query", required=True)
ap.add_argument("--gt", required=True); ap.add_argument("--index", default="hnsw")
ap.add_argument("--k", type=int, default=10); ap.add_argument("--sweep", default="16,32,64,128,256")
ap.add_argument("--M", type=int, default=16); ap.add_argument("--efc", type=int, default=200)
ap.add_argument("--repeat", type=int, default=3); ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--max-queries", type=int, default=-1); ap.add_argument("--out", default="")
a = ap.parse_args()

faiss.omp_set_num_threads(a.threads)
base = read_vecs(a.base, np.float32); xq = read_vecs(a.query, np.float32)
gt = read_vecs(a.gt, np.int32)
if a.max_queries > 0: xq, gt = xq[:a.max_queries], gt[:a.max_queries]
nq, d, k = len(xq), base.shape[1], a.k

rss0 = rss_mb(); t0 = time.perf_counter()
if a.index == "flat":
    idx = faiss.IndexFlatL2(d); name = "faiss_flat"; sweep = [0]
else:
    idx = faiss.IndexHNSWFlat(d, a.M); idx.hnsw.efConstruction = a.efc
    name = f"faiss_hnsw_M{a.M}"; sweep = [int(x) for x in a.sweep.split(",")]
idx.add(base)
build_s = time.perf_counter() - t0
index_mb = rss_mb() - rss0

rows = ["index,param,k,threads,recall,qps,p50_us,p99_us,build_s,index_mb"]
for p in sweep:
    if a.index == "hnsw": idx.hnsw.efSearch = p
    best_qps, best_lat, best_res = 0, None, None
    for _ in range(a.repeat):
        lat = np.empty(nq); res = np.empty((nq, k), dtype=np.int64)
        ts = time.perf_counter()
        for q in range(nq):
            s = time.perf_counter()
            _, ids = idx.search(xq[q:q+1], k)
            lat[q] = (time.perf_counter() - s) * 1e6
            res[q] = ids[0]
        qps = nq / (time.perf_counter() - ts)
        if qps > best_qps: best_qps, best_lat, best_res = qps, lat, res
    hit = sum(len(set(best_res[q]) & set(gt[q, :k])) for q in range(nq))
    recall = hit / (nq * k)
    lat = np.sort(best_lat)
    rows.append(f"{name},{p},{k},{a.threads},{recall:.6f},{best_qps:.3f},"
                f"{lat[nq//2]:.2f},{lat[min(nq-1,int(nq*0.99))]:.2f},{build_s:.3f},{index_mb:.1f}")
out = "\n".join(rows) + "\n"
if a.out: open(a.out, "w").write(out)
print(out, end="")

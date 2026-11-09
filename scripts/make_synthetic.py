#!/usr/bin/env python3
"""Tiny synthetic dataset in fvecs/ivecs format to smoke-test the harness without SIFT1M."""
import os, sys
import numpy as np
out = sys.argv[1] if len(sys.argv) > 1 else "data/synth"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
d = int(sys.argv[3]) if len(sys.argv) > 3 else 128
nq, k = 200, 100
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
base = rng.standard_normal((n, d)).astype(np.float32)
q = rng.standard_normal((nq, d)).astype(np.float32)
def wf(path, a):
    with open(path, "wb") as f:
        for r in a:
            np.int32(len(r)).tofile(f); r.tofile(f)
d2 = (q**2).sum(1)[:, None] - 2 * q @ base.T + (base**2).sum(1)[None, :]
gt = np.argsort(d2, axis=1)[:, :k].astype(np.int32)
wf(f"{out}/base.fvecs", base); wf(f"{out}/query.fvecs", q); wf(f"{out}/gt.ivecs", gt)
print("wrote", out)

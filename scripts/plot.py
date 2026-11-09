#!/usr/bin/env python3
"""Plot recall@k vs QPS from bench CSVs: plot.py out.png a.csv b.csv ..."""
import sys
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out, files = sys.argv[1], sys.argv[2:]
plt.figure(figsize=(6, 4))
for f in files:
    df = pd.read_csv(f)
    for name, g in df.groupby("index"):
        g = g.sort_values("recall")
        plt.plot(g["recall"], g["qps"], marker="o", label=name)
plt.yscale("log"); plt.xlabel("recall@10"); plt.ylabel("QPS (log)"); plt.grid(alpha=.3); plt.legend()
plt.tight_layout(); plt.savefig(out, dpi=150)

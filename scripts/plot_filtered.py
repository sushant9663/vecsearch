#!/usr/bin/env python3
"""plot_filtered.py a.csv b.csv out.png : recall vs QPS per (filter type, selectivity), per strategy."""
import sys
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
*files, out = sys.argv[1:]
df = pd.concat([pd.read_csv(f) for f in files])
rows = sorted(df.groupby(["attr_mode", "filter"]).groups)
sels = sorted(df.selectivity.unique())
fig, axes = plt.subplots(len(rows), len(sels), figsize=(4 * len(sels), 3.2 * len(rows)), squeeze=False)
for i, (mode, ft) in enumerate(rows):
    for j, s in enumerate(sels):
        ax = axes[i][j]
        sub = df[(df.attr_mode == mode) & (df["filter"] == ft) & (df.selectivity == s)]
        for strat, g in sub.groupby("strategy"):
            g = g.sort_values("recall")
            ax.plot(g.recall, g.qps, marker="o", label=strat)
        ax.set_yscale("log"); ax.grid(alpha=.3); ax.set_title(f"{mode}/{ft} sel={s:g}", fontsize=9)
        if i == len(rows) - 1: ax.set_xlabel("recall@10")
        if j == 0: ax.set_ylabel("QPS (log)")
        if i == 0 and j == 0: ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(out, dpi=140)

#!/usr/bin/env python3
"""plot_stream.py stream.csv out.png : query p99 and QPS, static vs mixed load."""
import sys
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
df = pd.read_csv(sys.argv[1])
df["name"] = df.label + "\n" + df["mode"] + "\nI=" + df.insert_threads.astype(str)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(df.name, df.p99_us / 1000); ax[0].set_ylabel("query p99 (ms)")
ax[1].bar(df.name, df.insert_per_s); ax[1].set_ylabel("insert throughput (vec/s)")
for a in ax: a.tick_params(axis="x", labelsize=6); a.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=140)

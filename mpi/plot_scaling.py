#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, os
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------- helpers ----------
def read_series(filepath, required_cols):
    rows = []
    with open(filepath, newline="") as f:
        r = csv.DictReader(f)
        miss = [c for c in required_cols if c not in r.fieldnames]
        if miss:
            raise SystemExit(f"{filepath} missing columns: {miss}")
        for row in r:
            rows.append(row)
    return rows

def groups(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return g

def ensure_outdir(d):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

# ---------- plotting ----------
def plot_strong(strong_csv, outdir):
    if not strong_csv: return
    rows = read_series(strong_csv, ["label", "ranks", "ms"])
    for r in rows:
        r["ranks"] = int(r["ranks"]); r["ms"] = float(r["ms"])
    bylab = groups(rows, "label")

    # Time vs ranks
    plt.figure(figsize=(7,4.5))
    for lab, rs in bylab.items():
        rs = sorted(rs, key=lambda x: x["ranks"])
        P  = [x["ranks"] for x in rs]
        T  = [x["ms"]/1000.0 for x in rs]
        plt.plot(P, T, marker="o", label=lab)
    plt.xlabel("MPI ranks (P)")
    plt.ylabel("Runtime (s)")
    plt.title("Strong Scaling — Runtime")
    plt.grid(True, alpha=0.4); plt.legend()
    savefig(os.path.join(outdir, "strong_runtime.png"))

    # Speedup & efficiency (baseline = P=1 time per label, or smallest P if 1 not present)
    plt.figure(figsize=(7,4.5))
    for lab, rs in bylab.items():
        rs = sorted(rs, key=lambda x: x["ranks"])
        P  = [x["ranks"] for x in rs]
        T  = [x["ms"]/1000.0 for x in rs]
        # baseline:
        baseline_time = None
        if 1 in P: baseline_time = T[P.index(1)]
        else:      baseline_time = T[0]
        S  = [baseline_time/t for t in T]
        plt.plot(P, S, marker="o", label=lab)
    # ideal line (S=P) over union of P
    allP = sorted(set([int(r["ranks"]) for r in rows]))
    plt.plot(allP, allP, "--", label="ideal (S=P)")
    plt.xlabel("MPI ranks (P)"); plt.ylabel("Speedup S(P)")
    plt.title("Strong Scaling — Speedup")
    plt.grid(True, alpha=0.4); plt.legend()
    savefig(os.path.join(outdir, "strong_speedup.png"))

    # Efficiency
    plt.figure(figsize=(7,4.5))
    for lab, rs in bylab.items():
        rs = sorted(rs, key=lambda x: x["ranks"])
        P  = [x["ranks"] for x in rs]
        T  = [x["ms"]/1000.0 for x in rs]
        baseline_time = T[P.index(1)] if 1 in P else T[0]
        S  = [baseline_time/t for t in T]
        E  = [100.0*s/p for s,p in zip(S,P)]
        plt.plot(P, E, marker="o", label=lab)
    plt.xlabel("MPI ranks (P)"); plt.ylabel("Efficiency (%)")
    plt.title("Strong Scaling — Efficiency")
    plt.grid(True, alpha=0.4); plt.legend()
    savefig(os.path.join(outdir, "strong_efficiency.png"))

def plot_batch(bs_csv, outdir):
    if not bs_csv: return
    rows = read_series(bs_csv, ["label", "bs", "ms"])
    for r in rows:
        r["bs"] = int(r["bs"]); r["ms"] = float(r["ms"])
    bylab = groups(rows, "label")

    plt.figure(figsize=(7,4.5))
    for lab, rs in bylab.items():
        rs = sorted(rs, key=lambda x: x["bs"])
        X  = [x["bs"] for x in rs]
        Y  = [x["ms"]/1000.0 for x in rs]
        plt.plot(X, Y, marker="o", label=lab)
    plt.xlabel("Batch size")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs Batch Size")
    plt.grid(True, alpha=0.4); plt.legend()
    savefig(os.path.join(outdir, "batch_runtime.png"))

def plot_hybrid(hybrid_csv, outdir):
    if not hybrid_csv: return
    rows = read_series(hybrid_csv, ["label","P","T","ms"])
    for r in rows:
        r["P"] = int(r["P"]); r["T"] = int(r["T"]); r["ms"] = float(r["ms"])
    bylab = groups(rows, "label")

    plt.figure(figsize=(7,4.5))
    for lab, rs in bylab.items():
        rs = sorted(rs, key=lambda x: x["P"])
        X  = [f'{x["P"]}×{x["T"]}' for x in rs]
        Y  = [x["ms"]/1000.0 for x in rs]
        plt.plot(X, Y, marker="o", label=lab)
    plt.xlabel("(MPI ranks × OMP threads) — constant total cores")
    plt.ylabel("Runtime (s)")
    plt.title("Hybrid Splits")
    plt.grid(True, alpha=0.4); plt.legend()
    savefig(os.path.join(outdir, "hybrid_runtime.png"))

# ---------- main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot MPI/OpenMP benchmark results.")
    ap.add_argument("--strong", help="CSV with columns: label,ranks,ms")
    ap.add_argument("--batch",  help="CSV with columns: label,bs,ms")
    ap.add_argument("--hybrid", help="CSV with columns: label,P,T,ms")
    ap.add_argument("--outdir", default="plots", help="Directory for output PNGs")
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    plot_strong(args.strong, args.outdir)
    plot_batch(args.batch, args.outdir)
    plot_hybrid(args.hybrid, args.outdir)

    print("Saved plots to:", os.path.abspath(args.outdir))

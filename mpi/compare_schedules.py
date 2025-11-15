#!/usr/bin/env python3
import argparse
import csv
import os
import matplotlib.pyplot as plt


def load_curve(path):
    epochs = []
    losses = []
    accs = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
            accs.append(float(row["accuracy"]))
    return epochs, losses, accs


def main():
    ap = argparse.ArgumentParser(description="Compare LR schedules (loss & accuracy).")
    ap.add_argument("--logdir", default="output")
    ap.add_argument("--prefix", default="leaky")
    args = ap.parse_args()

    configs = [
        ("constant",   "const"),
        ("time-based", "timebased"),
        ("exponential","exp"),
        ("step-based", "step"),
    ]

    curves = {}

    for label, suffix in configs:
        fname = f"{args.prefix}_{suffix}.csv"
        path = os.path.join(args.logdir, fname)
        if not os.path.isfile(path):
            print(f"[WARN] Missing: {path}")
            continue
        e, l, a = load_curve(path)
        curves[label] = (e, l, a)
        print(f"[INFO] Loaded {label} ({len(e)} points)")

    if not curves:
        print("No CSV files loaded. Check your directory.")
        return

    # ---------- Plot LOSS ----------
    plt.figure(figsize=(9, 6))
    for label, (e, l, _) in curves.items():
        plt.plot(e, l, marker="o", label=label)
    plt.title("Convergence (Loss) vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_vs_epoch.png", dpi=120)
    print("✔ Saved: loss_vs_epoch.png")

    # ---------- Plot ACC ----------
    plt.figure(figsize=(9, 6))
    for label, (e, _, a) in curves.items():
        plt.plot(e, a, marker="o", label=label)
    plt.title("Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_vs_epoch.png", dpi=120)
    print("✔ Saved: accuracy_vs_epoch.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import os

import matplotlib.pyplot as plt


def load_curve(path):
    """Return (epochs, losses, accuracies, lrs) from a log CSV."""
    epochs = []
    losses = []
    accs = []
    lrs = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
            accs.append(float(row["accuracy"]))
            # lr column from your C code: "epoch,loss,lr,accuracy"
            if "lr" in row and row["lr"] != "":
                lrs.append(float(row["lr"]))
            else:
                lrs.append(float("nan"))
    return epochs, losses, accs, lrs


def main():
    ap = argparse.ArgumentParser(
        description="Compare LR schedules (loss, accuracy, and LR vs epoch)."
    )
    ap.add_argument(
        "--logdir",
        default="output",
        help="Directory containing *_const.csv, *_timebased.csv, *_exp.csv, *_step.csv",
    )
    ap.add_argument(
        "--prefix",
        default="leaky",
        help="Filename prefix before schedule name (e.g. leaky_const.csv)",
    )
    args = ap.parse_args()

    # Map: human label -> filename suffix (must match decay_name() in C code)
    configs = [
        ("constant", "const"),
        ("time-based", "timebased"),
        ("exponential", "exp"),
        ("step-based", "step"),
    ]

    curves = {}

    for label, suffix in configs:
        fname = f"{args.prefix}_{suffix}.csv"
        path = os.path.join(args.logdir, fname)
        if not os.path.isfile(path):
            print(f"[WARN] Missing file: {path} (skipping {label})")
            continue
        epochs, losses, accs, lrs = load_curve(path)
        curves[label] = (epochs, losses, accs, lrs)
        print(
            f"[INFO] Loaded {label} from {path} with {len(epochs)} points."
        )

    if not curves:
        print("No curves loaded. Check --logdir and --prefix.")
        return

    # ---------- Loss vs epoch ----------
    plt.figure(figsize=(8, 6))
    for label, (epochs, losses, _, _) in curves.items():
        plt.plot(epochs, losses, marker="o", linewidth=2, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Convergence (Loss) vs Epoch")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_vs_epoch.png", dpi=120)
    print("Saved loss plot to loss_vs_epoch.png")

    # ---------- Accuracy vs epoch ----------
    plt.figure(figsize=(8, 6))
    for label, (epochs, _, accs, _) in curves.items():
        plt.plot(epochs, accs, marker="o", linewidth=2, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_vs_epoch.png", dpi=120)
    print("Saved accuracy plot to accuracy_vs_epoch.png")

    # ---------- Learning rate vs epoch ----------
    plt.figure(figsize=(8, 6))
    for label, (epochs, _, _, lrs) in curves.items():
        plt.plot(epochs, lrs, marker="o", linewidth=2, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate ηₜ")
    plt.title("Learning Rate vs Epoch")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lr_vs_epoch.png", dpi=120)
    print("Saved LR plot to lr_vs_epoch.png")


if __name__ == "__main__":
    main()

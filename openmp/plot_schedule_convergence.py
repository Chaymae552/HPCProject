#!/usr/bin/env python3
import pandas as pd, matplotlib.pyplot as plt

files = {
    "const": "output/leaky_const.csv",
    "time":  "output/leaky_timebased.csv",
    "exp":   "output/leaky_exp.csv",
    "step":  "output/leaky_step.csv",
}

plt.figure(figsize=(8,4))
for label, path in files.items():
    df = pd.read_csv(path)
    plt.plot(df["epoch"], df["loss"], label=f"{label} (loss)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch — different LR schedules")
plt.legend()
plt.tight_layout()
plt.savefig("loss_vs_epoch.png", dpi=150)

plt.figure(figsize=(8,4))
for label, path in files.items():
    df = pd.read_csv(path)
    plt.plot(df["epoch"], df["accuracy"], label=f"{label} (acc)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch — different LR schedules")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_vs_epoch.png", dpi=150)

print("Saved: loss_vs_epoch.png, accuracy_vs_epoch.png")
plt.figure(figsize=(8,4))
for label, path in files.items():
    df = pd.read_csv(path)
    plt.plot(df["epoch"], df["lr"], label=f"{label} (lr)")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate vs Epoch — schedule evolution")
plt.legend()
plt.tight_layout()
plt.savefig("lr_vs_epoch.png", dpi=150)
print("Saved: lr_vs_epoch.png")

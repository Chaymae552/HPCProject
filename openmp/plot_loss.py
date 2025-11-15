# plot_loss.py
import csv, sys
import matplotlib.pyplot as plt

in_csv  = sys.argv[1] if len(sys.argv) > 1 else "output/loss_minibatch.csv"
out_png = sys.argv[2] if len(sys.argv) > 2 else "output/loss_minibatch.png"
title   = sys.argv[3] if len(sys.argv) > 3 else "Mini-batch Training Loss"

epochs, losses = [], []
with open(in_csv, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        epochs.append(int(row["epoch"]))
        losses.append(float(row["loss"]))

plt.figure(figsize=(8,5))
plt.plot(epochs, losses, marker="", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss (cross-entropy)")
plt.title(title)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(out_png, dpi=150)
print(f"Saved {out_png}")

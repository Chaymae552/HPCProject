import pandas as pd
import matplotlib.pyplot as plt

# Load the benchmark data
df = pd.read_csv("output/bench.csv")

# Compute speedup relative to single-thread time
t1 = df[df["threads"] == 1]["time_ms"].mean()
df["speedup"] = t1 / df["time_ms"]

plt.figure(figsize=(6,4))
plt.plot(df["threads"], df["speedup"], "o-", lw=2)
plt.title("OpenMP Speedup Scaling")
plt.xlabel("Number of Threads")
plt.ylabel("Speedup (T1 / Tn)")
plt.grid(True)
plt.xticks(df["threads"])
plt.tight_layout()
plt.show()

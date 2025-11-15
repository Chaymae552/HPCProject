import pandas as pd, matplotlib.pyplot as plt, os
files=[("output/tanh_timebased.csv","tanh"),
       ("output/relu_timebased.csv","relu"),
       ("output/sigmoid_timebased.csv","sigmoid"),
       ("output/leaky_timebased.csv","leaky-relu")]
dfs=[]
for fn,label in files:
    if os.path.exists(fn):
        df=pd.read_csv(fn); df["act"]=label; dfs.append(df)
if not dfs: raise SystemExit("No activation CSVs found.")
all_df=pd.concat(dfs, ignore_index=True)

plt.figure()
for k,g in all_df.groupby("act"):
    plt.plot(g.epoch, g.loss, label=k)
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch by Activation")
plt.grid(True); plt.legend(); plt.savefig("output/act_compare_loss.png", dpi=180)

plt.figure()
for k,g in all_df.groupby("act"):
    plt.plot(g.epoch, g.accuracy, label=k)
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy vs Epoch by Activation")
plt.grid(True); plt.legend(); plt.savefig("output/act_compare_acc.png", dpi=180)
print("Saved: output/act_compare_loss.png, output/act_compare_acc.png")


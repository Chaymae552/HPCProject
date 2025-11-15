# generate_moon.py
from sklearn.datasets import make_moons
import numpy as np
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--n-samples", type=int, default=100, help="number of points")
parser.add_argument("--noise", type=float, default=0.20)
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--outdir", type=str, default="data")
args = parser.parse_args()

X, y = make_moons(args.n_samples, noise=args.noise, random_state=args.seed)
os.makedirs(args.outdir, exist_ok=True)
np.savetxt(f"{args.outdir}/data_X.txt", X)
np.savetxt(f"{args.outdir}/data_y.txt", y, fmt="%d")
print(f"Wrote {len(y)} samples to {args.outdir}/data_X.txt and data_y.txt")

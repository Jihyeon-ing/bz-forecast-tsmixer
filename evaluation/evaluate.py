"""Reproduce the Bz evaluation numbers of the paper from a predictions file.

    python evaluation/evaluate.py                      # released predictions
    python evaluation/evaluate.py --pred results/my_run.npz

Prints  (all in nT, test split 2019-2025, N = 1,335 windows)
  Figure 2 : MAE and bias vs lead time, model vs persistence (last observed Bz held constant)
  Table 2  : point-level MAE / bias by observed-Bz range over the first 6 h (whole test set),
             and the window-minimum bias for windows with observed minimum <= -20 nT
  Table 1  : the same ranges on the windows coincident with Son et al. (2025), for both models
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

BZ_CHANNEL = 1
RANGES = [("Bz > -10", lambda t: t > -10),
          ("-20 <= Bz <= -10", lambda t: (t <= -10) & (t >= -20)),
          ("Bz < -20", lambda t: t < -20)]


def range_table(pred, tar, label):
    print(f"\n{label}")
    print(f"  {'range':18s} {'N':>7s} {'MAE':>7s} {'bias':>7s}")
    p, t = pred.ravel(), tar.ravel()
    for name, f in RANGES:
        m = f(t)
        print(f"  {name:18s} {m.sum():7d} {np.abs(p[m] - t[m]).mean():7.2f} {(p[m] - t[m]).mean():+7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results/test_result.npz")
    ap.add_argument("--baseline", default="results/test_result_son2025.npz")
    args = ap.parse_args()

    r = np.load(args.pred, allow_pickle=True)
    bz_min, bz_max = float(r["bz_min"]), float(r["bz_max"])
    dn = lambda x: np.asarray(x) * (bz_max - bz_min) + bz_min
    pred, tar = dn(r["pred"]), dn(r["tar"])
    persist = np.repeat(dn(r["inp"][:, -1, BZ_CHANNEL])[:, None], pred.shape[1], axis=1)
    tar_t = pd.to_datetime(r["tar_t"][:, 0])
    print(f"Test samples: {len(pred)}  ({tar_t.min().date()} .. {tar_t.max().date()})")

    print("\nFigure 2: lead [h]  MAE_model  MAE_persist  bias_model  bias_persist")
    for i in range(1, pred.shape[1], 2):
        print(f"  {(i+1)/2:4.1f}  {np.abs(pred[:, i]-tar[:, i]).mean():7.2f}  {np.abs(persist[:, i]-tar[:, i]).mean():7.2f}"
              f"  {(pred[:, i]-tar[:, i]).mean():+7.2f}  {(persist[:, i]-tar[:, i]).mean():+7.2f}")

    range_table(pred[:, :12], tar[:, :12], "Table 2 (full-model row): first 6 h, whole test set")
    mn_t, mn_p = tar[:, :12].min(1), pred[:, :12].min(1)
    m = mn_t <= -20
    print(f"  window-minimum bias (obs. min <= -20 nT, N={m.sum()}): {(mn_p[m]-mn_t[m]).mean():+.2f} nT")

    if os.path.exists(args.baseline):
        s = np.load(args.baseline, allow_pickle=True)          # already in nT, 12 steps
        s_t = pd.to_datetime(s["tar_t"][:, 0])
        idx = {t: i for i, t in enumerate(tar_t)}
        pairs = [(idx[t], j) for j, t in enumerate(s_t) if t in idx]
        ii, jj = map(np.array, zip(*pairs))
        print(f"\nTable 1: {len(pairs)} windows coincident with Son et al. (2025), first 6 h")
        range_table(pred[ii, :12], tar[ii, :12], "  This study")
        range_table(s["pred"][jj], s["tar"][jj], "  Son et al. (2025)")


if __name__ == "__main__":
    main()

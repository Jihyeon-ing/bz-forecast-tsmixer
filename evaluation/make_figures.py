# -*- coding: utf-8 -*-
"""
Paper figure generation for:
  "Deep Learning-based IMF Bz Forecasting and Its Application to
   Geomagnetic Index Prediction" (Space Weather)

Manuscript figure -> function -> output file
  Figure 2 : fig_bz_leadtime()   -> Figure/Figure_bz_leadtime.png
  Figure 3 : fig_bz_examples()   -> Figure/Figure2.png  (Bz example panels)
  Figure 4 : fig_dst_scatter()   -> Figure/Figure3.png  (DL-input)
  Figure 5 : fig_dst_scatter()   -> Figure/Figure4.png  (Persistent-input)
  Figure 6 : fig_dst_metrics()   -> Figure/Figure5.png  (CC/RMSE vs lead)
  Figure 7 : fig_dst_storms()    -> Figure/Figure6.png  (storm examples)
  Figure 8 : fig_symh()          -> Figure/Figure7.png  (SYM-H extension)

Usage:  python evaluation/make_figures.py   (run from anywhere; paths are relative to the repo root)
"""
import datetime
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from scipy.stats import pearsonr

import dst_models as dst_eqs

# ---------------------------------------------------------------- data loading
import json
_s = json.load(open('trained_model/tsmixer_bz/scaler.json'))
MINV, MAXV = np.array(_s['min_values']), np.array(_s['max_values'])
BZ_MIN, BZ_MAX = _s['bz_min'], _s['bz_max']
SYMH_MIN, SYMH_MAX = -475.3333333333333, 106.0   # scaler of the SYM-H model (Son et al., 2026)

OMNI_CSV = "data/omni_hourly_2001_2025.csv"


def load_results():
    r = np.load("results/test_result.npz", allow_pickle=True)
    pred = r["pred"] * (BZ_MAX - BZ_MIN) + BZ_MIN
    tar = r["tar"] * (BZ_MAX - BZ_MIN) + BZ_MIN
    inp = r["inp"] * (MAXV - MINV) + MINV
    tar_t = pd.to_datetime(r["tar_t"])
    return inp, tar, pred, tar_t


def load_omni():
    omni = pd.read_csv(OMNI_CSV)
    omni["time"] = pd.to_datetime(omni["time"])
    omni.set_index("time", inplace=True)
    return omni


# ------------------------------------------------- Figure 2: Bz lead-time skill
def fig_bz_leadtime(out="figures/Figure_bz_leadtime.png"):
    inp, tar, pred, _ = load_results()
    persist = np.repeat(inp[:, -1, 1][:, None], 24, axis=1)
    leads = np.arange(1, 25) * 0.5
    strong = tar.min(axis=1) <= -10

    def curves(y, yh, m):
        mae = [np.mean(np.abs(yh[m, k] - y[m, k])) for k in range(24)]
        bias = [np.mean(yh[m, k] - y[m, k]) for k in range(24)]
        return np.array(mae), np.array(bias)

    allm = np.ones(len(tar), bool)
    m_mae, m_bias = curves(tar, pred, allm)
    p_mae, p_bias = curves(tar, persist, allm)
    ms_mae, ms_bias = curves(tar, pred, strong)
    ps_mae, ps_bias = curves(tar, persist, strong)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), facecolor="white")
    panels = [("(a) MAE", "MAE [nT]", m_mae, p_mae, ms_mae, ps_mae),
              ("(b) Bias", "Bias [nT]", m_bias, p_bias, ms_bias, ps_bias)]
    for ax, (title, ylab, a, b, c, d) in zip(axes, panels):
        ax.plot(leads, a, "o-", c="blue", ms=4, label="This study (all)")
        ax.plot(leads, b, "*-", c="forestgreen", ms=6, label="Persistence (all)")
        ax.plot(leads, c, "o--", c="blue", ms=4, alpha=0.55,
                label=r"This study ($B_z^{min} \leq -10$ nT)")
        ax.plot(leads, d, "*--", c="forestgreen", ms=6, alpha=0.55,
                label=r"Persistence ($B_z^{min} \leq -10$ nT)")
        if "Bias" in title:
            ax.axhline(0, color="k", linewidth=0.8)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=11, fontweight="bold")
        ax.set_xlabel("Forecast time [hours]", fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(0, 13, 2))
        ax.grid(alpha=0.4)
    axes[0].legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("saved", out)


# -------------------------------------- Figure 3: Bz example panels (a)-(d)
def fig_bz_examples(out="figures/Figure3.png"):
    """Four representative Bz predictions with the previous model overlaid.

    Panels: (a) southward turning near the prediction time, (b) moderate
    event, (c) strongest storm in the test set, (d) failure case.
    """
    inp, tar, pred, tar_t = load_results()
    prev = np.load("results/test_result_son2025.npz", allow_pickle=True)
    prev_pred = prev["pred"]  # already in nT, 12 steps (6 hours)
    prev_t0 = pd.to_datetime([t[0] for t in prev["tar_t"]])
    prev_map = {t: j for j, t in enumerate(prev_t0)}

    panels = [(585, "(a)"), (774, "(b)"), (951, "(c)"), (615, "(d)")]
    t_obs = np.concatenate([np.arange(-5, 1) * 0.5, np.arange(1, 25) * 0.5])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="white")
    for ax, (i, lab) in zip(axes.flat, panels):
        ptime = tar_t[i][0] - datetime.timedelta(minutes=30)
        obs = np.concatenate([inp[i, -6:, 1], tar[i]])
        ax.plot(t_obs, obs, label="Target Bz", color="dimgray",
                linewidth=2, marker="D", markersize=5)
        ax.plot(np.arange(0, 25) * 0.5,
                np.concatenate([[inp[i, -1, 1]], pred[i]]),
                label="Predicted Bz (This study)", color="blue",
                linestyle="--", linewidth=2, marker="o", markersize=5)
        j = prev_map.get(tar_t[i][0])
        if j is not None:
            ax.plot(np.arange(0, 13) * 0.5,
                    np.concatenate([[inp[i, -1, 1]], prev_pred[j]]),
                    label="Predicted Bz (Son+2025)", color="red",
                    linestyle="--", linewidth=1.5, marker="*", markersize=6)
        ax.axvline(0, color="red", linestyle="dashed")
        ax.set_title(f"{lab} Prediction time: {str(ptime)[:-3]} UTC",
                     fontsize=12, fontweight="bold")
        ax.set_ylim(-50, 15)
        ax.set_xticks(np.arange(-2, 13, 2))
        ax.set_xticklabels(["-2h", "0", "+2h", "+4h", "+6h", "+8h",
                            "+10h", "+12h"])
        ax.set_ylabel("Bz [nT]", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.4)
        ax.legend(fontsize=8, loc="lower right")
    for ax in axes[1]:
        ax.set_xlabel("Hours from the prediction time",
                      fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("saved", out)


# ------------------------------ O-M evaluation matrices (shared by Figs 4,5,6)
def om_eval_matrices():
    """Time-aligned O-M Dst evaluation at leads +1..+12 h.

    Model values are taken at the exact lead; hourly observed Dst is
    linearly interpolated for events starting on the half hour.
    Returns (obs, dl, per, obz), each (N, 12) in nT.
    """
    inp, tar, pred, tar_t = load_results()
    omni = load_omni()
    dst = omni["DST"]
    N = len(tar_t)
    t_grid = np.arange(0, 12.5, 0.5)
    obs_m = np.full((N, 12), np.nan)
    mats = {k: np.full((N, 12), np.nan) for k in ("dl", "per", "obz")}

    for i in range(N):
        ptime = tar_t[i][0] - datetime.timedelta(minutes=30)
        t_init = ptime.floor("h") if ptime.minute else ptime
        try:
            dst0 = dst.loc[t_init]
        except KeyError:
            continue
        if not np.isfinite(dst0):
            continue
        v = np.full_like(t_grid, inp[i, -1, 2])
        n = np.full_like(t_grid, inp[i, -1, 3])
        runs = {
            "dl": dst_eqs.obrien_mcpherron_dst(
                t_grid, v, n, np.append(inp[i, -1, 1], pred[i]), dst0),
            "per": dst_eqs.obrien_mcpherron_dst(
                t_grid, v, n, np.full_like(t_grid, inp[i, -1, 1]), dst0),
            "obz": dst_eqs.obrien_mcpherron_dst(
                t_grid, v, n, np.append(inp[i, -1, 1], tar[i]), dst0),
        }
        for h in range(1, 13):
            t_h = ptime + datetime.timedelta(hours=h)
            if t_h.minute == 0:
                o = dst.get(t_h, np.nan)
            else:
                lo = dst.get(t_h.floor("h"), np.nan)
                hi = dst.get(t_h.ceil("h"), np.nan)
                o = 0.5 * (lo + hi)
            obs_m[i, h - 1] = o
            for k in mats:
                mats[k][i, h - 1] = runs[k][2 * h]
    return obs_m, mats["dl"], mats["per"], mats["obz"]


# ------------------------------------- Figures 4 & 5: Dst scatter (12 panels)
def fig_dst_scatter():
    obs_m, dl_m, per_m, _ = om_eval_matrices()

    def one(y_mat, ylabel, fname):
        fig, axes = plt.subplots(3, 4, figsize=(12, 8))
        for h in range(12):
            ax = axes.flat[h]
            x, y = obs_m[:, h], y_mat[:, h]
            m = np.isfinite(x) & np.isfinite(y)
            cc = pearsonr(x[m], y[m])[0]
            rmse = np.sqrt(np.mean((x[m] - y[m]) ** 2))
            ax.scatter(x[m], y[m], color="k", marker="D", s=20)
            ax.plot([-400, 50], [-400, 50], "r--", lw=1.3, zorder=10)
            ax.set_title(f"{h+1}h prediction \n (CC={cc:.2f},  RMSE={rmse:.2f} nT)")
            ax.set_xlim(-400, 50)
            ax.set_ylim(-400, 50)
        fig.supxlabel("Observed Dst [nT]", fontsize=14, fontweight="bold")
        fig.supylabel(ylabel, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
        print("saved", fname)

    one(dl_m, "DL-input Dst [nT]", "figures/Figure4.png")
    one(per_m, "Persistent-input Dst [nT]", "figures/Figure5.png")


# --------------------------------- Figure 6: CC / RMSE vs forecast hour (Dst)
def fig_dst_metrics(out="figures/Figure6.png"):
    obs_m, dl_m, per_m, obz_m = om_eval_matrices()
    leads = np.arange(1, 13)
    cc, rmse = {}, {}
    for k, mat in [("dl", dl_m), ("per", per_m), ("obz", obz_m)]:
        cc[k], rmse[k] = np.zeros(12), np.zeros(12)
        for h in range(12):
            x, y = obs_m[:, h], mat[:, h]
            m = np.isfinite(x) & np.isfinite(y)
            cc[k][h] = pearsonr(x[m], y[m])[0]
            rmse[k][h] = np.sqrt(np.mean((x[m] - y[m]) ** 2))

    styles = {
        "obz": dict(c="darkorange", marker="s", label="Observed Bz"),
        "dl": dict(c="blue", marker="o", label="Predicted Bz (this study)"),
        "per": dict(c="forestgreen", marker="*", markersize=9,
                    label="Persistent Bz"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    for ax, metric, name in [(axes[0], cc, "CC"), (axes[1], rmse, "RMSE [nT]")]:
        ax.fill_between(leads, metric["per"], metric["obz"],
                        color="gray", alpha=0.15, zorder=0)
        for k in ("obz", "dl", "per"):
            s = dict(styles[k])
            lab = s.pop("label")
            ax.plot(leads, metric[k], label=lab, linewidth=2,
                    markersize=s.pop("markersize", 6), **s)
        ax.set_xlabel("Forecast hour", fontsize=12, fontweight="bold")
        ax.set_ylabel(name, fontsize=12, fontweight="bold")
        ax.set_xticks(leads)
        ax.grid(alpha=0.4)
    axes[0].set_title("(a) Correlation coefficient", fontsize=12, fontweight="bold")
    axes[1].set_title("(b) RMSE", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10, loc="lower left")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("saved", out)


# ----------------------------------------- Figure 7: Dst storm examples (a,b)
def fig_dst_storms(out="figures/Figure7.png"):
    inp, tar, pred, tar_t = load_results()
    omni = load_omni()
    t_grid = np.arange(0, 12.5, 0.5)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8.5), facecolor="white")
    panels = [(774, "(a)", (-135, 60)), (951, "(b)", (-350, 60))]
    for ax, (i, lab, ylim) in zip(axes, panels):
        ptime = tar_t[i][0] - datetime.timedelta(minutes=30)
        w = omni[tar_t[i][0] - datetime.timedelta(hours=10):tar_t[i][-1]]
        wd, wv = w["DST"].values, w["V"].values
        wbz, wn = w["BZ_GSM"].values, w["N"].values
        dst0 = wd[9]
        v = np.full_like(t_grid, inp[i, -1, 2])
        n = np.full_like(t_grid, inp[i, -1, 3])
        d_dl = dst_eqs.obrien_mcpherron_dst(
            t_grid, v, n, np.append(inp[i, -1, 1], pred[i]), dst0)
        d_per = dst_eqs.obrien_mcpherron_dst(
            t_grid, v, n, np.full_like(t_grid, inp[i, -1, 1]), dst0)
        d_obz = dst_eqs.obrien_mcpherron_dst(
            t_grid, v, n, np.append(inp[i, -1, 1], tar[i]), dst0)
        d_hind = dst_eqs.obrien_mcpherron_dst(
            np.arange(0, 13, 1), wv[9:], wn[9:], wbz[9:],
            wd[9:9 + 16], prediction=False)

        w1 = omni[tar_t[i][0] - datetime.timedelta(hours=5):
                  tar_t[i][-1] + datetime.timedelta(hours=5)]
        wd1 = w1["DST"].values
        x0 = (ptime - w1.index[0]).total_seconds() / 3600.0
        x_om0 = ((tar_t[i][0] - datetime.timedelta(hours=10)) -
                 w1.index[0]).total_seconds() / 3600.0 + 9

        ax.plot(np.arange(len(wd1)), wd1, label="Dst (Ground Truth)",
                marker="D", markersize=5, color="dimgray", linewidth=2)
        ax.plot(x0 + t_grid, d_dl, label="O-M with predicted Bz",
                marker="o", markersize=5, c="blue")
        ax.plot(x0 + t_grid, d_per, label="O-M with persistent Bz",
                marker="*", markersize=8, c="forestgreen")
        ax.plot(x0 + t_grid, d_obz,
                label="O-M with observed Bz + persistent V, N",
                marker="s", markersize=4, c="darkorange")
        ax.plot(x_om0 + np.arange(13), d_hind,
                label="O-M 1-hr prediction (hourly re-initialized)",
                marker="^", markersize=6, c="red")
        ax.axvspan(x0, x0 + 12, alpha=0.2, color="pink")
        ax.vlines(x=x0, ymin=-400, ymax=200, color="crimson", linestyle="dashed")
        ax.vlines(x=x0 + 12, ymin=-400, ymax=200, color="crimson",
                  linestyle="dashed")
        ax.set_title(f"{lab} Prediction time: {str(ptime)[:-3]} UTC",
                     fontsize=13, fontweight="bold")
        ax.legend(loc="lower left", fontsize=8)
        ax.set_xticks(x0 + np.array([1, 3, 6, 9, 12]))
        ax.set_xticklabels(["+1h", "+3h", "+6h", "+9h", "+12h"])
        ax.xaxis.set_minor_locator(FixedLocator(np.arange(0, 25, 1)))
        ax.tick_params(axis="x", which="minor", length=0)
        ax.grid(True, which="major", axis="y", alpha=0.5)
        ax.grid(True, which="minor", axis="x", alpha=0.5)
        ax.grid(True, which="major", axis="x", alpha=0.5)
        ax.set_ylabel("Dst [nT]", fontsize=12, fontweight="bold")
        ax.set_ylim(*ylim)
    axes[1].set_xlabel("Hours from the prediction time", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("saved", out)


# --------------------------------------------- Figure 8: SYM-H extension (a,b)
def fig_symh(out="figures/Figure8.png"):
    symh_pred = (np.load("results/symh_pred_direct_test.npy", allow_pickle=True)
                 * (SYMH_MAX - SYMH_MIN) + SYMH_MIN)
    ext = {
        "a": np.load("results/symh_pred_extended_20240303.npy", allow_pickle=True)[0],
        "b": np.load("results/symh_pred_extended_20241010.npy", allow_pickle=True)[0],
    }  # v2 files are already in physical units (nT)
    total = pd.read_csv("data/symh_30min_1995_2025.csv", engine="python")
    time = pd.to_datetime(total["time"])
    sym_h = total["sym_h"].values

    events = {"a": ("2024-03-03 12:30:00", 739, (-250, 50)),
              "b": ("2024-10-10 19:00:00", 910, (-480, 60))}
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), facecolor="white")
    for ax, key, lab in zip(axes, ["a", "b"], ["(a)", "(b)"]):
        t0_str, idx, ylim = events[key]
        ts = np.where(time == t0_str)[0][0]
        ptime = time[ts - 1]
        obs = sym_h[ts - 24: ts + 48]
        ax.plot(np.arange(72), obs, color="dimgray", linewidth=2,
                marker="D", markersize=3, label="SYM-H (Ground Truth)")
        ax.plot(np.arange(23, 48), [sym_h[ts - 1]] + list(symh_pred[idx]),
                color="red", marker="^", markersize=4,
                label="Son+2026 with observed solar wind")
        ax.plot(np.arange(47, 72), [symh_pred[idx][-1]] + list(ext[key]),
                color="blue", marker="o", markersize=4,
                label="Son+2026 with predicted Bz")
        ax.axvspan(23, 47, alpha=0.15, color="forestgreen")
        ax.axvspan(47, 71, alpha=0.25, color="pink")
        for x in (23, 47, 71):
            ax.axvline(x, color="black", linestyle="dashed", linewidth=1)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, 71)
        ax.set_xticks([5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71])
        ax.set_xticklabels(["-9h", "-6h", "-3h", "0", "+3h", "+6h", "+9h",
                            "+12h", "+15h", "+18h", "+21h", "+24h"])
        ax.set_ylabel("SYM-H [nT]", fontsize=12, fontweight="bold")
        ax.set_title(f"{lab} Prediction time: {str(ptime)[:-3]} UTC",
                     fontsize=13, fontweight="bold")
        ax.grid(alpha=0.4)
        ax.legend(loc="lower left", fontsize=9)
    axes[1].set_xlabel("Hours from the prediction time", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("saved", out)


if __name__ == "__main__":
    fig_bz_leadtime()
    fig_bz_examples()
    fig_dst_scatter()
    fig_dst_metrics()
    fig_dst_storms()
    fig_symh()

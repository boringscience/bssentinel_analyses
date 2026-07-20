"""
dca.py — Decision Curve Analysis for bssentinel

Calibrates both XGBoost and NEWS2 outputs to actual event probabilities,
then computes net benefit across clinically relevant decision thresholds (0.5–8%).

Uses out-of-fold predictions from oof_predictions.parquet.

Usage:
    python dca.py [--oof artifacts/oof_predictions.parquet] [--out artifacts/figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ── Calibration ────────────────────────────────────────────────────────────────

def isotonic_calibrate(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    Isotonic regression calibration: maps raw scores → P(Y=1|score).
    Monotone, non-parametric.
    """
    ir = IsotonicRegression(out_of_bounds="clip")
    return ir.fit_transform(scores, y.astype(float))


def logistic_calibrate(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Platt scaling (sigmoid) for NEWS2 raw scores."""
    X = scores.reshape(-1, 1)
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(X_sc, y)
    return lr.predict_proba(X_sc)[:, 1]


# ── Net benefit ────────────────────────────────────────────────────────────────

def net_benefit_curve(y: np.ndarray, p_cal: np.ndarray,
                      thresholds: np.ndarray) -> np.ndarray:
    """
    NB(t) = TP/n - FP/n × t/(1-t)
    where classification rule is p_cal ≥ t.
    """
    n = len(y)
    nb = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        pred = p_cal >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        nb[i] = tp / n - fp / n * (t / (1.0 - t + 1e-15))
    return nb


def nb_treat_all(y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    prev = y.mean()
    return prev - (1.0 - prev) * thresholds / (1.0 - thresholds + 1e-15)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_dca(oof_path: Path, out_dir: Path) -> dict:
    df = pd.read_parquet(oof_path)
    y         = df["label"].values.astype(int)
    p_xgb_raw = df["xgb_proba"].values.astype(float)
    news2_raw = df["news2_total"].fillna(0).values.astype(float)

    prev = y.mean()
    n    = len(y)
    print(f"[dca] n={n:,}  events={y.sum():,}  prevalence={prev:.4f} ({prev*100:.2f}%)")

    # ── Calibrate both predictors → actual P(event) ────────────────────────────
    # XGBoost: isotonic regression (model is well-ranked, just miscalibrated)
    p_xgb_cal = isotonic_calibrate(y, p_xgb_raw)

    # NEWS2: Platt scaling on raw score
    p_news2_cal = logistic_calibrate(y, news2_raw)

    print(f"[dca] XGBoost calibrated range:  {p_xgb_cal.min():.4f} – {p_xgb_cal.max():.4f}")
    print(f"[dca] NEWS2   calibrated range:  {p_news2_cal.min():.4f} – {p_news2_cal.max():.4f}")

    # ── DCA in clinical probability space ──────────────────────────────────────
    # Clinically relevant range: 0.5% – 8% (low but realistic for ICU surveillance)
    t_clin = np.linspace(0.005, 0.08, 300)

    nb_xgb  = net_benefit_curve(y, p_xgb_cal,   t_clin)
    nb_news = net_benefit_curve(y, p_news2_cal,  t_clin)
    nb_all  = nb_treat_all(y, t_clin)
    nb_none = np.zeros(len(t_clin))

    # ── Reference: NEWS2 cut-off points ───────────────────────────────────────
    news2_cutoffs = {"NEWS2 ≥ 3": 3, "NEWS2 ≥ 5": 5, "NEWS2 ≥ 7": 7}
    news2_pts = {}
    for label, cut in news2_cutoffs.items():
        flag = (news2_raw >= cut)
        tp   = int(np.sum(flag & (y == 1)))
        fp   = int(np.sum(flag & (y == 0)))
        ppv  = tp / (tp + fp) if (tp + fp) else 0
        # Compute NB at the PPV threshold (where NB is maximised for a binary rule)
        nb_v = tp / n - fp / n * (ppv / (1 - ppv + 1e-15))
        news2_pts[label] = {
            "cut": cut, "threshold": ppv, "nb": nb_v,
            "sensitivity": tp / y.sum(), "ppv": ppv,
        }
        print(f"[dca] {label:10s}  sens={tp/y.sum():.3f}  PPV={ppv:.4f}  NB@PPV={nb_v:.5f}")

    # Also evaluate NB for NEWS2 ≥ 5 at each threshold on the curve for comparison
    flag_5 = (news2_raw >= 5).astype(bool)
    tp5 = int(np.sum(flag_5 & (y == 1)))
    fp5 = int(np.sum(flag_5 & (y == 0)))
    # NB for a binary rule at clinical threshold t
    nb_news5_curve = np.array([tp5 / n - fp5 / n * (t / (1 - t + 1e-15)) for t in t_clin])

    # ── Figure ─────────────────────────────────────────────────────────────────
    STYLE = dict(
        bg="#ffffff", border="#cccccc",
        text="#222222", muted="#666666",
        blue="#0072B2", orange="#D55E00",
        green="#009E73", gray="#888888",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2),
                             facecolor=STYLE["bg"])
    fig.subplots_adjust(wspace=0.36, left=0.07, right=0.97,
                        top=0.88, bottom=0.13)

    for ax in axes:
        ax.set_facecolor(STYLE["bg"])
        for sp in ax.spines.values():
            sp.set_edgecolor(STYLE["border"])
        ax.tick_params(colors=STYLE["text"], labelsize=9)

    t_pct = t_clin * 100  # x-axis in %

    # ── Panel A: full 0.5–8% range ─────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t_pct, nb_xgb,  color=STYLE["blue"],   lw=2.2,
            label="bssentinel (XGBoost)")
    ax.plot(t_pct, nb_news, color=STYLE["orange"],  lw=1.8, ls="--",
            label="NEWS2 (Platt-calibrated)")
    ax.plot(t_pct, nb_news5_curve, color=STYLE["orange"], lw=1.2, ls=":",
            label="NEWS2 ≥ 5 (clinical rule)")
    ax.plot(t_pct, nb_all,  color=STYLE["green"],   lw=1.4, ls=(0,(4,2)),
            label="Treat-all")
    ax.axhline(0, color=STYLE["gray"], lw=1.0, label="Treat-none")

    # Shade bssentinel advantage region
    adv_mask = nb_xgb > np.maximum(nb_news, 0)
    ax.fill_between(t_pct[adv_mask], 0, nb_xgb[adv_mask],
                    alpha=0.10, color=STYLE["blue"])

    # NEWS2 clinical cut-off scatter
    markers = {"NEWS2 ≥ 3": "^", "NEWS2 ≥ 5": "s", "NEWS2 ≥ 7": "D"}
    for lbl, d in news2_pts.items():
        if 0.005 <= d["threshold"] <= 0.08:
            ax.scatter(d["threshold"] * 100, d["nb"],
                       color=STYLE["orange"], marker=markers[lbl],
                       s=55, zorder=5, clip_on=False)
            ax.annotate(lbl, xy=(d["threshold"] * 100, d["nb"]),
                        xytext=(d["threshold"] * 100 + 0.2, d["nb"] + 0.0002),
                        fontsize=7.5, color=STYLE["orange"])

    ax.set_xlabel("Threshold probability (%)", fontsize=11, color=STYLE["text"])
    ax.set_ylabel("Net benefit (per patient)", fontsize=11, color=STYLE["text"])
    ax.set_title("A  Decision Curve Analysis",
                 fontsize=12, fontweight="bold", color=STYLE["text"], loc="left")
    ax.set_xlim(0.5, 8.0)
    ax.set_ylim(bottom=min(nb_all.min(), nb_news.min()) * 1.1)
    ax.legend(fontsize=8.5, framealpha=0.95, edgecolor=STYLE["border"],
              facecolor=STYLE["bg"], labelcolor=STYLE["text"])
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

    # ── Panel B: zoom 1–5% (highlight clinical sweet spot) ────────────────────
    ax2 = axes[1]
    mask_zoom = (t_clin >= 0.01) & (t_clin <= 0.05)
    tz = t_pct[mask_zoom]

    ax2.plot(tz, nb_xgb[mask_zoom],  color=STYLE["blue"],   lw=2.5,
             label="bssentinel")
    ax2.plot(tz, nb_news[mask_zoom], color=STYLE["orange"],  lw=2.0, ls="--",
             label="NEWS2 (calibrated)")
    ax2.plot(tz, nb_all[mask_zoom],  color=STYLE["green"],   lw=1.6, ls=(0,(4,2)),
             label="Treat-all")
    ax2.axhline(0, color=STYLE["gray"], lw=1.0, label="Treat-none")

    # Fill where XGBoost > NEWS2
    adv2 = nb_xgb[mask_zoom] > nb_news[mask_zoom]
    ax2.fill_between(tz[adv2],
                     nb_news[mask_zoom][adv2],
                     nb_xgb[mask_zoom][adv2],
                     alpha=0.18, color=STYLE["blue"],
                     label="bssentinel advantage")

    # Annotate NB gap at 2%
    t2_idx = np.argmin(np.abs(t_clin[mask_zoom] - 0.02))
    nb_x2  = nb_xgb[mask_zoom][t2_idx]
    nb_n2  = nb_news[mask_zoom][t2_idx]
    if nb_x2 > nb_n2:
        mid = (nb_x2 + nb_n2) / 2
        ax2.annotate(
            f"ΔNB = {nb_x2 - nb_n2:.4f}\n(per patient at 2%)",
            xy=(2.0, mid),
            xytext=(3.0, mid + abs(nb_x2 - nb_n2) * 0.8),
            fontsize=8, color=STYLE["blue"],
            arrowprops=dict(arrowstyle="->", color=STYLE["blue"], lw=1.1),
        )

    ax2.set_xlabel("Threshold probability (%)", fontsize=11, color=STYLE["text"])
    ax2.set_ylabel("Net benefit (per patient)", fontsize=11, color=STYLE["text"])
    ax2.set_title("B  Clinical zone: 1–5% threshold (zoomed)",
                  fontsize=12, fontweight="bold", color=STYLE["text"], loc="left")
    ax2.set_xlim(1, 5)
    ax2.legend(fontsize=8.5, framealpha=0.95, edgecolor=STYLE["border"],
               facecolor=STYLE["bg"], labelcolor=STYLE["text"])
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

    fig.suptitle(
        "Decision Curve Analysis — bssentinel vs NEWS2 "
        "(MIMIC-IV, n\u202f=\u202f1,334,773 windows, prevalence 1.96%)",
        fontsize=12, fontweight="bold", color=STYLE["text"], y=0.97,
    )

    out_path = out_dir / "dca_curve.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"\n[dca] Figure saved → {out_path}")

    # ── Summary stats for paper text ───────────────────────────────────────────
    probe_t = [0.01, 0.02, 0.03, 0.05]
    summary = {
        "nb_at_thresholds": {},
        "xgb_dominates_news2_1_5pct": bool(
            np.all(nb_xgb[mask_zoom] >= nb_news[mask_zoom])
        ),
        "news2_cutpoints": {k: {"threshold_pct": round(v["threshold"]*100,3),
                                "sensitivity": round(v["sensitivity"],3),
                                "nb": round(v["nb"],6)}
                            for k, v in news2_pts.items()},
    }
    print("\n[dca] Net benefit summary (calibrated probabilities as thresholds):")
    print(f"{'Threshold':>10}  {'bssentinel NB':>16}  {'NEWS2 NB':>12}  {'Treat-all NB':>14}  {'XGBoost > NEWS2':>16}")
    for t in probe_t:
        idx = np.argmin(np.abs(t_clin - t))
        nx  = float(nb_xgb[idx])
        nn  = float(nb_news[idx])
        na  = float(nb_all[idx])
        summary["nb_at_thresholds"][f"{int(t*100)}pct"] = {
            "bssentinel": round(nx, 6),
            "news2":       round(nn, 6),
            "treat_all":   round(na, 6),
            "xgb_wins":   nx > nn,
        }
        flag = "✓" if nx > nn else "✗"
        print(f"{t*100:>9.0f}%  {nx:>16.5f}  {nn:>12.5f}  {na:>14.5f}  {flag:>16}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", default="artifacts/oof_predictions.parquet")
    parser.add_argument("--out", default="artifacts/figures")
    args = parser.parse_args()
    oof_path = Path(args.oof)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_dca(oof_path, out_dir)
    (out_dir.parent / "dca_summary.json").write_text(json.dumps(summary, indent=2))
    print("[dca] Summary saved → artifacts/dca_summary.json")

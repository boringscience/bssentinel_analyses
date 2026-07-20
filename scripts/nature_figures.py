"""
nature_figures.py — Modern Nature-style publication figures for bssentinel revision.

Nature house style (2024+):
  - Panel labels: lowercase bold (a, b, c), 10pt, positioned outside axes
  - Font: Helvetica/Arial, 6-7pt body, 7-8pt axis labels
  - Figure widths: single column 89mm (3.5in), 1.5 col 120mm (4.7in), double 183mm (7.2in)
  - Resolution: 300 dpi raster, 600 dpi line art
  - Colours: muted, accessible, contemporary palette
  - No gridlines (or extremely faint)
  - No box around legends
  - Tick marks outside, thin (0.4pt)
  - Spines: bottom + left only, 0.5pt
  - White background, high data-ink ratio
  - Subtle fills for visual hierarchy
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupKFold

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
ART = BASE / "ml" / "artifacts"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Modern Nature palette ─────────────────────────────────────────────────────
# Inspired by Nature Medicine / Nature Methods 2024 figure conventions
C = {
    "ink":      "#2D2D2D",   # primary text/lines
    "blue":     "#2166AC",   # primary accent — trustworthy, cool
    "red":      "#B2182B",   # alerts, events
    "green":    "#1B7837",   # positive outcomes
    "orange":   "#D6604D",   # warm secondary
    "purple":   "#7B3294",   # tertiary
    "teal":     "#35978F",   # quaternary
    "gray":     "#737373",   # muted elements
    "lgray":    "#D9D9D9",   # very light structural
    "vlgray":   "#F0F0F0",   # background fills
    "bg":       "#FFFFFF",
    # Soft fills for panels
    "blue_bg":  "#DEEBF7",
    "red_bg":   "#FEE0D2",
    "green_bg": "#D9F0D3",
    "orange_bg":"#FDD0A2",
    "purple_bg":"#DADAEB",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.4,
    "legend.labelspacing": 0.3,
    "legend.columnspacing": 1.0,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "xtick.color": C["ink"],
    "ytick.color": C["ink"],
    "axes.edgecolor": C["ink"],
    "axes.labelcolor": C["ink"],
    "text.color": C["ink"],
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "mathtext.default": "regular",
})


def _spine(ax):
    """Nature-style: left + bottom only, thin."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_linewidth(0.5)
        ax.spines[s].set_color(C["ink"])


def _panel(ax, letter, x=-0.12, y=1.08):
    """Nature lowercase bold panel label."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left",
            color=C["ink"], family="sans-serif")


# ── Load data ─────────────────────────────────────────────────────────────────
with open(ART / "metrics.json") as f:
    metrics = json.load(f)
oof = pd.read_parquet(ART / "oof_predictions.parquet")
fi = pd.read_csv(ART / "feature_importance.csv")

y = oof["label"].values
xgb_prob = oof["xgb_proba"].values
lr_prob = oof["lr_proba"].values
news2 = oof["news2_total"].values
groups = oof["hadm_id"].values


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Study design
# ══════════════════════════════════════════════════════════════════════════════

def fig1():
    fig = plt.figure(figsize=(7.2, 6.8))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[2.0, 1.0],
                           hspace=0.18, wspace=0.30, left=0.03, right=0.97, top=0.96, bottom=0.03)
    ax_tl = fig.add_subplot(gs[0, :])
    ax_in = fig.add_subplot(gs[1, 0])
    ax_ou = fig.add_subplot(gs[1, 1])

    # ── a: Timeline with data streams ────────────────────────────────────
    for sp in ax_tl.spines.values():
        sp.set_visible(False)
    ax_tl.set_xlim(-0.8, 15.0)
    ax_tl.set_ylim(-6.0, 2.8)
    ax_tl.set_xticks([])
    ax_tl.set_yticks([])
    _panel(ax_tl, "a", x=-0.005, y=1.02)
    TL = 1.2  # timeline y position

    # ── Shaded regions for lookback & prediction windows ──
    # Lookback window (blue shading) from +15h to +21h → x=5.5 to 7.5
    ax_tl.axvspan(5.5, 7.8, ymin=0.18, ymax=0.78, alpha=0.08, color=C["blue"], zorder=0)
    # Prediction horizon (red/pink shading) from +21h to event → x=7.8 to 12.8
    ax_tl.axvspan(7.8, 12.8, ymin=0.18, ymax=0.78, alpha=0.06, color=C["red"], zorder=0)

    # Admission timeline bar
    ax_tl.barh(TL, 13.2, left=0.4, height=0.12, color=C["vlgray"],
               edgecolor=C["lgray"], lw=0.4, zorder=1)
    ax_tl.plot([0.5, 13.5], [TL, TL], "-", color=C["lgray"], lw=1.5, zorder=2, solid_capstyle="round")

    # Admission marker
    ax_tl.plot(0.5, TL, "D", color=C["green"], ms=5.5, zorder=5, markeredgewidth=0)
    ax_tl.text(0.5, TL + 0.35, "Admission", ha="center", va="bottom",
               fontsize=6.5, color=C["green"], fontweight="bold")

    # Discharge marker
    ax_tl.plot(13.5, TL, "s", color=C["gray"], ms=4, zorder=5, markeredgewidth=0)
    ax_tl.text(13.5, TL + 0.32, "Discharge", ha="center", va="bottom",
               fontsize=5.5, color=C["gray"])

    # Check/Alert timepoints
    checks = [1.5, 3.5, 5.5, 7.5]
    alerts = [9.5, 11.5]
    for x in checks + alerts:
        alert = x in alerts
        fc = C["red"] if alert else C["blue"]
        ax_tl.plot([x, x], [TL - 0.05, TL - 0.22], "-", color=fc, lw=0.4, zorder=3, alpha=0.6)
        ax_tl.plot(x, TL, "o", color=fc, ms=6.5, zorder=6,
                   markeredgewidth=0.8, markeredgecolor="white")
        lbl = "Alert" if alert else "Check"
        ax_tl.text(x, TL - 0.38, lbl, ha="center", va="top", fontsize=5.5,
                   color=C["red"] if alert else C["gray"],
                   fontweight="bold" if alert else "normal")
        ax_tl.text(x, TL + 0.32, f"+{int((x - 0.5) * 3)}h", ha="center",
                   va="bottom", fontsize=5, color=C["gray"])

    # Event star
    ax_tl.plot(12.6, TL, "*", color=C["red"], ms=11, zorder=7, markeredgewidth=0)
    ax_tl.text(12.6, TL - 0.38, "Event", ha="center", va="top",
               fontsize=5.5, color=C["red"], fontweight="bold")
    ax_tl.text(13.8, TL - 0.10, "Observed Event\n(e.g., Sepsis diagnosis)",
               ha="left", va="center", fontsize=4.5, color=C["gray"], style="italic")

    # Early warning arrow
    arr_y = TL + 1.0
    ax_tl.annotate("", xy=(12.6, arr_y), xytext=(9.5, arr_y),
                   arrowprops=dict(arrowstyle="<->", color=C["orange"],
                                   lw=1.0, mutation_scale=9))
    ax_tl.text(11.05, arr_y + 0.15, "Early warning period\n(up to 9 h before Event)",
               ha="center", va="bottom", fontsize=5.5, color=C["orange"], fontweight="bold",
               bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # ── Data stream tracks below timeline ──
    track_y0 = TL - 1.3  # starting y for first track
    track_h = 0.7        # height per track
    track_gap = 0.15
    track_labels = [
        ("Vital Signs", C["blue"]),
        ("Laboratory", C["green"]),
        ("Medications", C["orange"]),
        ("Temporal", C["purple"]),
        ("Missingness", C["teal"]),
    ]

    np.random.seed(42)
    for i, (label, color) in enumerate(track_labels):
        ty = track_y0 - i * (track_h + track_gap)
        # Track label with colour marker
        ax_tl.plot(-0.65, ty, "s", color=color, ms=4, zorder=2, markeredgewidth=0)
        ax_tl.text(-0.45, ty, label, ha="left", va="center", fontsize=5.5,
                   fontweight="bold", color=C["gray"])
        # Thin separator line
        ax_tl.plot([0.5, 13.5], [ty - track_h * 0.45, ty - track_h * 0.45],
                   "-", color=C["vlgray"], lw=0.3, zorder=0)

        if i == 0:  # Vital signs — wavy lines
            xs = np.linspace(0.5, 13.5, 200)
            # HR-like trace
            hr = np.sin(xs * 2.5) * 0.08 + np.random.normal(0, 0.015, len(xs))
            ax_tl.plot(xs, ty + hr + 0.08, "-", color=C["blue"], lw=0.5, alpha=0.7, zorder=2)
            # BP-like trace
            bp = np.cos(xs * 1.8) * 0.06 + np.random.normal(0, 0.012, len(xs))
            ax_tl.plot(xs, ty + bp - 0.08, "-", color=C["red"], lw=0.5, alpha=0.5, zorder=2)
            # Dots for measurement points
            for cx in checks:
                ax_tl.plot(cx, ty, "o", color=C["lgray"], ms=2, zorder=3, alpha=0.6)

        elif i == 1:  # Laboratory — scattered coloured dots
            lab_colors = [C["blue"], C["red"], C["green"], C["green"],
                         "#D4A017", "#D4A017", C["orange"], C["purple"]]
            lab_xs = [1.0, 2.0, 3.0, 3.8, 4.5, 5.2, 6.5, 7.0]
            for j, lx in enumerate(lab_xs):
                ax_tl.plot(lx, ty + np.random.uniform(-0.08, 0.08),
                          "o", color=lab_colors[j % len(lab_colors)],
                          ms=3.5, zorder=3, alpha=0.8, markeredgewidth=0)
            # One in prediction zone
            ax_tl.plot(10.0, ty, "o", color=C["red"], ms=3.5, zorder=3, alpha=0.8,
                      markeredgewidth=0)

        elif i == 2:  # Medications — horizontal bars/blocks
            med_ranges = [(2.5, 5.0), (4.0, 6.5), (7.2, 9.5)]
            med_colors_list = [C["orange"], C["green"], C["orange"]]
            for j, ((x0, x1), mc) in enumerate(zip(med_ranges, med_colors_list)):
                ax_tl.add_patch(plt.Rectangle((x0, ty - 0.12), x1 - x0, 0.24,
                               facecolor=mc, alpha=0.4, edgecolor=mc,
                               lw=0.5, zorder=2))
                # Small triangle marker at end
                ax_tl.plot(x1, ty + 0.12, "v", color=mc, ms=2.5, zorder=3)

        elif i == 3:  # Temporal — smooth gradient fill using imshow
            from matplotlib.colors import LinearSegmentedColormap
            import matplotlib.colors as mcolors
            # Create gradient image
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            cmap_purple = LinearSegmentedColormap.from_list("purp",
                          ["white", C["purple"]], N=256)
            ax_tl.imshow(gradient, aspect="auto", cmap=cmap_purple,
                        extent=[0.5, 13.5, ty - 0.2, ty + 0.2],
                        alpha=0.6, zorder=1)

        elif i == 4:  # Missingness — coloured blocks with dashed border
            block_xs = np.linspace(0.5, 13.0, 18)
            miss_colors = [C["teal"], C["green"], C["orange"], C["red"],
                          C["blue"], C["purple"]]
            for j, bx in enumerate(block_xs):
                mc = miss_colors[j % len(miss_colors)]
                ax_tl.add_patch(plt.Rectangle((bx, ty - 0.15), 0.65, 0.30,
                               facecolor=mc, alpha=0.5, edgecolor="none", zorder=2))
            # Dashed border lines
            ax_tl.plot([0.5, 13.5], [ty + 0.15, ty + 0.15], "--",
                      color=C["gray"], lw=0.4, zorder=3, alpha=0.5)
            ax_tl.plot([0.5, 13.5], [ty - 0.15, ty - 0.15], "--",
                      color=C["gray"], lw=0.4, zorder=3, alpha=0.5)

    # ── Lookback / horizon arrows below tracks ──
    arr_base = track_y0 - 5 * (track_h + track_gap) - 0.2
    # 6h lookback arrow
    ax_tl.annotate("", xy=(5.5, arr_base), xytext=(7.8, arr_base),
                   arrowprops=dict(arrowstyle="<->", color=C["blue"],
                                   lw=0.7, mutation_scale=7))
    ax_tl.text(6.65, arr_base + 0.18, "6 h lookback", ha="center", va="bottom",
               fontsize=5.5, color=C["blue"], fontweight="semibold")
    ax_tl.text(6.65, arr_base - 0.18, "Lookback window\nfor feature extraction",
               ha="center", va="top", fontsize=4.5, color=C["blue"], alpha=0.7)

    # 24h prediction horizon arrow
    ax_tl.annotate("", xy=(7.8, arr_base), xytext=(12.8, arr_base),
                   arrowprops=dict(arrowstyle="<->", color=C["red"],
                                   lw=0.7, mutation_scale=7))
    ax_tl.text(10.3, arr_base + 0.18, "24 h prediction horizon", ha="center",
               va="bottom", fontsize=5.5, color=C["red"], fontweight="semibold")
    ax_tl.text(10.3, arr_base - 0.18, "Model prediction\nwindow (horizon)",
               ha="center", va="top", fontsize=4.5, color=C["red"], alpha=0.7)

    # ── b: Model Inputs (table style) ────────────────────────────────────
    for sp in ax_in.spines.values():
        sp.set_visible(False)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.set_xlim(0, 1)
    ax_in.set_ylim(0, 7.0)
    _panel(ax_in, "b", x=-0.04, y=1.06)

    ax_in.text(0.04, 6.8, "Model Inputs", fontsize=7.5, fontweight="bold",
               va="top", color=C["ink"])
    ax_in.text(0.30, 6.8, "(83 total features)", fontsize=6.5, va="top", color=C["gray"])
    ax_in.plot([0.04, 0.96], [6.4, 6.4], "-", color=C["lgray"], lw=0.5)

    input_data = [
        ("s", C["blue"],   "Vital Signs",  "HR, SBP, DBP, SpO$_2$, RR, Temp, GCS, +19 derived stats  (28)"),
        ("s", C["green"],  "Laboratory",   "Lactate, Creatinine, WBC, INR, +8 core labs  (12)"),
        ("s", C["orange"], "Medications",  "Vasopressors, Antibiotics, Anticoagulants  (3)"),
        ("s", C["purple"], "Temporal",     "Hours since admission, Admission source, Admission type  (12)"),
        ("s", C["teal"],   "Missingness",  "Binary missing flags for 20 lab features  (20)"),
    ]
    for i, (marker, col, cat, det) in enumerate(input_data):
        ry = 5.9 - i * 1.10
        # Colour square marker
        ax_in.plot(0.04, ry, marker, color=col, ms=6, zorder=2,
                   markeredgewidth=0)
        ax_in.text(0.08, ry + 0.05, cat, va="bottom", fontsize=6.5,
                   fontweight="bold", color=C["ink"])
        # Separator
        ax_in.plot([0.08, 0.96], [ry - 0.25, ry - 0.25], "-", color=C["vlgray"], lw=0.3)
        ax_in.text(0.08, ry - 0.10, det, va="top", fontsize=5, color=C["gray"])

    # ── c: Performance (table style with CIs) ────────────────────────────
    for sp in ax_ou.spines.values():
        sp.set_visible(False)
    ax_ou.set_xticks([])
    ax_ou.set_yticks([])
    ax_ou.set_xlim(0, 1)
    ax_ou.set_ylim(0, 7.0)
    _panel(ax_ou, "c", x=-0.04, y=1.06)

    ax_ou.text(0.04, 6.8, "Performance", fontsize=7.5, fontweight="bold",
               va="top", color=C["ink"])
    ax_ou.text(0.30, 6.8, "(Threshold = 0.31)", fontsize=6.5, va="top", color=C["gray"])
    ax_ou.plot([0.04, 0.96], [6.4, 6.4], "-", color=C["lgray"], lw=0.5)

    perf_data = [
        ("AUROC",               "0.758", "(0.754\u20130.763)"),
        ("Sensitivity",         "90.0%", "(89.1\u201390.9)"),
        ("FPR (1\u2212spec.)",  "62.1%", "(61.2\u201363.0)"),
        ("NPV",                 "99.5%", "(99.4\u201399.6)"),
        ("Gain vs mNEWS2",      "+0.190", "($P$ < 10\u207b\u00b9\u2075)"),
    ]
    for i, (lab, val, ci) in enumerate(perf_data):
        ry = 5.9 - i * 1.10
        ax_ou.text(0.04, ry, lab, va="center", fontsize=6.5, color=C["ink"])
        ax_ou.text(0.52, ry, val, va="center", fontsize=6.5, color=C["blue"],
                   fontweight="bold")
        ax_ou.text(0.68, ry, ci, va="center", fontsize=5.5, color=C["gray"])
        # Separator
        ax_ou.plot([0.04, 0.96], [ry - 0.30, ry - 0.30], "-", color=C["vlgray"], lw=0.3)

    fig.savefig(OUT / "fig1_study_design.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "fig1_study_design.pdf", facecolor="white")
    plt.close(fig)
    print("  fig1")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Discrimination
# ══════════════════════════════════════════════════════════════════════════════

def fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.8),
                                    gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(wspace=0.38, left=0.08, right=0.97, top=0.92, bottom=0.13)
    fig.patch.set_facecolor("white")

    # ── a: ROC ───────────────────────────────────────────────────────────
    _spine(ax1)
    _panel(ax1, "a")

    fpr_x, tpr_x, _ = roc_curve(y, xgb_prob)
    fpr_l, tpr_l, _ = roc_curve(y, lr_prob)
    v = ~np.isnan(news2)
    fpr_n, tpr_n, _ = roc_curve(y[v], news2[v])

    # Subtle AUC shading for primary model
    ax1.fill_between(fpr_x, tpr_x, alpha=0.08, color=C["blue"], zorder=1)

    # ROC curves with distinct styles
    ax1.plot(fpr_x, tpr_x, color=C["blue"], lw=1.4, label="bssentinel (0.758)",
             zorder=4, solid_capstyle="round")
    ax1.plot(fpr_l, tpr_l, color=C["purple"], lw=1.0, ls="--",
             label="Logistic reg. (0.708)", zorder=3)
    ax1.plot(fpr_n, tpr_n, color=C["orange"], lw=1.0, ls=":",
             label="mNEWS2 (0.568)", zorder=3)
    ax1.plot([0, 1], [0, 1], "-", color=C["lgray"], lw=0.4, zorder=1)

    # Operating point
    hs = metrics["high_sensitivity_point"]
    op_x, op_y = 1 - hs["specificity"], hs["sensitivity"]
    ax1.plot(op_x, op_y, "o", color=C["red"], ms=5, zorder=7,
             markeredgewidth=0.8, markeredgecolor="white")
    ax1.annotate("90% sensitivity\n(threshold = 0.31)",
                xy=(op_x, op_y), xytext=(0.22, 0.72),
                fontsize=5.5, color=C["red"], ha="center",
                arrowprops=dict(arrowstyle="-", color=C["red"], lw=0.5,
                               connectionstyle="arc3,rad=-0.2"))

    ax1.set_xlabel("1 \u2212 Specificity")
    ax1.set_ylabel("Sensitivity")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.04)
    ax1.set_aspect("equal", adjustable="box")
    ax1.legend(loc="lower right", fontsize=5.5, borderpad=0.4)

    # ── b: AUROC bars ────────────────────────────────────────────────────
    _spine(ax2)
    _panel(ax2, "b")

    models = ["MEWS", "Shock Index", "qSOFA", "mNEWS2", "Logistic reg.", "bssentinel"]
    aucs = [0.533, 0.539, 0.549, 0.568, 0.708, 0.758]
    cis_lo = [0.530, 0.535, 0.546, 0.565, 0.703, 0.754]
    cis_hi = [0.537, 0.542, 0.552, 0.572, 0.713, 0.763]

    # Gradient from light to saturated for visual hierarchy
    bar_colors = [C["lgray"], C["lgray"], C["lgray"],
                  C["orange"], C["purple"], C["blue"]]
    bar_alphas = [0.6, 0.6, 0.6, 0.85, 0.85, 1.0]

    yp = np.arange(len(models))
    for i in range(len(models)):
        ax2.barh(yp[i], aucs[i], color=bar_colors[i], height=0.52,
                 edgecolor="none", zorder=2, alpha=bar_alphas[i])
        ax2.errorbar(aucs[i], yp[i],
                     xerr=[[aucs[i] - cis_lo[i]], [cis_hi[i] - aucs[i]]],
                     fmt="none", color=C["ink"], capsize=2, lw=0.5, zorder=3)
        ax2.text(cis_hi[i] + 0.012, yp[i], f"{aucs[i]:.3f}",
                 va="center", fontsize=6, color=C["ink"], fontweight="semibold")

    ax2.axvline(0.5, color=C["lgray"], lw=0.5, ls="--", zorder=1)

    # Gain bracket — positioned above bars with enough clearance
    ax2.annotate("", xy=(0.758, 5.7), xytext=(0.568, 5.7),
                 arrowprops=dict(arrowstyle="<->", color=C["green"],
                                 lw=0.9, mutation_scale=8))
    ax2.text(0.663, 6.2, "\u0394 = 0.190", ha="center", fontsize=6,
             color=C["green"], fontweight="bold")
    ax2.text(0.663, 5.35, "p < 10\u207b\u00b9\u2075", ha="center", fontsize=5,
             color=C["green"])

    ax2.set_yticks(yp)
    ax2.set_yticklabels(models, fontsize=6.5)
    ax2.set_xlabel("AUROC")
    ax2.set_xlim(0.45, 0.84)
    ax2.set_ylim(-0.6, 6.8)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(left=False)

    fig.savefig(OUT / "fig2_discrimination.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "fig2_discrimination.pdf", facecolor="white")
    plt.close(fig)
    print("  fig2")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Clinical utility
# ══════════════════════════════════════════════════════════════════════════════

def fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4),
                                    gridspec_kw={"width_ratios": [1, 1.1]})
    fig.subplots_adjust(wspace=0.30, left=0.06, right=0.97, top=0.88, bottom=0.15)
    fig.patch.set_facecolor("white")

    # ── a: Confusion per 1000 ────────────────────────────────────────────
    ax1.axis("off")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-0.5, 9.0)
    _panel(ax1, "a", x=-0.02, y=1.06)

    tp, fn, fp, tn = 18, 2, 608, 372

    def cell(x, yy, w, h, fc, ec, lw=0.5):
        ax1.add_patch(FancyBboxPatch((x, yy), w, h, boxstyle="round,pad=0.12",
                      facecolor=fc, edgecolor=ec, lw=lw, zorder=2))

    # Header
    ax1.text(5.0, 8.6, "Per 1,000 windows", ha="center",
             fontsize=7.5, fontweight="bold", color=C["ink"])
    ax1.text(5.0, 8.1, "threshold = 0.31", ha="center",
             fontsize=6, color=C["gray"])

    # Column / row headers
    ax1.text(3.5, 7.5, "Alert", ha="center", fontsize=7, fontweight="bold",
             color=C["blue"])
    ax1.text(7.0, 7.5, "No alert", ha="center", fontsize=7, fontweight="bold",
             color=C["gray"])
    ax1.text(0.7, 5.5, "Event", ha="center", fontsize=6.5, fontweight="semibold",
             color=C["red"])
    ax1.text(0.7, 5.0, f"(n = 20)", ha="center", fontsize=5.5, color=C["gray"])
    ax1.text(0.7, 2.5, "Stable", ha="center", fontsize=6.5, fontweight="semibold",
             color=C["green"])
    ax1.text(0.7, 2.0, f"(n = 980)", ha="center", fontsize=5.5, color=C["gray"])

    # TP — blue
    cell(2, 4.2, 3, 2.6, C["blue_bg"], C["blue"], 0.6)
    ax1.text(3.5, 5.9, str(tp), ha="center", fontsize=20, fontweight="bold",
             color=C["blue"])
    ax1.text(3.5, 4.7, "True positive", ha="center", fontsize=6, color=C["ink"])

    # FN — warm
    cell(5.5, 4.2, 3, 2.6, C["red_bg"], C["orange"], 0.6)
    ax1.text(7.0, 5.9, str(fn), ha="center", fontsize=20, fontweight="bold",
             color=C["orange"])
    ax1.text(7.0, 4.7, "False negative", ha="center", fontsize=6, color=C["ink"])

    # FP — warm
    cell(2, 1.2, 3, 2.6, C["orange_bg"], C["orange"], 0.6)
    ax1.text(3.5, 2.9, str(fp), ha="center", fontsize=20, fontweight="bold",
             color=C["orange"])
    ax1.text(3.5, 1.7, "False positive", ha="center", fontsize=6, color=C["ink"])

    # TN — green
    cell(5.5, 1.2, 3, 2.6, C["green_bg"], C["green"], 0.6)
    ax1.text(7.0, 2.9, str(tn), ha="center", fontsize=20, fontweight="bold",
             color=C["green"])
    ax1.text(7.0, 1.7, "True negative", ha="center", fontsize=6, color=C["ink"])

    # Summary metrics (hardcoded to match manuscript exact values)
    ax1.text(3.5, 0.65, "PPV = 2.8%", ha="center", fontsize=6.5,
             color=C["orange"], fontweight="bold")
    ax1.text(7.0, 0.65, "NPV = 99.5%", ha="center", fontsize=6.5,
             color=C["green"], fontweight="bold")

    # ── b: Threshold curves ──────────────────────────────────────────────
    _spine(ax2)
    _panel(ax2, "b", x=-0.14)

    td = metrics["threshold_analysis"]
    ths = np.array([r["threshold"] for r in td])

    sens = [r["sensitivity"] * 100 for r in td]
    spec = [r["specificity"] * 100 for r in td]
    ppvs = [r["ppv"] * 100 for r in td]

    ax2.plot(ths, sens, "o-", color=C["blue"], lw=1.2, ms=3.5,
             markeredgecolor="white", markeredgewidth=0.5, label="Sensitivity", zorder=4)
    ax2.plot(ths, spec, "s-", color=C["green"], lw=1.2, ms=3.5,
             markeredgecolor="white", markeredgewidth=0.5, label="Specificity", zorder=4)
    ax2.plot(ths, ppvs, "^-", color=C["red"], lw=1.2, ms=3.5,
             markeredgecolor="white", markeredgewidth=0.5, label="PPV", zorder=4)

    # Operating threshold
    ax2.axvline(0.31, color=C["orange"], lw=0.8, ls="--", zorder=1, alpha=0.8)
    ax2.text(0.315, 52, "0.31", fontsize=6, color=C["orange"], fontweight="semibold",
             ha="left", va="center",
             bbox=dict(facecolor="white", edgecolor="none", pad=1))

    ax2.set_xlabel("Decision threshold")
    ax2.set_ylabel("(%)")
    ax2.set_xlim(0.08, 0.53)
    ax2.set_ylim(-2, 108)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.legend(loc="center right", fontsize=6, handletextpad=0.5)

    fig.savefig(OUT / "fig3_clinical_utility.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "fig3_clinical_utility.pdf", facecolor="white")
    plt.close(fig)
    print("  fig3")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Feature importance (lollipop chart)
# ══════════════════════════════════════════════════════════════════════════════

def fig4():
    LABELS = {
        "lactate_missing": "Lactate not measured", "spo2_worst": "Min. SpO$_2$",
        "gcs_verbal_worst": "Worst verbal GCS", "gcs_motor_worst": "Worst motor GCS",
        "news2_gcs": "mNEWS2 consciousness", "gcs_motor": "GCS motor (recent)",
        "respiratory_rate_worst": "Max. respiratory rate", "admit_offset_hours": "Time since admission",
        "vasopressors": "Vasopressor active", "news2_total": "mNEWS2 total",
        "adm_type_OBSERVATION ADMIT": "Observation admission",
        "adm_type_SURGICAL SAME DAY ADMISSION": "Same-day surgical",
        "antibiotics": "Antibiotic active", "anticoagulants": "Anticoagulant active",
        "gcs_total": "GCS total", "news2_hr": "mNEWS2 heart rate", "lactate": "Lactate value",
    }
    CAT_COL = {
        "lactate_missing": C["teal"], "spo2_worst": C["blue"],
        "gcs_verbal_worst": C["purple"], "gcs_motor_worst": C["purple"],
        "news2_gcs": C["purple"], "respiratory_rate_worst": C["blue"],
        "admit_offset_hours": C["gray"], "vasopressors": C["orange"],
        "news2_total": C["gray"], "adm_type_OBSERVATION ADMIT": C["gray"],
        "antibiotics": C["orange"], "anticoagulants": C["orange"],
        "gcs_total": C["purple"], "news2_hr": C["gray"], "lactate": C["green"],
    }
    CAT_NAME = {
        "lactate_missing": "Missingness", "spo2_worst": "Respiratory",
        "gcs_verbal_worst": "Neurology", "gcs_motor_worst": "Neurology",
        "news2_gcs": "Neurology", "respiratory_rate_worst": "Respiratory",
        "admit_offset_hours": "Context", "vasopressors": "Medication",
        "news2_total": "Context", "adm_type_OBSERVATION ADMIT": "Context",
        "antibiotics": "Medication", "anticoagulants": "Medication",
        "gcs_total": "Neurology", "news2_hr": "Context", "lactate": "Laboratory",
    }

    top = fi[fi["importance"] > 0].nlargest(15, "importance").reset_index(drop=True)
    top["label"] = top["feature"].map(lambda f: LABELS.get(f, f))
    top["color"] = top["feature"].map(lambda f: CAT_COL.get(f, C["gray"]))
    top["cat"] = top["feature"].map(lambda f: CAT_NAME.get(f, "Other"))
    top["pct"] = top["importance"] * 100
    top = top.iloc[::-1].reset_index(drop=True)
    n = 15

    fig, ax = plt.subplots(figsize=(4.7, 4.5))
    fig.subplots_adjust(left=0.38, right=0.92, top=0.92, bottom=0.10)
    fig.patch.set_facecolor("white")
    _spine(ax)
    _panel(ax, "a", x=-0.52, y=1.04)

    yp = np.arange(n)

    # Lollipop stems
    for i in range(n):
        ax.plot([0, top["pct"].iloc[i]], [yp[i], yp[i]],
                "-", color=top["color"].iloc[i], lw=0.8, alpha=0.5, zorder=2)

    # Lollipop dots
    ax.scatter(top["pct"], yp, c=top["color"], s=40, zorder=5,
               edgecolors="white", linewidths=0.6)

    # Value labels — offset from dots to avoid overlap
    for i in range(n):
        ax.text(top["pct"].iloc[i] + 0.35, yp[i],
                f"{top['pct'].iloc[i]:.1f}%", va="center", fontsize=5.5,
                color=C["ink"])

    # Subtle horizontal guides
    for i in range(n):
        ax.axhline(yp[i], color=C["vlgray"], lw=0.3, zorder=1)

    ax.set_yticks(yp)
    ax.set_yticklabels(top["label"], fontsize=6)
    ax.set_xlabel("Gain importance (%)")
    ax.set_xlim(0, 9.0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    # Legend
    seen = {}
    for _, r in top.iterrows():
        if r["cat"] not in seen:
            seen[r["cat"]] = r["color"]
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=5, label=cat, markeredgewidth=0)
               for cat, c in seen.items()]
    ax.legend(handles=handles, fontsize=5.5, loc="lower right", title="Domain",
              title_fontsize=6, handletextpad=0.3, borderpad=0.4)

    # Ablation note — positioned to avoid overlapping lollipop elements
    ax.annotate("Domain ablation:\nmissingness \u0394AUROC = 0.000\n(redundant with other domains)",
               xy=(top.iloc[-1]["pct"] + 0.1, n - 1), xytext=(5.5, n - 6),
               fontsize=5.5, color=C["teal"], va="center",
               arrowprops=dict(arrowstyle="-", color=C["teal"], lw=0.5,
                              connectionstyle="arc3,rad=-0.3"),
               bbox=dict(facecolor="#EDF7F6", edgecolor=C["teal"],
                         boxstyle="round,pad=0.3", lw=0.4))

    fig.savefig(OUT / "fig4_feature_importance.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "fig4_feature_importance.pdf", facecolor="white")
    plt.close(fig)
    print("  fig4")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — DCA
# ══════════════════════════════════════════════════════════════════════════════

def fig5():
    ir = IsotonicRegression(out_of_bounds="clip")
    p_xgb = ir.fit_transform(xgb_prob, y.astype(float))
    sc = StandardScaler()
    X_n = sc.fit_transform(news2.reshape(-1, 1))
    lr_ = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr_.fit(X_n, y)
    p_news = lr_.predict_proba(X_n)[:, 1]
    t = np.linspace(0.005, 0.08, 300)
    tp_ = t * 100
    n_ = len(y)

    def nb(p):
        return np.array([
            np.sum((p >= th) & (y == 1)) / n_ -
            np.sum((p >= th) & (y == 0)) / n_ * (th / (1 - th + 1e-15))
            for th in t
        ])

    nb_x = nb(p_xgb)
    nb_n = nb(p_news)
    prev = y.mean()
    nb_a = prev - (1 - prev) * t / (1 - t + 1e-15)
    f5 = news2 >= 5
    t5 = np.sum(f5 & (y == 1))
    fp5 = np.sum(f5 & (y == 0))
    nb5 = np.array([t5 / n_ - fp5 / n_ * (th / (1 - th + 1e-15)) for th in t])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(wspace=0.32, left=0.08, right=0.97, top=0.86, bottom=0.17)
    fig.patch.set_facecolor("white")

    for ax in [ax1, ax2]:
        _spine(ax)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}%"))

    _panel(ax1, "a")
    _panel(ax2, "b")

    # Panel a: full range
    ax1.plot(tp_, nb_x, color=C["blue"], lw=1.4, label="bssentinel",
             solid_capstyle="round", zorder=4)
    ax1.plot(tp_, nb_n, color=C["orange"], lw=1.0, ls="--",
             label="mNEWS2 (calibrated)", zorder=3)
    ax1.plot(tp_, nb5, color=C["orange"], lw=0.7, ls=":",
             label="mNEWS2 \u2265 5", zorder=3)
    ax1.plot(tp_, nb_a, color=C["green"], lw=0.7, ls=(0, (3, 2)),
             label="Treat all", zorder=2)
    ax1.axhline(0, color=C["lgray"], lw=0.4)

    # Advantage shading
    adv = nb_x > np.maximum(nb_n, 0)
    ax1.fill_between(tp_[adv], 0, nb_x[adv], alpha=0.10, color=C["blue"], zorder=1)

    ax1.set_xlabel("Threshold probability")
    ax1.set_ylabel("Net benefit")
    ax1.set_xlim(0.5, 8)
    ax1.set_ylim(min(nb_a.min(), nb_n.min()) * 1.05, None)
    ax1.legend(loc="upper right", fontsize=5.5, bbox_to_anchor=(1.0, 1.02))

    # Panel b: zoomed
    m = (t >= 0.01) & (t <= 0.05)
    tz = tp_[m]
    ax2.plot(tz, nb_x[m], color=C["blue"], lw=1.6, label="bssentinel",
             solid_capstyle="round", zorder=4)
    ax2.plot(tz, nb_n[m], color=C["orange"], lw=1.0, ls="--",
             label="mNEWS2", zorder=3)
    ax2.plot(tz, nb_a[m], color=C["green"], lw=0.7, ls=(0, (3, 2)),
             label="Treat all", zorder=2)
    ax2.axhline(0, color=C["lgray"], lw=0.4)

    a2 = nb_x[m] > nb_n[m]
    ax2.fill_between(tz[a2], nb_n[m][a2], nb_x[m][a2], alpha=0.15,
                     color=C["blue"], label="Advantage", zorder=1)

    ti = np.argmin(np.abs(t[m] - 0.02))
    nx = nb_x[m][ti]
    nn = nb_n[m][ti]
    if nx > nn:
        ax2.annotate(f"\u0394NB = {nx - nn:.4f}", xy=(2, (nx + nn) / 2),
                     xytext=(3.8, nx * 1.1), fontsize=6, color=C["blue"],
                     fontweight="semibold",
                     arrowprops=dict(arrowstyle="-", color=C["blue"], lw=0.5),
                     bbox=dict(facecolor=C["blue_bg"], edgecolor=C["blue"],
                              boxstyle="round,pad=0.25", lw=0.4))

    ax2.set_xlabel("Threshold probability")
    ax2.set_ylabel("Net benefit")
    ax2.set_xlim(1, 5)
    ax2.legend(loc="lower left", fontsize=5.5)

    fig.savefig(OUT / "fig5_dca.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "fig5_dca.pdf", facecolor="white")
    plt.close(fig)
    print("  fig5")


# ══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY
# ══════════════════════════════════════════════════════════════════════════════

def fig_s1():
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.subplots_adjust(left=0.16, right=0.95, top=0.92, bottom=0.14)
    fig.patch.set_facecolor("white")
    _spine(ax)

    fpr_x, tpr_x, _ = roc_curve(y, xgb_prob)
    fpr_l, tpr_l, _ = roc_curve(y, lr_prob)
    v = ~np.isnan(news2)
    fpr_n, tpr_n, _ = roc_curve(y[v], news2[v])

    ax.fill_between(fpr_x, tpr_x, alpha=0.06, color=C["blue"])
    ax.plot(fpr_x, tpr_x, color=C["blue"], lw=1.2, label="bssentinel (0.758)")
    ax.plot(fpr_l, tpr_l, color=C["purple"], lw=0.8, ls="--", label="Logistic reg. (0.708)")
    ax.plot(fpr_n, tpr_n, color=C["orange"], lw=0.8, ls=":", label="mNEWS2 (0.568)")
    ax.plot([0, 1], [0, 1], "-", color=C["lgray"], lw=0.4)

    ax.set_xlabel("1 \u2212 Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.03)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=5.5)

    fig.savefig(OUT / "fig_s1_roc.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "fig_s1_roc.pdf", facecolor="white")
    plt.close(fig)
    print("  fig_s1")


def fig_s2():
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.subplots_adjust(left=0.16, right=0.95, top=0.92, bottom=0.14)
    fig.patch.set_facecolor("white")
    _spine(ax)

    prec_x, rec_x, _ = precision_recall_curve(y, xgb_prob)
    prec_l, rec_l, _ = precision_recall_curve(y, lr_prob)

    ax.plot(rec_x, prec_x, color=C["blue"], lw=1.0,
            label=f"bssentinel ({metrics['auprc']['xgboost']:.3f})")
    ax.plot(rec_l, prec_l, color=C["purple"], lw=0.8, ls="--",
            label=f"Logistic reg. ({metrics['auprc']['logistic_reg']:.3f})")
    ax.axhline(y.mean(), color=C["lgray"], ls="--", lw=0.5,
               label=f"No skill ({y.mean():.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.01, 1.01)
    ax.legend(loc="upper right", fontsize=5.5)

    fig.savefig(OUT / "fig_s2_pr.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "fig_s2_pr.pdf", facecolor="white")
    plt.close(fig)
    print("  fig_s2")


def fig_s3():
    xgb_cal = np.zeros(len(y))
    lr_cal_p = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(xgb_prob, y, groups):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(xgb_prob[tr], y[tr])
        xgb_cal[te] = ir.transform(xgb_prob[te])
        ir2 = IsotonicRegression(out_of_bounds="clip")
        ir2.fit(lr_prob[tr], y[tr])
        lr_cal_p[te] = ir2.transform(lr_prob[te])

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.subplots_adjust(left=0.16, right=0.95, top=0.92, bottom=0.14)
    fig.patch.set_facecolor("white")
    _spine(ax)

    for p, lab, col in [(xgb_cal, "bssentinel", C["blue"]),
                         (lr_cal_p, "Logistic reg.", C["purple"])]:
        frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, "o-", color=col, lw=0.8, ms=3.5,
                markeredgecolor="white", markeredgewidth=0.4, label=lab)

    ax.plot([0, 0.1], [0, 0.1], "--", color=C["lgray"], lw=0.5, label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction")
    ax.set_xlim(0, 0.10)
    ax.set_ylim(0, 0.10)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=5.5)

    fig.savefig(OUT / "fig_s3_calibration.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "fig_s3_calibration.pdf", facecolor="white")
    plt.close(fig)
    print("  fig_s3")


def fig_s4():
    fig, ax = plt.subplots(figsize=(3.5, 5.5))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.02)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)

    def box(x, yy, w, h, text, ec=C["blue"], fc="#F7F9FC", fs=6):
        ax.add_patch(FancyBboxPatch((x, yy), w, h, boxstyle="round,pad=0.15",
                     facecolor=fc, edgecolor=ec, lw=0.6, zorder=2))
        ax.text(x + w / 2, yy + h / 2, text, ha="center", va="center",
                fontsize=fs, color=C["ink"], zorder=3)

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=C["gray"],
                                   lw=0.6, mutation_scale=7))

    box(2, 10.8, 6, 0.7, "MIMIC-IV v3.1\nn = 526,748 admissions")
    arr(5, 10.8, 5, 10.3)
    box(2, 9.4, 6, 0.7,
        "Age \u226518, LOS \u22656 h, \u22651 ICU stay\nn = 80,442 admissions | 62,025 patients",
        ec=C["green"], fc=C["green_bg"])
    ax.text(8.8, 10.5, "Excluded\nn = 446,306", fontsize=5, color=C["gray"], ha="center")
    ax.annotate("", xy=(8.8, 10.1), xytext=(7.5, 10.5),
                arrowprops=dict(arrowstyle="-", color=C["lgray"], lw=0.5))
    arr(5, 9.4, 5, 8.9)
    box(2, 8.0, 6, 0.7, "6-hourly windows\nn = 8,560,730 candidates")
    arr(5, 8.0, 5, 7.5)
    box(2, 6.6, 6, 0.7, "No vital signs \u2014 excluded\nn = 7,225,957 removed",
        ec=C["red"], fc=C["red_bg"])
    arr(5, 6.6, 5, 6.1)
    box(2, 5.2, 6, 0.7,
        "Analysis cohort\nn = 1,334,773 windows | 80,442 admissions",
        ec=C["blue"], fc=C["blue_bg"])
    arr(5, 5.2, 3.2, 4.6)
    arr(5, 5.2, 6.8, 4.6)
    box(1, 3.7, 4, 0.7, "Positive: 26,201 (1.96%)\nDeterioration \u226424 h",
        ec=C["red"], fc=C["red_bg"], fs=5.5)
    box(5.5, 3.7, 4, 0.7, "Negative: 1,308,572\nNo event \u226424 h",
        ec=C["green"], fc=C["green_bg"], fs=5.5)
    arr(3, 3.7, 5, 3.1)
    arr(7.5, 3.7, 5, 3.1)
    box(2, 2.2, 6, 0.7,
        "5-fold GroupKFold (patient-stratified)\nPooled out-of-fold predictions",
        ec=C["blue"], fc=C["blue_bg"], fs=5.5)

    fig.savefig(OUT / "fig_s4_cohort.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "fig_s4_cohort.pdf", facecolor="white")
    plt.close(fig)
    print("  fig_s4")


def fig_s5():
    miss = {
        "Bilirubin": 75.9, "Lactate": 72.5, "CRP": 98.5, "INR": 51.4,
        "BUN": 23.8, "Creatinine": 23.1, "Glucose": 22.4, "Potassium": 19.8,
        "Sodium": 19.5, "Haemoglobin": 19.2, "WBC": 19.0, "Platelets": 18.9,
        "Temperature": 3.8, "Resp. rate": 2.9, "SpO$_2$": 2.7,
        "Heart rate": 2.5, "Systolic BP": 2.4, "Diastolic BP": 2.4,
    }
    names = list(miss.keys())
    vals = list(miss.values())
    cols = [C["green"] if v > 10 else C["blue"] for v in vals]

    fig, ax = plt.subplots(figsize=(4.7, 4.2))
    fig.subplots_adjust(left=0.25, right=0.92, top=0.92, bottom=0.10)
    fig.patch.set_facecolor("white")
    _spine(ax)

    yp = np.arange(len(names))

    # Lollipop style
    for i in range(len(names)):
        ax.plot([0, vals[i]], [yp[i], yp[i]], "-", color=cols[i], lw=0.7,
                alpha=0.5, zorder=2)
    ax.scatter(vals, yp, c=cols, s=30, zorder=5, edgecolors="white", linewidths=0.5)

    for i, v in enumerate(vals):
        ax.text(v + 1.2, i, f"{v:.1f}%", va="center", fontsize=5.5, color=C["ink"])

    # Subtle guides
    for i in range(len(names)):
        ax.axhline(yp[i], color=C["vlgray"], lw=0.3, zorder=1)

    ax.set_yticks(yp)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Missingness (%)")
    ax.set_xlim(0, 108)
    ax.invert_yaxis()
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C["green"],
               markersize=5, label="Laboratory", markeredgewidth=0),
               plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C["blue"],
               markersize=5, label="Vital signs", markeredgewidth=0)]
    ax.legend(handles=handles, fontsize=5.5, loc="lower right")

    fig.savefig(OUT / "fig_s5_missingness.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "fig_s5_missingness.pdf", facecolor="white")
    plt.close(fig)
    print("  fig_s5")


def fig_s6():
    abl_path = Path(__file__).resolve().parent / "domain_ablation_results.json"
    if not abl_path.exists():
        print("  fig_s6 SKIPPED")
        return
    with open(abl_path) as f:
        abl = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5))
    fig.subplots_adjust(wspace=0.40, left=0.24, right=0.95, top=0.88, bottom=0.14)
    fig.patch.set_facecolor("white")

    full = abl["full_model"]["auroc"]
    doms = list(abl["domain_exclusion"].keys())
    drops = [abl["domain_exclusion"][d]["auroc_drop"] for d in doms]
    order = np.argsort(drops)[::-1]

    # ── a: Drop when removed ──
    _spine(ax1)
    _panel(ax1, "a", x=-0.30)
    yp = np.arange(len(doms))
    ds = [doms[i] for i in order]
    dv = [drops[i] for i in order]

    bar_cols = [C["red"] if d > 0.01 else C["orange"] if d > 0.002 else C["lgray"]
                for d in dv]

    # Lollipop style
    for i in range(len(ds)):
        ax1.plot([0, dv[i]], [yp[i], yp[i]], "-", color=bar_cols[i], lw=0.7,
                 alpha=0.5, zorder=2)
    ax1.scatter(dv, yp, c=bar_cols, s=35, zorder=5, edgecolors="white", linewidths=0.5)

    for i, d in enumerate(dv):
        ax1.text(d + 0.003, i,
                 f"\u2212{d:.4f}" if d > 0 else f"{d:+.4f}",
                 fontsize=5.5, va="center", color=C["ink"])

    for i in range(len(ds)):
        ax1.axhline(yp[i], color=C["vlgray"], lw=0.3, zorder=1)

    ax1.set_yticks(yp)
    ax1.set_yticklabels(ds, fontsize=6)
    ax1.set_xlabel("\u0394AUROC when removed")
    ax1.set_xlim(0, max(dv) * 1.25)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(left=False)

    # ── b: Domain-only AUROC ──
    _spine(ax2)
    _panel(ax2, "b", x=-0.16)
    d_only = abl["domain_only"]
    ns = list(d_only.keys())
    avals = [d_only[n]["auroc"] for n in ns]
    o2 = np.argsort(avals)
    ns2 = [ns[i] for i in o2]
    av2 = [avals[i] for i in o2]
    yp2 = np.arange(len(ns2))

    # Lollipop
    for i in range(len(ns2)):
        ax2.plot([0.45, av2[i]], [yp2[i], yp2[i]], "-", color=C["blue"],
                 lw=0.7, alpha=0.4, zorder=2)
    ax2.scatter(av2, yp2, c=C["blue"], s=35, zorder=5, alpha=0.85,
                edgecolors="white", linewidths=0.5)

    for i, a in enumerate(av2):
        ax2.text(a + 0.008, i, f"{a:.3f}", fontsize=5.5, va="center", color=C["ink"])

    for i in range(len(ns2)):
        ax2.axhline(yp2[i], color=C["vlgray"], lw=0.3, zorder=1)

    ax2.axvline(0.5, color=C["lgray"], lw=0.5, ls="--")
    ax2.axvline(full, color=C["red"], lw=0.6, ls="--")
    ax2.text(full - 0.005, len(ns2) - 0.3, f"Full\n{full:.3f}",
             fontsize=5.5, color=C["red"], ha="right", va="top", fontweight="semibold")

    ax2.set_yticks(yp2)
    ax2.set_yticklabels(ns2, fontsize=6)
    ax2.set_xlabel("AUROC")
    ax2.set_xlim(0.45, 0.80)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(left=False)

    fig.savefig(OUT / "fig_s6_ablation.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "fig_s6_ablation.pdf", facecolor="white")
    plt.close(fig)
    print("  fig_s6")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating modern Nature-style figures...")
    print("\nMain:")
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("\nSupplementary:")
    fig_s1(); fig_s2(); fig_s3(); fig_s4(); fig_s5(); fig_s6()
    print(f"\nAll saved to {OUT}/ (.png + .pdf)")

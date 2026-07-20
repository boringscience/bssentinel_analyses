"""
evaluate.py
Validation and reporting for bssentinel.

Produces:
  - AUROC with 95% bootstrap CI
  - AUPRC (precision-recall, appropriate for imbalanced outcomes)
  - Sensitivity / specificity / PPV / NPV at operating thresholds
  - Calibration plot (reliability diagram)
  - Model vs NEWS2 comparison table
  - DeLong test for AUC difference significance
  - Saves all plots to output_dir/figures/
  - Saves metrics to output_dir/metrics.json
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

log = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    log.warning("matplotlib not installed — plots will not be generated")


# ── Bootstrap AUC with CI ──────────────────────────────────────────────────────

def bootstrap_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute AUC with bootstrap confidence interval.

    Returns: (auc, lower_bound, upper_bound)
    """
    rng   = np.random.default_rng(seed)
    aucs  = []
    n     = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))

    aucs  = np.array(aucs)
    alpha = (1 - ci) / 2
    return (
        float(roc_auc_score(y_true, y_proba)),
        float(np.percentile(aucs, alpha * 100)),
        float(np.percentile(aucs, (1 - alpha) * 100)),
    )


def delong_test(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
) -> tuple[float, float]:
    """
    DeLong et al. (1988) test for AUC difference.
    Returns (z_statistic, p_value).

    This is a simplified implementation via Mann-Whitney U kernel variance.
    For production use fastDeLong from the original R pROC package.
    """
    def auc_var(y, p):
        pos = p[y == 1]
        neg = p[y == 0]
        n1, n0 = len(pos), len(neg)
        # Placement values (kernel)
        v10 = np.array([np.mean(pi > neg) + 0.5 * np.mean(pi == neg) for pi in pos])
        v01 = np.array([np.mean(pj < pos) + 0.5 * np.mean(pj == pos) for pj in neg])
        s10 = np.var(v10, ddof=1) / n1
        s01 = np.var(v01, ddof=1) / n0
        return s10 + s01

    auc_a = roc_auc_score(y_true, proba_a)
    auc_b = roc_auc_score(y_true, proba_b)
    var_a = auc_var(y_true, proba_a)
    var_b = auc_var(y_true, proba_b)

    # Correlation term (simplified: assume independent models — conservative)
    se   = np.sqrt(var_a + var_b)
    z    = (auc_a - auc_b) / (se + 1e-12)
    p    = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


# ── Threshold metrics ──────────────────────────────────────────────────────────

def threshold_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    """Sensitivity, specificity, PPV, NPV at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    ppv  = tp / (tp + fp + 1e-12)
    npv  = tn / (tn + fn + 1e-12)
    return {
        "threshold": threshold,
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "ppv":         round(float(ppv),  4),
        "npv":         round(float(npv),  4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def news2_auc(y_true: np.ndarray, news2_scores: np.ndarray) -> float:
    """AUROC for raw NEWS2 total score as a predictor."""
    valid = ~np.isnan(news2_scores)
    if valid.sum() < 10:
        return np.nan
    return float(roc_auc_score(y_true[valid], news2_scores[valid]))


# ── Calibration ────────────────────────────────────────────────────────────────

def calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration metrics.
    Returns Brier score + fraction_of_positives per bin.
    """
    brier = float(brier_score_loss(y_true, y_proba))
    from sklearn.calibration import calibration_curve as sk_cal
    frac_pos, mean_pred = sk_cal(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return {
        "brier_score": round(brier, 4),
        "calibration_bins": [
            {"mean_predicted": round(float(p), 4), "fraction_positive": round(float(f), 4)}
            for p, f in zip(mean_pred, frac_pos)
        ],
    }


# ── Full evaluation ────────────────────────────────────────────────────────────

def evaluate(
    oof_df: pd.DataFrame,
    output_dir: str | Path,
    model_name: str = "XGBoost",
) -> dict:
    """
    Full evaluation pipeline from OOF predictions dataframe.

    Args:
        oof_df:     DataFrame with columns: label, xgb_proba, lr_proba, news2_total
        output_dir: Where to write figures/ and metrics.json
        model_name: Display name for primary model

    Returns: metrics dict
    """
    output_dir = Path(output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    y         = oof_df["label"].values
    xgb_prob  = oof_df["xgb_proba"].values
    lr_prob   = oof_df["lr_proba"].values
    news2     = oof_df["news2_total"].values if "news2_total" in oof_df.columns else np.full(len(y), np.nan)

    log.info("Computing evaluation metrics...")

    # AUROC + CI
    xgb_auc, xgb_lo, xgb_hi = bootstrap_auc(y, xgb_prob)
    lr_auc,  lr_lo,  lr_hi  = bootstrap_auc(y, lr_prob)
    n2_auc = news2_auc(y, news2)

    log.info(f"  XGBoost AUROC:  {xgb_auc:.4f} (95% CI {xgb_lo:.4f}–{xgb_hi:.4f})")
    log.info(f"  LR AUROC:       {lr_auc:.4f} (95% CI {lr_lo:.4f}–{lr_hi:.4f})")
    log.info(f"  NEWS2 AUROC:    {n2_auc:.4f}")

    # DeLong XGBoost vs NEWS2
    delong_z, delong_p = np.nan, np.nan
    if not np.isnan(n2_auc):
        valid_n2 = ~np.isnan(news2)
        delong_z, delong_p = delong_test(y[valid_n2], xgb_prob[valid_n2], news2[valid_n2])
        log.info(f"  DeLong XGB vs NEWS2: z={delong_z:.3f}, p={delong_p:.4f}")

    # AUPRC
    xgb_auprc = float(average_precision_score(y, xgb_prob))
    lr_auprc  = float(average_precision_score(y, lr_prob))
    baseline_auprc = float(y.mean())  # random classifier AUPRC = prevalence

    # Threshold analysis
    thresholds_report = []
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        thresholds_report.append(threshold_metrics(y, xgb_prob, t))

    # Find threshold for 90% sensitivity (high-recall operating point)
    fpr, tpr, roc_thresholds = roc_curve(y, xgb_prob)
    sens_90_threshold = float(roc_thresholds[np.argmax(tpr >= 0.90)])
    high_sensitivity_metrics = threshold_metrics(y, xgb_prob, sens_90_threshold)
    log.info(f"  @ 90% sensitivity: specificity={high_sensitivity_metrics['specificity']:.3f}, threshold={sens_90_threshold:.3f}")

    # Calibration
    xgb_cal = calibration_metrics(y, xgb_prob)
    lr_cal  = calibration_metrics(y, lr_prob)

    # Compile metrics
    metrics = {
        "n_observations":    int(len(y)),
        "n_positive":        int(y.sum()),
        "prevalence":        round(float(y.mean()), 4),
        "primary_model":     model_name,
        "auroc": {
            "xgboost":      {"auc": xgb_auc, "ci_lo": xgb_lo, "ci_hi": xgb_hi},
            "logistic_reg": {"auc": lr_auc,  "ci_lo": lr_lo,  "ci_hi": lr_hi},
            "news2_only":   {"auc": float(n2_auc) if not np.isnan(n2_auc) else None},
        },
        "auprc": {
            "xgboost":       xgb_auprc,
            "logistic_reg":  lr_auprc,
            "baseline_prev": baseline_auprc,
        },
        "delong_xgb_vs_news2": {
            "z": float(delong_z) if not np.isnan(delong_z) else None,
            "p": float(delong_p) if not np.isnan(delong_p) else None,
        },
        "threshold_analysis":       thresholds_report,
        "high_sensitivity_point":   high_sensitivity_metrics,
        "calibration_xgb":          xgb_cal,
        "calibration_lr":           lr_cal,
    }

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved to {metrics_path}")

    # Generate plots
    if HAS_MPL:
        _plot_roc(y, xgb_prob, lr_prob, news2, metrics, output_dir)
        _plot_pr_curve(y, xgb_prob, lr_prob, output_dir)
        _plot_calibration(y, xgb_prob, lr_prob, output_dir)
        _plot_threshold_analysis(thresholds_report, output_dir)

    return metrics


# ── Plotting ───────────────────────────────────────────────────────────────────

STYLE = {
    "bg":      "#ffffff",
    "surface": "#ffffff",
    "border":  "#cccccc",
    "accent":  "#0072B2",
    "text":    "#222222",
    "muted":   "#555555",
    "green":   "#009E73",
    "orange":  "#D55E00",
    "red":     "#CC3333",
    "blue":    "#0072B2",
}


def _apply_bss_style(ax, fig):
    """Apply publication white theme to matplotlib axes."""
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["surface"])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["border"])
    ax.tick_params(colors=STYLE["text"])
    ax.xaxis.label.set_color(STYLE["text"])
    ax.yaxis.label.set_color(STYLE["text"])
    ax.title.set_color(STYLE["text"])
    ax.grid(True, color=STYLE["border"], linewidth=0.5, alpha=0.7)


def _plot_roc(y, xgb_prob, lr_prob, news2, metrics, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    _apply_bss_style(ax, fig)

    # XGBoost
    fpr_x, tpr_x, _ = roc_curve(y, xgb_prob)
    auc_x = metrics["auroc"]["xgboost"]["auc"]
    ci_lo = metrics["auroc"]["xgboost"]["ci_lo"]
    ci_hi = metrics["auroc"]["xgboost"]["ci_hi"]
    ax.plot(fpr_x, tpr_x, color=STYLE["accent"], lw=2,
            label=f"XGBoost  AUC={auc_x:.3f} (95% CI {ci_lo:.3f}–{ci_hi:.3f})")

    # Logistic Regression
    fpr_l, tpr_l, _ = roc_curve(y, lr_prob)
    auc_l = metrics["auroc"]["logistic_reg"]["auc"]
    ax.plot(fpr_l, tpr_l, color=STYLE["blue"], lw=1.5, linestyle="--",
            label=f"Logistic Reg  AUC={auc_l:.3f}")

    # NEWS2
    valid = ~np.isnan(news2)
    if valid.sum() > 0:
        fpr_n, tpr_n, _ = roc_curve(y[valid], news2[valid])
        auc_n = metrics["auroc"]["news2_only"]["auc"]
        ax.plot(fpr_n, tpr_n, color=STYLE["orange"], lw=1.5, linestyle=":",
                label=f"NEWS2 only  AUC={auc_n:.3f}")

    ax.plot([0, 1], [0, 1], color=STYLE["muted"], linestyle="--", lw=1)
    ax.set_xlabel("1 – Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("ROC Curve — Deterioration Prediction")
    ax.legend(fontsize=8, facecolor=STYLE["surface"], edgecolor=STYLE["border"],
              labelcolor=STYLE["text"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved roc_curve.png")


def _plot_pr_curve(y, xgb_prob, lr_prob, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    _apply_bss_style(ax, fig)

    prec_x, rec_x, _ = precision_recall_curve(y, xgb_prob)
    prec_l, rec_l, _ = precision_recall_curve(y, lr_prob)
    auprc_x = float(average_precision_score(y, xgb_prob))
    auprc_l = float(average_precision_score(y, lr_prob))
    prev     = float(y.mean())

    ax.plot(rec_x, prec_x, color=STYLE["accent"], lw=2,  label=f"XGBoost  AUPRC={auprc_x:.3f}")
    ax.plot(rec_l, prec_l, color=STYLE["blue"],   lw=1.5, linestyle="--", label=f"Logistic Reg  AUPRC={auprc_l:.3f}")
    ax.axhline(prev, color=STYLE["muted"], linestyle="--", lw=1, label=f"Baseline (prev={prev:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Deterioration Prediction")
    ax.legend(fontsize=8, facecolor=STYLE["surface"], edgecolor=STYLE["border"], labelcolor=STYLE["text"])
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved pr_curve.png")


def _plot_calibration(y, xgb_prob, lr_prob, output_dir):
    from sklearn.calibration import calibration_curve as sk_cal
    fig, ax = plt.subplots(figsize=(7, 6))
    _apply_bss_style(ax, fig)

    for proba, label, color in [
        (xgb_prob, "XGBoost",      STYLE["accent"]),
        (lr_prob,  "Logistic Reg", STYLE["blue"]),
    ]:
        frac_pos, mean_pred = sk_cal(y, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", color=color, lw=2, markersize=5, label=label)

    ax.plot([0, 1], [0, 1], "--", color=STYLE["muted"], lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Plot (Reliability Diagram)")
    ax.legend(fontsize=8, facecolor=STYLE["surface"], edgecolor=STYLE["border"], labelcolor=STYLE["text"])
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved calibration.png")


def _plot_threshold_analysis(thresholds_report: list[dict], output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_bss_style(ax, fig)

    t   = [r["threshold"]   for r in thresholds_report]
    sen = [r["sensitivity"] for r in thresholds_report]
    spe = [r["specificity"] for r in thresholds_report]
    ppv = [r["ppv"]         for r in thresholds_report]

    ax.plot(t, sen, "o-", color=STYLE["accent"], lw=2, markersize=5, label="Sensitivity")
    ax.plot(t, spe, "s-", color=STYLE["blue"],   lw=2, markersize=5, label="Specificity")
    ax.plot(t, ppv, "^-", color=STYLE["orange"], lw=2, markersize=5, label="PPV")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Sensitivity / Specificity / PPV by Threshold")
    ax.legend(fontsize=8, facecolor=STYLE["surface"], edgecolor=STYLE["border"], labelcolor=STYLE["text"])
    ax.set_xlim(0.05, 0.55)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "threshold_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved threshold_analysis.png")


def print_summary(metrics: dict) -> None:
    """Print a human-readable evaluation summary."""
    print("\n" + "═" * 60)
    print("  bssentinel — Validation Summary")
    print("═" * 60)
    print(f"  Dataset:  {metrics['n_observations']:,} windows  |  {metrics['n_positive']:,} events  |  prevalence {metrics['prevalence']:.1%}")
    print()
    print("  AUROC (95% CI)")
    for k, v in metrics["auroc"].items():
        if v and v.get("auc"):
            ci = f"({v['ci_lo']:.3f}–{v['ci_hi']:.3f})" if "ci_lo" in v else ""
            print(f"    {k:<20}  {v['auc']:.4f}  {ci}")
    print()
    print("  AUPRC")
    for k, v in metrics["auprc"].items():
        print(f"    {k:<20}  {v:.4f}")
    print()
    dl = metrics.get("delong_xgb_vs_news2", {})
    if dl.get("p") is not None:
        sig = "✓ significant" if dl["p"] < 0.05 else "✗ not significant"
        print(f"  DeLong XGB vs NEWS2:  z={dl['z']:.3f}  p={dl['p']:.4f}  {sig}")
    print()
    hs = metrics.get("high_sensitivity_point", {})
    if hs:
        print(f"  @ 90% Sensitivity:  specificity={hs['specificity']:.3f}  PPV={hs['ppv']:.3f}  threshold={hs['threshold']:.3f}")
    print("═" * 60 + "\n")

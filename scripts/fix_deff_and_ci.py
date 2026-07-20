"""
fix_deff_and_ci.py
Corrects the design-effect (DEFF) calculation and recomputes confidence intervals.

Problem: The manuscript reports cluster size = 4.4, but:
  1,334,773 windows / 80,442 admissions = 16.6 windows/admission
  The 4.4 value is patient-DAYS per admission (333,693 / 80,442 = 4.15).

This script recomputes DEFF with the correct cluster size and provides
corrected 95% CIs for AUROC.

Usage:
    python new_analyses/fix_deff_and_ci.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ── Load data ────────────────────────────────────────────────────────────────
oof_path = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "oof_predictions.parquet"
out_dir = Path(__file__).resolve().parent

oof = pd.read_parquet(oof_path)
y = oof["label"].values
xgb_prob = oof["xgb_proba"].values
lr_prob = oof["lr_proba"].values
news2 = oof["news2_total"].values
groups = oof["hadm_id"].values

n_windows = len(y)
n_admissions = oof["hadm_id"].nunique()
windows_per_admission = oof.groupby("hadm_id").size()

mean_cluster_size = windows_per_admission.mean()
median_cluster_size = windows_per_admission.median()

print(f"Total windows: {n_windows:,}")
print(f"Total admissions: {n_admissions:,}")
print(f"Mean windows/admission: {mean_cluster_size:.1f}")
print(f"Median windows/admission: {median_cluster_size:.1f}")
print(f"IQR: {windows_per_admission.quantile(0.25):.0f}–{windows_per_admission.quantile(0.75):.0f}")
print(f"Monitored patient-days: {n_windows * 6 / 24:,.0f}")
print(f"Patient-days/admission: {(n_windows * 6 / 24) / n_admissions:.1f}  ← manuscript's '4.4'")

# ── Bootstrap window-level AUROC CI ──────────────────────────────────────────
def bootstrap_auc(y_true, y_proba, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
    aucs = np.array(aucs)
    auc = roc_auc_score(y_true, y_proba)
    return auc, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

xgb_auc, xgb_lo_win, xgb_hi_win = bootstrap_auc(y, xgb_prob)
lr_auc, lr_lo_win, lr_hi_win = bootstrap_auc(y, lr_prob)

print(f"\nWindow-level bootstrap CIs:")
print(f"  XGBoost: {xgb_auc:.4f} ({xgb_lo_win:.4f}–{xgb_hi_win:.4f})")
print(f"  LR:      {lr_auc:.4f} ({lr_lo_win:.4f}–{lr_hi_win:.4f})")

# ── DEFF correction ──────────────────────────────────────────────────────────
# DEFF = 1 + (m̄ - 1) × ICC
# where m̄ = mean cluster size, ICC = intraclass correlation

ICC = 0.10  # conservative estimate (same as manuscript)

# WRONG (manuscript): cluster_size = 4.4 → DEFF = 1 + 3.4 × 0.10 = 1.34
deff_wrong = 1 + (4.4 - 1) * ICC

# CORRECT: cluster_size = 16.6 → DEFF = 1 + 15.6 × 0.10 = 2.56
deff_correct = 1 + (mean_cluster_size - 1) * ICC

print(f"\nDEFF comparison:")
print(f"  Manuscript (cluster=4.4):  DEFF = {deff_wrong:.2f}")
print(f"  Corrected (cluster={mean_cluster_size:.1f}): DEFF = {deff_correct:.2f}")
print(f"  CI widening factor: √({deff_correct:.2f}/{deff_wrong:.2f}) = {np.sqrt(deff_correct / deff_wrong):.3f}")

# Apply DEFF to bootstrap CIs
# Corrected SE = window-level SE × √DEFF
# CI_corrected = AUC ± z_0.975 × SE_corrected

def apply_deff(auc, ci_lo, ci_hi, deff):
    """Widen a CI by the design-effect factor."""
    se_window = (ci_hi - ci_lo) / (2 * 1.96)
    se_corrected = se_window * np.sqrt(deff)
    return (
        round(auc - 1.96 * se_corrected, 4),
        round(auc + 1.96 * se_corrected, 4),
    )

# Corrected CIs
xgb_lo_corr, xgb_hi_corr = apply_deff(xgb_auc, xgb_lo_win, xgb_hi_win, deff_correct)
lr_lo_corr, lr_hi_corr = apply_deff(lr_auc, lr_lo_win, lr_hi_win, deff_correct)

# Old manuscript CIs (with wrong DEFF)
xgb_lo_old, xgb_hi_old = apply_deff(xgb_auc, xgb_lo_win, xgb_hi_win, deff_wrong)

print(f"\nCorrected patient-level CIs (DEFF={deff_correct:.2f}):")
print(f"  XGBoost: {xgb_auc:.4f} ({xgb_lo_corr:.4f}–{xgb_hi_corr:.4f})")
print(f"  LR:      {lr_auc:.4f} ({lr_lo_corr:.4f}–{lr_hi_corr:.4f})")
print(f"\nOld manuscript CIs (DEFF={deff_wrong:.2f}):")
print(f"  XGBoost: {xgb_auc:.4f} ({xgb_lo_old:.4f}–{xgb_hi_old:.4f})")

# ── Also compute with alternative ICC values for sensitivity ─────────────────
print(f"\nSensitivity to ICC assumption (cluster size = {mean_cluster_size:.1f}):")
print(f"  {'ICC':>5}  {'DEFF':>6}  {'XGBoost 95% CI':>22}")
for icc in [0.05, 0.10, 0.15, 0.20]:
    d = 1 + (mean_cluster_size - 1) * icc
    lo, hi = apply_deff(xgb_auc, xgb_lo_win, xgb_hi_win, d)
    print(f"  {icc:>5.2f}  {d:>6.2f}  {xgb_auc:.4f} ({lo:.4f}–{hi:.4f})")

# ── Save results ─────────────────────────────────────────────────────────────
results = {
    "description": "Corrected DEFF and CI calculation",
    "n_windows": n_windows,
    "n_admissions": n_admissions,
    "mean_windows_per_admission": round(mean_cluster_size, 1),
    "median_windows_per_admission": float(median_cluster_size),
    "patient_days_per_admission": round((n_windows * 6 / 24) / n_admissions, 1),
    "ICC": ICC,
    "original_deff": round(deff_wrong, 2),
    "original_cluster_size": 4.4,
    "corrected_deff": round(deff_correct, 2),
    "corrected_cluster_size": round(mean_cluster_size, 1),
    "window_level_ci": {
        "xgboost": {"auc": round(xgb_auc, 4), "ci": [round(xgb_lo_win, 4), round(xgb_hi_win, 4)]},
        "lr": {"auc": round(lr_auc, 4), "ci": [round(lr_lo_win, 4), round(lr_hi_win, 4)]},
    },
    "corrected_patient_level_ci": {
        "xgboost": {"auc": round(xgb_auc, 4), "ci": [xgb_lo_corr, xgb_hi_corr]},
        "lr": {"auc": round(lr_auc, 4), "ci": [lr_lo_corr, lr_hi_corr]},
    },
    "manuscript_changes": {
        "methods_statistical_analysis": (
            f"Replace cluster size 4.4 with {mean_cluster_size:.1f} (mean windows per admission). "
            f"DEFF changes from 1.34 to {deff_correct:.2f}. "
            f"Explain that 4.4 was patient-days/admission, not windows/admission."
        ),
        "table_3": (
            f"Update patient-level 95% CI from (0.754–0.762) to ({xgb_lo_corr}–{xgb_hi_corr})"
        ),
        "results_text": (
            f"Update all CI references: XGBoost AUROC 0.758 "
            f"(patient-level 95% CI {xgb_lo_corr}–{xgb_hi_corr})"
        ),
    },
}

out_file = out_dir / "deff_corrected.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_file}")

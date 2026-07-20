"""
fix_brier_and_calibration.py
Recomputes the Brier score using properly calibrated probabilities.

Problem: The original Brier score (0.175) was computed on raw XGBoost
out-of-fold probabilities that are inflated by scale_pos_weight.
The correct Brier should be computed after isotonic regression calibration.

Usage:
    python new_analyses/fix_brier_and_calibration.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold

# ── Load OOF predictions ──────────────────────────────────────────────────────
oof_path = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "oof_predictions.parquet"
out_dir = Path(__file__).resolve().parent

oof = pd.read_parquet(oof_path)
y = oof["label"].values
xgb_raw = oof["xgb_proba"].values
lr_raw = oof["lr_proba"].values
groups = oof["hadm_id"].values

n = len(y)
prevalence = y.mean()

print(f"n = {n:,}  |  events = {y.sum():,}  |  prevalence = {prevalence:.4f}")
print(f"Raw XGBoost proba range: {xgb_raw.min():.4f} – {xgb_raw.max():.4f}")

# ── Cross-validated isotonic calibration ──────────────────────────────────────
# We must calibrate in a cross-validated manner to avoid overfitting the
# calibration mapping. Use GroupKFold by hadm_id (same as training).

xgb_calibrated = np.zeros(n)
lr_calibrated = np.zeros(n)

gkf = GroupKFold(n_splits=5)

for fold, (cal_train_idx, cal_test_idx) in enumerate(gkf.split(xgb_raw, y, groups)):
    # Fit isotonic calibration on cal_train, apply to cal_test
    ir_xgb = IsotonicRegression(out_of_bounds="clip")
    ir_xgb.fit(xgb_raw[cal_train_idx], y[cal_train_idx])
    xgb_calibrated[cal_test_idx] = ir_xgb.transform(xgb_raw[cal_test_idx])

    ir_lr = IsotonicRegression(out_of_bounds="clip")
    ir_lr.fit(lr_raw[cal_train_idx], y[cal_train_idx])
    lr_calibrated[cal_test_idx] = ir_lr.transform(lr_raw[cal_test_idx])

    print(f"  Fold {fold+1}: XGB cal range = {xgb_calibrated[cal_test_idx].min():.5f}–{xgb_calibrated[cal_test_idx].max():.5f}")

# ── Compute corrected Brier scores ───────────────────────────────────────────
brier_xgb_raw = brier_score_loss(y, xgb_raw)
brier_xgb_cal = brier_score_loss(y, xgb_calibrated)
brier_lr_raw = brier_score_loss(y, lr_raw)
brier_lr_cal = brier_score_loss(y, lr_calibrated)
brier_naive = prevalence * (1 - prevalence)  # always-predict-prevalence model

print(f"\n{'Metric':<35} {'Value':>10}")
print("=" * 47)
print(f"{'Brier (naive predict prevalence)':<35} {brier_naive:>10.4f}")
print(f"{'Brier (XGBoost raw — ORIGINAL)':<35} {brier_xgb_raw:>10.4f}")
print(f"{'Brier (XGBoost calibrated — FIXED)':<35} {brier_xgb_cal:>10.4f}")
print(f"{'Brier (LR raw)':<35} {brier_lr_raw:>10.4f}")
print(f"{'Brier (LR calibrated)':<35} {brier_lr_cal:>10.4f}")

# Brier skill score (BSS): 1 - Brier/Brier_ref
bss_xgb = 1 - brier_xgb_cal / brier_naive
bss_lr = 1 - brier_lr_cal / brier_naive
print(f"\n{'Brier Skill Score (XGBoost)':<35} {bss_xgb:>10.4f}")
print(f"{'Brier Skill Score (LR)':<35} {bss_lr:>10.4f}")

# ── Calibration curve (reliability diagram data) ─────────────────────────────
frac_pos_xgb, mean_pred_xgb = calibration_curve(y, xgb_calibrated, n_bins=10, strategy="quantile")
frac_pos_lr, mean_pred_lr = calibration_curve(y, lr_calibrated, n_bins=10, strategy="quantile")

print(f"\nCalibration bins (XGBoost, calibrated):")
print(f"  {'Mean Predicted':>15}  {'Fraction Positive':>18}")
for mp, fp in zip(mean_pred_xgb, frac_pos_xgb):
    print(f"  {mp:>15.4f}  {fp:>18.4f}")

# ── Save results ─────────────────────────────────────────────────────────────
results = {
    "description": "Corrected Brier scores using cross-validated isotonic calibration",
    "original_brier_xgb_raw": round(brier_xgb_raw, 4),
    "corrected_brier_xgb_calibrated": round(brier_xgb_cal, 4),
    "corrected_brier_lr_calibrated": round(brier_lr_cal, 4),
    "brier_naive_prevalence": round(brier_naive, 4),
    "brier_skill_score_xgb": round(bss_xgb, 4),
    "brier_skill_score_lr": round(bss_lr, 4),
    "calibration_bins_xgb": [
        {"mean_predicted": round(float(mp), 4), "fraction_positive": round(float(fp), 4)}
        for mp, fp in zip(mean_pred_xgb, frac_pos_xgb)
    ],
    "calibration_bins_lr": [
        {"mean_predicted": round(float(mp), 4), "fraction_positive": round(float(fp), 4)}
        for mp, fp in zip(mean_pred_lr, frac_pos_lr)
    ],
    "manuscript_changes": {
        "table_3": f"Replace Brier score 0.175 with {brier_xgb_cal:.4f} (XGBoost) and 0.213 with {brier_lr_cal:.4f} (LR)",
        "results_text": "Update 'Post-hoc isotonic regression calibration reduced the Brier score to ...'",
    },
}

out_file = out_dir / "brier_score_corrected.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {out_file}")

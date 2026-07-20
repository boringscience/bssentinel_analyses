"""
fix_table4_and_fpr.py
Recomputes Table 4 with correct values, adds TP column, fixes FPR definition.

Problems:
  1. Total alerts at threshold 0.25 is off (980,572 vs 981,185)
  2. FPR is reported as FP/total_windows (60.9%) instead of standard FPR = 1 - specificity (62.1%)
  3. No TP column — reviewers can't verify arithmetic

Usage:
    python new_analyses/fix_table4_and_fpr.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

# ── Load OOF predictions ─────────────────────────────────────────────────────
oof_path = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "oof_predictions.parquet"
out_dir = Path(__file__).resolve().parent

oof = pd.read_parquet(oof_path)
y = oof["label"].values
xgb_prob = oof["xgb_proba"].values

n_total = len(y)
n_positive = int(y.sum())
n_negative = n_total - n_positive
prevalence = y.mean()
monitored_patient_days = n_total * 6 / 24

print(f"n = {n_total:,}  |  events = {n_positive:,}  |  prevalence = {prevalence:.4f}")
print(f"Monitored patient-days = {monitored_patient_days:,.0f}")
print()


def compute_threshold_row(y_true, y_proba, threshold, patient_days):
    """Compute all metrics for a single threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    total_alerts = tp + fp
    alerts_per_day = total_alerts / patient_days
    # Standard FPR = 1 - specificity = FP / (FP + TN)
    fpr_standard = fp / (fp + tn)
    # What manuscript reported: FP / total windows
    false_alert_proportion = fp / len(y_true)

    return {
        "threshold": threshold,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "ppv": round(ppv, 4),
        "npv": round(npv, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total_alerts": int(total_alerts),
        "alerts_per_patient_day": round(alerts_per_day, 2),
        "fpr_standard": round(fpr_standard, 4),
        "false_alert_proportion": round(false_alert_proportion, 4),
    }


# ── Find exact threshold for 90% sensitivity ────────────────────────────────
fpr_curve, tpr_curve, roc_thresholds = roc_curve(y, xgb_prob)
sens_90_idx = np.argmax(tpr_curve >= 0.90)
threshold_90 = float(roc_thresholds[sens_90_idx])
print(f"Threshold for 90% sensitivity: {threshold_90:.4f}")

# ── Compute corrected Table 4 ───────────────────────────────────────────────
thresholds = [0.10, 0.20, 0.25, threshold_90, 0.35, 0.40, 0.50]

print(f"\n{'Thr':>6} {'Sens':>7} {'Spec':>7} {'PPV':>6} {'NPV':>6} "
      f"{'TP':>7} {'FP':>9} {'Total':>9} {'Alerts/Day':>11} "
      f"{'FPR(std)':>9} {'FP/N':>7}")
print("=" * 105)

table4_corrected = []
for t in thresholds:
    row = compute_threshold_row(y, xgb_prob, t, monitored_patient_days)
    table4_corrected.append(row)

    marker = "*" if abs(t - threshold_90) < 0.01 else " "
    print(f"{t:>5.2f}{marker} {row['sensitivity']:>6.1%} {row['specificity']:>6.1%} "
          f"{row['ppv']:>5.1%} {row['npv']:>5.1%} "
          f"{row['tp']:>7,} {row['fp']:>9,} {row['total_alerts']:>9,} "
          f"{row['alerts_per_patient_day']:>11.2f} "
          f"{row['fpr_standard']:>8.1%} {row['false_alert_proportion']:>6.1%}")

# ── Comparison with manuscript values ────────────────────────────────────────
print("\n\nDiscrepancies with manuscript Table 4:")
manuscript_total_alerts_025 = 980572
actual_total_alerts_025 = table4_corrected[2]["total_alerts"]  # threshold 0.25
print(f"  Threshold 0.25 total alerts: manuscript={manuscript_total_alerts_025:,} actual={actual_total_alerts_025:,} "
      f"(diff={actual_total_alerts_025 - manuscript_total_alerts_025:+,})")

# FPR issue at recommended threshold
rec_row = [r for r in table4_corrected if abs(r["threshold"] - threshold_90) < 0.01][0]
print(f"\n  At ~90% sensitivity threshold ({threshold_90:.4f}):")
print(f"    Standard FPR (1-specificity): {rec_row['fpr_standard']:.1%}")
print(f"    False-alert proportion (FP/N): {rec_row['false_alert_proportion']:.1%}")
print(f"    Manuscript reported: 60.9% (was FP/N, labelled as 'false-positive rate')")
print(f"    → Should use standard FPR = {rec_row['fpr_standard']:.1%} or rename to 'false-alert proportion'")

# ── Save results ─────────────────────────────────────────────────────────────
results = {
    "description": "Corrected Table 4 with TP column and fixed FPR definition",
    "threshold_for_90pct_sensitivity": round(threshold_90, 4),
    "n_total_windows": n_total,
    "n_events": n_positive,
    "prevalence": round(prevalence, 4),
    "monitored_patient_days": int(monitored_patient_days),
    "corrected_table4": table4_corrected,
    "manuscript_changes": {
        "table_4": [
            "Add TP column for arithmetic transparency",
            f"Fix total alerts at threshold 0.25: {manuscript_total_alerts_025:,} → {actual_total_alerts_025:,}",
            "Rename 'False-positive rate per window' to either 'Standard FPR (1-specificity)' or 'False-alert proportion (FP/total windows)'",
            "Use recommended threshold = {:.4f} (not 0.31)".format(threshold_90),
        ],
        "table_3": "Replace 'False-positive rate per window 60.9%' with 'FPR (1-specificity) = {:.1%}'".format(rec_row["fpr_standard"]),
    },
}

out_file = out_dir / "table4_corrected.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_file}")

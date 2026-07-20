"""
sensitivity_outcome.py
Sensitivity analysis restricting the outcome to mechanical ventilation + mortality only
(excluding ICU transfer, which includes planned post-operative admissions).

Addresses Reviewer 1, Reviewer 5, and Editor comments about outcome contamination
from planned ICU admissions.

This script requires access to the full feature matrix WITH the original
per-component outcome labels. If unavailable, it can be run with the cached
feature matrix augmented with outcome components.

Usage:
    python new_analyses/sensitivity_outcome.py --parquet /path/to/feature_matrix_with_outcomes.parquet

If you don't have component-level labels, this script provides an alternative
approach using the existing data by examining the surgical admission subgroup.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "auc",
    "early_stopping_rounds": 30,
    "random_state": 42,
    "n_jobs": -1,
}

N_SPLITS = 5


def train_and_evaluate(X, y, groups, label="model"):
    """Train XGBoost with GroupKFold CV and return OOF AUROC + CI."""
    n_pos = y.sum()
    if n_pos < 10:
        log.warning(f"  {label}: Only {n_pos} positive samples — skipping")
        return None

    pos_weight = (len(y) - n_pos) / n_pos
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_proba = np.zeros(len(y))

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = xgb.XGBClassifier(scale_pos_weight=pos_weight, **XGB_PARAMS)
        model.fit(
            X.iloc[train_idx], y[train_idx],
            eval_set=[(X.iloc[val_idx], y[val_idx])],
            verbose=False,
        )
        oof_proba[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    elapsed = time.time() - t0

    auroc = roc_auc_score(y, oof_proba)
    auprc = average_precision_score(y, oof_proba)

    rng = np.random.default_rng(42)
    aucs = []
    for _ in range(1000):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], oof_proba[idx]))
    ci_lo = float(np.percentile(aucs, 2.5))
    ci_hi = float(np.percentile(aucs, 97.5))

    log.info(f"  {label}: AUROC={auroc:.4f} ({ci_lo:.4f}–{ci_hi:.4f}) "
             f"AUPRC={auprc:.4f} prev={y.mean():.4f} [{elapsed:.0f}s]")

    return {
        "auroc": round(auroc, 4),
        "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4),
        "auprc": round(auprc, 4),
        "n_total": int(len(y)),
        "n_events": int(n_pos),
        "prevalence": round(float(y.mean()), 4),
        "elapsed_seconds": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis — restricted outcome")
    parser.add_argument("--parquet", type=str, help="Feature matrix with component outcome columns")
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    out_dir = Path(args.output)
    feature_cols_path = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "feature_cols.json"

    with open(feature_cols_path) as f:
        all_feature_cols = json.load(f)

    results = {}

    if args.parquet and Path(args.parquet).exists():
        df = pd.read_parquet(args.parquet)
        log.info(f"Loaded feature matrix: {len(df):,} rows")

        available_cols = [c for c in all_feature_cols if c in df.columns]
        X = df[available_cols]
        groups = df["hadm_id"].values

        # ── Analysis 1: Original composite outcome ───────────────────────
        log.info("\n[1] Original composite outcome (ICU + MV + mortality)...")
        y_composite = df["label"].values
        results["original_composite"] = train_and_evaluate(X, y_composite, groups, "Composite (original)")

        # ── Analysis 2: MV + mortality only (if columns exist) ───────────
        if "label_mv_mortality" in df.columns:
            log.info("\n[2] Restricted outcome (MV + mortality only)...")
            y_restricted = df["label_mv_mortality"].values
            results["mv_mortality_only"] = train_and_evaluate(X, y_restricted, groups, "MV + Mortality only")
        else:
            log.info("\n[2] Column 'label_mv_mortality' not found.")
            log.info("    To create it, add to feature_engineering.py:")
            log.info("      label_mv_mortality = 1 if MV or death within 24h, 0 otherwise")
            log.info("      (exclude ICU transfer events)")

        # ── Analysis 3: Exclude surgical same-day admissions ─────────────
        log.info("\n[3] Excluding same-day surgical admissions...")
        surg_col = "adm_type_SURGICAL SAME DAY ADMISSION"
        if surg_col in df.columns:
            mask_nonsurg = df[surg_col] == 0
            X_nonsurg = X[mask_nonsurg]
            y_nonsurg = y_composite[mask_nonsurg]
            groups_nonsurg = groups[mask_nonsurg]
            results["exclude_surgical"] = train_and_evaluate(
                X_nonsurg, y_nonsurg, groups_nonsurg, "Exclude surgical admissions"
            )
        else:
            log.warning(f"  Column '{surg_col}' not found")

        # ── Analysis 4: Emergency admissions only ────────────────────────
        log.info("\n[4] Emergency admissions only...")
        emer_cols = [c for c in df.columns if "EMER" in c and c.startswith("adm_type")]
        if emer_cols:
            mask_emer = df[emer_cols].sum(axis=1) > 0
            X_emer = X[mask_emer]
            y_emer = y_composite[mask_emer]
            groups_emer = groups[mask_emer]
            results["emergency_only"] = train_and_evaluate(
                X_emer, y_emer, groups_emer, "Emergency admissions only"
            )

    else:
        # ── Fallback: use OOF predictions + admission type proxy ─────────
        log.info("No feature matrix provided. Using OOF predictions for proxy analysis.")
        log.info("NOTE: Full retraining with restricted outcome requires the feature matrix.")

        oof_path = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "oof_predictions.parquet"
        oof = pd.read_parquet(oof_path)
        y = oof["label"].values
        xgb_prob = oof["xgb_proba"].values

        results["note"] = (
            "Full sensitivity analysis requires the feature matrix with component "
            "outcome labels (label_mv_mortality). The analysis below uses the existing "
            "OOF predictions and cannot retrain; it only re-evaluates the existing model."
        )

        log.info(f"\nOriginal AUROC (all admissions): {roc_auc_score(y, xgb_prob):.4f}")

        results["original_auroc"] = round(roc_auc_score(y, xgb_prob), 4)

        print("\n" + "=" * 70)
        print("TO RUN FULL SENSITIVITY ANALYSIS:")
        print("=" * 70)
        print("1. Modify feature_engineering.py to save component labels:")
        print("   - label_icu_transfer: 1 if ICU transfer within 24h")
        print("   - label_mech_vent: 1 if MV within 24h")
        print("   - label_mortality: 1 if death within 24h")
        print("   - label_mv_mortality: label_mech_vent | label_mortality")
        print("2. Re-run training pipeline to cache feature matrix:")
        print("   python ml/train.py --mimic /path/to/mimic --output ml/artifacts --cache ml/cache")
        print("3. Save feature matrix:")
        print("   df.to_parquet('ml/cache/feature_matrix.parquet')")
        print("4. Re-run this script:")
        print("   python new_analyses/sensitivity_outcome.py --parquet ml/cache/feature_matrix.parquet")
        print("=" * 70)

    # ── Save ─────────────────────────────────────────────────────────────
    results["description"] = (
        "Sensitivity analyses for outcome definition. "
        "'original_composite' = ICU transfer + MV + mortality. "
        "'mv_mortality_only' = MV + mortality (excludes ICU transfers including planned post-op). "
        "'exclude_surgical' = removes same-day surgical admissions entirely. "
        "'emergency_only' = only emergency/urgent admissions."
    )

    out_file = out_dir / "sensitivity_outcome_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()

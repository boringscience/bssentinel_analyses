"""
domain_ablation.py
Domain-level ablation analysis for bssentinel.

Trains XGBoost models with each feature domain systematically excluded
to quantify the incremental contribution of each clinical domain.

This directly addresses Reviewer 4 / Editor's requirement:
  "The authors should provide a domain-level ablation analysis showing how
   detection performance changes when each domain is included or excluded."

Requires the full feature matrix (cached or rebuilt from MIMIC-IV).

Usage:
    python new_analyses/domain_ablation.py --cache /path/to/ml/cache
    python new_analyses/domain_ablation.py --parquet /path/to/feature_matrix.parquet
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Feature domain definitions ───────────────────────────────────────────────
# Each domain maps to the set of column names it comprises.

VITAL_SIGNS = [
    "heart_rate", "heart_rate_trend", "heart_rate_worst",
    "systolic_bp", "systolic_bp_trend", "systolic_bp_worst",
    "diastolic_bp", "diastolic_bp_trend", "diastolic_bp_worst",
    "spo2", "spo2_trend", "spo2_worst",
    "respiratory_rate", "respiratory_rate_trend", "respiratory_rate_worst",
    "temperature", "temperature_trend", "temperature_worst",
    "gcs_eye", "gcs_eye_trend", "gcs_eye_worst",
    "gcs_verbal", "gcs_verbal_trend", "gcs_verbal_worst",
    "gcs_motor", "gcs_motor_trend", "gcs_motor_worst",
    "gcs_total",
]

LABORATORY = [
    "wbc", "hemoglobin", "platelets", "creatinine", "bun",
    "sodium", "potassium", "lactate", "glucose", "bilirubin", "inr", "crp",
]

MEDICATIONS = ["vasopressors", "antibiotics", "anticoagulants"]

NEWS2_SUBSCORES = [
    "news2_rr", "news2_spo2", "news2_sbp", "news2_hr",
    "news2_temp", "news2_gcs", "news2_o2", "news2_total",
]

MISSINGNESS = [
    "heart_rate_missing", "systolic_bp_missing", "diastolic_bp_missing",
    "spo2_missing", "respiratory_rate_missing", "temperature_missing",
    "wbc_missing", "hemoglobin_missing", "platelets_missing",
    "creatinine_missing", "bun_missing", "sodium_missing",
    "potassium_missing", "lactate_missing", "glucose_missing",
    "bilirubin_missing", "inr_missing", "crp_missing",
    "n_vitals_missing", "n_labs_missing",
]

TEMPORAL = [
    "admit_offset_hours", "age", "gender",
    "adm_type_AMBULATORY OBSERVATION", "adm_type_DIRECT EMER.",
    "adm_type_DIRECT OBSERVATION", "adm_type_ELECTIVE",
    "adm_type_EU OBSERVATION", "adm_type_EW EMER.",
    "adm_type_OBSERVATION ADMIT", "adm_type_SURGICAL SAME DAY ADMISSION",
    "adm_type_URGENT",
]

DOMAINS = {
    "Vital signs": VITAL_SIGNS,
    "Laboratory values": LABORATORY,
    "Medications": MEDICATIONS,
    "NEWS2 subscores": NEWS2_SUBSCORES,
    "Missingness indicators": MISSINGNESS,
    "Temporal & demographics": TEMPORAL,
}

# XGBoost hyperparameters (same as original training)
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
    """Train XGBoost with GroupKFold CV and return OOF AUROC."""
    n_pos = y.sum()
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

    # Bootstrap CI
    rng = np.random.default_rng(42)
    aucs = []
    for _ in range(1000):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], oof_proba[idx]))
    ci_lo = float(np.percentile(aucs, 2.5))
    ci_hi = float(np.percentile(aucs, 97.5))

    log.info(f"  {label}: AUROC={auroc:.4f} ({ci_lo:.4f}–{ci_hi:.4f})  AUPRC={auprc:.4f}  [{elapsed:.0f}s]")

    return {
        "auroc": round(auroc, 4),
        "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4),
        "auprc": round(auprc, 4),
        "n_features": X.shape[1],
        "elapsed_seconds": round(elapsed, 1),
    }


def load_feature_matrix(args):
    """Load the full feature matrix from cache or parquet."""
    if args.parquet:
        log.info(f"Loading feature matrix from {args.parquet}")
        return pd.read_parquet(args.parquet)

    # Try to find cached feature matrix
    cache_candidates = [
        Path(args.cache) / "feature_matrix.parquet" if args.cache else None,
        Path(__file__).resolve().parent.parent / "ml" / "cache" / "feature_matrix.parquet",
        Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "feature_matrix.parquet",
    ]
    for p in cache_candidates:
        if p and p.exists():
            log.info(f"Loading cached feature matrix from {p}")
            return pd.read_parquet(p)

    print(
        "ERROR: No feature matrix found. Please provide one:\n"
        "  --parquet /path/to/feature_matrix.parquet\n"
        "  --cache /path/to/cache/dir (containing feature_matrix.parquet)\n\n"
        "To generate the feature matrix, run:\n"
        "  cd ml && python train.py --mimic /path/to/mimic --output artifacts --cache cache\n"
        "  (this will cache feature_matrix.parquet for reuse)\n\n"
        "Alternatively, you can save the feature matrix during training by adding:\n"
        "  df.to_parquet('ml/cache/feature_matrix.parquet')\n"
        "after the build_feature_matrix() call in train.py"
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Domain-level ablation analysis")
    parser.add_argument("--parquet", type=str, help="Path to feature_matrix.parquet")
    parser.add_argument("--cache", type=str, help="Path to cache directory")
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parent),
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load feature matrix
    df = load_feature_matrix(args)
    log.info(f"Feature matrix: {len(df):,} rows, {len(df.columns)} columns")

    # Load feature column names
    feature_cols_path = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "feature_cols.json"
    with open(feature_cols_path) as f:
        all_feature_cols = json.load(f)

    # Ensure all feature columns exist
    available = [c for c in all_feature_cols if c in df.columns]
    missing_cols = set(all_feature_cols) - set(available)
    if missing_cols:
        log.warning(f"Missing {len(missing_cols)} feature columns: {missing_cols}")
    all_feature_cols = available

    y = df["label"].values
    groups = df["hadm_id"].values

    log.info(f"Features: {len(all_feature_cols)}  |  Events: {y.sum():,}/{len(y):,}  ({y.mean():.4f})")

    # ── 1. Full model (baseline) ─────────────────────────────────────────────
    log.info("\n[1/8] Training FULL model (all domains)...")
    X_full = df[all_feature_cols]
    full_result = train_and_evaluate(X_full, y, groups, "FULL MODEL")

    # ── 2. Domain exclusion (leave-one-domain-out) ───────────────────────────
    ablation_results = {"full_model": full_result, "domain_exclusion": {}}

    for i, (domain_name, domain_cols) in enumerate(DOMAINS.items()):
        log.info(f"\n[{i+2}/8] Training WITHOUT {domain_name}...")
        cols_to_remove = [c for c in domain_cols if c in all_feature_cols]
        remaining_cols = [c for c in all_feature_cols if c not in cols_to_remove]

        if len(remaining_cols) == len(all_feature_cols):
            log.warning(f"  No features removed for {domain_name} — skipping")
            continue

        X_ablated = df[remaining_cols]
        result = train_and_evaluate(X_ablated, y, groups, f"WITHOUT {domain_name}")
        result["features_removed"] = len(cols_to_remove)
        result["features_remaining"] = len(remaining_cols)
        result["auroc_drop"] = round(full_result["auroc"] - result["auroc"], 4)

        ablation_results["domain_exclusion"][domain_name] = result

    # ── 3. Domain-only models (each domain in isolation) ─────────────────────
    ablation_results["domain_only"] = {}

    for i, (domain_name, domain_cols) in enumerate(DOMAINS.items()):
        cols_available = [c for c in domain_cols if c in all_feature_cols]
        if len(cols_available) < 2:
            log.warning(f"  Too few features for {domain_name}-only model — skipping")
            continue

        log.info(f"\n  Training {domain_name} ONLY ({len(cols_available)} features)...")
        X_only = df[cols_available]
        result = train_and_evaluate(X_only, y, groups, f"{domain_name} ONLY")
        result["n_features_used"] = len(cols_available)

        ablation_results["domain_only"][domain_name] = result

    # ── 4. Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("DOMAIN ABLATION SUMMARY")
    print("=" * 90)

    print(f"\n{'Configuration':<35} {'AUROC':>7} {'95% CI':>18} {'AUPRC':>7} {'Drop':>7} {'Feats':>6}")
    print("-" * 90)
    print(f"{'Full model (all domains)':<35} {full_result['auroc']:>7.4f} "
          f"({full_result['ci_lo']:.4f}–{full_result['ci_hi']:.4f}) "
          f"{full_result['auprc']:>7.4f} {'—':>7} {full_result['n_features']:>6}")

    print("\nLeave-one-domain-out:")
    for domain_name, result in ablation_results["domain_exclusion"].items():
        print(f"  Without {domain_name:<27} {result['auroc']:>7.4f} "
              f"({result['ci_lo']:.4f}–{result['ci_hi']:.4f}) "
              f"{result['auprc']:>7.4f} {result['auroc_drop']:>+7.4f} {result['features_remaining']:>6}")

    print("\nDomain-only models:")
    for domain_name, result in ablation_results["domain_only"].items():
        print(f"  {domain_name:<33} {result['auroc']:>7.4f} "
              f"({result['ci_lo']:.4f}–{result['ci_hi']:.4f}) "
              f"{result['auprc']:>7.4f} {'—':>7} {result['n_features_used']:>6}")

    print("=" * 90)

    # ── Save results ─────────────────────────────────────────────────────────
    ablation_results["description"] = (
        "Domain-level ablation analysis. "
        "'domain_exclusion' shows AUROC when each domain is removed (leave-one-out). "
        "'domain_only' shows AUROC when only that domain is used. "
        "auroc_drop = full_model AUROC - ablated AUROC (positive = domain helps)."
    )

    out_file = out_dir / "domain_ablation_results.json"
    with open(out_file, "w") as f:
        json.dump(ablation_results, f, indent=2)
    log.info(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()

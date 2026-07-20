"""
train.py
End-to-end training pipeline orchestrator for bssentinel.

Usage:
    python train.py --mimic /path/to/mimic_iv --output ./artifacts --horizon 24
    python train.py --mimic /path/to/mimic_iv --output ./artifacts --horizon 24 --dev
                                                                       (^^ fast dev run, 2000 admissions)

After training, the artifacts/ directory contains:
    model_xgb.pkl           — calibrated XGBoost (used by server)
    model_lr.pkl            — calibrated logistic regression (baseline)
    xgb_raw.pkl             — uncalibrated XGBoost (used by SHAP explainer)
    feature_cols.json       — feature column names in order
    feature_importance.csv  — XGBoost gain importance
    shap_global.csv         — global SHAP feature importance
    oof_predictions.parquet — out-of-fold predictions for diagnostics
    metrics.json            — AUROC, AUPRC, calibration, threshold analysis
    figures/                — ROC, PR, calibration, threshold, SHAP plots
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="bssentinel training pipeline")
    p.add_argument("--mimic",   required=True,  help="Path to MIMIC-IV root directory")
    p.add_argument("--output",  required=True,  help="Output directory for model artifacts")
    p.add_argument("--horizon", type=int, default=24,
                   choices=[6, 12, 24, 48],   help="Prediction horizon in hours (default: 24)")
    p.add_argument("--window",  type=int, default=6,
                   help="Vital aggregation window in hours (default: 6)")
    p.add_argument("--dev",     action="store_true",
                   help="Development mode: limit to 10,000 admissions for fast iteration")
    p.add_argument("--cache",   default=None,   help="Cache directory for intermediate data")
    p.add_argument("--no-shap", action="store_true", help="Skip SHAP computation (faster)")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  bssentinel — MIMIC-IV Training Pipeline")
    log.info("=" * 60)
    log.info(f"  MIMIC-IV dir:    {args.mimic}")
    log.info(f"  Output dir:      {output_dir}")
    log.info(f"  Horizon:         {args.horizon}h")
    log.info(f"  Window:          {args.window}h")
    log.info(f"  Dev mode:        {args.dev}")
    log.info("=" * 60)

    # ── Step 1: Load cohort ───────────────────────────────────────────
    log.info("\n[1/5] Loading MIMIC-IV cohort...")
    from data_loader import load_cohort
    cohort = load_cohort(
        mimic_dir=args.mimic,
        horizon_hours=args.horizon,
        window_hours=args.window,
        max_admissions=10_000 if args.dev else None,
        cache_dir=args.cache,
    )

    log.info(f"  Admissions: {len(cohort['admissions']):,}")
    log.info(f"  Windows:    {len(cohort['windows']):,}")
    log.info(f"  Vitals:     {len(cohort['vitals']):,} measurements")
    log.info(f"  Labs:       {len(cohort['labs']):,} measurements")

    # ── Step 2: Build feature matrix ─────────────────────────────────
    log.info("\n[2/5] Building feature matrix...")
    from feature_engineering import build_feature_matrix, get_feature_columns
    df = build_feature_matrix(
        cohort=cohort,
        horizon_hours=args.horizon,
        window_hours=float(args.window),
    )

    feature_cols = get_feature_columns(df)
    log.info(f"  Features: {len(feature_cols)}")
    log.info(f"  Positive rate: {df['label'].mean():.3f}")

    # Drop windows with no vitals at all (no features to learn from)
    vital_cols = ["heart_rate", "systolic_bp", "spo2", "respiratory_rate", "temperature"]
    vital_present = [c for c in vital_cols if c in df.columns]
    df = df.dropna(subset=vital_present, how="all")
    log.info(f"  After dropping all-missing vitals: {len(df):,} windows")

    # ── Step 3: Train models ──────────────────────────────────────────
    log.info("\n[3/5] Training models...")
    from model import train_models
    results = train_models(
        df=df,
        feature_cols=feature_cols,
        output_dir=output_dir,
    )

    # ── Step 4: Evaluate ──────────────────────────────────────────────
    log.info("\n[4/5] Evaluating models...")
    from evaluate import evaluate, print_summary
    metrics = evaluate(
        oof_df=results["oof_df"],
        output_dir=output_dir,
    )
    print_summary(metrics)

    # ── Step 5: SHAP global analysis ─────────────────────────────────
    if not args.no_shap:
        log.info("\n[5/5] Computing global SHAP values...")
        try:
            from explain import compute_shap_global
            X = df[feature_cols]
            compute_shap_global(
                model_path=output_dir / "xgb_raw.pkl",
                X=X,
                feature_cols=feature_cols,
                output_dir=output_dir,
                max_samples=5000,
            )
        except ImportError:
            log.warning("  shap not installed — skipping. Run: pip install shap")
        except Exception as e:
            log.error(f"  SHAP failed: {e}")
    else:
        log.info("\n[5/5] SHAP skipped (--no-shap)")

    # ── Summary ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("  Training complete.")
    log.info(f"  AUROC:  {metrics['auroc']['xgboost']['auc']:.4f}")
    log.info(f"  AUPRC:  {metrics['auprc']['xgboost']:.4f}")
    log.info(f"  Artifacts: {output_dir}/")
    log.info("=" * 60)

    _print_next_steps(output_dir, metrics)


def _print_next_steps(output_dir: Path, metrics: dict):
    auc = metrics["auroc"]["xgboost"]["auc"]
    n2  = metrics["auroc"].get("news2_only", {}).get("auc")
    dl  = metrics.get("delong_xgb_vs_news2", {})

    print("\n  Next steps:")
    print(f"  1. Review metrics.json and figures/ in {output_dir}")
    if n2:
        gain = auc - n2
        sig  = "(p<0.05 ✓)" if dl.get("p") is not None and dl["p"] < 0.05 else "(p≥0.05 — not significant)"
        print(f"  2. AUC gain over NEWS2 alone: +{gain:.4f} {sig}")
    print("  3. Deploy: copy artifacts/ to the VPS and update server/main.py to load the model")
    print("     See: softwares/bssentinel/ml/explain.py → build_server_explain_fn()")
    print("  4. Retrain at 6h, 12h, 48h horizons for multi-horizon support\n")


if __name__ == "__main__":
    main()

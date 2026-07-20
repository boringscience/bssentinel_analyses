"""
additional_comparators.py
Computes qSOFA, MEWS, and Shock Index from existing features as additional
comparators for bssentinel.

Addresses Reviewer 4: "The authors should also consider additional comparators
that can be derived from the available variables, such as MEWS, qSOFA, SIRS,
or shock index variants."

Usage:
    python new_analyses/additional_comparators.py
    python new_analyses/additional_comparators.py --parquet /path/to/feature_matrix.parquet
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def compute_qsofa(df):
    """
    qSOFA (quick SOFA): 0–3 points
      +1 if SBP ≤ 100 mmHg
      +1 if RR ≥ 22 breaths/min
      +1 if GCS < 15

    All three components are available in the feature matrix.
    """
    score = pd.Series(0, index=df.index, dtype=float)

    if "systolic_bp" in df.columns:
        score += (df["systolic_bp"] <= 100).astype(float)
        # NaN SBP → NaN contribution
        score = score.where(df["systolic_bp"].notna(), np.nan)

    if "respiratory_rate" in df.columns:
        rr_component = (df["respiratory_rate"] >= 22).astype(float)
        score = score + rr_component.where(df["respiratory_rate"].notna(), 0)

    if "gcs_total" in df.columns:
        gcs_component = (df["gcs_total"] < 15).astype(float)
        score = score + gcs_component.where(df["gcs_total"].notna(), 0)
    elif all(c in df.columns for c in ["gcs_eye", "gcs_verbal", "gcs_motor"]):
        gcs = df[["gcs_eye", "gcs_verbal", "gcs_motor"]].sum(axis=1, min_count=1)
        gcs_component = (gcs < 15).astype(float)
        score = score + gcs_component.where(gcs.notna(), 0)

    return score


def compute_mews(df):
    """
    Modified Early Warning Score (MEWS): 0–14 points
      SBP:  ≤70 → 3; 71–80 → 2; 81–100 → 1; 101–199 → 0; ≥200 → 2
      HR:   <40 → 2; 41–50 → 1; 51–100 → 0; 101–110 → 1; 111–129 → 2; ≥130 → 3
      RR:   <9 → 2; 9–14 → 0; 15–20 → 1; 21–29 → 2; ≥30 → 3
      Temp: <35 → 2; 35–38.4 → 0; ≥38.5 → 2
      GCS:  15 → 0; 14 → 1; 9–13 → 2; ≤8 → 3
    """
    score = pd.Series(0.0, index=df.index)

    # SBP
    if "systolic_bp" in df.columns:
        sbp = df["systolic_bp"]
        sbp_score = pd.Series(0.0, index=df.index)
        sbp_score = sbp_score.where(~(sbp <= 70), 3)
        sbp_score = sbp_score.where(~((sbp > 70) & (sbp <= 80)), 2)
        sbp_score = sbp_score.where(~((sbp > 80) & (sbp <= 100)), 1)
        sbp_score = sbp_score.where(~(sbp >= 200), 2)
        score += sbp_score.fillna(0)

    # HR
    if "heart_rate" in df.columns:
        hr = df["heart_rate"]
        hr_score = pd.Series(0.0, index=df.index)
        hr_score = hr_score.where(~(hr < 40), 2)
        hr_score = hr_score.where(~((hr >= 40) & (hr <= 50)), 1)
        hr_score = hr_score.where(~((hr >= 101) & (hr <= 110)), 1)
        hr_score = hr_score.where(~((hr >= 111) & (hr <= 129)), 2)
        hr_score = hr_score.where(~(hr >= 130), 3)
        score += hr_score.fillna(0)

    # RR
    if "respiratory_rate" in df.columns:
        rr = df["respiratory_rate"]
        rr_score = pd.Series(0.0, index=df.index)
        rr_score = rr_score.where(~(rr < 9), 2)
        rr_score = rr_score.where(~((rr >= 15) & (rr <= 20)), 1)
        rr_score = rr_score.where(~((rr >= 21) & (rr <= 29)), 2)
        rr_score = rr_score.where(~(rr >= 30), 3)
        score += rr_score.fillna(0)

    # Temperature
    if "temperature" in df.columns:
        temp = df["temperature"]
        temp_score = pd.Series(0.0, index=df.index)
        temp_score = temp_score.where(~(temp < 35), 2)
        temp_score = temp_score.where(~(temp >= 38.5), 2)
        score += temp_score.fillna(0)

    # GCS
    gcs = None
    if "gcs_total" in df.columns:
        gcs = df["gcs_total"]
    elif all(c in df.columns for c in ["gcs_eye", "gcs_verbal", "gcs_motor"]):
        gcs = df[["gcs_eye", "gcs_verbal", "gcs_motor"]].sum(axis=1, min_count=1)

    if gcs is not None:
        gcs_score = pd.Series(0.0, index=df.index)
        gcs_score = gcs_score.where(~(gcs == 14), 1)
        gcs_score = gcs_score.where(~((gcs >= 9) & (gcs <= 13)), 2)
        gcs_score = gcs_score.where(~(gcs <= 8), 3)
        score += gcs_score.fillna(0)

    return score


def compute_shock_index(df):
    """
    Shock Index = HR / SBP
    Normal: 0.5–0.7. Elevated (>0.9) suggests hemodynamic compromise.
    """
    if "heart_rate" in df.columns and "systolic_bp" in df.columns:
        si = df["heart_rate"] / df["systolic_bp"].replace(0, np.nan)
        return si
    return pd.Series(np.nan, index=df.index)


def compute_auroc_with_ci(y, scores, name, n_boot=2000):
    """Compute AUROC with bootstrap CI for a score."""
    valid = ~np.isnan(scores)
    if valid.sum() < 100 or len(np.unique(y[valid])) < 2:
        log.warning(f"  {name}: insufficient valid data ({valid.sum()} valid, "
                    f"{y[valid].sum()} events)")
        return None

    y_v = y[valid]
    s_v = scores[valid]
    auroc = roc_auc_score(y_v, s_v)
    auprc = average_precision_score(y_v, s_v)

    rng = np.random.default_rng(42)
    aucs = []
    n = len(y_v)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_v[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_v[idx], s_v[idx]))
    ci_lo = float(np.percentile(aucs, 2.5))
    ci_hi = float(np.percentile(aucs, 97.5))

    log.info(f"  {name}: AUROC={auroc:.4f} ({ci_lo:.4f}–{ci_hi:.4f}) AUPRC={auprc:.4f} "
             f"n={valid.sum():,} valid")

    return {
        "auroc": round(auroc, 4),
        "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4),
        "auprc": round(auprc, 4),
        "n_valid": int(valid.sum()),
        "n_events": int(y_v.sum()),
        "pct_valid": round(valid.mean() * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, help="Path to feature_matrix.parquet")
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    out_dir = Path(args.output)

    # Try to load feature matrix or fall back to OOF predictions
    df = None
    oof_path = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "oof_predictions.parquet"

    if args.parquet and Path(args.parquet).exists():
        df = pd.read_parquet(args.parquet)
        log.info(f"Loaded feature matrix: {len(df):,} rows")
        y = df["label"].values
    else:
        # Try cache
        cache_candidates = [
            Path(__file__).resolve().parent.parent / "ml" / "cache" / "feature_matrix.parquet",
        ]
        for p in cache_candidates:
            if p.exists():
                df = pd.read_parquet(p)
                log.info(f"Loaded cached feature matrix: {len(df):,} rows from {p}")
                y = df["label"].values
                break

    if df is None:
        log.info("No feature matrix found. Using OOF predictions (limited — no raw vitals for scoring).")
        oof = pd.read_parquet(oof_path)
        df = oof
        y = oof["label"].values

    # ── Compute scores ───────────────────────────────────────────────────────
    results = {}

    # NEWS2 (modified — already computed)
    if "news2_total" in df.columns:
        news2 = df["news2_total"].values
        log.info("\n[1] Modified NEWS2 (excluding O2 supplementation):")
        results["mNEWS2"] = compute_auroc_with_ci(y, news2, "mNEWS2")

    # qSOFA
    if "systolic_bp" in df.columns:
        log.info("\n[2] qSOFA:")
        qsofa = compute_qsofa(df).values
        results["qSOFA"] = compute_auroc_with_ci(y, qsofa, "qSOFA")

        # qSOFA >= 2 operating characteristics
        if results["qSOFA"]:
            qsofa_pos = (qsofa >= 2)
            valid = ~np.isnan(qsofa)
            tp = int(np.sum(qsofa_pos[valid] & (y[valid] == 1)))
            fp = int(np.sum(qsofa_pos[valid] & (y[valid] == 0)))
            fn = int(np.sum(~qsofa_pos[valid] & (y[valid] == 1)))
            tn = int(np.sum(~qsofa_pos[valid] & (y[valid] == 0)))
            results["qSOFA"]["at_cutoff_2"] = {
                "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) > 0 else None,
                "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else None,
                "ppv": round(tp / (tp + fp), 4) if (tp + fp) > 0 else None,
                "npv": round(tn / (tn + fn), 4) if (tn + fn) > 0 else None,
            }
    else:
        log.info("\n[2] qSOFA: raw vitals not available in OOF predictions — skipping")

    # MEWS
    if "systolic_bp" in df.columns:
        log.info("\n[3] MEWS:")
        mews = compute_mews(df).values
        results["MEWS"] = compute_auroc_with_ci(y, mews, "MEWS")
    else:
        log.info("\n[3] MEWS: raw vitals not available — skipping")

    # Shock Index
    if "heart_rate" in df.columns and "systolic_bp" in df.columns:
        log.info("\n[4] Shock Index (HR/SBP):")
        si = compute_shock_index(df).values
        results["Shock_Index"] = compute_auroc_with_ci(y, si, "Shock Index")
    else:
        log.info("\n[4] Shock Index: raw vitals not available — skipping")

    # XGBoost (from OOF) for reference
    if "xgb_proba" in df.columns:
        log.info("\n[5] bssentinel XGBoost (reference):")
        results["bssentinel_XGBoost"] = compute_auroc_with_ci(y, df["xgb_proba"].values, "bssentinel")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARATOR SUMMARY")
    print("=" * 80)
    print(f"{'Model':<25} {'AUROC':>7} {'95% CI':>18} {'AUPRC':>7} {'n valid':>10}")
    print("-" * 80)
    for name, r in results.items():
        if r and isinstance(r, dict) and "auroc" in r:
            print(f"{name:<25} {r['auroc']:>7.4f} ({r['ci_lo']:.4f}–{r['ci_hi']:.4f}) "
                  f"{r['auprc']:>7.4f} {r.get('n_valid', 'N/A'):>10,}")
    print("=" * 80)

    # ── Save ─────────────────────────────────────────────────────────────────
    results["description"] = (
        "AUROC comparison of bssentinel with traditional early warning scores. "
        "mNEWS2 = modified NEWS2 (without supplemental O2). "
        "qSOFA = quick SOFA (SBP ≤ 100, RR ≥ 22, GCS < 15). "
        "MEWS = Modified Early Warning Score. "
        "Shock Index = HR / SBP."
    )
    results["manuscript_changes"] = {
        "methods": "Add computation of qSOFA, MEWS, and Shock Index from available features",
        "table_3": "Add rows for qSOFA, MEWS, Shock Index AUROCs",
        "results_text": "Report comparative AUROCs for all scores",
        "NEWS2_labelling": "Rename NEWS2 → mNEWS2 (modified NEWS2, excluding supplemental O2) throughout",
    }

    out_file = out_dir / "additional_comparators_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()

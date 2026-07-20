"""
explain.py
Real SHAP explanations for bssentinel predictions.

Uses the TreeExplainer (exact, fast for XGBoost) to compute per-prediction
Shapley values. Also computes global feature importance via SHAP.

Functions:
  compute_shap_global()   — global SHAP summary for a dataset
  explain_single()        — SHAP values for one patient observation
  shap_to_factors()       — convert SHAP values to the API's top-factors format
  update_server_explain() — replaces the heuristic explanation in server/main.py
                            with real SHAP once a model is trained
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    log.warning("shap not installed — run: pip install shap")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

STYLE = {
    "bg":     "#0a0a0a",
    "surface": "#111111",
    "accent": "#c8ff00",
    "text":   "#e0e0e0",
    "muted":  "#666666",
    "red":    "#f87171",
    "blue":   "#60a5fa",
}

FEATURE_LABELS = {
    "heart_rate":            "Heart Rate",
    "systolic_bp":           "Systolic BP",
    "diastolic_bp":          "Diastolic BP",
    "spo2":                  "SpO₂",
    "respiratory_rate":      "Respiratory Rate",
    "temperature":           "Temperature",
    "heart_rate_trend":      "Heart Rate (trend)",
    "spo2_trend":            "SpO₂ (trend)",
    "respiratory_rate_trend":"RR (trend)",
    "systolic_bp_worst":     "Systolic BP (worst)",
    "spo2_worst":            "SpO₂ (worst)",
    "wbc":                   "WBC",
    "hemoglobin":            "Hemoglobin",
    "platelets":             "Platelets",
    "creatinine":            "Creatinine",
    "bun":                   "BUN",
    "sodium":                "Sodium",
    "potassium":             "Potassium",
    "lactate":               "Lactate",
    "glucose":               "Glucose",
    "bilirubin":             "Bilirubin",
    "inr":                   "INR",
    "crp":                   "CRP",
    "vasopressors":          "Vasopressor Use",
    "antibiotics":           "Antibiotic Use",
    "anticoagulants":        "Anticoagulants",
    "news2_total":           "NEWS2 Total Score",
    "news2_rr":              "NEWS2 — Respiratory Rate",
    "news2_spo2":            "NEWS2 — SpO₂",
    "news2_sbp":             "NEWS2 — Systolic BP",
    "news2_hr":              "NEWS2 — Heart Rate",
    "news2_temp":            "NEWS2 — Temperature",
    "news2_gcs":             "NEWS2 — Consciousness",
    "age":                   "Patient Age",
    "admit_offset_hours":    "Time Since Admission",
    "n_vitals_missing":      "Missing Vitals Count",
    "n_labs_missing":        "Missing Labs Count",
}

FEATURE_UNITS = {
    "heart_rate":         "bpm",
    "systolic_bp":        "mmHg",
    "diastolic_bp":       "mmHg",
    "spo2":               "%",
    "respiratory_rate":   "breaths/min",
    "temperature":        "°C",
    "wbc":                "×10³/µL",
    "hemoglobin":         "g/dL",
    "creatinine":         "mg/dL",
    "lactate":            "mmol/L",
    "glucose":            "mg/dL",
    "bilirubin":          "mg/dL",
    "potassium":          "mEq/L",
    "sodium":             "mEq/L",
    "age":                "yrs",
    "admit_offset_hours": "h",
}


# ── Global SHAP analysis ───────────────────────────────────────────────────────

def compute_shap_global(
    model_path: str | Path,
    X: pd.DataFrame,
    feature_cols: list[str],
    output_dir: str | Path,
    max_samples: int = 5000,
) -> pd.DataFrame:
    """
    Compute global SHAP summary for the trained XGBoost model.

    Args:
        model_path:   Path to xgb_raw.pkl (raw XGBoost, not calibrated wrapper)
        X:            Feature matrix (can be test set or full OOF)
        feature_cols: Feature column names
        output_dir:   Where to save SHAP summary plots + CSV
        max_samples:  Subsample for speed (TreeExplainer is O(n))

    Returns:
        DataFrame with mean |SHAP| per feature, sorted by importance.
    """
    if not HAS_SHAP:
        raise ImportError("shap package required: pip install shap")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_feat = X[feature_cols].copy()
    if len(X_feat) > max_samples:
        X_feat = X_feat.sample(max_samples, random_state=42)

    log.info(f"Computing SHAP values for {len(X_feat):,} samples...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_feat)

    # Global importance: mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_df  = pd.DataFrame({
        "feature":    feature_cols,
        "label":      [FEATURE_LABELS.get(f, f) for f in feature_cols],
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    output_dir = Path(output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(output_dir / "shap_global.csv", index=False)
    log.info(f"SHAP global summary saved to {output_dir / 'shap_global.csv'}")

    if HAS_MPL:
        _plot_shap_summary(shap_df, output_dir)
        _plot_shap_beeswarm(shap_values, X_feat, feature_cols, output_dir)

    return shap_df


def _plot_shap_summary(shap_df: pd.DataFrame, output_dir: Path, top_n: int = 20):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["surface"])

    top = shap_df.head(top_n)
    bars = ax.barh(
        range(len(top)), top["mean_abs_shap"],
        color=STYLE["accent"], alpha=0.85,
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label"], color=STYLE["text"], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|", color=STYLE["text"])
    ax.set_title("Feature Importance (SHAP)", color=STYLE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e1e1e")
    ax.tick_params(colors=STYLE["muted"])
    ax.grid(axis="x", color="#1e1e1e", lw=0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved shap_importance.png")


def _plot_shap_beeswarm(shap_values, X_feat, feature_cols, output_dir, top_n=15):
    """SHAP beeswarm (summary) plot using shap's built-in renderer."""
    try:
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor(STYLE["bg"])
        shap.summary_plot(
            shap_values, X_feat,
            feature_names=[FEATURE_LABELS.get(f, f) for f in feature_cols],
            max_display=top_n,
            show=False,
            plot_type="dot",
        )
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        log.info("  Saved shap_beeswarm.png")
    except Exception as e:
        log.warning(f"Beeswarm plot failed: {e}")


# ── Single-prediction SHAP ─────────────────────────────────────────────────────

def explain_single(
    model_path: str | Path,
    feature_dict: dict,
    feature_cols: list[str],
) -> list[dict]:
    """
    Compute SHAP values for a single patient observation.

    Args:
        model_path:   Path to xgb_raw.pkl
        feature_dict: {feature_name: value}  (NaN for missing)
        feature_cols: All feature names the model expects

    Returns:
        List of dicts sorted by |SHAP|:
        [{feature, label, value, shap_value, direction, explanation}, ...]
    """
    if not HAS_SHAP:
        raise ImportError("shap package required: pip install shap")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    row = {col: feature_dict.get(col, np.nan) for col in feature_cols}
    X   = pd.DataFrame([row])[feature_cols]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]  # shape: (n_features,)

    factors = []
    for i, feat in enumerate(feature_cols):
        sv  = float(shap_values[i])
        val = feature_dict.get(feat, np.nan)
        if abs(sv) < 1e-6:
            continue

        unit    = FEATURE_UNITS.get(feat, "")
        if isinstance(val, bool):
            val_str = "Yes" if val else "No"
        elif val is None or (isinstance(val, float) and np.isnan(val)):
            val_str = "—"
        else:
            val_str = f"{val:.1f} {unit}".strip()

        factors.append({
            "feature":     feat,
            "label":       FEATURE_LABELS.get(feat, feat),
            "value":       val_str,
            "shap_value":  round(sv, 5),
            "contribution": round(abs(sv), 5),
            "direction":   "increases_risk" if sv > 0 else "decreases_risk",
            "explanation": _shap_explanation(feat, val, sv),
        })

    return sorted(factors, key=lambda x: -x["contribution"])


def shap_to_factors(shap_factors: list[dict], top_k: int = 3) -> list[dict]:
    """Convert explain_single() output to the API top_factors format."""
    return [
        {
            "feature":     f["feature"],
            "label":       f["label"],
            "value":       f["value"],
            "contribution": f["contribution"],
            "direction":   f["direction"],
            "explanation": f["explanation"],
        }
        for f in shap_factors[:top_k]
    ]


def _shap_explanation(feature: str, value, shap_val: float) -> str:
    """Generate plain-language explanation from SHAP direction and magnitude."""
    def fmt(v, decimals=1):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:.{decimals}f}"

    direction = "increases" if shap_val > 0 else "reduces"
    magnitude = abs(shap_val)
    strength  = "strongly" if magnitude > 0.1 else "moderately" if magnitude > 0.04 else "slightly"

    explanations = {
        "heart_rate":         f"Heart rate {fmt(value)} bpm {strength} {direction} risk — outside safe range (51–90 bpm).",
        "systolic_bp":        f"Systolic BP {fmt(value)} mmHg {strength} {direction} risk — haemodynamic instability.",
        "spo2":               f"SpO₂ {fmt(value)}% {strength} {direction} risk — blood oxygen saturation.",
        "respiratory_rate":   f"Respiratory rate {fmt(value)} br/min {strength} {direction} risk — normal range 12–20.",
        "temperature":        f"Temperature {fmt(value)}°C {strength} {direction} risk — fever or hypothermia.",
        "lactate":            f"Lactate {fmt(value)} mmol/L {strength} {direction} risk — tissue hypoperfusion marker.",
        "creatinine":         f"Creatinine {fmt(value)} mg/dL {strength} {direction} risk — renal function.",
        "wbc":                f"WBC {fmt(value)} ×10³/µL {strength} {direction} risk — immune/inflammatory response.",
        "inr":                f"INR {fmt(value)} {strength} {direction} risk — coagulation status.",
        "potassium":          f"Potassium {fmt(value)} mEq/L {strength} {direction} risk — electrolyte balance.",
        "sodium":             f"Sodium {fmt(value)} mEq/L {strength} {direction} risk — electrolyte balance.",
        "vasopressors":       f"Vasopressor use {strength} {direction} risk — indicates circulatory failure.",
        "antibiotics":        f"Antibiotic use {strength} {direction} risk — suggests active infection or sepsis.",
        "news2_total":        f"NEWS2 score {fmt(value, 0)} {strength} {direction} deterioration risk.",
        "age":                f"Age {fmt(value, 0)} years {strength} {direction} risk — vulnerability factor.",
        "admit_offset_hours": f"{fmt(value, 0)}h since admission {strength} {direction} risk — temporal pattern.",
        "n_vitals_missing":   f"Missing {fmt(value, 0)} vital signs {strength} {direction} risk — sparse monitoring.",
        "n_labs_missing":     f"Missing {fmt(value, 0)} lab values {strength} {direction} risk — incomplete workup.",
    }
    return explanations.get(
        feature,
        f"{FEATURE_LABELS.get(feature, feature)} {strength} {direction}s deterioration risk."
    )


# ── Integration with server ────────────────────────────────────────────────────

def build_server_explain_fn(
    model_dir: str | Path,
    feature_cols: list[str],
):
    """
    Returns a predict+explain function suitable for use in server/main.py.
    This replaces the heuristic run_risk_engine() once a model is trained.

    Usage in server/main.py:
        from ml.explain import build_server_explain_fn
        _predict_and_explain = build_server_explain_fn(MODEL_DIR, FEATURE_COLS)

        # In run_risk_engine():
        result = _predict_and_explain(feature_dict, horizon_hours)
    """
    model_dir = Path(model_dir)
    xgb_cal_path = model_dir / "model_xgb.pkl"
    xgb_raw_path = model_dir / "xgb_raw.pkl"

    with open(xgb_cal_path, "rb") as f:
        calibrated_model = pickle.load(f)

    if HAS_SHAP:
        with open(xgb_raw_path, "rb") as f:
            raw_model = pickle.load(f)
        explainer = shap.TreeExplainer(raw_model)
    else:
        explainer = None

    def predict_and_explain(feature_dict: dict, horizon_hours: int = 24) -> dict:
        row = {col: feature_dict.get(col, np.nan) for col in feature_cols}
        X   = pd.DataFrame([row])[feature_cols]

        risk = float(calibrated_model.predict_proba(X)[0, 1])

        # SHAP factors
        if explainer is not None:
            sv = explainer.shap_values(X)[0]
            factors = []
            for i, feat in enumerate(feature_cols):
                if abs(sv[i]) < 1e-6:
                    continue
                val = feature_dict.get(feat, np.nan)
                factors.append({
                    "feature":      feat,
                    "label":        FEATURE_LABELS.get(feat, feat),
                    "value":        _fmt_value(val, feat),
                    "contribution": round(abs(float(sv[i])), 5),
                    "direction":    "increases_risk" if sv[i] > 0 else "decreases_risk",
                    "explanation":  _shap_explanation(feat, val, float(sv[i])),
                })
            factors.sort(key=lambda x: -x["contribution"])
        else:
            factors = []

        return {"risk_score": round(risk, 4), "factors": factors}

    return predict_and_explain


def _fmt_value(val, feat: str) -> str:
    unit = FEATURE_UNITS.get(feat, "")
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.1f} {unit}".strip()

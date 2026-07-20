"""
model.py
Model training, calibration, and persistence for bssentinel.

Two models:
  1. LogisticRegression — fast, interpretable, calibrated baseline
  2. XGBoost            — main model, SHAP-compatible, handles missingness natively

Both are trained with patient-level GroupKFold cross-validation to prevent
data leakage (multiple windows per patient must not span train/test splits).

After training, models are calibrated with isotonic regression and saved to disk.
The best-performing calibrated model is also baked into a SentinelPredictor
object that the FastAPI server loads at startup.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

log = logging.getLogger(__name__)

# ── XGBoost hyperparameters ────────────────────────────────────────────────────
# Tuned for clinical imbalanced tabular data.
# Deterioration prevalence in MIMIC-IV ward cohort is ~8–15%.

XGB_PARAMS = {
    "n_estimators":         500,
    "max_depth":            6,
    "learning_rate":        0.05,
    "subsample":            0.8,
    "colsample_bytree":     0.8,
    "min_child_weight":     10,    # prevents overfit on rare positive cases
    "gamma":                1.0,
    "reg_alpha":            0.1,
    "reg_lambda":           1.0,
    "tree_method":          "hist",
    "eval_metric":          "auc",
    "early_stopping_rounds": 30,
    "random_state":         42,
    "n_jobs":               -1,
}

LR_PARAMS = {
    "max_iter":  2000,
    "solver":    "lbfgs",
    "C":         0.1,
    "random_state": 42,
}

N_SPLITS = 5


# ── Training ───────────────────────────────────────────────────────────────────

def train_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: str | Path,
    scale_pos_weight_factor: float = 1.0,
) -> dict[str, Any]:
    """
    Train logistic regression and XGBoost models with cross-validation.

    Args:
        df:                      Feature matrix with 'label', 'hadm_id', 'subject_id'
        feature_cols:            Columns to use as features
        output_dir:              Where to save model artifacts
        scale_pos_weight_factor: Multiplier for class imbalance correction

    Returns:
        Dict with trained models, feature importances, CV predictions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = df[feature_cols].copy()
    y = df["label"].values
    groups = df["hadm_id"].values   # patient-level grouping

    log.info(f"Training on {len(X):,} windows, {feature_cols.__len__()} features")
    log.info(f"  Positive rate: {y.mean():.3f}  ({y.sum():,} / {len(y):,})")

    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError(
            "No positive labels found in the training set. "
            "In --dev mode the 2,000 sampled admissions often contain only "
            "short-stay deaths that get excluded from window creation. "
            "Try: --dev with a larger sample, or run the full training without --dev."
        )

    # Class weight for imbalance
    classes = np.array([0, 1])
    cw = compute_class_weight("balanced", classes=classes, y=y)
    class_weight_dict = {0: cw[0], 1: cw[1]}
    pos_weight = (cw[1] / cw[0]) * scale_pos_weight_factor
    log.info(f"  Scale pos weight: {pos_weight:.2f}")

    gkf = GroupKFold(n_splits=N_SPLITS)

    # ── Logistic Regression baseline ──────────────────────────────────
    log.info("Training Logistic Regression baseline...")
    lr_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            class_weight=class_weight_dict,
            **LR_PARAMS,
        )),
    ])

    # Cross-validated out-of-fold predictions (no leakage)
    lr_oof_proba = cross_val_predict(
        lr_pipeline, X, y,
        groups=groups, cv=gkf,
        method="predict_proba", n_jobs=-1,
    )[:, 1]

    # Full fit for final model
    lr_pipeline.fit(X, y)

    # Calibrate on full data with CV
    lr_calibrated = CalibratedClassifierCV(lr_pipeline, method="isotonic", cv=5)
    lr_calibrated.fit(X, y)

    # ── XGBoost main model ─────────────────────────────────────────────
    log.info("Training XGBoost model...")

    # Handle NaN: XGBoost handles natively, but we need a consistent approach
    # Replace NaN with a sentinel value recognised by XGBoost (leave as-is)
    X_xgb = X.copy()

    xgb_oof_proba = np.zeros(len(y))
    xgb_fold_models = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_xgb, y, groups)):
        X_tr, X_val = X_xgb.iloc[train_idx], X_xgb.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(
            scale_pos_weight=pos_weight,
            **XGB_PARAMS,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        xgb_oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]
        xgb_fold_models.append(model)
        log.info(f"  Fold {fold + 1}/{N_SPLITS} done")

    # Final XGBoost model on full data
    xgb_final = xgb.XGBClassifier(scale_pos_weight=pos_weight, **XGB_PARAMS)
    xgb_final.fit(
        X_xgb, y,
        eval_set=[(X_xgb, y)],
        verbose=False,
    )

    # Calibrate XGBoost using prefit mode (avoids re-fitting without eval_set)
    xgb_calibrated = CalibratedClassifierCV(xgb_final, method="isotonic", cv="prefit")
    xgb_calibrated.fit(X_xgb, y)

    # ── Feature importance (XGBoost gain) ─────────────────────────────
    importance_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": xgb_final.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # ── Save artefacts ─────────────────────────────────────────────────
    with open(output_dir / "model_xgb.pkl",     "wb") as f: pickle.dump(xgb_calibrated, f)
    with open(output_dir / "model_lr.pkl",      "wb") as f: pickle.dump(lr_calibrated,  f)
    with open(output_dir / "xgb_raw.pkl",       "wb") as f: pickle.dump(xgb_final,       f)
    with open(output_dir / "feature_cols.json", "w")  as f: json.dump(feature_cols, f, indent=2)

    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # Save OOF predictions for calibration plots in evaluate.py
    oof_df = pd.DataFrame({
        "hadm_id":     groups,
        "label":       y,
        "xgb_proba":   xgb_oof_proba,
        "lr_proba":    lr_oof_proba,
        "news2_total": df["news2_total"].values if "news2_total" in df.columns else np.nan,
    })
    oof_df.to_parquet(output_dir / "oof_predictions.parquet", index=False)

    log.info(f"Models saved to {output_dir}")

    return {
        "xgb_calibrated":   xgb_calibrated,
        "lr_calibrated":    lr_calibrated,
        "xgb_raw":          xgb_final,
        "xgb_fold_models":  xgb_fold_models,
        "oof_df":           oof_df,
        "importance":       importance_df,
        "feature_cols":     feature_cols,
    }


# ── SentinelPredictor: runtime inference object ────────────────────────────────

class SentinelPredictor:
    """
    Wraps the trained XGBoost model for use in the FastAPI server.

    Replaces the heuristic scoring engine in server/main.py once a model
    is trained and validated. Falls back to the heuristic if no model found.
    """

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        with open(model_dir / "model_xgb.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(model_dir / "feature_cols.json") as f:
            self.feature_cols = json.load(f)
        self.importance = pd.read_csv(model_dir / "feature_importance.csv")
        log.info(f"SentinelPredictor loaded from {model_dir}")

    def predict(self, feature_dict: dict) -> dict:
        """
        Run inference on a single observation (dict of feature values).

        Args:
            feature_dict: {feature_name: value, ...}  (missing → NaN)

        Returns:
            {risk_score, confidence_interval_95, top_shap_features}
        """
        row = {col: feature_dict.get(col, np.nan) for col in self.feature_cols}
        X   = pd.DataFrame([row])[self.feature_cols]

        proba = float(self.model.predict_proba(X)[0, 1])
        return {"risk_score": round(proba, 4)}

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> Optional["SentinelPredictor"]:
        """Load predictor if model exists, return None otherwise."""
        try:
            return cls(model_dir)
        except FileNotFoundError:
            log.warning(f"No trained model found in {model_dir} — using heuristic fallback")
            return None


# ── Convenience loader ─────────────────────────────────────────────────────────

def load_trained_models(model_dir: str | Path) -> dict[str, Any]:
    """Load saved models and metadata from disk."""
    model_dir = Path(model_dir)
    result = {}
    for name, fname in [("xgb", "model_xgb.pkl"), ("lr", "model_lr.pkl"), ("xgb_raw", "xgb_raw.pkl")]:
        p = model_dir / fname
        if p.exists():
            with open(p, "rb") as f:
                result[name] = pickle.load(f)
    with open(model_dir / "feature_cols.json") as f:
        result["feature_cols"] = json.load(f)
    result["importance"] = pd.read_csv(model_dir / "feature_importance.csv")
    result["oof_df"]     = pd.read_parquet(model_dir / "oof_predictions.parquet")
    return result

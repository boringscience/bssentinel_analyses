"""
compute_feature_importance.py

Regenerates the XGBoost gain importances used by Figure 4.

The original ml/artifacts/feature_importance.csv is no longer available. The
published values are reproduced by a SINGLE model fitted to the whole dataset
(not by averaging the five cross-validation folds), using XGBoost's "gain"
importance normalised to sum to 1:

    published   this script
    lactate_missing        5.8   5.77
    spo2_worst             5.0   4.96
    gcs_verbal_worst       4.4   4.36
    gcs_motor_worst        3.9   3.92
    news2_gcs              3.7   3.65

Averaging gain across the CV folds instead gives a materially different ranking
(lactate_missing 4.0%, fourth rather than first), so the distinction matters and
is recorded here rather than left implicit. These rankings describe within-model
attribution for one fit and are reported as exploratory; the domain ablation in
Table 5 is the analysis that speaks to incremental contribution.

Output: new_analyses/feature_importance.csv
"""

import json
import logging
from pathlib import Path

import pandas as pd
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
OUT = HERE / "feature_importance.csv"

# Same hyperparameters as the cross-validated model, minus early stopping, which
# needs a held-out fold and does not apply to a single whole-dataset fit.
XGB_PARAMS = {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10,
    "gamma": 1.0, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "tree_method": "hist", "eval_metric": "auc",
    "random_state": 42, "n_jobs": -1,
}
IMPORTANCE_TYPE = "gain"


def main():
    df = pd.read_parquet(HERE / "feature_matrix.parquet")
    feats = json.loads((HERE / "feature_cols.json").read_text())
    X, y = df[feats], df["label"].values.astype(int)

    pos_weight = (len(y) - y.sum()) / y.sum()
    log.info(f"Fitting single model on all {len(y):,} windows "
             f"(scale_pos_weight={pos_weight:.1f})")
    model = xgb.XGBClassifier(scale_pos_weight=pos_weight, **XGB_PARAMS)
    model.fit(X, y, verbose=False)

    score = model.get_booster().get_score(importance_type=IMPORTANCE_TYPE)
    # get_score omits features never chosen for a split; those are zero-importance.
    gain = pd.Series({f: score.get(f, 0.0) for f in feats})
    gain = (gain / gain.sum()).sort_values(ascending=False)

    out = gain.reset_index()
    out.columns = ["feature", "gain_normalised"]
    out["gain_pct"] = (out["gain_normalised"] * 100).round(2)
    out.to_csv(OUT, index=False)

    log.info(f"Wrote {OUT}")
    print(out.head(16).to_string(index=False))


if __name__ == "__main__":
    main()

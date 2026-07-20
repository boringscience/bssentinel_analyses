"""
feature_engineering.py
Transforms raw MIMIC-IV data into a flat feature matrix for each observation window.

For each observation window at time t, aggregates:
  - Most recent vital sign within the preceding window_hours (+ trend)
  - Most recent lab value within the preceding 24h (labs change slowly)
  - Medication flags active at time t
  - NEWS2 subscore components (from vitals)
  - Missingness indicator flags (critical for clinical ML)
  - Temporal context features

Output: one row per observation window, suitable for XGBoost/sklearn.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Feature names ──────────────────────────────────────────────────────────────

VITAL_FEATURES = [
    "heart_rate", "systolic_bp", "diastolic_bp",
    "spo2", "respiratory_rate", "temperature",
]

LAB_FEATURES = [
    "wbc", "hemoglobin", "platelets", "creatinine",
    "bun", "sodium", "potassium", "lactate",
    "glucose", "bilirubin", "inr", "crp",
]

MED_FEATURES = ["vasopressors", "antibiotics", "anticoagulants"]

GCS_FEATURES = ["gcs_eye", "gcs_verbal", "gcs_motor"]


# ── NEWS2 scoring (vectorised) ─────────────────────────────────────────────────

def news2_respiratory_rate(rr: pd.Series) -> pd.Series:
    """NEWS2 respiratory rate subscore (0–3)."""
    s = pd.Series(np.nan, index=rr.index)
    s = s.where(rr.isna(), 0)
    s = s.where(~((rr >= 9)  & (rr <= 11)), 1)
    s = s.where(~((rr >= 21) & (rr <= 24)), 2)
    s = s.where(~(rr <= 8),  3)
    s = s.where(~(rr >= 25), 3)
    return s


def news2_spo2(spo2: pd.Series) -> pd.Series:
    """NEWS2 SpO2 subscore (Scale 1, 0–3)."""
    s = pd.Series(np.nan, index=spo2.index)
    s = s.where(spo2.isna(), 0)
    s = s.where(~((spo2 >= 94) & (spo2 <= 95)), 1)
    s = s.where(~((spo2 >= 92) & (spo2 <= 93)), 2)
    s = s.where(~(spo2 <= 91), 3)
    return s


def news2_systolic_bp(sbp: pd.Series) -> pd.Series:
    """NEWS2 systolic BP subscore (0–3)."""
    s = pd.Series(np.nan, index=sbp.index)
    s = s.where(sbp.isna(), 0)
    s = s.where(~((sbp >= 101) & (sbp <= 110)), 1)
    s = s.where(~((sbp >= 91)  & (sbp <= 100)), 2)
    s = s.where(~(sbp <= 90),  3)
    s = s.where(~(sbp >= 220), 3)
    return s


def news2_heart_rate(hr: pd.Series) -> pd.Series:
    """NEWS2 heart rate subscore (0–3)."""
    s = pd.Series(np.nan, index=hr.index)
    s = s.where(hr.isna(), 0)
    s = s.where(~((hr >= 41) & (hr <= 50)),   1)
    s = s.where(~((hr >= 91) & (hr <= 110)),  1)
    s = s.where(~((hr >= 111) & (hr <= 130)), 2)
    s = s.where(~(hr <= 40),  3)
    s = s.where(~(hr >= 131), 3)
    return s


def news2_temperature(temp: pd.Series) -> pd.Series:
    """NEWS2 temperature subscore (0–3)."""
    s = pd.Series(np.nan, index=temp.index)
    s = s.where(temp.isna(), 0)
    s = s.where(~((temp >= 35.1) & (temp <= 36.0)), 1)
    s = s.where(~((temp >= 38.1) & (temp <= 39.0)), 1)
    s = s.where(~(temp >= 39.1), 2)
    s = s.where(~(temp <= 35.0), 3)
    return s


def news2_gcs(gcs_total: pd.Series) -> pd.Series:
    """NEWS2 consciousness subscore: 0 if GCS=15, 3 otherwise."""
    s = pd.Series(np.nan, index=gcs_total.index)
    s = s.where(gcs_total.isna(), 0)
    s = s.where(~(gcs_total < 15), 3)
    return s


def compute_news2_total(df: pd.DataFrame) -> pd.Series:
    """
    Compute total NEWS2 score from feature columns in a wide dataframe.
    Expects columns: heart_rate, systolic_bp, spo2, respiratory_rate, temperature,
                     gcs_eye, gcs_verbal, gcs_motor, on_supplemental_o2 (optional).
    Returns: Series of NEWS2 total scores (NaN where all vitals missing).
    """
    subscores = pd.DataFrame(index=df.index)

    if "respiratory_rate" in df.columns:
        subscores["news2_rr"]   = news2_respiratory_rate(df["respiratory_rate"])
    if "spo2" in df.columns:
        subscores["news2_spo2"] = news2_spo2(df["spo2"])
    if "systolic_bp" in df.columns:
        subscores["news2_sbp"]  = news2_systolic_bp(df["systolic_bp"])
    if "heart_rate" in df.columns:
        subscores["news2_hr"]   = news2_heart_rate(df["heart_rate"])
    if "temperature" in df.columns:
        subscores["news2_temp"] = news2_temperature(df["temperature"])

    # GCS → consciousness subscore
    gcs_cols = [c for c in ["gcs_eye", "gcs_verbal", "gcs_motor"] if c in df.columns]
    if gcs_cols:
        gcs_total = df[gcs_cols].sum(axis=1, min_count=1)
        subscores["news2_gcs"] = news2_gcs(gcs_total)

    # Supplemental O2: 2 points if on O2
    if "on_supplemental_o2" in df.columns:
        subscores["news2_o2"] = df["on_supplemental_o2"].fillna(0) * 2

    # Total = sum of available subscores (at least one required)
    news2_total = subscores.sum(axis=1, min_count=1)
    return news2_total


# ── Aggregation per observation window ────────────────────────────────────────

def _most_recent(
    measurements: pd.DataFrame,
    value_col: str,
    group_col: str,
    time_col: str,
    obs_time: pd.Timestamp,
    lookback_hours: float,
) -> float | None:
    """Return most recent value within lookback window before obs_time."""
    t_min = obs_time - pd.Timedelta(hours=lookback_hours)
    mask = (
        (measurements[time_col] >= t_min) &
        (measurements[time_col] < obs_time)
    )
    sub = measurements[mask]
    if sub.empty:
        return np.nan
    return sub.loc[sub[time_col].idxmax(), value_col]


def aggregate_vitals_for_window(
    vitals_adm: pd.DataFrame,
    obs_time: pd.Timestamp,
    window_hours: float = 6.0,
) -> dict[str, float]:
    """
    For a single observation window, compute:
    - Most recent value per vital
    - Rate of change over window (value_now - value_window_start)
    - Worst value in window (max for HR/RR, min for SpO2/BP)
    """
    t_min = obs_time - pd.Timedelta(hours=window_hours)
    window = vitals_adm[
        (vitals_adm["charttime"] >= t_min) &
        (vitals_adm["charttime"] < obs_time)
    ]

    features: dict[str, float] = {}
    vital_names = vitals_adm["vital"].unique() if not vitals_adm.empty else []

    for vital in VITAL_FEATURES + GCS_FEATURES:
        sub = window[window["vital"] == vital]
        if sub.empty:
            features[vital]              = np.nan
            features[f"{vital}_trend"]   = np.nan
            features[f"{vital}_worst"]   = np.nan
        else:
            sub = sub.sort_values("charttime")
            features[vital] = sub.iloc[-1]["valuenum"]   # most recent

            # Trend: last value minus median of first half of window
            if len(sub) >= 3:
                mid = len(sub) // 2
                features[f"{vital}_trend"] = sub.iloc[-1]["valuenum"] - sub.iloc[:mid]["valuenum"].median()
            else:
                features[f"{vital}_trend"] = np.nan

            # Worst: direction depends on the vital
            if vital in ("spo2", "systolic_bp", "diastolic_bp", "hemoglobin"):
                features[f"{vital}_worst"] = sub["valuenum"].min()
            else:
                features[f"{vital}_worst"] = sub["valuenum"].max()

    return features


def aggregate_labs_for_window(
    labs_adm: pd.DataFrame,
    obs_time: pd.Timestamp,
    lookback_hours: float = 24.0,
) -> dict[str, float]:
    """
    Labs are measured less frequently — look back up to 24h.
    Returns most recent value per lab.
    """
    t_min = obs_time - pd.Timedelta(hours=lookback_hours)
    window = labs_adm[
        (labs_adm["charttime"] >= t_min) &
        (labs_adm["charttime"] < obs_time)
    ]
    features: dict[str, float] = {}
    for lab in LAB_FEATURES:
        sub = window[window["lab"] == lab]
        features[lab] = sub.loc[sub["charttime"].idxmax(), "valuenum"] if not sub.empty else np.nan
    return features


def aggregate_meds_for_window(
    meds_adm: pd.DataFrame,
    obs_time: pd.Timestamp,
) -> dict[str, int]:
    """
    Return 1/0 flags for whether each medication class is active at obs_time.
    """
    active = meds_adm[
        (meds_adm["starttime"] <= obs_time) &
        (meds_adm["stoptime"].isna() | (meds_adm["stoptime"] > obs_time))
    ]
    flags = {}
    for flag in MED_FEATURES:
        flags[flag] = int((active["flag"] == flag).any())
    return flags


def label_window(
    outcomes_adm: pd.DataFrame,
    obs_time: pd.Timestamp,
    horizon_hours: int,
) -> int:
    """
    Return 1 if any deterioration event occurs within horizon_hours of obs_time.
    """
    t_max = obs_time + pd.Timedelta(hours=horizon_hours)
    future_events = outcomes_adm[
        (outcomes_adm["event_time"] > obs_time) &
        (outcomes_adm["event_time"] <= t_max)
    ]
    return int(not future_events.empty)


# ── Build full feature matrix ──────────────────────────────────────────────────

def build_feature_matrix(
    cohort: dict[str, pd.DataFrame],
    horizon_hours: int = 24,
    window_hours: float = 6.0,
    lab_lookback_hours: float = 24.0,
) -> pd.DataFrame:
    """
    Build the complete feature matrix from the loaded cohort.

    Each row = one observation window.
    Columns = raw vitals + labs + meds + NEWS2 subscores + missingness flags + metadata.

    Args:
        cohort:             Output of data_loader.load_cohort()
        horizon_hours:      Prediction horizon for labelling
        window_hours:       Vital aggregation window
        lab_lookback_hours: Lab lookback window (labs measured infrequently)

    Returns:
        DataFrame with feature columns and 'label' column (0/1).
    """
    windows     = cohort["windows"]
    vitals_all  = cohort["vitals"]
    labs_all    = cohort["labs"]
    meds_all    = cohort["medications"]
    outcomes_all = cohort["outcomes"]

    all_rows = []
    hadm_ids = windows["hadm_id"].unique()
    n = len(hadm_ids)

    log.info(f"Building feature matrix for {n:,} admissions, {len(windows):,} windows...")

    for i, hadm_id in enumerate(hadm_ids):
        if i % 500 == 0:
            log.info(f"  Processing admission {i:,}/{n:,}...")

        adm_windows  = windows[windows["hadm_id"] == hadm_id]
        vitals_adm   = vitals_all[vitals_all["hadm_id"] == hadm_id]
        labs_adm     = labs_all[labs_all["hadm_id"] == hadm_id]
        meds_adm     = meds_all[meds_all["hadm_id"] == hadm_id] if not meds_all.empty else pd.DataFrame()
        outcomes_adm = outcomes_all[outcomes_all["hadm_id"] == hadm_id]

        for _, win in adm_windows.iterrows():
            obs_time = win["obs_time"]

            # Aggregate features
            vital_feats = aggregate_vitals_for_window(vitals_adm, obs_time, window_hours)
            lab_feats   = aggregate_labs_for_window(labs_adm, obs_time, lab_lookback_hours)
            med_feats   = aggregate_meds_for_window(meds_adm, obs_time) if not meds_adm.empty else {f: 0 for f in MED_FEATURES}

            # Label
            label = label_window(outcomes_adm, obs_time, horizon_hours)

            row = {
                "hadm_id":            hadm_id,
                "subject_id":         win["subject_id"],
                "obs_time":           obs_time,
                "admit_offset_hours": win["admit_offset_hours"],
                "age":                win["age"],
                "gender":             win["gender"],
                **vital_feats,
                **lab_feats,
                **med_feats,
                "label":              label,
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df = _add_news2_and_missingness(df)
    df = _add_admission_type_features(df, cohort["admissions"])

    log.info(f"Feature matrix: {len(df):,} rows, {len(df.columns):,} columns")
    log.info(f"  Label prevalence: {df['label'].mean():.3f} ({df['label'].sum():,} positive windows)")
    return df


def _add_news2_and_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Add NEWS2 total + subscores + missingness indicator columns."""
    df = df.copy()

    # NEWS2 subscores
    df["news2_rr"]   = news2_respiratory_rate(df.get("respiratory_rate", pd.Series(dtype=float)))
    df["news2_spo2"] = news2_spo2(df.get("spo2", pd.Series(dtype=float)))
    df["news2_sbp"]  = news2_systolic_bp(df.get("systolic_bp", pd.Series(dtype=float)))
    df["news2_hr"]   = news2_heart_rate(df.get("heart_rate", pd.Series(dtype=float)))
    df["news2_temp"] = news2_temperature(df.get("temperature", pd.Series(dtype=float)))

    gcs_cols = [c for c in ["gcs_eye", "gcs_verbal", "gcs_motor"] if c in df.columns]
    if gcs_cols:
        gcs_total = df[gcs_cols].sum(axis=1, min_count=1)
        df["news2_gcs"]   = news2_gcs(gcs_total)
        df["gcs_total"]   = gcs_total
    df["news2_o2"] = df.get("vasopressors", 0) * 0  # placeholder; real O2 flag not in chartevents by default

    news2_cols = [c for c in df.columns if c.startswith("news2_")]
    df["news2_total"] = df[news2_cols].sum(axis=1, min_count=1)

    # Missingness indicator flags (1 = missing, 0 = observed)
    # These are predictive features themselves — missingness is not random in ICU
    for feat in VITAL_FEATURES + LAB_FEATURES:
        if feat in df.columns:
            df[f"{feat}_missing"] = df[feat].isna().astype(int)

    # Count missing vitals / labs
    vital_miss_cols = [f"{v}_missing" for v in VITAL_FEATURES if f"{v}_missing" in df.columns]
    lab_miss_cols   = [f"{l}_missing" for l in LAB_FEATURES   if f"{l}_missing" in df.columns]
    df["n_vitals_missing"] = df[vital_miss_cols].sum(axis=1) if vital_miss_cols else 0
    df["n_labs_missing"]   = df[lab_miss_cols].sum(axis=1)   if lab_miss_cols   else 0

    return df


def _add_admission_type_features(df: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode admission type."""
    adm_type = admissions[["hadm_id", "admission_type"]].drop_duplicates("hadm_id")
    df = df.merge(adm_type, on="hadm_id", how="left", suffixes=("", "_adm"))
    dummies = pd.get_dummies(df["admission_type"], prefix="adm_type", drop_first=False)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=["admission_type"], errors="ignore")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature columns to use for modelling
    (excludes metadata and label).
    """
    exclude = {"hadm_id", "subject_id", "obs_time", "label", "admittime"}
    return [c for c in df.columns if c not in exclude]

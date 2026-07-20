"""
data_loader.py
MIMIC-IV cohort extraction for bssentinel deterioration risk model.

Loads and merges the relevant MIMIC-IV tables to produce a cohort of
patient observation windows with clinical features and outcome labels.

Expected MIMIC-IV directory layout:
    mimic_iv/
        hosp/
            admissions.csv
            patients.csv
            labevents.csv
            d_labitems.csv
            prescriptions.csv
        icu/
            icustays.csv
            chartevents.csv
            inputevents.csv

Usage:
    from data_loader import load_cohort
    cohort = load_cohort("/path/to/mimic_iv", horizon_hours=24)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _csv(directory: Path, stem: str) -> Path:
    """Return the path for a CSV file, preferring .csv.gz if .csv is absent."""
    gz = directory / f"{stem}.csv.gz"
    plain = directory / f"{stem}.csv"
    if gz.exists():
        return gz
    if plain.exists():
        return plain
    raise FileNotFoundError(
        f"Neither {stem}.csv nor {stem}.csv.gz found in {directory}"
    )


# ── MIMIC-IV item IDs ──────────────────────────────────────────────────────────

VITAL_ITEMIDS: dict[str, list[int]] = {
    "heart_rate":       [220045],
    "systolic_bp":      [220179, 220050],   # non-invasive, invasive arterial
    "diastolic_bp":     [220180, 220051],
    "spo2":             [220277],
    "respiratory_rate": [220210],
    "temperature_c":    [223762],
    "temperature_f":    [223761, 678],
    "gcs_eye":          [220739],
    "gcs_verbal":       [223900],
    "gcs_motor":        [223901],
}

LAB_ITEMIDS: dict[str, list[int]] = {
    "wbc":        [51301],
    "hemoglobin": [51222],
    "platelets":  [51265],
    "creatinine": [50912],
    "bun":        [51006],
    "sodium":     [50983],
    "potassium":  [50971],
    "lactate":    [50813],
    "glucose":    [50931],
    "bilirubin":  [50885],
    "inr":        [51237],
    "crp":        [50889],
}

# All vasopressor drug names in MIMIC-IV prescriptions
VASOPRESSOR_DRUGS = [
    "norepinephrine", "epinephrine", "dopamine", "vasopressin",
    "phenylephrine", "dobutamine", "milrinone",
]

ANTIBIOTIC_DRUGS = [
    "vancomycin", "piperacillin", "meropenem", "cefepime", "ceftriaxone",
    "metronidazole", "levofloxacin", "azithromycin", "ciprofloxacin",
    "ampicillin", "clindamycin", "linezolid", "daptomycin",
]

ANTICOAGULANT_DRUGS = [
    "heparin", "enoxaparin", "warfarin", "rivaroxaban",
    "apixaban", "dabigatran", "fondaparinux",
]

# Vasopressor ICU inputevent item IDs (backup)
VASOPRESSOR_ITEMIDS = [221906, 221289, 221662, 222315, 221749]

# Intubation procedure event item IDs
INTUBATION_ITEMIDS = [224385, 225794]


# ── Cohort definition ──────────────────────────────────────────────────────────

def load_admissions(hosp_dir: Path) -> pd.DataFrame:
    """
    Load and filter admissions.

    Inclusion: adults ≥18, has a recorded admit/discharge time.
    Returns one row per hospital admission with age and mortality flag.
    """
    log.info("Loading admissions...")
    adm = pd.read_csv(
        _csv(hosp_dir, "admissions"),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                 "deathtime", "admission_type", "hospital_expire_flag"],
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    pts = pd.read_csv(
        _csv(hosp_dir, "patients"),
        usecols=["subject_id", "anchor_age", "gender"],
    )
    df = adm.merge(pts, on="subject_id", how="left")
    df = df[df["anchor_age"] >= 18].copy()
    df = df.dropna(subset=["admittime", "dischtime"])
    df["los_hours"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 3600
    df = df[df["los_hours"] >= 6]   # need at least 6h of data
    log.info(f"  {len(df):,} admissions after filtering")
    return df


def load_icustays(icu_dir: Path) -> pd.DataFrame:
    """Load ICU stay records with intime/outtime. Returns empty df if file is absent."""
    log.info("Loading ICU stays...")
    try:
        icu = pd.read_csv(
            _csv(icu_dir, "icustays"),
            usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "first_careunit"],
            parse_dates=["intime", "outtime"],
        )
        log.info(f"  {len(icu):,} ICU stays")
        return icu
    except FileNotFoundError:
        log.warning("  icustays not found — ICU transfer outcomes will be excluded")
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "first_careunit"])


def load_vitals(icu_dir: Path, hadm_ids: set) -> pd.DataFrame:
    """
    Load chartevents for the specified admissions.
    chartevents is large (~3-4 GB) — reads in chunks and filters.
    Returns long-format dataframe: hadm_id, charttime, vital, value.
    """
    log.info("Loading chartevents (this may take a few minutes)...")

    # Build reverse lookup: itemid → vital name
    itemid_to_vital: dict[int, str] = {}
    for name, ids in VITAL_ITEMIDS.items():
        for iid in ids:
            itemid_to_vital[iid] = name
    all_itemids = set(itemid_to_vital.keys())

    chunks = []
    chunk_size = 500_000

    reader = pd.read_csv(
        _csv(icu_dir, "chartevents"),
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
        chunksize=chunk_size,
        low_memory=False,
    )
    while True:
        try:
            chunk = next(reader)
        except StopIteration:
            break
        except EOFError:
            log.warning("  chartevents.csv.gz appears truncated — using data loaded so far")
            break
        chunk = chunk[
            chunk["hadm_id"].isin(hadm_ids) &
            chunk["itemid"].isin(all_itemids) &
            chunk["valuenum"].notna()
        ].copy()
        if not chunk.empty:
            chunk["vital"] = chunk["itemid"].map(itemid_to_vital)
            chunks.append(chunk[["hadm_id", "charttime", "vital", "valuenum"]])

    if not chunks:
        return pd.DataFrame(columns=["hadm_id", "charttime", "vital", "valuenum"])

    df = pd.concat(chunks, ignore_index=True)
    df = _clean_vitals(df)
    log.info(f"  {len(df):,} vital measurements loaded")
    return df


def _clean_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """Plausibility filters and unit conversions."""
    # Convert temperature_f → temperature_c
    mask_f = df["vital"] == "temperature_f"
    df.loc[mask_f, "valuenum"] = (df.loc[mask_f, "valuenum"] - 32) / 1.8
    df.loc[mask_f, "vital"] = "temperature_c"

    # Rename to canonical name
    df["vital"] = df["vital"].str.replace("_c", "", regex=False)
    df.loc[df["vital"] == "temperature", "vital"] = "temperature"

    # Plausibility ranges
    PLAUSIBLE = {
        "heart_rate":       (10,  300),
        "systolic_bp":      (40,  300),
        "diastolic_bp":     (10,  250),
        "spo2":             (50,  100),
        "respiratory_rate": (2,   80),
        "temperature":      (25,  45),
        "gcs_eye":          (1,   4),
        "gcs_verbal":       (1,   5),
        "gcs_motor":        (1,   6),
    }
    mask = pd.Series(True, index=df.index)
    for vital, (lo, hi) in PLAUSIBLE.items():
        m = df["vital"] == vital
        mask &= ~(m & ((df["valuenum"] < lo) | (df["valuenum"] > hi)))
    return df[mask].copy()


def load_labs(hosp_dir: Path, hadm_ids: set) -> pd.DataFrame:
    """
    Load labevents for the specified admissions.
    Returns long-format: hadm_id, charttime, lab, value.
    """
    log.info("Loading labevents...")
    itemid_to_lab: dict[int, str] = {}
    for name, ids in LAB_ITEMIDS.items():
        for iid in ids:
            itemid_to_lab[iid] = name
    all_itemids = set(itemid_to_lab.keys())

    chunks = []
    reader = pd.read_csv(
        _csv(hosp_dir, "labevents"),
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
        chunksize=500_000,
        low_memory=False,
    )
    while True:
        try:
            chunk = next(reader)
        except StopIteration:
            break
        except EOFError:
            log.warning("  labevents.csv.gz appears truncated — using data loaded so far")
            break
        chunk = chunk[
            chunk["hadm_id"].isin(hadm_ids) &
            chunk["itemid"].isin(all_itemids) &
            chunk["valuenum"].notna()
        ].copy()
        if not chunk.empty:
            chunk["lab"] = chunk["itemid"].map(itemid_to_lab)
            chunks.append(chunk[["hadm_id", "charttime", "lab", "valuenum"]])

    if not chunks:
        return pd.DataFrame(columns=["hadm_id", "charttime", "lab", "valuenum"])

    df = pd.concat(chunks, ignore_index=True)
    log.info(f"  {len(df):,} lab measurements loaded")
    return df


def load_medications(hosp_dir: Path, hadm_ids: set) -> pd.DataFrame:
    """
    Load prescriptions and return per-admission flags:
    vasopressors, antibiotics, anticoagulants (with starttime).
    """
    log.info("Loading prescriptions...")
    rx = pd.read_csv(
        _csv(hosp_dir, "prescriptions"),
        usecols=["hadm_id", "drug", "starttime", "stoptime"],
        parse_dates=["starttime", "stoptime"],
        low_memory=False,
    )
    rx = rx[rx["hadm_id"].isin(hadm_ids) & rx["drug"].notna()].copy()
    rx["drug_lower"] = rx["drug"].str.lower()

    def flag(keywords: list[str]) -> pd.Series:
        mask = rx["drug_lower"].str.contains("|".join(keywords), na=False)
        return rx[mask][["hadm_id", "starttime", "stoptime"]].copy()

    vaso  = flag(VASOPRESSOR_DRUGS);   vaso["flag"]  = "vasopressors"
    abx   = flag(ANTIBIOTIC_DRUGS);    abx["flag"]   = "antibiotics"
    anticoag = flag(ANTICOAGULANT_DRUGS); anticoag["flag"] = "anticoagulants"

    meds = pd.concat([vaso, abx, anticoag], ignore_index=True)
    log.info(f"  {len(meds):,} medication records loaded")
    return meds


# ── Outcome labelling ──────────────────────────────────────────────────────────

def build_outcomes(
    admissions: pd.DataFrame,
    icustays: pd.DataFrame,
    icu_dir: Path,
    hadm_ids: set,
) -> pd.DataFrame:
    """
    Build per-admission outcome events (with timestamps).

    Returns: hadm_id, event_type, event_time
      event_type: 'icu_transfer' | 'intubation' | 'mortality'
    """
    events = []

    # 1. ICU transfer (for ward patients)
    icu_first = (
        icustays[icustays["hadm_id"].isin(hadm_ids)]
        .sort_values("intime")
        .groupby("hadm_id")
        .first()
        .reset_index()[["hadm_id", "intime"]]
        .rename(columns={"intime": "event_time"})
    )
    icu_first["event_type"] = "icu_transfer"
    events.append(icu_first)

    # 2. Mechanical ventilation (intubation events)
    log.info("Loading procedure events for intubation...")
    try:
        proc = pd.read_csv(
            _csv(icu_dir, "procedureevents"),
            usecols=["hadm_id", "itemid", "starttime"],
            parse_dates=["starttime"],
            low_memory=False,
        )
        intub = proc[
            proc["hadm_id"].isin(hadm_ids) &
            proc["itemid"].isin(INTUBATION_ITEMIDS)
        ][["hadm_id", "starttime"]].rename(columns={"starttime": "event_time"})
        intub["event_type"] = "intubation"
        events.append(intub)
    except FileNotFoundError:
        log.warning("procedureevents.csv not found — skipping intubation events")

    # 3. In-hospital mortality
    mort = admissions[
        admissions["hadm_id"].isin(hadm_ids) &
        (admissions["hospital_expire_flag"] == 1)
    ][["hadm_id", "dischtime"]].rename(columns={"dischtime": "event_time"}).copy()
    mort["event_type"] = "mortality"
    events.append(mort)

    outcome_df = pd.concat(events, ignore_index=True)
    outcome_df = outcome_df.dropna(subset=["event_time"])
    log.info(f"  {len(outcome_df):,} outcome events across {outcome_df['hadm_id'].nunique():,} admissions")
    return outcome_df


# ── Sliding observation window ─────────────────────────────────────────────────

def build_observation_windows(
    admissions: pd.DataFrame,
    window_hours: int = 6,
    horizon_hours: int = 24,
    min_obs_hours: int = 1,
) -> pd.DataFrame:
    """
    Create observation windows at regular intervals for each admission.

    Each row represents a point in time during an admission.
    Features will be aggregated from the preceding window_hours.
    Labels will be derived from events in the following horizon_hours.

    Returns: hadm_id, obs_time, admit_offset_hours
    """
    rows = []
    for _, row in admissions.iterrows():
        t = row["admittime"] + pd.Timedelta(hours=min_obs_hours)
        t_max = row["dischtime"] - pd.Timedelta(hours=horizon_hours)
        if t_max <= t:
            continue
        while t <= t_max:
            rows.append({
                "hadm_id":           row["hadm_id"],
                "subject_id":        row["subject_id"],
                "obs_time":          t,
                "admittime":         row["admittime"],
                "admit_offset_hours": (t - row["admittime"]).total_seconds() / 3600,
                "age":               row["anchor_age"],
                "gender":            1 if row.get("gender") == "M" else 0,
                "admission_type":    row["admission_type"],
            })
            t += pd.Timedelta(hours=window_hours)

    df = pd.DataFrame(rows)
    log.info(f"  {len(df):,} observation windows for {df['hadm_id'].nunique():,} admissions")
    return df


# ── Main cohort loader ─────────────────────────────────────────────────────────

def load_cohort(
    mimic_dir: str | Path,
    horizon_hours: int = 24,
    window_hours: int = 6,
    max_admissions: Optional[int] = None,
    cache_dir: Optional[str | Path] = None,
) -> dict[str, pd.DataFrame]:
    """
    Full MIMIC-IV cohort extraction pipeline.

    Args:
        mimic_dir:       Path to MIMIC-IV root (contains hosp/ and icu/)
        horizon_hours:   Prediction horizon (default 24h)
        window_hours:    Feature aggregation window (default 6h)
        max_admissions:  Limit for faster iteration during development
        cache_dir:       If set, save/load intermediate results here

    Returns dict with keys:
        'windows'     — observation windows (features go here)
        'vitals'      — long-format vital measurements
        'labs'        — long-format lab measurements
        'medications' — long-format medication intervals
        'outcomes'    — outcome events with timestamps
        'admissions'  — admission metadata
    """
    mimic_dir = Path(mimic_dir)
    hosp_dir  = mimic_dir / "hosp"
    icu_dir   = mimic_dir / "icu"

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"cohort_h{horizon_hours}_w{window_hours}.pkl"
        if cache_file.exists():
            log.info(f"Loading cached cohort from {cache_file}")
            return pd.read_pickle(cache_file)

    # Step 1: admissions + ICU stays
    admissions = load_admissions(hosp_dir)
    if max_admissions:
        admissions = admissions.head(max_admissions)

    icustays = load_icustays(icu_dir)
    hadm_ids = set(admissions["hadm_id"].unique())

    # Step 2: clinical data
    vitals      = load_vitals(icu_dir, hadm_ids)
    labs        = load_labs(hosp_dir, hadm_ids)
    medications = load_medications(hosp_dir, hadm_ids)

    # Step 3: outcomes
    outcomes = build_outcomes(admissions, icustays, icu_dir, hadm_ids)

    # Step 4: observation windows
    windows = build_observation_windows(admissions, window_hours, horizon_hours)

    result = {
        "windows":     windows,
        "vitals":      vitals,
        "labs":        labs,
        "medications": medications,
        "outcomes":    outcomes,
        "admissions":  admissions,
    }

    if cache_dir:
        pd.to_pickle(result, cache_file)
        log.info(f"Cohort cached to {cache_file}")

    return result

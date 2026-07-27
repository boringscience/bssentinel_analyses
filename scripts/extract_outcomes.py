"""
extract_outcomes.py
Pull the per-admission outcome events (with event_type) out of the cached cohort
pickle so downstream sensitivity analyses can build component-restricted labels
without reloading the 3 GB cohort.

Output: new_analyses/outcome_events.parquet  (hadm_id, event_type, event_time)
"""

import logging
import pickle
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

here = Path(__file__).resolve().parent
cohort_path = here.parent / "ml" / "cache" / "cohort_h24_w6.pkl"
out_path = here / "outcome_events.parquet"

if out_path.exists():
    log.info(f"Already exists: {out_path}")
    raise SystemExit(0)

log.info(f"Loading cohort from {cohort_path} (~3 GB, be patient)...")
with open(cohort_path, "rb") as f:
    cohort = pickle.load(f)

outcomes = cohort["outcomes"][["hadm_id", "event_type", "event_time"]].copy()
log.info(f"  {len(outcomes):,} outcome events")
log.info(f"  event_type counts:\n{outcomes['event_type'].value_counts()}")

outcomes.to_parquet(out_path, index=False)
log.info(f"Saved {out_path}")

"""
alert_strategies.py

Completes Supplementary Table S5 by evaluating the three candidate
de-duplicated alert definitions discussed in the Discussion, alongside the
window-level baseline, at the model operating threshold.

  1. Window-level              — every above-threshold window is an alert.
  2. Suppression window        — after an alert fires, no re-alert for the same
                                 admission within the suppression interval.
  3. Consecutive-alert logic   — an escalation fires on the 2nd consecutive
                                 above-threshold window of a run.
  4. Alert episode             — a maximal run of contiguous above-threshold
                                 windows counts once.

For the de-duplicated definitions, "sensitivity" is event-level: the proportion
of distinct deterioration episodes (maximal runs of label-positive windows
within an admission) for which at least one alert fired. "PPV" is the
proportion of fired alerts that coincide with a label-positive window.

Outputs new_analyses/alert_strategies_results.json
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
FEATURE_MATRIX = HERE / "feature_matrix.parquet"
OOF = HERE / "_oof_cache" / "xgb_primary_hadm_id.npy"
RESULTS = HERE / "final_revision_results.json"
OUT = HERE / "alert_strategies_results.json"

WINDOW_H = 6
SUPPRESSION_H = 12          # no re-alert within 12 h of an alert for that patient
HORIZON_H = 24              # a notification counts as a warning if it fires within
                            # this lead time before the deterioration episode


def main():
    R = json.loads(RESULTS.read_text())
    monitored_patient_days = R["alert_episodes"]["monitored_patient_days"]

    df = pd.read_parquet(FEATURE_MATRIX, columns=["hadm_id", "obs_time", "label"])
    df["score"] = np.load(OOF)

    # Recover the exact operating threshold rather than the 4-dp reported value,
    # so the window-level row reproduces main-text Table 4 exactly.
    _y = df["label"].values.astype(int)
    _s = df["score"].values
    _order = np.argsort(-_s)
    _k = int(np.searchsorted(np.cumsum(_y[_order]), int(np.ceil(0.90 * _y.sum()))))
    thr = float(_s[_order][_k])
    log.info(f"operating threshold {thr!r}")
    df = df.sort_values(["hadm_id", "obs_time"], kind="mergesort").reset_index(drop=True)

    adm = df["hadm_id"].values
    t = df["obs_time"].values
    y = df["label"].values.astype(bool)
    above = df["score"].values >= thr

    # Episode boundaries: new admission, or a gap wider than one 6-hourly step.
    gap_h = np.empty(len(df))
    gap_h[0] = np.inf
    gap_h[1:] = (t[1:] - t[:-1]) / np.timedelta64(1, "h")
    discontinuous = (adm != np.roll(adm, 1)) | (gap_h > WINDOW_H + 1e-6)
    discontinuous[0] = True

    def runs(flag):
        start = flag & (~np.roll(flag, 1) | discontinuous)
        start[0] = flag[0]
        return np.where(flag, np.cumsum(start), -1)

    outcome_ep = runs(y)
    n_outcome_ep = int(pd.Series(outcome_ep[outcome_ep >= 0]).nunique())
    log.info(f"{n_outcome_ep:,} distinct deterioration episodes")

    lead_windows = int(round(HORIZON_H / WINDOW_H))

    def detected(fired):
        """
        Outcome episodes for which a notification fired inside the episode or in
        the HORIZON_H lead-up, without crossing an admission/continuity break.
        """
        covered = fired.copy()
        for k in range(1, lead_windows + 1):
            shifted = np.roll(fired, k)
            shifted[:k] = False
            # Discard credit that would cross a discontinuity.
            blocked = np.zeros(len(fired), dtype=bool)
            for j in range(k):
                b = np.roll(discontinuous, j)
                b[:j] = True
                blocked |= b
            covered |= shifted & ~blocked
        m = outcome_ep >= 0
        return pd.Series(covered[m]).groupby(outcome_ep[m]).any()

    def summarise(name, fired, episode_count=None):
        """fired = boolean mask of windows at which a notification is raised."""
        n_alerts = int(fired.sum()) if episode_count is None else episode_count
        sens = float(detected(fired).mean())
        ppv = float(y[fired].mean()) if fired.any() else float("nan")
        return {
            "definition": name,
            "alerts": n_alerts,
            "alerts_per_patient_day": round(n_alerts / monitored_patient_days, 3),
            "event_level_sensitivity": round(sens, 4),
            "ppv": round(ppv, 4),
        }

    out = {}

    # 1. Window-level baseline
    out["window_level"] = summarise("Window-level", above)
    out["window_level"]["window_level_sensitivity"] = round(
        float(above[y].mean()), 4)

    # 2. Suppression window
    step = int(round(SUPPRESSION_H / WINDOW_H))   # windows suppressed after a fire
    fired = np.zeros(len(df), dtype=bool)
    cooldown = 0
    for i in range(len(df)):
        if discontinuous[i]:
            cooldown = 0
        if above[i] and cooldown == 0:
            fired[i] = True
            cooldown = step
        elif cooldown > 0:
            cooldown -= 1
    out["suppression_12h"] = summarise(
        f"Suppression window ({SUPPRESSION_H} h)", fired)

    # 3. Consecutive-alert escalation: fire on the 2nd window of a run
    prev_above = np.roll(above, 1)
    prev_above[0] = False
    prev_above[discontinuous] = False
    run_pos = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        run_pos[i] = (run_pos[i - 1] + 1) if (above[i] and prev_above[i]) else (
            1 if above[i] else 0)
    escalate = run_pos == 2
    out["consecutive_2"] = summarise("Consecutive-alert escalation (≥2)", escalate)

    # 4. Alert episode
    alert_ep = runs(above)
    n_ep = int(pd.Series(alert_ep[alert_ep >= 0]).nunique())
    first_of_ep = above & (~np.roll(above, 1) | discontinuous)
    first_of_ep[0] = above[0]
    # The notification fires once, at the start of the run, but the patient stays
    # in an alerted state for the whole run — so detection is assessed over the
    # run's windows, exactly as for the window-level row. Only the notification
    # COUNT differs between those two rows.
    ep = summarise("Alert episode (contiguous run counted once)", above,
                   episode_count=n_ep)
    ep["notifications_fired"] = int(first_of_ep.sum())
    # Episode-level PPV: episodes overlapping >=1 outcome-positive window.
    ov = pd.Series(y[alert_ep >= 0]).groupby(alert_ep[alert_ep >= 0]).any()
    ep["episode_level_ppv"] = round(float(ov.mean()), 4)
    out["alert_episode"] = ep

    payload = {
        "description": (
            "Alert burden under the window-level definition and three de-duplicated "
            "definitions, at the model operating threshold. 'event_level_sensitivity' "
            "is the proportion of distinct deterioration episodes for which at least "
            "one notification fired. 'ppv' is the proportion of fired notifications "
            "landing on a label-positive window; for the alert-episode row, "
            "'episode_level_ppv' additionally reports the proportion of episodes "
            "overlapping at least one label-positive window."
        ),
        "threshold": round(float(thr), 4),
        "suppression_hours": SUPPRESSION_H,
        "monitored_patient_days": monitored_patient_days,
        "n_outcome_episodes": n_outcome_ep,
        **out,
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print()
    hdr = f"{'definition':42s} {'alerts':>10s} {'/pt-day':>8s} {'sens':>7s} {'PPV':>7s}"
    print(hdr); print("-" * len(hdr))
    for k in ("window_level", "suppression_12h", "consecutive_2", "alert_episode"):
        r = out[k]
        print(f"{r['definition']:42s} {r['alerts']:>10,} "
              f"{r['alerts_per_patient_day']:>8.2f} "
              f"{r['event_level_sensitivity']*100:>6.1f}% {r['ppv']*100:>6.1f}%")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

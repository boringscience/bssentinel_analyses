# bssentinel — Clinical Deterioration Prediction in ICU-Linked Inpatients

**Development and internal validation of a multi-domain electronic health record machine learning model for detecting clinical deterioration in ICU-linked inpatients: a retrospective study using MIMIC-IV**

An XGBoost model trained on 83 multi-domain EHR features (vital signs, laboratory values, medications, temporal context, missingness indicators) to predict ICU transfer, mechanical ventilation, or in-hospital mortality within 24 hours.

**Key results:** AUROC 0.758 (95% CI 0.754–0.763), outperforming mNEWS2 (0.568), qSOFA (0.549), MEWS (0.533), and Shock Index (0.539).

---

## Repository Structure

```
├── scripts/
│   │
│   │  # ── Core ML pipeline ──────────────────────────────
│   ├── data_loader.py              # MIMIC-IV cohort extraction, vital/lab/med loading, outcome labelling
│   ├── feature_engineering.py      # 83-feature matrix construction, NEWS2 scoring, missingness flags
│   ├── model.py                    # XGBoost + LR training, cross-validation, calibration
│   ├── evaluate.py                 # AUROC/AUPRC with CIs, DeLong test, threshold analysis
│   ├── explain.py                  # SHAP values, global importance, server integration
│   ├── dca.py                      # Decision curve analysis
│   ├── train.py                    # End-to-end pipeline orchestrator
│   ├── requirements.txt            # Python dependencies
│   │
│   │  # ── Revision analyses ─────────────────────────────
│   ├── build_feature_matrix_cache.py  # Feature matrix cache builder for revision scripts
│   ├── domain_ablation.py          # Domain-level ablation analysis (13 models → Table 5)
│   ├── additional_comparators.py   # qSOFA, MEWS, Shock Index evaluation
│   ├── sensitivity_outcome.py      # Sensitivity analyses (excl. surgical, emergency-only)
│   ├── fix_brier_and_calibration.py # Corrected Brier score (0.175 → 0.018) and BSS
│   ├── fix_deff_and_ci.py          # Corrected DEFF (1.34 → 2.56) and patient-level CIs
│   ├── fix_table4_and_fpr.py       # Corrected Table 4 arithmetic, added TP column, fixed FPR
│   └── nature_figures.py           # All main (Figs 1–5) and supplementary (S1–S6) figures
│
├── results/                        # Revision analysis outputs (JSON)
│   ├── domain_ablation_results.json
│   ├── additional_comparators_results.json
│   ├── sensitivity_outcome_results.json
│   ├── brier_score_corrected.json
│   ├── deff_corrected.json
│   └── table4_corrected.json
│
├── artifacts/                      # Model outputs (from training pipeline)
│   ├── metrics.json                # Full validation metrics
│   ├── feature_cols.json           # Feature names in correct order
│   ├── feature_importance.csv      # XGBoost gain importance (top features)
│   └── analysis.txt                # Summary analysis report
│
├── figures/
│   ├── main/                       # Main manuscript figures (PNG, 300 dpi)
│   │   ├── fig1_study_design.png
│   │   ├── fig2_discrimination.png
│   │   ├── fig3_clinical_utility.png
│   │   ├── fig4_feature_importance.png
│   │   └── fig5_dca.png
│   └── supplementary/
│       ├── fig_s1_roc.png
│       ├── fig_s2_pr.png
│       ├── fig_s3_calibration.png
│       ├── fig_s4_cohort.png
│       ├── fig_s5_missingness.png
│       └── fig_s6_ablation.png
│
└── README.md
```

---

## Prerequisites

### 1. MIMIC-IV Access

MIMIC-IV v3.1 requires credentialing via PhysioNet:
1. Complete the CITI "Data or Specimens Only Research" course
2. Register at https://physionet.org and link your credential
3. Request access to `physionet.org/content/mimiciv/`
4. Download and extract — you need the `hosp/` and `icu/` modules

Expected directory structure:
```
mimic_iv/
├── hosp/
│   ├── admissions.csv
│   ├── patients.csv
│   ├── labevents.csv
│   ├── d_labitems.csv
│   └── prescriptions.csv
└── icu/
    ├── icustays.csv
    ├── chartevents.csv      ← ~3.5 GB
    ├── inputevents.csv
    └── procedureevents.csv
```

### 2. Python Environment

```bash
pip install -r scripts/requirements.txt
```

Dependencies: `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `scipy`

---

## Running the Pipeline

### Step 1: Train the model

```bash
# Development run (fast, 2000 admissions, ~10–20 min)
python scripts/train.py \
  --mimic /path/to/mimic_iv \
  --output ./artifacts \
  --horizon 24 \
  --dev

# Full training run (~2–4 hours)
python scripts/train.py \
  --mimic /path/to/mimic_iv \
  --output ./artifacts \
  --horizon 24 \
  --cache ./cache
```

This produces `artifacts/` with trained models (`model_xgb.pkl`, `model_lr.pkl`), out-of-fold predictions, metrics, and feature importance.

### Step 2: Build feature matrix cache (for revision analyses)

```bash
python scripts/build_feature_matrix_cache.py
```

### Step 3: Run revision analyses (order-independent)

```bash
python scripts/domain_ablation.py
python scripts/additional_comparators.py
python scripts/sensitivity_outcome.py
python scripts/fix_brier_and_calibration.py
python scripts/fix_deff_and_ci.py
python scripts/fix_table4_and_fpr.py
```

### Step 4: Generate all figures

```bash
python scripts/nature_figures.py
```

---

## Summary of Results

### Model Performance (Table 3)

| Model | AUROC | 95% CI |
|-------|-------|--------|
| **bssentinel (XGBoost)** | **0.758** | **0.754–0.763** |
| Logistic Regression | 0.708 | 0.703–0.713 |
| mNEWS2 | 0.568 | 0.565–0.572 |
| qSOFA | 0.549 | 0.546–0.552 |
| Shock Index | 0.539 | 0.535–0.542 |
| MEWS | 0.533 | 0.530–0.537 |

### Domain Ablation (Table 5)

| Removed Domain | ΔAUROC | Interpretation |
|----------------|--------|----------------|
| Vital signs | −0.051 | Dominant domain |
| Temporal/demographics | −0.014 | Second most important |
| Medications | −0.004 | Modest contribution |
| Laboratory values | −0.004 | Modest contribution |
| Missingness indicators | 0.000 | Redundant with other features |
| NEWS2 subscores | 0.000 | Subsumed by raw vitals |

### Operating Characteristics (threshold = 0.31)

| Metric | Value |
|--------|-------|
| Sensitivity | 90.0% |
| Specificity | 37.9% |
| PPV | 2.8% |
| NPV | 99.5% |
| Alerts per patient-day | 2.51 |

---

## Cohort

- **Source:** MIMIC-IV v3.1 (Beth Israel Deaconess Medical Center, 2008–2022)
- **Admissions:** 80,442 (from 62,025 unique patients)
- **Monitoring windows:** 1,334,773 (6-hourly)
- **Outcome prevalence:** 1.96% (26,201 positive windows)
- **Composite outcome:** ICU transfer, invasive mechanical ventilation, or in-hospital mortality within 24 hours

---

## Files Not Included

| File | Reason |
|------|--------|
| `model_xgb.pkl` / `model_lr.pkl` | Trained model binaries (~2 MB each); reproducible via `train.py` |
| `oof_predictions.parquet` | Out-of-fold predictions (21 MB); reproducible via `train.py` |
| `feature_matrix.parquet` | Cached feature matrix (55 MB); reproducible via `build_feature_matrix_cache.py` |

---

## Data Availability

This study uses MIMIC-IV v3.1, a publicly available de-identified EHR dataset. Access requires PhysioNet credentialing and CITI training. All results are fully reproducible from MIMIC-IV source tables using the provided scripts.

## License

Code is provided for research purposes. See MIMIC-IV data use agreement for data usage terms.

## Citation

If you use this code, please cite the manuscript (citation details to follow upon publication).

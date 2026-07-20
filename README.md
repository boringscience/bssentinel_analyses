# bssentinel — Revision Analyses

Revision analyses for: **"Development and internal validation of a multi-domain electronic health record machine learning model for detecting clinical deterioration in ICU-linked inpatients: a retrospective study using MIMIC-IV"**

## Repository Structure

```
revision_package/
├── scripts/                        # Analysis and figure generation code
│   ├── domain_ablation.py          # Domain-level ablation (13 models, Table 5)
│   ├── additional_comparators.py   # qSOFA, MEWS, Shock Index evaluation
│   ├── sensitivity_outcome.py      # Sensitivity analyses (excl. surgical, emergency-only)
│   ├── fix_brier_and_calibration.py# Corrected Brier score and calibration
│   ├── fix_deff_and_ci.py          # Corrected DEFF (1.34 → 2.56) and CIs
│   ├── fix_table4_and_fpr.py       # Corrected Table 4 arithmetic and FPR
│   ├── build_feature_matrix_cache.py # Feature matrix cache builder
│   └── nature_figures.py           # All main and supplementary figures
│
├── results/                        # Analysis outputs (JSON)
│   ├── domain_ablation_results.json
│   ├── additional_comparators_results.json
│   ├── sensitivity_outcome_results.json
│   ├── brier_score_corrected.json
│   ├── deff_corrected.json
│   └── table4_corrected.json
│
├── figures/
│   ├── main/                       # Main manuscript figures (PNG, 300 dpi)
│   │   ├── fig1_study_design.png   # Study design and system overview
│   │   ├── fig2_discrimination.png # ROC curves and AUROC comparison
│   │   ├── fig3_clinical_utility.png # Threshold analysis and alert burden
│   │   ├── fig4_feature_importance.png # XGBoost gain importance
│   │   └── fig5_dca.png            # Decision curve analysis
│   │
│   └── supplementary/              # Supplementary figures (PNG, 300 dpi)
│       ├── fig_s1_roc.png          # Five-fold ROC curves
│       ├── fig_s2_pr.png           # Precision-recall curves
│       ├── fig_s3_calibration.png  # Calibration reliability diagrams
│       ├── fig_s4_cohort.png       # Cohort flow diagram
│       ├── fig_s5_missingness.png  # Feature missingness reference
│       └── fig_s6_ablation.png     # Domain ablation visualization
│
└── README.md
```

## Key Revision Changes

| Analysis | Script | Result | Manuscript Section |
|----------|--------|--------|--------------------|
| Domain ablation (13 models) | `domain_ablation.py` | `domain_ablation_results.json` | New Table 5, Methods, Results, Discussion |
| Additional comparators (qSOFA, MEWS, SI) | `additional_comparators.py` | `additional_comparators_results.json` | Table 3, Figure 2 |
| Sensitivity analyses | `sensitivity_outcome.py` | `sensitivity_outcome_results.json` | Suppl. Table S4 |
| Brier score correction (0.175 → 0.018) | `fix_brier_and_calibration.py` | `brier_score_corrected.json` | Table 3, Results |
| DEFF correction (1.34 → 2.56) | `fix_deff_and_ci.py` | `deff_corrected.json` | Methods, Table 3 |
| Table 4 + FPR correction | `fix_table4_and_fpr.py` | `table4_corrected.json` | Table 4 |
| All figures redrawn | `nature_figures.py` | `figures/` | Figures 1–5, Suppl. S1–S6 |

## Requirements

- Python ≥ 3.10
- MIMIC-IV v3.1 access via PhysioNet
- Dependencies: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `shap`

## Reproducing Results

1. Obtain MIMIC-IV v3.1 credentials from [PhysioNet](https://physionet.org/content/mimiciv/3.1/)
2. Run the main bssentinel pipeline to generate `ml/artifacts/` (see main repository)
3. Build the feature matrix cache:
   ```bash
   python scripts/build_feature_matrix_cache.py
   ```
4. Run analyses (order-independent):
   ```bash
   python scripts/domain_ablation.py
   python scripts/additional_comparators.py
   python scripts/sensitivity_outcome.py
   python scripts/fix_brier_and_calibration.py
   python scripts/fix_deff_and_ci.py
   python scripts/fix_table4_and_fpr.py
   ```
5. Generate all figures:
   ```bash
   python scripts/nature_figures.py
   ```

## Data Availability

This study uses MIMIC-IV v3.1, a publicly available de-identified EHR dataset. Access requires PhysioNet credentialing and CITI training. The feature matrix (55 MB) is not included in this repository but is fully reproducible from MIMIC-IV source tables using the provided scripts.

## Citation

If you use this code, please cite the manuscript (citation details to follow upon publication).

# Routine Blood Markers for Cognitive Screening in Parkinson's Disease

Analysis code evaluating the incremental value of routine blood markers for concurrent cognitive screening in Parkinson's disease across PPMI and an external clinical cohort.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Paper:** Jing Wang, Yanfang Chen, and Kunpeng Xia. *Incremental value of routine blood markers for cognitive screening in Parkinson's disease: a cross-cohort evaluation*.

## Overview

Six algorithms—logistic regression (LR), support vector machine (SVM), random forest (RF), XGBoost, CatBoost, and a three-hidden-layer neural network (NN)—are crossed with three nested predictor sets: **Age-only set** (1 predictor), **Basic set** (5 demographic and anthropometric predictors), and **Full set** (21 basic and routine blood predictors). All 18 strategies share the same 969 PPMI training participants, 416 PPMI internal-test participants, and 197 external Luoyang PD (LY-PD) participants. Each strategy uses an independent 100-trial training-only tree-structured Parzen estimator (TPE) search with five-fold stratified cross-validation (seed `20261014`). Evaluation quantifies discrimination (AUROC), probability accuracy (Brier score), calibration, and paired within-algorithm Full-set minus Basic-set increments using 2,000 participant bootstrap resamples.

## Data Access and Setup

No raw PPMI tables, external participant records, predictions, or trained models are distributed in this repository.

- **PPMI exports (13 October 2026)**: Obtain via the [PPMI data portal](https://www.ppmi-info.org/access-data-specimens/download-data) (RRID:SCR_006431) under the PPMI Data Use Agreement.
- **External LY-PD cohort**: Obtain `published_cohorts.xlsx` (`FAHHAUST-PD` worksheet) from the supplementary materials of Wang and Yan (2025), [doi:10.3389/fnagi.2025.1688653](https://doi.org/10.3389/fnagi.2025.1688653).

Place the PPMI CSVs in `data/raw/ppmi/` (or set `PPMI_DATA_DIR` to a shared PPMI directory) and `published_cohorts.xlsx` in `data/raw/`. See [data/README.md](data/README.md) for the file list.

```bash
export PPMI_DATA_DIR=/path/to/ppmi/blood_clinical_20261014
# PowerShell: $env:PPMI_DATA_DIR = 'D:\path\to\ppmi\blood_clinical_20261014'
```

## Environment and Usage

Model training used Python 3.12.15 with package versions pinned in [requirements.txt](requirements.txt) (see also [environment_snapshot.json](environment_snapshot.json)).

```bash
python -m pip install -r requirements.txt

# 1. Cohort alignment, split verification, and cleaned input arrays
python prepare_cog_ppmi.py
python rematch_body_pm3.py
python freeze_cog_split.py --check
python prepare_bayes100_inputs.py --check

# 2. Model training (8 full/LR strategies + 10 reduced-input factorial strategies)
python bayes100.py --trials 100 --workers 1 --models Age_LR Basic_LR LR SVM RF XGBoost CatBoost NN
python factorial_preflight.py
python factorial_bayes100.py --workers 1

# 3. Performance estimation, tables, and empirical result figures
python factorial_statistics.py
python factorial_tables.py
python factorial_figures.py
```

Run the synthetic unit tests without private data:

```bash
python -m unittest discover -s tests -t .
```

Generated statistical tables and result figures are written locally under `outputs/`.

## Repository Structure

```text
pd-blood-cognitive-screening/
├── README.md                          # Overview and reproduction instructions
├── LICENSE                            # MIT license
├── CITATION.cff                       # Citation metadata
├── requirements.txt                   # Pinned Python dependencies
├── environment_snapshot.json          # Recorded training/postprocessing environment metadata
├── prepare_current_ppmi.py / prepare_cog_ppmi.py / rematch_body_pm3.py
├── body_rules.py / body_window.py / common_preprocessing.py / freeze_cog_split.py
├── prepare_bayes100_inputs.py / factorial_protocol.py / factorial_preprocessing.py
├── bayes100.py / factorial_preflight.py / factorial_bayes100.py / finalize_bayes100.py
├── factorial_results.py / factorial_statistics.py / prediction_intervals.py
├── calibration_likelihood.py / display_labels.py / pilot.py / validate_bayes100.py
├── factorial_tables.py / paper_bayes100_tables.py / six_model_tables.py
├── factorial_figures.py / calibration_bayes100_figure.py / manuscript_statistics.py
├── tests/                             # Synthetic unit tests
│   ├── test_body_window.py
│   └── test_scientific_behavior.py
└── data/
    └── README.md                      # Data access and directory setup
```

## License and Contact

Released under the [MIT License](LICENSE). PPMI data and the external cohort remain subject to their respective data-use agreements and licenses.

Correspondence: Jing Wang (`wangjing@xynu.edu.cn`).

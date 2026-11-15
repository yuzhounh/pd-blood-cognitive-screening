# Routine blood markers for cognitive screening in Parkinson's disease

Analysis code for **Incremental value of routine blood markers for cognitive screening in Parkinson's disease: a cross-cohort evaluation**.

## Current analysis: v4.0, 18 strategies

Six algorithms (LR, SVM, RF, XGBoost, CatBoost and a three-hidden-layer neural network) are each applied to age alone (Set 1), five demographic and anthropometric predictors (Set 2), and all 21 basic and blood predictors (Full set). All 18 strategies use the same 969 PPMI training participants, 416 internal-test participants and 197 Luoyang PD cohort (LY-PD) participants. The outcome is concurrent composite cognitive-screening positivity, not longitudinal cognitive decline.

Each strategy has an independent 100-trial training-only TPE search selected by mean five-fold AUROC. Eight compatible cleaned v3.2 fits were retained; ten reduced-input strategies completed the matrix. The factorial extension is exploratory. Within-algorithm Full set minus Set 2 comparisons assess the blood-panel increment: external AUROC point differences were negative for all six algorithms, while Brier-error changes depended on algorithm.

## Code and reproduction

See the [reproduction guide](code/README.md) for prerequisites, the eight-plus-ten training sequence, performance estimation, tables, figures and validation. `code/factorial_protocol.py` defines the complete registry; `code/bayes100.py` supplies the original eight fits and shared learners; `code/factorial_bayes100.py` trains the ten additional fits. Current postprocessing uses `factorial_statistics.py`, `factorial_tables.py` and `factorial_figures.py`.

Main Figure 2 reports all 18 strategies in both cohorts; Figure 3 reports six within-algorithm blood-panel increments. Supplementary ROC panels contain 36 curves, and calibration covers all 18 strategies. Main Figure 4 retains the three LR calibration comparisons. The editable [18-strategy workflow SVG](manuscript/figures/figure1_factorial_design.svg) is supplied together with its [base SVG](manuscript/figures/figure1_study_design.svg), which the exporter reads. Table formatting preserves the sign of very small nonzero interval bounds.

Training used Python 3.12.15 and the versions pinned in `requirements.txt`. `environment_snapshot.json` records all 18 training environments and the separate executed postprocessing environment. Historical helper modules are retained as dependencies; their old main functions are not the current analysis route.

## Data and artifact access

PPMI exports must be obtained by authorized researchers through the [PPMI data portal](https://www.ppmi-info.org/access-data-specimens/download-data). The external LY-PD cohort originated in the FAHHAUST-PD worksheet in [Wang and Yan's supplementary materials](https://doi.org/10.3389/fnagi.2025.1688653). Input versions, participant order, outcome labels and the frozen split must match the recorded protocol before a reproduction is comparable.

This is a source-code release. Raw or derived participant records, identifiers, individual predictions, fitted models, private input manifests, and the full author manuscript workspace are not distributed. The scripts require authorized inputs and, for postprocessing-only reproduction, the retained predictions and supporting aggregate receipts described in the guide. A code checkout alone cannot build the manuscript or reproduce private saved estimates. No new model training was performed for this code synchronization.

## Contact

Jing Wang, Xinyang Normal University. Email: wangjing@xynu.edu.cn.

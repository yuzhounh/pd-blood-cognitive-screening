# Routine blood markers for cognitive screening in Parkinson's disease

Analysis code for **Incremental value of routine blood markers for cognitive screening in Parkinson's disease: a cross-cohort evaluation**.

## Current analysis: v4.0, 18 strategies

Six algorithms (LR, SVM, RF, XGBoost, CatBoost and a three-hidden-layer neural network) are each applied to age alone (Age-only set), five demographic and anthropometric predictors (Basic set), and all 21 basic and blood predictors (Full set). All 18 strategies use the same 969 PPMI training participants, 416 internal-test participants and 197 Luoyang PD cohort (LY-PD) participants. The outcome is concurrent composite cognitive-screening positivity, not longitudinal cognitive decline.

Each strategy has an independent 100-trial training-only TPE search selected by mean five-fold AUROC. Eight compatible cleaned v3.2 fits were retained; ten reduced-input strategies completed the matrix. The factorial extension is exploratory. Within-algorithm Full set minus Basic set comparisons assess the blood-panel increment: external AUROC point differences were negative for all six algorithms, while Brier-error changes depended on algorithm.

## Code and reproduction

See the [reproduction guide](code/README.md) for prerequisites, the eight-plus-ten training sequence, performance estimation, tables, figures and validation. `code/factorial_protocol.py` defines the complete registry; `code/bayes100.py` supplies the original eight fits and shared learners; `code/factorial_bayes100.py` trains the ten additional fits. Current postprocessing uses `factorial_statistics.py`, `factorial_tables.py` and `factorial_figures.py`.

Main Figure 2 reports all 18 strategies in both cohorts; Figure 3 reports six within-algorithm blood-panel increments. Supplementary ROC panels contain 36 curves, and calibration covers all 18 strategies. Main Figure 4 retains the three LR calibration comparisons. The editable [18-strategy workflow SVG](manuscript/figures/figure1_factorial_design.svg) is the maintained source used by the current exporter; the [earlier eight-strategy SVG](manuscript/figures/figure1_study_design.svg) is retained for historical helper compatibility. Table formatting preserves the sign of very small nonzero interval bounds.

Training used Python 3.12.15 and the versions pinned in `requirements.txt`. `environment_snapshot.json` records all 18 training environments and the separate executed postprocessing environment. Historical helper modules are retained as dependencies; their old main functions are not the current analysis route.

The current presentation adds A–F algorithm labels to supplementary ROC panels and A–E labels to supplementary calibration panels, retaining the algorithm names and matching internal/external order. Content-based table widths, two-line unit/CI headers, unbreakable dense body cells, compact S1 and a one-page S2B layout are documented in the reproduction guide. S2B CV AUROC and threshold displays use three decimal places; saved full-precision thresholds drive the reported classifications. S8 preserves the source publication's two-decimal reported AUROCs. Use the documented presentation-only route with retained results to refresh displays without retraining or re-estimating bootstrap intervals.

Figure 1's maintained SVG now has uniformly sized stage icons, the `3 predictor sets × 6 algorithms` explanation, balanced stage-3 vertical spacing, and equal stage-2 card-row gaps with a shared bottom edge against stage 3. The S2/S3 generator uses separate left-aligned name and AUROC legend columns. Figure 1's editable SVG and the Python plotting source for S2/S3 are included; required private prediction inputs remain subject to the documented prerequisites. The latest source synchronization did not rerun experiments.

## Data and artifact access

PPMI exports must be obtained by authorized researchers through the [PPMI data portal](https://www.ppmi-info.org/access-data-specimens/download-data). The external LY-PD cohort originated in the FAHHAUST-PD worksheet in [Wang and Yan's supplementary materials](https://doi.org/10.3389/fnagi.2025.1688653). Input versions, participant order, outcome labels and the frozen split must match the recorded protocol before a reproduction is comparable.

This is a source-code release. Raw or derived participant records, identifiers, individual predictions, fitted models, private input manifests, and the full author manuscript workspace are not distributed. The scripts require authorized inputs and, for postprocessing-only reproduction, the retained predictions and supporting aggregate receipts described in the guide. A code checkout alone cannot build the manuscript or reproduce private saved estimates. No new model training was performed for this code synchronization.

## Contact

Jing Wang, Xinyang Normal University. Email: wangjing@xynu.edu.cn.

## Predictor-set display names

Reader-facing labels are Age-only set (1 predictor), Basic set (5 predictors), and Full set (21 predictors). `display_labels.py` maps the immutable result keys `Set 1` and `Set 2` to the semantic names. Existing strategy IDs, receipt filenames, prediction arrays, and unrounded result CSVs retain their original keys for compatibility.

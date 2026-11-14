# Routine blood markers for cognitive screening in Parkinson's disease

Analysis code for **Incremental predictive value of routine blood markers for cognitive screening in Parkinson's disease: development in PPMI and external validation in FAHHAUST**.

The study evaluates a concurrent cognitive-screening outcome. It compares age-only, five-predictor basic, and 21-predictor full-panel logistic regression, with six full-panel algorithms providing context. The development, internal-test, and external samples contain 969, 416, and 197 participants, respectively. Adding the blood panel lowered external logistic-regression AUROC relative to the basic reference; these results do not establish a clinically deployable screening tool.

## Code and environment

The final training runner is `code/bayes100.py`. Each of eight strategies uses 100 training-only TPE trials and mean five-fold AUROC for selection. The out-of-fold Youden threshold is fixed before evaluation. See [the reproduction guide](code/README.md) for cohort construction, model development, performance estimation, tables, and figures.

Use Python 3.12 and the dependencies in `requirements.txt`. `environment_snapshot.json` records the package versions used for the completed study. The retained `pilot.py`, `prepare_current_ppmi.py`, `six_model_tables.py`, and `manuscript_statistics.py` provide imports needed by the final workflow; their historical main functions are not the final model-development procedure.

The current four-figure presentation uses `code/main_presentation_figures.py` to export the editable [study workflow SVG](manuscript/figures/figure1_study_design.svg) with Inkscape and to draw internal/external ROC curves from the retained evaluation predictions. All 16 empirical AUROCs are checked against the saved results. This step does not refit models or change thresholds. The calibration plot is Figure 4, and the retained case-mix table is Table 4.

## Data access

PPMI exports must be obtained by authorized researchers through the [PPMI data portal](https://www.ppmi-info.org/access-data-specimens/download-data), subject to its access conditions. The external FAHHAUST-PD worksheet is identified in [Wang and Yan's supplementary materials](https://doi.org/10.3389/fnagi.2025.1688653). Place authorized inputs at the paths documented in the reproduction guide. Restricted records, participant identifiers, individual predictions, and fitted model objects are not distributed here.

This is a code release with the editable workflow SVG. The manuscript-building scripts also require the remaining manuscript sources, figure assets, and aggregate result files from the author workflow; those additional assets are not part of this code repository. Reproducing model estimates from fresh inputs requires running the documented analysis, rather than running the manuscript builder alone.

## Contact

Jing Wang, Xinyang Normal University. Email: wangjing@xynu.edu.cn.

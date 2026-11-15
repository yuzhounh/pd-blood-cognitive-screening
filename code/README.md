# v4.0: 18-strategy analysis and reproduction guide

The current analysis crosses six algorithms with three nested predictor sets on the same participants. Age-only set is age alone; Basic set is age, sex, height, weight and BMI; Full set adds 16 routine blood predictors. The samples contain 969 training, 416 internal-test and 197 external LY-PD participants. Positivity is MoCA <26 or NP1COG >=1, with both observed. The main comparison is Full set minus Basic set within each algorithm.

The v4.0 author release is fixed. Eight compatible cleaned v3.2 fits are reused and ten reduced-input fits complete the 18-strategy matrix. The independent retained-measurement sensitivity still contains eight strategies; the earlier-paper reconstruction is separate. The table-display correction dated 2026-11-15 changes one rounded interval bound, not fitted estimates or scientific conclusions.

## Environment and required private artifacts

Training used Python 3.12.15, NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1, scikit-learn 1.5.1, Optuna 4.7.0, XGBoost 3.0.0 and CatBoost 1.2.8. `requirements.txt` specifies the training dependencies; `environment_snapshot.json` also records the separate executed postprocessing environment. The author's temporary package-loading directory is a local implementation detail, not a distributed dependency. Exact retraining may depend on package versions, ordered inputs and platform.

Run commands from the project root in an authorized reproduction workspace. Code checkout alone does not supply these prerequisites:

- PPMI exports dated 13 October 2026 under `data/raw/ppmi_2026-10-14/`, and the source external workbook at `data/raw/published_cohorts.xlsx`.
- The aligned cohort and participant split in `data/derived/cog_pm3/`, external table in `data/derived/cog_v1/`, and cleaned arrays `training.npz`, `internal.npz`, `external.npz` plus `input_manifest.json` in `data/derived/bayes100_input/`.
- For postprocessing-only reproduction, baseline prediction packets in `data/derived/bayes100/`, additional packets in `data/derived/B129 Full Factorial Predictions/`, and their receipts in `manuscript/results/bayes100/` and `manuscript/results/B129 Full Factorial/`.
- Baseline compatibility/protection records, saved baseline calibration intervals, and independently retained B128 measurement-sensitivity results. The table generator reads that comparison to display S9. Full figure/manuscript generation also requires the author's descriptive aggregate results and manuscript sources.

These artifacts contain private data or belong to the full author workspace and are not part of this public code release. The cohort-building modules (`prepare_cog_ppmi.py`, `rematch_body_pm3.py`, `prepare_bayes100_inputs.py`, `body_window.py`, `body_rules.py`) document the underlying rules. `freeze_cog_split.py` reproduces the current stratified split from the aligned cohort in its original row order; it verifies an existing split and refuses to replace a different one. Historical builders depend on prior derived artifacts and are not a single clean-checkout bootstrap command. Do not run their historical main functions over an existing frozen release.

Cleaned measurements require height 100–250 cm inclusive and finite positive weight before nearest complete-pair matching within three calendar months of the index. BMI is recomputed from the selected pair. Invalid or missing predictors do not remove participants. Frozen participant allocation and outcome labels are preserved.

## Model development: eight baseline fits plus ten additional fits

The registry in `factorial_protocol.py` defines all 18 strategies and their source paths. `bayes100.py` supplies the shared learner, search and stopping routines. With the required cleaned input artifacts already prepared, the baseline training command is:

```powershell
python code/freeze_cog_split.py --check
python code/prepare_bayes100_inputs.py --check
python code/bayes100.py --trials 100 --workers 1 --models Age_LR Basic_LR LR SVM RF XGBoost CatBoost NN
```

The original study already had those eight compatible fits. Its extension checked and reused them, then trained only ten additional strategies:

```powershell
python code/factorial_preflight.py
python code/factorial_bayes100.py --workers 1
```

The preflight checks input/software compatibility, participant identity and columnwise preprocessing equivalence; it also performs ten small training-fold smoke fits. The full runner performs ten independent 100-trial TPE searches. Every strategy uses the same five stratified folds and seed 20261014. Boosted trees and NN use training-only early stopping; preprocessing and models are finally refitted on all 969 training participants with stopping counts fixed. OOF Youden thresholds are fixed before either evaluation cohort is scored. No feature selection, class weighting or outcome resampling is used in the primary analysis.

These commands perform scientific computation and are documentation, not a claim that a fresh run was performed for this release. Existing compatible receipts are reused; incompatible receipts are rejected. Work in a separate reproduction workspace for a new run, preserving the frozen author results.

## Postprocessing from completed predictions

With all private prerequisites available, the current sequence is:

```powershell
python code/factorial_statistics.py
python code/factorial_tables.py
python code/factorial_figures.py
python code/build_manuscript.py --factorial
python code/validate_factorial_manuscript.py
```

Statistics produce 36 strategy–cohort performance records, 72 absolute-metric intervals, 72 paired-metric intervals, 36 calibration records and 72 operating-point records. AUROC/Brier and paired differences use 2,000 shared outcome-stratified participant bootstrap samples per cohort. Calibration uses 2,000 ordinary participant bootstrap samples, with compatible baseline intervals reused; classification uses Wilson intervals. These are conditional on fitted prediction models, without model-development bootstrap. The validation script independently re-estimates selected paired bootstrap intervals; it does not retrain prediction models. Model comparisons are exploratory and unadjusted for multiplicity; intervals crossing zero do not establish equivalence.

`factorial_tables.py` renders the current main performance/increment tables and supplementary tables. `difference_interval` adds decimal places when a nonzero difference or bound would otherwise round to zero. For example, the internal XGBoost Brier interval displays `0.0002 (-0.00003 to 0.0005)`. Stored unrounded estimates are unchanged.

`factorial_figures.py` exports the 18-strategy workflow, both cohorts' 18-strategy AUROC plots, six-algorithm increment plots, 36 ROC curves, and non-LR calibration panels. Main Figure 4 retains the three LR calibration comparisons; its generator is `calibration_bayes100_figure.py`. Age case mix is Supplementary Figure S1. Figure export requires Arial and Inkscape; PDF compilation requires TeX Live/`latexmk` and `pdftotext`.

Main Tables 1 and 4 describe the common cohort and case mix. Table 2 reports six Full-set models; Table 3 reports their within-algorithm blood increments. S2B contains 18 selected configurations; S3A/B contain 18 performances per cohort; S4 contains 36 calibration rows; S5A/B contain 36 classification rows per cohort. S6 holds secondary predictor-set contrasts. The earlier-publication reconstruction is S7/S8; the retained eight-strategy measurement sensitivity is S9. No additional 18-strategy sensitivity experiment is implied.

Publication scripts include older helper modules for imports and provenance. Their historical main functions are superseded by the current route above. Restricted inputs, individual predictions, fitted objects and manuscript snapshots remain outside this public repository.

## Predictor-set display names

Reader-facing labels are Age-only set (1 predictor), Basic set (5 predictors), and Full set (21 predictors). `display_labels.py` maps the immutable result keys `Set 1` and `Set 2` to the semantic names. Existing strategy IDs, receipt filenames, prediction arrays, and unrounded result CSVs retain their original keys for compatibility.

For a display-only refresh from existing results, run `python code/factorial_tables.py`, `python code/factorial_figures.py --presentation-only`, and `python code/build_manuscript.py --factorial`. The figure command reuses calibration bins and Wilson intervals, leaves the age figure unchanged, and forms ROC plotting coordinates from frozen predictions; it does not train estimators or rerun bootstrap inference. When refreshing a frozen active workspace, only the generated presentation CSVs and build/figure metadata need write access; frozen analysis and prediction files remain protected.

Run `python code/validate_factorial_manuscript.py --presentation-only` to check display names, figure/result bindings, table coverage and compiled PDFs without replaying bootstrap inference.

Figure 2 uses a 9.4 by 5.22 inch layout with plotting-panel height/width of 9/8; its plotting height is 10% shorter than the preceding 5/4-panel layout. Figure 3 uses cohort rows (internal, external) and metric columns (AUROC, Brier), with equal limits and ticks within each metric column. Table 1 keeps names and units in the first column; its note specifies n or n (%), mean (SD) for age and MoCA, and median [Q1, Q3] for anthropometry and blood predictors.

## Current presentation details

The current generators preserve the semantic predictor-set names and the following display choices. These changes do not alter saved predictions, model selection, or statistical estimates.

Figure 1 is exported from the maintained `figure1_factorial_design.svg`. Its first three icons share a 70-unit visual bounding frame with proportional scaling and equal effective stroke width. The third-stage heading is `18 prediction strategies` above `3 predictor sets × 6 algorithms`; the repeated sentence about crossing every algorithm with all sets is omitted. The exporter validates the registry counts and icon metadata before exporting the SVG, rather than recreating it from the earlier eight-strategy figure.

The third-stage algorithm list has baselines at 246/286, the divider is at 309, and the arrow starts at 525 while retaining its endpoint at 567. This gives extra separation below the combination heading, balanced visible gaps around the divider, and a shorter gap above the arrow. The SVG remains the authority for these coordinates; the exporter checks the separation between the heading and algorithm list.

Stage 2 has card-row tops at 211, 319, 437 and 555, with respective heights of 90, 100, 100 and 182 SVG units. All three intervening gaps are 18 units. The paired training/internal-test cards share the first row; the Full set card and stage-3 Final model application card have a common bottom at 737. The exporter checks equal row gaps and bottom alignment.

- Supplementary ROC Figures S2/S3 label the algorithms A–F: LR, SVM, RF, XGBoost, CatBoost, NN. Each ROC legend uses separate left-aligned predictor-set and AUROC columns, with corresponding rows sharing a baseline. Calibration Figures S4/S5 use A–E: SVM, RF, XGBoost, CatBoost, NN. Internal and external panels use matching letter/algorithm order. The author manuscript scales Figure S1 to 0.80 of the text width without modifying its vector data figure.
- S2A keeps `Trials per set` on one line. S2B uses 9-point type, 3-point column padding and a wider parameter column; the author layout fits all 18 rows and the note on page 3. Its CV AUROC and OOF cut-off columns display three decimal places. Operating-point results use full-precision stored thresholds, not those rounded display values. Floating hyperparameters retain six significant digits; tree/epoch counts remain integers.
- S4 uses local 8.5/10.5-point type and 2.5-point column padding. S5A/B and S6 retain 9/11-point type. Their body cells are unbreakable, with content-sized columns; headers may occupy two lines. `header_break='ci'` places `(95% CI)` on the second line; `header_break='percent_ci'` places `% (95% CI)` there and omits a redundant trailing comma from the metric name. S8 uses `Workflow A` above `AUROC (95% CI)`; S9 separates metric names from `cleaned`/`retained` in the header.
- S8's `Reported AUROC` column retains the original publication's two decimal places, while reconstructed AUROCs use three. This is source precision, not a different underlying calculation.
- S1 is a centered 12-cm short table with content-based columns, retained 9-point type and unchanged counts. `paper_bayes100_tables.write_missing_table()` regenerates it from the saved missingness table. `six_model_tables.write_reconstruction_table()` regenerates S8 from saved ancillary metrics without invoking the superseded historical main routine. For either scoped helper, import it with `code/` on the Python module path and call only that function.

The author's supplementary source now cites every supplementary figure and table in its corresponding explanatory text, uses unique PDF destinations for A/B table parts, and places S1 on the first page. These source-layout choices require the author's supplementary TeX file, which is outside this code-only release. Display and layout checks were completed without rerunning experiments or bootstrap inference for this synchronization.

The editable source for Figure 1 is its maintained SVG. Figures S2/S3 are code-generated statistical figures: their source is `factorial_figures.py` (`roc_panels`), which reads the retained prediction packets and performance records described above; no separate manually maintained SVG is required. This synchronization updates those sources and the guide only, reusing the completed figure/layout checks and without running training or inference commands.

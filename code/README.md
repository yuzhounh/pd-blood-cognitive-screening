# v3.0 analysis and manuscript reproduction

The primary analysis uses updated raw PPMI records and the released FAHHAUST-PD worksheet. The earlier publication's PPMI table is used only for Supplementary Tables S5–S6. The final modeling runner is `bayes100.py`; historical pilot runners are not required for the main comparison.

## Inputs

Restricted input files are not distributed with the code. Authorized researchers must place the PPMI exports under `data/raw/ppmi_2026-10-14/` and the source supplementary workbook at `data/raw/published_cohorts.xlsx`. The cohort builder specifies the required table names. Export versions, input hashes, ordered predictor names, and cohort counts are recorded in the source and search manifests. `environment_snapshot.json` records the environments actually used; per-strategy receipts retain exact package versions. Installing a compatible environment alone is not a claim of bitwise identical retraining.

From the project root, the input-building sequence is:

```powershell
python code/prepare_cog_ppmi.py
python code/prepare_bayes100_inputs.py
python code/prepare_bayes100_inputs.py --check
```

The second command matches complete height/weight records within ±3 calendar months, calculates BMI, and exports the fixed 969/416/197 input arrays. Existing split IDs and input manifests are checked and never silently replaced. Current verified arrays already exist; rebuilding them is unnecessary for manuscript reproduction.

## Model development

`bayes100.py` performs 100 TPE trials for each strategy, maximizing arithmetic mean training five-fold AUROC. It fits preprocessing within model-fitting subsets, chooses early-stopping counts within training data, refits on all 969 training participants, fixes an OOF Youden threshold, and only then opens held-out input arrays. The six full-panel strategies use 21 predictors; `Age_LR` and `Basic_LR` receive one and five columns. Class weights and outcome-based resampling are disabled.

For a fresh authorized compute run, a single-worker example is:

```powershell
python code/bayes100.py --trials 100 --workers 1 --models LR SVM RF XGBoost CatBoost NN Age_LR Basic_LR
```

Completed predictions are private files in `data/derived/bayes100/`; aggregate receipts and all trial configurations are in `manuscript/results/bayes100/`. The actual v3.0 run used Server 141 for seven strategies and Server 140 for XGBoost. Code, preprocessing, input, environment, and prediction signatures are recorded. No new training is needed to inspect the completed analysis.

## Reproduce the current outputs

The current source/PDF pairs are `manuscript/manuscript.tex` and `manuscript.pdf`, and `manuscript/supplementary_materials.tex` and `supplementary_materials.pdf`. `build_manuscript.py` compiles these names directly. For a formatting-only revision, run only that builder; the analysis and figure regeneration commands below are not needed. The common template uses 11-point Latin Modern text, A4 with 2.5 cm margins, double line spacing, 10/12-point captions, and 9/11-point tables and notes. Table generators retain the shared typography and account for column padding.

With the retained predictions, this sequence regenerates inference, tables, figures, PDFs, and checks:

```powershell
python code/finalize_bayes100.py
python code/validate_bayes100.py
python code/paper_bayes100_tables.py
python code/calibration_bayes100_figure.py
python code/manuscript_figures.py
python code/main_presentation_figures.py
python code/build_manuscript.py
python code/validate_bayes100_manuscript.py
```

The summarizer does not retrain predictive models. It uses 2,000 paired outcome-stratified participant bootstrap samples conditional on fitted models. Calibration models and probability-bin plots describe saved predictions without recalibration. Arial must be installed for figure export. PDF builds require TeX Live with `latexmk` and Poppler's `pdftotext`; auxiliary files stay under `tmp/latex/`, and final PDFs are copied to `manuscript/` after successful checks.

Tables 1/4, Supplementary Table S1, and the ancillary S5/S6 are retained descriptive/source-material outputs. They are unchanged by hyperparameter tuning. Table 4 was promoted from the supplement without recalculating its values. The four main figures cover the editable study workflow, AUROC/case mix, empirical ROC curves, and nested-LR calibration; the supplement contains six tables and no duplicated figures. `main_presentation_figures.py` exports the maintained workflow SVG using Inkscape and plots the ROC curves from frozen predictions, checking all 16 AUROCs against the retained results without refitting or updating thresholds. The calibration generator now writes `figure4_calibration.pdf/png`. `six_model_tables.py` and `manuscript_statistics.py` supply reusable formatting helpers to the current generator, but their historical main functions must not overwrite the current model tables. The earlier limited-search validator is likewise superseded by `validate_bayes100_manuscript.py`.

Raw records, identifiers, participant-level predictions, temporary build files, references, and caches are excluded from version control. Current final manuscript and supplementary PDFs are deliverables. The immutable v2.0 snapshot is documented in `review/version.md`; v3.0 has been tagged and snapshotted as a stable release.

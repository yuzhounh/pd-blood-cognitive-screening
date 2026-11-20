# Data Directory

This repository does not include raw PPMI tables, external cohort records, participant-level derivatives, or fitted model checkpoints.

## Required Inputs

1. **PPMI clinical and laboratory exports** (downloaded 13 October 2026): Obtain from the [PPMI data portal](https://www.ppmi-info.org/access-data-specimens/download-data) (RRID:SCR_006431) under the PPMI Data Use Agreement. Required tables include `Participant_Status*.csv`, `Demographics*.csv`, `Blood_Chemistry___Hematology_*.csv`, `Blood_Chemistry___Hematology-Archived_*.csv`, `Montreal_Cognitive_Assessment__MoCA_*.csv`, `MDS-UPDRS_Part_I*.csv`, and `Vital_Signs_*.csv`.
2. **External LY-PD cohort**: Originated from the `FAHHAUST-PD` worksheet in the supplementary materials of Wang and Yan (2025), [doi:10.3389/fnagi.2025.1688653](https://doi.org/10.3389/fnagi.2025.1688653), saved as `published_cohorts.xlsx`.

## Placement

Place the PPMI CSV tables in `data/raw/ppmi/` (or set `PPMI_DATA_DIR` to a shared folder containing the batch) and place `published_cohorts.xlsx` in `data/raw/`. All files under `data/raw/` and `data/derived/` are ignored by `.gitignore`.

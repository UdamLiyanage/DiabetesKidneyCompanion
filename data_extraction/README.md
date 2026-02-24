# Data Extraction Scripts

These scripts extract and analyze the diabetic CKD patient cohort from MIMIC-IV via Google BigQuery.

> **Note**: These scripts require [PhysioNet credentialed access](https://physionet.org/settings/credentialing/) to MIMIC-IV. They are included for reproducibility and to document the data pipeline.

## Prerequisites

1. **PhysioNet Account** with credentialed access to MIMIC-IV
2. **Google Cloud Project** linked to PhysioNet BigQuery access
3. **Dependencies**: `pip install google-cloud-bigquery pandas pyarrow db-dtypes`

## Scripts

### 1. `quick_explore.py`
Verifies BigQuery access and previews available data.

```bash
# Test with demo dataset (no credentials needed)
python quick_explore.py --project YOUR_PROJECT_ID --demo

# Full MIMIC-IV access
python quick_explore.py --project YOUR_PROJECT_ID
```

### 2. `extract_diabetes_ckd_cohort.py`
Main extraction pipeline — identifies patients with both diabetes and CKD, extracts labs, medications, and diagnoses.

```bash
python extract_diabetes_ckd_cohort.py --project YOUR_PROJECT_ID
```

**Outputs** (in `./extracted_data/`):
- `dm_ckd_cohort.csv` — Patient demographics
- `longitudinal_labs.csv` — All lab results over time
- `medications.csv` — Prescription records
- `diagnoses.csv` — ICD codes
- `omr_egfr_vitals.csv` — Outpatient eGFR and vitals

### 3. `analyze_cohort.py`
Post-extraction analysis — computes CKD staging (CKD-EPI 2021), flags medication concerns, and builds patient profiles.

```bash
python analyze_cohort.py --data-dir ./extracted_data
```

**Outputs**:
- `patient_profiles.csv` — Summary per patient
- `patient_profiles.json` — Detailed profiles for the app

## Cohort Statistics (from our extraction)

| Metric | Value |
|--------|-------|
| Total patients | 14,815 |
| Lab records | ~4.2M |
| Medication records | ~890K |

## ICD Codes Used

**Diabetes (ICD-9/10)**:
- `250%` / `E10%`, `E11%`, `E13%`

**CKD (ICD-9/10)**:
- `585%`, `586%`, `403%`, `404%` / `N18%`, `N19%`, `I12%`, `I13%`

## Notes

- MIMIC-IV contains ICU patients, so CKD staging reflects acute illness context
- eGFR is calculated using CKD-EPI 2021 (race-free equation) when not directly available
- The synthetic lab reports used by the main pipeline were generated with value distributions from this cohort

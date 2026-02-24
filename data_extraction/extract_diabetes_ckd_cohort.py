"""
MIMIC-IV BigQuery Extraction Pipeline - DiabetesKidney Companion
=================================================================
Extracts diabetic CKD patients with longitudinal labs and medications
from MIMIC-IV v3.1 via Google BigQuery.

Prerequisites:
    1. PhysioNet credentialed access to MIMIC-IV
    2. Google Cloud project linked to PhysioNet BigQuery access
    3. pip install google-cloud-bigquery pandas pyarrow db-dtypes

Usage:
    python extract_diabetes_ckd_cohort.py --project YOUR_GCP_PROJECT_ID

Author: DiabetesKidney Companion / MedGemma Impact Challenge
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

# =============================================================================
# Configuration
# =============================================================================

# BigQuery dataset references (MIMIC-IV v3.1)
# After Nov 25, 2024, the default datasets also contain v3.1
HOSP_DATASET = "physionet-data.mimiciv_3_1_hosp"  # or mimiciv_v3_1_hosp

# Output directory
OUTPUT_DIR = Path("./extracted_data")

# =============================================================================
# ICD Code Definitions
# =============================================================================

# ICD-9 codes for diabetes mellitus
ICD9_DIABETES = [
    "250%",  # Diabetes mellitus (all subtypes)
]

# ICD-10 codes for diabetes mellitus
ICD10_DIABETES = [
    "E10%",  # Type 1 diabetes mellitus
    "E11%",  # Type 2 diabetes mellitus
    "E13%",  # Other specified diabetes mellitus
]

# ICD-9 codes for CKD
ICD9_CKD = [
    "585%",  # Chronic kidney disease
    "586%",  # Renal failure, unspecified
    "403%",  # Hypertensive chronic kidney disease
    "404%",  # Hypertensive heart and chronic kidney disease
]

# ICD-10 codes for CKD
ICD10_CKD = [
    "N18%",  # Chronic kidney disease
    "N19%",  # Unspecified kidney failure
    "I12%",  # Hypertensive chronic kidney disease
    "I13%",  # Hypertensive heart and chronic kidney disease
    "E11.22",  # Type 2 diabetes with diabetic CKD
    "E11.65",  # Type 2 diabetes with hyperglycemia
    "E10.22",  # Type 1 diabetes with diabetic CKD
]

# =============================================================================
# Lab Item Definitions (common MIMIC-IV itemids)
# NOTE: These will be verified against d_labitems in Step 1
# =============================================================================

# We search by label rather than hardcoding itemids for robustness
LAB_LABELS = {
    "creatinine": ["Creatinine"],
    "bun": ["Urea Nitrogen"],
    "egfr": ["Estimated GFR"],  # Note: eGFR may also be in omr table
    "hba1c": ["%Hemoglobin A1c"],
    "glucose": ["Glucose"],
    "potassium": ["Potassium"],
    "phosphorus": ["Phosphate"],  # Often labeled as Phosphate in MIMIC
    "calcium": ["Calcium, Total"],
    "albumin": ["Albumin"],
    "sodium": ["Sodium"],
    "bicarbonate": ["Bicarbonate"],
    "hemoglobin": ["Hemoglobin"],
    "urine_protein": ["Protein", "Urine Protein"],
    "uacr": ["Albumin/Creatinine, Urine"],  # Urine albumin-to-creatinine ratio
}


def get_client(project_id: str) -> bigquery.Client:
    """Create authenticated BigQuery client."""
    return bigquery.Client(project=project_id)


def run_query(client: bigquery.Client, query: str, description: str = "") -> pd.DataFrame:
    """Execute a BigQuery query and return results as DataFrame."""
    if description:
        print(f"\n{'=' * 60}")
        print(f"  {description}")
        print(f"{'=' * 60}")

    print(f"  Executing query...")
    job = client.query(query)
    df = job.to_dataframe()
    print(f"  ✓ Returned {len(df):,} rows, {len(df.columns)} columns")

    if len(df) > 0:
        print(f"  Columns: {list(df.columns)}")

    return df


# =============================================================================
# STEP 1: Discover Lab Item IDs
# =============================================================================

def step1_discover_lab_items(client: bigquery.Client) -> pd.DataFrame:
    """
    Query d_labitems to find the exact itemids for our labs of interest.
    This is more robust than hardcoding itemids which can change between versions.
    """
    # Build LIKE conditions for all lab labels
    like_conditions = []
    for key, labels in LAB_LABELS.items():
        for label in labels:
            like_conditions.append(f"LOWER(label) LIKE LOWER('%{label}%')")

    where_clause = " OR ".join(like_conditions)

    query = f"""
    SELECT 
        itemid,
        label,
        fluid,
        category
    FROM `{HOSP_DATASET}.d_labitems`
    WHERE {where_clause}
    ORDER BY label
    """

    return run_query(client, query, "Step 1: Discovering Lab Item IDs from d_labitems")


# =============================================================================
# STEP 2: Identify Diabetic CKD Cohort
# =============================================================================

def step2_identify_cohort(client: bigquery.Client) -> pd.DataFrame:
    """
    Identify patients with BOTH diabetes AND CKD diagnoses.
    Uses ICD-9 and ICD-10 codes from diagnoses_icd table.
    """

    # Build ICD LIKE clauses
    def build_icd_likes(codes, alias="icd_code"):
        return " OR ".join([f"{alias} LIKE '{c}'" for c in codes])

    dm_icd9 = build_icd_likes(ICD9_DIABETES)
    dm_icd10 = build_icd_likes(ICD10_DIABETES)
    ckd_icd9 = build_icd_likes(ICD9_CKD)
    ckd_icd10 = build_icd_likes(ICD10_CKD)

    query = f"""
    -- Patients with diabetes diagnoses
    WITH dm_patients AS (
        SELECT DISTINCT subject_id
        FROM `{HOSP_DATASET}.diagnoses_icd`
        WHERE 
            (icd_version = 9 AND ({dm_icd9}))
            OR (icd_version = 10 AND ({dm_icd10}))
    ),

    -- Patients with CKD diagnoses
    ckd_patients AS (
        SELECT DISTINCT subject_id
        FROM `{HOSP_DATASET}.diagnoses_icd`
        WHERE 
            (icd_version = 9 AND ({ckd_icd9}))
            OR (icd_version = 10 AND ({ckd_icd10}))
    ),

    -- Intersection: patients with BOTH diabetes AND CKD
    dm_ckd_cohort AS (
        SELECT dm.subject_id
        FROM dm_patients dm
        INNER JOIN ckd_patients ckd ON dm.subject_id = ckd.subject_id
    )

    -- Join with patient demographics
    SELECT 
        p.subject_id,
        p.gender,
        p.anchor_age,
        p.anchor_year,
        p.anchor_year_group,
        p.dod,  -- date of death (if applicable)
        a.admittime AS first_admittime,
        a.race,
        COUNT(DISTINCT a.hadm_id) AS total_admissions
    FROM dm_ckd_cohort c
    INNER JOIN `{HOSP_DATASET}.patients` p 
        ON c.subject_id = p.subject_id
    INNER JOIN `{HOSP_DATASET}.admissions` a
        ON c.subject_id = a.subject_id
    GROUP BY 
        p.subject_id, p.gender, p.anchor_age, p.anchor_year,
        p.anchor_year_group, p.dod, a.admittime, a.race
    ORDER BY p.subject_id
    """

    return run_query(client, query, "Step 2: Identifying Diabetic + CKD Cohort")


# =============================================================================
# STEP 3: Extract Longitudinal Lab Results
# =============================================================================

def step3_extract_labs(client: bigquery.Client, lab_items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all relevant lab results for the DM+CKD cohort.
    Includes longitudinal data across all admissions.
    """
    # Get the discovered itemids
    itemids = lab_items_df['itemid'].tolist()
    if not itemids:
        print("  WARNING: No lab items discovered. Using fallback search.")
        itemid_clause = "1=1"  # will be slow but catch-all
    else:
        itemid_clause = f"le.itemid IN ({','.join(str(i) for i in itemids)})"

    # Build ICD clauses (same as step 2)
    dm_icd9 = " OR ".join([f"icd_code LIKE '{c}'" for c in ICD9_DIABETES])
    dm_icd10 = " OR ".join([f"icd_code LIKE '{c}'" for c in ICD10_DIABETES])
    ckd_icd9 = " OR ".join([f"icd_code LIKE '{c}'" for c in ICD9_CKD])
    ckd_icd10 = " OR ".join([f"icd_code LIKE '{c}'" for c in ICD10_CKD])

    query = f"""
    WITH dm_ckd_cohort AS (
        SELECT DISTINCT d1.subject_id
        FROM (
            SELECT DISTINCT subject_id FROM `{HOSP_DATASET}.diagnoses_icd`
            WHERE (icd_version = 9 AND ({dm_icd9}))
               OR (icd_version = 10 AND ({dm_icd10}))
        ) d1
        INNER JOIN (
            SELECT DISTINCT subject_id FROM `{HOSP_DATASET}.diagnoses_icd`
            WHERE (icd_version = 9 AND ({ckd_icd9}))
               OR (icd_version = 10 AND ({ckd_icd10}))
        ) d2 ON d1.subject_id = d2.subject_id
    )

    SELECT 
        le.subject_id,
        le.hadm_id,
        le.specimen_id,
        le.itemid,
        dl.label AS lab_name,
        dl.fluid,
        le.charttime,
        le.storetime,
        le.value,
        le.valuenum,
        le.valueuom,
        le.ref_range_lower,
        le.ref_range_upper,
        le.flag,  -- 'abnormal' flag
        le.priority,
        le.comments
    FROM `{HOSP_DATASET}.labevents` le
    INNER JOIN dm_ckd_cohort c ON le.subject_id = c.subject_id
    INNER JOIN `{HOSP_DATASET}.d_labitems` dl ON le.itemid = dl.itemid
    WHERE {itemid_clause}
    ORDER BY le.subject_id, le.charttime
    """

    return run_query(client, query, "Step 3: Extracting Longitudinal Lab Results")


# =============================================================================
# STEP 4: Extract eGFR from OMR Table
# =============================================================================

def step4_extract_omr_egfr(client: bigquery.Client) -> pd.DataFrame:
    """
    Extract eGFR from the Online Medical Record (OMR) table.
    OMR contains outpatient data including baseline eGFR values
    that may not be in labevents.
    """
    dm_icd9 = " OR ".join([f"icd_code LIKE '{c}'" for c in ICD9_DIABETES])
    dm_icd10 = " OR ".join([f"icd_code LIKE '{c}'" for c in ICD10_DIABETES])
    ckd_icd9 = " OR ".join([f"icd_code LIKE '{c}'" for c in ICD9_CKD])
    ckd_icd10 = " OR ".join([f"icd_code LIKE '{c}'" for c in ICD10_CKD])

    query = f"""
    WITH dm_ckd_cohort AS (
        SELECT DISTINCT d1.subject_id
        FROM (
            SELECT DISTINCT subject_id FROM `{HOSP_DATASET}.diagnoses_icd`
            WHERE (icd_version = 9 AND ({dm_icd9}))
               OR (icd_version = 10 AND ({dm_icd10}))
        ) d1
        INNER JOIN (
            SELECT DISTINCT subject_id FROM `{HOSP_DATASET}.diagnoses_icd`
            WHERE (icd_version = 9 AND ({ckd_icd9}))
               OR (icd_version = 10 AND ({ckd_icd10}))
        ) d2 ON d1.subject_id = d2.subject_id
    )

    SELECT
        o.subject_id,
        o.chartdate,
        o.result_name,
        o.result_value
    FROM `{HOSP_DATASET}.omr` o
    INNER JOIN dm_ckd_cohort c ON o.subject_id = c.subject_id
    WHERE LOWER(o.result_name) LIKE '%egfr%'
       OR LOWER(o.result_name) LIKE '%glomerular%'
       OR LOWER(o.result_name) LIKE '%blood pressure%'
       OR LOWER(o.result_name) LIKE '%weight%'
       OR LOWER(o.result_name) LIKE '%bmi%'
    ORDER BY o.subject_id, o.chartdate
    """

    return run_query(client, query, "Step 4: Extracting eGFR & Vitals from OMR Table")


# =============================================================================
# STEP 5: Extract Medications (Prescriptions)
# =============================================================================

def step5_extract_medications(client: bigquery.Client) -> pd.DataFrame:
    """
    Extract medication prescriptions for the DM+CKD cohort.
    Focus on diabetes meds, nephrotoxic drugs, and drugs requiring
    renal dose adjustment.
    """
    # Medications of interest for diabetes-kidney interaction
    med_keywords = [
        # Diabetes medications
        "metformin", "glipizide", "glyburide", "glimepiride",
        "sitagliptin", "linagliptin", "saxagliptin", "alogliptin",
        "empagliflozin", "dapagliflozin", "canagliflozin",  # SGLT2i
        "liraglutide", "semaglutide", "dulaglutide",  # GLP-1 RAs
        "insulin",
        "pioglitazone", "rosiglitazone",
        # ACE inhibitors / ARBs (renoprotective)
        "lisinopril", "enalapril", "ramipril", "captopril",
        "losartan", "valsartan", "irbesartan", "olmesartan",
        # Diuretics
        "furosemide", "hydrochlorothiazide", "spironolactone",
        "bumetanide", "torsemide", "metolazone",
        # Nephrotoxic or renal-adjusted drugs
        "nsaid", "ibuprofen", "naproxen", "celecoxib",
        "gentamicin", "vancomycin", "tobramycin",
        "lithium",
        # Phosphate binders & CKD-specific
        "sevelamer", "calcium acetate", "lanthanum",
        "epoetin", "darbepoetin",  # ESAs
        "calcitriol", "paricalcitol",
        "sodium bicarbonate",
        "kayexalate", "patiromer",  # Potassium binders
    ]

    med_like_clauses = " OR ".join(
        [f"LOWER(drug) LIKE '%{m}%'" for m in med_keywords]
    )

    dm_icd9 = " OR ".join([f"icd_code LIKE '{c}'" for c in ICD9_DIABETES])
    dm_icd10 = " OR ".join([f"icd_code LIKE '{c}'" for c in ICD10_DIABETES])
    ckd_icd9 = " OR ".join([f"icd_code LIKE '{c}'" for c in ICD9_CKD])
    ckd_icd10 = " OR ".join([f"icd_code LIKE '{c}'" for c in ICD10_CKD])

    query = f"""
    WITH dm_ckd_cohort AS (
        SELECT DISTINCT d1.subject_id
        FROM (
            SELECT DISTINCT subject_id FROM `{HOSP_DATASET}.diagnoses_icd`
            WHERE (icd_version = 9 AND ({dm_icd9}))
               OR (icd_version = 10 AND ({dm_icd10}))
        ) d1
        INNER JOIN (
            SELECT DISTINCT subject_id FROM `{HOSP_DATASET}.diagnoses_icd`
            WHERE (icd_version = 9 AND ({ckd_icd9}))
               OR (icd_version = 10 AND ({ckd_icd10}))
        ) d2 ON d1.subject_id = d2.subject_id
    )

    SELECT 
        rx.subject_id,
        rx.hadm_id,
        rx.pharmacy_id,
        rx.starttime,
        rx.stoptime,
        rx.drug_type,
        rx.drug,
        rx.prod_strength,
        rx.form_rx,
        rx.dose_val_rx,
        rx.dose_unit_rx,
        rx.form_val_disp,
        rx.form_unit_disp,
        rx.doses_per_24_hrs,
        rx.route
    FROM `{HOSP_DATASET}.prescriptions` rx
    INNER JOIN dm_ckd_cohort c ON rx.subject_id = c.subject_id
    WHERE {med_like_clauses}
    ORDER BY rx.subject_id, rx.starttime
    """

    return run_query(client, query, "Step 5: Extracting Medications (Prescriptions)")


# =============================================================================
# STEP 6: Extract Detailed Diagnoses for Cohort
# =============================================================================

def step6_extract_diagnoses(client: bigquery.Client) -> pd.DataFrame:
    """
    Extract all diagnoses for the DM+CKD cohort for CKD staging
    and comorbidity analysis.
    """
    dm_icd9 = " OR ".join([f"icd_code LIKE '{c}'" for c in ICD9_DIABETES])
    dm_icd10 = " OR ".join([f"icd_code LIKE '{c}'" for c in ICD10_DIABETES])
    ckd_icd9 = " OR ".join([f"icd_code LIKE '{c}'" for c in ICD9_CKD])
    ckd_icd10 = " OR ".join([f"icd_code LIKE '{c}'" for c in ICD10_CKD])

    query = f"""
    WITH dm_ckd_cohort AS (
        SELECT DISTINCT d1.subject_id
        FROM (
            SELECT DISTINCT subject_id FROM `{HOSP_DATASET}.diagnoses_icd`
            WHERE (icd_version = 9 AND ({dm_icd9}))
               OR (icd_version = 10 AND ({dm_icd10}))
        ) d1
        INNER JOIN (
            SELECT DISTINCT subject_id FROM `{HOSP_DATASET}.diagnoses_icd`
            WHERE (icd_version = 9 AND ({ckd_icd9}))
               OR (icd_version = 10 AND ({ckd_icd10}))
        ) d2 ON d1.subject_id = d2.subject_id
    )

    SELECT 
        dx.subject_id,
        dx.hadm_id,
        dx.seq_num,
        dx.icd_code,
        dx.icd_version,
        d.long_title AS icd_description
    FROM `{HOSP_DATASET}.diagnoses_icd` dx
    INNER JOIN dm_ckd_cohort c ON dx.subject_id = c.subject_id
    LEFT JOIN `{HOSP_DATASET}.d_icd_diagnoses` d 
        ON dx.icd_code = d.icd_code AND dx.icd_version = d.icd_version
    ORDER BY dx.subject_id, dx.hadm_id, dx.seq_num
    """

    return run_query(client, query, "Step 6: Extracting All Diagnoses for Cohort")


# =============================================================================
# STEP 7: Summary Statistics
# =============================================================================

def compute_summary(
        cohort_df: pd.DataFrame,
        labs_df: pd.DataFrame,
        meds_df: pd.DataFrame,
        omr_df: pd.DataFrame,
) -> dict:
    """Compute and print summary statistics for the extracted data."""

    summary = {
        "total_patients": cohort_df["subject_id"].nunique() if len(cohort_df) > 0 else 0,
        "total_lab_records": len(labs_df),
        "total_medication_records": len(meds_df),
        "total_omr_records": len(omr_df),
    }

    if len(cohort_df) > 0:
        summary["gender_distribution"] = cohort_df.groupby("gender")["subject_id"].nunique().to_dict()
        summary["age_stats"] = {
            "mean": round(cohort_df["anchor_age"].mean(), 1),
            "median": round(cohort_df["anchor_age"].median(), 1),
            "min": int(cohort_df["anchor_age"].min()),
            "max": int(cohort_df["anchor_age"].max()),
        }
        summary["admissions_per_patient"] = {
            "mean": round(cohort_df["total_admissions"].mean(), 1),
            "max": int(cohort_df["total_admissions"].max()),
        }

    if len(labs_df) > 0:
        summary["lab_counts_by_type"] = labs_df.groupby("lab_name").size().sort_values(ascending=False).head(
            20).to_dict()

        # Patients with longitudinal data (>1 lab measurement over time)
        pts_with_multi_labs = (
            labs_df.groupby(["subject_id", "lab_name"])
            .size()
            .reset_index(name="count")
        )
        pts_with_multi_labs = pts_with_multi_labs[pts_with_multi_labs["count"] > 1]
        summary["patients_with_longitudinal_labs"] = pts_with_multi_labs["subject_id"].nunique()

    if len(meds_df) > 0:
        summary["top_medications"] = meds_df["drug"].value_counts().head(20).to_dict()

    print(f"\n{'=' * 60}")
    print(f"  EXTRACTION SUMMARY")
    print(f"{'=' * 60}")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"\n  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    return summary


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract Diabetic CKD cohort from MIMIC-IV via BigQuery"
    )
    parser.add_argument(
        "--project", required=True,
        help="Google Cloud Project ID with BigQuery access to MIMIC-IV"
    )
    parser.add_argument(
        "--output-dir", default="./extracted_data",
        help="Directory to save extracted CSV files"
    )
    parser.add_argument(
        "--dataset", default="physionet-data.mimiciv_hosp",
        help="BigQuery dataset (default: physionet-data.mimiciv_hosp)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print queries without executing"
    )

    args = parser.parse_args()

    global HOSP_DATASET
    HOSP_DATASET = args.dataset
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DiabetesKidney Companion - MIMIC-IV Data Extraction")
    print("=" * 60)
    print(f"  Project: {args.project}")
    print(f"  Dataset: {HOSP_DATASET}")
    print(f"  Output:  {output_dir.resolve()}")

    # Initialize client
    client = get_client(args.project)

    # Step 1: Discover lab item IDs
    lab_items_df = step1_discover_lab_items(client)
    lab_items_df.to_csv(output_dir / "lab_item_mapping.csv", index=False)
    print(f"\n  Lab items discovered:")
    for _, row in lab_items_df.iterrows():
        print(f"    itemid={row['itemid']}: {row['label']} ({row['fluid']})")

    # Step 2: Identify DM+CKD cohort
    cohort_df = step2_identify_cohort(client)
    cohort_df.to_csv(output_dir / "dm_ckd_cohort.csv", index=False)

    # Step 3: Extract longitudinal labs
    labs_df = step3_extract_labs(client, lab_items_df)
    labs_df.to_csv(output_dir / "longitudinal_labs.csv", index=False)

    # Step 4: Extract OMR data (eGFR, vitals)
    omr_df = step4_extract_omr_egfr(client)
    omr_df.to_csv(output_dir / "omr_egfr_vitals.csv", index=False)

    # Step 5: Extract medications
    meds_df = step5_extract_medications(client)
    meds_df.to_csv(output_dir / "medications.csv", index=False)

    # Step 6: Extract diagnoses
    dx_df = step6_extract_diagnoses(client)
    dx_df.to_csv(output_dir / "diagnoses.csv", index=False)

    # Step 7: Summary
    summary = compute_summary(cohort_df, labs_df, meds_df, omr_df)

    # Save summary
    import json
    with open(output_dir / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  ✓ All data saved to: {output_dir.resolve()}")
    print(f"{'=' * 60}")
    print(f"\n  Files created:")
    for f in sorted(output_dir.glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

"""
Quick Explorer — Verify BigQuery Access & Preview Data
========================================================
Run this FIRST to verify your BigQuery setup works and to
explore what's available in the MIMIC-IV tables.

Usage:
    python quick_explore.py --project YOUR_GCP_PROJECT_ID

    # For the demo dataset (no PhysioNet credentials needed):
    python quick_explore.py --project YOUR_GCP_PROJECT_ID --demo
"""

import argparse
from google.cloud import bigquery


def main():
    parser = argparse.ArgumentParser(description="Quick MIMIC-IV BigQuery explorer")
    parser.add_argument("--project", required=True, help="GCP Project ID")
    parser.add_argument("--demo", action="store_true", help="Use MIMIC-IV Demo dataset")
    args = parser.parse_args()

    dataset = "physionet-data.mimiciv_demo_hosp" if args.demo else "physionet-data.mimiciv_3_1_hosp"
    client = bigquery.Client(project=args.project)

    def q(sql, desc=""):
        if desc:
            print(f"\n--- {desc} ---")
        print(f"Query: {sql[:120]}...")
        result = client.query(sql).to_dataframe()
        print(result.to_string(max_rows=20))
        print(f"({len(result)} rows)\n")
        return result

    # 1. Check access & count patients
    q(f"SELECT COUNT(DISTINCT subject_id) as n_patients FROM `{dataset}.patients`",
      "Total patients in database")

    # 2. Preview lab items relevant to kidney/diabetes
    q(f"""
    SELECT itemid, label, fluid, category 
    FROM `{dataset}.d_labitems`
    WHERE LOWER(label) LIKE '%creatinine%'
       OR LOWER(label) LIKE '%gfr%'
       OR LOWER(label) LIKE '%a1c%'
       OR LOWER(label) LIKE '%glucose%'
       OR LOWER(label) LIKE '%potassium%'
       OR LOWER(label) LIKE '%urea%'
       OR LOWER(label) LIKE '%phosph%'
       OR LOWER(label) LIKE '%albumin%'
       OR LOWER(label) LIKE '%bicarbonate%'
    ORDER BY label
    """, "Lab items relevant to diabetes & kidney")

    # 3. Count diabetes patients
    q(f"""
    SELECT COUNT(DISTINCT subject_id) as dm_patients
    FROM `{dataset}.diagnoses_icd`
    WHERE (icd_version = 9 AND icd_code LIKE '250%')
       OR (icd_version = 10 AND (icd_code LIKE 'E10%' OR icd_code LIKE 'E11%'))
    """, "Patients with diabetes diagnosis")

    # 4. Count CKD patients
    q(f"""
    SELECT COUNT(DISTINCT subject_id) as ckd_patients
    FROM `{dataset}.diagnoses_icd`
    WHERE (icd_version = 9 AND icd_code LIKE '585%')
       OR (icd_version = 10 AND icd_code LIKE 'N18%')
    """, "Patients with CKD diagnosis")

    # 5. Count DM + CKD overlap
    q(f"""
    SELECT COUNT(DISTINCT d1.subject_id) as dm_ckd_patients
    FROM (
        SELECT DISTINCT subject_id FROM `{dataset}.diagnoses_icd`
        WHERE (icd_version = 9 AND icd_code LIKE '250%')
           OR (icd_version = 10 AND (icd_code LIKE 'E10%' OR icd_code LIKE 'E11%'))
    ) d1
    INNER JOIN (
        SELECT DISTINCT subject_id FROM `{dataset}.diagnoses_icd`
        WHERE (icd_version = 9 AND icd_code LIKE '585%')
           OR (icd_version = 10 AND icd_code LIKE 'N18%')
    ) d2 ON d1.subject_id = d2.subject_id
    """, "Patients with BOTH diabetes AND CKD")

    # 6. Preview OMR table (eGFR, vitals)
    q(f"""
    SELECT result_name, COUNT(*) as n
    FROM `{dataset}.omr`
    GROUP BY result_name
    ORDER BY n DESC
    """, "OMR table — available result types")

    # 7. Sample prescriptions for diabetes meds
    q(f"""
    SELECT drug, COUNT(*) as n
    FROM `{dataset}.prescriptions`
    WHERE LOWER(drug) LIKE '%metformin%'
       OR LOWER(drug) LIKE '%insulin%'
       OR LOWER(drug) LIKE '%glipizide%'
       OR LOWER(drug) LIKE '%empagliflozin%'
       OR LOWER(drug) LIKE '%lisinopril%'
       OR LOWER(drug) LIKE '%losartan%'
    GROUP BY drug
    ORDER BY n DESC
    LIMIT 20
    """, "Top diabetes/kidney medications in prescriptions")

    print("=" * 60)
    print("  ✓ BigQuery access verified!")
    print(f"  Dataset: {dataset}")
    print("  You're ready to run extract_diabetes_ckd_cohort.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

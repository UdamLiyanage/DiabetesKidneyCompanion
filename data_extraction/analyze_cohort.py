"""
Post-Extraction Analysis - DiabetesKidney Companion
=====================================================
Validates extracted data, computes CKD staging, and creates
the longitudinal patient profiles needed for the app.

Run AFTER extract_diabetes_ckd_cohort.py has completed.

Usage:
    python analyze_cohort.py --data-dir ./extracted_data
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np


# =============================================================================
# CKD Staging Logic (KDIGO 2012 Guidelines)
# =============================================================================

def classify_ckd_stage(egfr: float) -> str:
    """
    Classify CKD stage based on eGFR (mL/min/1.73m²).
    KDIGO 2012 classification.
    """
    if pd.isna(egfr):
        return "Unknown"
    if egfr >= 90:
        return "G1"  # Normal or high
    elif egfr >= 60:
        return "G2"  # Mildly decreased
    elif egfr >= 45:
        return "G3a"  # Mildly to moderately decreased
    elif egfr >= 30:
        return "G3b"  # Moderately to severely decreased
    elif egfr >= 15:
        return "G4"  # Severely decreased
    else:
        return "G5"  # Kidney failure


def classify_hba1c_control(hba1c: float) -> str:
    """Classify glycemic control based on HbA1c (%)."""
    if pd.isna(hba1c):
        return "Unknown"
    if hba1c < 5.7:
        return "Normal"
    elif hba1c < 6.5:
        return "Prediabetes"
    elif hba1c < 7.0:
        return "Well-controlled DM"
    elif hba1c < 8.0:
        return "Moderately controlled DM"
    elif hba1c < 9.0:
        return "Poorly controlled DM"
    else:
        return "Very poorly controlled DM"


# =============================================================================
# Medication Safety Flags Based on eGFR
# =============================================================================

RENAL_DOSE_ADJUSTMENTS = {
    "metformin": [
        {"egfr_threshold": 30, "action": "CONTRAINDICATED - discontinue metformin"},
        {"egfr_threshold": 45, "action": "CAUTION - do not initiate; consider dose reduction if already on it"},
        {"egfr_threshold": 60, "action": "MONITOR - check renal function more frequently"},
    ],
    "glyburide": [
        {"egfr_threshold": 30,
         "action": "AVOID - use glipizide instead (glyburide has active metabolites cleared by kidney)"},
    ],
    "sitagliptin": [
        {"egfr_threshold": 30, "action": "REDUCE to 25mg daily"},
        {"egfr_threshold": 45, "action": "REDUCE to 50mg daily"},
    ],
    "canagliflozin": [
        {"egfr_threshold": 30, "action": "AVOID initiating; may continue 100mg if already on it"},
        {"egfr_threshold": 45, "action": "LIMIT to 100mg daily"},
    ],
    "empagliflozin": [
        {"egfr_threshold": 20, "action": "DISCONTINUE if no heart failure indication"},
        {"egfr_threshold": 45, "action": "MONITOR - reduced glycemic efficacy but cardiorenal benefits persist"},
    ],
    "dapagliflozin": [
        {"egfr_threshold": 25, "action": "DISCONTINUE if no heart failure/CKD indication"},
        {"egfr_threshold": 45, "action": "MONITOR - may continue for cardiorenal protection"},
    ],
    "lisinopril": [
        {"egfr_threshold": 30, "action": "CAUTION - start low dose, monitor potassium and creatinine closely"},
    ],
    "ibuprofen": [
        {"egfr_threshold": 30, "action": "AVOID - high risk of acute kidney injury"},
        {"egfr_threshold": 60, "action": "CAUTION - use lowest dose for shortest duration"},
    ],
    "naproxen": [
        {"egfr_threshold": 30, "action": "AVOID - high risk of acute kidney injury"},
        {"egfr_threshold": 60, "action": "CAUTION - use lowest dose for shortest duration"},
    ],
    "gentamicin": [
        {"egfr_threshold": 60, "action": "ADJUST dose and extend interval - monitor drug levels"},
    ],
    "vancomycin": [
        {"egfr_threshold": 60, "action": "ADJUST dose - monitor trough levels closely"},
    ],
    "gabapentin": [
        {"egfr_threshold": 15, "action": "REDUCE to 100-300mg daily"},
        {"egfr_threshold": 30, "action": "REDUCE to 200-700mg daily"},
        {"egfr_threshold": 60, "action": "REDUCE to 400-1400mg daily"},
    ],
}


def flag_medication_concerns(drug_name: str, egfr: float) -> list:
    """
    Check if a medication requires dose adjustment or is contraindicated
    given the patient's current eGFR.
    """
    flags = []
    drug_lower = drug_name.lower()

    for med_key, thresholds in RENAL_DOSE_ADJUSTMENTS.items():
        if med_key in drug_lower:
            for rule in thresholds:
                if egfr < rule["egfr_threshold"]:
                    flags.append({
                        "drug": drug_name,
                        "egfr": egfr,
                        "threshold": rule["egfr_threshold"],
                        "action": rule["action"],
                    })
                    break  # Only the most severe applicable rule

    return flags


# =============================================================================
# Build Longitudinal Patient Profiles
# =============================================================================

def build_patient_profiles(
        cohort_df: pd.DataFrame,
        labs_df: pd.DataFrame,
        meds_df: pd.DataFrame,
        omr_df: pd.DataFrame,
) -> list:
    """
    Build per-patient longitudinal profiles combining labs, meds, and CKD staging.
    These profiles are what the DiabetesKidney Companion app will display.
    """
    profiles = []
    patient_ids = cohort_df["subject_id"].unique()

    print(f"\nBuilding profiles for {len(patient_ids)} patients...")

    for i, pid in enumerate(patient_ids):
        if i % 500 == 0 and i > 0:
            print(f"  Processed {i}/{len(patient_ids)} patients...")

        # Patient demographics
        pt_demo = cohort_df[cohort_df["subject_id"] == pid].iloc[0]

        # Patient labs (chronological)
        pt_labs = labs_df[labs_df["subject_id"] == pid].copy()

        # Get most recent eGFR and creatinine
        egfr_labs = pt_labs[pt_labs["lab_name"].str.contains("GFR|Creatinine", case=False, na=False)]
        creatinine_labs = pt_labs[pt_labs["lab_name"].str.contains("Creatinine", case=False, na=False)]
        hba1c_labs = pt_labs[pt_labs["lab_name"].str.contains("A1c", case=False, na=False)]

        # Also check OMR for eGFR
        pt_omr = omr_df[omr_df["subject_id"] == pid] if len(omr_df) > 0 else pd.DataFrame()

        # Get latest eGFR value
        latest_egfr = None
        if len(egfr_labs) > 0:
            latest_row = egfr_labs.sort_values("charttime", ascending=False).iloc[0]
            latest_egfr = latest_row.get("valuenum")

        # If no eGFR in labs, estimate from creatinine (CKD-EPI 2021)
        if latest_egfr is None and len(creatinine_labs) > 0:
            latest_cr = creatinine_labs.sort_values("charttime", ascending=False).iloc[0]
            cr_val = latest_cr.get("valuenum")
            if cr_val and cr_val > 0:
                # Simplified CKD-EPI 2021 (race-free)
                age = pt_demo.get("anchor_age", 60)
                sex = pt_demo.get("gender", "M")
                latest_egfr = estimate_egfr_ckd_epi_2021(cr_val, age, sex)

        # CKD stage
        ckd_stage = classify_ckd_stage(latest_egfr)

        # HbA1c control
        latest_hba1c = None
        if len(hba1c_labs) > 0:
            latest_hba1c_row = hba1c_labs.sort_values("charttime", ascending=False).iloc[0]
            latest_hba1c = latest_hba1c_row.get("valuenum")
        glycemic_control = classify_hba1c_control(latest_hba1c)

        # Medication safety flags
        pt_meds = meds_df[meds_df["subject_id"] == pid] if len(meds_df) > 0 else pd.DataFrame()
        med_flags = []
        if len(pt_meds) > 0 and latest_egfr is not None:
            for _, med_row in pt_meds.iterrows():
                flags = flag_medication_concerns(med_row["drug"], latest_egfr)
                med_flags.extend(flags)

        # Compute eGFR trajectory (if multiple measurements)
        egfr_trajectory = []
        if len(creatinine_labs) > 1:
            for _, lab_row in creatinine_labs.sort_values("charttime").iterrows():
                cr = lab_row.get("valuenum")
                if cr and cr > 0:
                    age = pt_demo.get("anchor_age", 60)
                    sex = pt_demo.get("gender", "M")
                    egfr_est = estimate_egfr_ckd_epi_2021(cr, age, sex)
                    egfr_trajectory.append({
                        "charttime": str(lab_row["charttime"]),
                        "creatinine": cr,
                        "egfr_estimated": round(egfr_est, 1),
                        "ckd_stage": classify_ckd_stage(egfr_est),
                    })

        # Number of unique lab tests
        lab_types = pt_labs["lab_name"].nunique() if len(pt_labs) > 0 else 0
        lab_count = len(pt_labs)

        profile = {
            "subject_id": int(pid),
            "gender": pt_demo.get("gender"),
            "anchor_age": int(pt_demo.get("anchor_age", 0)),
            "total_admissions": int(pt_demo.get("total_admissions", 0)),
            "latest_egfr": round(latest_egfr, 1) if latest_egfr else None,
            "ckd_stage": ckd_stage,
            "latest_hba1c": round(latest_hba1c, 1) if latest_hba1c else None,
            "glycemic_control": glycemic_control,
            "total_lab_records": lab_count,
            "unique_lab_types": lab_types,
            "total_medications": len(pt_meds),
            "medication_safety_flags": med_flags,
            "egfr_trajectory_points": len(egfr_trajectory),
            "egfr_trajectory": egfr_trajectory[:50],  # Cap at 50 points
        }

        profiles.append(profile)

    return profiles


def estimate_egfr_ckd_epi_2021(creatinine: float, age: int, sex: str) -> float:
    """
    CKD-EPI 2021 equation (race-free).
    Creatinine in mg/dL, age in years.
    Returns eGFR in mL/min/1.73m².
    """
    if sex.upper() in ("F", "FEMALE"):
        if creatinine <= 0.7:
            egfr = 142 * (creatinine / 0.7) ** (-0.241) * (0.9938 ** age) * 1.012
        else:
            egfr = 142 * (creatinine / 0.7) ** (-1.200) * (0.9938 ** age) * 1.012
    else:
        if creatinine <= 0.9:
            egfr = 142 * (creatinine / 0.9) ** (-0.302) * (0.9938 ** age)
        else:
            egfr = 142 * (creatinine / 0.9) ** (-1.200) * (0.9938 ** age)

    return max(egfr, 0)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze extracted MIMIC-IV cohort")
    parser.add_argument("--data-dir", default="./extracted_data", help="Directory with extracted CSVs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("  DiabetesKidney Companion - Cohort Analysis")
    print("=" * 60)

    # Load data
    print("\nLoading extracted data...")
    cohort_df = pd.read_csv(data_dir / "dm_ckd_cohort.csv")
    labs_df = pd.read_csv(data_dir / "longitudinal_labs.csv")
    meds_df = pd.read_csv(data_dir / "medications.csv")

    omr_path = data_dir / "omr_egfr_vitals.csv"
    omr_df = pd.read_csv(omr_path) if omr_path.exists() else pd.DataFrame()

    print(f"  Cohort:      {len(cohort_df):,} rows ({cohort_df['subject_id'].nunique():,} patients)")
    print(f"  Labs:        {len(labs_df):,} rows")
    print(f"  Medications: {len(meds_df):,} rows")
    print(f"  OMR:         {len(omr_df):,} rows")

    # Build patient profiles
    profiles = build_patient_profiles(cohort_df, labs_df, meds_df, omr_df)

    # Save profiles
    profiles_df = pd.DataFrame(profiles)
    profiles_df.to_csv(data_dir / "patient_profiles.csv", index=False)

    # Save full profiles as JSON (for the app)
    with open(data_dir / "patient_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2, default=str)

    # Summary statistics
    print(f"\n{'=' * 60}")
    print(f"  COHORT ANALYSIS RESULTS")
    print(f"{'=' * 60}")

    print(f"\n  Total patients: {len(profiles)}")

    # CKD stage distribution
    ckd_dist = profiles_df["ckd_stage"].value_counts()
    print(f"\n  CKD Stage Distribution:")
    for stage, count in ckd_dist.items():
        pct = count / len(profiles) * 100
        print(f"    {stage}: {count} ({pct:.1f}%)")

    # Glycemic control
    glyc_dist = profiles_df["glycemic_control"].value_counts()
    print(f"\n  Glycemic Control Distribution:")
    for level, count in glyc_dist.items():
        pct = count / len(profiles) * 100
        print(f"    {level}: {count} ({pct:.1f}%)")

    # Patients with medication safety flags
    pts_with_flags = profiles_df[profiles_df["medication_safety_flags"].apply(
        lambda x: len(x) > 0 if isinstance(x, list) else False
    )]
    print(
        f"\n  Patients with medication safety flags: {len(pts_with_flags)} ({len(pts_with_flags) / len(profiles) * 100:.1f}%)")

    # Patients with eGFR trajectory
    pts_with_traj = profiles_df[profiles_df["egfr_trajectory_points"] > 1]
    print(
        f"  Patients with eGFR trajectory (>1 point): {len(pts_with_traj)} ({len(pts_with_traj) / len(profiles) * 100:.1f}%)")

    print(f"\n  Output files:")
    print(f"    {data_dir / 'patient_profiles.csv'}")
    print(f"    {data_dir / 'patient_profiles.json'}")
    print(f"\n  ✓ Analysis complete!")


if __name__ == "__main__":
    main()

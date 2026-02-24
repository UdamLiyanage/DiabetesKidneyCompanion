#!/usr/bin/env python3
"""
medication_alerts.py — Rule-based KDIGO medication safety alerts.

Reads extraction JSONs from ./output/extractions/, applies clinical alert rules,
writes one alert JSON per patient to ./output/alerts/.

No GPU required — pure rule logic.
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent
EXTR_DIR     = PROJECT_DIR / "output" / "extractions"
ALERTS_DIR   = PROJECT_DIR / "output" / "alerts"
SUMMARY_PATH = PROJECT_DIR / "output" / "alerts_summary.json"
ALERTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Lab name normalisation (mirrors batch_extraction.py) ─

_LAB_ALIASES = {
    "egfr": [
        "egfr", "estimated gfr", "estimated glomerular filtration rate",
        "egfr (ckd-epi 2021)", "gfr", "egfr (ckd-epi)",
    ],
    "creatinine": ["creatinine", "creatinine, serum"],
    "potassium":  ["potassium", "potassium, serum", "k"],
    "hba1c": [
        "hba1c", "hemoglobin a1c", "hba1c (%)", "glycated hemoglobin",
        "a1c", "hemoglobin a1c (hba1c)", "% hemoglobin a1c",
    ],
    "glucose": [
        "glucose", "glucose (fasting)", "glucose, fasting", "fasting glucose",
    ],
    "bun": [
        "bun", "bun (urea)", "blood urea nitrogen", "blood urea nitrogen (bun)",
        "urea nitrogen", "urea", "bun/urea",
    ],
    "sodium":    ["sodium", "sodium, serum", "na"],
    "bicarbonate": [
        "bicarbonate", "co2", "co2 (bicarbonate)", "carbon dioxide",
        "total co2", "co2, total",
    ],
}

def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9/ ()-]", "", name.lower().strip())

# Build reverse alias → canonical key map once at import time
_ALIAS_MAP: dict[str, str] = {}
for _key, _aliases in _LAB_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_MAP[_norm(_alias)] = _key


def extract_lab_values(lab_results: list) -> dict[str, float | None]:
    """
    Convert lab_results list into {canonical_key: float_value} dict.
    Returns None for any key not found or not parseable.
    """
    values: dict[str, float | None] = {k: None for k in _LAB_ALIASES}
    for lab in lab_results:
        norm = _norm(lab.get("test_name", ""))
        key  = _ALIAS_MAP.get(norm)
        if key is None:
            continue
        raw = lab.get("value")
        try:
            values[key] = float(raw)
        except (TypeError, ValueError):
            pass
    return values


# ── CKD staging ────────────────────────────────────────────

def ckd_stage(egfr: float | None) -> str:
    """Return KDIGO CKD GFR category string, or 'unknown'."""
    if egfr is None:
        return "unknown"
    if egfr >= 90:
        return "G1"
    if egfr >= 60:
        return "G2"
    if egfr >= 45:
        return "G3a"
    if egfr >= 30:
        return "G3b"
    if egfr >= 15:
        return "G4"
    return "G5"


# ── Glycaemic control ──────────────────────────────────────

def glycemic_control(hba1c: float | None) -> str:
    """Return 'good', 'moderate', 'poor', or 'unknown'."""
    if hba1c is None:
        return "unknown"
    if hba1c < 7.0:
        return "good"
    if hba1c <= 8.0:
        return "moderate"
    return "poor"


# ── Alert builders ─────────────────────────────────────────
# Each builder returns a list of alert dicts (may be empty).
# Severity: "warning" > "caution" > "info"

def _alert(category, severity, medication_class, message, clinical_basis):
    return {
        "category":        category,
        "severity":        severity,
        "medication_class": medication_class,
        "message":         message,
        "clinical_basis":  clinical_basis,
    }


def alerts_egfr_contraindications(egfr: float | None) -> list:
    if egfr is None:
        return []
    alerts = []
    basis  = f"eGFR {egfr:.0f} mL/min/1.73m²"

    if egfr < 15:
        alerts.append(_alert(
            "medication_contraindication", "warning",
            "oral_diabetes_agents",
            "Most oral diabetes medications are contraindicated at eGFR <15. "
            "Insulin is preferred; consult endocrinology.",
            basis,
        ))
    if egfr < 30:
        alerts.append(_alert(
            "medication_contraindication", "warning",
            "metformin",
            "Metformin is contraindicated at eGFR <30 due to risk of lactic acidosis. "
            "Discontinue and reassess glycaemic management.",
            basis,
        ))
        alerts.append(_alert(
            "medication_contraindication", "caution",
            "sglt2_inhibitors",
            "SGLT2 inhibitors have reduced efficacy and require review at eGFR <30. "
            "Most agents should be discontinued.",
            basis,
        ))
    elif egfr < 45:
        alerts.append(_alert(
            "medication_contraindication", "caution",
            "metformin",
            "Metformin dose reduction required at eGFR 30–44. "
            "Maximum dose 1000 mg/day; monitor renal function every 3–6 months.",
            basis,
        ))

    return alerts


def alerts_nephrotoxic(egfr: float | None) -> list:
    if egfr is None:
        return []
    alerts = []
    basis  = f"eGFR {egfr:.0f} mL/min/1.73m²"

    if egfr < 60:
        alerts.append(_alert(
            "nephrotoxic_drug_warning", "warning",
            "nsaids",
            "NSAIDs should be avoided in CKD (eGFR <60). "
            "Use paracetamol for analgesia; avoid ibuprofen, naproxen, diclofenac.",
            basis,
        ))
    if egfr < 30:
        alerts.append(_alert(
            "nephrotoxic_drug_warning", "caution",
            "aminoglycosides",
            "Aminoglycosides require careful dose adjustment and therapeutic drug "
            "monitoring at eGFR <30. Avoid if alternatives exist.",
            basis,
        ))

    # Contrast dye applies at any CKD stage
    alerts.append(_alert(
        "nephrotoxic_drug_warning", "info",
        "iodinated_contrast",
        "Any CKD: iodinated contrast dye requires hydration protocol and "
        "pre/post-procedure renal function monitoring.",
        basis,
    ))

    return alerts


def alerts_potassium(potassium: float | None) -> list:
    if potassium is None:
        return []
    alerts = []
    basis  = f"K⁺ {potassium:.1f} mmol/L"

    if potassium > 6.0:
        alerts.append(_alert(
            "electrolyte_medication_interaction", "warning",
            "potassium_sparing_drugs",
            f"K⁺ {potassium:.1f} mmol/L — urgent review required. "
            "Hold potassium-sparing diuretics, ACE inhibitors, and ARBs until "
            "potassium is corrected.",
            basis,
        ))
    elif potassium > 5.5:
        alerts.append(_alert(
            "electrolyte_medication_interaction", "caution",
            "ace_inhibitors_arbs",
            f"K⁺ {potassium:.1f} mmol/L — ACE inhibitors/ARBs may need dose "
            "reduction. Recheck potassium within 1 week.",
            basis,
        ))
    elif potassium > 5.0:
        alerts.append(_alert(
            "electrolyte_medication_interaction", "info",
            "ace_inhibitors_arbs",
            f"K⁺ {potassium:.1f} mmol/L — ACE inhibitors/ARBs require close "
            "monitoring. Avoid additional potassium-raising agents.",
            basis,
        ))

    return alerts


def alerts_ckd_monitoring(stage: str) -> list:
    alerts = []

    if stage in ("G3a", "G3b"):
        alerts.append(_alert(
            "ckd_monitoring", "info",
            None,
            "CKD G3: quarterly monitoring of renal function, electrolytes, "
            "blood pressure, and urine ACR recommended.",
            f"CKD stage {stage}",
        ))
    elif stage == "G4":
        alerts.append(_alert(
            "ckd_monitoring", "caution",
            None,
            "CKD G4: monthly monitoring recommended. Nephrology referral indicated "
            "for dialysis preparation and medication review.",
            f"CKD stage {stage}",
        ))
    elif stage == "G5":
        alerts.append(_alert(
            "ckd_monitoring", "warning",
            None,
            "CKD G5: urgent nephrology referral required. Initiate dialysis "
            "planning discussion and intensive medication review.",
            f"CKD stage {stage}",
        ))

    return alerts


def alerts_glycemic_ckd(
    hba1c:    float | None,
    control:  str,
    stage:    str,
    egfr:     float | None,
) -> list:
    alerts = []

    # Insulin preference in advanced CKD with poor control
    if stage in ("G4", "G5") and control == "poor":
        alerts.append(_alert(
            "glycemic_ckd_interaction", "caution",
            "oral_diabetes_agents",
            "Poor glycaemic control with CKD G4–G5: insulin is preferred over "
            "oral agents. Most oral medications are contraindicated or less effective "
            "at this level of kidney function.",
            f"CKD stage {stage}, HbA1c {hba1c:.1f}%" if hba1c else f"CKD stage {stage}",
        ))

    # Relaxed HbA1c target in advanced CKD
    if stage in ("G4", "G5") and hba1c is not None and hba1c < 7.0:
        alerts.append(_alert(
            "glycemic_ckd_interaction", "info",
            None,
            "In advanced CKD (G4–G5) an HbA1c target <8% is acceptable to reduce "
            "hypoglycaemia risk. Tight control (<7%) may not be appropriate.",
            f"HbA1c {hba1c:.1f}%, CKD stage {stage}",
        ))

    # Hypoglycaemia risk warning
    if egfr is not None and egfr < 45 and control in ("good", "moderate"):
        alerts.append(_alert(
            "glycemic_ckd_interaction", "caution",
            "insulin_sulfonylureas",
            "Hypoglycaemia risk increases as eGFR falls. Reduced insulin clearance "
            "and impaired gluconeogenesis at eGFR <45 heighten risk. "
            "Review insulin doses and avoid long-acting sulfonylureas.",
            f"eGFR {egfr:.0f} mL/min/1.73m²",
        ))

    return alerts


# ── Per-patient alert generation ───────────────────────────

def generate_alerts(extraction_record: dict) -> dict:
    """
    Given a single extraction record (as saved by 2_batch_extraction.py),
    return a complete alert document for that patient.
    """
    patient_id = extraction_record.get("patient_id", "unknown")
    extraction  = extraction_record.get("extraction", {})
    lab_results = extraction.get("lab_results", [])

    labs    = extract_lab_values(lab_results)
    egfr    = labs["egfr"]
    potassium = labs["potassium"]
    hba1c   = labs["hba1c"]

    stage   = ckd_stage(egfr)
    control = glycemic_control(hba1c)

    alerts: list = []
    alerts += alerts_egfr_contraindications(egfr)
    alerts += alerts_nephrotoxic(egfr)
    alerts += alerts_potassium(potassium)
    alerts += alerts_ckd_monitoring(stage)
    alerts += alerts_glycemic_ckd(hba1c, control, stage, egfr)

    # Sort: warning first, then caution, then info
    _order = {"warning": 0, "caution": 1, "info": 2}
    alerts.sort(key=lambda a: _order.get(a["severity"], 9))

    return {
        "patient_id":       patient_id,
        "ckd_stage":        stage,
        "egfr":             egfr,
        "potassium":        potassium,
        "hba1c":            hba1c,
        "glycemic_control": control,
        "alert_count":      len(alerts),
        "alerts":           alerts,
        "generated_at":     datetime.now(timezone.utc).isoformat(),
    }


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DiabetesKidney Companion — Medication Safety Alerts")
    print("=" * 60)

    extraction_files = sorted(EXTR_DIR.glob("*_extraction.json"))
    if not extraction_files:
        print(f"ERROR: No extraction files found in {EXTR_DIR}")
        sys.exit(1)

    print(f"Found {len(extraction_files)} extraction files\n")

    n_processed = n_skipped = n_failed = 0
    severity_counts = {"warning": 0, "caution": 0, "info": 0}
    category_counts: dict[str, int] = {}
    stage_counts:    dict[str, int] = {}
    total_alerts = 0

    for extr_path in extraction_files:
        try:
            with open(extr_path) as f:
                record = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  SKIP {extr_path.name}: could not read ({e})")
            n_skipped += 1
            continue

        if record.get("parse_error"):
            n_skipped += 1
            continue

        alert_doc = generate_alerts(record)

        out_path = ALERTS_DIR / f"{alert_doc['patient_id']}_alerts.json"
        with open(out_path, "w") as f:
            json.dump(alert_doc, f, indent=2)

        # Aggregate stats
        stage_counts[alert_doc["ckd_stage"]] = (
            stage_counts.get(alert_doc["ckd_stage"], 0) + 1
        )
        for alert in alert_doc["alerts"]:
            severity_counts[alert["severity"]] = (
                severity_counts.get(alert["severity"], 0) + 1
            )
            cat = alert["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
            total_alerts += 1

        n_processed += 1

    # ── Summary ────────────────────────────────────────────
    summary = {
        "total_extractions":  len(extraction_files),
        "processed":          n_processed,
        "skipped_parse_error": n_skipped,
        "total_alerts":       total_alerts,
        "avg_alerts_per_patient": round(total_alerts / max(n_processed, 1), 1),
        "by_severity":        severity_counts,
        "by_category":        category_counts,
        "by_ckd_stage":       dict(sorted(stage_counts.items())),
        "generated_at":       datetime.now(timezone.utc).isoformat(),
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Processed:  {n_processed} patients")
    print(f"Skipped:    {n_skipped}  (parse errors / unreadable)")
    print(f"Total alerts generated: {total_alerts}")
    print(f"Avg alerts per patient: {summary['avg_alerts_per_patient']}")

    print(f"\n{'Severity':<10} {'Count':>6}")
    print("─" * 18)
    for sev in ("warning", "caution", "info"):
        print(f"  {sev:<8} {severity_counts.get(sev, 0):>6}")

    print(f"\n{'Category':<38} {'Count':>6}")
    print("─" * 46)
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<36} {count:>6}")

    print(f"\n{'CKD Stage':<12} {'Patients':>8}")
    print("─" * 22)
    for stage, count in sorted(stage_counts.items()):
        print(f"  {stage:<10} {count:>8}")

    print(f"\nAlert JSONs:  {ALERTS_DIR}")
    print(f"Summary:      {SUMMARY_PATH}")

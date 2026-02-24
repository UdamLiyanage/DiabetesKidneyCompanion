#!/usr/bin/env python3
"""
patient_explanations.py — MedGemma-generated patient-friendly lab explanations.

Reads extraction + alert JSONs, calls MedGemma 27B to produce plain-language
summaries, parses the response into structured sections, saves one JSON per patient.

Run AFTER scripts batch_extraction and medication_alerts have completed.
Resumable: already-written explanation files are skipped.
"""

import os
import sys

# Must be set before importing transformers/torch hub.
os.environ["HF_HOME"] = "/data/gemma/.cache/huggingface"

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────
PROJECT_DIR   = Path(__file__).parent
OUTPUT_DIR    = PROJECT_DIR / "output"
EXTR_DIR      = OUTPUT_DIR / "extractions"
ALERTS_DIR    = OUTPUT_DIR / "alerts"
EXPL_DIR      = OUTPUT_DIR / "explanations"
ERROR_LOG     = OUTPUT_DIR / "explanation_errors.log"
SUMMARY_PATH  = OUTPUT_DIR / "explanations_summary.json"
CONFIG_PATH   = OUTPUT_DIR / "config.json"

EXPL_DIR.mkdir(parents=True, exist_ok=True)

MODELS_TO_TRY = [
    "google/medgemma-27b-it",
    "google/medgemma-1.5-4b-it",
]

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    filename=str(ERROR_LOG),
    level=logging.ERROR,
    format="%(asctime)s  %(levelname)s  %(message)s",
)


# ── Model loading ──────────────────────────────────────────

def load_model(preferred_model_id=None):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"GPU:       {torch.cuda.get_device_name(0)}")
    print(f"VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"VRAM free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB")

    order = MODELS_TO_TRY[:]
    if preferred_model_id and preferred_model_id in order:
        order.insert(0, order.pop(order.index(preferred_model_id)))

    for model_id in order:
        print(f"\nTrying to load {model_id}...")
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",
            )
            processor = AutoProcessor.from_pretrained(model_id)
            vram_used = torch.cuda.memory_allocated() / 1e9
            print(f"  Loaded. VRAM used: {vram_used:.1f} GB")
            return model, processor, model_id
        except Exception as e:
            print(f"  Failed: {e}")
            torch.cuda.empty_cache()

    print("ERROR: No model could be loaded!")
    sys.exit(1)


# ── Prompt construction ────────────────────────────────────

_SEVERITY_LABELS = {"warning": "Important", "caution": "Note", "info": "FYI"}

def build_prompt(extr: dict, alert_doc: dict | None) -> str:
    """Construct the patient-educator prompt from extraction and alert data."""
    labs = extr.get("extraction", {}).get("lab_results", [])

    def find(names):
        for lab in labs:
            if lab.get("test_name", "").lower().strip() in names:
                v = lab.get("value")
                return str(v) if v is not None else "not measured"
        return "not measured"

    egfr_val  = find({"egfr", "estimated gfr", "gfr", "egfr (ckd-epi 2021)", "egfr (ckd-epi)"})
    hba1c_val = find({"hba1c", "hemoglobin a1c", "a1c", "hba1c (%)", "glycated hemoglobin"})
    creat_val = find({"creatinine", "creatinine, serum"})
    potk_val  = find({"potassium", "potassium, serum", "k"})

    stage   = (alert_doc or {}).get("ckd_stage", "unknown")
    control = (alert_doc or {}).get("glycemic_control", "unknown")

    # Format alerts as a numbered list; omit section if none
    alert_lines = []
    for i, alert in enumerate((alert_doc or {}).get("alerts", []), 1):
        label = _SEVERITY_LABELS.get(alert.get("severity", "info"), "Note")
        msg   = alert.get("message", "")
        alert_lines.append(f"  {i}. [{label}] {msg}")
    alerts_block = "\n".join(alert_lines) if alert_lines else "  No active alerts."

    return f"""\
You are a patient educator helping someone understand their lab results.
Explain in simple, reassuring language that a high school graduate would understand.
Avoid medical jargon. Use analogies where helpful.

Patient's Lab Results:
  Kidney function (eGFR): {egfr_val} (Stage: {stage})
  Blood sugar control (HbA1c): {hba1c_val}%
  Creatinine: {creat_val}
  Potassium: {potk_val}
  Glycaemic control category: {control}

Active Alerts:
{alerts_block}

Please provide exactly these four sections using the exact headers shown:

OVERALL SUMMARY:
[2-3 sentences on overall kidney and diabetes status]

YOUR NUMBERS EXPLAINED:
- Kidney function (eGFR): [what this number means day-to-day]
- Blood sugar control (HbA1c): [plain-language meaning]
- Creatinine: [plain-language meaning]
- Potassium: [plain-language meaning]

ALERTS — WHAT THEY MEAN FOR YOU:
[One sentence per alert explaining why it matters and a question to ask the doctor]

ONE POSITIVE STEP:
[A single, specific, encouraging action the patient can take]

Keep the total response under 300 words. Be warm and supportive."""


# ── Model inference ────────────────────────────────────────

def generate_explanation(prompt: str, model, processor,
                         max_new_tokens: int = 512) -> str:
    """Call MedGemma with a text-only prompt and return the raw response string."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
        )

    response = processor.decode(output[0][input_len:], skip_special_tokens=True)
    if not response.strip():
        # Fallback: decode with special tokens and strip tags
        response = processor.decode(output[0][input_len:], skip_special_tokens=False)
        response = re.sub(r'<[a-z_/][^>]*>', '', response).strip()

    return response


# ── Response parsing ───────────────────────────────────────

def _extract_section(text: str, header: str, next_headers: list[str]) -> str:
    """Extract the content between `header` and the next known header."""
    pattern = re.escape(header) + r"\s*[:\-]?\s*"
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return ""
    start = m.end()
    # Find where the next section begins
    end = len(text)
    for nh in next_headers:
        nm = re.search(re.escape(nh) + r"\s*[:\-]?\s*", text[start:], re.IGNORECASE)
        if nm:
            end = min(end, start + nm.start())
    return text[start:end].strip()


_HEADERS = [
    "OVERALL SUMMARY",
    "YOUR NUMBERS EXPLAINED",
    "ALERTS — WHAT THEY MEAN FOR YOU",
    "ONE POSITIVE STEP",
]


def parse_response(raw: str) -> dict:
    """
    Split MedGemma's response into structured sections.
    Falls back to storing the full raw text if sections cannot be found.
    """
    # Summary
    summary_text = _extract_section(raw, "OVERALL SUMMARY", _HEADERS[1:])

    # Lab explanations — parse bullet lines
    numbers_block = _extract_section(raw, "YOUR NUMBERS EXPLAINED", _HEADERS[2:])
    lab_explanations: dict[str, str] = {}
    field_map = {
        "egfr":        ["kidney function", "egfr"],
        "hba1c":       ["blood sugar control", "hba1c", "a1c"],
        "creatinine":  ["creatinine"],
        "potassium":   ["potassium"],
    }
    for line in numbers_block.splitlines():
        line = line.strip().lstrip("-•*").strip()
        lower = line.lower()
        for key, keywords in field_map.items():
            if key not in lab_explanations and any(kw in lower for kw in keywords):
                # Strip the label prefix up to the first colon
                explanation = re.sub(r"^[^:]+:\s*", "", line).strip()
                lab_explanations[key] = explanation or line
                break

    # Alert explanations — one sentence per line, strip leading bullets
    alerts_block = _extract_section(
        raw, "ALERTS — WHAT THEY MEAN FOR YOU", _HEADERS[3:]
    )
    alert_explanations = [
        re.sub(r"^\d+\.\s*", "", ln).lstrip("-•*").strip()
        for ln in alerts_block.splitlines()
        if ln.strip() and not ln.strip().startswith("ALERTS")
    ]

    # Action step — take the first non-empty line
    step_block = _extract_section(raw, "ONE POSITIVE STEP", [])
    action_step = next(
        (ln.strip() for ln in step_block.splitlines() if ln.strip()), ""
    )

    # If parsing found nothing useful, store the whole response as the summary
    if not summary_text and not lab_explanations and not alert_explanations:
        return {
            "summary":           raw.strip(),
            "lab_explanations":  {},
            "alert_explanations": [],
            "action_step":       "",
            "_parse_fallback":   True,
        }

    return {
        "summary":           summary_text,
        "lab_explanations":  lab_explanations,
        "alert_explanations": alert_explanations,
        "action_step":       action_step,
    }


# ── Per-patient processing ─────────────────────────────────

def already_done(patient_id: str) -> bool:
    return (EXPL_DIR / f"{patient_id}_explanation.json").exists()


def process_patient(patient_id: str, extr: dict, alert_doc: dict | None,
                    model, processor, model_id: str) -> dict:
    prompt   = build_prompt(extr, alert_doc)
    t0       = time.time()
    raw      = generate_explanation(prompt, model, processor)
    elapsed  = time.time() - t0

    parsed   = parse_response(raw)

    doc = {
        "patient_id":         patient_id,
        "summary":            parsed["summary"],
        "lab_explanations":   parsed["lab_explanations"],
        "alert_explanations": parsed["alert_explanations"],
        "action_step":        parsed["action_step"],
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "model":              model_id.split("/")[-1],
        "elapsed_s":          round(elapsed, 2),
    }
    if parsed.get("_parse_fallback"):
        doc["_parse_fallback"] = True

    out_path = EXPL_DIR / f"{patient_id}_explanation.json"
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    return doc


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DiabetesKidney Companion — Patient Explanations")
    print("=" * 60)

    # Read preferred model from config written by script 1
    preferred = None
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            preferred = json.load(f).get("model_id")
        print(f"Config found. Preferred model: {preferred}")

    model, processor, model_id = load_model(preferred_model_id=preferred)

    # Collect patients that have a successful extraction
    extr_files = sorted(EXTR_DIR.glob("*_extraction.json"))
    if not extr_files:
        print(f"ERROR: No extraction files in {EXTR_DIR}")
        sys.exit(1)

    patients = []
    for path in extr_files:
        try:
            with open(path) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if rec.get("parse_error"):
            continue
        patients.append((path.stem.replace("_extraction", ""), rec))

    print(f"\nEligible patients: {len(patients)}")

    pending = [(pid, rec) for pid, rec in patients if not already_done(pid)]
    print(f"Already explained:  {len(patients) - len(pending)}")
    print(f"To process:         {len(pending)}")

    if not pending:
        print("\nAll patients already explained. Delete files in output/explanations/ to redo.")
        sys.exit(0)

    n_ok = n_fail = 0
    elapsed_times = []

    for patient_id, extr in tqdm(pending, desc="Explaining", unit="patient"):
        # Load alert doc if available
        alert_path = ALERTS_DIR / f"{patient_id}_alerts.json"
        alert_doc  = None
        if alert_path.exists():
            try:
                with open(alert_path) as f:
                    alert_doc = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        try:
            doc = process_patient(
                patient_id, extr, alert_doc, model, processor, model_id
            )
            n_ok += 1
            elapsed_times.append(doc["elapsed_s"])
            fallback_note = "  [parse fallback]" if doc.get("_parse_fallback") else ""
            tqdm.write(f"  {patient_id}  {doc['elapsed_s']:.1f}s{fallback_note}")

        except Exception as exc:
            n_fail += 1
            msg = f"{patient_id}: {exc}"
            tqdm.write(f"  FAIL {msg}")
            logging.error(msg, exc_info=True)

    # ── Summary ────────────────────────────────────────────
    avg_time = round(sum(elapsed_times) / len(elapsed_times), 2) if elapsed_times else None

    summary = {
        "model":              model_id,
        "total_eligible":     len(patients),
        "newly_generated":    n_ok,
        "failed":             n_fail,
        "already_existed":    len(patients) - len(pending),
        "success_rate_pct":   round(100 * n_ok / max(len(pending), 1), 1),
        "avg_time_per_patient_s": avg_time,
        "generated_at":       datetime.now(timezone.utc).isoformat(),
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Model:            {model_id}")
    print(f"Newly generated:  {n_ok}")
    print(f"Failed:           {n_fail}  (see {ERROR_LOG.name})")
    print(f"Success rate:     {summary['success_rate_pct']}%")
    if avg_time:
        print(f"Avg time/patient: {avg_time}s")
    print(f"\nExplanation JSONs: {EXPL_DIR}")
    print(f"Summary:           {SUMMARY_PATH}")
    if n_fail:
        print(f"Error log:         {ERROR_LOG}")

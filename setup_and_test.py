#!/usr/bin/env python3
"""
setup_and_test.py — MedGemma setup and single-image extraction test.
"""

import os
import sys

# Redirect all Hugging Face downloads away from root (~/.cache) to /data/gemma.
# Must be set before importing transformers/torch hub.
os.environ["HF_HOME"] = "/data/gemma/.cache/huggingface"

import json
import time
import re
import torch
from pathlib import Path
from PIL import Image

# ── Configuration ──────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "synthetic_output"
IMAGE_DIR = DATA_DIR / "images"
GT_DIR = DATA_DIR / "patients"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Model preference order
MODELS_TO_TRY = [
    "google/medgemma-27b-it",      # Best quality, needs ~55GB VRAM
    "google/medgemma-1.5-4b-it",   # Lighter fallback
]


def select_and_load_model():
    """Try loading models in preference order."""
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"VRAM free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB")

    for model_id in MODELS_TO_TRY:
        print(f"\nTrying to load {model_id}...")
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",
            )
            processor = AutoProcessor.from_pretrained(model_id)

            vram_used = torch.cuda.memory_allocated() / 1e9
            print(f"  Loaded successfully. VRAM used: {vram_used:.1f} GB")
            return model, processor, model_id

        except Exception as e:
            print(f"  Failed: {e}")
            torch.cuda.empty_cache()
            continue

    print("ERROR: No model could be loaded!")
    sys.exit(1)


# ── Extraction components ──────────────────────────────────

EXTRACTION_PROMPT = """\
Extract all lab test results from this clinical lab report image.
Return ONLY a JSON object — no thinking, no explanation, no markdown fences.

Required JSON structure:
{"patient_info":{"name":"...","age":0,"sex":"M or F","patient_id":"...","report_date":"..."},"lab_results":[{"test_name":"...","value":0,"unit":"...","reference_range":"...","flag":"normal or high or low"}],"medications_listed":["..."]}

Rules:
- Every test row on the report must appear in lab_results
- Use exact numeric values as printed
- flag: "high" if marked H, "low" if marked L, "normal" otherwise
- Output raw JSON only. Do NOT wrap in markdown. Do NOT explain."""


def resize_image(image, max_size=1024):
    """Resize so longest side is max_size."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def strip_thinking(response):
    """Remove MedGemma's internal thinking block, keeping only the JSON output."""
    candidates = []
    for m in re.finditer(r'\{', response):
        snippet = response[m.start():m.start()+50].lower()
        if '"patient_info"' in snippet:
            candidates.append(m.start())
    if candidates:
        return response[candidates[-1]:]
    cleaned = re.sub(r'<unused\d+>thought.*?(?=\{)', '', response, flags=re.DOTALL)
    return cleaned.strip()


def parse_json_response(response):
    """Parse JSON from response, handling common issues."""
    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
        fixed = re.sub(r",\s*([}\]])", r"\1", match.group())
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    return {"_parse_error": True, "_raw_response": response}


def extract_lab_data(image_path, model, processor, max_new_tokens=4096):
    """Extract structured lab data from a report image."""
    image = Image.open(image_path).convert("RGB")
    image = resize_image(image, max_size=1024)

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": EXTRACTION_PROMPT},
        ]}
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
            do_sample=False,
        )

    response = processor.decode(output[0][input_len:], skip_special_tokens=True)
    response = strip_thinking(response)
    parsed = parse_json_response(response)
    return parsed, response


# ── Ground truth comparison ────────────────────────────────

NAME_MAP = {
    "glucose":       ["glucose", "glucose (fasting)", "glucose, fasting", "fasting glucose"],
    "bun":           ["bun", "blood urea nitrogen", "blood urea nitrogen (bun)", "urea nitrogen"],
    "creatinine":    ["creatinine", "creatinine, serum"],
    "egfr":          ["egfr", "estimated gfr", "estimated glomerular filtration rate",
                      "egfr (ckd-epi 2021)", "gfr", "egfr (ckd-epi)"],
    "sodium":        ["sodium", "sodium, serum", "na"],
    "potassium":     ["potassium", "potassium, serum", "k"],
    "chloride":      ["chloride", "chloride, serum", "cl"],
    "bicarbonate":   ["bicarbonate", "co2", "co2 (bicarbonate)", "carbon dioxide",
                      "total co2", "co2, total"],
    "calcium":       ["calcium", "calcium, total", "calcium, serum"],
    "phosphate":     ["phosphate", "phosphorus", "phosphorus, serum", "inorganic phosphorus"],
    "albumin":       ["albumin", "albumin, serum"],
    "total_protein": ["total protein", "protein, total", "total protein, serum"],
    "alt":           ["alt", "alt (sgpt)", "alanine aminotransferase", "sgpt"],
    "ast":           ["ast", "ast (sgot)", "aspartate aminotransferase", "sgot"],
    "hemoglobin":    ["hemoglobin", "hemoglobin, blood", "hgb", "hb"],
    "hba1c":         ["hba1c", "hemoglobin a1c", "hba1c (%)", "glycated hemoglobin",
                      "a1c", "hemoglobin a1c (hba1c)", "% hemoglobin a1c"],
    "uacr":          ["uacr", "urine albumin-to-creatinine ratio",
                      "albumin/creatinine ratio", "urine albumin/creatinine",
                      "urine acr", "albumin-to-creatinine ratio", "urine alb/creat ratio"],
    "pth":           ["pth", "intact pth", "parathyroid hormone",
                      "intact parathyroid hormone"],
}

def _normalise(name):
    return re.sub(r"[^a-z0-9/ ()-]", "", name.lower().strip())

def match_extracted_to_gt(extracted_labs, gt_labs):
    alias_to_key = {}
    for gt_key, aliases in NAME_MAP.items():
        for alias in aliases:
            alias_to_key[_normalise(alias)] = gt_key

    extracted_by_key = {}
    for lab in extracted_labs:
        norm = _normalise(lab.get("test_name", ""))
        if norm in alias_to_key:
            extracted_by_key[alias_to_key[norm]] = lab

    comparison = {}
    for gt_key, gt_entry in gt_labs.items():
        gt_val = gt_entry["value"]
        if gt_key in extracted_by_key:
            ext_val = extracted_by_key[gt_key].get("value")
            try:
                gt_num, ext_num = float(gt_val), float(ext_val)
                abs_diff = abs(gt_num - ext_num)
                rel_diff = abs_diff / max(abs(gt_num), 0.01)
                match = abs_diff <= 0.15 or rel_diff <= 0.02
            except (TypeError, ValueError):
                match = str(gt_val) == str(ext_val)
            comparison[gt_key] = {"gt_value": gt_val, "extracted_value": ext_val, "match": match}
        else:
            comparison[gt_key] = {"gt_value": gt_val, "extracted_value": None, "match": False}
    return comparison


def run_test(image_path, gt_path, model, processor, label):
    """Run extraction on one image, compare to ground truth, print results."""
    print(f"\n--- Testing on {image_path.name} ({label}) ---")

    start = time.time()
    extraction, raw = extract_lab_data(image_path, model, processor)
    elapsed = time.time() - start

    print(f"Extraction took {elapsed:.1f}s")

    if extraction.get("_parse_error"):
        print("  WARNING: JSON parse failed. Raw response snippet:")
        print(" ", raw[:300])
        return None

    print(f"Patient: {extraction.get('patient_info', {}).get('name', 'N/A')}")
    print(f"Labs found: {len(extraction.get('lab_results', []))}")
    print(f"Meds found: {extraction.get('medications_listed', [])}")

    with open(gt_path) as f:
        gt = json.load(f)

    comparison = match_extracted_to_gt(
        extraction.get("lab_results", []),
        gt["lab_panel"]
    )

    print(f"\n{'Test':<22} {'Ground Truth':>12} {'Extracted':>12} {'Match':>7}")
    print("─" * 60)
    matched = total = 0
    for key, vals in comparison.items():
        total += 1
        if vals["match"]:
            matched += 1
        gt_s = str(vals["gt_value"]) if vals["gt_value"] is not None else "-"
        ex_s = str(vals["extracted_value"]) if vals["extracted_value"] is not None else "-"
        mark = "OK" if vals["match"] else "MISS"
        print(f"  {key:<20} {gt_s:>12} {ex_s:>12} {mark:>7}")

    print("─" * 60)
    accuracy = 100 * matched / max(total, 1)
    print(f"  Accuracy: {matched}/{total} ({accuracy:.1f}%)")

    return {"image": image_path.name, "matched": matched, "total": total, "accuracy": accuracy}


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DiabetesKidney Companion — MedGemma Setup & Test")
    print("=" * 60)

    # Load model
    model, processor, model_id = select_and_load_model()

    # Save config for subsequent scripts
    config = {
        "model_id": model_id,
        "project_dir": str(PROJECT_DIR),
        "data_dir": str(DATA_DIR),
        "output_dir": str(OUTPUT_DIR),
    }
    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")

    results = []

    # Test extraction on SYN-0001
    test1_image = IMAGE_DIR / "SYN-0001.png"
    test1_gt = GT_DIR / "SYN-0001.json"
    r1 = run_test(test1_image, test1_gt, model, processor, label="baseline")
    if r1:
        results.append(r1)

    # Also test a complex case
    test2_image = IMAGE_DIR / "SYN-0190.png"
    test2_gt = GT_DIR / "SYN-0190.json"
    if test2_image.exists() and test2_gt.exists():
        r2 = run_test(test2_image, test2_gt, model, processor, label="complex case")
        if r2:
            results.append(r2)
    else:
        print(f"\nSkipping SYN-0190 (not found)")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Model used: {model_id}")
    if results:
        overall_matched = sum(r["matched"] for r in results)
        overall_total = sum(r["total"] for r in results)
        print(f"Overall accuracy: {overall_matched}/{overall_total} "
              f"({100 * overall_matched / max(overall_total, 1):.1f}%)")
        for r in results:
            print(f"  {r['image']}: {r['matched']}/{r['total']} ({r['accuracy']:.1f}%)")

    # Save results
    results_path = OUTPUT_DIR / "setup_test_results.json"
    with open(results_path, "w") as f:
        json.dump({"model_id": model_id, "tests": results}, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print("\nSetup complete. Ready for batch extraction.")

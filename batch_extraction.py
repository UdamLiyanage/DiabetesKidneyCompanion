#!/usr/bin/env python3
"""
batch_extraction.py — Batch MedGemma extraction over all 195 synthetic lab images.
Run AFTER setup_and_test.py has passed.

Resumable: already-processed images are skipped automatically.
"""

import os
import sys

# Must be set before importing transformers/torch hub.
os.environ["HF_HOME"] = "/data/gemma/.cache/huggingface"

import json
import time
import re
import torch
from collections import defaultdict
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_DIR    = PROJECT_DIR / "synthetic_output"
IMAGE_DIR   = DATA_DIR / "images"
GT_DIR      = DATA_DIR / "patients"
OUTPUT_DIR  = PROJECT_DIR / "output"
EXTR_DIR    = OUTPUT_DIR / "extractions"

OUTPUT_DIR.mkdir(exist_ok=True)
EXTR_DIR.mkdir(exist_ok=True)

MODELS_TO_TRY = [
    "google/medgemma-27b-it",
    "google/medgemma-1.5-4b-it",
]


# ── Model loading ──────────────────────────────────────────

def select_and_load_model(preferred_model_id=None):
    """
    Load a model.  If preferred_model_id is supplied (e.g. read from config.json
    written by script 1) it is tried first; otherwise MODELS_TO_TRY order applies.
    """
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


# ── Extraction components (kept in sync with script 1) ─────

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
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def strip_thinking(response):
    """Remove MedGemma's internal thinking block, keeping only the JSON output."""
    # Method 1: Find last {"patient_info" occurrence (skip thinking drafts)
    candidates = []
    for m in re.finditer(r'\{', response):
        snippet = response[m.start():m.start()+50].lower()
        if '"patient_info"' in snippet:
            candidates.append(m.start())
    if candidates:
        return response[candidates[-1]:]

    # Method 2: Find any JSON with lab_results or lab_values
    for m in re.finditer(r'\{', response):
        snippet = response[m.start():m.start()+100].lower()
        if '"lab_results"' in snippet or '"lab_values"' in snippet:
            return response[m.start():]

    # Method 3: Strip special token tags like <unused94>, <start_of_turn>, etc.
    cleaned = re.sub(r'<[^>]+>', '', response)
    cleaned = cleaned.strip()

    # Method 4: Never return empty
    return cleaned if cleaned else response


def sanitize_json_string(s):
    """Fix common JSON errors in MedGemma output."""
    # 1. "value":0.7-1.3 -> "value":"0.7-1.3" (range pasted as value)
    s = re.sub(r'("value"\s*:\s*)(\d+\.?\d*\s*-\s*\d+\.?\d*)', r'\1"\2"', s)

    # 2. "value":H, or "value":L, (bare flag letter as value)
    s = re.sub(r'("value"\s*:\s*)([HL])\s*([,}])', r'\1"\2"\3', s)

    # 3. "value">60 or "value">=60 (missing colon, comparison as value)
    s = re.sub(r'("value"\s*"?\s*)(>|>=|<|<=)(\s*\d+\.?\d*)', r'"value":"\2\3"', s)

    # 4. "reference_range">= or ">60" patterns with missing colon
    s = re.sub(r'("reference_range"\s*"?\s*)(>|>=|<|<=)(\s*\d+\.?\d*")', r'"reference_range":"\2\3', s)

    # 5. "value":10.4" -> "value":10.4 (trailing quote on number)
    s = re.sub(r'("value"\s*:\s*)(\d+\.?\d*)"(\s*[,}])', r'\1\2\3', s)

    # 6. "value":<30 or "value":>60 (colon present but comparison operator)
    s = re.sub(r'("value"\s*:\s*)(>|>=|<|<=)(\s*\d+\.?\d*)\s*([,}])', r'\1"\2\3"\4', s)

    return s


def parse_json_response(response):
    """Parse JSON from model response — bulletproof version."""
    cleaned = response.strip()
    # Strip special tokens
    cleaned = re.sub(r'<[a-z_/][^>]*>', '', cleaned)
    cleaned = cleaned.strip()

    # Strip markdown fences (do it manually since regex can be fragile)
    if cleaned.startswith('```'):
        first_newline = cleaned.index('\n') if '\n' in cleaned else len(cleaned)
        cleaned = cleaned[first_newline+1:]
    if cleaned.rstrip().endswith('```'):
        cleaned = cleaned.rstrip()
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Sanitize known MedGemma JSON errors
    cleaned = sanitize_json_string(cleaned)

    # Attempt 1: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: raw_decode from first {
    try:
        idx = cleaned.index('{')
        result, _ = json.JSONDecoder().raw_decode(cleaned, idx)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 3: extract JSON by matching braces
    try:
        start = cleaned.index('{')
        depth = 0
        in_string = False
        escape = False
        end = start
        for i in range(start, len(cleaned)):
            c = cleaned[i]
            if escape:
                escape = False
                continue
            if c == '\\':
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        candidate = cleaned[start:end]
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 4: fix trailing commas on the brace-matched candidate
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError, NameError):
        pass

    return {"_parse_error": True, "_raw_response": response}


def normalize_extraction(parsed):
    """Normalize variant JSON structures to our expected format."""
    if parsed.get("_parse_error"):
        return parsed

    # Handle "lab_values" -> "lab_results"
    if "lab_values" in parsed and "lab_results" not in parsed:
        parsed["lab_results"] = parsed.pop("lab_values")

    # Handle "result" -> "value" in each lab entry
    for lab in parsed.get("lab_results", []):
        if "result" in lab and "value" not in lab:
            lab["value"] = lab.pop("result")

    # Handle missing patient_info gracefully
    if "patient_info" not in parsed:
        parsed["patient_info"] = {}

    # Handle missing medications
    if "medications_listed" not in parsed:
        parsed["medications_listed"] = parsed.get("medications", [])

    return parsed


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

    new_tokens = output.shape[-1] - input_len

    # Decode both ways
    response_clean = processor.decode(output[0][input_len:], skip_special_tokens=True)
    response_raw   = processor.decode(output[0][input_len:], skip_special_tokens=False)

    print(f"    [debug] new_tokens={new_tokens}, clean_len={len(response_clean)}, raw_len={len(response_raw)}")

    # Parse from clean version (no special tokens like <end_of_turn>).
    # Fall back to raw only if clean is empty.
    parse_source = response_clean if response_clean.strip() else response_raw

    cleaned = strip_thinking(parse_source)
    parsed  = parse_json_response(cleaned)
    parsed  = normalize_extraction(parsed)

    # Always return raw version for debugging/saving
    return parsed, response_raw


# ── Ground truth comparison ────────────────────────────────

NAME_MAP = {
    "glucose":       ["glucose", "glucose (fasting)", "glucose, fasting", "fasting glucose"],
    "bun":           ["bun", "bun (urea)", "blood urea nitrogen", "blood urea nitrogen (bun)",
                      "urea nitrogen", "urea", "bun/urea"],
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
    "hemoglobin":    ["hemoglobin", "hemoglobin, blood", "hgb", "hb",
                      "hemoglobin (hgb)", "hgb/hb", "hemoglobin a",
                      "hemoglobin (blood)", "cbc hemoglobin"],
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
            comparison[gt_key] = {
                "gt_value": gt_val, "extracted_value": ext_val, "match": match
            }
        else:
            comparison[gt_key] = {
                "gt_value": gt_val, "extracted_value": None, "match": False
            }
    return comparison


# ── Batch helpers ──────────────────────────────────────────

def output_path_for(image_path):
    """Return the extraction JSON path for a given image."""
    return EXTR_DIR / f"{image_path.stem}_extraction.json"


def already_processed(image_path):
    p = output_path_for(image_path)
    if not p.exists():
        return False
    try:
        with open(p) as f:
            data = json.load(f)
        # Treat a previously-failed parse as not done so it can be retried
        return not data.get("_parse_error", False)
    except (json.JSONDecodeError, OSError):
        return False


def process_image(image_path, model, processor):
    """
    Extract labs from one image, save to disk, return a result dict.
    Never raises — errors are captured and returned as failed records.
    """
    out_path = output_path_for(image_path)
    patient_id = image_path.stem          # e.g. "SYN-0042"

    try:
        t0 = time.time()
        extraction, raw_response = extract_lab_data(image_path, model, processor)
        elapsed = time.time() - t0

        record = {
            "patient_id":    patient_id,
            "image":         image_path.name,
            "elapsed_s":     round(elapsed, 2),
            "extraction":    extraction,
            "raw_response":  raw_response,
            "parse_error":   extraction.get("_parse_error", False),
        }

        # Ground-truth comparison when available
        gt_path = GT_DIR / f"{patient_id}.json"
        if gt_path.exists() and not record["parse_error"]:
            with open(gt_path) as f:
                gt = json.load(f)
            comparison = match_extracted_to_gt(
                extraction.get("lab_results", []),
                gt["lab_panel"],
            )
            matched = sum(1 for v in comparison.values() if v["match"])
            total   = len(comparison)
            record["comparison"] = comparison
            record["matched"]    = matched
            record["total"]      = total
            record["accuracy"]   = round(100 * matched / max(total, 1), 2)
        else:
            record["comparison"] = {}
            record["matched"]    = 0
            record["total"]      = 0
            record["accuracy"]   = None

        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

        return record

    except Exception as exc:
        record = {
            "patient_id":   patient_id,
            "image":        image_path.name,
            "elapsed_s":    None,
            "extraction":   {},
            "raw_response": "",
            "parse_error":  True,
            "error":        str(exc),
            "comparison":   {},
            "matched":      0,
            "total":        0,
            "accuracy":     None,
        }
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        return record


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DiabetesKidney Companion — Batch Extraction")
    print("=" * 60)

    # Use model preferred by script 1 if available
    preferred = None
    config_path = OUTPUT_DIR / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        preferred = cfg.get("model_id")
        print(f"Config found. Preferred model: {preferred}")
    else:
        print("No config.json found — will try models in default order.")

    model, processor, model_id = select_and_load_model(preferred_model_id=preferred)

    # Collect and sort all images
    image_files = sorted(IMAGE_DIR.glob("SYN-*.png"))
    if not image_files:
        print(f"ERROR: No images found in {IMAGE_DIR}")
        sys.exit(1)
    print(f"\nFound {len(image_files)} images in {IMAGE_DIR}")

    # Identify already-done vs pending
    pending   = [p for p in image_files if not already_processed(p)]
    n_skipped = len(image_files) - len(pending)
    print(f"Already processed (skipping): {n_skipped}")
    print(f"To process:                   {len(pending)}")

    if not pending:
        print("\nAll images already processed. Nothing to do.")
        print("Delete files in output/extractions/ to reprocess.")
    else:
        # ── Batch loop ─────────────────────────────────────────
        batch_start = time.time()
        results = []

        for img_path in tqdm(pending, desc="Extracting", unit="img"):
            record = process_image(img_path, model, processor)
            results.append(record)

            status = "FAIL" if record["parse_error"] else f"{record['accuracy']:.1f}%"
            tqdm.write(f"  {record['patient_id']}  {record.get('elapsed_s', '?'):.1f}s  {status}")

        batch_elapsed = time.time() - batch_start

    # ── Aggregate over ALL processed files (including previously skipped) ──
    all_records = []
    for p in image_files:
        out = output_path_for(p)
        if out.exists():
            try:
                with open(out) as f:
                    all_records.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass

    n_total     = len(image_files)
    n_processed = len(all_records)
    n_failed    = sum(1 for r in all_records if r.get("parse_error") or r.get("error"))
    n_ok        = n_processed - n_failed

    # Per-test accuracy across entire corpus
    per_test_matched = defaultdict(int)
    per_test_total   = defaultdict(int)
    overall_matched  = 0
    overall_total    = 0

    for r in all_records:
        for test_key, vals in r.get("comparison", {}).items():
            per_test_total[test_key]   += 1
            overall_total              += 1
            if vals.get("match"):
                per_test_matched[test_key] += 1
                overall_matched            += 1

    per_test_stats = {
        k: {
            "matched": per_test_matched[k],
            "total":   per_test_total[k],
            "pct":     round(100 * per_test_matched[k] / max(per_test_total[k], 1), 1),
        }
        for k in sorted(per_test_total)
    }

    overall_pct = round(100 * overall_matched / max(overall_total, 1), 2)

    # Timing stats (only from this run's newly processed batch)
    times = [r["elapsed_s"] for r in all_records if r.get("elapsed_s") is not None]
    avg_time = round(sum(times) / len(times), 2) if times else None

    summary = {
        "model":                    model_id,
        "total_images":             n_total,
        "processed":                n_processed,
        "successful":               n_ok,
        "failed":                   n_failed,
        "avg_time_per_image_s":     avg_time,
        "accuracy": {
            "overall_pct":  overall_pct,
            "matched":      overall_matched,
            "total":        overall_total,
            "per_test":     per_test_stats,
        },
    }

    summary_path = OUTPUT_DIR / "extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Console summary ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Batch Extraction Summary")
    print("=" * 60)
    print(f"Model:          {model_id}")
    print(f"Images total:   {n_total}")
    print(f"Processed:      {n_processed}  (successful: {n_ok}, failed: {n_failed})")
    if avg_time:
        print(f"Avg time/image: {avg_time}s")
    print(f"\nOverall accuracy: {overall_matched}/{overall_total} ({overall_pct}%)")
    print(f"\n{'Test':<22} {'Matched':>8} {'Total':>7} {'%':>7}")
    print("─" * 48)
    for k, s in per_test_stats.items():
        print(f"  {k:<20} {s['matched']:>8} {s['total']:>7} {s['pct']:>6.1f}%")
    print("─" * 48)
    print(f"\nPer-image JSONs: {EXTR_DIR}")
    print(f"Summary saved:   {summary_path}")

    if n_failed:
        failed_ids = [r["patient_id"] for r in all_records if r.get("parse_error") or r.get("error")]
        print(f"\nFailed images ({n_failed}): {', '.join(failed_ids)}")

    # ── Hemoglobin mismatch diagnostic ─────────────────────
    mismatches = []
    for r in all_records[:10]:
        if r.get("parse_error"):
            continue
        comp = r.get("comparison", {})
        if "hemoglobin" in comp and not comp["hemoglobin"]["match"]:
            mismatches.append({
                "id":  r["patient_id"],
                "gt":  comp["hemoglobin"]["gt_value"],
                "ext": comp["hemoglobin"]["extracted_value"],
            })
    if mismatches:
        print("\nHemoglobin mismatches (first 10 records):")
        for m in mismatches:
            print(f"  {m['id']}: GT={m['gt']} EXT={m['ext']}")

"""
FIR 2024 - Qwen3-VL-235B-A22B-Instruct item-wise text+image API benchmark (2 runs)
STRICT OUTPUT VERSION - SHORT WINDOWS PATH VERSION

Fixed model:
  MODEL_REQUESTED = "qwen3-vl-235b-a22b-instruct"

This script evaluates the 210 questions of the Spanish FIR 2024 Pharmacy exam
using Alibaba Cloud Model Studio / DashScope OpenAI-compatible Chat Completions,
one question at a time.

For image-containing FIR questions, it sends the local PNG/JPG image as a
Base64 data URL via image_url. Expected image folder:
  images/fir/

Input files expected in the same folder as this script:
  - fir2024_questions_210.csv
  - fir2024_answer_key_210.csv
  - images/fir/Q001.png, Q006.png, ...
  - .env containing:
      DASHSCOPE_API_KEY=...
      QWEN_BASE_URL=.../compatible-mode/v1

Cheap probe:
  python run_fir2024_qwen3_vl_235b_a22b_instruct_itemwise_textplusimage_2runs_STRICT.py --limit 5

Probe only image questions:
  python run_fir2024_qwen3_vl_235b_a22b_instruct_itemwise_textplusimage_2runs_STRICT.py --only-images --limit 5

Full run:
  python run_fir2024_qwen3_vl_235b_a22b_instruct_itemwise_textplusimage_2runs_STRICT.py

Requirements:
  pip install openai python-dotenv pandas truststore
"""

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

try:
    import truststore
    truststore.inject_into_ssl()
    TRUSTSTORE_ENABLED = True
except Exception:
    TRUSTSTORE_ENABLED = False


# =========================
# STUDY CONFIGURATION
# =========================

PROVIDER = "Alibaba Cloud Model Studio / Qwen"
INTERFACE = "API"
API_ENDPOINT = "OpenAI-compatible Chat Completions"

MODEL_REQUESTED = "qwen3-vl-235b-a22b-instruct"

PROMPT_TEMPLATE_VERSION = "fir2024_itemwise_textplusimage_examlike_blank_strict_single_answer_v1"
SCRIPT_VERSION = "fir2024_qwen3_vl_235b_a22b_instruct_itemwise_textplusimage_2runs_strict_v1"

GENERATION_PARAMETERS = {
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 300,
    "response_format": {"type": "json_object"},
    "stream": False,
    "tools_enabled": False,
    "web_search_enabled": False,
    "file_input_enabled": False,
    "image_input_enabled": True,
    "image_transport": "base64_data_url_via_image_url",
    "enable_thinking": False,
}

N_REPLICATES_DEFAULT = 2
REQUEST_TIMEOUT_SECONDS = 180
MAX_ATTEMPTS_PER_CALL = 3
RETRY_SLEEP_SECONDS = 5

QUESTIONS_FILE = Path("fir2024_questions_210.csv")
ANSWER_KEY_FILE = Path("fir2024_answer_key_210.csv")
IMAGE_ROOT = Path("images") / "fir"
OUTPUT_ROOT = Path("outputs")


# =========================
# PROMPT
# =========================

PROMPT_PREAMBLE = """You are taking an examination for Pharmacy residency access.

This is a multiple-choice exam. Each question has four options, numbered 1 to 4, and only one is correct. In the official exam, correct answers score +1 point, incorrect answers score -1/3 points, and blank answers score 0 points. Act as a human candidate and try to get the highest score possible.

For image-containing questions, use both the question text and the attached image. Do not use web search, tools, external references, or an answer key.

Return only valid JSON with exactly this shape:
{"answer": "1|2|3|4|blank", "rationale": "brief explanation"}

Output rules:
- The value of "answer" must be exactly one single string: "1", "2", "3", "4", or "blank".
- Do not return multiple options.
- Do not return values such as "1,2", "1 or 2", "3/4", "4,", lists, arrays, ranges, or explanations inside the "answer" field.
- If you cannot choose one single best option, return "blank".
- Keep the rationale brief and do not include alternative final answers.
"""


def build_prompt(row: pd.Series) -> str:
    has_image = bool_like(row.get("has_image", False))
    image_note = "\nThis question includes an attached image." if has_image else ""
    return f"""{PROMPT_PREAMBLE}
Question {int(row['original_question_number'])}:{image_note}
{row['question_text']}

Options:
1. {row['option_1']}
2. {row['option_2']}
3. {row['option_3']}
4. {row['option_4']}
"""


# =========================
# HELPERS
# =========================

def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_model_name(model_name: str) -> str:
    return model_name.replace(".", "_").replace("-", "_").replace("/", "_").replace(":", "_")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def bool_like(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() in {"true", "1", "yes", "y", "si", "sí"}


def normalize_answer(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip().lower()

    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]

    if text in {"", "nan", "none", "null"}:
        return None

    if text in {"1", "2", "3", "4", "blank"}:
        return text

    return text


def parse_model_answer(raw_text: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns parsed_answer, rationale, parse_error.

    Invalid answer-field values are scored as missing_or_invalid = 0 and fully
    documented in parse_error/raw_text.
    """
    if not raw_text:
        return None, None, "missing_raw_output"

    text = raw_text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                data = json.loads(text[start:end + 1])
            else:
                raise
        except Exception as e:
            return None, None, f"json_decode_error: {str(e)}"

    if not isinstance(data, dict):
        return None, None, "json_not_object"

    raw_answer = data.get("answer")
    answer = normalize_answer(raw_answer)
    rationale = data.get("rationale")

    if answer not in {"1", "2", "3", "4", "blank"}:
        return None, rationale, f"invalid_answer_value: {raw_answer}"

    return answer, rationale, None


def score_answer(model_answer: Optional[str], correct_answer: Optional[str]) -> Tuple[Optional[bool], Optional[float], str]:
    if correct_answer is None:
        return None, None, "not_evaluable"

    if model_answer is None:
        return False, 0.0, "missing_or_invalid"

    if model_answer == "blank":
        return False, 0.0, "blank"

    if model_answer == correct_answer:
        return True, 1.0, "correct"

    return False, -1.0 / 3.0, "incorrect"


def get_correct_answer(answer_key: pd.DataFrame, question_id: str) -> Optional[str]:
    row = answer_key.loc[answer_key["question_id"] == question_id]
    if row.empty:
        return None

    if "official_answer_available" in row.columns:
        if not bool(row.iloc[0]["official_answer_available"]):
            return None

    if "is_annulled_or_without_official_answer" in row.columns:
        if bool(row.iloc[0]["is_annulled_or_without_official_answer"]):
            return None

    return normalize_answer(row.iloc[0]["correct_answer"])


def get_image_path(row: pd.Series) -> Optional[Path]:
    if not bool_like(row.get("has_image", False)):
        return None

    # Prefer an explicit CSV path if present and valid.
    for col in ["image_path", "image_file", "local_image_path", "image"]:
        if col in row.index:
            value = row.get(col)
            if value is not None:
                try:
                    if pd.isna(value):
                        continue
                except Exception:
                    pass
                text = str(value).strip()
                if text:
                    p = Path(text.replace("\\", "/"))
                    if p.exists():
                        return p
                    # If the CSV path is relative but includes only filename.
                    p2 = IMAGE_ROOT / p.name
                    if p2.exists():
                        return p2

    qn = int(row["original_question_number"])
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = IMAGE_ROOT / f"Q{qn:03d}{ext}"
        if p.exists():
            return p

    return IMAGE_ROOT / f"Q{qn:03d}.png"


def file_to_data_url(path: Path) -> Tuple[str, str, int, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing image file: {path}")

    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    sha = hashlib.sha256(raw).hexdigest()
    data_url = f"data:{mime_type};base64,{b64}"
    return data_url, mime_type, len(raw), sha


def validate_required_images(questions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in questions.iterrows():
        if not bool_like(row.get("has_image", False)):
            continue
        qn = int(row["original_question_number"])
        p = get_image_path(row)
        exists = p.exists() if p else False
        size_bytes = p.stat().st_size if exists else None
        sha = sha256_file(p) if exists else None
        rows.append({
            "question_id": row["question_id"],
            "original_question_number": qn,
            "image_path": str(p) if p else None,
            "exists": exists,
            "size_bytes": size_bytes,
            "sha256": sha,
        })
    manifest = pd.DataFrame(rows)
    missing = manifest[manifest["exists"] != True] if len(manifest) else pd.DataFrame()
    if len(missing):
        print("Missing image files:")
        print(missing.to_string(index=False))
        raise FileNotFoundError("One or more required FIR image files are missing. See list above.")
    return manifest


def build_message_content(prompt: str, image_path: Optional[Path]) -> Tuple[Any, Dict[str, Any]]:
    """
    For non-image questions, returns plain text content.
    For image questions, returns OpenAI-compatible multimodal content list.
    """
    image_meta = {
        "image_sent": False,
        "image_path": None,
        "image_mime_type": None,
        "image_size_bytes": None,
        "image_sha256": None,
    }

    if image_path is None:
        return prompt, image_meta

    data_url, mime_type, size_bytes, sha = file_to_data_url(image_path)
    image_meta.update({
        "image_sent": True,
        "image_path": str(image_path),
        "image_mime_type": mime_type,
        "image_size_bytes": size_bytes,
        "image_sha256": sha,
    })

    content = [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": prompt},
    ]
    return content, image_meta


def call_qwen(client: OpenAI, prompt: str, image_path: Optional[Path]) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], Dict[str, Any]]:
    content, image_meta = build_message_content(prompt, image_path)

    kwargs = dict(
        model=MODEL_REQUESTED,
        messages=[{"role": "user", "content": content}],
        temperature=GENERATION_PARAMETERS["temperature"],
        top_p=GENERATION_PARAMETERS["top_p"],
        max_tokens=GENERATION_PARAMETERS["max_tokens"],
        response_format=GENERATION_PARAMETERS["response_format"],
        extra_body={"enable_thinking": False},
    )

    try:
        response = client.chat.completions.create(**kwargs)
        response_dict = response.model_dump()
        choices = response_dict.get("choices") or []
        raw_content = None
        if choices:
            raw_content = (choices[0].get("message") or {}).get("content")
        return response_dict, raw_content, None, image_meta
    except Exception as e:
        return None, None, repr(e), image_meta


def extract_usage(response_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not response_dict:
        return {}
    return response_dict.get("usage") or {}


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for replicate, group in results.groupby("replicate"):
        evaluable = group[group["has_official_answer"] == True].copy()
        n_eval = len(evaluable)
        correct = int((evaluable["answer_status"] == "correct").sum())
        incorrect = int((evaluable["answer_status"] == "incorrect").sum())
        blank = int((evaluable["answer_status"] == "blank").sum())
        missing_invalid = int((evaluable["answer_status"] == "missing_or_invalid").sum())
        api_errors = int(group["api_error"].notna().sum())
        parse_errors = int(group["parse_error"].notna().sum())
        score = evaluable["score"].sum(skipna=True)

        rows.append({
            "replicate": replicate,
            "n_questions_total": len(group),
            "n_evaluable": n_eval,
            "correct": correct,
            "incorrect": incorrect,
            "blank": blank,
            "missing_or_invalid": missing_invalid,
            "not_evaluable": int((group["answer_status"] == "not_evaluable").sum()),
            "accuracy_evaluable": correct / n_eval if n_eval else None,
            "score_evaluable": score,
            "api_errors": api_errors,
            "parse_errors": parse_errors,
            "mean_latency_seconds": group["latency_seconds"].mean(),
            "total_prompt_tokens": group["prompt_tokens"].sum(skipna=True),
            "total_completion_tokens": group["completion_tokens"].sum(skipna=True),
            "total_tokens": group["total_tokens"].sum(skipna=True),
            "total_prompt_cache_hit_tokens": group["prompt_cache_hit_tokens"].sum(skipna=True),
            "total_prompt_cache_miss_tokens": group["prompt_cache_miss_tokens"].sum(skipna=True),
            "n_image_questions_sent": int(group["image_sent"].sum()),
        })

    return pd.DataFrame(rows)


def build_summary_by_image_status(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (replicate, has_image), group in results.groupby(["replicate", "has_image"], dropna=False):
        evaluable = group[group["has_official_answer"] == True].copy()
        n_eval = len(evaluable)
        correct = int((evaluable["answer_status"] == "correct").sum())
        incorrect = int((evaluable["answer_status"] == "incorrect").sum())
        blank = int((evaluable["answer_status"] == "blank").sum())
        missing_invalid = int((evaluable["answer_status"] == "missing_or_invalid").sum())
        score = evaluable["score"].sum(skipna=True)

        rows.append({
            "replicate": replicate,
            "has_image": bool(has_image),
            "n_questions_total": len(group),
            "n_evaluable": n_eval,
            "correct": correct,
            "incorrect": incorrect,
            "blank": blank,
            "missing_or_invalid": missing_invalid,
            "not_evaluable": int((group["answer_status"] == "not_evaluable").sum()),
            "accuracy_evaluable": correct / n_eval if n_eval else None,
            "score_evaluable": score,
            "api_errors": int(group["api_error"].notna().sum()),
            "parse_errors": int(group["parse_error"].notna().sum()),
            "n_images_sent": int(group["image_sent"].sum()),
            "total_tokens": group["total_tokens"].sum(skipna=True),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Optional cheap probe, e.g. --limit 5.")
    parser.add_argument("--replicates", type=int, default=N_REPLICATES_DEFAULT, help="Number of replicates. Default: 2.")
    parser.add_argument("--only-images", action="store_true", help="Run only questions flagged as image-containing.")
    parser.add_argument("--only-text", action="store_true", help="Run only questions not flagged as image-containing.")
    args = parser.parse_args()

    if args.only_images and args.only_text:
        raise RuntimeError("Use either --only-images or --only-text, not both.")

    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL")

    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY in .env")
    if not base_url:
        raise RuntimeError("Missing QWEN_BASE_URL in .env")

    if base_url.rstrip("/").endswith("/chat/completions"):
        raise RuntimeError("QWEN_BASE_URL must end in /compatible-mode/v1, not /chat/completions.")

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing {QUESTIONS_FILE}")
    if not ANSWER_KEY_FILE.exists():
        raise FileNotFoundError(f"Missing {ANSWER_KEY_FILE}")

    questions = pd.read_csv(QUESTIONS_FILE)
    answer_key = pd.read_csv(ANSWER_KEY_FILE)
    questions = questions.sort_values("original_question_number").reset_index(drop=True)

    if "has_image" not in questions.columns:
        questions["has_image"] = False

    if args.only_images:
        questions = questions[questions["has_image"].apply(bool_like)].copy()
    elif args.only_text:
        questions = questions[~questions["has_image"].apply(bool_like)].copy()

    if args.limit is not None:
        questions = questions.head(args.limit).copy()

    # Validate only images that will be used in this run.
    image_manifest = validate_required_images(questions)

    run_timestamp = utc_timestamp()
    safe_name = safe_model_name(MODEL_REQUESTED)
    mode_suffix = "_onlyimages" if args.only_images else ("_onlytext" if args.only_text else "")
    limit_suffix = f"_limit{args.limit}" if args.limit is not None else ""
    # Keep paths short for Windows MAX_PATH compatibility.
    run_name = f"fir2024_qwen235b_ti2r_STRICT{mode_suffix}{limit_suffix}_{run_timestamp}"
    output_dir = OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_outputs_file = output_dir / f"{run_name}_raw_outputs.jsonl"
    results_file = output_dir / f"{run_name}_scored_results.csv"
    summary_file = output_dir / f"{run_name}_summary_by_replicate.csv"
    image_summary_file = output_dir / f"{run_name}_summary_by_image_status.csv"
    metadata_file = output_dir / f"{run_name}_run_metadata.json"
    image_manifest_file = output_dir / f"{run_name}_image_manifest.csv"

    image_manifest.to_csv(image_manifest_file, index=False, encoding="utf-8")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)

    metadata = {
        "run_name": run_name,
        "created_utc": run_timestamp,
        "provider": PROVIDER,
        "interface": INTERFACE,
        "api_endpoint": API_ENDPOINT,
        "model_requested": MODEL_REQUESTED,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_preamble_hash_sha256": sha256_text(PROMPT_PREAMBLE),
        "prompt_preamble": PROMPT_PREAMBLE,
        "script_version": SCRIPT_VERSION,
        "questions_file": str(QUESTIONS_FILE),
        "questions_file_sha256": sha256_file(QUESTIONS_FILE),
        "answer_key_file": str(ANSWER_KEY_FILE),
        "answer_key_file_sha256": sha256_file(ANSWER_KEY_FILE),
        "image_root": str(IMAGE_ROOT),
        "image_manifest_file": str(image_manifest_file),
        "n_replicates": args.replicates,
        "n_questions_attempted": int(len(questions)),
        "n_image_questions_attempted": int(questions["has_image"].apply(bool_like).sum()),
        "generation_parameters": GENERATION_PARAMETERS,
        "scoring": {
            "correct": 1,
            "incorrect": -1/3,
            "blank": 0,
            "missing_or_invalid": 0,
            "invalid_answer_field": 0,
            "questions_without_official_answer_excluded": True,
        },
        "base_url_logged": base_url,
        "truststore_enabled": TRUSTSTORE_ENABLED,
    }

    all_rows = []

    print(f"Run name: {run_name}")
    print(f"Provider: {PROVIDER}")
    print(f"Fixed model: {MODEL_REQUESTED}")
    print(f"Base URL: {base_url}")
    print(f"Questions: {len(questions)}")
    print(f"Image questions in run: {int(questions['has_image'].apply(bool_like).sum())}")
    print(f"Replicates: {args.replicates}")
    print(f"Output directory: {output_dir}")
    print("enable_thinking param sent: True")
    print("Strict single-answer prompt: True")
    print()

    with raw_outputs_file.open("w", encoding="utf-8") as raw_f:
        for replicate in range(1, args.replicates + 1):
            print(f"=== Replicate {replicate}/{args.replicates} ===")

            for _, row in questions.iterrows():
                question_id = row["question_id"]
                original_question_number = int(row["original_question_number"])
                has_image = bool_like(row.get("has_image", False))
                image_path = get_image_path(row) if has_image else None
                prompt = build_prompt(row)
                correct_answer = get_correct_answer(answer_key, question_id)
                has_official_answer = correct_answer is not None

                response_dict = None
                raw_text = None
                api_error = None
                parse_error = None
                model_answer = None
                rationale = None
                latency_seconds = None
                attempts_used = 0
                image_meta = {
                    "image_sent": False,
                    "image_path": str(image_path) if image_path else None,
                    "image_mime_type": None,
                    "image_size_bytes": None,
                    "image_sha256": None,
                }

                for attempt in range(1, MAX_ATTEMPTS_PER_CALL + 1):
                    attempts_used = attempt
                    start = time.time()
                    response_dict, raw_text, api_error, image_meta = call_qwen(
                        client=client,
                        prompt=prompt,
                        image_path=image_path,
                    )
                    latency_seconds = time.time() - start

                    if api_error is None:
                        model_answer, rationale, parse_error = parse_model_answer(raw_text)
                        break

                    if attempt < MAX_ATTEMPTS_PER_CALL:
                        time.sleep(RETRY_SLEEP_SECONDS)

                is_correct, score, answer_status = score_answer(model_answer, correct_answer)
                usage = extract_usage(response_dict)

                prompt_tokens_details = usage.get("prompt_tokens_details") or {}
                completion_tokens_details = usage.get("completion_tokens_details") or {}

                model_returned = None
                finish_reason = None
                if response_dict:
                    model_returned = response_dict.get("model")
                    choices = response_dict.get("choices") or []
                    if choices:
                        finish_reason = choices[0].get("finish_reason")

                record = {
                    "run_name": run_name,
                    "provider": PROVIDER,
                    "api_endpoint": API_ENDPOINT,
                    "model_requested": MODEL_REQUESTED,
                    "model_returned": model_returned,
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    "replicate": replicate,
                    "question_id": question_id,
                    "original_question_number": original_question_number,
                    "is_reserve": bool(row.get("is_reserve", False)),
                    "has_image": has_image,
                    "prompt_hash_sha256": sha256_text(prompt),
                    "prompt": prompt,
                    "model_answer": model_answer,
                    "rationale": rationale,
                    "correct_answer": correct_answer,
                    "has_official_answer": has_official_answer,
                    "is_correct": is_correct,
                    "score": score,
                    "answer_status": answer_status,
                    "raw_text": raw_text,
                    "parse_error": parse_error,
                    "api_error": api_error,
                    "attempts_used": attempts_used,
                    "enable_thinking_param_sent": True,
                    "strict_single_answer_prompt": True,
                    "latency_seconds": latency_seconds,
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "reasoning_tokens": completion_tokens_details.get("reasoning_tokens"),
                    "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens") or prompt_tokens_details.get("cached_tokens"),
                    "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
                    **image_meta,
                    "response": response_dict,
                }

                raw_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                raw_f.flush()

                all_rows.append({k: v for k, v in record.items() if k not in {"prompt", "raw_text", "usage", "response"}})

                status = answer_status
                if api_error:
                    status = "API_ERROR"
                elif parse_error:
                    status = "PARSE_ERROR"

                img_flag = " img=Y" if has_image else ""
                print(
                    f"rep={replicate} q={original_question_number:03d}{img_flag} "
                    f"ans={model_answer} correct={correct_answer} status={status} "
                    f"latency={latency_seconds:.2f}s"
                )

    results = pd.DataFrame(all_rows)
    results.to_csv(results_file, index=False, encoding="utf-8")

    summary = build_summary(results)
    summary.to_csv(summary_file, index=False, encoding="utf-8")

    image_summary = build_summary_by_image_status(results)
    image_summary.to_csv(image_summary_file, index=False, encoding="utf-8")

    metadata["completed_utc"] = utc_timestamp()
    metadata["outputs"] = {
        "raw_outputs_file": str(raw_outputs_file),
        "scored_results_file": str(results_file),
        "summary_file": str(summary_file),
        "summary_by_image_status_file": str(image_summary_file),
        "metadata_file": str(metadata_file),
    }

    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nDONE")
    print(f"Raw outputs: {raw_outputs_file}")
    print(f"Scored results: {results_file}")
    print(f"Summary: {summary_file}")
    print(f"Summary by image status: {image_summary_file}")
    print(f"Image manifest: {image_manifest_file}")
    print(f"Metadata: {metadata_file}")
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nSummary by image status:")
    print(image_summary.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

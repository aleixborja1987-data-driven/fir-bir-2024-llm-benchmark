"""
BIR 2024 - OpenAI GPT-5.2 item-wise API benchmark (2 runs)

This script evaluates the 210 questions of the Spanish BIR 2024 Biology exam
using direct OpenAI Responses API calls, one question at a time, with controlled
sampling parameters and full per-question logging.

Input files expected in the same folder as this script:
  - bir2024_questions_210.csv
  - bir2024_answer_key_210.csv
  - .env containing OPENAI_API_KEY=...

Main outputs are written to a timestamped folder under outputs/:
  - *_raw_outputs.jsonl       Full raw per-call logs, one JSON object per API call
  - *_scored_results.csv      Flat scored per-call table for analysis/Excel
  - *_summary_by_replicate.csv
  - *_run_metadata.json

Requirements:
  pip install openai python-dotenv pandas
"""

import csv
import hashlib
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# STUDY CONFIGURATION
# =========================

MODEL_REQUESTED = "gpt-5.2-2025-12-11"
PROVIDER = "OpenAI"
INTERFACE = "API"
API_ENDPOINT = "Responses API"
PROMPT_TEMPLATE_VERSION = "bir2024_text_itemwise_examlike_blank_v1"
SCRIPT_VERSION = "bir2024_gpt52_itemwise_2runs_v1"

# Controlled generation parameters.
# Important: do NOT send reasoning.effort. Prior compatibility testing showed that
# GPT-5.2 accepts temperature/top_p with Responses API when reasoning is not sent.
GENERATION_PARAMETERS = {
    "temperature": 0,
    "top_p": 1,
    "max_output_tokens": 300,
    "reasoning_effort": "not_sent",
    "stream": False,
    "store": False,
    "tools_enabled": False,
    "web_search_enabled": False,
}

N_REPLICATES = 2
REQUEST_TIMEOUT_SECONDS = 120
MAX_ATTEMPTS_PER_CALL = 3
RETRY_SLEEP_SECONDS = 5

QUESTIONS_FILE = Path("bir2024_questions_210.csv")
ANSWER_KEY_FILE = Path("bir2024_answer_key_210.csv")
OUTPUT_ROOT = Path("outputs")

RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
SAFE_MODEL_NAME = MODEL_REQUESTED.replace(".", "_").replace("-", "_")
RUN_NAME = f"bir2024_{SAFE_MODEL_NAME}_itemwise_2runs_{RUN_TIMESTAMP}"
OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME

RAW_OUTPUTS_FILE = OUTPUT_DIR / f"{RUN_NAME}_raw_outputs.jsonl"
RESULTS_FILE = OUTPUT_DIR / f"{RUN_NAME}_scored_results.csv"
SUMMARY_FILE = OUTPUT_DIR / f"{RUN_NAME}_summary_by_replicate.csv"
METADATA_FILE = OUTPUT_DIR / f"{RUN_NAME}_run_metadata.json"


# =========================
# PROMPT AND SCHEMA
# =========================

PROMPT_PREAMBLE = """You are taking an examination for Biology residency access.

This is a multiple-choice exam. Each question has four options, numbered 1 to 4, and only one is correct. In the official exam, correct answers score +1 point, incorrect answers score -1/3 points, and blank answers score 0 points. Act as a human candidate and try to get the highest score possible.

Return only valid JSON with:
{"answer": "1|2|3|4|blank", "rationale": "brief explanation"}
"""

JSON_SCHEMA = {
    "format": {
        "type": "json_schema",
        "name": "bir2024_mcq_answer_with_blank",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "enum": ["1", "2", "3", "4", "blank"],
                },
                "rationale": {
                    "type": "string",
                },
            },
            "required": ["answer", "rationale"],
            "additionalProperties": False,
        },
    }
}


def build_prompt(row: pd.Series) -> str:
    """Builds the exact prompt sent to the model. The correct answer is never included."""
    return f"""{PROMPT_PREAMBLE}
Question {int(row['original_question_number'])}:
{row['question_text']}

Options:
1. {row['option_1']}
2. {row['option_2']}
3. {row['option_3']}
4. {row['option_4']}
"""


# =========================
# HELPER FUNCTIONS
# =========================


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


def normalize_answer(value: Any) -> Optional[str]:
    if value is None:
        return None
    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    # Handle values read as floats from CSV, e.g. 2.0
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]

    if text in {"", "nan", "none", "null"}:
        return None

    if text in {"1", "2", "3", "4", "blank"}:
        return text

    return text


def parse_model_answer(raw_text: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Returns parsed_answer, rationale, parse_error."""
    if not raw_text:
        return None, None, "missing_raw_output"

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return None, None, f"json_decode_error: {str(e)}"

    answer = normalize_answer(data.get("answer"))
    rationale = data.get("rationale")

    if answer is None:
        return None, rationale, "missing_answer"

    if answer not in ["1", "2", "3", "4", "blank"]:
        return answer, rationale, "invalid_answer"

    return answer, rationale, None


def object_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    try:
        return dict(obj)
    except Exception:
        return {"repr": repr(obj)}


def get_usage_dict(response: Any) -> Dict[str, Any]:
    return object_to_dict(getattr(response, "usage", None))


def get_response_dump(response: Any) -> Dict[str, Any]:
    return object_to_dict(response)


def flatten_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(usage, dict):
        usage = {}

    output_details = usage.get("output_tokens_details") or {}
    input_details = usage.get("input_tokens_details") or {}

    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": output_details.get("reasoning_tokens"),
        "cached_input_tokens": input_details.get("cached_tokens"),
    }


def safe_tokens_per_second(tokens: Optional[float], latency_seconds: Optional[float]) -> Optional[float]:
    if tokens is None or latency_seconds is None:
        return None
    try:
        if latency_seconds <= 0:
            return None
        return float(tokens) / float(latency_seconds)
    except Exception:
        return None


def get_finish_reason(response: Any) -> Optional[str]:
    """Responses API objects may expose finish reasons in nested outputs depending on model."""
    try:
        output = getattr(response, "output", None)
        if output and len(output) > 0:
            first = output[0]
            return getattr(first, "finish_reason", None)
    except Exception:
        return None
    return None


def compute_scoring_status_and_score(row: pd.Series) -> Tuple[str, Optional[bool], bool, Optional[float]]:
    """Returns scoring_status, is_correct, is_blank, exam_score."""
    error_type = row.get("error_type")
    parse_error = row.get("parse_error")
    answer = normalize_answer(row.get("parsed_answer"))
    correct_answer = normalize_answer(row.get("correct_answer"))
    official_answer_available = row.get("official_answer_available")

    if pd.notna(error_type) and str(error_type).strip() not in ["", "None", "nan"]:
        return "api_error", None, False, None

    if pd.notna(parse_error) and str(parse_error).strip() not in ["", "None", "nan"]:
        return "parse_error", None, False, None

    if official_answer_available is False or correct_answer is None:
        return "no_official_answer", None, answer == "blank", None

    if answer == "blank":
        return "blank", False, True, 0.0

    if answer == correct_answer:
        return "correct", True, False, 1.0

    return "incorrect", False, False, -1.0 / 3.0


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_FILE}")
    if not ANSWER_KEY_FILE.exists():
        raise FileNotFoundError(f"Answer key file not found: {ANSWER_KEY_FILE}")

    questions = pd.read_csv(QUESTIONS_FILE)
    answer_key = pd.read_csv(ANSWER_KEY_FILE)

    required_question_cols = {
        "exam",
        "question_id",
        "original_question_number",
        "is_reserve",
        "subject",
        "has_image",
        "question_text",
        "option_1",
        "option_2",
        "option_3",
        "option_4",
        "source_pdf_page",
    }
    required_key_cols = {
        "exam",
        "question_id",
        "original_question_number",
        "is_reserve",
        "correct_answer",
        "official_answer_available",
        "is_annulled_or_without_official_answer",
    }

    missing_q = required_question_cols - set(questions.columns)
    missing_k = required_key_cols - set(answer_key.columns)

    if missing_q:
        raise ValueError(f"Missing columns in {QUESTIONS_FILE}: {missing_q}")
    if missing_k:
        raise ValueError(f"Missing columns in {ANSWER_KEY_FILE}: {missing_k}")

    if len(questions) != 210:
        raise ValueError(f"Expected 210 questions, found {len(questions)}")

    if len(answer_key) != 210:
        raise ValueError(f"Expected 210 answer-key rows, found {len(answer_key)}")

    # Normalize answer key now to prevent float-like values (e.g. 2.0) from causing mismatches.
    answer_key["correct_answer"] = answer_key["correct_answer"].apply(normalize_answer)

    return questions, answer_key


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# =========================
# API CALL
# =========================


def call_openai_responses(client: OpenAI, prompt: str) -> Dict[str, Any]:
    """Makes one API call and returns a normalized record.

    The model is not provided with any tools, web search, file search, or external retrieval.
    Internet lookup during inference is therefore not available through this API call.
    """
    error_type = None
    error_message = None
    error_traceback = None
    raw_text = None
    model_returned = None
    finish_reason = None
    response_id = None
    usage_dict: Dict[str, Any] = {}
    response_dump: Dict[str, Any] = {}
    attempts_used = 0

    start = time.perf_counter()

    for attempt in range(1, MAX_ATTEMPTS_PER_CALL + 1):
        attempts_used = attempt
        try:
            response = client.responses.create(
                model=MODEL_REQUESTED,
                input=prompt,
                temperature=GENERATION_PARAMETERS["temperature"],
                top_p=GENERATION_PARAMETERS["top_p"],
                max_output_tokens=GENERATION_PARAMETERS["max_output_tokens"],
                text=JSON_SCHEMA,
                store=GENERATION_PARAMETERS["store"],
            )

            raw_text = response.output_text
            model_returned = getattr(response, "model", None)
            response_id = getattr(response, "id", None)
            finish_reason = get_finish_reason(response)
            usage_dict = get_usage_dict(response)
            response_dump = get_response_dump(response)
            error_type = None
            error_message = None
            error_traceback = None
            break

        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            error_traceback = traceback.format_exc()

            # Retry only transient-looking issues. For invalid requests, repeated attempts will not help.
            non_retryable_markers = [
                "Unsupported parameter",
                "invalid_request_error",
                "model_not_found",
                "insufficient_quota",
                "authentication",
            ]
            if any(marker.lower() in error_message.lower() for marker in non_retryable_markers):
                break

            if attempt < MAX_ATTEMPTS_PER_CALL:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)

    latency_seconds = time.perf_counter() - start

    parsed_answer, rationale, parse_error = parse_model_answer(raw_text)
    usage_flat = flatten_usage(usage_dict)

    output_tps = safe_tokens_per_second(usage_flat.get("output_tokens"), latency_seconds)
    total_tps = safe_tokens_per_second(usage_flat.get("total_tokens"), latency_seconds)

    return {
        "model_returned": model_returned,
        "response_id": response_id,
        "finish_reason": finish_reason,
        "raw_output": raw_text,
        "parsed_answer": parsed_answer,
        "rationale": rationale,
        "parse_error": parse_error,
        "usage": usage_dict,
        "response_dump": response_dump,
        "error_type": error_type,
        "error_message": error_message,
        "error_traceback": error_traceback,
        "latency_seconds": latency_seconds,
        "attempts_used": attempts_used,
        **usage_flat,
        "output_tokens_per_second": output_tps,
        "total_tokens_per_second": total_tps,
    }


# =========================
# MAIN EXECUTION
# =========================


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Check your .env file.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    questions, answer_key = load_inputs()

    questions_hash = sha256_file(QUESTIONS_FILE)
    answer_key_hash = sha256_file(ANSWER_KEY_FILE)
    prompt_preamble_hash = sha256_text(PROMPT_PREAMBLE)

    metadata = {
        "run_name": RUN_NAME,
        "run_timestamp_utc": RUN_TIMESTAMP,
        "script_version": SCRIPT_VERSION,
        "provider": PROVIDER,
        "interface": INTERFACE,
        "api_endpoint": API_ENDPOINT,
        "model_requested": MODEL_REQUESTED,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_preamble": PROMPT_PREAMBLE,
        "prompt_preamble_sha256": prompt_preamble_hash,
        "generation_parameters": GENERATION_PARAMETERS,
        "n_replicates": N_REPLICATES,
        "questions_file": str(QUESTIONS_FILE),
        "answer_key_file": str(ANSWER_KEY_FILE),
        "questions_file_sha256": questions_hash,
        "answer_key_file_sha256": answer_key_hash,
        "n_questions": int(len(questions)),
        "n_answer_key_rows": int(len(answer_key)),
        "tools_or_web_search_provided_to_model": False,
        "notes": (
            "The official answer key is not included in prompts or API requests. "
            "It is merged only after model generation for offline scoring. "
            "No tools, web search, file search, or external retrieval are provided to the model."
        ),
    }

    METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    RAW_OUTPUTS_FILE.write_text("", encoding="utf-8")

    client = OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)

    flat_rows = []
    total_calls = len(questions) * N_REPLICATES
    call_index = 0

    print(f"Run name: {RUN_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total API calls planned: {total_calls}")
    print("Important: this will spend API tokens. Close Excel files before running.\n")

    for replicate_id in range(1, N_REPLICATES + 1):
        for _, row in questions.iterrows():
            call_index += 1
            prompt = build_prompt(row)
            prompt_hash = sha256_text(prompt)

            request_time_utc = datetime.now(timezone.utc).isoformat()
            run_id = (
                f"{RUN_TIMESTAMP}_rep{replicate_id}_"
                f"{row['exam']}_{row['question_id']}_{SAFE_MODEL_NAME}"
            )

            print(
                f"[{call_index}/{total_calls}] "
                f"rep={replicate_id} {row['exam']} {row['question_id']} "
                f"(original {row['original_question_number']})"
            )

            api_result = call_openai_responses(client, prompt)
            response_time_utc = datetime.now(timezone.utc).isoformat()

            base_record = {
                "run_id": run_id,
                "run_name": RUN_NAME,
                "replicate_id": replicate_id,
                "provider": PROVIDER,
                "interface": INTERFACE,
                "api_endpoint": API_ENDPOINT,
                "model_requested": MODEL_REQUESTED,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "script_version": SCRIPT_VERSION,
                "generation_parameters": GENERATION_PARAMETERS,
                "exam": row["exam"],
                "question_id": row["question_id"],
                "original_question_number": int(row["original_question_number"]),
                "is_reserve": bool(row["is_reserve"]),
                "subject": row.get("subject"),
                "has_image": bool(row["has_image"]),
                "source_pdf_page": int(row["source_pdf_page"]),
                "question_text": row["question_text"],
                "option_1": row["option_1"],
                "option_2": row["option_2"],
                "option_3": row["option_3"],
                "option_4": row["option_4"],
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
                "prompt_preamble_sha256": prompt_preamble_hash,
                "questions_file_sha256": questions_hash,
                "answer_key_file_sha256": answer_key_hash,
                "request_time_utc": request_time_utc,
                "response_time_utc": response_time_utc,
                "temperature": GENERATION_PARAMETERS["temperature"],
                "top_p": GENERATION_PARAMETERS["top_p"],
                "max_output_tokens": GENERATION_PARAMETERS["max_output_tokens"],
                "reasoning_effort": GENERATION_PARAMETERS["reasoning_effort"],
                "stream": GENERATION_PARAMETERS["stream"],
                "store": GENERATION_PARAMETERS["store"],
                "tools_enabled": GENERATION_PARAMETERS["tools_enabled"],
                "web_search_enabled": GENERATION_PARAMETERS["web_search_enabled"],
            }

            raw_record = {**base_record, **api_result}
            write_jsonl(RAW_OUTPUTS_FILE, raw_record)

            flat_row = {
                **base_record,
                "model_returned": api_result.get("model_returned"),
                "response_id": api_result.get("response_id"),
                "finish_reason": api_result.get("finish_reason"),
                "raw_output": api_result.get("raw_output"),
                "parsed_answer": api_result.get("parsed_answer"),
                "rationale": api_result.get("rationale"),
                "parse_error": api_result.get("parse_error"),
                "error_type": api_result.get("error_type"),
                "error_message": api_result.get("error_message"),
                "latency_seconds": round(api_result.get("latency_seconds"), 6)
                if api_result.get("latency_seconds") is not None else None,
                "attempts_used": api_result.get("attempts_used"),
                "input_tokens": api_result.get("input_tokens"),
                "output_tokens": api_result.get("output_tokens"),
                "total_tokens": api_result.get("total_tokens"),
                "reasoning_tokens": api_result.get("reasoning_tokens"),
                "cached_input_tokens": api_result.get("cached_input_tokens"),
                "output_tokens_per_second": round(api_result.get("output_tokens_per_second"), 6)
                if api_result.get("output_tokens_per_second") is not None else None,
                "total_tokens_per_second": round(api_result.get("total_tokens_per_second"), 6)
                if api_result.get("total_tokens_per_second") is not None else None,
            }
            flat_rows.append(flat_row)

    results_df = pd.DataFrame(flat_rows)

    # Merge answer key only after generation. The model never sees these columns.
    scored = results_df.merge(
        answer_key,
        on=["exam", "question_id", "original_question_number", "is_reserve"],
        how="left",
    )

    # Normalize for safe comparisons.
    scored["parsed_answer"] = scored["parsed_answer"].apply(normalize_answer)
    scored["correct_answer"] = scored["correct_answer"].apply(normalize_answer)

    scoring_results = scored.apply(compute_scoring_status_and_score, axis=1, result_type="expand")
    scoring_results.columns = ["scoring_status", "is_correct", "is_blank", "exam_score"]
    scored = pd.concat([scored, scoring_results], axis=1)

    scored.to_csv(RESULTS_FILE, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")

    # Summary by replicate. Score/accuracy only among rows with official answer available and no API/parse errors.
    summary_rows = []
    for replicate_id, group in scored.groupby("replicate_id"):
        evaluable = group[group["official_answer_available"] == True].copy()
        valid = evaluable[evaluable["scoring_status"].isin(["correct", "incorrect", "blank"])]

        n_evaluable = len(evaluable)
        n_valid = len(valid)
        n_correct = int((valid["scoring_status"] == "correct").sum())
        n_incorrect = int((valid["scoring_status"] == "incorrect").sum())
        n_blank = int((valid["scoring_status"] == "blank").sum())
        n_api_error = int((group["scoring_status"] == "api_error").sum())
        n_parse_error = int((group["scoring_status"] == "parse_error").sum())
        n_no_official_answer = int((group["scoring_status"] == "no_official_answer").sum())

        accuracy_answered = n_correct / (n_correct + n_incorrect) if (n_correct + n_incorrect) else None
        accuracy_evaluable = n_correct / n_evaluable if n_evaluable else None
        total_exam_score = valid["exam_score"].sum(skipna=True)

        summary_rows.append({
            "run_name": RUN_NAME,
            "replicate_id": replicate_id,
            "model_requested": MODEL_REQUESTED,
            "n_total_questions_run": len(group),
            "n_evaluable_with_official_answer": n_evaluable,
            "n_valid_scored": n_valid,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_blank": n_blank,
            "n_api_error": n_api_error,
            "n_parse_error": n_parse_error,
            "n_no_official_answer": n_no_official_answer,
            "accuracy_among_answered_nonblank": accuracy_answered,
            "accuracy_among_evaluable_questions_counting_blank_as_not_correct": accuracy_evaluable,
            "official_exam_score_sum": total_exam_score,
            "mean_latency_seconds": valid["latency_seconds"].mean(),
            "median_latency_seconds": valid["latency_seconds"].median(),
            "mean_output_tokens_per_second": valid["output_tokens_per_second"].mean(),
            "mean_total_tokens_per_second": valid["total_tokens_per_second"].mean(),
            "total_input_tokens": group["input_tokens"].sum(skipna=True),
            "total_output_tokens": group["output_tokens"].sum(skipna=True),
            "total_tokens": group["total_tokens"].sum(skipna=True),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_FILE, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Raw outputs saved to: {RAW_OUTPUTS_FILE}")
    print(f"Scored results saved to: {RESULTS_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")
    print("\nSummary by replicate:")
    print(summary[[
        "replicate_id",
        "n_evaluable_with_official_answer",
        "n_correct",
        "n_incorrect",
        "n_blank",
        "n_api_error",
        "n_parse_error",
        "official_exam_score_sum",
        "accuracy_among_evaluable_questions_counting_blank_as_not_correct",
        "mean_latency_seconds",
        "mean_output_tokens_per_second",
        "total_tokens",
    ]])


if __name__ == "__main__":
    main()

"""
BIR 2024 - Qwen3-VL-235B-A22B-Instruct item-wise API benchmark (2 runs)
STRICT OUTPUT VERSION

Fixed model:
  MODEL_REQUESTED = "qwen3-vl-235b-a22b-instruct"

Why this strict version exists:
  Qwen's OpenAI-compatible response_format={"type":"json_object"} enforces JSON,
  but it does not enforce an enum for the "answer" field. In an initial run,
  the model sometimes returned values such as "3,4" or "4,", which are not
  valid exam answers. This version keeps the same study conditions but makes
  the output constraints explicit in the prompt and scores invalid outputs as
  missing_or_invalid = 0, matching the parse-error handling used elsewhere.

Input files expected in the same folder as this script:
  - bir2024_questions_210.csv
  - bir2024_answer_key_210.csv
  - .env containing:
      DASHSCOPE_API_KEY=...
      QWEN_BASE_URL=.../compatible-mode/v1

Cheap probe:
  python run_bir2024_qwen3_vl_235b_a22b_instruct_itemwise_2runs_STRICT.py --limit 5

Full run:
  python run_bir2024_qwen3_vl_235b_a22b_instruct_itemwise_2runs_STRICT.py

Requirements:
  pip install openai python-dotenv pandas truststore
"""

import argparse
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

PROMPT_TEMPLATE_VERSION = "bir2024_text_itemwise_examlike_blank_strict_single_answer_v2"
SCRIPT_VERSION = "bir2024_qwen3_vl_235b_a22b_instruct_itemwise_2runs_strict_v2"

GENERATION_PARAMETERS = {
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 300,
    "response_format": {"type": "json_object"},
    "stream": False,
    "tools_enabled": False,
    "web_search_enabled": False,
    "file_input_enabled": False,
    "image_input_enabled": False,
    "enable_thinking": False,
}

N_REPLICATES_DEFAULT = 2
REQUEST_TIMEOUT_SECONDS = 120
MAX_ATTEMPTS_PER_CALL = 3
RETRY_SLEEP_SECONDS = 5

QUESTIONS_FILE = Path("bir2024_questions_210.csv")
ANSWER_KEY_FILE = Path("bir2024_answer_key_210.csv")
OUTPUT_ROOT = Path("outputs")


# =========================
# PROMPT
# =========================

PROMPT_PREAMBLE = """You are taking an examination for Biology residency access.

This is a multiple-choice exam. Each question has four options, numbered 1 to 4, and only one is correct. In the official exam, correct answers score +1 point, incorrect answers score -1/3 points, and blank answers score 0 points. Act as a human candidate and try to get the highest score possible.

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

    Invalid answer-field values are returned as model_answer=None so that they
    are scored as missing_or_invalid = 0 rather than as ordinary incorrect
    answers. The invalid raw value is still documented in parse_error.
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


def call_qwen(client: OpenAI, prompt: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    kwargs = dict(
        model=MODEL_REQUESTED,
        messages=[{"role": "user", "content": prompt}],
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
        return response_dict, raw_content, None
    except Exception as e:
        return None, None, repr(e)


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
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Optional cheap probe, e.g. --limit 5.")
    parser.add_argument("--replicates", type=int, default=N_REPLICATES_DEFAULT, help="Number of replicates. Default: 2.")
    args = parser.parse_args()

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

    if args.limit is not None:
        questions = questions.head(args.limit).copy()

    run_timestamp = utc_timestamp()
    safe_name = safe_model_name(MODEL_REQUESTED)
    limit_suffix = f"_limit{args.limit}" if args.limit is not None else ""
    run_name = f"bir2024_{safe_name}_itemwise_2runs_STRICT{limit_suffix}_{run_timestamp}"
    output_dir = OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_outputs_file = output_dir / f"{run_name}_raw_outputs.jsonl"
    results_file = output_dir / f"{run_name}_scored_results.csv"
    summary_file = output_dir / f"{run_name}_summary_by_replicate.csv"
    metadata_file = output_dir / f"{run_name}_run_metadata.json"

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
        "n_replicates": args.replicates,
        "n_questions_attempted": int(len(questions)),
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

                for attempt in range(1, MAX_ATTEMPTS_PER_CALL + 1):
                    attempts_used = attempt
                    start = time.time()
                    response_dict, raw_text, api_error = call_qwen(client=client, prompt=prompt)
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
                    "has_image": bool(row.get("has_image", False)),
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

                print(
                    f"rep={replicate} q={original_question_number:03d} "
                    f"ans={model_answer} correct={correct_answer} status={status} "
                    f"latency={latency_seconds:.2f}s"
                )

    results = pd.DataFrame(all_rows)
    results.to_csv(results_file, index=False, encoding="utf-8")

    summary = build_summary(results)
    summary.to_csv(summary_file, index=False, encoding="utf-8")

    metadata["completed_utc"] = utc_timestamp()
    metadata["outputs"] = {
        "raw_outputs_file": str(raw_outputs_file),
        "scored_results_file": str(results_file),
        "summary_file": str(summary_file),
        "metadata_file": str(metadata_file),
    }

    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nDONE")
    print(f"Raw outputs: {raw_outputs_file}")
    print(f"Scored results: {results_file}")
    print(f"Summary: {summary_file}")
    print(f"Metadata: {metadata_file}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

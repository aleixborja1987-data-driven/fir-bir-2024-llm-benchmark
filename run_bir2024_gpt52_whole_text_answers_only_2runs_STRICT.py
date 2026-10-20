"""
BIR 2024 - GPT-5.2 whole-text API benchmark (2 runs)
ANSWERS-ONLY STRICT VERSION

Fixed model:
  MODEL_REQUESTED = "gpt-5.2-2025-12-11"

Condition:
  Whole-text answers-only.
  The full BIR 2024 exam text is sent in one API call per replicate.
  No PDF file and no images are sent.

Why answers-only:
  This mirrors the Qwen whole-text answers-only condition, allowing a fairer
  comparison with a purely textual full-exam prompt rather than OpenAI-native
  PDF input. Rationales are not requested to reduce output length and avoid
  making this condition depend on long generated explanations.

Input files expected in the same folder as this script:
  - bir2024_questions_210.csv
  - bir2024_answer_key_210.csv
  - .env containing:
      OPENAI_API_KEY=...

Probe with first 30 questions:
  python run_bir2024_gpt52_whole_text_answers_only_2runs_STRICT.py --limit 30

Full run:
  python run_bir2024_gpt52_whole_text_answers_only_2runs_STRICT.py

Optional single replicate:
  python run_bir2024_gpt52_whole_text_answers_only_2runs_STRICT.py --replicates 1

Requirements:
  pip install openai python-dotenv pandas truststore
"""

import argparse
import hashlib
import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Windows / corporate SSL compatibility. Harmless if not needed.
try:
    import truststore
    truststore.inject_into_ssl()
    TRUSTSTORE_ENABLED = True
except Exception:
    TRUSTSTORE_ENABLED = False


# =========================
# STUDY CONFIGURATION
# =========================

PROVIDER = "OpenAI"
INTERFACE = "API"
API_ENDPOINT = "Responses API"

MODEL_REQUESTED = "gpt-5.2-2025-12-11"

PROMPT_TEMPLATE_VERSION = "bir2024_whole_text_examlike_blank_strict_answers_only_v1"
SCRIPT_VERSION = "bir2024_gpt52_whole_text_answers_only_2runs_strict_v1"

GENERATION_PARAMETERS = {
    "temperature": 0,
    "top_p": 1,
    "max_output_tokens": 10000,
    "text_format": "json_schema_strict",
    "store": False,
    "stream": False,
    "tools_enabled": False,
    "web_search_enabled": False,
    "file_input_enabled": False,
    "image_input_enabled": False,
    "reasoning_effort": "not_sent",
    "rationales_requested": False,
}

N_REPLICATES_DEFAULT = 2
REQUEST_TIMEOUT_SECONDS = 600
MAX_ATTEMPTS_PER_CALL = 3
RETRY_SLEEP_SECONDS = 10

QUESTIONS_FILE = Path("bir2024_questions_210.csv")
ANSWER_KEY_FILE = Path("bir2024_answer_key_210.csv")
OUTPUT_ROOT = Path("outputs")


# =========================
# PROMPT
# =========================

PROMPT_PREAMBLE = """You are taking an examination for Biology residency access.

This is a multiple-choice exam. Each question has four options, numbered 1 to 4, and only one is correct. In the official exam, correct answers score +1 point, incorrect answers score -1/3 points, and blank answers score 0 points. Act as a human candidate and try to get the highest score possible.

You will receive the full exam text, containing all questions and their four answer options. Do not use web search, tools, external references, files, images, or an answer key.

Return only valid JSON with exactly this shape:
{
  "answers": [
    {"question_number": 1, "answer": "1|2|3|4|blank"},
    {"question_number": 2, "answer": "1|2|3|4|blank"}
  ]
}

Output rules:
- Return exactly one answer object for every question included in the input.
- Use the integer question number shown in the input.
- The value of "answer" must be exactly one single string: "1", "2", "3", "4", or "blank".
- Do not return multiple options.
- Do not return values such as "1,2", "1 or 2", "3/4", "4,", lists, arrays, ranges, or explanations inside the "answer" field.
- If you cannot choose one single best option, return "blank".
- Do not include rationales, explanations, comments, markdown, or any other fields.
"""


def format_question(row: pd.Series) -> str:
    qn = int(row["original_question_number"])
    return f"""Question {qn}:
{row['question_text']}

Options:
1. {row['option_1']}
2. {row['option_2']}
3. {row['option_3']}
4. {row['option_4']}"""


def build_whole_exam_prompt(questions: pd.DataFrame) -> str:
    blocks = [format_question(row) for _, row in questions.iterrows()]
    return PROMPT_PREAMBLE + "\n\nFULL EXAM TEXT:\n\n" + "\n\n---\n\n".join(blocks)


def build_json_schema(n_questions: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "minItems": n_questions,
                "maxItems": n_questions,
                "items": {
                    "type": "object",
                    "properties": {
                        "question_number": {"type": "integer"},
                        "answer": {
                            "type": "string",
                            "enum": ["1", "2", "3", "4", "blank"],
                        },
                    },
                    "required": ["question_number", "answer"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["answers"],
        "additionalProperties": False,
    }


# =========================
# HELPERS
# =========================

def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def normalize_qnum(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()
    match = re.search(r"\d+", text)
    if not match:
        return None
    return int(match.group(0))


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


def extract_response_text(response: Any) -> Optional[str]:
    """
    Robustly extracts text from an OpenAI Responses API response.
    """
    if response is None:
        return None

    # Newer SDKs expose output_text.
    try:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text
    except Exception:
        pass

    # Dict fallback.
    try:
        d = response.model_dump() if hasattr(response, "model_dump") else response
    except Exception:
        d = None

    if not isinstance(d, dict):
        return None

    texts = []

    for item in d.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict):
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    texts.append(content.get("text"))
                elif content.get("type") == "message":
                    maybe_text = content.get("text")
                    if maybe_text:
                        texts.append(maybe_text)

    return "\n".join(texts) if texts else None


def load_json_robust(raw_text: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not raw_text:
        return None, "missing_raw_output"

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
            return None, f"json_decode_error: {str(e)}"

    if not isinstance(data, dict):
        return None, "json_not_object"

    return data, None


def extract_answer_records(data: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], Optional[str]]:
    records: Dict[int, Dict[str, Any]] = {}

    answers = None
    for key in ["answers", "responses", "results"]:
        if isinstance(data.get(key), list):
            answers = data.get(key)
            break

    if isinstance(answers, list):
        for item in answers:
            if not isinstance(item, dict):
                continue

            qn = None
            for qkey in ["question_number", "question", "q", "id", "number"]:
                if qkey in item:
                    qn = normalize_qnum(item.get(qkey))
                    break

            if qn is None:
                continue

            records[qn] = {
                "raw_answer": item.get("answer"),
                "raw_record": item,
            }

        return records, None

    possible_q_keys = [k for k in data.keys() if normalize_qnum(k) is not None]
    if possible_q_keys:
        for key in possible_q_keys:
            qn = normalize_qnum(key)
            value = data.get(key)

            if isinstance(value, dict):
                raw_answer = value.get("answer")
            else:
                raw_answer = value

            records[qn] = {
                "raw_answer": raw_answer,
                "raw_record": value,
            }

        return records, None

    return records, "answers_key_missing_or_not_list"


def call_openai(client: OpenAI, prompt: str, n_questions: int) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    Returns response_dict, raw_text, api_error.
    """
    schema = build_json_schema(n_questions)

    try:
        response = client.responses.create(
            model=MODEL_REQUESTED,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            temperature=GENERATION_PARAMETERS["temperature"],
            top_p=GENERATION_PARAMETERS["top_p"],
            max_output_tokens=GENERATION_PARAMETERS["max_output_tokens"],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "exam_answers_schema",
                    "strict": True,
                    "schema": schema,
                }
            },
            store=False,
        )

        response_dict = response.model_dump()
        raw_text = extract_response_text(response)
        return response_dict, raw_text, None

    except Exception as e:
        return None, None, repr(e)


def extract_usage(response_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not response_dict:
        return {}
    return response_dict.get("usage") or {}


def build_summary(results: pd.DataFrame, call_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for replicate, group in results.groupby("replicate"):
        evaluable = group[group["has_official_answer"] == True].copy()
        n_eval = len(evaluable)
        correct = int((evaluable["answer_status"] == "correct").sum())
        incorrect = int((evaluable["answer_status"] == "incorrect").sum())
        blank = int((evaluable["answer_status"] == "blank").sum())
        missing_invalid = int((evaluable["answer_status"] == "missing_or_invalid").sum())
        score = evaluable["score"].sum(skipna=True)

        call = call_rows[call_rows["replicate"] == replicate]
        call = call.iloc[0].to_dict() if len(call) else {}

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
            "api_errors": int(bool(call.get("api_error"))),
            "parse_errors": int(group["parse_error"].notna().sum()),
            "whole_response_parse_error": call.get("whole_response_parse_error"),
            "n_answer_records_returned": call.get("n_answer_records_returned"),
            "n_unique_answer_records_returned": call.get("n_unique_answer_records_returned"),
            "latency_seconds": call.get("latency_seconds"),
            "input_tokens": call.get("input_tokens"),
            "output_tokens": call.get("output_tokens"),
            "reasoning_tokens": call.get("reasoning_tokens"),
            "total_tokens": call.get("total_tokens"),
            "response_status": call.get("response_status"),
            "finish_reasons": call.get("finish_reasons"),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Optional probe, e.g. --limit 30.")
    parser.add_argument("--replicates", type=int, default=N_REPLICATES_DEFAULT, help="Number of replicates. Default: 2.")
    args = parser.parse_args()

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing {QUESTIONS_FILE}")
    if not ANSWER_KEY_FILE.exists():
        raise FileNotFoundError(f"Missing {ANSWER_KEY_FILE}")

    questions = pd.read_csv(QUESTIONS_FILE)
    answer_key = pd.read_csv(ANSWER_KEY_FILE)
    questions = questions.sort_values("original_question_number").reset_index(drop=True)

    if args.limit is not None:
        questions = questions.head(args.limit).copy()

    prompt = build_whole_exam_prompt(questions)
    prompt_hash = sha256_text(prompt)

    run_timestamp = utc_timestamp()
    limit_suffix = f"_limit{args.limit}" if args.limit is not None else ""
    run_name = f"bir2024_gpt52_wholetext_ansonly_2r_STRICT{limit_suffix}_{run_timestamp}"
    output_dir = OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_outputs_file = output_dir / f"{run_name}_raw_outputs.jsonl"
    scored_results_file = output_dir / f"{run_name}_scored_results.csv"
    summary_file = output_dir / f"{run_name}_summary_by_replicate.csv"
    metadata_file = output_dir / f"{run_name}_run_metadata.json"
    call_level_file = output_dir / f"{run_name}_call_level.csv"
    prompt_log_file = output_dir / f"{run_name}_prompt_log.txt"

    prompt_log_file.write_text(prompt, encoding="utf-8")

    client = OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)

    metadata = {
        "run_name": run_name,
        "created_utc": run_timestamp,
        "provider": PROVIDER,
        "interface": INTERFACE,
        "api_endpoint": API_ENDPOINT,
        "model_requested": MODEL_REQUESTED,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_preamble_hash_sha256": sha256_text(PROMPT_PREAMBLE),
        "prompt_hash_sha256": prompt_hash,
        "prompt_preamble": PROMPT_PREAMBLE,
        "prompt_log_file": str(prompt_log_file),
        "script_version": SCRIPT_VERSION,
        "questions_file": str(QUESTIONS_FILE),
        "questions_file_sha256": sha256_file(QUESTIONS_FILE),
        "answer_key_file": str(ANSWER_KEY_FILE),
        "answer_key_file_sha256": sha256_file(ANSWER_KEY_FILE),
        "n_replicates": args.replicates,
        "n_questions_attempted": int(len(questions)),
        "condition": "whole_text_answers_only",
        "generation_parameters": GENERATION_PARAMETERS,
        "scoring": {
            "correct": 1,
            "incorrect": -1/3,
            "blank": 0,
            "missing_or_invalid": 0,
            "invalid_answer_field": 0,
            "questions_without_official_answer_excluded": True,
        },
        "truststore_enabled": TRUSTSTORE_ENABLED,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
    }

    all_rows: List[Dict[str, Any]] = []
    call_rows: List[Dict[str, Any]] = []

    print(f"Run name: {run_name}")
    print(f"Provider: {PROVIDER}")
    print(f"Fixed model: {MODEL_REQUESTED}")
    print("Condition: whole_text_answers_only")
    print(f"Questions included: {len(questions)}")
    print(f"Replicates: {args.replicates}")
    print(f"Prompt characters: {len(prompt):,}")
    print(f"Output directory: {output_dir}")
    print("Strict JSON schema: True")
    print("Rationales requested: False")
    print("Reasoning effort sent: False")
    print()

    with raw_outputs_file.open("w", encoding="utf-8") as raw_f:
        for replicate in range(1, args.replicates + 1):
            print(f"=== Replicate {replicate}/{args.replicates} ===")

            response_dict = None
            raw_text = None
            api_error = None
            latency_seconds = None
            attempts_used = 0

            for attempt in range(1, MAX_ATTEMPTS_PER_CALL + 1):
                attempts_used = attempt
                start = time.time()
                response_dict, raw_text, api_error = call_openai(
                    client=client,
                    prompt=prompt,
                    n_questions=int(len(questions)),
                )
                latency_seconds = time.time() - start

                if api_error is None:
                    break

                if attempt < MAX_ATTEMPTS_PER_CALL:
                    time.sleep(RETRY_SLEEP_SECONDS)

            usage = extract_usage(response_dict)
            input_tokens_details = usage.get("input_tokens_details") or {}
            output_tokens_details = usage.get("output_tokens_details") or {}

            model_returned = None
            response_status = None
            finish_reasons = []
            if response_dict:
                model_returned = response_dict.get("model")
                response_status = response_dict.get("status")
                for item in response_dict.get("output", []) or []:
                    if isinstance(item, dict) and item.get("finish_reason"):
                        finish_reasons.append(item.get("finish_reason"))

            data = None
            whole_response_parse_error = None
            answer_records: Dict[int, Dict[str, Any]] = {}

            if api_error is None:
                data, whole_response_parse_error = load_json_robust(raw_text)
                if data is not None:
                    answer_records, answer_records_error = extract_answer_records(data)
                    if answer_records_error:
                        whole_response_parse_error = (
                            whole_response_parse_error + "; " + answer_records_error
                            if whole_response_parse_error else answer_records_error
                        )

            call_record = {
                "run_name": run_name,
                "provider": PROVIDER,
                "api_endpoint": API_ENDPOINT,
                "model_requested": MODEL_REQUESTED,
                "model_returned": model_returned,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "replicate": replicate,
                "condition": "whole_text_answers_only",
                "prompt_hash_sha256": prompt_hash,
                "n_questions_in_prompt": int(len(questions)),
                "n_answer_records_returned": len(answer_records),
                "n_unique_answer_records_returned": len(set(answer_records.keys())),
                "whole_response_parse_error": whole_response_parse_error,
                "api_error": api_error,
                "attempts_used": attempts_used,
                "strict_json_schema": True,
                "rationales_requested": False,
                "reasoning_effort_sent": False,
                "latency_seconds": latency_seconds,
                "response_status": response_status,
                "finish_reasons": ",".join(finish_reasons),
                "usage": usage,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "reasoning_tokens": output_tokens_details.get("reasoning_tokens"),
                "cached_tokens": input_tokens_details.get("cached_tokens"),
                "raw_text": raw_text,
                "response": response_dict,
            }

            raw_f.write(json.dumps(call_record, ensure_ascii=False) + "\n")
            raw_f.flush()
            call_rows.append({k: v for k, v in call_record.items() if k not in {"usage", "raw_text", "response"}})

            for _, row in questions.iterrows():
                qn = int(row["original_question_number"])
                question_id = row["question_id"]
                correct_answer = get_correct_answer(answer_key, question_id)
                has_official_answer = correct_answer is not None

                rec = answer_records.get(qn)
                raw_answer = None
                model_answer = None
                parse_error = None
                raw_record = None

                if api_error is not None:
                    parse_error = "api_error_no_answer"
                elif whole_response_parse_error is not None and rec is None:
                    parse_error = "whole_response_parse_error_no_answer"
                elif rec is None:
                    parse_error = "missing_answer_for_question"
                else:
                    raw_answer = rec.get("raw_answer")
                    raw_record = rec.get("raw_record")
                    model_answer = normalize_answer(raw_answer)
                    if model_answer not in {"1", "2", "3", "4", "blank"}:
                        parse_error = f"invalid_answer_value: {raw_answer}"
                        model_answer = None

                is_correct, score, answer_status = score_answer(model_answer, correct_answer)

                result_record = {
                    "run_name": run_name,
                    "provider": PROVIDER,
                    "api_endpoint": API_ENDPOINT,
                    "model_requested": MODEL_REQUESTED,
                    "model_returned": model_returned,
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    "replicate": replicate,
                    "condition": "whole_text_answers_only",
                    "question_id": question_id,
                    "original_question_number": qn,
                    "is_reserve": bool(row.get("is_reserve", False)),
                    "has_image": bool(row.get("has_image", False)),
                    "model_answer": model_answer,
                    "raw_answer": raw_answer,
                    "correct_answer": correct_answer,
                    "has_official_answer": has_official_answer,
                    "is_correct": is_correct,
                    "score": score,
                    "answer_status": answer_status,
                    "parse_error": parse_error,
                    "api_error": api_error,
                    "response_status": response_status,
                    "finish_reasons": ",".join(finish_reasons),
                    "latency_seconds_call": latency_seconds,
                    "input_tokens_call": usage.get("input_tokens"),
                    "output_tokens_call": usage.get("output_tokens"),
                    "total_tokens_call": usage.get("total_tokens"),
                    "raw_record": json.dumps(raw_record, ensure_ascii=False) if raw_record is not None else None,
                }

                all_rows.append(result_record)

            evaluable_rows = [r for r in all_rows if r["replicate"] == replicate and r["has_official_answer"]]
            correct = sum(1 for r in evaluable_rows if r["answer_status"] == "correct")
            n_eval = len(evaluable_rows)
            missing = sum(1 for r in evaluable_rows if r["answer_status"] == "missing_or_invalid")
            blanks = sum(1 for r in evaluable_rows if r["answer_status"] == "blank")
            incorrect = sum(1 for r in evaluable_rows if r["answer_status"] == "incorrect")

            print(
                f"rep={replicate} status={'API_ERROR' if api_error else 'ok'} "
                f"answers_returned={len(answer_records)} response_status={response_status} "
                f"correct={correct}/{n_eval} incorrect={incorrect} blank={blanks} "
                f"missing_invalid={missing} latency={latency_seconds:.2f}s"
            )

    results = pd.DataFrame(all_rows)
    call_df = pd.DataFrame(call_rows)

    results.to_csv(scored_results_file, index=False, encoding="utf-8")
    call_df.to_csv(call_level_file, index=False, encoding="utf-8")

    summary = build_summary(results, call_df)
    summary.to_csv(summary_file, index=False, encoding="utf-8")

    metadata["completed_utc"] = utc_timestamp()
    metadata["outputs"] = {
        "raw_outputs_file": str(raw_outputs_file),
        "scored_results_file": str(scored_results_file),
        "summary_file": str(summary_file),
        "call_level_file": str(call_level_file),
        "prompt_log_file": str(prompt_log_file),
        "metadata_file": str(metadata_file),
    }

    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nDONE")
    print(f"Raw outputs: {raw_outputs_file}")
    print(f"Scored results: {scored_results_file}")
    print(f"Summary: {summary_file}")
    print(f"Call-level: {call_level_file}")
    print(f"Prompt log: {prompt_log_file}")
    print(f"Metadata: {metadata_file}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

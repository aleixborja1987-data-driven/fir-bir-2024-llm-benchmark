"""
BIR 2024 — Kimi K2.6 whole-exam / whole-text NON-THINKING benchmark
Answers-only, strict JSON Schema, 2 replicates.

Official Moonshot API / OpenAI-compatible Chat Completions.

Condition:
- The complete BIR 2024 exam is concatenated as text and sent in ONE API call
  per replicate.
- Answers only; no rationales requested.
- No PDF, images, files, tools, web search, retrieval, or answer key during
  inference.
- The answer key is merged locally only after each response has been received.

Prespecified Kimi K2.6 non-thinking settings:
- model: kimi-k2.6
- thinking: explicitly disabled
- temperature: 0.6 (provider-supported fixed value)
- top_p: 0.95 (provider-supported fixed value)
- stream: true
- strict JSON Schema
- correct +1, incorrect -1/3, blank/missing/invalid 0
- questions without an official answer excluded
- valid but incorrect answers are never retried

Reasoning audit:
- Any non-empty reasoning_content is considered substantive reasoning.
- More than 1 reported reasoning token is considered substantive reasoning.
- Exactly 1 reported reasoning token with empty reasoning_content is logged as
  provider-accounting residue and is accepted.

Expected files in the working directory:
- .env with MOONSHOT_API_KEY=...
- bir2024_questions_210.csv
- bir2024_answer_key_210.csv

Install:
    pip install --upgrade openai python-dotenv pandas truststore

Recommended full-structure pilot:
    python "07_run_bir2024_kimi_k2_6_whole_text_nonthinking_2reps.py" `
      --replicates 1

Definitive run:
    python "07_run_bir2024_kimi_k2_6_whole_text_nonthinking_2reps.py" `
      --replicates 2

Optional partial-format probe only:
    python "07_run_bir2024_kimi_k2_6_whole_text_nonthinking_2reps.py" `
      --limit 30 `
      --replicates 1

NOTE:
A 30-question probe validates API/schema mechanics, but the recommended pilot
uses all 210 questions once because output completeness and token limits are
specific to the full whole-exam condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import truststore
    truststore.inject_into_ssl()
    TRUSTSTORE_STATUS = "enabled"
except Exception as exc:
    TRUSTSTORE_STATUS = f"not_enabled: {exc!r}"

import pandas as pd
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)


# ---------------------------------------------------------------------
# Prespecified study configuration
# ---------------------------------------------------------------------

MODEL = "kimi-k2.6"
PROVIDER = "Moonshot AI / Kimi"
INTERFACE = "API"
BASE_URL_DEFAULT = "https://api.moonshot.ai/v1"
ENDPOINT = "OpenAI-compatible Chat Completions"

CONDITION = "whole_text_answers_only_nonthinking"
PROMPT_VERSION = "bir2024_whole_text_examlike_blank_strict_answers_only_v1"
SCRIPT_VERSION = "bir2024_kimi_k2_6_whole_text_nonthinking_2reps_v1"

THINKING = {"type": "disabled"}
TEMPERATURE = 0.6
TOP_P = 0.95
PRESENCE_PENALTY = 0.0
FREQUENCY_PENALTY = 0.0
N = 1

RESIDUAL_REASONING_TOKEN_ALLOWANCE = 1

QUESTIONS_FILE = Path("bir2024_questions_210.csv")
ANSWER_KEY_FILE = Path("bir2024_answer_key_210.csv")
OUTPUT_ROOT = Path("outputs")

VALID_ANSWERS = {"1", "2", "3", "4", "blank"}

DEFAULT_REPLICATES = 2
DEFAULT_MAX_TOKENS = 12000
DEFAULT_TIMEOUT_SECONDS = 1800.0
DEFAULT_MAX_TOTAL_ATTEMPTS = 7
DEFAULT_BASE_SLEEP_SECONDS = 10.0
DEFAULT_MAX_BACKOFF_SECONDS = 120.0
DEFAULT_BETWEEN_REPLICATES_SLEEP_SECONDS = 2.0


PROMPT_PREAMBLE = """You are taking the Spanish BIR examination for Biology residency access.

This is a multiple-choice exam. Each question has four options, numbered 1 to 4, and only one is correct.

Scoring rule:
- Correct answer: +1 point.
- Incorrect answer: -1/3 points.
- Blank answer: 0 points.

Act as a human candidate trying to maximize the final score. Answer from your own knowledge only. Do not use web search, external tools, files, images, retrieval, or any answer key.

You will receive the complete exam text in a single message.

Return only valid JSON matching the required schema:
{
  "answers": [
    {"question_number": 1, "answer": "1|2|3|4|blank"},
    {"question_number": 2, "answer": "1|2|3|4|blank"}
  ]
}

Output rules:
- Return exactly one answer object for every question included in the input.
- Use the integer question number shown in the input.
- Preserve the same question order as in the input.
- The answer must be exactly "1", "2", "3", "4", or "blank".
- Do not return multiple options.
- If you cannot choose one single best option, return "blank".
- Do not include rationales, explanations, comments, markdown, or extra fields.
"""


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


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

    aliases = {
        "": None,
        "nan": None,
        "none": None,
        "null": None,
        "n/a": None,
        "na": None,
        "blanco": "blank",
        "en blanco": "blank",
        "no answer": "blank",
    }
    return aliases.get(text, text)


def normalize_qnum(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)

    match = re.search(r"\d+", str(value).strip())
    return int(match.group(0)) if match else None


def to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "si", "sí"}:
        return True
    if text in {"false", "0", "no", "n", "", "nan", "none", "null"}:
        return False
    return default


def dump_object(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    try:
        return dict(obj)
    except Exception:
        return {"repr": repr(obj)}


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------
# Inputs, prompt and schema
# ---------------------------------------------------------------------

def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing required file: {QUESTIONS_FILE}")
    if not ANSWER_KEY_FILE.exists():
        raise FileNotFoundError(f"Missing required file: {ANSWER_KEY_FILE}")

    questions = pd.read_csv(QUESTIONS_FILE)
    key = pd.read_csv(ANSWER_KEY_FILE)

    required_q = {
        "question_id",
        "original_question_number",
        "question_text",
        "option_1",
        "option_2",
        "option_3",
        "option_4",
    }
    required_k = {"question_id", "correct_answer"}

    missing_q = required_q - set(questions.columns)
    missing_k = required_k - set(key.columns)
    if missing_q:
        raise ValueError(f"Missing question columns: {sorted(missing_q)}")
    if missing_k:
        raise ValueError(f"Missing answer-key columns: {sorted(missing_k)}")

    if len(questions) != 210:
        raise ValueError(f"Expected 210 questions, found {len(questions)}")
    if len(key) != 210:
        raise ValueError(f"Expected 210 answer-key rows, found {len(key)}")

    questions = questions.sort_values("original_question_number").reset_index(drop=True)
    return questions, key


def select_questions(questions: pd.DataFrame, limit: Optional[int]) -> pd.DataFrame:
    out = questions.copy()
    if limit is not None:
        if limit < 1 or limit > len(out):
            raise ValueError(f"--limit must be between 1 and {len(out)}")
        out = out.head(limit)
    return out.reset_index(drop=True)


def format_question(row: pd.Series) -> str:
    qnum = int(row["original_question_number"])
    return f"""Question {qnum}:
{safe_str(row["question_text"]).strip()}

Options:
1. {safe_str(row["option_1"]).strip()}
2. {safe_str(row["option_2"]).strip()}
3. {safe_str(row["option_3"]).strip()}
4. {safe_str(row["option_4"]).strip()}"""


def build_whole_exam_prompt(questions: pd.DataFrame) -> str:
    blocks = [format_question(row) for _, row in questions.iterrows()]
    return (
        PROMPT_PREAMBLE
        + "\n\nFULL EXAM TEXT:\n\n"
        + "\n\n---\n\n".join(blocks)
    )


def build_answer_schema(n_questions: int) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "bir2024_whole_exam_answers",
            "strict": True,
            "schema": {
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
            },
        },
    }


def build_key_map(key: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}

    for _, row in key.iterrows():
        qid = safe_str(row["question_id"]).strip()
        answer = normalize_answer(row.get("correct_answer"))

        available = True
        if "official_answer_available" in key.columns:
            available = to_bool(row.get("official_answer_available"), True)
        if "is_annulled_or_without_official_answer" in key.columns:
            if to_bool(row.get("is_annulled_or_without_official_answer"), False):
                available = False
        if answer not in {"1", "2", "3", "4"}:
            available = False

        mapping[qid] = {
            "correct_answer": answer,
            "official_answer_available": available,
        }

    return mapping


# ---------------------------------------------------------------------
# Streaming call and reasoning audit
# ---------------------------------------------------------------------

def extract_usage_fields(usage: Dict[str, Any]) -> Dict[str, Any]:
    completion_details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}

    reasoning_tokens = completion_details.get("reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = usage.get("reasoning_tokens")

    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": prompt_details.get("cached_tokens") or usage.get("cached_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def call_once(
    client: OpenAI,
    prompt: str,
    answer_schema: Dict[str, Any],
    max_tokens: int,
) -> Dict[str, Any]:
    started = time.perf_counter()

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROMPT_PREAMBLE},
            {
                "role": "user",
                "content": prompt.removeprefix(PROMPT_PREAMBLE).lstrip(),
            },
        ],
        response_format=answer_schema,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        n=N,
        presence_penalty=PRESENCE_PENALTY,
        frequency_penalty=FREQUENCY_PENALTY,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"thinking": THINKING},
        # No tools, web search, files, images, or retrieval.
    )

    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    model_returned: Optional[str] = None
    response_id: Optional[str] = None
    finish_reason: Optional[str] = None
    usage_dict: Dict[str, Any] = {}
    n_chunks = 0

    for chunk in stream:
        n_chunks += 1
        chunk_dict = dump_object(chunk)

        if response_id is None:
            response_id = chunk_dict.get("id")
        if model_returned is None:
            model_returned = chunk_dict.get("model")

        if chunk_dict.get("usage"):
            usage_dict = chunk_dict.get("usage") or {}

        choices = chunk_dict.get("choices") or []
        for choice in choices:
            if choice.get("finish_reason") is not None:
                finish_reason = choice.get("finish_reason")

            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                content_parts.append(str(content))

            reasoning = (
                delta.get("reasoning_content")
                or delta.get("reasoning")
                or delta.get("analysis")
            )
            if reasoning:
                reasoning_parts.append(str(reasoning))

    elapsed = time.perf_counter() - started
    raw_content = "".join(content_parts)
    reasoning_content = "".join(reasoning_parts)
    usage_fields = extract_usage_fields(usage_dict)

    reasoning_tokens_raw = usage_fields.get("reasoning_tokens")
    try:
        reasoning_tokens_num = (
            float(reasoning_tokens_raw)
            if reasoning_tokens_raw is not None
            else 0.0
        )
    except (TypeError, ValueError):
        reasoning_tokens_num = 0.0

    reasoning_content_nonempty = bool(reasoning_content.strip())
    substantive_reasoning = (
        reasoning_content_nonempty
        or reasoning_tokens_num > RESIDUAL_REASONING_TOKEN_ALLOWANCE
    )
    residual_reasoning_token_only = (
        not reasoning_content_nonempty
        and reasoning_tokens_num == RESIDUAL_REASONING_TOKEN_ALLOWANCE
    )

    return {
        "response_id": response_id,
        "model_returned": model_returned,
        "finish_reason": finish_reason,
        "raw_content": raw_content,
        "reasoning_content": reasoning_content,
        "reasoning_content_nonempty": reasoning_content_nonempty,
        "substantive_reasoning": substantive_reasoning,
        "residual_reasoning_token_only": residual_reasoning_token_only,
        "elapsed_seconds": elapsed,
        "n_stream_chunks": n_chunks,
        "usage_raw": usage_dict,
        **usage_fields,
    }


# ---------------------------------------------------------------------
# Validation and scoring
# ---------------------------------------------------------------------

def parse_and_validate_output(
    raw_content: Optional[str],
    expected_question_numbers: Sequence[int],
) -> Tuple[Dict[int, str], Optional[str], Optional[Dict[str, Any]]]:
    if not raw_content or not raw_content.strip():
        return {}, "missing_output", None

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        return {}, f"json_decode_error: {exc}", None

    if not isinstance(parsed, dict):
        return {}, "top_level_not_object", parsed

    answers = parsed.get("answers")
    if not isinstance(answers, list):
        return {}, "answers_missing_or_not_array", parsed

    expected = list(expected_question_numbers)
    expected_set = set(expected)

    if len(answers) != len(expected):
        return {}, f"wrong_answer_count:{len(answers)}_expected:{len(expected)}", parsed

    records: Dict[int, str] = {}
    duplicates: List[int] = []
    invalid_items: List[str] = []

    for idx, item in enumerate(answers):
        if not isinstance(item, dict):
            invalid_items.append(f"index_{idx}:not_object")
            continue

        if set(item.keys()) != {"question_number", "answer"}:
            invalid_items.append(f"index_{idx}:wrong_fields")
            continue

        qnum = normalize_qnum(item.get("question_number"))
        answer = normalize_answer(item.get("answer"))

        if qnum is None:
            invalid_items.append(f"index_{idx}:invalid_question_number")
            continue
        if qnum in records:
            duplicates.append(qnum)
            continue
        if answer not in VALID_ANSWERS:
            invalid_items.append(f"q{qnum}:invalid_answer:{answer!r}")
            continue

        records[qnum] = answer

    if invalid_items:
        return {}, "invalid_items:" + "|".join(invalid_items[:10]), parsed
    if duplicates:
        return {}, f"duplicate_question_numbers:{duplicates}", parsed

    found_set = set(records)
    missing = sorted(expected_set - found_set)
    unexpected = sorted(found_set - expected_set)
    if missing or unexpected:
        return {}, f"question_set_mismatch:missing={missing};unexpected={unexpected}", parsed

    output_order = [normalize_qnum(item["question_number"]) for item in answers]
    if output_order != expected:
        return {}, "question_order_mismatch", parsed

    return records, None, parsed


def score_answer(
    model_answer: Optional[str],
    correct_answer: Optional[str],
    official_available: bool,
) -> Tuple[Optional[bool], Optional[float], str]:
    if not official_available:
        return None, None, "not_evaluable"
    if model_answer is None or model_answer not in VALID_ANSWERS:
        return False, 0.0, "missing_or_invalid"
    if model_answer == "blank":
        return False, 0.0, "blank"
    if model_answer == correct_answer:
        return True, 1.0, "correct"
    return False, -1.0 / 3.0, "incorrect"


def is_transient_exception(exc: Exception) -> bool:
    if isinstance(
        exc,
        (APIConnectionError, APITimeoutError, RateLimitError),
    ):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        return status in {408, 409, 429, 500, 502, 503, 504}
    return False


def backoff_seconds(
    attempt: int,
    base_sleep: float,
    max_backoff: float,
) -> float:
    delay = min(max_backoff, base_sleep * (2 ** max(0, attempt - 1)))
    return delay + random.uniform(0, min(2.0, delay * 0.1))


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def build_summary(results: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for replicate, group in results.groupby("replicate", sort=True):
        evaluable = group[group["official_answer_available"] == True].copy()
        call = calls[calls["replicate"] == replicate]
        call_row = call.iloc[-1].to_dict() if len(call) else {}

        n_eval = len(evaluable)
        n_correct = int((evaluable["answer_status"] == "correct").sum())
        n_incorrect = int((evaluable["answer_status"] == "incorrect").sum())
        n_blank = int((evaluable["answer_status"] == "blank").sum())
        n_missing = int((evaluable["answer_status"] == "missing_or_invalid").sum())
        score = float(evaluable["score"].sum(skipna=True))

        rows.append({
            "replicate": int(replicate),
            "n_questions_total": int(len(group)),
            "n_evaluable": int(n_eval),
            "n_not_evaluable": int((group["answer_status"] == "not_evaluable").sum()),
            "correct": n_correct,
            "incorrect": n_incorrect,
            "blank": n_blank,
            "missing_or_invalid": n_missing,
            "accuracy_evaluable": (n_correct / n_eval) if n_eval else None,
            "score": score,
            "max_score": n_eval,
            "score_fraction": (score / n_eval) if n_eval else None,
            "call_success": call_row.get("call_success"),
            "attempts_used": call_row.get("attempts_used"),
            "finish_reason": call_row.get("finish_reason"),
            "elapsed_seconds": call_row.get("elapsed_seconds"),
            "prompt_tokens": call_row.get("prompt_tokens"),
            "completion_tokens": call_row.get("completion_tokens"),
            "reasoning_tokens": call_row.get("reasoning_tokens"),
            "total_tokens": call_row.get("total_tokens"),
            "substantive_reasoning": call_row.get("substantive_reasoning"),
            "residual_reasoning_token_only": call_row.get("residual_reasoning_token_only"),
        })

    return pd.DataFrame(rows)


def build_discordance_table(results: pd.DataFrame) -> pd.DataFrame:
    if results["replicate"].nunique() < 2:
        return pd.DataFrame()

    pivot = results.pivot_table(
        index=[
            "question_id",
            "original_question_number",
            "correct_answer",
            "official_answer_available",
        ],
        columns="replicate",
        values=["model_answer", "answer_status", "score"],
        aggfunc="first",
    )
    pivot.columns = [
        f"{metric}_rep{replicate}" for metric, replicate in pivot.columns
    ]
    pivot = pivot.reset_index()

    answer_cols = sorted(c for c in pivot.columns if c.startswith("model_answer_rep"))
    if len(answer_cols) >= 2:
        pivot["same_answer"] = pivot[answer_cols].nunique(axis=1, dropna=False) == 1
        pivot = pivot[pivot["same_answer"] == False].copy()

    return pivot.sort_values("original_question_number").reset_index(drop=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--max-total-attempts",
        type=int,
        default=DEFAULT_MAX_TOTAL_ATTEMPTS,
    )
    parser.add_argument(
        "--base-sleep",
        type=float,
        default=DEFAULT_BASE_SLEEP_SECONDS,
    )
    parser.add_argument(
        "--max-backoff",
        type=float,
        default=DEFAULT_MAX_BACKOFF_SECONDS,
    )
    parser.add_argument(
        "--between-replicates-sleep",
        type=float,
        default=DEFAULT_BETWEEN_REPLICATES_SLEEP_SECONDS,
    )
    args = parser.parse_args()

    if args.replicates < 1:
        raise ValueError("--replicates must be >= 1")
    if args.max_tokens < 1000:
        raise ValueError("--max-tokens should be >= 1000")
    if args.max_total_attempts < 1:
        raise ValueError("--max-total-attempts must be >= 1")

    load_dotenv()
    api_key = os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")
    base_url = os.getenv("KIMI_BASE_URL", BASE_URL_DEFAULT)

    if not api_key:
        raise RuntimeError(
            "Missing MOONSHOT_API_KEY (or KIMI_API_KEY) in environment/.env"
        )

    questions_all, key = load_inputs()
    questions = select_questions(questions_all, args.limit)
    key_map = build_key_map(key)

    expected_qnums = [
        int(x) for x in questions["original_question_number"].tolist()
    ]
    answer_schema = build_answer_schema(len(questions))
    full_prompt = build_whole_exam_prompt(questions)

    timestamp = utc_compact()
    suffix = f"_limit{args.limit}" if args.limit is not None else ""
    run_id = (
        f"bir2024_kimi_k2_6_whole_text_nonthinking_"
        f"{args.replicates}r_STRICT{suffix}_{timestamp}"
    )
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    raw_file = output_dir / f"{run_id}_raw_results.jsonl"
    results_file = output_dir / f"{run_id}_results.csv"
    summary_file = output_dir / f"{run_id}_summary.csv"
    discordances_file = output_dir / f"{run_id}_discordances.csv"
    metadata_file = output_dir / f"{run_id}_metadata.json"
    prompt_file = output_dir / f"{run_id}_prompt_preamble.txt"
    full_prompt_file = output_dir / f"{run_id}_full_prompt.txt"

    prompt_file.write_text(PROMPT_PREAMBLE, encoding="utf-8")
    full_prompt_file.write_text(full_prompt, encoding="utf-8")

    metadata: Dict[str, Any] = {
        "run_id": run_id,
        "created_utc": timestamp,
        "provider": PROVIDER,
        "interface": INTERFACE,
        "endpoint": ENDPOINT,
        "base_url": base_url,
        "model_requested": MODEL,
        "condition": CONDITION,
        "prompt_template_version": PROMPT_VERSION,
        "script_version": SCRIPT_VERSION,
        "truststore_status": TRUSTSTORE_STATUS,
        "questions_file": str(QUESTIONS_FILE),
        "answer_key_file": str(ANSWER_KEY_FILE),
        "questions_file_sha256": sha256_file(QUESTIONS_FILE),
        "answer_key_file_sha256": sha256_file(ANSWER_KEY_FILE),
        "n_questions_in_file": int(len(questions_all)),
        "n_questions_selected": int(len(questions)),
        "is_full_definitive_selection": bool(
            len(questions) == 210 and args.limit is None
        ),
        "limit": args.limit,
        "replicates": args.replicates,
        "thinking": {
            "requested": THINKING,
            "policy": "explicitly_disabled",
            "reasoning_audit_rule": {
                "abort_if_reasoning_content_nonempty": True,
                "abort_if_reasoning_tokens_above": RESIDUAL_REASONING_TOKEN_ALLOWANCE,
                "residual_reasoning_token_allowance": RESIDUAL_REASONING_TOKEN_ALLOWANCE,
                "interpretation": (
                    "Exactly 1 reported reasoning token with empty "
                    "reasoning_content is logged as provider-accounting residue."
                ),
            },
        },
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "sampling_parameter_note": (
            "Provider-constrained settings for Kimi K2.6 non-thinking mode. "
            "temperature=0 and top_p=1 are unsupported."
        ),
        "n": N,
        "presence_penalty": PRESENCE_PENALTY,
        "frequency_penalty": FREQUENCY_PENALTY,
        "max_tokens": args.max_tokens,
        "response_format": "json_schema_strict",
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": "not_provided",
        "web_search": "not_provided",
        "file_input": False,
        "image_input": False,
        "rationales_requested": False,
        "request_timeout_seconds": args.timeout,
        "retry_policy": {
            "max_total_attempts": args.max_total_attempts,
            "base_sleep_seconds": args.base_sleep,
            "max_backoff_seconds": args.max_backoff,
            "retry_transient_api_failures": True,
            "retry_invalid_or_incomplete_structured_outputs": True,
            "retry_unexpected_finish_reason": True,
            "retry_valid_but_incorrect_answers": False,
        },
        "between_replicates_sleep_seconds": args.between_replicates_sleep,
        "scoring": {
            "correct": 1,
            "incorrect": -1 / 3,
            "blank": 0,
            "missing_or_invalid": 0,
            "questions_without_official_answer_excluded": True,
        },
        "prompt_preamble": PROMPT_PREAMBLE,
        "prompt_preamble_sha256": sha256_text(PROMPT_PREAMBLE),
        "full_prompt_sha256": sha256_text(full_prompt),
        "answer_schema": answer_schema,
        "output_files": {
            "metadata": str(metadata_file),
            "raw_results": str(raw_file),
            "results": str(results_file),
            "summary": str(summary_file),
            "discordances": str(discordances_file),
            "prompt_preamble": str(prompt_file),
            "full_prompt": str(full_prompt_file),
        },
    }
    metadata_file.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=args.timeout,
        max_retries=0,  # Retries are handled and logged by this script.
    )

    call_rows: List[Dict[str, Any]] = []
    result_rows: List[Dict[str, Any]] = []
    protocol_violation = False

    print(f"Run ID: {run_id}")
    print(f"Questions per call: {len(questions)}")
    print(f"Replicates: {args.replicates}")
    print(f"Output: {output_dir}")
    print()

    for replicate in range(1, args.replicates + 1):
        print(f"[Replicate {replicate}/{args.replicates}] starting...")
        accepted_call: Optional[Dict[str, Any]] = None
        accepted_answers: Dict[int, str] = {}
        attempts_log: List[Dict[str, Any]] = []

        for attempt in range(1, args.max_total_attempts + 1):
            attempt_started = utc_iso()
            try:
                call = call_once(
                    client=client,
                    prompt=full_prompt,
                    answer_schema=answer_schema,
                    max_tokens=args.max_tokens,
                )

                validation_answers, validation_error, parsed = (
                    parse_and_validate_output(
                        call.get("raw_content"),
                        expected_qnums,
                    )
                )

                finish_ok = call.get("finish_reason") in {"stop", None}
                substantive_reasoning = bool(call.get("substantive_reasoning"))

                attempt_record = {
                    "run_id": run_id,
                    "replicate": replicate,
                    "attempt": attempt,
                    "attempt_started_utc": attempt_started,
                    "attempt_completed_utc": utc_iso(),
                    "request_succeeded": True,
                    "validation_error": validation_error,
                    "finish_reason_ok": finish_ok,
                    "protocol_violation": substantive_reasoning,
                    **call,
                    "parsed_output": parsed,
                    "exception_type": None,
                    "exception_message": None,
                    "traceback": None,
                }
                append_jsonl(raw_file, attempt_record)
                attempts_log.append(attempt_record)

                if substantive_reasoning:
                    protocol_violation = True
                    print(
                        "  Protocol violation: substantive reasoning detected. "
                        "Aborting run."
                    )
                    raise RuntimeError(
                        "Substantive reasoning detected in a non-thinking run."
                    )

                if validation_error is None and finish_ok:
                    accepted_call = call
                    accepted_answers = validation_answers
                    print(
                        f"  Accepted on attempt {attempt}; "
                        f"{len(accepted_answers)} answers; "
                        f"finish_reason={call.get('finish_reason')}; "
                        f"reasoning_tokens={call.get('reasoning_tokens')}."
                    )
                    break

                reason = validation_error or (
                    f"unexpected_finish_reason:{call.get('finish_reason')}"
                )
                print(f"  Attempt {attempt} invalid: {reason}")

                if attempt < args.max_total_attempts:
                    sleep_for = backoff_seconds(
                        attempt,
                        args.base_sleep,
                        args.max_backoff,
                    )
                    print(f"  Retrying in {sleep_for:.1f}s...")
                    time.sleep(sleep_for)

            except RuntimeError as exc:
                if "Substantive reasoning detected" in str(exc):
                    raise
                raise
            except Exception as exc:
                transient = is_transient_exception(exc)
                attempt_record = {
                    "run_id": run_id,
                    "replicate": replicate,
                    "attempt": attempt,
                    "attempt_started_utc": attempt_started,
                    "attempt_completed_utc": utc_iso(),
                    "request_succeeded": False,
                    "validation_error": None,
                    "finish_reason_ok": False,
                    "protocol_violation": False,
                    "response_id": None,
                    "model_returned": None,
                    "finish_reason": None,
                    "raw_content": None,
                    "reasoning_content": None,
                    "reasoning_content_nonempty": None,
                    "substantive_reasoning": None,
                    "residual_reasoning_token_only": None,
                    "elapsed_seconds": None,
                    "n_stream_chunks": None,
                    "usage_raw": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "reasoning_tokens": None,
                    "cached_tokens": None,
                    "total_tokens": None,
                    "parsed_output": None,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "transient_exception": transient,
                }
                append_jsonl(raw_file, attempt_record)
                attempts_log.append(attempt_record)

                print(
                    f"  Attempt {attempt} exception: "
                    f"{type(exc).__name__}: {exc}"
                )

                if not transient or attempt >= args.max_total_attempts:
                    raise

                sleep_for = backoff_seconds(
                    attempt,
                    args.base_sleep,
                    args.max_backoff,
                )
                print(f"  Retrying in {sleep_for:.1f}s...")
                time.sleep(sleep_for)

        if accepted_call is None:
            raise RuntimeError(
                f"Replicate {replicate} failed after "
                f"{args.max_total_attempts} attempts."
            )

        call_rows.append({
            "run_id": run_id,
            "replicate": replicate,
            "call_success": True,
            "attempts_used": len(attempts_log),
            "response_id": accepted_call.get("response_id"),
            "model_returned": accepted_call.get("model_returned"),
            "finish_reason": accepted_call.get("finish_reason"),
            "elapsed_seconds": accepted_call.get("elapsed_seconds"),
            "n_stream_chunks": accepted_call.get("n_stream_chunks"),
            "prompt_tokens": accepted_call.get("prompt_tokens"),
            "completion_tokens": accepted_call.get("completion_tokens"),
            "reasoning_tokens": accepted_call.get("reasoning_tokens"),
            "cached_tokens": accepted_call.get("cached_tokens"),
            "total_tokens": accepted_call.get("total_tokens"),
            "reasoning_content_nonempty": accepted_call.get(
                "reasoning_content_nonempty"
            ),
            "substantive_reasoning": accepted_call.get(
                "substantive_reasoning"
            ),
            "residual_reasoning_token_only": accepted_call.get(
                "residual_reasoning_token_only"
            ),
        })

        for _, qrow in questions.iterrows():
            qid = safe_str(qrow["question_id"]).strip()
            qnum = int(qrow["original_question_number"])
            model_answer = accepted_answers.get(qnum)

            key_info = key_map.get(
                qid,
                {
                    "correct_answer": None,
                    "official_answer_available": False,
                },
            )
            correct_answer = key_info["correct_answer"]
            official_available = bool(
                key_info["official_answer_available"]
            )

            is_correct, score, status = score_answer(
                model_answer=model_answer,
                correct_answer=correct_answer,
                official_available=official_available,
            )

            result_rows.append({
                "run_id": run_id,
                "provider": PROVIDER,
                "model_requested": MODEL,
                "model_returned": accepted_call.get("model_returned"),
                "condition": CONDITION,
                "replicate": replicate,
                "question_id": qid,
                "original_question_number": qnum,
                "question_text": safe_str(qrow["question_text"]),
                "option_1": safe_str(qrow["option_1"]),
                "option_2": safe_str(qrow["option_2"]),
                "option_3": safe_str(qrow["option_3"]),
                "option_4": safe_str(qrow["option_4"]),
                "model_answer": model_answer,
                "correct_answer": correct_answer,
                "official_answer_available": official_available,
                "is_correct": is_correct,
                "answer_status": status,
                "score": score,
                "response_id": accepted_call.get("response_id"),
                "finish_reason": accepted_call.get("finish_reason"),
                "attempts_used": len(attempts_log),
                "prompt_tokens_call": accepted_call.get("prompt_tokens"),
                "completion_tokens_call": accepted_call.get("completion_tokens"),
                "reasoning_tokens_call": accepted_call.get("reasoning_tokens"),
                "total_tokens_call": accepted_call.get("total_tokens"),
                "elapsed_seconds_call": accepted_call.get("elapsed_seconds"),
            })

        pd.DataFrame(result_rows).to_csv(
            results_file,
            index=False,
            encoding="utf-8-sig",
        )

        if replicate < args.replicates:
            time.sleep(args.between_replicates_sleep)

    results_df = pd.DataFrame(result_rows)
    calls_df = pd.DataFrame(call_rows)
    summary_df = build_summary(results_df, calls_df)
    discordances_df = build_discordance_table(results_df)

    results_df.to_csv(results_file, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
    discordances_df.to_csv(
        discordances_file,
        index=False,
        encoding="utf-8-sig",
    )

    reasoning_token_values = [
        row.get("reasoning_tokens")
        for row in call_rows
        if row.get("reasoning_tokens") is not None
    ]
    reasoning_distribution: Dict[str, int] = {}
    for value in reasoning_token_values:
        key_value = str(value)
        reasoning_distribution[key_value] = (
            reasoning_distribution.get(key_value, 0) + 1
        )

    metadata.update({
        "completed_utc": utc_compact(),
        "definitive_run_completed_as_prespecified": bool(
            not protocol_violation
            and args.replicates == 2
            and len(questions) == 210
            and args.limit is None
            and len(call_rows) == 2
        ),
        "observed_reasoning_audit": {
            "n_calls": len(call_rows),
            "n_calls_with_reasoning_content": sum(
                bool(row.get("reasoning_content_nonempty"))
                for row in call_rows
            ),
            "n_calls_with_any_reasoning_tokens": sum(
                (float(row.get("reasoning_tokens") or 0) > 0)
                for row in call_rows
            ),
            "n_calls_with_residual_reasoning_token_only": sum(
                bool(row.get("residual_reasoning_token_only"))
                for row in call_rows
            ),
            "n_calls_with_substantive_reasoning": sum(
                bool(row.get("substantive_reasoning"))
                for row in call_rows
            ),
            "n_protocol_violations": int(protocol_violation),
            "total_reasoning_tokens": sum(
                float(row.get("reasoning_tokens") or 0)
                for row in call_rows
            ),
            "reasoning_token_distribution": reasoning_distribution,
            "residual_reasoning_token_allowance": (
                RESIDUAL_REASONING_TOKEN_ALLOWANCE
            ),
            "run_acceptance_criterion": (
                "All accepted calls must have empty reasoning_content and "
                "no call may report more than the residual-token allowance."
            ),
        },
        "call_level_summary": call_rows,
    })

    metadata_file.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print()
    print("Completed successfully.")
    print(summary_df.to_string(index=False))
    print()
    print(f"Results:       {results_file}")
    print(f"Summary:       {summary_file}")
    print(f"Discordances:  {discordances_file}")
    print(f"Raw results:   {raw_file}")
    print(f"Metadata:      {metadata_file}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nFATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)

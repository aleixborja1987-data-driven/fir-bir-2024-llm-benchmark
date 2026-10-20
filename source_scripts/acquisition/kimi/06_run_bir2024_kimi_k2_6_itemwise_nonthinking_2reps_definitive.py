"""
BIR 2024 — Kimi K2.6 item-wise NON-THINKING definitive benchmark (2 replicates)

Official Moonshot API / OpenAI-compatible Chat Completions.

Prespecified settings:
- model: kimi-k2.6
- thinking: explicitly disabled
- temperature: 0.6 (provider-fixed in non-thinking mode)
- top_p: 0.95 (provider-fixed)
- stream: true
- strict JSON Schema
- no tools, web, files, images, retrieval, or answer key during inference
- correct +1, incorrect -1/3, blank/missing/invalid 0
- questions without an official answer excluded
- valid but incorrect answers are never retried

Kimi K2.6 does not accept temperature=0 or top_p=1. The supported
non-thinking values 0.6 and 0.95 are sent explicitly and logged.

Expected files:
- .env with MOONSHOT_API_KEY=...
- bir2024_questions_210.csv
- bir2024_answer_key_210.csv

Install:
    pip install --upgrade openai python-dotenv pandas truststore

Definitive run:
    python "06_run_bir2024_kimi_k2_6_itemwise_nonthinking_2reps_definitive.py"

Equivalent explicit command:
    python "06_run_bir2024_kimi_k2_6_itemwise_nonthinking_2reps_definitive.py" `
      --replicates 2

Definitive:
    python "06_run_bir2024_kimi_k2_6_itemwise_nonthinking_2reps_definitive.py" `
      --replicates 2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
BASE_URL_DEFAULT = "https://api.moonshot.ai/v1"
ENDPOINT = "OpenAI-compatible Chat Completions"

CONDITION = "itemwise_text_nonthinking"
PROMPT_VERSION = "bir2024_text_itemwise_examlike_blank_strict_v1"
SCRIPT_VERSION = "bir2024_kimi_k2_6_itemwise_nonthinking_2reps_definitive_v1"

# Provider-constrained values for Kimi K2.6 non-thinking mode.
THINKING = {"type": "disabled"}
TEMPERATURE = 0.6
TOP_P = 0.95
PRESENCE_PENALTY = 0.0
FREQUENCY_PENALTY = 0.0
N = 1

# Definitive-run audit rule:
# - Any non-empty reasoning_content is substantive reasoning.
# - More than 1 reported reasoning token is substantive reasoning.
# - Exactly 1 token with empty reasoning_content is logged as a residual
#   provider-accounting token and does not abort the pilot.
RESIDUAL_REASONING_TOKEN_ALLOWANCE = 1

QUESTIONS_FILE = Path("bir2024_questions_210.csv")
ANSWER_KEY_FILE = Path("bir2024_answer_key_210.csv")
OUTPUT_ROOT = Path("outputs")

VALID_ANSWERS = {"1", "2", "3", "4", "blank"}

SYSTEM_PROMPT = """You are taking the Spanish BIR examination for Biology residency access.

This is a multiple-choice exam. Each question has four options, numbered 1 to 4, and only one is correct.

Scoring rule:
- Correct answer: +1 point.
- Incorrect answer: -1/3 points.
- Blank answer: 0 points.

Act as a human candidate trying to maximize the final score. Answer from your own knowledge only. Do not use web search, external tools, files, images, or any answer key.

Return only valid JSON matching the required schema:
{"answer":"1|2|3|4|blank","rationale":"brief rationale, max 160 characters"}
"""

ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "bir2024_mcq_answer",
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
                    "maxLength": 160,
                },
            },
            "required": ["answer", "rationale"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------
# Helpers
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
        "blanco": "blank",
        "en blanco": "blank",
    }
    return aliases.get(text, text)


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
    return str(value).strip().lower() in {"true", "1", "yes", "y", "sí", "si"}


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
# Inputs
# ---------------------------------------------------------------------

def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing: {QUESTIONS_FILE}")
    if not ANSWER_KEY_FILE.exists():
        raise FileNotFoundError(f"Missing: {ANSWER_KEY_FILE}")

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

    if required_q - set(questions.columns):
        raise ValueError(f"Missing question columns: {sorted(required_q - set(questions.columns))}")
    if required_k - set(key.columns):
        raise ValueError(f"Missing answer-key columns: {sorted(required_k - set(key.columns))}")
    if len(questions) != 210:
        raise ValueError(f"Expected 210 questions, found {len(questions)}")
    if len(key) != 210:
        raise ValueError(f"Expected 210 answer-key rows, found {len(key)}")

    questions = questions.sort_values("original_question_number").reset_index(drop=True)
    return questions, key


def select_questions(
    questions: pd.DataFrame,
    limit: Optional[int],
    start_q: Optional[int],
    end_q: Optional[int],
) -> pd.DataFrame:
    out = questions.copy()
    if start_q is not None:
        out = out[out["original_question_number"].astype(int) >= start_q]
    if end_q is not None:
        out = out[out["original_question_number"].astype(int) <= end_q]
    out = out.sort_values("original_question_number").reset_index(drop=True)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        out = out.head(limit)
    if out.empty:
        raise ValueError("No questions selected")
    return out


def build_key_map(key: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
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

        output[qid] = {
            "correct_answer": answer,
            "official_answer_available": available,
        }
    return output


def build_user_prompt(row: pd.Series) -> str:
    return f"""Question {int(row['original_question_number'])}:
{safe_str(row['question_text']).strip()}

Options:
1. {safe_str(row['option_1']).strip()}
2. {safe_str(row['option_2']).strip()}
3. {safe_str(row['option_3']).strip()}
4. {safe_str(row['option_4']).strip()}
"""


# ---------------------------------------------------------------------
# Streaming API call
# ---------------------------------------------------------------------

def usage_fields(usage: Dict[str, Any]) -> Dict[str, Any]:
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


def call_once(client: OpenAI, prompt: str, max_tokens: int) -> Dict[str, Any]:
    started = time.perf_counter()

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format=ANSWER_SCHEMA,
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
    chunk_dumps: List[Dict[str, Any]] = []
    model_returned = None
    response_id = None
    finish_reason = None
    usage: Dict[str, Any] = {}

    for chunk in stream:
        chunk_dumps.append(dump_object(chunk))
        model_returned = model_returned or getattr(chunk, "model", None)
        response_id = response_id or getattr(chunk, "id", None)

        if getattr(chunk, "usage", None) is not None:
            usage = dump_object(chunk.usage)

        for choice in (getattr(chunk, "choices", None) or []):
            if getattr(choice, "finish_reason", None) is not None:
                finish_reason = choice.finish_reason

            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            if getattr(delta, "content", None):
                content_parts.append(delta.content)

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)

    fields = usage_fields(usage)
    reasoning_text = "".join(reasoning_parts)

    reasoning_tokens_raw = fields.get("reasoning_tokens")
    try:
        reasoning_tokens_numeric = (
            float(reasoning_tokens_raw)
            if reasoning_tokens_raw is not None
            else 0.0
        )
    except (TypeError, ValueError):
        reasoning_tokens_numeric = float("inf")

    reasoning_content_observed = bool(reasoning_text.strip())
    any_reasoning_tokens_reported = (
        reasoning_tokens_raw is not None
        and reasoning_tokens_numeric > 0
    )
    residual_reasoning_token_only = (
        not reasoning_content_observed
        and 0 < reasoning_tokens_numeric
        <= RESIDUAL_REASONING_TOKEN_ALLOWANCE
    )
    substantive_reasoning_observed = (
        reasoning_content_observed
        or reasoning_tokens_numeric
        > RESIDUAL_REASONING_TOKEN_ALLOWANCE
    )

    return {
        "model_returned": model_returned,
        "response_id": response_id,
        "finish_reason": finish_reason,
        "raw_content": "".join(content_parts),
        "reasoning_content": reasoning_text or None,
        "reasoning_content_observed": reasoning_content_observed,
        "any_reasoning_tokens_reported": any_reasoning_tokens_reported,
        "residual_reasoning_token_only": residual_reasoning_token_only,
        "substantive_reasoning_observed": substantive_reasoning_observed,
        # Backward-compatible name: now means substantive reasoning only.
        "reasoning_observed": substantive_reasoning_observed,
        "reasoning_tokens_numeric": (
            None if reasoning_tokens_raw is None else reasoning_tokens_numeric
        ),
        "reasoning_audit_class": (
            "substantive_reasoning"
            if substantive_reasoning_observed
            else (
                "residual_accounting_token_only"
                if residual_reasoning_token_only
                else "no_reasoning_reported"
            )
        ),
        "usage": usage,
        "n_stream_chunks": len(chunk_dumps),
        "chunk_dumps": chunk_dumps,
        "attempt_latency_seconds": time.perf_counter() - started,
        **fields,
    }


# ---------------------------------------------------------------------
# Parsing and scoring
# ---------------------------------------------------------------------

def parse_output(
    raw_content: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    if not raw_content or not raw_content.strip():
        return None, None, "missing_raw_content", None

    try:
        data = json.loads(raw_content)
    except Exception as exc:
        return None, None, f"json_decode_error: {exc!r}", None

    if not isinstance(data, dict):
        return None, None, "json_not_object", data

    answer = normalize_answer(data.get("answer"))
    rationale = data.get("rationale")

    if set(data) - {"answer", "rationale"}:
        return answer, rationale if isinstance(rationale, str) else None, "unexpected_fields", data
    if answer not in VALID_ANSWERS:
        return answer, rationale if isinstance(rationale, str) else None, "invalid_answer", data
    if not isinstance(rationale, str):
        return answer, None, "rationale_not_string", data
    if len(rationale) > 160:
        return answer, rationale, f"rationale_too_long:{len(rationale)}", data

    return answer, rationale, None, data


def score(
    answer: Optional[str],
    correct: Optional[str],
    official_available: bool,
    parse_error: Optional[str],
    api_error: Optional[str],
) -> Dict[str, Any]:
    if api_error is not None:
        return {
            "status": "api_error",
            "score": 0.0 if official_available else None,
            "is_correct": False if official_available else None,
            "is_incorrect": False if official_available else None,
            "is_blank": False,
        }
    if parse_error is not None:
        return {
            "status": "invalid_or_missing",
            "score": 0.0 if official_available else None,
            "is_correct": False if official_available else None,
            "is_incorrect": False if official_available else None,
            "is_blank": False,
        }
    if not official_available:
        return {
            "status": "not_evaluable",
            "score": None,
            "is_correct": None,
            "is_incorrect": None,
            "is_blank": answer == "blank",
        }
    if answer == "blank":
        return {
            "status": "blank",
            "score": 0.0,
            "is_correct": False,
            "is_incorrect": False,
            "is_blank": True,
        }
    if answer == correct:
        return {
            "status": "correct",
            "score": 1.0,
            "is_correct": True,
            "is_incorrect": False,
            "is_blank": False,
        }
    return {
        "status": "incorrect",
        "score": -1.0 / 3.0,
        "is_correct": False,
        "is_incorrect": True,
        "is_blank": False,
    }


# ---------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------

def retryable(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return getattr(exc, "status_code", None) in {408, 409, 425, 429, 500, 502, 503, 504}
    return False


def wait_seconds(attempt: int, base: float, maximum: float) -> float:
    return min(base * (2 ** max(0, attempt - 1)), maximum) + random.uniform(0, 0.75)


def call_with_retries(
    client: OpenAI,
    prompt: str,
    max_tokens: int,
    max_attempts: int,
    base_sleep: float,
    max_backoff: float,
) -> Dict[str, Any]:
    started = time.perf_counter()
    attempt_log: List[Dict[str, Any]] = []
    last: Dict[str, Any] = {}

    for attempt in range(1, max_attempts + 1):
        try:
            result = call_once(client, prompt, max_tokens)
            last = result

            if result["substantive_reasoning_observed"]:
                message = (
                    "PROTOCOL VIOLATION: substantive reasoning observed despite "
                    'thinking={"type":"disabled"}. The pilot permits at most '
                    f"{RESIDUAL_REASONING_TOKEN_ALLOWANCE} reported residual "
                    "reasoning token when reasoning_content is empty."
                )
                attempt_log.append({
                    "attempt": attempt,
                    "error_type": "UnexpectedSubstantiveReasoningObserved",
                    "error": message,
                    "reasoning_tokens": result.get("reasoning_tokens"),
                    "reasoning_tokens_numeric": result.get(
                        "reasoning_tokens_numeric"
                    ),
                    "reasoning_content_length": len(
                        result.get("reasoning_content") or ""
                    ),
                    "reasoning_audit_class": result.get(
                        "reasoning_audit_class"
                    ),
                    "retry": False,
                })
                return {
                    **result,
                    "ok": False,
                    "protocol_violation": True,
                    "answer": None,
                    "rationale": None,
                    "parsed_json": None,
                    "parse_error": None,
                    "api_error": message,
                    "error_type": "UnexpectedSubstantiveReasoningObserved",
                    "attempts_used": attempt,
                    "attempt_log": attempt_log,
                    "latency_seconds": time.perf_counter() - started,
                }

            answer, rationale, parse_error, parsed_json = parse_output(result["raw_content"])
            valid_finish = result.get("finish_reason") == "stop"

            if parse_error is None and valid_finish:
                return {
                    **result,
                    "ok": True,
                    "protocol_violation": False,
                    "answer": answer,
                    "rationale": rationale,
                    "parsed_json": parsed_json,
                    "parse_error": None,
                    "api_error": None,
                    "error_type": None,
                    "attempts_used": attempt,
                    "attempt_log": attempt_log,
                    "latency_seconds": time.perf_counter() - started,
                }

            issue = "; ".join(
                part for part in [
                    parse_error,
                    None if valid_finish else f"unexpected_finish_reason:{result.get('finish_reason')}",
                ]
                if part
            )
            should_retry = attempt < max_attempts
            attempt_log.append({
                "attempt": attempt,
                "error_type": "StructuredOutputValidationError",
                "error": issue,
                "finish_reason": result.get("finish_reason"),
                "retry": should_retry,
            })
            if should_retry:
                time.sleep(wait_seconds(attempt, base_sleep, max_backoff))
                continue

            return {
                **result,
                "ok": False,
                "protocol_violation": False,
                "answer": answer,
                "rationale": rationale,
                "parsed_json": parsed_json,
                "parse_error": parse_error or issue,
                "api_error": None,
                "error_type": "StructuredOutputValidationError",
                "attempts_used": attempt,
                "attempt_log": attempt_log,
                "latency_seconds": time.perf_counter() - started,
            }

        except BadRequestError as exc:
            return {
                "ok": False,
                "protocol_violation": False,
                "reasoning_observed": False,
                "answer": None,
                "rationale": None,
                "parsed_json": None,
                "parse_error": None,
                "api_error": repr(exc),
                "error_type": type(exc).__name__,
                "attempts_used": attempt,
                "attempt_log": attempt_log + [{
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                    "retry": False,
                }],
                "traceback": traceback.format_exc(),
                "latency_seconds": time.perf_counter() - started,
            }

        except Exception as exc:
            should_retry = retryable(exc) and attempt < max_attempts
            attempt_log.append({
                "attempt": attempt,
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "status_code": getattr(exc, "status_code", None),
                "retry": should_retry,
            })
            if should_retry:
                time.sleep(wait_seconds(attempt, base_sleep, max_backoff))
                continue

            return {
                "ok": False,
                "protocol_violation": False,
                "reasoning_observed": False,
                "answer": None,
                "rationale": None,
                "parsed_json": None,
                "parse_error": None,
                "api_error": repr(exc),
                "error_type": type(exc).__name__,
                "attempts_used": attempt,
                "attempt_log": attempt_log,
                "traceback": traceback.format_exc(),
                "latency_seconds": time.perf_counter() - started,
            }

    return {
        **last,
        "ok": False,
        "protocol_violation": False,
        "answer": None,
        "rationale": None,
        "parsed_json": None,
        "parse_error": "max_attempts_exhausted",
        "api_error": None,
        "error_type": "MaxAttemptsExhausted",
        "attempts_used": max_attempts,
        "attempt_log": attempt_log,
        "latency_seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for replicate, group in results.groupby("replicate", sort=True):
        evaluable = group[group["official_answer_available"] == True]

        n_evaluable = len(evaluable)
        n_correct = int((evaluable["status"] == "correct").sum())
        n_incorrect = int((evaluable["status"] == "incorrect").sum())
        n_blank = int((evaluable["status"] == "blank").sum())
        n_invalid = int((evaluable["status"] == "invalid_or_missing").sum())
        n_api_error = int((evaluable["status"] == "api_error").sum())

        numeric = lambda column: pd.to_numeric(group[column], errors="coerce")

        rows.append({
            "replicate": int(replicate),
            "n_total_attempted": int(len(group)),
            "n_evaluable": int(n_evaluable),
            "n_official_unavailable": int((group["official_answer_available"] == False).sum()),
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_blank": n_blank,
            "n_invalid_or_missing": n_invalid,
            "n_api_error": n_api_error,
            "percent_correct_evaluable": (
                100.0 * n_correct / n_evaluable if n_evaluable else None
            ),
            "score": evaluable["score"].sum(skipna=True),
            "prompt_tokens": numeric("prompt_tokens").sum(min_count=1),
            "completion_tokens": numeric("completion_tokens").sum(min_count=1),
            "reasoning_tokens": numeric("reasoning_tokens").sum(min_count=1),
            "total_tokens": numeric("total_tokens").sum(min_count=1),
            "n_calls_with_reasoning_content": int(
                group["reasoning_content_observed"].sum()
            ),
            "n_calls_with_any_reasoning_tokens": int(
                group["any_reasoning_tokens_reported"].sum()
            ),
            "n_calls_with_residual_reasoning_token_only": int(
                group["residual_reasoning_token_only"].sum()
            ),
            "n_calls_with_substantive_reasoning": int(
                group["substantive_reasoning_observed"].sum()
            ),
            "n_calls_with_reasoning_observed": int(
                group["reasoning_observed"].sum()
            ),
            "n_protocol_violations": int(group["protocol_violation"].sum()),
            "min_reasoning_tokens": numeric("reasoning_tokens").min(),
            "median_reasoning_tokens": numeric("reasoning_tokens").median(),
            "max_reasoning_tokens": numeric("reasoning_tokens").max(),
            "mean_latency_seconds": numeric("latency_seconds").mean(),
            "median_latency_seconds": numeric("latency_seconds").median(),
            "total_attempts_used": numeric("attempts_used").sum(min_count=1),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-q", type=int, default=None)
    parser.add_argument("--end-q", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-attempts", type=int, default=7)
    parser.add_argument("--base-sleep", type=float, default=10.0)
    parser.add_argument("--max-backoff", type=float, default=120.0)
    parser.add_argument("--between-calls-sleep", type=float, default=0.15)
    parser.add_argument("--between-replicates-sleep", type=float, default=2.0)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.replicates < 1 or args.max_attempts < 1 or args.max_tokens < 1:
        raise ValueError("replicates, max-attempts and max-tokens must be >= 1")

    # Fail early if someone edits these to unsupported values.
    if TEMPERATURE != 0.6 or TOP_P != 0.95:
        raise RuntimeError(
            "Kimi K2.6 non-thinking requires temperature=0.6 and top_p=0.95."
        )

    load_dotenv()
    api_key = os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")
    if not api_key:
        raise RuntimeError("MOONSHOT_API_KEY or KIMI_API_KEY not found in .env")

    base_url = os.getenv("KIMI_BASE_URL", BASE_URL_DEFAULT)

    all_questions, answer_key = load_inputs()
    questions = select_questions(all_questions, args.limit, args.start_q, args.end_q)
    key_map = build_key_map(answer_key)

    is_full_definitive_selection = (
        args.limit is None
        and args.start_q is None
        and args.end_q is None
        and len(questions) == 210
        and args.replicates == 2
    )

    timestamp = utc_compact()
    suffix: List[str] = []
    if args.limit is not None:
        suffix.append(f"limit{args.limit}")
    if args.start_q is not None or args.end_q is not None:
        suffix.append(f"q{args.start_q or 'start'}-{args.end_q or 'end'}")
    selection_suffix = "_" + "_".join(suffix) if suffix else ""

    run_name = (
        f"bir2024_kimi_k2_6_itemwise_nonthinking_"
        f"{args.replicates}r_STRICT{selection_suffix}_{timestamp}"
    )
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    metadata_file = output_dir / f"{run_name}_metadata.json"
    raw_file = output_dir / f"{run_name}_raw_results.jsonl"
    results_file = output_dir / f"{run_name}_results.csv"
    summary_file = output_dir / f"{run_name}_summary.csv"
    prompt_file = output_dir / f"{run_name}_prompt_preamble.txt"

    raw_file.write_text("", encoding="utf-8")
    prompt_file.write_text(SYSTEM_PROMPT, encoding="utf-8")

    metadata: Dict[str, Any] = {
        "run_id": run_name,
        "created_utc": timestamp,
        "provider": PROVIDER,
        "interface": "API",
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
        "n_questions_in_file": int(len(all_questions)),
        "n_questions_selected": int(len(questions)),
        "is_full_definitive_selection": is_full_definitive_selection,
        "limit": args.limit,
        "start_q": args.start_q,
        "end_q": args.end_q,
        "replicates": args.replicates,
        "thinking": {
            "requested": THINKING,
            "policy": "explicitly_disabled",
            "reasoning_audit_rule": {
                "abort_if_reasoning_content_nonempty": True,
                "abort_if_reasoning_tokens_above": (
                    RESIDUAL_REASONING_TOKEN_ALLOWANCE
                ),
                "residual_reasoning_token_allowance": (
                    RESIDUAL_REASONING_TOKEN_ALLOWANCE
                ),
                "interpretation": (
                    "Exactly 1 reported reasoning token with empty "
                    "reasoning_content is logged as provider-accounting "
                    "residue, not substantive reasoning."
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
        "rationales_requested": True,
        "rationale_max_chars": 160,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "answer_schema": ANSWER_SCHEMA,
        "retry_policy": {
            "max_total_attempts": args.max_attempts,
            "base_sleep_seconds": args.base_sleep,
            "max_backoff_seconds": args.max_backoff,
            "retry_transient_api_failures": True,
            "retry_invalid_structured_outputs": True,
            "retry_unexpected_finish_reason": True,
            "retry_valid_but_incorrect_answers": False,
        },
        "request_timeout_seconds": args.timeout,
        "between_calls_sleep_seconds": args.between_calls_sleep,
        "between_replicates_sleep_seconds": args.between_replicates_sleep,
        "scoring": {
            "correct": 1,
            "incorrect": -1 / 3,
            "blank": 0,
            "missing_or_invalid": 0,
            "questions_without_official_answer_excluded": True,
        },
    }
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=args.timeout,
        max_retries=0,
    )

    print("=== BIR 2024 — Kimi K2.6 NON-THINKING definitive run ===")
    print("Run:", run_name)
    print("Questions:", len(questions))
    print("Replicates:", args.replicates)
    print("Full definitive selection:", is_full_definitive_selection)
    if not is_full_definitive_selection:
        print(
            "WARNING: this is not the prespecified 210-question × "
            "2-replicate definitive run."
        )
    print("thinking:", THINKING)
    print("temperature:", TEMPERATURE)
    print("top_p:", TOP_P)
    print("stream: True")
    print(
        "Residual reasoning-token allowance:",
        RESIDUAL_REASONING_TOKEN_ALLOWANCE,
        "(only with empty reasoning_content)",
    )
    print()

    rows: List[Dict[str, Any]] = []
    total_calls = len(questions) * args.replicates
    call_number = 0

    for replicate in range(1, args.replicates + 1):
        print(f"--- Replicate {replicate}/{args.replicates} ---", flush=True)

        for _, question in questions.iterrows():
            call_number += 1
            qid = safe_str(question["question_id"]).strip()
            qnum = int(question["original_question_number"])
            prompt = build_user_prompt(question)

            print(
                f"[{call_number}/{total_calls}] rep={replicate} "
                f"{qid} (original {qnum})",
                flush=True,
            )

            request_time = utc_iso()
            result = call_with_retries(
                client=client,
                prompt=prompt,
                max_tokens=args.max_tokens,
                max_attempts=args.max_attempts,
                base_sleep=args.base_sleep,
                max_backoff=args.max_backoff,
            )
            response_time = utc_iso()

            key_info = key_map.get(
                qid,
                {"correct_answer": None, "official_answer_available": False},
            )
            correct = key_info["correct_answer"]
            official_available = bool(key_info["official_answer_available"])

            scoring = score(
                result.get("answer"),
                correct,
                official_available,
                result.get("parse_error"),
                result.get("api_error"),
            )

            raw_record = {
                "run_id": run_name,
                "replicate": replicate,
                "question_id": qid,
                "original_question_number": qnum,
                "request_time_utc": request_time,
                "response_time_utc": response_time,
                "provider": PROVIDER,
                "model_requested": MODEL,
                "condition": CONDITION,
                "thinking_requested": THINKING,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "prompt_sha256": sha256_text(SYSTEM_PROMPT + "\n" + prompt),
                "user_prompt": prompt,
                **result,
                "official_answer_available": official_available,
                "correct_answer": correct,
                **scoring,
            }
            append_jsonl(raw_file, raw_record)

            row = {
                "run_id": run_name,
                "replicate": replicate,
                "question_id": qid,
                "original_question_number": qnum,
                "official_answer_available": official_available,
                "correct_answer": correct,
                "answer": result.get("answer"),
                "rationale": result.get("rationale"),
                "parse_error": result.get("parse_error"),
                "status": scoring["status"],
                "score": scoring["score"],
                "is_correct": scoring["is_correct"],
                "is_incorrect": scoring["is_incorrect"],
                "is_blank": scoring["is_blank"],
                "ok": result.get("ok"),
                "protocol_violation": result.get("protocol_violation", False),
                "reasoning_content_observed": result.get(
                    "reasoning_content_observed", False
                ),
                "any_reasoning_tokens_reported": result.get(
                    "any_reasoning_tokens_reported", False
                ),
                "residual_reasoning_token_only": result.get(
                    "residual_reasoning_token_only", False
                ),
                "substantive_reasoning_observed": result.get(
                    "substantive_reasoning_observed", False
                ),
                "reasoning_observed": result.get("reasoning_observed", False),
                "reasoning_audit_class": result.get("reasoning_audit_class"),
                "reasoning_tokens_numeric": result.get(
                    "reasoning_tokens_numeric"
                ),
                "thinking_requested": "disabled",
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "attempts_used": result.get("attempts_used"),
                "error_type": result.get("error_type"),
                "error": result.get("api_error"),
                "model_returned": result.get("model_returned"),
                "response_id": result.get("response_id"),
                "finish_reason": result.get("finish_reason"),
                "prompt_tokens": result.get("prompt_tokens"),
                "completion_tokens": result.get("completion_tokens"),
                "reasoning_tokens": result.get("reasoning_tokens"),
                "cached_tokens": result.get("cached_tokens"),
                "total_tokens": result.get("total_tokens"),
                "n_stream_chunks": result.get("n_stream_chunks"),
                "latency_seconds": result.get("latency_seconds"),
            }
            rows.append(row)

            results_df = pd.DataFrame(rows)
            results_df.to_csv(
                results_file,
                index=False,
                quoting=csv.QUOTE_ALL,
                encoding="utf-8-sig",
            )
            build_summary(results_df).to_csv(
                summary_file,
                index=False,
                quoting=csv.QUOTE_ALL,
                encoding="utf-8-sig",
            )

            if result.get("protocol_violation"):
                metadata["aborted_due_to_protocol_violation"] = True
                metadata["completed_utc"] = utc_compact()
                metadata_file.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    "\nRUN ABORTED: substantive reasoning was observed "
                    "despite thinking being explicitly disabled."
                )
                sys.exit(3)

            if args.between_calls_sleep > 0:
                time.sleep(args.between_calls_sleep)

        if replicate < args.replicates and args.between_replicates_sleep > 0:
            time.sleep(args.between_replicates_sleep)

    results_df = pd.DataFrame(rows)
    summary_df = build_summary(results_df)

    results_df.to_csv(
        results_file,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        summary_file,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding="utf-8-sig",
    )

    reasoning_values = pd.to_numeric(results_df["reasoning_tokens"], errors="coerce")
    total_reasoning_tokens = reasoning_values.sum(min_count=1)

    metadata["completed_utc"] = utc_compact()
    metadata["definitive_run_completed_as_prespecified"] = (
        is_full_definitive_selection and len(results_df) == 420
    )
    reasoning_token_distribution = (
        reasoning_values.dropna().value_counts().sort_index()
    )
    reasoning_token_distribution_dict = {
        str(int(value) if float(value).is_integer() else float(value)): int(count)
        for value, count in reasoning_token_distribution.items()
    }

    metadata["observed_reasoning_audit"] = {
        "n_calls_with_reasoning_content": int(
            results_df["reasoning_content_observed"].sum()
        ),
        "n_calls_with_any_reasoning_tokens": int(
            results_df["any_reasoning_tokens_reported"].sum()
        ),
        "n_calls_with_residual_reasoning_token_only": int(
            results_df["residual_reasoning_token_only"].sum()
        ),
        "n_calls_with_substantive_reasoning": int(
            results_df["substantive_reasoning_observed"].sum()
        ),
        "n_calls_with_reasoning_observed": int(
            results_df["reasoning_observed"].sum()
        ),
        "n_protocol_violations": int(
            results_df["protocol_violation"].sum()
        ),
        "total_reasoning_tokens": (
            None if pd.isna(total_reasoning_tokens)
            else float(total_reasoning_tokens)
        ),
        "reasoning_token_distribution": reasoning_token_distribution_dict,
        "residual_reasoning_token_allowance": (
            RESIDUAL_REASONING_TOKEN_ALLOWANCE
        ),
        "run_acceptance_criterion": (
            "All calls must have empty reasoning_content and no call may "
            "report more than the residual-token allowance."
        ),
    }
    metadata["output_files"] = {
        "metadata": str(metadata_file),
        "raw_results": str(raw_file),
        "results": str(results_file),
        "summary": str(summary_file),
        "prompt_preamble": str(prompt_file),
    }
    metadata_file.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Final summary ===")
    print(summary_df.to_string(index=False))

    print("\n=== Reasoning audit ===")
    print(
        "Calls with non-empty reasoning_content:",
        int(results_df["reasoning_content_observed"].sum()),
    )
    print(
        "Calls with residual accounting token only:",
        int(results_df["residual_reasoning_token_only"].sum()),
    )
    print(
        "Calls with substantive reasoning:",
        int(results_df["substantive_reasoning_observed"].sum()),
    )
    print(
        "Reasoning-token distribution:",
        reasoning_token_distribution_dict,
    )

    run_passed = (
        int(results_df["reasoning_content_observed"].sum()) == 0
        and int(results_df["substantive_reasoning_observed"].sum()) == 0
        and int(results_df["protocol_violation"].sum()) == 0
    )
    expected_records = len(questions) * args.replicates
    record_count_passed = len(results_df) == expected_records
    definitive_run_passed = (
        is_full_definitive_selection
        and len(results_df) == 420
        and run_passed
    )

    print(
        "NON-THINKING DEFINITIVE RUN AUDIT:",
        "PASS" if run_passed else "FAIL",
    )
    print(
        "EXPECTED RECORD COUNT AUDIT:",
        "PASS" if record_count_passed else "FAIL",
        f"({len(results_df)}/{expected_records})",
    )
    print(
        "PRESPECIFIED 210 × 2 DEFINITIVE RUN:",
        "PASS" if definitive_run_passed else "FAIL",
    )
    print("\nFiles written:")
    for path in [metadata_file, raw_file, results_file, summary_file, prompt_file]:
        print(" ", path)


if __name__ == "__main__":
    main()

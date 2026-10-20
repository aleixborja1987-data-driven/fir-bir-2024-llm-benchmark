"""
BIR 2024 — xAI Grok 4.3 whole-exam, answers-only, NON-REASONING
================================================================

Definitive benchmark condition
------------------------------
- Model: grok-4.3
- Endpoint: xAI OpenAI-compatible Chat Completions
- Full BIR exam text in one request per replicate
- 210 questions per request in the definitive condition
- Two independent replicates by default
- Text-only input
- Answers-only strict JSON Schema output
- No rationales: this avoids any rationale-length validation artefact
- Official answer key merged locally only after inference
- No tools, web search, X search, files, images, retrieval, or answer key
  are supplied to the model

Generation parameters
---------------------
- reasoning_effort = "none"
- temperature = 0
- top_p = 1
- max_tokens = 10000
- stream = False
- response_format = strict json_schema

Non-reasoning safeguard
-----------------------
Every successful response must report:
- completion_tokens_details.reasoning_tokens == 0
- empty/absent message.reasoning_content

The entire run aborts immediately if:
- positive reasoning tokens are observed;
- reasoning_content is non-empty; or
- the provider does not expose an explicit numeric reasoning-token count.

Structured-output safeguards
----------------------------
- Exactly one answer record per question
- Exactly the expected question numbers
- No duplicates, missing or unexpected question numbers
- Same order as the input
- Answer restricted to "1", "2", "3", "4", or "blank"
- No rationales or extra fields
- Invalid/incomplete outputs may be retried
- Valid answers are never retried because they are incorrect

Connection and filesystem safeguards
------------------------------------
- truststore is injected before importing the OpenAI client
- short output paths for Windows
- path-length preflight before any API call

Expected local files
--------------------
- .env containing XAI_API_KEY=...
- bir2024_questions_210.csv
- bir2024_answer_key_210.csv

Installation
------------
    python -m pip install --upgrade openai pandas python-dotenv truststore

Dry run without API calls
-------------------------
    python "run_bir2024_grok43_whole_NONREASONING.py" --dry-run

Small structural pilot
----------------------
    python "run_bir2024_grok43_whole_NONREASONING.py" `
      --limit 10 `
      --replicates 1

Definitive run
--------------
    python "run_bir2024_grok43_whole_NONREASONING.py" `
      --replicates 2
"""

from __future__ import annotations

import argparse
import csv
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
from typing import Any, Dict, List, Optional, Tuple

try:
    import truststore
except ImportError as exc:
    raise RuntimeError(
        "Falta truststore. Instálalo con: "
        "python -m pip install --upgrade truststore"
    ) from exc

# Inject before importing the OpenAI/httpx client.
truststore.inject_into_ssl()
TRUSTSTORE_STATUS = "enabled"

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# Study configuration
# ============================================================

MODEL_REQUESTED = "grok-4.3"
PROVIDER = "xAI"
INTERFACE = "API"
API_ENDPOINT = "OpenAI-compatible Chat Completions"
BASE_URL = "https://api.x.ai/v1"

CONDITION = "whole_text_answers_only_nonthinking"
PROMPT_TEMPLATE_VERSION = (
    "bir2024_whole_text_examlike_blank_"
    "answers_only_strict_json_nonthinking_v1"
)
SCRIPT_VERSION = (
    "bir2024_grok43_whole_text_answers_only_"
    "strict_json_reasoning_audit_v1"
)

TEMPERATURE = 0
TOP_P = 1
MAX_TOKENS = 10000
STREAM = False

DEFAULT_REPLICATES = 2
REQUEST_TIMEOUT_SECONDS = 900.0
MAX_ATTEMPTS_PER_REPLICATE = 4
BASE_RETRY_SLEEP_SECONDS = 10.0
MAX_RETRY_SLEEP_SECONDS = 120.0
BETWEEN_REPLICATES_SLEEP_SECONDS = 2.0

REQUIRE_EXPLICIT_ZERO_REASONING_TOKENS = True

QUESTIONS_FILE = Path("bir2024_questions_210.csv")
ANSWER_KEY_FILE = Path("bir2024_answer_key_210.csv")
OUTPUT_ROOT = Path("outputs")

VALID_ANSWERS = {"1", "2", "3", "4", "blank"}


PROMPT_PREAMBLE = """You are taking an examination for Biology residency access.

This is a multiple-choice exam. Each question has four options, numbered 1 to 4, and only one is correct. In the official exam, correct answers score +1 point, incorrect answers score -1/3 points, and blank answers score 0 points. Act as a human candidate and try to get the highest score possible.

You will receive the full exam text, containing all questions and their four answer options.

Return exactly one JSON object matching the supplied schema.

Output rules:
- Return exactly one answer object for every question included in the input.
- Use the integer question number shown in the input.
- Preserve the same question order as in the input.
- The value of "answer" must be exactly one single string: "1", "2", "3", "4", or "blank".
- Do not return multiple options.
- If you cannot choose one single best option, return "blank".
- Do not include rationales, explanations, comments, markdown, or any other fields.
- Do not use web search, X search, external references, files, images, retrieval, tools, or an answer key.

FULL EXAM TEXT:
"""


GENERATION_PARAMETERS: Dict[str, Any] = {
    "reasoning_effort": "none",
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "max_tokens": MAX_TOKENS,
    "stream": STREAM,
    "structured_output": "strict_json_schema",
    "rationales_requested": False,
    "tools_enabled": False,
    "web_search_enabled": False,
    "x_search_enabled": False,
    "file_input_enabled": False,
    "image_input_enabled": False,
    "retrieval_enabled": False,
    "truststore_status": TRUSTSTORE_STATUS,
}


# ============================================================
# General helpers
# ============================================================

def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}

    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    try:
        return dict(obj)
    except Exception:
        return {"repr": repr(obj)}


def bool_like(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "si",
        "sí",
    }


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

    return text


def normalize_question_number(value: Any) -> Optional[int]:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    match = re.search(r"\d+", str(value).strip())
    return int(match.group(0)) if match else None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return None


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


def retry_sleep_seconds(attempt: int) -> float:
    base = BASE_RETRY_SLEEP_SECONDS * (
        2 ** max(0, attempt - 1)
    )
    return min(base, MAX_RETRY_SLEEP_SECONDS) + random.uniform(
        0,
        0.75,
    )


# ============================================================
# Input and prompt construction
# ============================================================

def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"No existe {QUESTIONS_FILE}"
        )

    if not ANSWER_KEY_FILE.exists():
        raise FileNotFoundError(
            f"No existe {ANSWER_KEY_FILE}"
        )

    questions = pd.read_csv(QUESTIONS_FILE)
    answer_key = pd.read_csv(ANSWER_KEY_FILE)

    required_question_columns = {
        "exam",
        "question_id",
        "original_question_number",
        "is_reserve",
        "question_text",
        "option_1",
        "option_2",
        "option_3",
        "option_4",
    }
    required_key_columns = {
        "exam",
        "question_id",
        "correct_answer",
    }

    missing_questions = (
        required_question_columns - set(questions.columns)
    )
    missing_key = (
        required_key_columns - set(answer_key.columns)
    )

    if missing_questions:
        raise ValueError(
            "Faltan columnas en questions: "
            f"{sorted(missing_questions)}"
        )

    if missing_key:
        raise ValueError(
            "Faltan columnas en answer key: "
            f"{sorted(missing_key)}"
        )

    if len(questions) != 210:
        raise ValueError(
            f"Se esperaban 210 preguntas BIR, hay {len(questions)}"
        )

    if len(answer_key) != 210:
        raise ValueError(
            "Se esperaban 210 filas en el answer key, "
            f"hay {len(answer_key)}"
        )

    questions = questions.sort_values(
        "original_question_number"
    ).reset_index(drop=True)

    answer_key["correct_answer"] = answer_key[
        "correct_answer"
    ].apply(normalize_answer)

    return questions, answer_key


def select_questions(
    questions: pd.DataFrame,
    limit: Optional[int],
    start_q: Optional[int],
    end_q: Optional[int],
) -> pd.DataFrame:
    selected = questions.copy()

    if start_q is not None:
        selected = selected[
            selected["original_question_number"].astype(int)
            >= start_q
        ]

    if end_q is not None:
        selected = selected[
            selected["original_question_number"].astype(int)
            <= end_q
        ]

    selected = selected.sort_values(
        "original_question_number"
    ).reset_index(drop=True)

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit debe ser >= 1")
        selected = selected.head(limit).copy()

    if selected.empty:
        raise ValueError("No se seleccionó ninguna pregunta.")

    return selected


def format_question(row: pd.Series) -> str:
    question_number = int(
        row["original_question_number"]
    )

    return f"""Question {question_number}:
{row['question_text']}

Options:
1. {row['option_1']}
2. {row['option_2']}
3. {row['option_3']}
4. {row['option_4']}
"""


def build_full_prompt(
    questions: pd.DataFrame,
) -> str:
    parts: List[str] = [
        PROMPT_PREAMBLE,
        "\n\n",
    ]

    for index, (_, row) in enumerate(
        questions.iterrows(),
        start=1,
    ):
        parts.append(format_question(row))

        if index < len(questions):
            parts.append("\n---\n\n")

    return "".join(parts)


def build_answer_schema(
    n_questions: int,
) -> Dict[str, Any]:
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
                        "question_number": {
                            "type": "integer",
                        },
                        "answer": {
                            "type": "string",
                            "enum": [
                                "1",
                                "2",
                                "3",
                                "4",
                                "blank",
                            ],
                        },
                    },
                    "required": [
                        "question_number",
                        "answer",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["answers"],
        "additionalProperties": False,
    }


def build_response_format(
    n_questions: int,
) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "bir_all_answers",
            "strict": True,
            "schema": build_answer_schema(
                n_questions
            ),
        },
    }


# ============================================================
# Response parsing and reasoning audit
# ============================================================

def extract_reasoning_tokens(
    usage_dict: Dict[str, Any],
) -> Optional[int]:
    if not isinstance(usage_dict, dict):
        return None

    details = (
        usage_dict.get("completion_tokens_details")
        or usage_dict.get("output_tokens_details")
        or {}
    )

    if isinstance(details, dict):
        value = details.get("reasoning_tokens")
        if value is None:
            value = details.get("thinking_tokens")

        if value is not None:
            return safe_int(value)

    direct = usage_dict.get("reasoning_tokens")
    if direct is not None:
        return safe_int(direct)

    return None


def flatten_usage(
    usage_dict: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(usage_dict, dict):
        usage_dict = {}

    prompt_details = (
        usage_dict.get("prompt_tokens_details")
        or usage_dict.get("input_tokens_details")
        or {}
    )

    input_tokens = usage_dict.get(
        "prompt_tokens",
        usage_dict.get("input_tokens"),
    )
    output_tokens = usage_dict.get(
        "completion_tokens",
        usage_dict.get("output_tokens"),
    )
    total_tokens = usage_dict.get("total_tokens")

    if total_tokens is None:
        try:
            total_tokens = int(input_tokens or 0) + int(
                output_tokens or 0
            )
        except Exception:
            total_tokens = None

    cached_tokens = None
    if isinstance(prompt_details, dict):
        cached_tokens = prompt_details.get(
            "cached_tokens"
        )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": extract_reasoning_tokens(
            usage_dict
        ),
        "cached_input_tokens": cached_tokens,
    }


def extract_message_fields(
    response: Any,
) -> Tuple[
    Optional[str],
    Optional[str],
    Optional[str],
]:
    choices = getattr(response, "choices", None) or []

    if not choices:
        return None, None, None

    choice = choices[0]
    message = getattr(choice, "message", None)

    if message is None:
        return (
            None,
            None,
            getattr(choice, "finish_reason", None),
        )

    return (
        getattr(message, "content", None),
        getattr(message, "reasoning_content", None),
        getattr(choice, "finish_reason", None),
    )


def parse_all_answers(
    raw_text: Optional[str],
    expected_question_numbers: List[int],
) -> Tuple[
    Dict[int, Dict[str, Any]],
    Optional[str],
    Dict[str, Any],
]:
    if not raw_text:
        return {}, "missing_raw_output", {}

    try:
        data = json.loads(raw_text)
    except Exception as exc:
        return (
            {},
            f"json_decode_error:{type(exc).__name__}:{exc}",
            {"raw_text": raw_text},
        )

    if not isinstance(data, dict):
        return {}, "json_not_object", {
            "parsed_json": data,
        }

    if set(data.keys()) != {"answers"}:
        return {}, "unexpected_top_level_fields", {
            "parsed_json": data,
        }

    answers = data.get("answers")

    if not isinstance(answers, list):
        return {}, "answers_missing_or_not_list", {
            "parsed_json": data,
        }

    records: Dict[int, Dict[str, Any]] = {}
    duplicates: List[int] = []
    invalid_records: List[Dict[str, Any]] = []

    for item in answers:
        if not isinstance(item, dict):
            invalid_records.append({
                "reason": "record_not_object",
                "record": item,
            })
            continue

        if set(item.keys()) != {
            "question_number",
            "answer",
        }:
            invalid_records.append({
                "reason": "unexpected_record_fields",
                "record": item,
            })
            continue

        question_number = normalize_question_number(
            item.get("question_number")
        )
        answer = normalize_answer(
            item.get("answer")
        )

        if question_number is None:
            invalid_records.append({
                "reason": "invalid_question_number",
                "record": item,
            })
            continue

        if answer not in VALID_ANSWERS:
            invalid_records.append({
                "reason": "invalid_answer",
                "record": item,
            })
            continue

        if question_number in records:
            duplicates.append(question_number)

        records[question_number] = {
            "raw_answer": item.get("answer"),
            "answer": answer,
            "raw_record": item,
        }

    returned_numbers = sorted(records.keys())
    expected_set = set(expected_question_numbers)
    returned_set = set(returned_numbers)

    missing_numbers = sorted(
        expected_set - returned_set
    )
    unexpected_numbers = sorted(
        returned_set - expected_set
    )

    diagnostics = {
        "raw_json": data,
        "n_answer_records_returned": len(answers),
        "n_unique_valid_answer_records": len(records),
        "duplicate_question_numbers": sorted(
            set(duplicates)
        ),
        "missing_question_numbers": missing_numbers,
        "unexpected_question_numbers": (
            unexpected_numbers
        ),
        "invalid_records": invalid_records,
        "returned_question_numbers": returned_numbers,
    }

    errors: List[str] = []

    if len(answers) != len(
        expected_question_numbers
    ):
        errors.append(
            "wrong_answer_record_count:"
            f"{len(answers)}/"
            f"{len(expected_question_numbers)}"
        )

    if duplicates:
        errors.append(
            "duplicate_question_numbers:"
            + ",".join(
                map(
                    str,
                    sorted(set(duplicates)),
                )
            )
        )

    if invalid_records:
        errors.append(
            f"invalid_answer_records:{len(invalid_records)}"
        )

    if missing_numbers:
        errors.append(
            "missing_question_numbers:"
            + ",".join(map(str, missing_numbers))
        )

    if unexpected_numbers:
        errors.append(
            "unexpected_question_numbers:"
            + ",".join(map(str, unexpected_numbers))
        )

    if returned_numbers != expected_question_numbers:
        errors.append("question_order_or_set_mismatch")

    return (
        records,
        "; ".join(errors) if errors else None,
        diagnostics,
    )


def is_transient_exception(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)

    if status_code in {
        408,
        409,
        425,
        429,
        500,
        502,
        503,
        504,
        529,
    }:
        return True

    name = type(exc).__name__.lower()

    return any(
        marker in name
        for marker in [
            "timeout",
            "connection",
            "ratelimit",
            "overloaded",
            "internalserver",
        ]
    )


def call_grok_once(
    client: OpenAI,
    full_prompt: str,
    response_format: Dict[str, Any],
    expected_question_numbers: List[int],
) -> Dict[str, Any]:
    started = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL_REQUESTED,
        messages=[
            {
                "role": "user",
                "content": full_prompt,
            }
        ],
        reasoning_effort="none",
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        response_format=response_format,
        stream=STREAM,
    )

    latency_seconds = time.perf_counter() - started

    raw_text, reasoning_content, finish_reason = (
        extract_message_fields(response)
    )
    usage_dict = object_to_dict(
        getattr(response, "usage", None)
    )
    usage_flat = flatten_usage(usage_dict)

    answer_records, parse_error, diagnostics = (
        parse_all_answers(
            raw_text=raw_text,
            expected_question_numbers=(
                expected_question_numbers
            ),
        )
    )

    reasoning_tokens = usage_flat[
        "reasoning_tokens"
    ]
    reasoning_content_nonempty = bool(
        isinstance(reasoning_content, str)
        and reasoning_content.strip()
    )
    missing_reasoning_audit = (
        REQUIRE_EXPLICIT_ZERO_REASONING_TOKENS
        and reasoning_tokens is None
    )
    positive_reasoning_tokens = (
        reasoning_tokens is not None
        and reasoning_tokens > 0
    )
    protocol_violation = (
        missing_reasoning_audit
        or positive_reasoning_tokens
        or reasoning_content_nonempty
    )

    if finish_reason != "stop":
        finish_error = (
            f"unexpected_finish_reason:{finish_reason}"
        )
        parse_error = (
            f"{parse_error}; {finish_error}"
            if parse_error
            else finish_error
        )

    return {
        "model_returned": getattr(
            response,
            "model",
            None,
        ),
        "response_id": getattr(response, "id", None),
        "finish_reason": finish_reason,
        "raw_output": raw_text,
        "answer_records": answer_records,
        "whole_response_parse_error": parse_error,
        "diagnostics": diagnostics,
        "reasoning_content": reasoning_content,
        "reasoning_content_nonempty": (
            reasoning_content_nonempty
        ),
        "missing_reasoning_audit": (
            missing_reasoning_audit
        ),
        "positive_reasoning_tokens": (
            positive_reasoning_tokens
        ),
        "protocol_violation": protocol_violation,
        "usage": usage_dict,
        "response_dump": object_to_dict(response),
        "latency_seconds": latency_seconds,
        **usage_flat,
    }


def call_grok_with_retries(
    client: OpenAI,
    full_prompt: str,
    response_format: Dict[str, Any],
    expected_question_numbers: List[int],
) -> Dict[str, Any]:
    overall_started = time.perf_counter()
    attempt_log: List[Dict[str, Any]] = []
    last_result: Optional[Dict[str, Any]] = None

    for attempt in range(
        1,
        MAX_ATTEMPTS_PER_REPLICATE + 1,
    ):
        try:
            result = call_grok_once(
                client=client,
                full_prompt=full_prompt,
                response_format=response_format,
                expected_question_numbers=(
                    expected_question_numbers
                ),
            )
            last_result = result

            if result["protocol_violation"]:
                return {
                    **result,
                    "call_success": False,
                    "error_type": (
                        "NonReasoningProtocolViolation"
                    ),
                    "error_message": (
                        "Expected explicit zero reasoning "
                        "tokens and empty reasoning_content."
                    ),
                    "attempts_used": attempt,
                    "attempt_log": attempt_log,
                    "elapsed_seconds_including_retries": (
                        time.perf_counter()
                        - overall_started
                    ),
                }

            if (
                result["whole_response_parse_error"]
                is None
            ):
                return {
                    **result,
                    "call_success": True,
                    "error_type": None,
                    "error_message": None,
                    "attempts_used": attempt,
                    "attempt_log": attempt_log,
                    "elapsed_seconds_including_retries": (
                        time.perf_counter()
                        - overall_started
                    ),
                }

            retry = (
                attempt
                < MAX_ATTEMPTS_PER_REPLICATE
            )
            attempt_log.append({
                "attempt": attempt,
                "error_type": (
                    "StructuredOutputValidationError"
                ),
                "error_message": result[
                    "whole_response_parse_error"
                ],
                "retry": retry,
            })

            if retry:
                time.sleep(
                    retry_sleep_seconds(attempt)
                )
                continue

        except Exception as exc:
            retry = (
                is_transient_exception(exc)
                and attempt
                < MAX_ATTEMPTS_PER_REPLICATE
            )

            attempt_log.append({
                "attempt": attempt,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "status_code": getattr(
                    exc,
                    "status_code",
                    None,
                ),
                "retry": retry,
            })

            if retry:
                time.sleep(
                    retry_sleep_seconds(attempt)
                )
                continue

            return {
                "call_success": False,
                "protocol_violation": False,
                "model_returned": None,
                "response_id": None,
                "finish_reason": None,
                "raw_output": None,
                "answer_records": {},
                "whole_response_parse_error": None,
                "diagnostics": {},
                "reasoning_content": None,
                "reasoning_content_nonempty": False,
                "missing_reasoning_audit": False,
                "positive_reasoning_tokens": False,
                "usage": {},
                "response_dump": None,
                "latency_seconds": None,
                "elapsed_seconds_including_retries": (
                    time.perf_counter()
                    - overall_started
                ),
                "attempts_used": attempt,
                "attempt_log": attempt_log,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_traceback": traceback.format_exc(),
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "reasoning_tokens": None,
                "cached_input_tokens": None,
            }

    assert last_result is not None

    return {
        **last_result,
        "call_success": False,
        "error_type": "StructuredOutputValidationError",
        "error_message": last_result[
            "whole_response_parse_error"
        ],
        "attempts_used": (
            MAX_ATTEMPTS_PER_REPLICATE
        ),
        "attempt_log": attempt_log,
        "elapsed_seconds_including_retries": (
            time.perf_counter() - overall_started
        ),
    }


# ============================================================
# Scoring
# ============================================================

def official_answer_available(row: pd.Series) -> bool:
    if "official_answer_available" in row.index:
        if not bool_like(
            row.get("official_answer_available")
        ):
            return False

    if (
        "is_annulled_or_without_official_answer"
        in row.index
    ):
        if bool_like(
            row.get(
                "is_annulled_or_without_official_answer"
            )
        ):
            return False

    return normalize_answer(
        row.get("correct_answer")
    ) in {"1", "2", "3", "4"}


def score_answer(
    model_answer: Optional[str],
    correct_answer: Optional[str],
    has_official_answer: bool,
    parse_error: Optional[str],
    api_error: Optional[str],
) -> Tuple[
    str,
    Optional[bool],
    bool,
    Optional[float],
]:
    if api_error:
        return (
            "api_or_protocol_error",
            None if not has_official_answer else False,
            False,
            0.0 if has_official_answer else None,
        )

    if parse_error:
        return (
            "missing_or_invalid",
            None if not has_official_answer else False,
            False,
            0.0 if has_official_answer else None,
        )

    if not has_official_answer:
        return (
            "no_official_answer",
            None,
            model_answer == "blank",
            None,
        )

    if model_answer == "blank":
        return "blank", False, True, 0.0

    if model_answer == correct_answer:
        return "correct", True, False, 1.0

    return "incorrect", False, False, -1.0 / 3.0


def build_discordances(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for question_id, group in scored.groupby(
        "question_id",
        sort=False,
    ):
        group = group.sort_values("replicate_id")

        if len(group) != 2:
            continue

        first = group.iloc[0]
        second = group.iloc[1]

        if first["model_answer"] == second["model_answer"]:
            continue

        rows.append({
            "run_name": first["run_name"],
            "exam": first["exam"],
            "question_id": question_id,
            "original_question_number": int(
                first["original_question_number"]
            ),
            "has_official_answer": bool(
                first["has_official_answer"]
            ),
            "correct_answer": first["correct_answer"],
            "replicate_1_answer": first[
                "model_answer"
            ],
            "replicate_1_status": first[
                "scoring_status"
            ],
            "replicate_1_score": first["exam_score"],
            "replicate_2_answer": second[
                "model_answer"
            ],
            "replicate_2_status": second[
                "scoring_status"
            ],
            "replicate_2_score": second["exam_score"],
        })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Piloto con las primeras N preguntas.",
    )
    parser.add_argument(
        "--start-q",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--end-q",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=DEFAULT_REPLICATES,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida entradas sin llamar a la API.",
    )

    args = parser.parse_args()

    if args.replicates < 1:
        raise ValueError("--replicates debe ser >= 1")

    load_dotenv(".env")

    questions_all, answer_key = load_inputs()
    questions = select_questions(
        questions=questions_all,
        limit=args.limit,
        start_q=args.start_q,
        end_q=args.end_q,
    )

    expected_question_numbers = [
        int(value)
        for value in questions[
            "original_question_number"
        ].tolist()
    ]

    full_prompt = build_full_prompt(questions)
    response_format = build_response_format(
        len(questions)
    )

    is_full_definitive_selection = (
        args.limit is None
        and args.start_q is None
        and args.end_q is None
        and len(questions) == 210
        and args.replicates == 2
    )

    print(
        "=== BIR 2024 — Grok 4.3 whole-exam "
        "answers-only NON-REASONING ==="
    )
    print("Questions included:", len(questions))
    print("Replicates:", args.replicates)
    print(
        "Full definitive selection:",
        is_full_definitive_selection,
    )
    print("Model:", MODEL_REQUESTED)
    print("Reasoning effort:", "none")
    print("Temperature:", TEMPERATURE)
    print("Top_p:", TOP_P)
    print("Max tokens:", MAX_TOKENS)
    print("Rationales requested:", False)
    print("Strict JSON Schema:", True)
    print("Truststore:", TRUSTSTORE_STATUS)
    print(
        "Require explicit zero reasoning tokens:",
        REQUIRE_EXPLICIT_ZERO_REASONING_TOKENS,
    )
    print(
        "Expected answer records per replicate:",
        len(expected_question_numbers),
    )
    print(
        "Prompt SHA-256:",
        sha256_text(full_prompt),
    )

    if args.dry_run:
        print()
        print("DRY RUN COMPLETED — no API calls made.")
        return

    api_key = os.getenv("XAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No encuentro XAI_API_KEY en .env"
        )

    timestamp = utc_timestamp()

    selection_parts: List[str] = []
    if args.limit is not None:
        selection_parts.append(
            f"limit{args.limit}"
        )
    if (
        args.start_q is not None
        or args.end_q is not None
    ):
        selection_parts.append(
            f"q{args.start_q or 'start'}-"
            f"{args.end_q or 'end'}"
        )

    selection_suffix = (
        "_"
        + "_".join(selection_parts)
        if selection_parts
        else ""
    )

    run_name = (
        f"bir24_grok43_whole_nr_{args.replicates}r"
        f"{selection_suffix}_{timestamp}"
    )
    output_dir = OUTPUT_ROOT / run_name
    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    raw_file = output_dir / "raw.jsonl"
    results_file = output_dir / "results.csv"
    summary_file = output_dir / "summary.csv"
    calls_file = output_dir / "calls.csv"
    discordances_file = (
        output_dir / "discordances.csv"
    )
    metadata_file = output_dir / "metadata.json"
    prompt_file = output_dir / "prompt.txt"

    planned_paths = [
        raw_file,
        results_file,
        summary_file,
        calls_file,
        discordances_file,
        metadata_file,
        prompt_file,
    ]

    if os.name == "nt":
        resolved_lengths = {
            str(path.resolve()): len(
                str(path.resolve())
            )
            for path in planned_paths
        }
        longest_path, longest_length = max(
            resolved_lengths.items(),
            key=lambda item: item[1],
        )

        print(
            "Longest planned output path:",
            longest_length,
            "characters",
        )

        if longest_length >= 240:
            raise OSError(
                "Ruta de salida demasiado larga para "
                "Windows "
                f"({longest_length} caracteres): "
                f"{longest_path}"
            )

    prompt_file.write_text(
        full_prompt,
        encoding="utf-8",
    )
    raw_file.write_text("", encoding="utf-8")

    prompt_hash = sha256_text(full_prompt)
    questions_hash = sha256_file(QUESTIONS_FILE)
    answer_key_hash = sha256_file(
        ANSWER_KEY_FILE
    )

    metadata: Dict[str, Any] = {
        "run_name": run_name,
        "created_utc": timestamp,
        "provider": PROVIDER,
        "interface": INTERFACE,
        "api_endpoint": API_ENDPOINT,
        "base_url": BASE_URL,
        "model_requested": MODEL_REQUESTED,
        "condition": CONDITION,
        "prompt_template_version": (
            PROMPT_TEMPLATE_VERSION
        ),
        "script_version": SCRIPT_VERSION,
        "generation_parameters": {
            **GENERATION_PARAMETERS,
            "response_format": response_format,
        },
        "answer_schema": response_format[
            "json_schema"
        ]["schema"],
        "reasoning_audit_rule": {
            "reasoning_effort_sent": "none",
            "require_explicit_numeric_reasoning_tokens": True,
            "required_reasoning_tokens_per_call": 0,
            "require_empty_reasoning_content": True,
            "abort_on_violation": True,
        },
        "structured_output_rule": {
            "answers_only": True,
            "rationales_requested": False,
            "expected_answer_records_per_call": int(
                len(questions)
            ),
            "require_exact_question_set": True,
            "require_input_order": True,
            "retry_invalid_or_incomplete_output": True,
            "retry_valid_but_incorrect_answers": False,
        },
        "connection_configuration": {
            "truststore_status": TRUSTSTORE_STATUS,
            "truststore_injected_before_openai_import": True,
        },
        "n_replicates": args.replicates,
        "n_questions_in_file": int(
            len(questions_all)
        ),
        "n_questions_attempted": int(
            len(questions)
        ),
        "expected_question_numbers": (
            expected_question_numbers
        ),
        "selection": {
            "limit": args.limit,
            "start_q": args.start_q,
            "end_q": args.end_q,
        },
        "is_full_definitive_selection": (
            is_full_definitive_selection
        ),
        "questions_file": str(QUESTIONS_FILE),
        "answer_key_file": str(ANSWER_KEY_FILE),
        "questions_file_sha256": questions_hash,
        "answer_key_file_sha256": (
            answer_key_hash
        ),
        "prompt_file": str(prompt_file),
        "prompt_sha256": prompt_hash,
        "retry_policy": {
            "max_attempts_per_replicate": (
                MAX_ATTEMPTS_PER_REPLICATE
            ),
            "retry_transient_api_errors": True,
            "retry_invalid_or_incomplete_structured_outputs": True,
            "retry_valid_but_incorrect_answers": False,
        },
        "tools_or_web_search_provided_to_model": False,
        "notes": (
            "The official answer key is merged only "
            "after inference for local scoring. No "
            "rationales are requested, preventing the "
            "rationale-length artefact observed in the "
            "item-wise condition."
        ),
    }

    metadata_file.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    all_rows: List[Dict[str, Any]] = []
    call_rows: List[Dict[str, Any]] = []

    print()
    print("Run name:", run_name)
    print("Output directory:", output_dir)

    for replicate_id in range(
        1,
        args.replicates + 1,
    ):
        print()
        print(
            f"=== Replicate {replicate_id}/"
            f"{args.replicates} ===",
            flush=True,
        )

        request_time = utc_iso()

        result = call_grok_with_retries(
            client=client,
            full_prompt=full_prompt,
            response_format=response_format,
            expected_question_numbers=(
                expected_question_numbers
            ),
        )

        response_time = utc_iso()
        diagnostics = result.get(
            "diagnostics"
        ) or {}
        answer_records = result.get(
            "answer_records"
        ) or {}

        raw_call = {
            "run_name": run_name,
            "replicate_id": replicate_id,
            "provider": PROVIDER,
            "interface": INTERFACE,
            "api_endpoint": API_ENDPOINT,
            "base_url": BASE_URL,
            "model_requested": MODEL_REQUESTED,
            "model_returned": result.get(
                "model_returned"
            ),
            "response_id": result.get(
                "response_id"
            ),
            "finish_reason": result.get(
                "finish_reason"
            ),
            "condition": CONDITION,
            "prompt_template_version": (
                PROMPT_TEMPLATE_VERSION
            ),
            "script_version": SCRIPT_VERSION,
            "n_questions_in_prompt": int(
                len(questions)
            ),
            "n_answer_records_returned": (
                diagnostics.get(
                    "n_answer_records_returned"
                )
            ),
            "n_unique_valid_answer_records": (
                diagnostics.get(
                    "n_unique_valid_answer_records"
                )
            ),
            "duplicate_question_numbers": (
                diagnostics.get(
                    "duplicate_question_numbers"
                )
            ),
            "missing_question_numbers": (
                diagnostics.get(
                    "missing_question_numbers"
                )
            ),
            "unexpected_question_numbers": (
                diagnostics.get(
                    "unexpected_question_numbers"
                )
            ),
            "invalid_records": diagnostics.get(
                "invalid_records"
            ),
            "whole_response_parse_error": (
                result.get(
                    "whole_response_parse_error"
                )
            ),
            "call_success": result.get(
                "call_success"
            ),
            "protocol_violation": result.get(
                "protocol_violation"
            ),
            "reasoning_content_nonempty": (
                result.get(
                    "reasoning_content_nonempty"
                )
            ),
            "missing_reasoning_audit": (
                result.get(
                    "missing_reasoning_audit"
                )
            ),
            "positive_reasoning_tokens": (
                result.get(
                    "positive_reasoning_tokens"
                )
            ),
            "reasoning_tokens": result.get(
                "reasoning_tokens"
            ),
            "error_type": result.get(
                "error_type"
            ),
            "error_message": result.get(
                "error_message"
            ),
            "latency_seconds": safe_float(
                result.get("latency_seconds")
            ),
            "elapsed_seconds_including_retries": (
                safe_float(
                    result.get(
                        "elapsed_seconds_including_retries"
                    )
                )
            ),
            "attempts_used": result.get(
                "attempts_used"
            ),
            "input_tokens": result.get(
                "input_tokens"
            ),
            "output_tokens": result.get(
                "output_tokens"
            ),
            "total_tokens": result.get(
                "total_tokens"
            ),
            "cached_input_tokens": result.get(
                "cached_input_tokens"
            ),
            "request_time_utc": request_time,
            "response_time_utc": response_time,
            "attempt_log": result.get(
                "attempt_log"
            ),
            "usage": result.get("usage"),
            "raw_output": result.get(
                "raw_output"
            ),
            "reasoning_content": result.get(
                "reasoning_content"
            ),
            "raw_response": result.get(
                "response_dump"
            ),
        }

        append_jsonl(raw_file, raw_call)
        call_rows.append(raw_call)

        if result.get("protocol_violation"):
            pd.DataFrame(call_rows).to_csv(
                calls_file,
                index=False,
                quoting=csv.QUOTE_ALL,
                encoding="utf-8-sig",
            )

            metadata["completed_utc"] = (
                utc_timestamp()
            )
            metadata[
                "aborted_due_to_reasoning_protocol_violation"
            ] = True
            metadata[
                "definitive_run_completed_as_prespecified"
            ] = False
            metadata[
                "observed_reasoning_audit"
            ] = {
                "n_calls_completed": len(call_rows),
                "protocol_violations": int(
                    sum(
                        bool(
                            row[
                                "protocol_violation"
                            ]
                        )
                        for row in call_rows
                    )
                ),
            }

            metadata_file.write_text(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                "RUN ABORTED: non-reasoning "
                "protocol violation.",
                flush=True,
            )
            sys.exit(3)

        for _, question_row in questions.iterrows():
            question_number = int(
                question_row[
                    "original_question_number"
                ]
            )
            record = answer_records.get(
                question_number
            )

            raw_answer = None
            model_answer = None
            parse_error = None
            raw_record = None

            if not result.get("call_success"):
                parse_error = (
                    "whole_exam_call_failed"
                )
            elif record is None:
                parse_error = (
                    "missing_answer_for_question"
                )
            else:
                raw_answer = record.get(
                    "raw_answer"
                )
                model_answer = normalize_answer(
                    record.get("answer")
                )
                raw_record = record.get(
                    "raw_record"
                )

                if model_answer not in VALID_ANSWERS:
                    parse_error = (
                        "invalid_answer_for_question"
                    )
                    model_answer = None

            all_rows.append({
                "run_name": run_name,
                "replicate_id": replicate_id,
                "provider": PROVIDER,
                "interface": INTERFACE,
                "api_endpoint": API_ENDPOINT,
                "base_url": BASE_URL,
                "model_requested": MODEL_REQUESTED,
                "model_returned": result.get(
                    "model_returned"
                ),
                "condition": CONDITION,
                "prompt_template_version": (
                    PROMPT_TEMPLATE_VERSION
                ),
                "script_version": SCRIPT_VERSION,
                "exam": question_row["exam"],
                "question_id": question_row[
                    "question_id"
                ],
                "original_question_number": (
                    question_number
                ),
                "is_reserve": bool_like(
                    question_row.get("is_reserve")
                ),
                "subject": question_row.get(
                    "subject"
                ),
                "has_image": False,
                "source_pdf_page": safe_int(
                    question_row.get(
                        "source_pdf_page"
                    )
                ),
                "model_answer": model_answer,
                "raw_answer": raw_answer,
                "parse_error": parse_error,
                "raw_record": (
                    json.dumps(
                        raw_record,
                        ensure_ascii=False,
                    )
                    if raw_record is not None
                    else None
                ),
                "response_id": result.get(
                    "response_id"
                ),
                "finish_reason": result.get(
                    "finish_reason"
                ),
                "call_success": result.get(
                    "call_success"
                ),
                "protocol_violation": result.get(
                    "protocol_violation"
                ),
                "reasoning_content_nonempty_call": (
                    result.get(
                        "reasoning_content_nonempty"
                    )
                ),
                "reasoning_tokens_call": result.get(
                    "reasoning_tokens"
                ),
                "latency_seconds_call": result.get(
                    "latency_seconds"
                ),
                "attempts_used": result.get(
                    "attempts_used"
                ),
                "input_tokens_call": result.get(
                    "input_tokens"
                ),
                "output_tokens_call": result.get(
                    "output_tokens"
                ),
                "total_tokens_call": result.get(
                    "total_tokens"
                ),
                "cached_input_tokens_call": result.get(
                    "cached_input_tokens"
                ),
            })

        print(
            "rep="
            f"{replicate_id} "
            "success="
            f"{result.get('call_success')} "
            "answers="
            f"{diagnostics.get('n_answer_records_returned')} "
            "unique_valid="
            f"{diagnostics.get('n_unique_valid_answer_records')} "
            "reasoning_tokens="
            f"{result.get('reasoning_tokens')} "
            "reasoning_content="
            f"{result.get('reasoning_content_nonempty')} "
            "attempts="
            f"{result.get('attempts_used')} "
            "model="
            f"{result.get('model_returned')}",
            flush=True,
        )

        if (
            replicate_id < args.replicates
            and BETWEEN_REPLICATES_SLEEP_SECONDS > 0
        ):
            time.sleep(
                BETWEEN_REPLICATES_SLEEP_SECONDS
            )

    unscored = pd.DataFrame(all_rows)
    call_level = pd.DataFrame(call_rows)

    answer_key_columns = [
        "exam",
        "question_id",
        "correct_answer",
    ]
    for column in [
        "official_answer_available",
        "is_annulled_or_without_official_answer",
    ]:
        if column in answer_key.columns:
            answer_key_columns.append(column)

    scored = unscored.merge(
        answer_key[answer_key_columns],
        on=["exam", "question_id"],
        how="left",
    )

    scored["model_answer"] = scored[
        "model_answer"
    ].apply(normalize_answer)
    scored["correct_answer"] = scored[
        "correct_answer"
    ].apply(normalize_answer)

    official_flags: List[bool] = []
    statuses: List[str] = []
    correct_flags: List[Optional[bool]] = []
    blank_flags: List[bool] = []
    scores: List[Optional[float]] = []

    for _, row in scored.iterrows():
        has_official = official_answer_available(
            row
        )

        status, is_correct, is_blank, score = (
            score_answer(
                model_answer=normalize_answer(
                    row.get("model_answer")
                ),
                correct_answer=normalize_answer(
                    row.get("correct_answer")
                ),
                has_official_answer=has_official,
                parse_error=row.get(
                    "parse_error"
                ),
                api_error=(
                    None
                    if bool(
                        row.get("call_success")
                    )
                    else "call_failed"
                ),
            )
        )

        official_flags.append(has_official)
        statuses.append(status)
        correct_flags.append(is_correct)
        blank_flags.append(is_blank)
        scores.append(score)

    scored["has_official_answer"] = official_flags
    scored["scoring_status"] = statuses
    scored["is_correct"] = correct_flags
    scored["is_blank"] = blank_flags
    scored["exam_score"] = scores

    scored.to_csv(
        results_file,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding="utf-8-sig",
    )
    call_level.to_csv(
        calls_file,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding="utf-8-sig",
    )

    summary_rows: List[Dict[str, Any]] = []

    for replicate_id, group in scored.groupby(
        "replicate_id",
        sort=True,
    ):
        evaluable = group[
            group["has_official_answer"] == True
        ]
        valid = evaluable[
            evaluable["scoring_status"].isin(
                ["correct", "incorrect", "blank"]
            )
        ]
        call = call_level[
            call_level["replicate_id"]
            == replicate_id
        ].iloc[0]

        n_evaluable = len(evaluable)
        n_correct = int(
            (valid["scoring_status"] == "correct").sum()
        )
        n_incorrect = int(
            (
                valid["scoring_status"]
                == "incorrect"
            ).sum()
        )
        n_blank = int(
            (valid["scoring_status"] == "blank").sum()
        )

        summary_rows.append({
            "run_name": run_name,
            "replicate_id": replicate_id,
            "model_requested": MODEL_REQUESTED,
            "model_returned_unique": ";".join(
                sorted(
                    str(value)
                    for value in group[
                        "model_returned"
                    ].dropna().unique()
                )
            ),
            "n_total_questions_run": len(group),
            "n_evaluable_with_official_answer": (
                n_evaluable
            ),
            "n_valid_scored": len(valid),
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_blank": n_blank,
            "n_api_or_protocol_error": int(
                (
                    evaluable["scoring_status"]
                    == "api_or_protocol_error"
                ).sum()
            ),
            "n_missing_or_invalid": int(
                (
                    evaluable["scoring_status"]
                    == "missing_or_invalid"
                ).sum()
            ),
            "n_no_official_answer": int(
                (
                    group["has_official_answer"]
                    == False
                ).sum()
            ),
            "n_answer_records_returned": call[
                "n_answer_records_returned"
            ],
            "n_unique_valid_answer_records": call[
                "n_unique_valid_answer_records"
            ],
            "whole_response_parse_error": call[
                "whole_response_parse_error"
            ],
            "accuracy_among_answered_nonblank": (
                n_correct
                / (n_correct + n_incorrect)
                if (n_correct + n_incorrect)
                else None
            ),
            "accuracy_among_evaluable_questions_counting_blank_as_not_correct": (
                n_correct / n_evaluable
                if n_evaluable
                else None
            ),
            "official_exam_score_sum": valid[
                "exam_score"
            ].sum(skipna=True),
            "latency_seconds": call[
                "latency_seconds"
            ],
            "elapsed_seconds_including_retries": call[
                "elapsed_seconds_including_retries"
            ],
            "attempts_used": call[
                "attempts_used"
            ],
            "input_tokens": call["input_tokens"],
            "output_tokens": call[
                "output_tokens"
            ],
            "total_tokens": call["total_tokens"],
            "reasoning_tokens": call[
                "reasoning_tokens"
            ],
            "cached_input_tokens": call[
                "cached_input_tokens"
            ],
            "reasoning_content_nonempty": call[
                "reasoning_content_nonempty"
            ],
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        summary_file,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding="utf-8-sig",
    )

    discordances = build_discordances(scored)
    discordances.to_csv(
        discordances_file,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding="utf-8-sig",
    )

    expected_rows = len(questions) * args.replicates
    record_count_pass = len(scored) == expected_rows
    all_calls_successful = bool(
        call_level["call_success"].fillna(False).all()
    )
    all_reasoning_explicit_zero = bool(
        call_level["reasoning_tokens"].notna().all()
        and (
            call_level["reasoning_tokens"] == 0
        ).all()
    )
    no_reasoning_content = not bool(
        call_level[
            "reasoning_content_nonempty"
        ].fillna(False).astype(bool).any()
    )
    no_protocol_violations = not bool(
        call_level[
            "protocol_violation"
        ].fillna(False).astype(bool).any()
    )

    definitive_pass = (
        is_full_definitive_selection
        and record_count_pass
        and all_calls_successful
        and all_reasoning_explicit_zero
        and no_reasoning_content
        and no_protocol_violations
    )

    metadata["completed_utc"] = utc_timestamp()
    metadata[
        "definitive_run_completed_as_prespecified"
    ] = definitive_pass
    metadata["observed_reasoning_audit"] = {
        "n_calls_completed": int(
            len(call_level)
        ),
        "all_reasoning_token_values_explicit": bool(
            call_level[
                "reasoning_tokens"
            ].notna().all()
        ),
        "unique_reasoning_token_values": sorted(
            int(value)
            for value in call_level[
                "reasoning_tokens"
            ].dropna().unique()
        ),
        "total_reasoning_tokens": int(
            call_level[
                "reasoning_tokens"
            ].sum(skipna=True)
        ),
        "calls_with_positive_reasoning_tokens": int(
            (
                call_level[
                    "reasoning_tokens"
                ].fillna(0)
                > 0
            ).sum()
        ),
        "calls_with_nonempty_reasoning_content": int(
            call_level[
                "reasoning_content_nonempty"
            ].fillna(False).astype(bool).sum()
        ),
        "protocol_violations": int(
            call_level[
                "protocol_violation"
            ].fillna(False).astype(bool).sum()
        ),
    }
    metadata["output_files"] = {
        "raw_outputs": str(raw_file),
        "scored_results": str(
            results_file
        ),
        "summary_by_replicate": str(
            summary_file
        ),
        "call_level": str(calls_file),
        "discordances": str(
            discordances_file
        ),
        "prompt": str(prompt_file),
        "run_metadata": str(metadata_file),
    }

    metadata_file.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=== Summary ===")
    print(summary.to_string(index=False))
    print()
    print(
        "NON-REASONING AUDIT:",
        (
            "PASS"
            if all_reasoning_explicit_zero
            and no_reasoning_content
            and no_protocol_violations
            else "FAIL"
        ),
    )
    print(
        "EXPECTED RECORD COUNT:",
        "PASS" if record_count_pass else "FAIL",
        f"({len(scored)}/{expected_rows})",
    )
    print(
        "PRESPECIFIED 210 x 2 DEFINITIVE RUN:",
        "PASS" if definitive_pass else "FAIL",
    )
    print()
    print("Files written:")
    print(" ", raw_file)
    print(" ", results_file)
    print(" ", summary_file)
    print(" ", calls_file)
    print(" ", discordances_file)
    print(" ", prompt_file)
    print(" ", metadata_file)


if __name__ == "__main__":
    main()

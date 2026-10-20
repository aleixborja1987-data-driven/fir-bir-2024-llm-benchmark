"""
BIR 2024 — xAI Grok 4.3 item-wise, NON-REASONING, 2 replicates
================================================================

Definitive benchmark condition
------------------------------
- Model: grok-4.3
- Endpoint: xAI OpenAI-compatible Chat Completions
- One BIR question per API call
- Two independent replicates by default
- Text-only input
- Strict JSON Schema output with answer + short rationale
- Official answer key merged locally only after inference
- No tools, web search, X search, files, images, retrieval, or answer key
  are supplied to the model

Generation parameters
---------------------
- reasoning_effort = "none"
- temperature = 0
- top_p = 1
- max_tokens = 300
- stream = False
- response_format = strict json_schema
- rationale maximum length = 160 characters

Non-reasoning safeguard
-----------------------
Every successful response must report:
- completion_tokens_details.reasoning_tokens == 0
- empty/absent message.reasoning_content

The entire run aborts immediately if:
- positive reasoning tokens are observed;
- reasoning_content is non-empty; or
- the provider does not expose an explicit numeric reasoning-token count.

This strict requirement ensures that the definitive condition is empirically
verified as zero-reasoning rather than inferred only from the request parameter.

Connection safeguard
--------------------
truststore is injected before importing the OpenAI client so that the operating
system certificate store is used. Output paths are intentionally short for
Windows compatibility.

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
    python "run_bir2024_grok43_itemwise_NONREASONING.py" --dry-run

Pilot
-----
    python "run_bir2024_grok43_itemwise_NONREASONING.py" `
      --limit 10 `
      --replicates 1

Definitive run
--------------
    python "run_bir2024_grok43_itemwise_NONREASONING.py" `
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

CONDITION = "itemwise_text_rationale_nonthinking"
PROMPT_TEMPLATE_VERSION = (
    "bir2024_text_itemwise_examlike_blank_"
    "strict_json_short_rationale_nonthinking_v1"
)
SCRIPT_VERSION = (
    "bir2024_grok43_itemwise_nonthinking_"
    "strict_json_reasoning_audit_v1"
)

TEMPERATURE = 0
TOP_P = 1
MAX_TOKENS = 300
RATIONALE_MAX_CHARS = 160
STREAM = False

DEFAULT_REPLICATES = 2
REQUEST_TIMEOUT_SECONDS = 240.0
MAX_ATTEMPTS_PER_CALL = 5
BASE_RETRY_SLEEP_SECONDS = 5.0
MAX_RETRY_SLEEP_SECONDS = 60.0
BETWEEN_CALLS_SLEEP_SECONDS = 0.0

REQUIRE_EXPLICIT_ZERO_REASONING_TOKENS = True

QUESTIONS_FILE = Path("bir2024_questions_210.csv")
ANSWER_KEY_FILE = Path("bir2024_answer_key_210.csv")
OUTPUT_ROOT = Path("outputs")

VALID_ANSWERS = {"1", "2", "3", "4", "blank"}

ANSWER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": ["1", "2", "3", "4", "blank"],
        },
        "rationale": {
            "type": "string",
            "maxLength": RATIONALE_MAX_CHARS,
        },
    },
    "required": ["answer", "rationale"],
    "additionalProperties": False,
}

RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "bir_exam_answer",
        "strict": True,
        "schema": ANSWER_SCHEMA,
    },
}

GENERATION_PARAMETERS: Dict[str, Any] = {
    "reasoning_effort": "none",
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "max_tokens": MAX_TOKENS,
    "stream": STREAM,
    "response_format": RESPONSE_FORMAT,
    "rationale_max_chars": RATIONALE_MAX_CHARS,
    "tools_enabled": False,
    "web_search_enabled": False,
    "x_search_enabled": False,
    "file_input_enabled": False,
    "image_input_enabled": False,
    "retrieval_enabled": False,
    "truststore_status": TRUSTSTORE_STATUS,
}

PROMPT_PREAMBLE = """You are taking an examination for Biology residency access.

This is a multiple-choice exam. Each question has four options, numbered 1 to 4, and only one is correct. In the official exam, correct answers score +1 point, incorrect answers score -1/3 points, and blank answers score 0 points. Act as a human candidate and try to get the highest score possible.

Return exactly one JSON object matching the supplied schema.
The "answer" field must be exactly "1", "2", "3", "4", or "blank".
The "rationale" field must be brief and must not exceed 160 characters.
Do not use web search, X search, external references, files, images, retrieval, tools, or an answer key.
"""


# ============================================================
# Helpers
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


def safe_tokens_per_second(
    tokens: Optional[float],
    seconds: Optional[float],
) -> Optional[float]:
    if tokens is None or seconds is None or seconds <= 0:
        return None

    try:
        return float(tokens) / float(seconds)
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
# Inputs and prompt
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
        cached_tokens = prompt_details.get("cached_tokens")

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
        return None, None, getattr(
            choice,
            "finish_reason",
            None,
        )

    content = getattr(message, "content", None)
    reasoning_content = getattr(
        message,
        "reasoning_content",
        None,
    )
    finish_reason = getattr(
        choice,
        "finish_reason",
        None,
    )

    return content, reasoning_content, finish_reason


def parse_model_answer(
    raw_text: Optional[str],
) -> Tuple[
    Optional[str],
    Optional[str],
    Optional[str],
]:
    if not raw_text:
        return None, None, "missing_raw_output"

    try:
        data = json.loads(raw_text)
    except Exception as exc:
        return (
            None,
            None,
            f"json_decode_error:{type(exc).__name__}:{exc}",
        )

    if not isinstance(data, dict):
        return None, None, "json_not_object"

    if set(data.keys()) != {"answer", "rationale"}:
        return (
            None,
            data.get("rationale"),
            "unexpected_json_fields",
        )

    answer = normalize_answer(data.get("answer"))
    rationale = data.get("rationale")

    if answer not in VALID_ANSWERS:
        return answer, rationale, "invalid_answer"

    if not isinstance(rationale, str):
        return answer, rationale, "invalid_rationale_type"

    if len(rationale) > RATIONALE_MAX_CHARS:
        return answer, rationale, "rationale_too_long"

    return answer, rationale, None


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
    prompt: str,
) -> Dict[str, Any]:
    started = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL_REQUESTED,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        reasoning_effort="none",
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        response_format=RESPONSE_FORMAT,
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

    parsed_answer, rationale, parse_error = (
        parse_model_answer(raw_text)
    )

    reasoning_tokens = usage_flat["reasoning_tokens"]
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

    return {
        "model_returned": getattr(
            response,
            "model",
            None,
        ),
        "response_id": getattr(response, "id", None),
        "finish_reason": finish_reason,
        "raw_output": raw_text,
        "parsed_answer": parsed_answer,
        "rationale": rationale,
        "parse_error": parse_error,
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
        "output_tokens_per_second": (
            safe_tokens_per_second(
                usage_flat["output_tokens"],
                latency_seconds,
            )
        ),
        "total_tokens_per_second": (
            safe_tokens_per_second(
                usage_flat["total_tokens"],
                latency_seconds,
            )
        ),
    }


def call_grok_with_retries(
    client: OpenAI,
    prompt: str,
) -> Dict[str, Any]:
    overall_started = time.perf_counter()
    attempt_log: List[Dict[str, Any]] = []
    last_result: Optional[Dict[str, Any]] = None

    for attempt in range(
        1,
        MAX_ATTEMPTS_PER_CALL + 1,
    ):
        try:
            result = call_grok_once(client, prompt)
            last_result = result

            if result["protocol_violation"]:
                return {
                    **result,
                    "call_success": False,
                    "error_type": (
                        "NonReasoningProtocolViolation"
                    ),
                    "error_message": (
                        "Expected explicit zero reasoning tokens "
                        "and empty reasoning_content."
                    ),
                    "attempts_used": attempt,
                    "attempt_log": attempt_log,
                    "elapsed_seconds_including_retries": (
                        time.perf_counter()
                        - overall_started
                    ),
                }

            if result["parse_error"] is None:
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

            retry = attempt < MAX_ATTEMPTS_PER_CALL
            attempt_log.append({
                "attempt": attempt,
                "error_type": (
                    "StructuredOutputValidationError"
                ),
                "error_message": result["parse_error"],
                "retry": retry,
            })

            if retry:
                time.sleep(retry_sleep_seconds(attempt))
                continue

        except Exception as exc:
            retry = (
                is_transient_exception(exc)
                and attempt < MAX_ATTEMPTS_PER_CALL
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
                time.sleep(retry_sleep_seconds(attempt))
                continue

            return {
                "call_success": False,
                "protocol_violation": False,
                "model_returned": None,
                "response_id": None,
                "finish_reason": None,
                "raw_output": None,
                "parsed_answer": None,
                "rationale": None,
                "parse_error": None,
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
                "output_tokens_per_second": None,
                "total_tokens_per_second": None,
            }

    assert last_result is not None

    return {
        **last_result,
        "call_success": False,
        "error_type": "StructuredOutputValidationError",
        "error_message": last_result["parse_error"],
        "attempts_used": MAX_ATTEMPTS_PER_CALL,
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


def score_row(
    row: pd.Series,
) -> Tuple[
    str,
    Optional[bool],
    bool,
    Optional[float],
]:
    has_official = official_answer_available(row)
    answer = normalize_answer(row.get("parsed_answer"))
    correct = normalize_answer(row.get("correct_answer"))

    if not bool_like(row.get("call_success")):
        return (
            "api_or_protocol_error",
            None if not has_official else False,
            False,
            0.0 if has_official else None,
        )

    parse_error = row.get("parse_error")
    if (
        parse_error is not None
        and str(parse_error).strip()
        not in {"", "None", "nan"}
    ):
        return (
            "parse_error",
            None if not has_official else False,
            False,
            0.0 if has_official else None,
        )

    if not has_official:
        return (
            "no_official_answer",
            None,
            answer == "blank",
            None,
        )

    if answer == "blank":
        return "blank", False, True, 0.0

    if answer == correct:
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

        if first["parsed_answer"] == second["parsed_answer"]:
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
                "parsed_answer"
            ],
            "replicate_1_status": first[
                "scoring_status"
            ],
            "replicate_1_score": first["exam_score"],
            "replicate_2_answer": second[
                "parsed_answer"
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

    is_full_definitive_selection = (
        args.limit is None
        and args.start_q is None
        and args.end_q is None
        and len(questions) == 210
        and args.replicates == 2
    )

    print(
        "=== BIR 2024 — Grok 4.3 item-wise "
        "NON-REASONING ==="
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
    print("Strict JSON Schema:", True)
    print("Truststore:", TRUSTSTORE_STATUS)
    print(
        "Require explicit zero reasoning tokens:",
        REQUIRE_EXPLICIT_ZERO_REASONING_TOKENS,
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
        selection_parts.append(f"limit{args.limit}")
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
        f"bir24_grok43_item_nr_{args.replicates}r"
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
    discordances_file = (
        output_dir / "discordances.csv"
    )
    metadata_file = output_dir / "metadata.json"

    planned_paths = [
        raw_file,
        results_file,
        summary_file,
        discordances_file,
        metadata_file,
    ]

    if os.name == "nt":
        longest = max(
            len(str(path.resolve()))
            for path in planned_paths
        )
        print(
            "Longest planned output path:",
            longest,
            "characters",
        )
        if longest >= 240:
            raise OSError(
                "Ruta de salida demasiado larga para Windows."
            )

    prompt_preamble_hash = sha256_text(
        PROMPT_PREAMBLE
    )
    questions_hash = sha256_file(QUESTIONS_FILE)
    answer_key_hash = sha256_file(ANSWER_KEY_FILE)

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
        "generation_parameters": (
            GENERATION_PARAMETERS
        ),
        "answer_schema": ANSWER_SCHEMA,
        "reasoning_audit_rule": {
            "reasoning_effort_sent": "none",
            "require_explicit_numeric_reasoning_tokens": True,
            "required_reasoning_tokens_per_call": 0,
            "require_empty_reasoning_content": True,
            "abort_on_violation": True,
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
        "prompt_preamble": PROMPT_PREAMBLE,
        "prompt_preamble_sha256": (
            prompt_preamble_hash
        ),
        "retry_policy": {
            "max_attempts_per_call": (
                MAX_ATTEMPTS_PER_CALL
            ),
            "retry_transient_api_errors": True,
            "retry_invalid_structured_outputs": True,
            "retry_valid_but_incorrect_answers": False,
        },
        "tools_or_web_search_provided_to_model": False,
        "notes": (
            "The official answer key is merged only after "
            "inference for local scoring. No tools, web "
            "search, X search, files, images, retrieval, or "
            "answer key are included in API requests."
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
    raw_file.write_text("", encoding="utf-8")

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    total_calls = len(questions) * args.replicates
    call_index = 0
    flat_rows: List[Dict[str, Any]] = []

    print()
    print("Run name:", run_name)
    print("Output directory:", output_dir)
    print("Total planned API calls:", total_calls)

    for replicate_id in range(
        1,
        args.replicates + 1,
    ):
        for _, row in questions.iterrows():
            call_index += 1
            prompt = build_prompt(row)

            print(
                f"[{call_index}/{total_calls}] "
                f"rep={replicate_id} "
                f"{row['question_id']} "
                f"(Q{int(row['original_question_number'])})",
                flush=True,
            )

            request_time = utc_iso()
            result = call_grok_with_retries(
                client=client,
                prompt=prompt,
            )
            response_time = utc_iso()

            base_record = {
                "run_id": (
                    f"{timestamp}_rep{replicate_id}_"
                    f"{row['question_id']}_grok43"
                ),
                "run_name": run_name,
                "replicate_id": replicate_id,
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
                "generation_parameters": (
                    GENERATION_PARAMETERS
                ),
                "exam": row["exam"],
                "question_id": row["question_id"],
                "original_question_number": int(
                    row["original_question_number"]
                ),
                "is_reserve": bool_like(
                    row.get("is_reserve")
                ),
                "subject": row.get("subject"),
                "has_image": bool_like(
                    row.get("has_image")
                ),
                "source_pdf_page": safe_int(
                    row.get("source_pdf_page")
                ),
                "question_text": row["question_text"],
                "option_1": row["option_1"],
                "option_2": row["option_2"],
                "option_3": row["option_3"],
                "option_4": row["option_4"],
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "prompt_preamble_sha256": (
                    prompt_preamble_hash
                ),
                "questions_file_sha256": questions_hash,
                "answer_key_file_sha256": (
                    answer_key_hash
                ),
                "request_time_utc": request_time,
                "response_time_utc": response_time,
                "reasoning_effort": "none",
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
                "stream": STREAM,
                "tools_enabled": False,
                "web_search_enabled": False,
                "x_search_enabled": False,
            }

            raw_record = {
                **base_record,
                **result,
            }
            append_jsonl(raw_file, raw_record)

            flat_rows.append({
                **base_record,
                "call_success": result.get(
                    "call_success"
                ),
                "model_returned": result.get(
                    "model_returned"
                ),
                "response_id": result.get(
                    "response_id"
                ),
                "finish_reason": result.get(
                    "finish_reason"
                ),
                "raw_output": result.get(
                    "raw_output"
                ),
                "parsed_answer": result.get(
                    "parsed_answer"
                ),
                "rationale": result.get("rationale"),
                "parse_error": result.get(
                    "parse_error"
                ),
                "protocol_violation": result.get(
                    "protocol_violation"
                ),
                "reasoning_content_nonempty": (
                    result.get(
                        "reasoning_content_nonempty"
                    )
                ),
                "missing_reasoning_audit": result.get(
                    "missing_reasoning_audit"
                ),
                "positive_reasoning_tokens": (
                    result.get(
                        "positive_reasoning_tokens"
                    )
                ),
                "error_type": result.get("error_type"),
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
                "reasoning_tokens": result.get(
                    "reasoning_tokens"
                ),
                "cached_input_tokens": result.get(
                    "cached_input_tokens"
                ),
                "output_tokens_per_second": result.get(
                    "output_tokens_per_second"
                ),
                "total_tokens_per_second": result.get(
                    "total_tokens_per_second"
                ),
            })

            print(
                "    success="
                f"{result.get('call_success')} "
                "answer="
                f"{result.get('parsed_answer')} "
                "reasoning_tokens="
                f"{result.get('reasoning_tokens')} "
                "attempts="
                f"{result.get('attempts_used')} "
                "model="
                f"{result.get('model_returned')}",
                flush=True,
            )

            if result.get("protocol_violation"):
                partial = pd.DataFrame(flat_rows)
                partial.to_csv(
                    results_file,
                    index=False,
                    quoting=csv.QUOTE_ALL,
                    encoding="utf-8-sig",
                )

                metadata["completed_utc"] = utc_timestamp()
                metadata[
                    "aborted_due_to_reasoning_protocol_violation"
                ] = True
                metadata[
                    "definitive_run_completed_as_prespecified"
                ] = False
                metadata_file.write_text(
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                print(
                    "RUN ABORTED: non-reasoning protocol "
                    "violation.",
                    flush=True,
                )
                sys.exit(3)

            if BETWEEN_CALLS_SLEEP_SECONDS > 0:
                time.sleep(BETWEEN_CALLS_SLEEP_SECONDS)

    results = pd.DataFrame(flat_rows)

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

    scored = results.merge(
        answer_key[answer_key_columns],
        on=["exam", "question_id"],
        how="left",
    )

    scored["parsed_answer"] = scored[
        "parsed_answer"
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
        has_official = official_answer_available(row)
        status, is_correct, is_blank, score = (
            score_row(row)
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
                    group["scoring_status"]
                    == "api_or_protocol_error"
                ).sum()
            ),
            "n_parse_error": int(
                (
                    group["scoring_status"]
                    == "parse_error"
                ).sum()
            ),
            "n_no_official_answer": int(
                (
                    group["has_official_answer"]
                    == False
                ).sum()
            ),
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
            "mean_latency_seconds": group[
                "latency_seconds"
            ].mean(),
            "median_latency_seconds": group[
                "latency_seconds"
            ].median(),
            "total_input_tokens": group[
                "input_tokens"
            ].sum(skipna=True),
            "total_output_tokens": group[
                "output_tokens"
            ].sum(skipna=True),
            "total_tokens": group[
                "total_tokens"
            ].sum(skipna=True),
            "total_reasoning_tokens": group[
                "reasoning_tokens"
            ].sum(skipna=True),
            "calls_with_positive_reasoning_tokens": int(
                group[
                    "positive_reasoning_tokens"
                ].fillna(False).astype(bool).sum()
            ),
            "calls_with_missing_reasoning_audit": int(
                group[
                    "missing_reasoning_audit"
                ].fillna(False).astype(bool).sum()
            ),
            "calls_with_reasoning_content": int(
                group[
                    "reasoning_content_nonempty"
                ].fillna(False).astype(bool).sum()
            ),
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
    all_calls_successful = bool(
        scored["call_success"].fillna(False).all()
    )
    all_reasoning_explicit_zero = bool(
        scored["reasoning_tokens"].notna().all()
        and (scored["reasoning_tokens"] == 0).all()
    )
    no_reasoning_content = not bool(
        scored[
            "reasoning_content_nonempty"
        ].fillna(False).astype(bool).any()
    )
    record_count_pass = len(scored) == expected_rows

    definitive_pass = (
        is_full_definitive_selection
        and record_count_pass
        and all_calls_successful
        and all_reasoning_explicit_zero
        and no_reasoning_content
    )

    metadata["completed_utc"] = utc_timestamp()
    metadata[
        "definitive_run_completed_as_prespecified"
    ] = definitive_pass
    metadata["observed_reasoning_audit"] = {
        "n_calls_completed": int(len(scored)),
        "all_reasoning_token_values_explicit": bool(
            scored["reasoning_tokens"].notna().all()
        ),
        "unique_reasoning_token_values": sorted(
            int(value)
            for value in scored[
                "reasoning_tokens"
            ].dropna().unique()
        ),
        "total_reasoning_tokens": int(
            scored["reasoning_tokens"].sum(
                skipna=True
            )
        ),
        "calls_with_positive_reasoning_tokens": int(
            (
                scored["reasoning_tokens"].fillna(0)
                > 0
            ).sum()
        ),
        "calls_with_nonempty_reasoning_content": int(
            scored[
                "reasoning_content_nonempty"
            ].fillna(False).astype(bool).sum()
        ),
        "protocol_violations": int(
            scored[
                "protocol_violation"
            ].fillna(False).astype(bool).sum()
        ),
    }
    metadata["output_files"] = {
        "raw_outputs": str(raw_file),
        "scored_results": str(results_file),
        "summary_by_replicate": str(summary_file),
        "discordances": str(discordances_file),
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
    print(" ", discordances_file)
    print(" ", metadata_file)


if __name__ == "__main__":
    main()

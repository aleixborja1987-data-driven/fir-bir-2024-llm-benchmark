"""
BIR 2024 — Claude Opus 4.8 whole-exam, answers-only, NON-THINKING + TRUSTSTORE
=================================================================

Definitive benchmark:
- 210 BIR questions in one text-only request per replicate.
- 2 independent replicates by default.
- Answers-only output through a forced client-side tool schema.
- No rationales.
- No web search, retrieval, files, images, external tools, or answer key
  are supplied during inference.
- The official answer key is merged locally only after generation.

Claude Opus 4.8 configuration
-----------------------------
- model: claude-opus-4-8
- thinking: OMITTED
  On Claude Opus 4.8, thinking is off unless thinking={"type":"adaptive"}
  is explicitly sent.
- output_config / effort: OMITTED
- temperature: OMITTED
- top_p: OMITTED
- top_k: OMITTED
  Claude Opus 4.8 does not support non-default sampling parameters. Omitting
  them matches the three already completed Opus 4.8 benchmark conditions.
- max_tokens: 10000
- stream: False
- structured output: forced record_all_answers tool schema

Connection safeguard
--------------------
- truststore.inject_into_ssl() is called before importing Anthropic.
- The operating-system certificate store is therefore used by the SDK.
- This is intended to avoid intermittent APIConnectionError events observed
  behind a firewall performing HTTPS inspection.
- Output folders and filenames are intentionally short.
- On Windows, all planned output paths are checked before any API call; the
  run aborts locally if the longest path is 240 characters or more.

Protocol safeguards
-------------------
- Abort the entire run if any `thinking` content block is returned.
- Abort if the API reports a positive thinking-token count.
- Retry only transient API failures or invalid/incomplete structured outputs.
- Never retry a valid response because its answers are incorrect.
- Require exactly one valid answer for every question in the prompt.
- Require the definitive run to contain 210 questions x 2 replicates.

Expected local files
--------------------
- .env with ANTHROPIC_API_KEY=...
- bir2024_questions_210.csv
- bir2024_answer_key_210.csv

Install
-------
    pip install --upgrade anthropic pandas python-dotenv truststore

Optional dry run, without API expenditure
-----------------------------------------
    python "run_bir2024_opus48_whole_TRUSTSTORE_SHORTPATHS.py" `
      --dry-run

Small structural pilot
----------------------
    python "run_bir2024_opus48_whole_TRUSTSTORE_SHORTPATHS.py" `
      --limit 10 `
      --replicates 1

Definitive run
--------------
    python "run_bir2024_opus48_whole_TRUSTSTORE_SHORTPATHS.py"
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
        "Falta el paquete truststore. Instálalo con: pip install truststore"
    ) from exc

# Must be injected before importing/creating the Anthropic client so that
# httpx/OpenSSL use the operating-system certificate store. This is especially
# relevant on networks with HTTPS inspection.
truststore.inject_into_ssl()
TRUSTSTORE_STATUS = "enabled"

import anthropic
import pandas as pd
from dotenv import load_dotenv


# ============================================================
# Study configuration
# ============================================================

MODEL_REQUESTED = "claude-opus-4-8"
PROVIDER = "Anthropic Claude"
INTERFACE = "API"
API_ENDPOINT = "Messages API with forced client-side tool schema"

CONDITION = "whole_text_answers_only_nonthinking"
PROMPT_TEMPLATE_VERSION = (
    "bir2024_whole_text_examlike_blank_answers_only_tool_schema_v1"
)
SCRIPT_VERSION = (
    "bir2024_claude_opus48_whole_text_answers_only_"
    "tool_schema_no_sampling_params_reasoning_audit_truststore_shortpaths_v3"
)

GENERATION_PARAMETERS: Dict[str, Any] = {
    "thinking": "not_sent",
    "thinking_policy": "disabled_by_omission_for_claude_opus_4_8",
    "output_config_effort": "not_sent",
    "temperature": "not_sent",
    "top_p": "not_sent",
    "top_k": "not_sent",
    "sampling_policy": (
        "provider_default; Opus 4.8 rejects non-default sampling values"
    ),
    "max_tokens": 10000,
    "structured_output": "forced_tool_use_input_schema",
    "rationales_requested": False,
    "stream": False,
    "external_tools_enabled": False,
    "web_search_enabled": False,
    "file_input_enabled": False,
    "image_input_enabled": False,
    "truststore_status": TRUSTSTORE_STATUS,
}

REQUEST_TIMEOUT_SECONDS = 600.0
MAX_TOTAL_ATTEMPTS_PER_REPLICATE = 3
BASE_SLEEP_SECONDS = 20.0
MAX_BACKOFF_SECONDS = 120.0
BETWEEN_REPLICATES_SLEEP_SECONDS = 2.0

QUESTIONS_FILE = Path("bir2024_questions_210.csv")
ANSWER_KEY_FILE = Path("bir2024_answer_key_210.csv")
OUTPUT_ROOT = Path("outputs")

VALID_ANSWERS = {"1", "2", "3", "4", "blank"}


PROMPT_PREAMBLE = """You are taking an examination for Biology residency access.

This is a multiple-choice exam. Each question has four options, numbered 1 to 4, and only one is correct. In the official exam, correct answers score +1 point, incorrect answers score -1/3 points, and blank answers score 0 points. Act as a human candidate and try to get the highest score possible.

You will receive the full exam text, containing all questions and their four answer options.

Do not use web search, external references, files, images, retrieval, or an answer key.

Use the record_all_answers tool.

Tool input rules:
- Return exactly one answer object for every question included in the input.
- Use the integer question number shown in the input.
- Preserve the same question order as in the input.
- The value of "answer" must be exactly one single string: "1", "2", "3", "4", or "blank".
- Do not return multiple options.
- If you cannot choose one single best option, return "blank".
- Do not include rationales, explanations, comments, markdown, or any other fields.

FULL EXAM TEXT:
"""


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


def backoff_seconds(attempt_number: int) -> float:
    exponential = BASE_SLEEP_SECONDS * (
        2 ** max(0, attempt_number - 1)
    )
    return min(exponential, MAX_BACKOFF_SECONDS) + random.uniform(
        0.0,
        0.75,
    )


# ============================================================
# Input and prompt construction
# ============================================================

def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"No existe {QUESTIONS_FILE}")

    if not ANSWER_KEY_FILE.exists():
        raise FileNotFoundError(f"No existe {ANSWER_KEY_FILE}")

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
    missing_key = required_key_columns - set(answer_key.columns)

    if missing_questions:
        raise ValueError(
            "Faltan columnas en el fichero de preguntas: "
            f"{sorted(missing_questions)}"
        )

    if missing_key:
        raise ValueError(
            "Faltan columnas en el answer key: "
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
    question_number = int(row["original_question_number"])

    return f"""Question {question_number}:
{row['question_text']}

Options:
1. {row['option_1']}
2. {row['option_2']}
3. {row['option_3']}
4. {row['option_4']}
"""


def build_full_prompt(questions: pd.DataFrame) -> str:
    parts: List[str] = [PROMPT_PREAMBLE, "\n\n"]

    for index, (_, row) in enumerate(
        questions.iterrows(),
        start=1,
    ):
        parts.append(format_question(row))

        if index < len(questions):
            parts.append("\n---\n\n")

    return "".join(parts)


def build_answer_tool(n_questions: int) -> Dict[str, Any]:
    return {
        "name": "record_all_answers",
        "description": (
            "Record all answers to a multiple-choice examination."
        ),
        "input_schema": {
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
        },
    }


# ============================================================
# Response and reasoning audit
# ============================================================

def extract_thinking_token_count(
    usage_dict: Dict[str, Any],
) -> Optional[int]:
    """
    Supports both current and legacy-like response shapes.

    Expected observed non-thinking result:
    - no thinking blocks;
    - thinking_tokens absent or equal to zero.
    """
    if not isinstance(usage_dict, dict):
        return None

    direct = usage_dict.get("thinking_tokens")
    if direct is not None:
        return safe_int(direct)

    output_details = usage_dict.get("output_tokens_details")
    if isinstance(output_details, dict):
        value = output_details.get("thinking_tokens")
        if value is not None:
            return safe_int(value)

    completion_details = usage_dict.get(
        "completion_tokens_details"
    )
    if isinstance(completion_details, dict):
        value = completion_details.get("reasoning_tokens")
        if value is not None:
            return safe_int(value)

    return None


def flatten_usage(
    usage_dict: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(usage_dict, dict):
        usage_dict = {}

    input_tokens = usage_dict.get("input_tokens")
    output_tokens = usage_dict.get("output_tokens")
    cache_creation = usage_dict.get(
        "cache_creation_input_tokens"
    )
    cache_read = usage_dict.get("cache_read_input_tokens")
    thinking_tokens = extract_thinking_token_count(
        usage_dict
    )

    try:
        total_tokens = int(input_tokens or 0) + int(
            output_tokens or 0
        )
    except Exception:
        total_tokens = None

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "thinking_tokens": thinking_tokens,
        "reasoning_tokens": thinking_tokens,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
    }


def audit_thinking_blocks(response: Any) -> Dict[str, Any]:
    blocks = getattr(response, "content", None) or []
    thinking_blocks: List[Dict[str, Any]] = []

    for block in blocks:
        if getattr(block, "type", None) == "thinking":
            thinking_blocks.append(object_to_dict(block))

    nonempty_thinking_text = False

    for block in thinking_blocks:
        text = block.get("thinking")
        if isinstance(text, str) and text.strip():
            nonempty_thinking_text = True
            break

    return {
        "n_thinking_blocks": len(thinking_blocks),
        "thinking_blocks_present": bool(thinking_blocks),
        "thinking_content_nonempty": nonempty_thinking_text,
        "thinking_blocks": thinking_blocks,
    }


def parse_tool_answers(
    response: Any,
    expected_question_numbers: List[int],
) -> Tuple[
    Dict[int, Dict[str, Any]],
    Optional[str],
    Dict[str, Any],
]:
    tool_input = None
    n_matching_tool_blocks = 0

    for block in getattr(response, "content", None) or []:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None)
            == "record_all_answers"
        ):
            n_matching_tool_blocks += 1
            if tool_input is None:
                tool_input = getattr(block, "input", None)

    if tool_input is None:
        return {}, "missing_record_all_answers_tool_use", {
            "n_matching_tool_blocks": n_matching_tool_blocks,
        }

    if n_matching_tool_blocks != 1:
        return {}, (
            "unexpected_record_all_answers_tool_count:"
            f"{n_matching_tool_blocks}"
        ), {
            "n_matching_tool_blocks": n_matching_tool_blocks,
            "raw_tool_input": tool_input,
        }

    if not isinstance(tool_input, dict):
        return {}, "tool_input_not_object", {
            "raw_tool_input": tool_input,
        }

    answers = tool_input.get("answers")

    if not isinstance(answers, list):
        return {}, "answers_missing_or_not_list", {
            "raw_tool_input": tool_input,
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
        answer = normalize_answer(item.get("answer"))

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

    missing_numbers = sorted(expected_set - returned_set)
    unexpected_numbers = sorted(returned_set - expected_set)

    diagnostics = {
        "raw_tool_input": tool_input,
        "n_matching_tool_blocks": n_matching_tool_blocks,
        "n_answer_records_returned": len(answers),
        "n_unique_valid_answer_records": len(records),
        "duplicate_question_numbers": sorted(
            set(duplicates)
        ),
        "missing_question_numbers": missing_numbers,
        "unexpected_question_numbers": unexpected_numbers,
        "invalid_records": invalid_records,
        "returned_question_numbers": returned_numbers,
    }

    errors: List[str] = []

    if len(answers) != len(expected_question_numbers):
        errors.append(
            "wrong_answer_record_count:"
            f"{len(answers)}/"
            f"{len(expected_question_numbers)}"
        )

    if duplicates:
        errors.append(
            "duplicate_question_numbers:"
            + ",".join(map(str, sorted(set(duplicates))))
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


def call_claude_once(
    client: anthropic.Anthropic,
    full_prompt: str,
    answer_tool: Dict[str, Any],
    expected_question_numbers: List[int],
) -> Dict[str, Any]:
    started = time.perf_counter()

    # Intentionally NOT sent:
    # - thinking
    # - output_config / effort
    # - temperature
    # - top_p
    # - top_k
    response = client.messages.create(
        model=MODEL_REQUESTED,
        max_tokens=GENERATION_PARAMETERS["max_tokens"],
        messages=[
            {
                "role": "user",
                "content": full_prompt,
            }
        ],
        tools=[answer_tool],
        tool_choice={
            "type": "tool",
            "name": "record_all_answers",
        },
    )

    latency_seconds = time.perf_counter() - started

    usage_dict = object_to_dict(
        getattr(response, "usage", None)
    )
    usage_flat = flatten_usage(usage_dict)
    thinking_audit = audit_thinking_blocks(response)

    thinking_tokens = usage_flat.get("thinking_tokens")
    positive_thinking_tokens = (
        thinking_tokens is not None
        and thinking_tokens > 0
    )

    protocol_violation = (
        thinking_audit["thinking_blocks_present"]
        or positive_thinking_tokens
    )

    answer_records, parse_error, diagnostics = (
        parse_tool_answers(
            response=response,
            expected_question_numbers=(
                expected_question_numbers
            ),
        )
    )

    stop_reason = getattr(response, "stop_reason", None)

    if stop_reason != "tool_use":
        additional = f"unexpected_stop_reason:{stop_reason}"
        parse_error = (
            f"{parse_error}; {additional}"
            if parse_error
            else additional
        )

    return {
        "model_returned": getattr(
            response,
            "model",
            None,
        ),
        "response_id": getattr(response, "id", None),
        "stop_reason": stop_reason,
        "answer_records": answer_records,
        "whole_response_parse_error": parse_error,
        "diagnostics": diagnostics,
        "raw_response": object_to_dict(response),
        "usage": usage_dict,
        "latency_seconds": latency_seconds,
        "protocol_violation": protocol_violation,
        "positive_thinking_tokens": (
            positive_thinking_tokens
        ),
        **thinking_audit,
        **usage_flat,
    }


def call_claude_with_retries(
    client: anthropic.Anthropic,
    full_prompt: str,
    answer_tool: Dict[str, Any],
    expected_question_numbers: List[int],
) -> Dict[str, Any]:
    overall_started = time.perf_counter()
    attempt_log: List[Dict[str, Any]] = []
    last_result: Optional[Dict[str, Any]] = None

    for attempt in range(
        1,
        MAX_TOTAL_ATTEMPTS_PER_REPLICATE + 1,
    ):
        try:
            result = call_claude_once(
                client=client,
                full_prompt=full_prompt,
                answer_tool=answer_tool,
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
                        "UnexpectedThinkingProtocolViolation"
                    ),
                    "error_message": (
                        "Claude Opus 4.8 returned a thinking "
                        "block or positive thinking-token count "
                        "despite the thinking field being omitted."
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
                < MAX_TOTAL_ATTEMPTS_PER_REPLICATE
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
                time.sleep(backoff_seconds(attempt))
                continue

        except Exception as exc:
            retry = (
                is_transient_exception(exc)
                and attempt
                < MAX_TOTAL_ATTEMPTS_PER_REPLICATE
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
                time.sleep(backoff_seconds(attempt))
                continue

            return {
                "call_success": False,
                "protocol_violation": False,
                "model_returned": None,
                "response_id": None,
                "stop_reason": None,
                "answer_records": {},
                "whole_response_parse_error": None,
                "diagnostics": {},
                "raw_response": None,
                "usage": {},
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
                "n_thinking_blocks": 0,
                "thinking_blocks_present": False,
                "thinking_content_nonempty": False,
                "thinking_blocks": [],
                "positive_thinking_tokens": False,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "thinking_tokens": None,
                "reasoning_tokens": None,
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None,
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
            MAX_TOTAL_ATTEMPTS_PER_REPLICATE
        ),
        "attempt_log": attempt_log,
        "elapsed_seconds_including_retries": (
            time.perf_counter() - overall_started
        ),
    }


# ============================================================
# Scoring and summaries
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
) -> Tuple[str, Optional[bool], bool, Optional[float]]:
    if api_error:
        return (
            "api_error",
            None,
            False,
            0.0 if has_official_answer else None,
        )

    if parse_error:
        return (
            "missing_or_invalid",
            False if has_official_answer else None,
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
            "is_reserve": bool(first["is_reserve"]),
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
        help="Piloto estructural con las primeras N preguntas.",
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
        default=2,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Valida entradas, prompt y esquema sin llamar "
            "a la API."
        ),
    )

    args = parser.parse_args()

    if args.replicates < 1:
        raise ValueError("--replicates debe ser >= 1")

    load_dotenv(dotenv_path=".env")

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
    answer_tool = build_answer_tool(len(questions))

    is_full_definitive_selection = (
        args.limit is None
        and args.start_q is None
        and args.end_q is None
        and len(questions) == 210
        and args.replicates == 2
    )

    print(
        "=== BIR 2024 — Claude Opus 4.8 whole-exam "
        "answers-only NON-THINKING ==="
    )
    print("Questions included:", len(questions))
    print("Replicates:", args.replicates)
    print(
        "Full definitive selection:",
        is_full_definitive_selection,
    )
    print("Thinking field sent: False")
    print("Output effort sent: False")
    print("Temperature sent: False")
    print("Top_p sent: False")
    print("Top_k sent: False")
    print("Images sent: 0")
    print("Truststore status:", TRUSTSTORE_STATUS)
    print(
        "Structured output: forced "
        "record_all_answers tool schema"
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

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No encuentro ANTHROPIC_API_KEY en .env"
        )

    timestamp = utc_timestamp()
    safe_model = MODEL_REQUESTED.replace(
        ".",
        "_",
    ).replace("-", "_")

    selection_suffix_parts: List[str] = []

    if args.limit is not None:
        selection_suffix_parts.append(
            f"limit{args.limit}"
        )

    if (
        args.start_q is not None
        or args.end_q is not None
    ):
        selection_suffix_parts.append(
            f"q{args.start_q or 'start'}-"
            f"{args.end_q or 'end'}"
        )

    selection_suffix = (
        "_"
        + "_".join(selection_suffix_parts)
        if selection_suffix_parts
        else ""
    )

    # Short Windows-safe run and file names.
    # The scientific condition is recorded in metadata; it does not need to be
    # repeated in every filesystem path.
    run_name = (
        f"bir24_opus48_whole_trust_{args.replicates}r"
        f"{selection_suffix}_{timestamp}"
    )

    output_dir = OUTPUT_ROOT / run_name
    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    raw_file = output_dir / "raw.jsonl"
    scored_file = output_dir / "results.csv"
    summary_file = output_dir / "summary.csv"
    call_level_file = output_dir / "calls.csv"
    discordances_file = output_dir / "discordances.csv"
    metadata_file = output_dir / "metadata.json"
    prompt_log_file = output_dir / "prompt.txt"

    # Prevent spending API calls if Windows would reject one of the output
    # paths. The conservative 240-character threshold leaves headroom below
    # the traditional MAX_PATH limit.
    planned_output_paths = [
        raw_file,
        scored_file,
        summary_file,
        call_level_file,
        discordances_file,
        metadata_file,
        prompt_log_file,
    ]

    if os.name == "nt":
        resolved_lengths = {
            str(path.resolve()): len(str(path.resolve()))
            for path in planned_output_paths
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
                "Ruta de salida demasiado larga para una ejecución segura en "
                f"Windows ({longest_length} caracteres): {longest_path}"
            )

    raw_file.write_text("", encoding="utf-8")
    prompt_log_file.write_text(
        full_prompt,
        encoding="utf-8",
    )

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

    answer_key_min = answer_key[
        answer_key_columns
    ].copy()

    metadata: Dict[str, Any] = {
        "run_name": run_name,
        "created_utc": timestamp,
        "provider": PROVIDER,
        "interface": INTERFACE,
        "api_endpoint": API_ENDPOINT,
        "model_requested": MODEL_REQUESTED,
        "condition": CONDITION,
        "truststore_status": TRUSTSTORE_STATUS,
        "truststore_note": (
            "truststore.inject_into_ssl() was called before importing and "
            "creating the Anthropic client so that the operating-system "
            "certificate store is used. This was added after repeated "
            "APIConnectionError events on a network using HTTPS inspection."
        ),
        "prompt_template_version": (
            PROMPT_TEMPLATE_VERSION
        ),
        "script_version": SCRIPT_VERSION,
        "generation_parameters": (
            GENERATION_PARAMETERS
        ),
        "connection_configuration": {
            "truststore_injected_before_anthropic_import": True,
            "truststore_status": TRUSTSTORE_STATUS,
            "target_host": "api.anthropic.com",
        },
        "thinking_audit_rule": {
            "thinking_parameter_sent": False,
            "abort_if_any_thinking_block_returned": True,
            "abort_if_positive_thinking_tokens_reported": True,
            "expected_for_opus_4_8": (
                "No thinking blocks and zero or absent "
                "thinking-token count."
            ),
        },
        "answer_tool_schema": answer_tool,
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
        "is_full_definitive_selection": (
            is_full_definitive_selection
        ),
        "selection": {
            "limit": args.limit,
            "start_q": args.start_q,
            "end_q": args.end_q,
        },
        "questions_file": str(QUESTIONS_FILE),
        "answer_key_file": str(ANSWER_KEY_FILE),
        "questions_file_sha256": sha256_file(
            QUESTIONS_FILE
        ),
        "answer_key_file_sha256": sha256_file(
            ANSWER_KEY_FILE
        ),
        "prompt_log_file": str(prompt_log_file),
        "prompt_hash_sha256": sha256_text(
            full_prompt
        ),
        "request_timeout_seconds": (
            REQUEST_TIMEOUT_SECONDS
        ),
        "retry_policy": {
            "max_total_attempts_per_replicate": (
                MAX_TOTAL_ATTEMPTS_PER_REPLICATE
            ),
            "base_sleep_seconds": (
                BASE_SLEEP_SECONDS
            ),
            "max_backoff_seconds": (
                MAX_BACKOFF_SECONDS
            ),
            "retry_transient_api_failures": True,
            "retry_invalid_or_incomplete_structured_outputs": True,
            "retry_valid_but_incorrect_answers": False,
        },
        "between_replicates_sleep_seconds": (
            BETWEEN_REPLICATES_SLEEP_SECONDS
        ),
        "scoring": {
            "correct": 1,
            "incorrect": -1 / 3,
            "blank": 0,
            "missing_or_invalid": 0,
            "questions_without_official_answer_excluded": True,
        },
        "tools_or_web_search_provided_to_model": False,
        "structured_output_note": (
            "The forced record_all_answers tool is used "
            "only as a client-side structured-output schema. "
            "No external function is executed."
        ),
        "sampling_note": (
            "temperature, top_p and top_k are omitted. "
            "Claude Opus 4.8 does not support non-default "
            "sampling values, and omission matches the "
            "three previously completed Opus 4.8 runs."
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

    client = anthropic.Anthropic(
        api_key=api_key,
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

        result = call_claude_with_retries(
            client=client,
            full_prompt=full_prompt,
            answer_tool=answer_tool,
            expected_question_numbers=(
                expected_question_numbers
            ),
        )

        response_time = utc_iso()

        diagnostics = result.get("diagnostics") or {}
        answer_records = (
            result.get("answer_records") or {}
        )

        raw_call = {
            "run_name": run_name,
            "replicate_id": replicate_id,
            "provider": PROVIDER,
            "interface": INTERFACE,
            "api_endpoint": API_ENDPOINT,
            "model_requested": MODEL_REQUESTED,
            "model_returned": result.get(
                "model_returned"
            ),
            "response_id": result.get("response_id"),
            "stop_reason": result.get("stop_reason"),
            "condition": CONDITION,
            "prompt_template_version": (
                PROMPT_TEMPLATE_VERSION
            ),
            "script_version": SCRIPT_VERSION,
            "n_questions_in_prompt": len(questions),
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
            "n_thinking_blocks": result.get(
                "n_thinking_blocks"
            ),
            "thinking_blocks_present": result.get(
                "thinking_blocks_present"
            ),
            "thinking_content_nonempty": result.get(
                "thinking_content_nonempty"
            ),
            "positive_thinking_tokens": result.get(
                "positive_thinking_tokens"
            ),
            "thinking_tokens": result.get(
                "thinking_tokens"
            ),
            "error_type": result.get("error_type"),
            "error_message": result.get(
                "error_message"
            ),
            "latency_seconds": result.get(
                "latency_seconds"
            ),
            "elapsed_seconds_including_retries": (
                result.get(
                    "elapsed_seconds_including_retries"
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
            "cache_creation_input_tokens": (
                result.get(
                    "cache_creation_input_tokens"
                )
            ),
            "cache_read_input_tokens": result.get(
                "cache_read_input_tokens"
            ),
            "request_time_utc": request_time,
            "response_time_utc": response_time,
            "attempt_log": result.get("attempt_log"),
            "usage": result.get("usage"),
            "raw_tool_input": diagnostics.get(
                "raw_tool_input"
            ),
            "thinking_blocks": result.get(
                "thinking_blocks"
            ),
            "raw_response": result.get(
                "raw_response"
            ),
        }

        append_jsonl(raw_file, raw_call)
        call_rows.append(raw_call)

        if result.get("protocol_violation"):
            pd.DataFrame(call_rows).to_csv(
                call_level_file,
                index=False,
                quoting=csv.QUOTE_ALL,
                encoding="utf-8-sig",
            )

            metadata[
                "completed_utc"
            ] = utc_timestamp()
            metadata[
                "aborted_due_to_reasoning_protocol_violation"
            ] = True
            metadata["observed_reasoning_audit"] = {
                "n_calls_completed": len(call_rows),
                "n_calls_with_thinking_blocks": sum(
                    bool(row["thinking_blocks_present"])
                    for row in call_rows
                ),
                "n_calls_with_positive_thinking_tokens": sum(
                    bool(row["positive_thinking_tokens"])
                    for row in call_rows
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
                "RUN ABORTED: unexpected thinking was "
                "observed.",
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
                raw_answer = record.get("raw_answer")
                model_answer = normalize_answer(
                    record.get("answer")
                )
                raw_record = record.get("raw_record")

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
                "subject": question_row.get("subject"),
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
                "stop_reason": result.get(
                    "stop_reason"
                ),
                "call_success": result.get(
                    "call_success"
                ),
                "protocol_violation": result.get(
                    "protocol_violation"
                ),
                "n_thinking_blocks_call": result.get(
                    "n_thinking_blocks"
                ),
                "thinking_tokens_call": result.get(
                    "thinking_tokens"
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
                "cache_creation_input_tokens_call": (
                    result.get(
                        "cache_creation_input_tokens"
                    )
                ),
                "cache_read_input_tokens_call": (
                    result.get(
                        "cache_read_input_tokens"
                    )
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
            "thinking_blocks="
            f"{result.get('n_thinking_blocks')} "
            "thinking_tokens="
            f"{result.get('thinking_tokens')} "
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

    scored = unscored.merge(
        answer_key_min,
        on=["exam", "question_id"],
        how="left",
    )

    scoring_statuses: List[str] = []
    correct_flags: List[Optional[bool]] = []
    blank_flags: List[bool] = []
    exam_scores: List[Optional[float]] = []
    official_flags: List[bool] = []

    for _, row in scored.iterrows():
        has_official = official_answer_available(row)

        status, is_correct, is_blank, score = (
            score_answer(
                model_answer=normalize_answer(
                    row.get("model_answer")
                ),
                correct_answer=normalize_answer(
                    row.get("correct_answer")
                ),
                has_official_answer=has_official,
                parse_error=row.get("parse_error"),
                api_error=(
                    None
                    if bool(row.get("call_success"))
                    else "call_failed"
                ),
            )
        )

        official_flags.append(has_official)
        scoring_statuses.append(status)
        correct_flags.append(is_correct)
        blank_flags.append(is_blank)
        exam_scores.append(score)

    scored["has_official_answer"] = official_flags
    scored["scoring_status"] = scoring_statuses
    scored["is_correct"] = correct_flags
    scored["is_blank"] = blank_flags
    scored["exam_score"] = exam_scores

    scored.to_csv(
        scored_file,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding="utf-8-sig",
    )

    call_level = pd.DataFrame(call_rows)
    call_level.to_csv(
        call_level_file,
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
        ].copy()

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
        missing_or_invalid = int(
            (
                evaluable["scoring_status"]
                == "missing_or_invalid"
            ).sum()
        )
        n_api_error = int(
            (
                evaluable["scoring_status"]
                == "api_error"
            ).sum()
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
            "missing_or_invalid": (
                missing_or_invalid
            ),
            "n_api_error": n_api_error,
            "n_no_official_answer": int(
                (
                    group[
                        "has_official_answer"
                    ]
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
            "attempts_used": call["attempts_used"],
            "input_tokens": call["input_tokens"],
            "output_tokens": call["output_tokens"],
            "total_tokens": call["total_tokens"],
            "thinking_tokens": call[
                "thinking_tokens"
            ],
            "n_thinking_blocks": call[
                "n_thinking_blocks"
            ],
            "cache_creation_input_tokens": call[
                "cache_creation_input_tokens"
            ],
            "cache_read_input_tokens": call[
                "cache_read_input_tokens"
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

    n_calls_with_thinking_blocks = int(
        call_level["thinking_blocks_present"].fillna(
            False
        ).astype(bool).sum()
    )
    n_calls_with_positive_thinking_tokens = int(
        call_level["positive_thinking_tokens"].fillna(
            False
        ).astype(bool).sum()
    )

    expected_rows = len(questions) * args.replicates

    metadata["completed_utc"] = utc_timestamp()
    metadata["definitive_run_completed_as_prespecified"] = (
        is_full_definitive_selection
        and len(scored) == 420
        and bool(call_level["call_success"].all())
        and n_calls_with_thinking_blocks == 0
        and n_calls_with_positive_thinking_tokens == 0
    )
    metadata["observed_reasoning_audit"] = {
        "n_calls_completed": int(len(call_level)),
        "n_calls_with_thinking_blocks": (
            n_calls_with_thinking_blocks
        ),
        "n_calls_with_positive_thinking_tokens": (
            n_calls_with_positive_thinking_tokens
        ),
        "thinking_token_values": [
            (
                None
                if pd.isna(value)
                else int(value)
            )
            for value in call_level[
                "thinking_tokens"
            ].tolist()
        ],
        "protocol_violations": int(
            call_level["protocol_violation"].fillna(
                False
            ).astype(bool).sum()
        ),
    }
    metadata["output_files"] = {
        "raw_outputs": str(raw_file),
        "scored_results": str(scored_file),
        "summary_by_replicate": str(
            summary_file
        ),
        "call_level": str(call_level_file),
        "discordances": str(
            discordances_file
        ),
        "prompt_log": str(prompt_log_file),
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

    nonthinking_pass = (
        n_calls_with_thinking_blocks == 0
        and n_calls_with_positive_thinking_tokens == 0
    )
    record_count_pass = len(scored) == expected_rows
    definitive_pass = metadata[
        "definitive_run_completed_as_prespecified"
    ]

    print()
    print("=== Summary ===")
    print(summary.to_string(index=False))
    print()
    print(
        "NON-THINKING AUDIT:",
        "PASS" if nonthinking_pass else "FAIL",
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
    print(" ", scored_file)
    print(" ", summary_file)
    print(" ", call_level_file)
    print(" ", discordances_file)
    print(" ", prompt_log_file)
    print(" ", metadata_file)


if __name__ == "__main__":
    main()

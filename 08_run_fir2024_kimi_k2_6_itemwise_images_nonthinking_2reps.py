"""
FIR 2024 — Kimi K2.6 item-wise TEXT+IMAGE NON-THINKING benchmark
Strict JSON Schema, brief rationales, 2 replicates.

Expected files in the working directory:
- .env with MOONSHOT_API_KEY=...
- fir2024_questions_210.csv
- fir2024_answer_key_210.csv
- fir2024_images/ containing the local question images, unless explicit paths
  are already present in the questions CSV.

Required question columns:
- question_id
- original_question_number
- question_text
- option_1
- option_2
- option_3
- option_4

Optional image columns:
- image_path, image_paths, image_file, image_files
- figure_path, figure_paths, figure_file, figure_files
- has_image

Method:
- One question per API call.
- Images attached locally as base64 data URLs when detected.
- No PDF, file upload, tools, web search, retrieval, or answer key at inference.
- The answer key is merged locally only after the response is received.
- Valid but incorrect answers are never retried.
- Transient API failures and invalid structured outputs are retried.
- Kimi K2.6 non-thinking:
    thinking={"type":"disabled"}
    temperature=0.6
    top_p=0.95
    stream=True
- Exactly one provider-reported reasoning token with empty reasoning_content is
  treated as accounting residue. Non-empty reasoning_content or >1 reasoning
  token aborts the run.

Install:
    pip install --upgrade openai python-dotenv pandas truststore

MANDATORY dry run before API calls:
    python "08_run_fir2024_kimi_k2_6_itemwise_images_nonthinking_2reps.py" `
      --dry-run `
      --require-expected-image-count 20 `
      --inspect-q 21

Pilot on one text question and one image question:
    python "08_run_fir2024_kimi_k2_6_itemwise_images_nonthinking_2reps.py" `
      --replicates 1 `
      --start-q 20 `
      --end-q 21 `
      --require-expected-image-count 1

Definitive run:
    python "08_run_fir2024_kimi_k2_6_itemwise_images_nonthinking_2reps.py" `
      --replicates 2 `
      --require-expected-image-count 20
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

PROVIDER = "Moonshot AI / Kimi"
INTERFACE = "API"
ENDPOINT = "OpenAI-compatible Chat Completions"
MODEL = "kimi-k2.6"
DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"

CONDITION = "itemwise_textplusimage_nonthinking"
PROMPT_VERSION = "fir2024_itemwise_textplusimage_examlike_blank_strict_v1"
SCRIPT_VERSION = "fir2024_kimi_k2_6_itemwise_images_nonthinking_2reps_v1"

QUESTIONS_FILE = Path("fir2024_questions_210.csv")
ANSWER_KEY_FILE = Path("fir2024_answer_key_210.csv")
DEFAULT_IMAGES_DIR = Path("fir2024_images")
OUTPUT_ROOT = Path("outputs")

THINKING = {"type": "disabled"}
TEMPERATURE = 0.6
TOP_P = 0.95
PRESENCE_PENALTY = 0.0
FREQUENCY_PENALTY = 0.0
MAX_TOKENS_DEFAULT = 500

RESIDUAL_REASONING_TOKEN_ALLOWANCE = 1
VALID_ANSWERS = {"1", "2", "3", "4", "blank"}

IMAGE_COLUMNS = [
    "image_path", "image_paths", "image_file", "image_files",
    "figure_path", "figure_paths", "figure_file", "figure_files",
]
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]


ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fir2024_mcq_answer",
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


SYSTEM_PROMPT = """You are taking the Spanish FIR examination for Clinical Physics residency access.

This is a multiple-choice exam. Each question has four options, numbered 1 to 4, and only one is correct.

Some questions may include an image, figure, graph, diagram, equation screenshot, or table. If an image is provided, use it as part of the question.

Scoring rule:
- Correct answer: +1 point.
- Incorrect answer: -1/3 points.
- Blank answer: 0 points.

Act as a human candidate trying to maximize the final score. Answer from your own knowledge and the provided question/images only. Do not use web search, external tools, files beyond the provided image, retrieval, or any answer key.

Return only valid JSON matching the required schema:
{"answer":"1|2|3|4|blank","rationale":"brief rationale, max 160 characters"}
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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        "": None, "nan": None, "none": None, "null": None,
        "na": None, "n/a": None, "blanco": "blank",
        "en blanco": "blank", "no answer": "blank",
    }
    if text in aliases:
        return aliases[text]

    for candidate in ("1", "2", "3", "4"):
        if text in {
            candidate,
            f"{candidate}.",
            f"option {candidate}",
            f"opción {candidate}",
        }:
            return candidate
    return text


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
    if text in {"true", "1", "yes", "y", "si", "sí", "x"}:
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
# Input loading and scoring key
# ---------------------------------------------------------------------

def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing required file: {QUESTIONS_FILE}")
    if not ANSWER_KEY_FILE.exists():
        raise FileNotFoundError(f"Missing required file: {ANSWER_KEY_FILE}")

    questions = pd.read_csv(QUESTIONS_FILE)
    key = pd.read_csv(ANSWER_KEY_FILE)

    required_q = {
        "question_id", "original_question_number", "question_text",
        "option_1", "option_2", "option_3", "option_4",
    }
    required_k = {"question_id", "correct_answer"}

    missing_q = required_q - set(questions.columns)
    missing_k = required_k - set(key.columns)
    if missing_q:
        raise ValueError(f"Missing question columns: {sorted(missing_q)}")
    if missing_k:
        raise ValueError(f"Missing answer-key columns: {sorted(missing_k)}")

    if len(questions) != 210:
        raise ValueError(f"Expected 210 FIR questions, found {len(questions)}")
    if len(key) != 210:
        raise ValueError(f"Expected 210 FIR answer-key rows, found {len(key)}")

    questions = questions.sort_values("original_question_number").reset_index(drop=True)
    return questions, key


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


def select_questions(
    questions: pd.DataFrame,
    start_q: Optional[int],
    end_q: Optional[int],
) -> pd.DataFrame:
    out = questions.copy()
    qnum = out["original_question_number"].astype(int)
    if start_q is not None:
        out = out[qnum >= start_q]
    if end_q is not None:
        out = out[out["original_question_number"].astype(int) <= end_q]
    return out.sort_values("original_question_number").reset_index(drop=True)


# ---------------------------------------------------------------------
# Image detection and transport
# ---------------------------------------------------------------------

def split_possible_paths(value: str) -> List[str]:
    value = value.strip()
    if not value:
        return []

    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [safe_str(x).strip() for x in parsed if safe_str(x).strip()]
        except Exception:
            pass

    return [
        x.strip().strip('"').strip("'")
        for x in re.split(r"[;|,]", value)
        if x.strip().strip('"').strip("'")
    ]


def resolve_image_path(raw_path: str, images_dir: Path, work_dir: Path) -> Optional[Path]:
    p = Path(raw_path.strip().strip('"').strip("'"))
    if not str(p):
        return None

    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([work_dir / p, images_dir / p])

    expanded: List[Path] = []
    for candidate in candidates:
        expanded.append(candidate)
        if not candidate.suffix:
            expanded.extend(candidate.with_suffix(ext) for ext in IMAGE_EXTENSIONS)

    for candidate in expanded:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def candidate_image_stems(qnum: int, qid: str) -> List[str]:
    qid_clean = re.sub(r"[^A-Za-z0-9_\-]", "_", qid)
    numbers = [str(qnum), f"{qnum:02d}", f"{qnum:03d}"]
    stems: List[str] = []

    for n in numbers:
        stems.extend([
            f"Q{n}", f"q{n}", n,
            f"FIR_Q{n}", f"fir_q{n}",
            f"FIR_{n}", f"fir_{n}",
            f"question_{n}", f"Question_{n}",
            f"pregunta_{n}", f"Pregunta_{n}",
        ])

    if qid_clean:
        stems.extend([qid_clean, qid_clean.lower(), qid_clean.upper()])

    return list(dict.fromkeys(stems))


def find_images_for_question(
    row: pd.Series,
    images_dir: Path,
    work_dir: Path,
) -> List[Path]:
    qnum = int(row["original_question_number"])
    qid = safe_str(row["question_id"]).strip()
    found: List[Path] = []

    # Explicit paths in CSV have priority.
    for column in IMAGE_COLUMNS:
        if column not in row.index:
            continue
        raw = safe_str(row[column]).strip()
        if not raw:
            continue
        for token in split_possible_paths(raw):
            resolved = resolve_image_path(token, images_dir, work_dir)
            if resolved and resolved not in found:
                found.append(resolved)

    # Filename-pattern fallback. This is what previously mattered for Q21.
    for stem in candidate_image_stems(qnum, qid):
        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists() and candidate.is_file():
                resolved = candidate.resolve()
                if resolved not in found:
                    found.append(resolved)

    return found


def collect_image_detection(
    questions: pd.DataFrame,
    images_dir: Path,
    work_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in questions.iterrows():
        paths = find_images_for_question(row, images_dir, work_dir)
        declared = to_bool(row.get("has_image"), False) if "has_image" in row.index else None
        rows.append({
            "question_id": safe_str(row["question_id"]).strip(),
            "original_question_number": int(row["original_question_number"]),
            "has_image_declared": declared,
            "has_image_detected": bool(paths),
            "n_images_detected": len(paths),
            "image_paths": ";".join(str(p) for p in paths),
            "declared_but_missing": bool(declared and not paths),
        })
    return pd.DataFrame(rows)


def mime_type_for_image(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime in {"image/png", "image/jpeg", "image/webp"}:
        return mime
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    raise ValueError(f"Unsupported image type: {path}")


def image_to_data_url(path: Path) -> str:
    mime = mime_type_for_image(path)
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def build_question_text(row: pd.Series, image_paths: Sequence[Path]) -> str:
    qnum = int(row["original_question_number"])
    if len(image_paths) == 1:
        image_note = "\nAn image/figure is attached after the question text. Use it.\n"
    elif len(image_paths) > 1:
        image_note = f"\n{len(image_paths)} images/figures are attached after the question text. Use them.\n"
    else:
        image_note = ""

    return f"""Question {qnum}:{image_note}
{safe_str(row["question_text"]).strip()}

Options:
1. {safe_str(row["option_1"]).strip()}
2. {safe_str(row["option_2"]).strip()}
3. {safe_str(row["option_3"]).strip()}
4. {safe_str(row["option_4"]).strip()}"""


def build_user_content(
    row: pd.Series,
    image_paths: Sequence[Path],
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": build_question_text(row, image_paths)}
    ]
    for path in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_data_url(path)},
        })
    return content


# ---------------------------------------------------------------------
# Streaming API call, parsing and reasoning audit
# ---------------------------------------------------------------------

def extract_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
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


def parse_output(raw_content: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Any]:
    if not raw_content or not raw_content.strip():
        return None, None, "missing_output", None

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        return None, None, f"json_decode_error:{exc}", None

    if not isinstance(parsed, dict):
        return None, None, "top_level_not_object", parsed
    if set(parsed) != {"answer", "rationale"}:
        return None, None, "wrong_fields", parsed

    answer = normalize_answer(parsed.get("answer"))
    rationale = parsed.get("rationale")

    if answer not in VALID_ANSWERS:
        return answer, rationale if isinstance(rationale, str) else None, "invalid_answer", parsed
    if not isinstance(rationale, str):
        return answer, None, "invalid_rationale", parsed
    if len(rationale) > 160:
        return answer, rationale, "rationale_too_long", parsed

    return answer, rationale, None, parsed


def call_once(
    client: OpenAI,
    user_content: List[Dict[str, Any]],
    max_tokens: int,
) -> Dict[str, Any]:
    started = time.perf_counter()

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=ANSWER_SCHEMA,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        presence_penalty=PRESENCE_PENALTY,
        frequency_penalty=FREQUENCY_PENALTY,
        n=1,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"thinking": THINKING},
        # No tools, web search, retrieval, file/PDF input.
    )

    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    response_id = None
    model_returned = None
    finish_reason = None
    usage_dict: Dict[str, Any] = {}
    n_chunks = 0

    for chunk in stream:
        n_chunks += 1
        data = dump_object(chunk)

        response_id = response_id or data.get("id")
        model_returned = model_returned or data.get("model")
        if data.get("usage"):
            usage_dict = data["usage"] or {}

        for choice in data.get("choices") or []:
            if choice.get("finish_reason") is not None:
                finish_reason = choice.get("finish_reason")
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_parts.append(str(delta["content"]))

            reasoning = (
                delta.get("reasoning_content")
                or delta.get("reasoning")
                or delta.get("analysis")
            )
            if reasoning:
                reasoning_parts.append(str(reasoning))

    raw_content = "".join(content_parts)
    reasoning_content = "".join(reasoning_parts)
    elapsed = time.perf_counter() - started
    usage_fields = extract_usage(usage_dict)

    try:
        reasoning_tokens_num = float(usage_fields.get("reasoning_tokens") or 0)
    except (TypeError, ValueError):
        reasoning_tokens_num = 0.0

    reasoning_content_nonempty = bool(reasoning_content.strip())
    substantive_reasoning = (
        reasoning_content_nonempty
        or reasoning_tokens_num > RESIDUAL_REASONING_TOKEN_ALLOWANCE
    )
    residual_only = (
        not reasoning_content_nonempty
        and reasoning_tokens_num == RESIDUAL_REASONING_TOKEN_ALLOWANCE
    )

    answer, rationale, parse_error, parsed = parse_output(raw_content)

    return {
        "response_id": response_id,
        "model_returned": model_returned,
        "finish_reason": finish_reason,
        "raw_content": raw_content,
        "parsed_output": parsed,
        "answer": answer,
        "rationale": rationale,
        "parse_error": parse_error,
        "reasoning_content": reasoning_content,
        "reasoning_content_nonempty": reasoning_content_nonempty,
        "substantive_reasoning": substantive_reasoning,
        "residual_reasoning_token_only": residual_only,
        "elapsed_seconds": elapsed,
        "n_stream_chunks": n_chunks,
        "usage_raw": usage_dict,
        **usage_fields,
    }


def is_transient(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return getattr(exc, "status_code", None) in {
            408, 409, 425, 429, 500, 502, 503, 504,
        }
    return False


def backoff(attempt: int, base: float, maximum: float) -> float:
    value = min(maximum, base * (2 ** max(0, attempt - 1)))
    return value + random.uniform(0, min(1.5, value * 0.1))


def score_answer(
    model_answer: Optional[str],
    correct_answer: Optional[str],
    official_available: bool,
) -> Dict[str, Any]:
    if not official_available:
        return {
            "is_evaluable": False,
            "is_correct": None,
            "is_incorrect": None,
            "is_blank": model_answer == "blank",
            "score": None,
            "answer_status": "not_evaluable",
        }
    if model_answer not in VALID_ANSWERS:
        return {
            "is_evaluable": True,
            "is_correct": False,
            "is_incorrect": False,
            "is_blank": False,
            "score": 0.0,
            "answer_status": "missing_or_invalid",
        }
    if model_answer == "blank":
        return {
            "is_evaluable": True,
            "is_correct": False,
            "is_incorrect": False,
            "is_blank": True,
            "score": 0.0,
            "answer_status": "blank",
        }
    if model_answer == correct_answer:
        return {
            "is_evaluable": True,
            "is_correct": True,
            "is_incorrect": False,
            "is_blank": False,
            "score": 1.0,
            "answer_status": "correct",
        }
    return {
        "is_evaluable": True,
        "is_correct": False,
        "is_incorrect": True,
        "is_blank": False,
        "score": -1.0 / 3.0,
        "answer_status": "incorrect",
    }


# ---------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------

def flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    image_paths = record.get("image_paths") or []
    return {
        "run_id": record.get("run_id"),
        "provider": PROVIDER,
        "model_requested": MODEL,
        "model_returned": record.get("model_returned"),
        "condition": CONDITION,
        "replicate": record.get("replicate"),
        "question_id": record.get("question_id"),
        "original_question_number": record.get("original_question_number"),
        "question_text": record.get("question_text"),
        "option_1": record.get("option_1"),
        "option_2": record.get("option_2"),
        "option_3": record.get("option_3"),
        "option_4": record.get("option_4"),
        "has_image_declared": record.get("has_image_declared"),
        "has_image_detected": record.get("has_image_detected"),
        "n_images_detected": record.get("n_images_detected"),
        "image_paths": ";".join(image_paths) if isinstance(image_paths, list) else safe_str(image_paths),
        "model_answer": record.get("answer"),
        "rationale": record.get("rationale"),
        "correct_answer": record.get("correct_answer"),
        "official_answer_available": record.get("official_answer_available"),
        "is_evaluable": record.get("is_evaluable"),
        "is_correct": record.get("is_correct"),
        "is_incorrect": record.get("is_incorrect"),
        "is_blank": record.get("is_blank"),
        "answer_status": record.get("answer_status"),
        "score": record.get("score"),
        "parse_error": record.get("parse_error"),
        "api_ok": record.get("api_ok"),
        "attempts_used": record.get("attempts_used"),
        "response_id": record.get("response_id"),
        "finish_reason": record.get("finish_reason"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "prompt_tokens": record.get("prompt_tokens"),
        "completion_tokens": record.get("completion_tokens"),
        "reasoning_tokens": record.get("reasoning_tokens"),
        "cached_tokens": record.get("cached_tokens"),
        "total_tokens": record.get("total_tokens"),
        "reasoning_content_nonempty": record.get("reasoning_content_nonempty"),
        "substantive_reasoning": record.get("substantive_reasoning"),
        "residual_reasoning_token_only": record.get("residual_reasoning_token_only"),
        "error_type": record.get("error_type"),
        "error_message": record.get("error_message"),
    }


def write_results(path: Path, records: List[Dict[str, Any]]) -> None:
    pd.DataFrame([flatten_record(r) for r in records]).to_csv(
        path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL
    )


def build_summary(records: List[Dict[str, Any]], replicates: Iterable[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for replicate in replicates:
        group = [r for r in records if r["replicate"] == replicate]
        evaluable = [r for r in group if r.get("is_evaluable") is True]
        image_eval = [
            r for r in evaluable if r.get("has_image_detected") is True
        ]
        text_eval = [
            r for r in evaluable if r.get("has_image_detected") is not True
        ]

        rows.append({
            "replicate": replicate,
            "n_questions_total": len(group),
            "n_evaluable": len(evaluable),
            "n_not_evaluable": sum(r.get("is_evaluable") is False for r in group),
            "n_correct": sum(r.get("answer_status") == "correct" for r in evaluable),
            "n_incorrect": sum(r.get("answer_status") == "incorrect" for r in evaluable),
            "n_blank": sum(r.get("answer_status") == "blank" for r in evaluable),
            "missing_or_invalid": sum(
                r.get("answer_status") == "missing_or_invalid" for r in evaluable
            ),
            "accuracy_evaluable": (
                sum(r.get("answer_status") == "correct" for r in evaluable) / len(evaluable)
                if evaluable else None
            ),
            "score": sum(float(r.get("score") or 0) for r in evaluable),
            "n_image_questions": sum(r.get("has_image_detected") is True for r in group),
            "n_image_evaluable": len(image_eval),
            "n_image_correct": sum(r.get("answer_status") == "correct" for r in image_eval),
            "image_accuracy_evaluable": (
                sum(r.get("answer_status") == "correct" for r in image_eval) / len(image_eval)
                if image_eval else None
            ),
            "n_text_evaluable": len(text_eval),
            "n_text_correct": sum(r.get("answer_status") == "correct" for r in text_eval),
            "text_accuracy_evaluable": (
                sum(r.get("answer_status") == "correct" for r in text_eval) / len(text_eval)
                if text_eval else None
            ),
            "api_errors": sum(not bool(r.get("api_ok")) for r in group),
            "parse_errors": sum(bool(r.get("parse_error")) for r in group),
            "attempts_total": sum(int(r.get("attempts_used") or 0) for r in group),
            "latency_seconds": sum(float(r.get("elapsed_seconds") or 0) for r in group),
            "mean_latency_seconds": (
                sum(float(r.get("elapsed_seconds") or 0) for r in group) / len(group)
                if group else None
            ),
            "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in group),
            "completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in group),
            "reasoning_tokens": sum(float(r.get("reasoning_tokens") or 0) for r in group),
            "cached_tokens": sum(int(r.get("cached_tokens") or 0) for r in group),
            "total_tokens": sum(int(r.get("total_tokens") or 0) for r in group),
        })
    return pd.DataFrame(rows)


def build_discordances(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame([flatten_record(r) for r in records])
    if df.empty or df["replicate"].nunique() < 2:
        return pd.DataFrame()

    pivot = df.pivot_table(
        index=[
            "question_id", "original_question_number",
            "correct_answer", "official_answer_available",
            "has_image_detected",
        ],
        columns="replicate",
        values=["model_answer", "answer_status", "score"],
        aggfunc="first",
    )
    pivot.columns = [f"{a}_rep{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    answer_cols = sorted(c for c in pivot.columns if c.startswith("model_answer_rep"))
    pivot["same_answer"] = pivot[answer_cols].nunique(axis=1, dropna=False) == 1
    return pivot[pivot["same_answer"] == False].sort_values(
        "original_question_number"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--start-q", type=int, default=None)
    parser.add_argument("--end-q", type=int, default=None)
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    parser.add_argument("--require-expected-image-count", type=int, default=None)
    parser.add_argument("--inspect-q", type=int, default=21)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_DEFAULT)
    parser.add_argument("--max-total-attempts", type=int, default=7)
    parser.add_argument("--base-sleep", type=float, default=10.0)
    parser.add_argument("--max-backoff", type=float, default=120.0)
    parser.add_argument("--between-calls-sleep", type=float, default=0.15)
    parser.add_argument("--between-replicates-sleep", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.replicates < 1:
        raise ValueError("--replicates must be >= 1")

    load_dotenv()
    api_key = os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")
    base_url = os.getenv("KIMI_BASE_URL", DEFAULT_BASE_URL)
    if not api_key and not args.dry_run:
        raise RuntimeError("Missing MOONSHOT_API_KEY or KIMI_API_KEY in .env")

    work_dir = Path.cwd()
    images_dir = Path(args.images_dir)
    if not images_dir.is_absolute():
        images_dir = (work_dir / images_dir).resolve()

    questions_all, key = load_inputs()
    questions = select_questions(questions_all, args.start_q, args.end_q)
    key_map = build_key_map(key)
    if questions.empty:
        raise ValueError("No questions selected.")

    image_detection = collect_image_detection(questions, images_dir, work_dir)
    n_image_questions = int(image_detection["has_image_detected"].sum())
    n_declared_missing = int(image_detection["declared_but_missing"].sum())

    if n_declared_missing:
        missing_rows = image_detection[image_detection["declared_but_missing"]]
        raise RuntimeError(
            "Questions declared as image questions have no detected image:\n"
            + missing_rows.to_string(index=False)
        )

    if (
        args.require_expected_image_count is not None
        and n_image_questions != args.require_expected_image_count
    ):
        raise RuntimeError(
            f"Expected {args.require_expected_image_count} image questions, "
            f"detected {n_image_questions}.\n\n"
            + image_detection[image_detection["has_image_detected"]].to_string(index=False)
        )

    timestamp = utc_compact()
    selection_suffix = ""
    if args.start_q is not None or args.end_q is not None:
        selection_suffix = f"_q{args.start_q or 1}-{args.end_q or 210}"
    run_id = (
        f"fir2024_kimi_k2_6_itemwise_images_nonthinking_"
        f"{args.replicates}r_STRICT{selection_suffix}_{timestamp}"
    )
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    raw_path = out_dir / f"{run_id}_raw_results.jsonl"
    results_path = out_dir / f"{run_id}_results.csv"
    summary_path = out_dir / f"{run_id}_summary.csv"
    discordances_path = out_dir / f"{run_id}_discordances.csv"
    image_detection_path = out_dir / f"{run_id}_image_detection.csv"
    image_manifest_path = out_dir / f"{run_id}_image_manifest.csv"
    metadata_path = out_dir / f"{run_id}_metadata.json"
    preamble_path = out_dir / f"{run_id}_prompt_preamble.txt"

    image_detection.to_csv(
        image_detection_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL
    )
    image_detection[image_detection["has_image_detected"]].to_csv(
        image_manifest_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL
    )
    preamble_path.write_text(SYSTEM_PROMPT, encoding="utf-8")

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
        "images_dir": str(images_dir),
        "n_questions_in_file": len(questions_all),
        "n_questions_selected": len(questions),
        "is_full_definitive_selection": (
            len(questions) == 210 and args.start_q is None and args.end_q is None
        ),
        "n_image_questions_detected": n_image_questions,
        "n_declared_image_questions_missing": n_declared_missing,
        "replicates": args.replicates,
        "start_q": args.start_q,
        "end_q": args.end_q,
        "thinking": {
            "requested": THINKING,
            "policy": "explicitly_disabled",
            "reasoning_audit_rule": {
                "abort_if_reasoning_content_nonempty": True,
                "abort_if_reasoning_tokens_above": RESIDUAL_REASONING_TOKEN_ALLOWANCE,
                "residual_reasoning_token_allowance": RESIDUAL_REASONING_TOKEN_ALLOWANCE,
            },
        },
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "sampling_parameter_note": (
            "Provider-constrained settings for Kimi K2.6 non-thinking mode."
        ),
        "presence_penalty": PRESENCE_PENALTY,
        "frequency_penalty": FREQUENCY_PENALTY,
        "max_tokens": args.max_tokens,
        "response_format": "json_schema_strict",
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": "not_provided",
        "web_search": "not_provided",
        "file_input": False,
        "pdf_input": False,
        "image_input": True,
        "image_transport": "local_base64_data_url",
        "rationales_requested": True,
        "rationale_max_chars": 160,
        "request_timeout_seconds": args.timeout,
        "retry_policy": {
            "max_total_attempts": args.max_total_attempts,
            "retry_transient_api_failures": True,
            "retry_invalid_structured_outputs": True,
            "retry_unexpected_finish_reason": True,
            "retry_valid_but_incorrect_answers": False,
        },
        "between_calls_sleep_seconds": args.between_calls_sleep,
        "between_replicates_sleep_seconds": args.between_replicates_sleep,
        "scoring": {
            "correct": 1,
            "incorrect": -1 / 3,
            "blank": 0,
            "missing_or_invalid": 0,
            "questions_without_official_answer_excluded": True,
        },
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "answer_schema": ANSWER_SCHEMA,
        "output_files": {
            "metadata": str(metadata_path),
            "raw_results": str(raw_path),
            "results": str(results_path),
            "summary": str(summary_path),
            "discordances": str(discordances_path),
            "image_detection": str(image_detection_path),
            "image_manifest": str(image_manifest_path),
            "prompt_preamble": str(preamble_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== FIR 2024 Kimi K2.6 item-wise TEXT+IMAGE NON-THINKING ===")
    print("Run ID:", run_id)
    print("Questions selected:", len(questions))
    print("Image questions detected:", n_image_questions)
    print("Images directory:", images_dir)
    print("Replicates:", args.replicates)
    print("Output directory:", out_dir)
    print()

    detected = image_detection[image_detection["has_image_detected"]]
    if len(detected):
        print("Detected image questions:")
        print(detected[
            ["original_question_number", "question_id", "n_images_detected", "image_paths"]
        ].to_string(index=False))
        print()

    inspect = questions[
        questions["original_question_number"].astype(int) == args.inspect_q
    ]
    if len(inspect):
        inspect_row = inspect.iloc[0]
        inspect_paths = find_images_for_question(inspect_row, images_dir, work_dir)
        print(f"Inspection Q{args.inspect_q}:")
        print(build_question_text(inspect_row, inspect_paths))
        print("Attached image paths:")
        for path in inspect_paths:
            print(" ", path)
        if not inspect_paths:
            print("  [NONE DETECTED]")
        print()

    if args.dry_run:
        metadata["dry_run"] = True
        metadata["completed_utc"] = utc_compact()
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("DRY RUN completed. No API calls were made.")
        return 0

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=args.timeout,
        max_retries=0,
    )

    all_records: List[Dict[str, Any]] = []
    protocol_violation = False
    total_planned = args.replicates * len(questions)
    counter = 0

    for replicate in range(1, args.replicates + 1):
        print(f"--- Replicate {replicate}/{args.replicates} ---")

        for _, row in questions.iterrows():
            counter += 1
            qnum = int(row["original_question_number"])
            qid = safe_str(row["question_id"]).strip()
            image_paths = find_images_for_question(row, images_dir, work_dir)
            user_content = build_user_content(row, image_paths)

            accepted: Optional[Dict[str, Any]] = None
            attempts_used = 0
            last_error_type = None
            last_error_message = None

            for attempt in range(1, args.max_total_attempts + 1):
                attempts_used = attempt
                attempt_started = utc_iso()
                try:
                    response = call_once(client, user_content, args.max_tokens)

                    raw_attempt = {
                        "run_id": run_id,
                        "replicate": replicate,
                        "question_id": qid,
                        "original_question_number": qnum,
                        "attempt": attempt,
                        "attempt_started_utc": attempt_started,
                        "attempt_completed_utc": utc_iso(),
                        "image_paths": [str(p) for p in image_paths],
                        "n_images_sent": len(image_paths),
                        "request_succeeded": True,
                        **response,
                    }
                    append_jsonl(raw_path, raw_attempt)

                    if response["substantive_reasoning"]:
                        protocol_violation = True
                        raise RuntimeError(
                            f"Substantive reasoning detected at replicate "
                            f"{replicate}, Q{qnum}."
                        )

                    finish_ok = response.get("finish_reason") in {"stop", None}
                    output_ok = response.get("parse_error") is None
                    if finish_ok and output_ok:
                        accepted = response
                        break

                    last_error_type = "InvalidStructuredOutput"
                    last_error_message = (
                        response.get("parse_error")
                        or f"unexpected_finish_reason:{response.get('finish_reason')}"
                    )

                    if attempt < args.max_total_attempts:
                        time.sleep(backoff(attempt, args.base_sleep, args.max_backoff))

                except RuntimeError as exc:
                    if "Substantive reasoning detected" in str(exc):
                        raise
                    raise
                except Exception as exc:
                    transient = is_transient(exc)
                    last_error_type = type(exc).__name__
                    last_error_message = str(exc)

                    append_jsonl(raw_path, {
                        "run_id": run_id,
                        "replicate": replicate,
                        "question_id": qid,
                        "original_question_number": qnum,
                        "attempt": attempt,
                        "attempt_started_utc": attempt_started,
                        "attempt_completed_utc": utc_iso(),
                        "image_paths": [str(p) for p in image_paths],
                        "n_images_sent": len(image_paths),
                        "request_succeeded": False,
                        "transient_exception": transient,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "traceback": traceback.format_exc(),
                    })

                    if not transient or attempt >= args.max_total_attempts:
                        break
                    time.sleep(backoff(attempt, args.base_sleep, args.max_backoff))

            key_info = key_map.get(qid, {
                "correct_answer": None,
                "official_answer_available": False,
            })

            if accepted is None:
                model_answer = None
                rationale = None
                response_fields: Dict[str, Any] = {
                    "api_ok": False,
                    "parse_error": last_error_message,
                    "response_id": None,
                    "model_returned": None,
                    "finish_reason": None,
                    "elapsed_seconds": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "reasoning_tokens": None,
                    "cached_tokens": None,
                    "total_tokens": None,
                    "reasoning_content_nonempty": None,
                    "substantive_reasoning": None,
                    "residual_reasoning_token_only": None,
                    "error_type": last_error_type,
                    "error_message": last_error_message,
                }
            else:
                model_answer = accepted["answer"]
                rationale = accepted["rationale"]
                response_fields = {
                    "api_ok": True,
                    "parse_error": accepted.get("parse_error"),
                    "response_id": accepted.get("response_id"),
                    "model_returned": accepted.get("model_returned"),
                    "finish_reason": accepted.get("finish_reason"),
                    "elapsed_seconds": accepted.get("elapsed_seconds"),
                    "prompt_tokens": accepted.get("prompt_tokens"),
                    "completion_tokens": accepted.get("completion_tokens"),
                    "reasoning_tokens": accepted.get("reasoning_tokens"),
                    "cached_tokens": accepted.get("cached_tokens"),
                    "total_tokens": accepted.get("total_tokens"),
                    "reasoning_content_nonempty": accepted.get("reasoning_content_nonempty"),
                    "substantive_reasoning": accepted.get("substantive_reasoning"),
                    "residual_reasoning_token_only": accepted.get(
                        "residual_reasoning_token_only"
                    ),
                    "error_type": None,
                    "error_message": None,
                }

            scored = score_answer(
                model_answer=model_answer,
                correct_answer=key_info["correct_answer"],
                official_available=bool(key_info["official_answer_available"]),
            )

            record = {
                "run_id": run_id,
                "replicate": replicate,
                "question_id": qid,
                "original_question_number": qnum,
                "question_text": safe_str(row["question_text"]),
                "option_1": safe_str(row["option_1"]),
                "option_2": safe_str(row["option_2"]),
                "option_3": safe_str(row["option_3"]),
                "option_4": safe_str(row["option_4"]),
                "has_image_declared": (
                    to_bool(row.get("has_image"), False)
                    if "has_image" in row.index else None
                ),
                "has_image_detected": bool(image_paths),
                "n_images_detected": len(image_paths),
                "image_paths": [str(p) for p in image_paths],
                "answer": model_answer,
                "rationale": rationale,
                "correct_answer": key_info["correct_answer"],
                "official_answer_available": bool(
                    key_info["official_answer_available"]
                ),
                "attempts_used": attempts_used,
                **scored,
                **response_fields,
            }
            all_records.append(record)
            write_results(results_path, all_records)

            answer_label = model_answer if model_answer is not None else "ERROR"
            print(
                f"[{counter}/{total_planned}] Rep {replicate} Q{qnum}: "
                f"{answer_label} | {record['answer_status']} | "
                f"images={len(image_paths)} | attempts={attempts_used}"
            )

            if counter < total_planned:
                time.sleep(args.between_calls_sleep)

        if replicate < args.replicates:
            time.sleep(args.between_replicates_sleep)

    results_df = pd.DataFrame([flatten_record(r) for r in all_records])
    summary_df = build_summary(all_records, range(1, args.replicates + 1))
    discordances_df = build_discordances(all_records)

    results_df.to_csv(
        results_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL
    )
    summary_df.to_csv(
        summary_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL
    )
    discordances_df.to_csv(
        discordances_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL
    )

    reasoning_values = [
        float(r.get("reasoning_tokens") or 0)
        for r in all_records if r.get("api_ok")
    ]
    distribution: Dict[str, int] = {}
    for value in reasoning_values:
        label = str(int(value)) if value.is_integer() else str(value)
        distribution[label] = distribution.get(label, 0) + 1

    metadata.update({
        "completed_utc": utc_compact(),
        "definitive_run_completed_as_prespecified": bool(
            not protocol_violation
            and args.replicates == 2
            and len(questions) == 210
            and args.start_q is None
            and args.end_q is None
            and len(all_records) == 420
            and all(bool(r.get("api_ok")) for r in all_records)
        ),
        "observed_reasoning_audit": {
            "n_calls_accepted": sum(bool(r.get("api_ok")) for r in all_records),
            "n_calls_with_reasoning_content": sum(
                bool(r.get("reasoning_content_nonempty")) for r in all_records
            ),
            "n_calls_with_any_reasoning_tokens": sum(v > 0 for v in reasoning_values),
            "n_calls_with_residual_reasoning_token_only": sum(
                bool(r.get("residual_reasoning_token_only")) for r in all_records
            ),
            "n_calls_with_substantive_reasoning": sum(
                bool(r.get("substantive_reasoning")) for r in all_records
            ),
            "n_protocol_violations": int(protocol_violation),
            "total_reasoning_tokens": sum(reasoning_values),
            "reasoning_token_distribution": distribution,
            "residual_reasoning_token_allowance": RESIDUAL_REASONING_TOKEN_ALLOWANCE,
        },
    })
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print()
    print("Completed.")
    print(summary_df.to_string(index=False))
    print()
    print("Results:      ", results_path)
    print("Summary:      ", summary_path)
    print("Discordances: ", discordances_path)
    print("Image manifest:", image_manifest_path)
    print("Metadata:     ", metadata_path)
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

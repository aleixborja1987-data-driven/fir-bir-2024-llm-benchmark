"""
FIR 2024 whole-exam text+image answers-only benchmark for Kimi K2.6
NON-THINKING (2 replicates).

Provider/model:
- Moonshot AI / Kimi Open Platform
- model: kimi-k2.6
- endpoint: OpenAI-compatible Chat Completions
- base_url: https://api.moonshot.ai/v1

Condition:
- The complete extracted FIR 2024 exam is sent in ONE API call per replicate.
- All detected images are attached in the same user message, immediately after
  the corresponding question text.
- Answers only: no rationales requested.
- No PDF, tools, web search, retrieval, external files, or answer key are
  provided during inference.
- The answer key is merged locally only after the response is received.

Prespecified Kimi K2.6 non-thinking settings:
- thinking: explicitly disabled with extra_body={"thinking":{"type":"disabled"}}
- temperature: 0.6 (provider-supported non-thinking value)
- top_p: 0.95 (provider-supported non-thinking value)
- stream: true
- response_format: strict JSON Schema
- max_tokens: 12000 by default
- valid but incorrect answers are never retried
- only transient API failures, unexpected finish reasons, or invalid/incomplete
  structured outputs are retried

Reasoning audit:
- Any non-empty reasoning_content is substantive reasoning and aborts the run.
- More than 1 provider-reported reasoning token is substantive reasoning and
  aborts the run.
- Exactly 1 reported reasoning token with empty reasoning_content is logged as
  provider-accounting residue and accepted, following the validated pilot and
  prior Kimi K2.6 runs.

Expected files in the working directory:
- .env with MOONSHOT_API_KEY=... (or KIMI_API_KEY=...)
- fir2024_questions_210.csv
- fir2024_answer_key_210.csv
- image files referenced in the questions CSV or available in --images-dir

Install:
    pip install --upgrade openai python-dotenv pandas truststore

Dry run / image validation:
    python "08_run_fir2024_kimi_k2_6_whole_exam_images_nonthinking_2reps.py" `
      --dry-run `
      --require-expected-image-count 20

Recommended full-structure pilot (210 questions, 20 images, 1 replicate):
    python "08_run_fir2024_kimi_k2_6_whole_exam_images_nonthinking_2reps.py" `
      --replicates 1 `
      --require-expected-image-count 20

Definitive run:
    python "08_run_fir2024_kimi_k2_6_whole_exam_images_nonthinking_2reps.py" `
      --replicates 2 `
      --require-expected-image-count 20

A partial --limit probe can test mechanics, but it does not validate output
completeness or token limits for the full whole-exam condition. When using a
partial selection, set --require-expected-image-count to the number of image
questions contained in that selection.
"""

from __future__ import annotations

import argparse
import base64
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import truststore
    truststore.inject_into_ssl()
    TRUSTSTORE_STATUS = "enabled"
except Exception as exc:
    TRUSTSTORE_STATUS = f"not_enabled: {repr(exc)}"

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


# ============================================================
# Study configuration
# ============================================================

PROVIDER = "Moonshot AI / Kimi"
INTERFACE = "API"
API_ENDPOINT = "OpenAI-compatible Chat Completions"
MODEL_REQUESTED = "kimi-k2.6"
DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"

CONDITION = "whole_exam_text_plus_images_answers_only_nonthinking"
PROMPT_TEMPLATE_VERSION = "fir2024_whole_exam_textplusimage_examlike_blank_strict_ansonly_nonthinking_v1"
SCRIPT_VERSION = "fir2024_kimi_k2_6_whole_exam_images_nonthinking_2reps_v1"

QUESTIONS_FILE = Path("fir2024_questions_210.csv")
ANSWER_KEY_FILE = Path("fir2024_answer_key_210.csv")
DEFAULT_IMAGES_DIR = Path("images") / "fir"
OUTPUT_ROOT = Path("outputs")

VALID_ANSWERS = {"1", "2", "3", "4", "blank"}

IMAGE_COLUMNS = [
    "image_path",
    "image_paths",
    "image_file",
    "image_files",
    "figure_path",
    "figure_paths",
    "figure_file",
    "figure_files",
]
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]

N_REPLICATES_DEFAULT = 2
THINKING = {"type": "disabled"}
TEMPERATURE = 0.6
TOP_P = 0.95
PRESENCE_PENALTY = 0.0
FREQUENCY_PENALTY = 0.0
N = 1
RESIDUAL_REASONING_TOKEN_ALLOWANCE = 1
MAX_TOKENS_DEFAULT = 12000
REQUEST_TIMEOUT_SECONDS_DEFAULT = 1800.0
MAX_RETRIES_DEFAULT = 7
BASE_SLEEP_SECONDS_DEFAULT = 10.0
MAX_BACKOFF_SECONDS_DEFAULT = 120.0
BETWEEN_REPLICATES_SLEEP_DEFAULT = 2.0


PROMPT_PREAMBLE = """You are taking the Spanish FIR examination for Clinical Physics residency access.

This is a multiple-choice exam. Each question has four options, numbered 1 to 4, and only one is correct. In the official exam, correct answers score +1 point, incorrect answers score -1/3 points, and blank answers score 0 points. Act as a human candidate and try to obtain the highest possible final score.

You will receive the complete exam in a single message. Some questions include one or more images, figures, graphs, diagrams, equation screenshots, or tables. When an image is attached immediately after a question, it belongs to that question and must be used to answer it.

Answer only from your own knowledge and the supplied question text and images. Do not use web search, tools, external references, other files, retrieval, or an answer key.

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
- Preserve the same question order as in the input.
- The value of "answer" must be exactly one string: "1", "2", "3", "4", or "blank".
- Do not return multiple options.
- If you cannot choose one single best option, return "blank".
- Do not include rationales, explanations, comments, markdown, or any other fields.

FULL EXAM:
"""


FINAL_REMINDER = """End of the exam.

Return only the required JSON object containing exactly one answer for every question above, in the same order. Do not include any rationale or additional text."""


# ============================================================
# Generic helpers
# ============================================================

def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def parse_bool(value: Any, default: bool = False) -> bool:
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
    if text in {"true", "1", "yes", "y", "si", "sí", "x", "image", "fig", "figure"}:
        return True
    if text in {"false", "0", "no", "n", "", "nan", "none", "null"}:
        return False
    return default


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
    if text in aliases:
        return aliases[text]
    return text


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


def object_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    try:
        return dict(obj)
    except Exception:
        return {"repr": repr(obj)}


# ============================================================
# Input loading
# ============================================================

def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing required file: {QUESTIONS_FILE}")
    if not ANSWER_KEY_FILE.exists():
        raise FileNotFoundError(f"Missing required file: {ANSWER_KEY_FILE}")

    questions = pd.read_csv(QUESTIONS_FILE)
    answer_key = pd.read_csv(ANSWER_KEY_FILE)

    required_question_columns = [
        "question_id",
        "original_question_number",
        "question_text",
        "option_1",
        "option_2",
        "option_3",
        "option_4",
    ]
    missing_questions = [c for c in required_question_columns if c not in questions.columns]
    if missing_questions:
        raise ValueError(f"{QUESTIONS_FILE} is missing columns: {missing_questions}")

    required_key_columns = ["question_id", "correct_answer"]
    missing_key = [c for c in required_key_columns if c not in answer_key.columns]
    if missing_key:
        raise ValueError(f"{ANSWER_KEY_FILE} is missing columns: {missing_key}")

    questions = questions.sort_values("original_question_number").reset_index(drop=True)
    return questions, answer_key


def filter_questions(
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
        out = out.head(limit).copy()

    return out


def build_answer_key_map(answer_key: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}

    for _, row in answer_key.iterrows():
        question_id = safe_str(row["question_id"]).strip()
        correct_answer = normalize_answer(row.get("correct_answer"))

        official_available = True
        if "official_answer_available" in answer_key.columns:
            official_available = parse_bool(
                row.get("official_answer_available"),
                default=True,
            )

        if "is_annulled_or_without_official_answer" in answer_key.columns:
            if parse_bool(
                row.get("is_annulled_or_without_official_answer"),
                default=False,
            ):
                official_available = False

        if correct_answer not in {"1", "2", "3", "4"}:
            official_available = False

        mapping[question_id] = {
            "correct_answer": correct_answer,
            "official_answer_available": official_available,
        }

    return mapping


# ============================================================
# Image detection and encoding
# ============================================================

def split_possible_paths(value: str) -> List[str]:
    value = value.strip()
    if not value:
        return []

    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [
                    safe_str(item).strip()
                    for item in parsed
                    if safe_str(item).strip()
                ]
        except Exception:
            pass

    chunks = re.split(r"[;|,]", value)
    return [
        chunk.strip().strip('"').strip("'")
        for chunk in chunks
        if chunk.strip().strip('"').strip("'")
    ]


def candidate_image_names(question_number: int, question_id: str) -> List[str]:
    qid_clean = re.sub(r"[^A-Za-z0-9_\-]", "_", question_id)
    numbers = [
        str(question_number),
        f"{question_number:02d}",
        f"{question_number:03d}",
    ]

    stems: List[str] = []
    for number in numbers:
        stems.extend([
            f"Q{number}",
            f"q{number}",
            number,
            f"FIR_Q{number}",
            f"fir_q{number}",
            f"FIR_{number}",
            f"fir_{number}",
            f"question_{number}",
            f"Question_{number}",
            f"pregunta_{number}",
            f"Pregunta_{number}",
        ])

    if qid_clean:
        stems.extend([qid_clean, qid_clean.lower(), qid_clean.upper()])

    seen = set()
    output: List[str] = []
    for stem in stems:
        if stem not in seen:
            output.append(stem)
            seen.add(stem)
    return output


def resolve_path(
    raw_path: str,
    images_dir: Path,
    work_dir: Path,
) -> Optional[Path]:
    raw_path = raw_path.strip().strip('"').strip("'")
    if not raw_path:
        return None

    path = Path(raw_path)
    candidates: List[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(work_dir / path)
        candidates.append(images_dir / path)

    expanded: List[Path] = []
    for candidate in candidates:
        expanded.append(candidate)
        if candidate.suffix == "":
            for extension in IMAGE_EXTENSIONS:
                expanded.append(candidate.with_suffix(extension))

    for candidate in expanded:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    return None


def find_images_for_question(
    row: pd.Series,
    images_dir: Path,
    work_dir: Path,
) -> List[Path]:
    question_number = int(row["original_question_number"])
    question_id = safe_str(row["question_id"]).strip()

    found: List[Path] = []

    # Explicit paths in the CSV take priority.
    for column in IMAGE_COLUMNS:
        if column in row.index:
            raw_value = safe_str(row[column]).strip()
            if raw_value:
                for token in split_possible_paths(raw_value):
                    resolved = resolve_path(
                        token,
                        images_dir=images_dir,
                        work_dir=work_dir,
                    )
                    if resolved and resolved not in found:
                        found.append(resolved)

    # Robust fallback to common filename patterns.
    for stem in candidate_image_names(question_number, question_id):
        for extension in IMAGE_EXTENSIONS:
            candidate = images_dir / f"{stem}{extension}"
            if candidate.exists() and candidate.is_file():
                resolved = candidate.resolve()
                if resolved not in found:
                    found.append(resolved)

    return found


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
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def collect_image_detection(
    questions: pd.DataFrame,
    images_dir: Path,
    work_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, row in questions.iterrows():
        paths = find_images_for_question(
            row,
            images_dir=images_dir,
            work_dir=work_dir,
        )
        rows.append({
            "question_id": safe_str(row["question_id"]).strip(),
            "original_question_number": int(row["original_question_number"]),
            "has_image_declared": parse_bool(row.get("has_image"), default=False),
            "has_image_detected": len(paths) > 0,
            "n_images_detected": len(paths),
            "image_paths": ";".join(str(path) for path in paths),
        })

    return pd.DataFrame(rows)


# ============================================================
# Prompt and multimodal request construction
# ============================================================

def format_question(row: pd.Series, n_images: int) -> str:
    question_number = int(row["original_question_number"])

    image_note = ""
    if n_images == 1:
        image_note = (
            "\nIMAGE NOTE: One image belonging to this question is attached "
            "immediately after this question block."
        )
    elif n_images > 1:
        image_note = (
            f"\nIMAGE NOTE: {n_images} images belonging to this question are attached "
            "immediately after this question block, in the displayed order."
        )

    return f"""Question {question_number}:{image_note}

{safe_str(row["question_text"]).strip()}

Options:
1. {safe_str(row["option_1"]).strip()}
2. {safe_str(row["option_2"]).strip()}
3. {safe_str(row["option_3"]).strip()}
4. {safe_str(row["option_4"]).strip()}"""


def build_multimodal_request(
    questions: pd.DataFrame,
    images_dir: Path,
    work_dir: Path,
) -> Tuple[List[Dict[str, Any]], str, pd.DataFrame]:
    """
    Build one user message with interleaved text and images.

    Returns:
    - API content parts, including base64 data URLs;
    - canonical textual log with image placeholders and hashes;
    - image manifest with request ordering information.
    """
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": PROMPT_PREAMBLE}
    ]
    canonical_parts: List[str] = [PROMPT_PREAMBLE]
    manifest_rows: List[Dict[str, Any]] = []

    image_order = 0

    for _, row in questions.iterrows():
        question_id = safe_str(row["question_id"]).strip()
        question_number = int(row["original_question_number"])
        image_paths = find_images_for_question(
            row,
            images_dir=images_dir,
            work_dir=work_dir,
        )

        question_text = format_question(row, n_images=len(image_paths))
        separator = "\n\n---\n\n" if len(content) > 1 else "\n\n"

        text_part = separator + question_text
        content.append({"type": "text", "text": text_part})
        canonical_parts.append(text_part)

        for image_index_within_question, image_path in enumerate(image_paths, start=1):
            image_order += 1
            mime_type = mime_type_for_image(image_path)
            image_hash = sha256_file(image_path)
            byte_size = image_path.stat().st_size

            marker = (
                f"\n[IMAGE {image_index_within_question}/{len(image_paths)} "
                f"FOR QUESTION {question_number}: "
                f"{image_path.name}; mime={mime_type}; "
                f"sha256={image_hash}; bytes={byte_size}]\n"
            )
            content.append({"type": "text", "text": marker})
            canonical_parts.append(marker)

            content_part_index = len(content)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(image_path),
                },
            })

            manifest_rows.append({
                "question_id": question_id,
                "original_question_number": question_number,
                "image_index_within_question": image_index_within_question,
                "n_images_for_question": len(image_paths),
                "image_order_within_request": image_order,
                "content_part_index_1_based": content_part_index + 1,
                "image_filename": image_path.name,
                "image_path": str(image_path),
                "image_mime_type": mime_type,
                "image_bytes": byte_size,
                "image_sha256": image_hash,
                "sent_to_model": True,
            })

    final_text = "\n\n---\n\n" + FINAL_REMINDER
    content.append({"type": "text", "text": final_text})
    canonical_parts.append(final_text)

    canonical_log = "".join(canonical_parts)
    manifest = pd.DataFrame(manifest_rows)

    return content, canonical_log, manifest


def build_answer_schema(n_questions: int) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "fir2024_whole_exam_answers",
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
                                "question_number": {
                                    "type": "integer",
                                },
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


# ============================================================
# Parsing and scoring
# ============================================================

def parse_and_validate_output(
    raw_content: Optional[str],
    expected_question_numbers: Sequence[int],
) -> Tuple[Dict[int, Dict[str, Any]], Optional[str], Optional[Dict[str, Any]]]:
    if raw_content is None or not raw_content.strip():
        return {}, "missing_content", None

    try:
        data = json.loads(raw_content)
    except Exception as exc:
        return {}, f"json_decode_error: {repr(exc)}", None

    if not isinstance(data, dict):
        return {}, "json_not_object", None

    answers = data.get("answers")
    if not isinstance(answers, list):
        return {}, "answers_missing_or_not_list", data

    records: Dict[int, Dict[str, Any]] = {}
    duplicate_numbers: List[int] = []
    invalid_items: List[str] = []

    for index, item in enumerate(answers):
        if not isinstance(item, dict):
            invalid_items.append(f"index_{index}:not_object")
            continue

        question_number = normalize_qnum(item.get("question_number"))
        answer = normalize_answer(item.get("answer"))

        if question_number is None:
            invalid_items.append(f"index_{index}:invalid_question_number")
            continue
        if answer not in VALID_ANSWERS:
            invalid_items.append(
                f"Q{question_number}:invalid_answer={repr(item.get('answer'))}"
            )
            continue
        if question_number in records:
            duplicate_numbers.append(question_number)
            continue

        records[question_number] = {
            "raw_answer": item.get("answer"),
            "model_answer": answer,
            "raw_record": item,
        }

    expected = list(expected_question_numbers)
    expected_set = set(expected)
    returned_set = set(records)

    missing = sorted(expected_set - returned_set)
    unexpected = sorted(returned_set - expected_set)

    issues: List[str] = []
    if len(answers) != len(expected):
        issues.append(
            f"wrong_answer_record_count:{len(answers)}_expected:{len(expected)}"
        )
    if invalid_items:
        issues.append("invalid_items:" + "|".join(invalid_items[:20]))
    if duplicate_numbers:
        issues.append(
            "duplicate_question_numbers:"
            + ",".join(map(str, sorted(set(duplicate_numbers))))
        )
    if missing:
        issues.append(
            "missing_question_numbers:" + ",".join(map(str, missing))
        )
    if unexpected:
        issues.append(
            "unexpected_question_numbers:" + ",".join(map(str, unexpected))
        )

    if issues:
        return records, "; ".join(issues), data

    return records, None, data


def score_answer(
    model_answer: Optional[str],
    correct_answer: Optional[str],
    official_available: bool,
) -> Dict[str, Any]:
    if not official_available:
        return {
            "has_official_answer": False,
            "is_correct": None,
            "is_incorrect": None,
            "is_blank": model_answer == "blank",
            "score": None,
            "answer_status": "not_evaluable",
        }

    if model_answer is None or model_answer not in VALID_ANSWERS:
        return {
            "has_official_answer": True,
            "is_correct": False,
            "is_incorrect": False,
            "is_blank": False,
            "score": 0.0,
            "answer_status": "missing_or_invalid",
        }

    if model_answer == "blank":
        return {
            "has_official_answer": True,
            "is_correct": False,
            "is_incorrect": False,
            "is_blank": True,
            "score": 0.0,
            "answer_status": "blank",
        }

    if model_answer == correct_answer:
        return {
            "has_official_answer": True,
            "is_correct": True,
            "is_incorrect": False,
            "is_blank": False,
            "score": 1.0,
            "answer_status": "correct",
        }

    return {
        "has_official_answer": True,
        "is_correct": False,
        "is_incorrect": True,
        "is_blank": False,
        "score": -1.0 / 3.0,
        "answer_status": "incorrect",
    }


# ============================================================
# API and retry logic
# ============================================================

def should_retry_exception(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True

    if isinstance(exc, APIStatusError):
        status_code = getattr(exc, "status_code", None)
        return status_code in {408, 409, 425, 429, 500, 502, 503, 504}

    return False


def backoff_seconds(
    attempt_index: int,
    base_sleep: float,
    max_backoff: float,
) -> float:
    exponential = base_sleep * (2 ** max(0, attempt_index - 1))
    return min(exponential, max_backoff) + random.uniform(0.0, 0.75)


def call_model_once(
    client: OpenAI,
    user_content: List[Dict[str, Any]],
    response_format: Dict[str, Any],
    max_tokens: int,
) -> Dict[str, Any]:
    """Perform one streamed multimodal call and audit hidden reasoning."""
    started = time.perf_counter()

    stream = client.chat.completions.create(
        model=MODEL_REQUESTED,
        messages=[
            {
                "role": "user",
                "content": user_content,
            }
        ],
        response_format=response_format,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        n=N,
        presence_penalty=PRESENCE_PENALTY,
        frequency_penalty=FREQUENCY_PENALTY,
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=max_tokens,
        extra_body={"thinking": THINKING},
        # No tools, web search, files, PDF, or retrieval.
    )

    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    chunk_dumps: List[Dict[str, Any]] = []
    model_returned: Optional[str] = None
    response_id: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = {}

    for chunk in stream:
        chunk_dump = object_to_dict(chunk)
        chunk_dumps.append(chunk_dump)

        model_returned = model_returned or getattr(chunk, "model", None)
        response_id = response_id or getattr(chunk, "id", None)

        if getattr(chunk, "usage", None) is not None:
            usage = object_to_dict(chunk.usage)

        for choice in (getattr(chunk, "choices", None) or []):
            if getattr(choice, "finish_reason", None) is not None:
                finish_reason = choice.finish_reason

            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            content = getattr(delta, "content", None)
            if content:
                content_parts.append(content)

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)

    raw_content = "".join(content_parts)
    reasoning_content = "".join(reasoning_parts)
    usage_fields = extract_usage_fields(usage)

    reasoning_tokens_raw = usage_fields.get("reasoning_tokens")
    try:
        reasoning_tokens_numeric = (
            float(reasoning_tokens_raw)
            if reasoning_tokens_raw is not None
            else 0.0
        )
    except (TypeError, ValueError):
        reasoning_tokens_numeric = float("inf")

    reasoning_content_observed = bool(reasoning_content.strip())
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
        "raw_content": raw_content,
        "reasoning_content": reasoning_content or None,
        "reasoning_content_observed": reasoning_content_observed,
        "any_reasoning_tokens_reported": any_reasoning_tokens_reported,
        "residual_reasoning_token_only": residual_reasoning_token_only,
        "substantive_reasoning_observed": substantive_reasoning_observed,
        "reasoning_observed": substantive_reasoning_observed,
        "reasoning_tokens_numeric": (
            None if reasoning_tokens_raw is None
            else reasoning_tokens_numeric
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
        **usage_fields,
    }


def call_model_with_retries(
    client: OpenAI,
    user_content: List[Dict[str, Any]],
    response_format: Dict[str, Any],
    expected_question_numbers: Sequence[int],
    max_tokens: int,
    max_retries: int,
    base_sleep: float,
    max_backoff: float,
) -> Dict[str, Any]:
    attempt_log: List[Dict[str, Any]] = []
    overall_started = time.perf_counter()

    last_result: Dict[str, Any] = {}
    last_records: Dict[int, Dict[str, Any]] = {}
    last_parse_error: Optional[str] = None
    last_parsed_json: Optional[Dict[str, Any]] = None

    for attempt_index in range(1, max_retries + 1):
        try:
            result = call_model_once(
                client=client,
                user_content=user_content,
                response_format=response_format,
                max_tokens=max_tokens,
            )
            last_result = result

            if result.get("substantive_reasoning_observed"):
                message = (
                    "PROTOCOL VIOLATION: substantive reasoning observed "
                    "despite thinking being explicitly disabled. The run "
                    f"allows at most {RESIDUAL_REASONING_TOKEN_ALLOWANCE} "
                    "reported residual reasoning token when "
                    "reasoning_content is empty."
                )
                attempt_log.append({
                    "attempt": attempt_index,
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
                    "strict_output_valid": False,
                    "protocol_violation": True,
                    "answer_records": {},
                    "parsed_json": None,
                    "parse_error": None,
                    "api_error": message,
                    "error_type": "UnexpectedSubstantiveReasoningObserved",
                    "attempts_used": attempt_index,
                    "attempt_log": attempt_log,
                    "latency_seconds_including_retries": (
                        time.perf_counter() - overall_started
                    ),
                }

            records, parse_error, parsed_json = parse_and_validate_output(
                raw_content=result.get("raw_content"),
                expected_question_numbers=expected_question_numbers,
            )
            last_records = records
            last_parse_error = parse_error
            last_parsed_json = parsed_json

            valid_finish_reason = result.get("finish_reason") == "stop"

            if parse_error is None and valid_finish_reason:
                return {
                    **result,
                    "ok": True,
                    "strict_output_valid": True,
                    "protocol_violation": False,
                    "answer_records": records,
                    "parsed_json": parsed_json,
                    "parse_error": None,
                    "api_error": None,
                    "error_type": None,
                    "attempts_used": attempt_index,
                    "attempt_log": attempt_log,
                    "latency_seconds_including_retries": (
                        time.perf_counter() - overall_started
                    ),
                }

            issues: List[str] = []
            if parse_error:
                issues.append(parse_error)
            if not valid_finish_reason:
                issues.append(
                    "unexpected_finish_reason:"
                    f"{result.get('finish_reason')}"
                )
            issue = "; ".join(issues)
            retry = attempt_index < max_retries

            attempt_log.append({
                "attempt": attempt_index,
                "error_type": "StructuredOutputValidationError",
                "error": issue,
                "finish_reason": result.get("finish_reason"),
                "attempt_latency_seconds": result.get(
                    "attempt_latency_seconds"
                ),
                "reasoning_audit_class": result.get(
                    "reasoning_audit_class"
                ),
                "retry": retry,
            })

            if retry:
                time.sleep(
                    backoff_seconds(
                        attempt_index=attempt_index,
                        base_sleep=base_sleep,
                        max_backoff=max_backoff,
                    )
                )
                continue

        except BadRequestError as exc:
            attempt_log.append({
                "attempt": attempt_index,
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "status_code": getattr(exc, "status_code", None),
                "retry": False,
            })
            return {
                "ok": False,
                "strict_output_valid": False,
                "protocol_violation": False,
                "answer_records": {},
                "parsed_json": None,
                "parse_error": None,
                "api_error": repr(exc),
                "error_type": type(exc).__name__,
                "attempts_used": attempt_index,
                "attempt_log": attempt_log,
                "traceback": traceback.format_exc(),
                "latency_seconds_including_retries": (
                    time.perf_counter() - overall_started
                ),
            }

        except Exception as exc:
            retry = (
                should_retry_exception(exc)
                and attempt_index < max_retries
            )
            attempt_log.append({
                "attempt": attempt_index,
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "status_code": getattr(exc, "status_code", None),
                "retry": retry,
            })

            if retry:
                time.sleep(
                    backoff_seconds(
                        attempt_index=attempt_index,
                        base_sleep=base_sleep,
                        max_backoff=max_backoff,
                    )
                )
                continue

            return {
                "ok": False,
                "strict_output_valid": False,
                "protocol_violation": False,
                "answer_records": {},
                "parsed_json": None,
                "parse_error": None,
                "api_error": repr(exc),
                "error_type": type(exc).__name__,
                "attempts_used": attempt_index,
                "attempt_log": attempt_log,
                "traceback": traceback.format_exc(),
                "latency_seconds_including_retries": (
                    time.perf_counter() - overall_started
                ),
            }

    return {
        **last_result,
        "ok": False,
        "strict_output_valid": False,
        "protocol_violation": False,
        "answer_records": last_records,
        "parsed_json": last_parsed_json,
        "parse_error": last_parse_error,
        "api_error": None,
        "error_type": "StructuredOutputValidationError",
        "attempts_used": max_retries,
        "attempt_log": attempt_log,
        "latency_seconds_including_retries": (
            time.perf_counter() - overall_started
        ),
    }


# ============================================================
# Output helpers
# ============================================================

def extract_usage_fields(usage: Dict[str, Any]) -> Dict[str, Any]:
    prompt_details = (
        usage.get("prompt_tokens_details")
        if isinstance(usage.get("prompt_tokens_details"), dict)
        else {}
    )
    completion_details = (
        usage.get("completion_tokens_details")
        if isinstance(usage.get("completion_tokens_details"), dict)
        else {}
    )

    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": completion_details.get("reasoning_tokens"),
        "cached_tokens": (
            prompt_details.get("cached_tokens")
            or usage.get("cached_tokens")
            or usage.get("prompt_cache_hit_tokens")
        ),
        "total_tokens": usage.get("total_tokens"),
    }


def summarize_replicate(
    results: pd.DataFrame,
    call_level: pd.DataFrame,
    replicate: int,
) -> Dict[str, Any]:
    group = results[results["replicate"] == replicate].copy()
    evaluable = group[group["has_official_answer"] == True].copy()

    call_rows = call_level[call_level["replicate"] == replicate]
    call = call_rows.iloc[0].to_dict() if len(call_rows) else {}

    n_evaluable = len(evaluable)
    n_correct = int((evaluable["answer_status"] == "correct").sum())
    n_incorrect = int((evaluable["answer_status"] == "incorrect").sum())
    n_blank = int((evaluable["answer_status"] == "blank").sum())
    n_missing_invalid = int(
        (evaluable["answer_status"] == "missing_or_invalid").sum()
    )

    return {
        "replicate": replicate,
        "model_requested": call.get("model_requested"),
        "model_returned": call.get("model_returned"),
        "n_questions_total": len(group),
        "n_evaluable": n_evaluable,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_blank": n_blank,
        "missing_or_invalid": n_missing_invalid,
        "not_evaluable": int(
            (group["answer_status"] == "not_evaluable").sum()
        ),
        "accuracy_evaluable": (
            n_correct / n_evaluable if n_evaluable else None
        ),
        "score": evaluable["score"].sum(skipna=True),
        "api_errors": int(bool(call.get("api_error"))),
        "whole_response_parse_errors": int(
            bool(call.get("whole_response_parse_error"))
        ),
        "n_answer_records_returned": call.get(
            "n_answer_records_returned"
        ),
        "n_unique_answer_records_returned": call.get(
            "n_unique_answer_records_returned"
        ),
        "n_image_questions_sent": call.get("n_image_questions_sent"),
        "n_images_sent": call.get("n_images_sent"),
        "attempts_used": call.get("attempts_used"),
        "latency_seconds": call.get("latency_seconds"),
        "final_attempt_latency_seconds": call.get(
            "final_attempt_latency_seconds"
        ),
        "prompt_tokens": call.get("prompt_tokens"),
        "completion_tokens": call.get("completion_tokens"),
        "reasoning_tokens": call.get("reasoning_tokens"),
        "cached_tokens": call.get("cached_tokens"),
        "total_tokens": call.get("total_tokens"),
        "finish_reason": call.get("finish_reason"),
        "response_status": call.get("response_status"),
        "reasoning_content_observed": call.get(
            "reasoning_content_observed"
        ),
        "residual_reasoning_token_only": call.get(
            "residual_reasoning_token_only"
        ),
        "substantive_reasoning_observed": call.get(
            "substantive_reasoning_observed"
        ),
        "reasoning_audit_class": call.get("reasoning_audit_class"),
        "protocol_violation": call.get("protocol_violation"),
    }


def summarize_by_image_status(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for replicate in sorted(
        results["replicate"].dropna().astype(int).unique().tolist()
    ):
        replicate_data = results[results["replicate"] == replicate]

        for has_image, label in [
            (False, "text_only"),
            (True, "image"),
        ]:
            group = replicate_data[
                replicate_data["has_image_detected"] == has_image
            ].copy()
            evaluable = group[
                group["has_official_answer"] == True
            ].copy()

            n_evaluable = len(evaluable)
            n_correct = int(
                (evaluable["answer_status"] == "correct").sum()
            )
            n_incorrect = int(
                (evaluable["answer_status"] == "incorrect").sum()
            )
            n_blank = int(
                (evaluable["answer_status"] == "blank").sum()
            )
            n_missing_invalid = int(
                (
                    evaluable["answer_status"]
                    == "missing_or_invalid"
                ).sum()
            )

            rows.append({
                "replicate": replicate,
                "image_status": label,
                "has_image_detected": has_image,
                "n_questions_total": len(group),
                "n_evaluable": n_evaluable,
                "n_correct": n_correct,
                "n_incorrect": n_incorrect,
                "n_blank": n_blank,
                "missing_or_invalid": n_missing_invalid,
                "not_evaluable": int(
                    (group["answer_status"] == "not_evaluable").sum()
                ),
                "accuracy_evaluable": (
                    n_correct / n_evaluable
                    if n_evaluable
                    else None
                ),
                "score": evaluable["score"].sum(skipna=True),
            })

    return pd.DataFrame(rows)


def write_outputs(
    all_rows: List[Dict[str, Any]],
    call_rows: List[Dict[str, Any]],
    scored_results_file: Path,
    call_level_file: Path,
    summary_file: Path,
    image_status_summary_file: Path,
) -> None:
    results = pd.DataFrame(all_rows)
    calls = pd.DataFrame(call_rows)

    results.to_csv(
        scored_results_file,
        index=False,
        encoding="utf-8-sig",
    )
    calls.to_csv(
        call_level_file,
        index=False,
        encoding="utf-8-sig",
    )

    completed_replicates = sorted(
        results["replicate"].dropna().astype(int).unique().tolist()
    )
    summary_rows = [
        summarize_replicate(
            results=results,
            call_level=calls,
            replicate=replicate,
        )
        for replicate in completed_replicates
    ]
    pd.DataFrame(summary_rows).to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    summarize_by_image_status(results).to_csv(
        image_status_summary_file,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Pilot/probe using the first N selected questions.",
    )
    parser.add_argument(
        "--start-q",
        type=int,
        default=None,
        help="First original question number to include.",
    )
    parser.add_argument(
        "--end-q",
        type=int,
        default=None,
        help="Last original question number to include.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=N_REPLICATES_DEFAULT,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS_DEFAULT,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=REQUEST_TIMEOUT_SECONDS_DEFAULT,
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES_DEFAULT,
        help="Maximum total attempts per replicate, including the first attempt.",
    )
    parser.add_argument(
        "--base-sleep",
        type=float,
        default=BASE_SLEEP_SECONDS_DEFAULT,
    )
    parser.add_argument(
        "--max-backoff",
        type=float,
        default=MAX_BACKOFF_SECONDS_DEFAULT,
    )
    parser.add_argument(
        "--between-replicates-sleep",
        type=float,
        default=BETWEEN_REPLICATES_SLEEP_DEFAULT,
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(DEFAULT_IMAGES_DIR),
    )
    parser.add_argument(
        "--require-expected-image-count",
        type=int,
        default=20,
        help=(
            "Stop unless exactly this many selected questions have at least "
            "one detected image. Default: 20 for FIR 2024."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs/images and write manifests without calling the API.",
    )
    args = parser.parse_args()

    if args.replicates < 1:
        raise ValueError("--replicates must be >= 1")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be >= 1")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be >= 1")

    load_dotenv()

    api_key = (
        os.getenv("MOONSHOT_API_KEY")
        or os.getenv("KIMI_API_KEY")
    )
    base_url = os.getenv(
        "KIMI_BASE_URL",
        DEFAULT_BASE_URL,
    )

    if not api_key and not args.dry_run:
        print(
            "ERROR: Missing MOONSHOT_API_KEY or KIMI_API_KEY in .env"
        )
        sys.exit(1)

    work_dir = Path.cwd()
    images_dir = Path(args.images_dir)
    if not images_dir.is_absolute():
        images_dir = work_dir / images_dir

    questions_all, answer_key = load_inputs()
    questions = filter_questions(
        questions=questions_all,
        limit=args.limit,
        start_q=args.start_q,
        end_q=args.end_q,
    )

    if questions.empty:
        raise ValueError("No questions selected.")

    answer_key_map = build_answer_key_map(answer_key)
    expected_question_numbers = [
        int(value)
        for value in questions["original_question_number"].tolist()
    ]

    image_detection = collect_image_detection(
        questions=questions,
        images_dir=images_dir,
        work_dir=work_dir,
    )
    n_image_questions_detected = int(
        image_detection["has_image_detected"].sum()
    )
    n_declared_image_questions_missing = int(
        (
            image_detection["has_image_declared"]
            & ~image_detection["has_image_detected"]
        ).sum()
    )

    if n_declared_image_questions_missing > 0:
        print("ERROR: One or more declared image questions have no detected image.")
        print(
            image_detection[
                image_detection["has_image_declared"]
                & ~image_detection["has_image_detected"]
            ].to_string(index=False)
        )
        sys.exit(2)

    if (
        args.require_expected_image_count is not None
        and n_image_questions_detected
        != args.require_expected_image_count
    ):
        print("ERROR: Unexpected number of image questions detected.")
        print("Expected:", args.require_expected_image_count)
        print("Detected:", n_image_questions_detected)
        print()
        print("Detected image questions:")
        print(
            image_detection[
                image_detection["has_image_detected"]
            ].to_string(index=False)
        )
        sys.exit(2)

    user_content, canonical_prompt_log, image_manifest = (
        build_multimodal_request(
            questions=questions,
            images_dir=images_dir,
            work_dir=work_dir,
        )
    )

    if len(image_manifest):
        n_images_sent = int(len(image_manifest))
    else:
        n_images_sent = 0

    is_full_definitive_selection = (
        args.limit is None
        and args.start_q is None
        and args.end_q is None
        and len(questions) == 210
        and n_image_questions_detected == 20
        and n_declared_image_questions_missing == 0
        and n_images_sent == 20
        and args.replicates == 2
    )

    request_hash_material = {
        "canonical_prompt_log_sha256": sha256_text(
            canonical_prompt_log
        ),
        "images": (
            image_manifest[
                [
                    "original_question_number",
                    "image_order_within_request",
                    "image_sha256",
                ]
            ].to_dict(orient="records")
            if len(image_manifest)
            else []
        ),
    }
    request_hash_sha256 = sha256_text(
        json.dumps(
            request_hash_material,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    response_format = build_answer_schema(
        n_questions=len(questions)
    )

    timestamp = utc_now_compact()
    suffixes: List[str] = []
    if args.limit is not None:
        suffixes.append(f"limit{args.limit}")
    if args.start_q is not None or args.end_q is not None:
        suffixes.append(
            f"q{args.start_q or 'start'}-{args.end_q or 'end'}"
        )
    selection_suffix = (
        "_" + "_".join(suffixes)
        if suffixes
        else ""
    )

    run_name = (
        "fir2024_kimi_k2_6_wholetext_textplusimage_ansonly_nonthinking_"
        f"{args.replicates}r_STRICT"
        f"{selection_suffix}_{timestamp}"
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else OUTPUT_ROOT / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    raw_outputs_file = (
        output_dir / f"{run_name}_raw_outputs.jsonl"
    )
    scored_results_file = (
        output_dir / f"{run_name}_scored_results.csv"
    )
    summary_file = (
        output_dir / f"{run_name}_summary_by_replicate.csv"
    )
    metadata_file = (
        output_dir / f"{run_name}_run_metadata.json"
    )
    call_level_file = (
        output_dir / f"{run_name}_call_level.csv"
    )
    image_manifest_file = (
        output_dir / f"{run_name}_image_manifest.csv"
    )
    image_detection_file = (
        output_dir / f"{run_name}_image_detection.csv"
    )
    image_status_summary_file = (
        output_dir / f"{run_name}_summary_by_image_status.csv"
    )
    prompt_log_file = (
        output_dir / f"{run_name}_prompt_log.txt"
    )

    image_detection.to_csv(
        image_detection_file,
        index=False,
        encoding="utf-8-sig",
    )
    image_manifest.to_csv(
        image_manifest_file,
        index=False,
        encoding="utf-8-sig",
    )
    prompt_log_file.write_text(
        canonical_prompt_log,
        encoding="utf-8",
    )

    metadata: Dict[str, Any] = {
        "run_name": run_name,
        "created_utc": timestamp,
        "provider": PROVIDER,
        "interface": INTERFACE,
        "api_endpoint": API_ENDPOINT,
        "base_url": base_url,
        "model_requested": MODEL_REQUESTED,
        "condition": CONDITION,
        "input_modality": "text+image",
        "output_mode": "answers_only",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "script_version": SCRIPT_VERSION,
        "prompt_preamble": PROMPT_PREAMBLE,
        "prompt_preamble_hash_sha256": sha256_text(
            PROMPT_PREAMBLE
        ),
        "canonical_prompt_log_hash_sha256": sha256_text(
            canonical_prompt_log
        ),
        "request_hash_sha256": request_hash_sha256,
        "prompt_log_file": str(prompt_log_file),
        "questions_file": str(QUESTIONS_FILE),
        "questions_file_sha256": sha256_file(QUESTIONS_FILE),
        "answer_key_file": str(ANSWER_KEY_FILE),
        "answer_key_file_sha256": sha256_file(
            ANSWER_KEY_FILE
        ),
        "images_dir": str(images_dir),
        "n_questions_in_file": int(len(questions_all)),
        "n_questions_attempted": int(len(questions)),
        "expected_question_numbers": expected_question_numbers,
        "n_image_questions_detected": (
            n_image_questions_detected
        ),
        "n_declared_image_questions_missing": (
            n_declared_image_questions_missing
        ),
        "n_images_attached_per_replicate": n_images_sent,
        "n_replicates": args.replicates,
        "is_full_definitive_selection": is_full_definitive_selection,
        "reasoning_audit_rule": {
            "abort_if_reasoning_content_nonempty": True,
            "abort_if_reasoning_tokens_above": RESIDUAL_REASONING_TOKEN_ALLOWANCE,
            "residual_reasoning_token_allowance": RESIDUAL_REASONING_TOKEN_ALLOWANCE,
        },
        "selection": {
            "limit": args.limit,
            "start_q": args.start_q,
            "end_q": args.end_q,
        },
        "generation_parameters": {
            "thinking": THINKING,
            "thinking_policy": "explicitly_disabled",
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "presence_penalty": PRESENCE_PENALTY,
            "frequency_penalty": FREQUENCY_PENALTY,
            "n": N,
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
            "image_placement": (
                "interleaved_immediately_after_corresponding_question"
            ),
            "rationales_requested": False,
        },
        "retry_policy": {
            "max_total_attempts_per_replicate": (
                args.max_retries
            ),
            "base_sleep_seconds": args.base_sleep,
            "max_backoff_seconds": args.max_backoff,
            "retry_transient_api_failures": True,
            "retry_invalid_or_incomplete_structured_outputs": True,
            "retry_valid_but_incorrect_answers": False,
        },
        "request_timeout_seconds": args.timeout,
        "between_replicates_sleep_seconds": (
            args.between_replicates_sleep
        ),
        "truststore_status": TRUSTSTORE_STATUS,
        "scoring": {
            "correct": 1,
            "incorrect": -1 / 3,
            "blank": 0,
            "missing_or_invalid": 0,
            "questions_without_official_answer_excluded": True,
        },
        "notes": (
            "Kimi K2.6 was run with thinking explicitly disabled. "
            "temperature=0.6 and top_p=0.95 were sent as the "
            "provider-supported non-thinking values. The full FIR exam "
            "and all detected images were sent in one user message per "
            "replicate. Exactly one residual provider-reported reasoning "
            "token with empty reasoning_content is accepted; any non-empty "
            "reasoning_content or more than one reasoning token aborts the run."
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
        "=== FIR 2024 Kimi K2.6 whole-exam text+image "
        "answers-only NON-THINKING ==="
    )
    print("Run name:", run_name)
    print("Output directory:", output_dir)
    print("Provider:", PROVIDER)
    print("Model:", MODEL_REQUESTED)
    print("Base URL:", base_url)
    print("Truststore:", TRUSTSTORE_STATUS)
    print("Condition:", CONDITION)
    print("Questions included:", len(questions))
    print(
        "Image questions detected:",
        n_image_questions_detected,
    )
    print(
        "Declared image questions missing:",
        n_declared_image_questions_missing,
    )
    print(
        "Images attached per replicate:",
        n_images_sent,
    )
    print("Replicates:", args.replicates)
    print(
        "Canonical prompt characters:",
        f"{len(canonical_prompt_log):,}",
    )
    print("thinking:", THINKING)
    print("temperature:", TEMPERATURE)
    print("top_p:", TOP_P)
    print(
        "Residual reasoning-token allowance:",
        RESIDUAL_REASONING_TOKEN_ALLOWANCE,
    )
    print("Full definitive selection:", is_full_definitive_selection)
    print("max_tokens:", args.max_tokens)
    print("response_format: json_schema strict")
    print("Rationales requested: False")
    print()

    if n_image_questions_detected:
        print("Detected image questions:")
        print(
            image_detection[
                image_detection["has_image_detected"]
            ][
                [
                    "original_question_number",
                    "question_id",
                    "n_images_detected",
                    "image_paths",
                ]
            ].to_string(index=False)
        )
        print()

    if args.dry_run:
        metadata["dry_run"] = True
        metadata["completed_utc"] = utc_now_compact()
        metadata["output_files"] = {
            "metadata": str(metadata_file),
            "image_manifest": str(image_manifest_file),
            "image_detection": str(image_detection_file),
            "prompt_log": str(prompt_log_file),
        }
        metadata_file.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("DRY RUN completed.")
        print("No API calls were made.")
        print("Files written:")
        print(" ", metadata_file)
        print(" ", image_manifest_file)
        print(" ", image_detection_file)
        print(" ", prompt_log_file)
        return

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=args.timeout,
        max_retries=0,
    )

    all_rows: List[Dict[str, Any]] = []
    call_rows: List[Dict[str, Any]] = []

    with raw_outputs_file.open(
        "w",
        encoding="utf-8",
    ) as raw_file:
        for replicate in range(1, args.replicates + 1):
            print(
                f"--- Replicate {replicate}/{args.replicates} ---",
                flush=True,
            )

            result = call_model_with_retries(
                client=client,
                user_content=user_content,
                response_format=response_format,
                expected_question_numbers=(
                    expected_question_numbers
                ),
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                base_sleep=args.base_sleep,
                max_backoff=args.max_backoff,
            )

            answer_records = (
                result.get("answer_records")
                if isinstance(
                    result.get("answer_records"),
                    dict,
                )
                else {}
            )
            usage = (
                result.get("usage")
                if isinstance(result.get("usage"), dict)
                else {}
            )
            usage_fields = extract_usage_fields(usage)

            raw_call_record = {
                "run_name": run_name,
                "provider": PROVIDER,
                "api_endpoint": API_ENDPOINT,
                "model_requested": MODEL_REQUESTED,
                "model_returned": result.get(
                    "model_returned"
                ),
                "replicate": replicate,
                "condition": CONDITION,
                "request_hash_sha256": request_hash_sha256,
                "n_questions_in_prompt": len(questions),
                "n_image_questions_sent": (
                    n_image_questions_detected
                ),
                "n_images_sent": n_images_sent,
                "n_answer_records_returned": len(
                    answer_records
                ),
                "n_unique_answer_records_returned": len(
                    set(answer_records)
                ),
                "strict_output_valid": result.get(
                    "strict_output_valid"
                ),
                "whole_response_parse_error": result.get(
                    "parse_error"
                ),
                "api_error": result.get("api_error"),
                "error_type": result.get("error_type"),
                "attempts_used": result.get(
                    "attempts_used"
                ),
                "attempt_log": result.get("attempt_log"),
                "latency_seconds_including_retries": (
                    result.get(
                        "latency_seconds_including_retries"
                    )
                ),
                "final_attempt_latency_seconds": (
                    result.get("attempt_latency_seconds")
                ),
                "finish_reason": result.get(
                    "finish_reason"
                ),
                "thinking_requested": THINKING,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
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
                "reasoning_audit_class": result.get(
                    "reasoning_audit_class"
                ),
                "protocol_violation": result.get(
                    "protocol_violation", False
                ),
                "max_tokens": args.max_tokens,
                "response_format": "json_schema_strict",
                "raw_content": result.get("raw_content"),
                "reasoning_content": result.get(
                    "reasoning_content"
                ),
                "parsed_json": result.get(
                    "parsed_json"
                ),
                "usage": usage,
                "n_stream_chunks": result.get("n_stream_chunks"),
                "chunk_dumps": result.get("chunk_dumps"),
            }
            raw_file.write(
                json.dumps(
                    raw_call_record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
            raw_file.flush()

            call_row = {
                "run_name": run_name,
                "provider": PROVIDER,
                "api_endpoint": API_ENDPOINT,
                "model_requested": MODEL_REQUESTED,
                "model_returned": result.get(
                    "model_returned"
                ),
                "replicate": replicate,
                "condition": CONDITION,
                "request_hash_sha256": request_hash_sha256,
                "n_questions_in_prompt": len(questions),
                "n_image_questions_sent": (
                    n_image_questions_detected
                ),
                "n_images_sent": n_images_sent,
                "n_answer_records_returned": len(
                    answer_records
                ),
                "n_unique_answer_records_returned": len(
                    set(answer_records)
                ),
                "strict_output_valid": result.get(
                    "strict_output_valid"
                ),
                "whole_response_parse_error": result.get(
                    "parse_error"
                ),
                "api_error": result.get("api_error"),
                "error_type": result.get("error_type"),
                "attempts_used": result.get(
                    "attempts_used"
                ),
                "latency_seconds": result.get(
                    "latency_seconds_including_retries"
                ),
                "final_attempt_latency_seconds": (
                    result.get("attempt_latency_seconds")
                ),
                "finish_reason": result.get(
                    "finish_reason"
                ),
                "response_status": (
                    "ok" if result.get("ok") else "failed"
                ),
                "thinking_requested": "disabled",
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
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
                "reasoning_audit_class": result.get(
                    "reasoning_audit_class"
                ),
                "protocol_violation": result.get(
                    "protocol_violation", False
                ),
                "n_stream_chunks": result.get("n_stream_chunks"),
                **usage_fields,
            }
            call_rows.append(call_row)

            for _, question_row in questions.iterrows():
                question_id = safe_str(
                    question_row["question_id"]
                ).strip()
                question_number = int(
                    question_row[
                        "original_question_number"
                    ]
                )

                key_info = answer_key_map.get(
                    question_id,
                    {
                        "correct_answer": None,
                        "official_answer_available": False,
                    },
                )
                correct_answer = key_info[
                    "correct_answer"
                ]
                official_available = bool(
                    key_info[
                        "official_answer_available"
                    ]
                )

                question_images = (
                    image_detection[
                        image_detection[
                            "original_question_number"
                        ]
                        == question_number
                    ].iloc[0]
                )
                has_image_detected = bool(
                    question_images[
                        "has_image_detected"
                    ]
                )
                n_images_detected = int(
                    question_images[
                        "n_images_detected"
                    ]
                )
                image_paths = question_images[
                    "image_paths"
                ]

                answer_record = answer_records.get(
                    question_number
                )
                model_answer = (
                    answer_record.get("model_answer")
                    if isinstance(answer_record, dict)
                    else None
                )
                raw_answer = (
                    answer_record.get("raw_answer")
                    if isinstance(answer_record, dict)
                    else None
                )

                item_parse_error = None
                if result.get("api_error") is not None:
                    item_parse_error = (
                        "api_error_no_answer"
                    )
                elif (
                    result.get("parse_error") is not None
                    and answer_record is None
                ):
                    item_parse_error = (
                        "whole_response_invalid_or_"
                        "incomplete_no_answer"
                    )
                elif answer_record is None:
                    item_parse_error = (
                        "missing_answer_for_question"
                    )
                elif model_answer not in VALID_ANSWERS:
                    item_parse_error = (
                        "invalid_answer_value:"
                        f"{repr(raw_answer)}"
                    )
                    model_answer = None

                scoring = score_answer(
                    model_answer=model_answer,
                    correct_answer=correct_answer,
                    official_available=official_available,
                )

                row_record = {
                    "run_name": run_name,
                    "provider": PROVIDER,
                    "api_endpoint": API_ENDPOINT,
                    "model_requested": MODEL_REQUESTED,
                    "model_returned": result.get(
                        "model_returned"
                    ),
                    "prompt_template_version": (
                        PROMPT_TEMPLATE_VERSION
                    ),
                    "replicate": replicate,
                    "condition": CONDITION,
                    "question_id": question_id,
                    "original_question_number": (
                        question_number
                    ),
                    "is_reserve": parse_bool(
                        question_row.get("is_reserve"),
                        default=False,
                    ),
                    "has_image_declared": parse_bool(
                        question_row.get("has_image"),
                        default=False,
                    ),
                    "has_image_detected": (
                        has_image_detected
                    ),
                    "n_images_detected": (
                        n_images_detected
                    ),
                    "image_paths": image_paths,
                    "images_sent": (
                        n_images_detected
                    ),
                    "request_hash_sha256": (
                        request_hash_sha256
                    ),
                    "model_answer": model_answer,
                    "raw_answer": raw_answer,
                    "rationale": None,
                    "correct_answer": correct_answer,
                    **scoring,
                    "parse_error": item_parse_error,
                    "api_error": result.get("api_error"),
                    "attempts_used": result.get(
                        "attempts_used"
                    ),
                    "thinking_requested": "disabled",
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "reasoning_content_observed": result.get(
                        "reasoning_content_observed", False
                    ),
                    "residual_reasoning_token_only": result.get(
                        "residual_reasoning_token_only", False
                    ),
                    "substantive_reasoning_observed": result.get(
                        "substantive_reasoning_observed", False
                    ),
                    "reasoning_audit_class": result.get(
                        "reasoning_audit_class"
                    ),
                    "strict_json_schema": True,
                    "latency_seconds": result.get(
                        "latency_seconds_including_retries"
                    ),
                    "finish_reason": result.get(
                        "finish_reason"
                    ),
                    **usage_fields,
                }
                all_rows.append(row_record)

            write_outputs(
                all_rows=all_rows,
                call_rows=call_rows,
                scored_results_file=(
                    scored_results_file
                ),
                call_level_file=call_level_file,
                summary_file=summary_file,
                image_status_summary_file=(
                    image_status_summary_file
                ),
            )

            results_df = pd.DataFrame(all_rows)
            calls_df = pd.DataFrame(call_rows)
            replicate_summary = summarize_replicate(
                results=results_df,
                call_level=calls_df,
                replicate=replicate,
            )
            print(
                json.dumps(
                    replicate_summary,
                    ensure_ascii=False,
                    indent=2,
                )
            )
            print()

            if result.get("protocol_violation", False):
                metadata["completed_utc"] = utc_now_compact()
                metadata["aborted_due_to_protocol_violation"] = True
                metadata["output_files"] = {
                    "raw_outputs": str(raw_outputs_file),
                    "scored_results": str(scored_results_file),
                    "summary_by_replicate": str(summary_file),
                    "call_level": str(call_level_file),
                    "image_manifest": str(image_manifest_file),
                    "image_detection": str(image_detection_file),
                    "summary_by_image_status": str(image_status_summary_file),
                    "prompt_log": str(prompt_log_file),
                }
                metadata_file.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    "RUN ABORTED: substantive reasoning was observed "
                    "despite thinking being explicitly disabled."
                )
                sys.exit(3)

            if (
                replicate < args.replicates
                and args.between_replicates_sleep > 0
            ):
                time.sleep(
                    args.between_replicates_sleep
                )

    metadata["completed_utc"] = utc_now_compact()

    calls_final = pd.DataFrame(call_rows)
    reasoning_values = pd.to_numeric(
        calls_final.get("reasoning_tokens", pd.Series(dtype=float)),
        errors="coerce",
    )
    reasoning_distribution = (
        reasoning_values.dropna().value_counts().sort_index()
    )
    reasoning_distribution_dict = {
        str(int(value) if float(value).is_integer() else float(value)): int(count)
        for value, count in reasoning_distribution.items()
    }

    metadata["definitive_run_completed_as_prespecified"] = (
        is_full_definitive_selection
        and len(call_rows) == 2
        and len(all_rows) == 420
        and all(bool(row.get("strict_output_valid")) for row in call_rows)
        and not any(bool(row.get("protocol_violation")) for row in call_rows)
    )
    metadata["observed_reasoning_audit"] = {
        "n_calls_accepted": int(len(call_rows)),
        "n_calls_with_reasoning_content": int(
            calls_final.get(
                "reasoning_content_observed", pd.Series(dtype=bool)
            ).fillna(False).astype(bool).sum()
        ),
        "n_calls_with_any_reasoning_tokens": int(
            calls_final.get(
                "any_reasoning_tokens_reported", pd.Series(dtype=bool)
            ).fillna(False).astype(bool).sum()
        ),
        "n_calls_with_residual_reasoning_token_only": int(
            calls_final.get(
                "residual_reasoning_token_only", pd.Series(dtype=bool)
            ).fillna(False).astype(bool).sum()
        ),
        "n_calls_with_substantive_reasoning": int(
            calls_final.get(
                "substantive_reasoning_observed", pd.Series(dtype=bool)
            ).fillna(False).astype(bool).sum()
        ),
        "n_protocol_violations": int(
            calls_final.get(
                "protocol_violation", pd.Series(dtype=bool)
            ).fillna(False).astype(bool).sum()
        ),
        "total_reasoning_tokens": (
            None
            if reasoning_values.dropna().empty
            else float(reasoning_values.sum())
        ),
        "reasoning_token_distribution": reasoning_distribution_dict,
        "residual_reasoning_token_allowance": (
            RESIDUAL_REASONING_TOKEN_ALLOWANCE
        ),
    }
    metadata["output_files"] = {
        "raw_outputs": str(raw_outputs_file),
        "scored_results": str(scored_results_file),
        "summary_by_replicate": str(summary_file),
        "call_level": str(call_level_file),
        "image_manifest": str(image_manifest_file),
        "image_detection": str(image_detection_file),
        "summary_by_image_status": str(
            image_status_summary_file
        ),
        "prompt_log": str(prompt_log_file),
    }
    metadata_file.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== Final summary ===")
    print(
        pd.read_csv(summary_file).to_string(index=False)
    )
    print()
    print("=== Summary by image status ===")
    print(
        pd.read_csv(
            image_status_summary_file
        ).to_string(index=False)
    )
    print()
    print("=== Non-thinking audit ===")
    print(
        calls_final[[
            "replicate",
            "reasoning_tokens",
            "reasoning_content_observed",
            "residual_reasoning_token_only",
            "substantive_reasoning_observed",
            "protocol_violation",
        ]].to_string(index=False)
    )
    print("Reasoning-token distribution:", reasoning_distribution_dict)
    print(
        "PRESPECIFIED 210 × 2 DEFINITIVE RUN:",
        "PASS"
        if metadata["definitive_run_completed_as_prespecified"]
        else "FAIL",
    )
    print()
    print("Files written:")
    print(" ", raw_outputs_file)
    print(" ", scored_results_file)
    print(" ", summary_file)
    print(" ", metadata_file)
    print(" ", call_level_file)
    print(" ", image_manifest_file)
    print(" ", image_detection_file)
    print(" ", image_status_summary_file)
    print(" ", prompt_log_file)


if __name__ == "__main__":
    main()

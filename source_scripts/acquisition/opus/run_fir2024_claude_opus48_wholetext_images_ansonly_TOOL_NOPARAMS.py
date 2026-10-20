import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
import anthropic


MODEL_REQUESTED = "claude-opus-4-8"
PROVIDER = "Anthropic Claude"
INTERFACE = "API"
API_ENDPOINT = "Messages API with forced client-side tool schema"

PROMPT_TEMPLATE_VERSION = "fir2024_whole_text_images_examlike_blank_answers_only_tool_schema_v1"
SCRIPT_VERSION = "fir2024_claude_opus48_whole_text_images_answers_only_tool_schema_no_sampling_params_v1"

GENERATION_PARAMETERS = {
    "temperature": "not_sent",
    "top_p": "not_sent",
    "max_tokens": 10000,
    "structured_output": "forced_tool_use_input_schema",
    "rationales_requested": False,
    "stream": False,
    "external_tools_enabled": False,
    "web_search_enabled": False,
    "file_input_enabled": False,
    "image_input_enabled": True,
    "image_transport": "base64_via_messages_image_block",
    "reasoning_effort": "not_sent",
}

REQUEST_TIMEOUT_SECONDS = 600
MAX_ATTEMPTS_PER_CALL = 3
RETRY_SLEEP_SECONDS = 20

QUESTIONS_FILE = Path("fir2024_questions_210.csv")
ANSWER_KEY_FILE = Path("fir2024_answer_key_210.csv")
IMAGE_ROOT = Path("images") / "fir"
OUTPUT_ROOT = Path("outputs")


PROMPT_PREAMBLE = """You are taking an examination for Pharmacy residency access.

This is a multiple-choice exam. Each question has four options, numbered 1 to 4, and only one is correct. In the official exam, correct answers score +1 point, incorrect answers score -1/3 points, and blank answers score 0 points. Act as a human candidate and try to get the highest score possible.

You will receive the full exam text, containing all questions and their four answer options. Some questions may include an image; when an image is provided, use it as part of the question.

Do not use web search, tools other than record_all_answers, external references, files, or an answer key.

Use the record_all_answers tool.

Tool input rules:
- Return exactly one answer object for every question included in the input.
- Use the integer question number shown in the input.
- The value of "answer" must be exactly one single string: "1", "2", "3", "4", or "blank".
- Do not return multiple options.
- Do not return values such as "1,2", "1 or 2", "3/4", "4,", lists, arrays, ranges, or explanations inside the "answer" field.
- If you cannot choose one single best option, return "blank".
- Do not include rationales, explanations, comments, markdown, or any other fields.

FULL EXAM TEXT:
"""


def build_answer_tool(n_questions: int) -> Dict[str, Any]:
    return {
        "name": "record_all_answers",
        "description": "Record all answers to a multiple-choice examination.",
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
    }


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def bool_like(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() in {"true", "1", "yes", "y", "si", "sí"}


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


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def object_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    try:
        return dict(obj)
    except Exception:
        return {"repr": repr(obj)}


def flatten_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(usage, dict):
        usage = {}

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    cache_creation_input_tokens = usage.get("cache_creation_input_tokens")
    cache_read_input_tokens = usage.get("cache_read_input_tokens")

    try:
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)
    except Exception:
        total_tokens = None

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": None,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
    }


def format_question(row: pd.Series) -> str:
    qn = int(row["original_question_number"])
    image_note = ""
    if bool_like(row.get("has_image")):
        image_note = "\n[Image for this question is attached immediately after this question block.]"

    return f"""Question {qn}:
{row['question_text']}

Options:
1. {row['option_1']}
2. {row['option_2']}
3. {row['option_3']}
4. {row['option_4']}{image_note}
"""


def resolve_image_path(row: pd.Series) -> Optional[Path]:
    if not bool_like(row.get("has_image")):
        return None

    raw_image_path = row.get("image_path")
    try:
        if raw_image_path is not None and not pd.isna(raw_image_path):
            text = str(raw_image_path).strip()
            if text and text.lower() not in {"nan", "none", "null"}:
                p = Path(text)
                if p.exists():
                    return p
    except Exception:
        pass

    qid = str(row["question_id"]).strip()
    qn = int(row["original_question_number"])

    candidates = [
        IMAGE_ROOT / f"{qid}.png",
        IMAGE_ROOT / f"{qid}.jpg",
        IMAGE_ROOT / f"{qid}.jpeg",
        IMAGE_ROOT / f"Q{qn:03d}.png",
        IMAGE_ROOT / f"Q{qn:03d}.jpg",
        IMAGE_ROOT / f"Q{qn:03d}.jpeg",
    ]

    for p in candidates:
        if p.exists():
            return p

    return candidates[0]


def image_block_from_path(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No existe la imagen requerida: {path}")

    media_type = mimetypes.guess_type(str(path))[0] or "image/png"

    with path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": b64,
        },
    }


def build_content_blocks_and_prompt_log(questions: pd.DataFrame) -> Tuple[list, str, list]:
    blocks = [{"type": "text", "text": PROMPT_PREAMBLE + "\n\n"}]
    prompt_log_parts = [PROMPT_PREAMBLE + "\n\n"]
    image_manifest = []

    for idx, (_, row) in enumerate(questions.iterrows(), start=1):
        q_text = format_question(row)
        separator = "\n---\n\n" if idx < len(questions) else "\n"

        blocks.append({"type": "text", "text": q_text})
        prompt_log_parts.append(q_text)

        image_path = resolve_image_path(row)
        has_image = image_path is not None

        image_sha256 = None
        image_size_bytes = None
        image_exists = False

        if image_path is not None:
            if not image_path.exists():
                raise FileNotFoundError(f"Pregunta con imagen pero archivo no encontrado: {row['question_id']} -> {image_path}")

            blocks.append(image_block_from_path(image_path))
            image_sha256 = sha256_file(image_path)
            image_size_bytes = image_path.stat().st_size
            image_exists = True

            prompt_log_parts.append(f"\n[ATTACHED IMAGE: {image_path}]\n")

        blocks.append({"type": "text", "text": separator})
        prompt_log_parts.append(separator)

        image_manifest.append({
            "question_id": row["question_id"],
            "original_question_number": int(row["original_question_number"]),
            "has_image_csv": bool_like(row.get("has_image")),
            "image_sent": has_image,
            "image_path": str(image_path) if image_path else None,
            "image_exists": image_exists,
            "image_sha256": image_sha256,
            "image_size_bytes": image_size_bytes,
        })

    return blocks, "".join(prompt_log_parts), image_manifest


def parse_tool_answers(response) -> Tuple[Dict[int, Dict[str, Any]], Optional[str], Dict[str, Any]]:
    tool_input = None

    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "record_all_answers":
            tool_input = block.input
            break

    if tool_input is None:
        return {}, "missing_record_all_answers_tool_use", {}

    if not isinstance(tool_input, dict):
        return {}, "tool_input_not_object", {"raw_tool_input": tool_input}

    answers = tool_input.get("answers")

    if not isinstance(answers, list):
        return {}, "answers_key_missing_or_not_list", {"raw_tool_input": tool_input}

    records = {}
    duplicate_qnums = []

    for item in answers:
        if not isinstance(item, dict):
            continue

        qn = normalize_qnum(item.get("question_number"))
        if qn is None:
            continue

        if qn in records:
            duplicate_qnums.append(qn)

        records[qn] = {
            "raw_answer": item.get("answer"),
            "raw_record": item,
        }

    diagnostics = {
        "raw_tool_input": tool_input,
        "n_answer_records_returned": len(answers),
        "n_unique_answer_records_returned": len(records),
        "duplicate_question_numbers": sorted(set(duplicate_qnums)),
    }

    return records, None, diagnostics


def call_claude(client, content_blocks: list, answer_tool: Dict[str, Any]) -> Dict[str, Any]:
    error_type = None
    error_message = None
    error_traceback = None

    response = None
    model_returned = None
    response_id = None
    stop_reason = None
    usage_dict = {}
    response_dump = {}
    answer_records = {}
    parse_error = None
    diagnostics = {}

    start = time.perf_counter()
    attempts_used = 0

    for attempt in range(1, MAX_ATTEMPTS_PER_CALL + 1):
        attempts_used = attempt

        try:
            response = client.messages.create(
                model=MODEL_REQUESTED,
                max_tokens=GENERATION_PARAMETERS["max_tokens"],
                messages=[{"role": "user", "content": content_blocks}],
                tools=[answer_tool],
                tool_choice={"type": "tool", "name": "record_all_answers"},
            )

            model_returned = getattr(response, "model", None)
            response_id = getattr(response, "id", None)
            stop_reason = getattr(response, "stop_reason", None)
            usage_dict = object_to_dict(getattr(response, "usage", None))
            response_dump = object_to_dict(response)

            answer_records, parse_error, diagnostics = parse_tool_answers(response)

            error_type = None
            error_message = None
            error_traceback = None
            break

        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            error_traceback = traceback.format_exc()

            print(f"    API ERROR attempt {attempt}/{MAX_ATTEMPTS_PER_CALL}: {error_type} - {error_message[:200]}")

            non_retryable_markers = [
                "authentication",
                "permission",
                "not_found",
                "invalid_request_error",
                "credit",
                "billing",
            ]

            if any(marker.lower() in error_message.lower() for marker in non_retryable_markers):
                break

            if attempt < MAX_ATTEMPTS_PER_CALL:
                time.sleep(RETRY_SLEEP_SECONDS)

    latency = time.perf_counter() - start
    usage_flat = flatten_usage(usage_dict)

    return {
        "model_returned": model_returned,
        "response_id": response_id,
        "stop_reason": stop_reason,
        "answer_records": answer_records,
        "whole_response_parse_error": parse_error,
        "diagnostics": diagnostics,
        "raw_response": response_dump,
        "usage": usage_dict,
        "error_type": error_type,
        "error_message": error_message,
        "error_traceback": error_traceback,
        "latency_seconds": latency,
        "attempts_used": attempts_used,
        **usage_flat,
    }


def get_official_available(row: pd.Series) -> bool:
    if "official_answer_available" in row.index:
        if not bool_like(row.get("official_answer_available")):
            return False

    if "is_annulled_or_without_official_answer" in row.index:
        if bool_like(row.get("is_annulled_or_without_official_answer")):
            return False

    return normalize_answer(row.get("correct_answer")) in {"1", "2", "3", "4"}


def score_answer(model_answer: Optional[str], correct_answer: Optional[str], official_available: bool):
    if not official_available or correct_answer is None:
        return "no_official_answer", None, None

    if model_answer is None:
        return "missing_or_invalid", False, 0.0

    if model_answer == "blank":
        return "blank", False, 0.0

    if model_answer == correct_answer:
        return "correct", True, 1.0

    return "incorrect", False, -1.0 / 3.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Probe, e.g. --limit 10")
    parser.add_argument("--replicates", type=int, default=2, help="Default 2")
    args = parser.parse_args()

    load_dotenv(dotenv_path=".env")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("No encuentro ANTHROPIC_API_KEY en .env")

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"No existe {QUESTIONS_FILE}")
    if not ANSWER_KEY_FILE.exists():
        raise FileNotFoundError(f"No existe {ANSWER_KEY_FILE}")

    questions = pd.read_csv(QUESTIONS_FILE)
    answer_key = pd.read_csv(ANSWER_KEY_FILE)

    questions = questions.sort_values("original_question_number").reset_index(drop=True)

    if len(questions) != 210:
        raise ValueError(f"Se esperaban 210 preguntas FIR, hay {len(questions)}")

    if args.limit is not None:
        questions = questions.head(args.limit).copy()

    answer_key["correct_answer"] = answer_key["correct_answer"].apply(normalize_answer)

    ak_cols = ["exam", "question_id", "correct_answer"]
    for col in ["official_answer_available", "is_annulled_or_without_official_answer"]:
        if col in answer_key.columns:
            ak_cols.append(col)

    answer_key_min = answer_key[ak_cols].copy()

    content_blocks, prompt_log, image_manifest = build_content_blocks_and_prompt_log(questions)
    answer_tool = build_answer_tool(len(questions))

    timestamp = utc_timestamp()
    safe_model = MODEL_REQUESTED.replace(".", "_").replace("-", "_")
    limit_suffix = f"_limit{args.limit}" if args.limit is not None else ""
    run_name = f"fir2024_{safe_model}_wholetext_images_ansonly_TOOL_{args.replicates}runs{limit_suffix}_{timestamp}"
    output_dir = OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_file = output_dir / f"{run_name}_raw_outputs.jsonl"
    scored_file = output_dir / f"{run_name}_scored_results.csv"
    summary_file = output_dir / f"{run_name}_summary_by_replicate.csv"
    image_summary_file = output_dir / f"{run_name}_summary_by_image_status.csv"
    metadata_file = output_dir / f"{run_name}_run_metadata.json"
    prompt_log_file = output_dir / f"{run_name}_prompt_log.txt"
    image_manifest_file = output_dir / f"{run_name}_image_manifest.csv"

    prompt_log_file.write_text(prompt_log, encoding="utf-8")
    pd.DataFrame(image_manifest).to_csv(image_manifest_file, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")

    metadata = {
        "run_name": run_name,
        "created_utc": timestamp,
        "provider": PROVIDER,
        "interface": INTERFACE,
        "api_endpoint": API_ENDPOINT,
        "model_requested": MODEL_REQUESTED,
        "condition": "whole_text_images_answers_only",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "script_version": SCRIPT_VERSION,
        "generation_parameters": GENERATION_PARAMETERS,
        "answer_tool_schema": answer_tool,
        "n_replicates": args.replicates,
        "n_questions_attempted": int(len(questions)),
        "questions_file": str(QUESTIONS_FILE),
        "answer_key_file": str(ANSWER_KEY_FILE),
        "questions_file_sha256": sha256_file(QUESTIONS_FILE),
        "answer_key_file_sha256": sha256_file(ANSWER_KEY_FILE),
        "prompt_log_file": str(prompt_log_file),
        "prompt_hash_sha256": sha256_text(prompt_log),
        "image_root": str(IMAGE_ROOT),
        "image_manifest_file": str(image_manifest_file),
        "tools_or_web_search_provided_to_model": False,
        "claude_sampling_note": (
            "Claude Opus evaluated as Anthropic-specific exploratory condition. "
            "temperature and top_p are not sent because Anthropic API rejected simultaneous specification."
        ),
    }

    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_file.write_text("", encoding="utf-8")

    client = anthropic.Anthropic(
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    all_rows = []
    call_rows = []

    print(f"Run name: {run_name}")
    print(f"Output directory: {output_dir}")
    print(f"Condition: FIR whole-text + images answers-only")
    print(f"Questions included: {len(questions)}")
    print(f"Images sent: {sum(1 for x in image_manifest if x['image_sent'])}")
    print(f"Replicates: {args.replicates}")
    print("Temperature sent: False")
    print("Top_p sent: False")
    print()

    for rep in range(1, args.replicates + 1):
        print(f"=== Replicate {rep}/{args.replicates} ===")

        request_time = datetime.now(timezone.utc).isoformat()
        api_result = call_claude(client, content_blocks, answer_tool)
        response_time = datetime.now(timezone.utc).isoformat()

        answer_records = api_result["answer_records"]
        diagnostics = api_result["diagnostics"] or {}

        raw_call_record = {
            "run_name": run_name,
            "replicate_id": rep,
            "provider": PROVIDER,
            "interface": INTERFACE,
            "api_endpoint": API_ENDPOINT,
            "model_requested": MODEL_REQUESTED,
            "model_returned": api_result.get("model_returned"),
            "response_id": api_result.get("response_id"),
            "stop_reason": api_result.get("stop_reason"),
            "condition": "whole_text_images_answers_only",
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "script_version": SCRIPT_VERSION,
            "n_questions_in_prompt": int(len(questions)),
            "n_answer_records_returned": diagnostics.get("n_answer_records_returned", len(answer_records)),
            "n_unique_answer_records_returned": diagnostics.get("n_unique_answer_records_returned", len(answer_records)),
            "duplicate_question_numbers": diagnostics.get("duplicate_question_numbers"),
            "whole_response_parse_error": api_result.get("whole_response_parse_error"),
            "error_type": api_result.get("error_type"),
            "error_message": api_result.get("error_message"),
            "error_traceback": api_result.get("error_traceback"),
            "latency_seconds": api_result.get("latency_seconds"),
            "attempts_used": api_result.get("attempts_used"),
            "input_tokens": api_result.get("input_tokens"),
            "output_tokens": api_result.get("output_tokens"),
            "total_tokens": api_result.get("total_tokens"),
            "reasoning_tokens": api_result.get("reasoning_tokens"),
            "cache_creation_input_tokens": api_result.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": api_result.get("cache_read_input_tokens"),
            "request_time_utc": request_time,
            "response_time_utc": response_time,
            "usage": api_result.get("usage"),
            "raw_tool_input": diagnostics.get("raw_tool_input"),
            "raw_response": api_result.get("raw_response"),
        }

        with raw_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(raw_call_record, ensure_ascii=False, default=str) + "\n")

        call_rows.append(raw_call_record)

        for _, row in questions.iterrows():
            qn = int(row["original_question_number"])
            rec = answer_records.get(qn)

            raw_answer = None
            model_answer = None
            parse_error = None
            raw_record = None

            if api_result.get("error_type"):
                parse_error = "api_error_no_answer"
            elif api_result.get("whole_response_parse_error") and rec is None:
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

            all_rows.append({
                "run_name": run_name,
                "replicate_id": rep,
                "provider": PROVIDER,
                "interface": INTERFACE,
                "api_endpoint": API_ENDPOINT,
                "model_requested": MODEL_REQUESTED,
                "model_returned": api_result.get("model_returned"),
                "condition": "whole_text_images_answers_only",
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "script_version": SCRIPT_VERSION,
                "exam": row["exam"],
                "question_id": row["question_id"],
                "original_question_number": qn,
                "is_reserve": bool_like(row.get("is_reserve")),
                "subject": row.get("subject"),
                "has_image": bool_like(row.get("has_image")),
                "source_pdf_page": safe_int(row.get("source_pdf_page")),
                "model_answer": model_answer,
                "raw_answer": raw_answer,
                "parse_error": parse_error,
                "error_type": api_result.get("error_type"),
                "error_message": api_result.get("error_message"),
                "whole_response_parse_error": api_result.get("whole_response_parse_error"),
                "raw_record": json.dumps(raw_record, ensure_ascii=False) if raw_record is not None else None,
                "response_id": api_result.get("response_id"),
                "stop_reason": api_result.get("stop_reason"),
                "latency_seconds_call": api_result.get("latency_seconds"),
                "attempts_used": api_result.get("attempts_used"),
                "input_tokens_call": api_result.get("input_tokens"),
                "output_tokens_call": api_result.get("output_tokens"),
                "total_tokens_call": api_result.get("total_tokens"),
                "reasoning_tokens_call": api_result.get("reasoning_tokens"),
            })

        print(
            f"rep={rep} status={'API_ERROR' if api_result.get('error_type') else 'ok'} "
            f"answers_returned={diagnostics.get('n_answer_records_returned', len(answer_records))} "
            f"unique={diagnostics.get('n_unique_answer_records_returned', len(answer_records))} "
            f"parse_error={api_result.get('whole_response_parse_error')} "
            f"model={api_result.get('model_returned')} "
            f"tokens={api_result.get('total_tokens')}"
        )

    results = pd.DataFrame(all_rows)

    scored = results.merge(answer_key_min, on=["exam", "question_id"], how="left")

    scored["model_answer"] = scored["model_answer"].apply(normalize_answer)
    scored["correct_answer"] = scored["correct_answer"].apply(normalize_answer)

    statuses = []
    is_corrects = []
    scores = []
    official_flags = []

    for _, row in scored.iterrows():
        official_available = get_official_available(row)
        status, is_correct, score = score_answer(
            model_answer=row.get("model_answer"),
            correct_answer=row.get("correct_answer"),
            official_available=official_available,
        )

        if row.get("error_type") not in [None, "", "nan"] and not pd.isna(row.get("error_type")):
            status = "api_error"
            is_correct = None
            score = None
        elif row.get("parse_error") not in [None, "", "nan"] and not pd.isna(row.get("parse_error")):
            if status != "no_official_answer":
                status = "parse_error"
                is_correct = None
                score = None

        statuses.append(status)
        is_corrects.append(is_correct)
        scores.append(score)
        official_flags.append(official_available)

    scored["has_official_answer"] = official_flags
    scored["scoring_status"] = statuses
    scored["is_correct"] = is_corrects
    scored["exam_score"] = scores

    scored.to_csv(scored_file, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")

    summary_rows = []

    for rep, group in scored.groupby("replicate_id"):
        evaluable = group[group["has_official_answer"] == True].copy()
        valid = evaluable[evaluable["scoring_status"].isin(["correct", "incorrect", "blank"])]

        n_eval = len(evaluable)
        n_correct = int((valid["scoring_status"] == "correct").sum())
        n_incorrect = int((valid["scoring_status"] == "incorrect").sum())
        n_blank = int((valid["scoring_status"] == "blank").sum())
        n_api_error = int((evaluable["scoring_status"] == "api_error").sum())
        n_parse_error = int((evaluable["scoring_status"] == "parse_error").sum())

        call = [x for x in call_rows if x["replicate_id"] == rep][0]

        summary_rows.append({
            "run_name": run_name,
            "replicate_id": rep,
            "model_requested": MODEL_REQUESTED,
            "model_returned_unique": ";".join(sorted([str(x) for x in group["model_returned"].dropna().unique()])),
            "n_total_questions_run": len(group),
            "n_evaluable_with_official_answer": n_eval,
            "n_valid_scored": len(valid),
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_blank": n_blank,
            "n_api_error": n_api_error,
            "n_parse_error": n_parse_error,
            "n_no_official_answer": int((group["has_official_answer"] == False).sum()),
            "n_answer_records_returned": call.get("n_answer_records_returned"),
            "n_unique_answer_records_returned": call.get("n_unique_answer_records_returned"),
            "whole_response_parse_error": call.get("whole_response_parse_error"),
            "accuracy_among_answered_nonblank": n_correct / (n_correct + n_incorrect) if (n_correct + n_incorrect) else None,
            "accuracy_among_evaluable_questions_counting_blank_as_not_correct": n_correct / n_eval if n_eval else None,
            "official_exam_score_sum": valid["exam_score"].sum(skipna=True),
            "latency_seconds": call.get("latency_seconds"),
            "attempts_used": call.get("attempts_used"),
            "input_tokens": call.get("input_tokens"),
            "output_tokens": call.get("output_tokens"),
            "total_tokens": call.get("total_tokens"),
            "reasoning_tokens": call.get("reasoning_tokens"),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_file, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")

    image_rows = []

    for (rep, has_image), group in scored.groupby(["replicate_id", "has_image"]):
        evaluable = group[group["has_official_answer"] == True].copy()
        valid = evaluable[evaluable["scoring_status"].isin(["correct", "incorrect", "blank"])]

        image_rows.append({
            "run_name": run_name,
            "replicate_id": rep,
            "has_image": has_image,
            "n_total_questions_run": len(group),
            "n_evaluable_with_official_answer": len(evaluable),
            "n_valid_scored": len(valid),
            "n_correct": int((valid["scoring_status"] == "correct").sum()),
            "n_incorrect": int((valid["scoring_status"] == "incorrect").sum()),
            "n_blank": int((valid["scoring_status"] == "blank").sum()),
            "n_api_error": int((evaluable["scoring_status"] == "api_error").sum()),
            "n_parse_error": int((evaluable["scoring_status"] == "parse_error").sum()),
            "official_exam_score_sum": valid["exam_score"].sum(skipna=True),
        })

    image_summary = pd.DataFrame(image_rows)
    image_summary.to_csv(image_summary_file, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")

    metadata["completed_utc"] = utc_timestamp()
    metadata["outputs"] = {
        "raw_outputs_file": str(raw_file),
        "scored_results_file": str(scored_file),
        "summary_file": str(summary_file),
        "summary_by_image_status_file": str(image_summary_file),
        "metadata_file": str(metadata_file),
        "prompt_log_file": str(prompt_log_file),
        "image_manifest_file": str(image_manifest_file),
    }
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("DONE")
    print(f"Raw outputs: {raw_file}")
    print(f"Scored results: {scored_file}")
    print(f"Summary: {summary_file}")
    print(f"Image summary: {image_summary_file}")
    print(f"Metadata: {metadata_file}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

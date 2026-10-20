import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import os
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

PROMPT_TEMPLATE_VERSION = "fir2024_text_image_itemwise_examlike_blank_tool_schema_short_rationale_v1"
SCRIPT_VERSION = "fir2024_claude_opus48_itemwise_tool_schema_no_sampling_params_v1"

GENERATION_PARAMETERS = {
    "temperature": "not_sent",
    "top_p": "not_sent",
    "max_tokens": 300,
    "structured_output": "forced_tool_use_input_schema",
    "rationale_max_chars": 160,
    "stream": False,
    "external_tools_enabled": False,
    "web_search_enabled": False,
    "file_input_enabled": False,
    "image_input_enabled": True,
    "image_transport": "base64_via_messages_image_block",
    "reasoning_effort": "not_sent",
}

REQUEST_TIMEOUT_SECONDS = 240
MAX_ATTEMPTS_PER_CALL = 4
RETRY_SLEEP_SECONDS = 20

QUESTIONS_FILE = Path("fir2024_questions_210.csv")
ANSWER_KEY_FILE = Path("fir2024_answer_key_210.csv")
IMAGE_ROOT = Path("images") / "fir"
OUTPUT_ROOT = Path("outputs")

ANSWER_TOOL = {
    "name": "record_answer",
    "description": "Record the answer to a multiple-choice examination question.",
    "input_schema": {
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
}

PROMPT_PREAMBLE = """You are taking an examination for Pharmacy residency access.

This is a multiple-choice exam. Each question has four options, numbered 1 to 4, and only one is correct. In the official exam, correct answers score +1 point, incorrect answers score -1/3 points, and blank answers score 0 points. Act as a human candidate and try to get the highest score possible.

Use the record_answer tool.
The "answer" field must be exactly one of: "1", "2", "3", "4", or "blank".
The "rationale" field must be brief and must not exceed 160 characters.
Do not use web search, tools other than record_answer, external references, files, or an answer key.
If an image is provided with the question, use it as part of the question.
"""


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


def safe_tps(tokens, seconds):
    try:
        if tokens is None or seconds is None or seconds <= 0:
            return None
        return float(tokens) / float(seconds)
    except Exception:
        return None


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


def resolve_image_path(row: pd.Series) -> Optional[Path]:
    has_image = bool_like(row.get("has_image"))
    if not has_image:
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
    candidates = [
        IMAGE_ROOT / f"{qid}.png",
        IMAGE_ROOT / f"{qid}.jpg",
        IMAGE_ROOT / f"{qid}.jpeg",
        IMAGE_ROOT / f"Q{int(row['original_question_number']):03d}.png",
        IMAGE_ROOT / f"Q{int(row['original_question_number']):03d}.jpg",
        IMAGE_ROOT / f"Q{int(row['original_question_number']):03d}.jpeg",
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


def build_content_blocks(row: pd.Series) -> Tuple[list, Optional[Path], Optional[str], Optional[int]]:
    prompt = build_prompt(row)
    blocks = [{"type": "text", "text": prompt}]

    image_path = resolve_image_path(row)
    image_sha256 = None
    image_size_bytes = None

    if image_path is not None:
        if not image_path.exists():
            raise FileNotFoundError(f"Pregunta con imagen pero archivo no encontrado: {row['question_id']} -> {image_path}")
        blocks.append(image_block_from_path(image_path))
        image_sha256 = sha256_file(image_path)
        image_size_bytes = image_path.stat().st_size

    return blocks, image_path, image_sha256, image_size_bytes


def validate_inputs(questions: pd.DataFrame, answer_key: pd.DataFrame):
    if len(questions) != 210:
        raise ValueError(f"Se esperaban 210 preguntas FIR, hay {len(questions)}")
    if len(answer_key) != 210:
        raise ValueError(f"Se esperaban 210 respuestas oficiales FIR, hay {len(answer_key)}")

    required_q_cols = [
        "exam",
        "question_id",
        "original_question_number",
        "is_reserve",
        "has_image",
        "question_text",
        "option_1",
        "option_2",
        "option_3",
        "option_4",
    ]

    for col in required_q_cols:
        if col not in questions.columns:
            raise ValueError(f"Falta columna en questions: {col}")

    if "correct_answer" not in answer_key.columns:
        raise ValueError("Falta columna correct_answer en answer key")


def parse_tool_answer(response):
    tool_input = None

    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "record_answer":
            tool_input = block.input
            break

    if tool_input is None:
        return None, None, "missing_record_answer_tool_use", None

    answer = normalize_answer(tool_input.get("answer"))
    rationale = tool_input.get("rationale")

    if answer not in ["1", "2", "3", "4", "blank"]:
        return answer, rationale, "invalid_answer", tool_input

    if not isinstance(rationale, str):
        return answer, rationale, "invalid_rationale_type", tool_input

    if len(rationale) > 160:
        return answer, rationale, "rationale_too_long", tool_input

    return answer, rationale, None, tool_input


def call_claude(client, content_blocks: list) -> Dict[str, Any]:
    error_type = None
    error_message = None
    error_traceback = None

    response = None
    model_returned = None
    response_id = None
    stop_reason = None
    usage_dict = {}
    response_dump = {}
    parsed_answer = None
    rationale = None
    parse_error = None
    tool_input = None
    attempts_used = 0

    start = time.perf_counter()

    for attempt in range(1, MAX_ATTEMPTS_PER_CALL + 1):
        attempts_used = attempt

        try:
            response = client.messages.create(
                model=MODEL_REQUESTED,
                max_tokens=GENERATION_PARAMETERS["max_tokens"],
                messages=[{"role": "user", "content": content_blocks}],
                tools=[ANSWER_TOOL],
                tool_choice={"type": "tool", "name": "record_answer"},
            )

            model_returned = getattr(response, "model", None)
            response_id = getattr(response, "id", None)
            stop_reason = getattr(response, "stop_reason", None)
            usage_dict = object_to_dict(getattr(response, "usage", None))
            response_dump = object_to_dict(response)

            parsed_answer, rationale, parse_error, tool_input = parse_tool_answer(response)

            error_type = None
            error_message = None
            error_traceback = None
            break

        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            error_traceback = traceback.format_exc()

            print(f"    API ERROR attempt {attempt}/{MAX_ATTEMPTS_PER_CALL}: {error_type} - {error_message[:180]}")

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
        "raw_tool_input": tool_input,
        "raw_response": response_dump,
        "parsed_answer": parsed_answer,
        "rationale": rationale,
        "parse_error": parse_error,
        "usage": usage_dict,
        "error_type": error_type,
        "error_message": error_message,
        "error_traceback": error_traceback,
        "latency_seconds": latency,
        "attempts_used": attempts_used,
        **usage_flat,
        "output_tokens_per_second": safe_tps(usage_flat.get("output_tokens"), latency),
        "total_tokens_per_second": safe_tps(usage_flat.get("total_tokens"), latency),
    }


def get_official_available(row: pd.Series) -> bool:
    if "official_answer_available" in row.index:
        return bool_like(row.get("official_answer_available"))
    if "is_annulled_or_without_official_answer" in row.index:
        if bool_like(row.get("is_annulled_or_without_official_answer")):
            return False
    return normalize_answer(row.get("correct_answer")) is not None


def compute_scoring(row):
    error_type = row.get("error_type")
    parse_error = row.get("parse_error")
    answer = normalize_answer(row.get("parsed_answer"))
    correct = normalize_answer(row.get("correct_answer"))
    official = get_official_available(row)

    if pd.notna(error_type) and str(error_type).strip() not in ["", "None", "nan"]:
        return "api_error", None, False, None

    if pd.notna(parse_error) and str(parse_error).strip() not in ["", "None", "nan"]:
        return "parse_error", None, False, None

    if not official or correct is None:
        return "no_official_answer", None, answer == "blank", None

    if answer == "blank":
        return "blank", False, True, 0.0

    if answer == correct:
        return "correct", True, False, 1.0

    return "incorrect", False, False, -1.0 / 3.0


def write_jsonl(path: Path, record: Dict[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


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

    validate_inputs(questions, answer_key)

    questions = questions.sort_values("original_question_number").reset_index(drop=True)
    answer_key["correct_answer"] = answer_key["correct_answer"].apply(normalize_answer)

    if args.limit is not None:
        questions = questions.head(args.limit).copy()

    missing_images = []
    image_manifest_rows = []

    for _, row in questions.iterrows():
        image_path = resolve_image_path(row)
        has_image = bool_like(row.get("has_image"))
        if has_image and (image_path is None or not image_path.exists()):
            missing_images.append(f"{row['question_id']} -> {image_path}")

        image_manifest_rows.append({
            "question_id": row["question_id"],
            "original_question_number": int(row["original_question_number"]),
            "has_image": has_image,
            "image_path": str(image_path) if image_path else None,
            "image_exists": bool(image_path.exists()) if image_path else False,
            "image_sha256": sha256_file(image_path) if image_path and image_path.exists() else None,
            "image_size_bytes": image_path.stat().st_size if image_path and image_path.exists() else None,
        })

    if missing_images:
        print("Faltan imágenes requeridas. No se llama a la API.")
        for x in missing_images:
            print("  " + x)
        raise FileNotFoundError("Hay preguntas FIR con has_image=True sin archivo de imagen local.")

    timestamp = utc_timestamp()
    safe_model = MODEL_REQUESTED.replace(".", "_").replace("-", "_")
    limit_suffix = f"_limit{args.limit}" if args.limit is not None else ""
    run_name = f"fir2024_{safe_model}_itemwise_TOOL_{args.replicates}runs{limit_suffix}_{timestamp}"
    output_dir = OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_file = output_dir / f"{run_name}_raw_outputs.jsonl"
    results_file = output_dir / f"{run_name}_scored_results.csv"
    summary_file = output_dir / f"{run_name}_summary_by_replicate.csv"
    image_summary_file = output_dir / f"{run_name}_summary_by_image_status.csv"
    image_manifest_file = output_dir / f"{run_name}_image_manifest.csv"
    metadata_file = output_dir / f"{run_name}_run_metadata.json"

    pd.DataFrame(image_manifest_rows).to_csv(image_manifest_file, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")

    q_hash = sha256_file(QUESTIONS_FILE)
    k_hash = sha256_file(ANSWER_KEY_FILE)
    preamble_hash = sha256_text(PROMPT_PREAMBLE)

    metadata = {
        "run_name": run_name,
        "created_utc": timestamp,
        "provider": PROVIDER,
        "interface": INTERFACE,
        "api_endpoint": API_ENDPOINT,
        "model_requested": MODEL_REQUESTED,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "script_version": SCRIPT_VERSION,
        "generation_parameters": GENERATION_PARAMETERS,
        "answer_tool_schema": ANSWER_TOOL,
        "n_replicates": args.replicates,
        "n_questions_attempted": int(len(questions)),
        "questions_file": str(QUESTIONS_FILE),
        "answer_key_file": str(ANSWER_KEY_FILE),
        "questions_file_sha256": q_hash,
        "answer_key_file_sha256": k_hash,
        "image_root": str(IMAGE_ROOT),
        "image_manifest_file": str(image_manifest_file),
        "tools_or_web_search_provided_to_model": False,
        "structured_output_note": (
            "A forced client-side tool schema is used only to obtain structured output. "
            "No external tools, web search, file access, or answer key are provided."
        ),
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

    total_calls = len(questions) * args.replicates
    call_index = 0
    rows = []

    print(f"Run name: {run_name}")
    print(f"Output directory: {output_dir}")
    print(f"Total API calls planned: {total_calls}")
    print(f"Questions included: {len(questions)}")
    print(f"Replicates: {args.replicates}")
    print("Condition: FIR item-wise text + image when available")
    print("Structured output: forced Claude tool_use input_schema")
    print("Temperature sent: False")
    print("Top_p sent: False")
    print("Rationale max chars: 160")
    print()

    for rep in range(1, args.replicates + 1):
        for _, row in questions.iterrows():
            call_index += 1

            content_blocks, image_path, image_sha256, image_size_bytes = build_content_blocks(row)
            prompt = build_prompt(row)
            prompt_hash = sha256_text(prompt)
            has_image = image_path is not None

            print(
                f"[{call_index}/{total_calls}] "
                f"rep={rep} {row['exam']} {row['question_id']} "
                f"(original {row['original_question_number']}) image={has_image}"
            )

            request_time = datetime.now(timezone.utc).isoformat()
            api_result = call_claude(client, content_blocks)
            response_time = datetime.now(timezone.utc).isoformat()

            status_preview = (
                "API_ERROR" if api_result.get("error_type")
                else "PARSE_ERROR" if api_result.get("parse_error")
                else f"answer={api_result.get('parsed_answer')}"
            )

            print(f"    {status_preview} model={api_result.get('model_returned')} attempts={api_result.get('attempts_used')}")

            base = {
                "run_id": f"{timestamp}_rep{rep}_{row['exam']}_{row['question_id']}_{safe_model}",
                "run_name": run_name,
                "replicate_id": rep,
                "provider": PROVIDER,
                "interface": INTERFACE,
                "api_endpoint": API_ENDPOINT,
                "model_requested": MODEL_REQUESTED,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "script_version": SCRIPT_VERSION,
                "generation_parameters": json.dumps(GENERATION_PARAMETERS, ensure_ascii=False),
                "exam": row["exam"],
                "question_id": row["question_id"],
                "original_question_number": int(row["original_question_number"]),
                "is_reserve": bool_like(row["is_reserve"]),
                "subject": row.get("subject"),
                "has_image": has_image,
                "image_path": str(image_path) if image_path else None,
                "image_sha256": image_sha256,
                "image_size_bytes": image_size_bytes,
                "source_pdf_page": safe_int(row.get("source_pdf_page")),
                "question_text": row["question_text"],
                "option_1": row["option_1"],
                "option_2": row["option_2"],
                "option_3": row["option_3"],
                "option_4": row["option_4"],
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
                "prompt_preamble_sha256": preamble_hash,
                "questions_file_sha256": q_hash,
                "answer_key_file_sha256": k_hash,
                "request_time_utc": request_time,
                "response_time_utc": response_time,
                "temperature": GENERATION_PARAMETERS["temperature"],
                "top_p": GENERATION_PARAMETERS["top_p"],
                "max_tokens": GENERATION_PARAMETERS["max_tokens"],
                "reasoning_effort": GENERATION_PARAMETERS["reasoning_effort"],
                "stream": GENERATION_PARAMETERS["stream"],
                "external_tools_enabled": GENERATION_PARAMETERS["external_tools_enabled"],
                "web_search_enabled": GENERATION_PARAMETERS["web_search_enabled"],
            }

            raw_record = {**base, **api_result}
            write_jsonl(raw_file, raw_record)

            rows.append({
                **base,
                "model_returned": api_result.get("model_returned"),
                "response_id": api_result.get("response_id"),
                "stop_reason": api_result.get("stop_reason"),
                "raw_tool_input": json.dumps(api_result.get("raw_tool_input"), ensure_ascii=False) if api_result.get("raw_tool_input") is not None else None,
                "parsed_answer": api_result.get("parsed_answer"),
                "rationale": api_result.get("rationale"),
                "parse_error": api_result.get("parse_error"),
                "error_type": api_result.get("error_type"),
                "error_message": api_result.get("error_message"),
                "latency_seconds": round(api_result.get("latency_seconds"), 6) if api_result.get("latency_seconds") is not None else None,
                "attempts_used": api_result.get("attempts_used"),
                "input_tokens": api_result.get("input_tokens"),
                "output_tokens": api_result.get("output_tokens"),
                "total_tokens": api_result.get("total_tokens"),
                "reasoning_tokens": api_result.get("reasoning_tokens"),
                "cache_creation_input_tokens": api_result.get("cache_creation_input_tokens"),
                "cache_read_input_tokens": api_result.get("cache_read_input_tokens"),
                "output_tokens_per_second": round(api_result.get("output_tokens_per_second"), 6) if api_result.get("output_tokens_per_second") is not None else None,
                "total_tokens_per_second": round(api_result.get("total_tokens_per_second"), 6) if api_result.get("total_tokens_per_second") is not None else None,
            })

    results = pd.DataFrame(rows)

    merge_keys = ["exam", "question_id"]
    for extra in ["original_question_number", "is_reserve"]:
        if extra in results.columns and extra in answer_key.columns:
            merge_keys.append(extra)

    scored = results.merge(answer_key, on=merge_keys, how="left")

    scored["parsed_answer"] = scored["parsed_answer"].apply(normalize_answer)
    scored["correct_answer"] = scored["correct_answer"].apply(normalize_answer)

    scoring = scored.apply(compute_scoring, axis=1, result_type="expand")
    scoring.columns = ["scoring_status", "is_correct", "is_blank", "exam_score"]
    scored = pd.concat([scored, scoring], axis=1)

    scored.to_csv(results_file, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")

    summary_rows = []

    for rep, group in scored.groupby("replicate_id"):
        evaluable = group[group["scoring_status"] != "no_official_answer"].copy()
        valid = evaluable[evaluable["scoring_status"].isin(["correct", "incorrect", "blank"])]

        n_eval = len(evaluable)
        n_correct = int((valid["scoring_status"] == "correct").sum())
        n_incorrect = int((valid["scoring_status"] == "incorrect").sum())
        n_blank = int((valid["scoring_status"] == "blank").sum())
        n_api_error = int((group["scoring_status"] == "api_error").sum())
        n_parse_error = int((group["scoring_status"] == "parse_error").sum())
        n_no_official = int((group["scoring_status"] == "no_official_answer").sum())

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
            "n_no_official_answer": n_no_official,
            "accuracy_among_answered_nonblank": n_correct / (n_correct + n_incorrect) if (n_correct + n_incorrect) else None,
            "accuracy_among_evaluable_questions_counting_blank_as_not_correct": n_correct / n_eval if n_eval else None,
            "official_exam_score_sum": valid["exam_score"].sum(skipna=True),
            "mean_latency_seconds": valid["latency_seconds"].mean(),
            "median_latency_seconds": valid["latency_seconds"].median(),
            "mean_output_tokens_per_second": valid["output_tokens_per_second"].mean(),
            "mean_total_tokens_per_second": valid["total_tokens_per_second"].mean(),
            "total_input_tokens": group["input_tokens"].sum(skipna=True),
            "total_output_tokens": group["output_tokens"].sum(skipna=True),
            "total_tokens": group["total_tokens"].sum(skipna=True),
            "total_reasoning_tokens": group["reasoning_tokens"].sum(skipna=True),
            "total_cache_creation_input_tokens": group["cache_creation_input_tokens"].sum(skipna=True),
            "total_cache_read_input_tokens": group["cache_read_input_tokens"].sum(skipna=True),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_file, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")

    image_rows = []
    for (rep, has_image), group in scored.groupby(["replicate_id", "has_image"]):
        evaluable = group[group["scoring_status"] != "no_official_answer"].copy()
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
            "n_api_error": int((group["scoring_status"] == "api_error").sum()),
            "n_parse_error": int((group["scoring_status"] == "parse_error").sum()),
            "official_exam_score_sum": valid["exam_score"].sum(skipna=True),
        })

    image_summary = pd.DataFrame(image_rows)
    image_summary.to_csv(image_summary_file, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")

    metadata["completed_utc"] = utc_timestamp()
    metadata["outputs"] = {
        "raw_outputs_file": str(raw_file),
        "scored_results_file": str(results_file),
        "summary_file": str(summary_file),
        "summary_by_image_status_file": str(image_summary_file),
        "image_manifest_file": str(image_manifest_file),
        "metadata_file": str(metadata_file),
    }
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("DONE")
    print(f"Raw outputs: {raw_file}")
    print(f"Scored results: {results_file}")
    print(f"Summary: {summary_file}")
    print(f"Image summary: {image_summary_file}")
    print(f"Image manifest: {image_manifest_file}")
    print(f"Metadata: {metadata_file}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

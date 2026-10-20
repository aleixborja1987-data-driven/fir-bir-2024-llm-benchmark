import os
import csv
import json
import time
import base64
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# STUDY CONFIGURATION
# =========================

PROVIDER = "OpenAI"
INTERFACE = "API"
API_ENDPOINT = "Responses API"

MODEL_REQUESTED = "gpt-5.2-2025-12-11"

PROMPT_TEMPLATE_VERSION = "fir2024_itemwise_textplusimage_examlike_blank_v1"

GENERATION_PARAMETERS = {
    "temperature": 0,
    "top_p": 1,
    "max_output_tokens": 300,
    "reasoning_effort": "not_sent",
    "stream": False,
    "store": False,
    "tools_enabled": False,
    "web_search_enabled": False,
    "image_input_enabled": True,
    "image_transport": "local_png_base64_data_url",
}

N_REPLICATES = 2

QUESTIONS_FILE = Path("fir2024_questions_210.csv")
ANSWER_KEY_FILE = Path("fir2024_answer_key_210.csv")

RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
SAFE_MODEL_NAME = MODEL_REQUESTED.replace(".", "_").replace("-", "_")

OUTPUT_DIR = Path("outputs") / f"fir2024_{SAFE_MODEL_NAME}_itemwise_textplusimage_{N_REPLICATES}runs_{RUN_TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUTPUTS_FILE = OUTPUT_DIR / f"fir2024_{SAFE_MODEL_NAME}_itemwise_textplusimage_{N_REPLICATES}runs_{RUN_TIMESTAMP}_raw_outputs.jsonl"
RESULTS_FILE = OUTPUT_DIR / f"fir2024_{SAFE_MODEL_NAME}_itemwise_textplusimage_{N_REPLICATES}runs_{RUN_TIMESTAMP}_scored_results.csv"
SUMMARY_FILE = OUTPUT_DIR / f"fir2024_{SAFE_MODEL_NAME}_itemwise_textplusimage_{N_REPLICATES}runs_{RUN_TIMESTAMP}_summary_by_replicate.csv"
METADATA_FILE = OUTPUT_DIR / f"fir2024_{SAFE_MODEL_NAME}_itemwise_textplusimage_{N_REPLICATES}runs_{RUN_TIMESTAMP}_run_metadata.json"
IMAGE_MANIFEST_FILE = OUTPUT_DIR / f"fir2024_{SAFE_MODEL_NAME}_itemwise_textplusimage_{N_REPLICATES}runs_{RUN_TIMESTAMP}_image_manifest.csv"


# =========================
# HELPER FUNCTIONS
# =========================

def as_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y", "sí", "si"}


def clean_optional_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    value = str(value).strip()
    if value in ["", "nan", "None", "NA", "<NA>"]:
        return None
    return value


def normalize_answer(answer):
    answer = clean_optional_value(answer)
    if answer is None:
        return None
    answer = str(answer).strip().lower()
    if answer in ["1", "2", "3", "4", "blank"]:
        return answer
    # Normalize CSV float artifacts if they appear
    if answer in ["1.0", "2.0", "3.0", "4.0"]:
        return answer[0]
    return answer


def normalize_correct_answer(answer):
    answer = clean_optional_value(answer)
    if answer is None:
        return None
    answer = str(answer).strip()
    if answer in ["1.0", "2.0", "3.0", "4.0"]:
        return answer[0]
    return answer


def nonempty_string(value) -> bool:
    value = clean_optional_value(value)
    return value is not None


def build_prompt(row: pd.Series) -> str:
    """
    Builds the exact text prompt shown to the model.
    IMPORTANT: the correct answer is never included here.
    If the question has an image, the image is attached separately in the API call.
    """
    image_note = ""
    if as_bool(row.get("has_image", False)):
        image_note = "\nThe original question includes an attached image. Use it together with the text below.\n"

    return f"""You are taking an examination for Pharmacy residency access.

This is a multiple-choice exam. Each question has four options, numbered 1 to 4, and only one is correct. In the official exam, correct answers score +1 point, incorrect answers score -1/3 points, and blank answers score 0 points. Act as a human candidate and try to get the highest score possible.{image_note}

Question:
{row["question_text"]}

Options:
1. {row["option_1"]}
2. {row["option_2"]}
3. {row["option_3"]}
4. {row["option_4"]}

Return only valid JSON with:
{{"answer": "1|2|3|4|blank", "rationale": "brief explanation"}}
"""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_png_dimensions(path: Path):
    """
    Minimal PNG dimension reader using only standard library.
    Returns (width, height) or (None, None).
    """
    try:
        with path.open("rb") as f:
            signature = f.read(8)
            if signature != b"\x89PNG\r\n\x1a\n":
                return None, None
            length = int.from_bytes(f.read(4), "big")
            chunk_type = f.read(4)
            if chunk_type != b"IHDR":
                return None, None
            data = f.read(length)
            width = int.from_bytes(data[0:4], "big")
            height = int.from_bytes(data[4:8], "big")
            return width, height
    except Exception:
        return None, None


def encode_image_as_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        raise ValueError(f"Unsupported image extension for {path}. Use PNG/JPG/WEBP.")
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def validate_inputs(questions: pd.DataFrame, answer_key: pd.DataFrame):
    required_question_cols = {
        "exam",
        "question_id",
        "original_question_number",
        "is_reserve",
        "subject",
        "has_image",
        "image_path",
        "image_detail",
        "question_text",
        "option_1",
        "option_2",
        "option_3",
        "option_4",
    }

    required_key_cols = {"exam", "question_id", "correct_answer"}

    missing_q = required_question_cols - set(questions.columns)
    missing_k = required_key_cols - set(answer_key.columns)

    if missing_q:
        raise ValueError(f"Faltan columnas en {QUESTIONS_FILE}: {missing_q}")

    if missing_k:
        raise ValueError(f"Faltan columnas en {ANSWER_KEY_FILE}: {missing_k}")

    if len(questions) != 210:
        raise ValueError(f"Se esperaban 210 preguntas, pero hay {len(questions)}")

    # Fail before any API calls if any image is missing
    image_manifest = []
    errors = []

    for _, row in questions.iterrows():
        if not as_bool(row["has_image"]):
            continue

        qid = str(row["question_id"])
        image_path_str = clean_optional_value(row.get("image_path"))
        image_detail = clean_optional_value(row.get("image_detail")) or "high"

        if not image_path_str:
            errors.append(f"{qid}: has_image=True pero image_path vacío")
            continue

        image_path = Path(image_path_str)
        if not image_path.exists():
            errors.append(f"{qid}: no existe {image_path}")
            continue

        size_bytes = image_path.stat().st_size
        if size_bytes <= 0:
            errors.append(f"{qid}: archivo vacío {image_path}")
            continue

        if image_detail not in {"low", "high", "auto", "original"}:
            errors.append(f"{qid}: image_detail inválido: {image_detail}")

        width, height = get_png_dimensions(image_path)
        image_manifest.append({
            "question_id": qid,
            "image_path": str(image_path),
            "image_detail": image_detail,
            "size_bytes": size_bytes,
            "width_px": width,
            "height_px": height,
            "sha256": sha256_file(image_path),
        })

    if errors:
        msg = "\n".join(f"- {e}" for e in errors)
        raise RuntimeError(f"Errores de validación de imágenes:\n{msg}")

    pd.DataFrame(image_manifest).to_csv(
        IMAGE_MANIFEST_FILE,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding="utf-8-sig"
    )


def parse_model_answer(raw_text: str):
    """
    Parses final answer from JSON.
    Returns parsed_answer, rationale, parse_error.
    """
    if raw_text is None or str(raw_text).strip() == "":
        return None, None, "empty_output"

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return None, None, f"json_decode_error: {str(e)}"

    answer = normalize_answer(data.get("answer"))
    rationale = data.get("rationale")

    if answer is None:
        return None, rationale, "missing_answer"

    if answer not in ["1", "2", "3", "4", "blank"]:
        return answer, rationale, "invalid_answer"

    return answer, rationale, None


def model_dump_safe(obj):
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    try:
        return dict(obj)
    except Exception:
        return repr(obj)


def get_usage_dict(response) -> dict:
    usage = getattr(response, "usage", None)
    dumped = model_dump_safe(usage)
    return dumped if isinstance(dumped, dict) else {}


def flatten_usage(usage: dict) -> dict:
    if not isinstance(usage, dict):
        usage = {}

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")

    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}

    cached_tokens = input_details.get("cached_tokens")
    reasoning_tokens = output_details.get("reasoning_tokens")

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_input_tokens": cached_tokens,
    }


def safe_divide(numerator, denominator):
    try:
        if numerator is None or denominator is None:
            return None
        if float(denominator) == 0:
            return None
        return float(numerator) / float(denominator)
    except Exception:
        return None


def score_answer(parsed_answer, correct_answer, has_official_answer):
    """
    Returns scoring_status, is_correct, is_blank, exam_score.
    """
    parsed_answer = normalize_answer(parsed_answer)
    correct_answer = normalize_correct_answer(correct_answer)

    if not has_official_answer:
        return "no_official_answer", None, parsed_answer == "blank", None

    if parsed_answer is None:
        return "missing_model_answer", None, False, None

    if parsed_answer == "blank":
        return "blank", False, True, 0.0

    if str(parsed_answer).strip() == str(correct_answer).strip():
        return "correct", True, False, 1.0

    return "incorrect", False, False, -1.0 / 3.0


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for rep, group in scored.groupby("replicate_id"):
        evaluable = group[group["has_official_answer"] == True].copy()

        n_total = len(group)
        n_evaluable = len(evaluable)

        n_correct = int((evaluable["scoring_status"] == "correct").sum())
        n_incorrect = int((evaluable["scoring_status"] == "incorrect").sum())
        n_blank = int((evaluable["scoring_status"] == "blank").sum())
        n_api_error = int(group["error_type"].apply(nonempty_string).sum())
        n_parse_error = int(group["parse_error"].apply(nonempty_string).sum())
        n_no_official = int((group["has_official_answer"] == False).sum())

        score = evaluable["exam_score"].dropna().sum()
        accuracy = n_correct / n_evaluable if n_evaluable else None

        rows.append({
            "replicate_id": rep,
            "n_total_questions_run": n_total,
            "n_evaluable_with_official_answer": n_evaluable,
            "n_no_official_answer": n_no_official,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_blank": n_blank,
            "n_api_error": n_api_error,
            "n_parse_error": n_parse_error,
            "accuracy_evaluable": accuracy,
            "exam_score_evaluable": score,
            "mean_latency_seconds": group["latency_seconds"].mean(),
            "median_latency_seconds": group["latency_seconds"].median(),
            "mean_output_tokens_per_second": group["output_tokens_per_second"].mean(),
            "mean_total_tokens_per_second": group["total_tokens_per_second"].mean(),
            "total_input_tokens": group["input_tokens"].fillna(0).sum(),
            "total_output_tokens": group["output_tokens"].fillna(0).sum(),
            "total_tokens": group["total_tokens"].fillna(0).sum(),
        })

    return pd.DataFrame(rows)


# =========================
# MAIN EXECUTION
# =========================

def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No se ha encontrado OPENAI_API_KEY. Revisa el archivo .env.")

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"No existe {QUESTIONS_FILE}")

    if not ANSWER_KEY_FILE.exists():
        raise FileNotFoundError(f"No existe {ANSWER_KEY_FILE}")

    client = OpenAI(api_key=api_key)

    questions = pd.read_csv(QUESTIONS_FILE)
    answer_key = pd.read_csv(ANSWER_KEY_FILE)

    validate_inputs(questions, answer_key)

    metadata = {
        "provider": PROVIDER,
        "interface": INTERFACE,
        "api_endpoint": API_ENDPOINT,
        "model_requested": MODEL_REQUESTED,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "generation_parameters": GENERATION_PARAMETERS,
        "n_replicates": N_REPLICATES,
        "questions_file": str(QUESTIONS_FILE),
        "answer_key_file": str(ANSWER_KEY_FILE),
        "run_timestamp_utc": RUN_TIMESTAMP,
        "important_note": (
            "Correct answers are never included in model inputs. The answer key is merged locally "
            "after generation for offline scoring. Image-containing items are sent as text plus a "
            "local base64-encoded PNG data URL."
        ),
    }

    METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    RAW_OUTPUTS_FILE.write_text("", encoding="utf-8")

    results = []
    total_calls = len(questions) * N_REPLICATES
    call_counter = 0

    for replicate_id in range(1, N_REPLICATES + 1):
        for _, row in questions.iterrows():
            call_counter += 1

            prompt = build_prompt(row)
            has_image = as_bool(row.get("has_image", False))
            image_path = Path(str(row.get("image_path")).strip()) if has_image else None
            image_detail = str(row.get("image_detail", "high")).strip() if has_image else None
            image_sha256 = sha256_file(image_path) if has_image else None
            image_size_bytes = image_path.stat().st_size if has_image else None
            image_width_px, image_height_px = get_png_dimensions(image_path) if has_image else (None, None)

            run_id = (
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
                f"rep{replicate_id}_{row['exam']}_{row['question_id']}_{SAFE_MODEL_NAME}"
            )

            print(
                f"[{call_counter}/{total_calls}] "
                f"rep={replicate_id} {row['exam']} {row['question_id']} "
                f"(original {row['original_question_number']}) "
                f"{'[image]' if has_image else '[text]'}"
            )

            request_time_utc = datetime.now(timezone.utc).isoformat()
            start = time.perf_counter()

            error_type = None
            error_message = None
            raw_text = None
            model_returned = None
            response_status = None
            incomplete_details = None
            response_id = None
            usage_dict = {}
            response_dump = None

            try:
                content = [{"type": "input_text", "text": prompt}]

                if has_image:
                    content.append({
                        "type": "input_image",
                        "image_url": encode_image_as_data_url(image_path),
                        "detail": image_detail,
                    })

                response = client.responses.create(
                    model=MODEL_REQUESTED,
                    input=[{
                        "role": "user",
                        "content": content,
                    }],
                    temperature=GENERATION_PARAMETERS["temperature"],
                    top_p=GENERATION_PARAMETERS["top_p"],
                    max_output_tokens=GENERATION_PARAMETERS["max_output_tokens"],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "mcq_answer_with_blank",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                        "enum": ["1", "2", "3", "4", "blank"]
                                    },
                                    "rationale": {
                                        "type": "string"
                                    }
                                },
                                "required": ["answer", "rationale"],
                                "additionalProperties": False
                            }
                        }
                    },
                    store=GENERATION_PARAMETERS["store"],
                )

                response_id = getattr(response, "id", None)
                model_returned = getattr(response, "model", None)
                response_status = getattr(response, "status", None)
                incomplete_details = model_dump_safe(getattr(response, "incomplete_details", None))
                usage_dict = get_usage_dict(response)
                raw_text = getattr(response, "output_text", None)
                response_dump = model_dump_safe(response)

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)

            latency_seconds = time.perf_counter() - start
            response_time_utc = datetime.now(timezone.utc).isoformat()

            parsed_answer, rationale, parse_error = (
                parse_model_answer(raw_text) if raw_text else (None, None, "empty_output" if not error_type else None)
            )

            usage_flat = flatten_usage(usage_dict)

            output_tps = safe_divide(usage_flat.get("output_tokens"), latency_seconds)
            total_tps = safe_divide(usage_flat.get("total_tokens"), latency_seconds)

            prompt_sha = sha256_text(prompt)

            raw_record = {
                "run_id": run_id,
                "replicate_id": replicate_id,
                "provider": PROVIDER,
                "interface": INTERFACE,
                "api_endpoint": API_ENDPOINT,
                "model_requested": MODEL_REQUESTED,
                "model_returned": model_returned,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "generation_parameters": GENERATION_PARAMETERS,
                "exam": row["exam"],
                "question_id": row["question_id"],
                "original_question_number": row["original_question_number"],
                "is_reserve": row["is_reserve"],
                "subject": row["subject"],
                "has_image": has_image,
                "image_path": str(image_path) if has_image else None,
                "image_detail": image_detail,
                "image_sha256": image_sha256,
                "image_size_bytes": image_size_bytes,
                "image_width_px": image_width_px,
                "image_height_px": image_height_px,
                "prompt": prompt,
                "prompt_sha256": prompt_sha,
                "request_time_utc": request_time_utc,
                "response_time_utc": response_time_utc,
                "latency_seconds": latency_seconds,
                "response_id": response_id,
                "response_status": response_status,
                "incomplete_details": incomplete_details,
                "raw_output": raw_text,
                "parsed_answer": parsed_answer,
                "rationale": rationale,
                "parse_error": parse_error,
                "usage": usage_dict,
                "output_tokens_per_second": output_tps,
                "total_tokens_per_second": total_tps,
                "error_type": error_type,
                "error_message": error_message,
                "response_dump": response_dump,
            }

            with RAW_OUTPUTS_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")

            results.append({
                "run_id": run_id,
                "replicate_id": replicate_id,
                "provider": PROVIDER,
                "interface": INTERFACE,
                "api_endpoint": API_ENDPOINT,
                "model_requested": MODEL_REQUESTED,
                "model_returned": model_returned,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "exam": row["exam"],
                "question_id": row["question_id"],
                "original_question_number": row["original_question_number"],
                "is_reserve": row["is_reserve"],
                "subject": row["subject"],
                "has_image": has_image,
                "image_path": str(image_path) if has_image else None,
                "image_detail": image_detail,
                "image_sha256": image_sha256,
                "image_size_bytes": image_size_bytes,
                "image_width_px": image_width_px,
                "image_height_px": image_height_px,
                "prompt_sha256": prompt_sha,
                "request_time_utc": request_time_utc,
                "response_time_utc": response_time_utc,
                "latency_seconds": round(latency_seconds, 4),
                "response_id": response_id,
                "response_status": response_status,
                "incomplete_details": json.dumps(incomplete_details, ensure_ascii=False) if incomplete_details else None,
                "raw_output": raw_text,
                "parsed_answer": parsed_answer,
                "rationale": rationale,
                "parse_error": parse_error,
                "error_type": error_type,
                "error_message": error_message,
                **usage_flat,
                "output_tokens_per_second": output_tps,
                "total_tokens_per_second": total_tps,
                "temperature": GENERATION_PARAMETERS["temperature"],
                "top_p": GENERATION_PARAMETERS["top_p"],
                "max_output_tokens": GENERATION_PARAMETERS["max_output_tokens"],
                "reasoning_effort": GENERATION_PARAMETERS["reasoning_effort"],
                "stream": GENERATION_PARAMETERS["stream"],
                "store": GENERATION_PARAMETERS["store"],
                "tools_enabled": GENERATION_PARAMETERS["tools_enabled"],
                "web_search_enabled": GENERATION_PARAMETERS["web_search_enabled"],
            })

    results_df = pd.DataFrame(results)

    answer_key["correct_answer"] = answer_key["correct_answer"].apply(normalize_correct_answer)

    scored = results_df.merge(
        answer_key,
        on=["exam", "question_id"],
        how="left"
    )

    scored["parsed_answer"] = scored["parsed_answer"].apply(normalize_answer)
    scored["correct_answer"] = scored["correct_answer"].apply(normalize_correct_answer)

    scored["has_official_answer"] = scored["correct_answer"].apply(nonempty_string)

    scoring = scored.apply(
        lambda r: score_answer(
            parsed_answer=r["parsed_answer"],
            correct_answer=r["correct_answer"],
            has_official_answer=r["has_official_answer"],
        ),
        axis=1,
        result_type="expand",
    )

    scoring.columns = ["scoring_status", "is_correct", "is_blank", "exam_score"]

    scored = pd.concat([scored, scoring], axis=1)

    scored.loc[scored["error_type"].apply(nonempty_string), "scoring_status"] = "api_error"
    scored.loc[scored["parse_error"].apply(nonempty_string), "scoring_status"] = "parse_error"

    scored.to_csv(
        RESULTS_FILE,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding="utf-8-sig"
    )

    summary = summarize(scored)
    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding="utf-8-sig"
    )

    print("\nDone.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Raw outputs saved to: {RAW_OUTPUTS_FILE}")
    print(f"Scored results saved to: {RESULTS_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")
    print(f"Image manifest saved to: {IMAGE_MANIFEST_FILE}")

    print("\nSummary by replicate:")
    print(summary)


if __name__ == "__main__":
    main()

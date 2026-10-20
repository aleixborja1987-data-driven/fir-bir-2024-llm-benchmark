import os
import csv
import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Optional but useful on some Windows/corporate setups.
# If truststore is not installed, the script will continue without it.
try:
    import truststore
    truststore.inject_into_ssl()
    TRUSTSTORE_ENABLED = True
except Exception:
    TRUSTSTORE_ENABLED = False


# =========================
# STUDY CONFIGURATION
# =========================

PROVIDER = "DeepSeek"
INTERFACE = "API"
API_ENDPOINT = "Chat Completions API - OpenAI-compatible"
BASE_URL = "https://api.deepseek.com"

MODEL_REQUESTED = "deepseek-v4-pro"

PROMPT_TEMPLATE_VERSION = "fir2024_itemwise_text_only_examlike_blank_v1"

# Primary controlled setting:
# DeepSeek V4 thinking mode defaults to enabled, but in thinking mode temperature/top_p
# do not take effect. Therefore, for a controlled sampling protocol, we explicitly
# disable thinking mode.
GENERATION_PARAMETERS = {
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 300,
    "stream": False,
    "response_format": {"type": "json_object"},
    "thinking": "disabled",
    "reasoning_effort": "not_sent",
    "store": "not_applicable",
    "tools_enabled": False,
    "web_search_enabled": False,
    "image_input_enabled": False,
    "image_handling": "image_blind_text_only_for_has_image_items",
}

N_REPLICATES = 2

QUESTIONS_FILE = Path("fir2024_questions_210.csv")
ANSWER_KEY_FILE = Path("fir2024_answer_key_210.csv")

RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
SAFE_MODEL_NAME = MODEL_REQUESTED.replace(".", "_").replace("-", "_")

OUTPUT_DIR = Path("outputs") / f"fir2024_{SAFE_MODEL_NAME}_textonly_itemwise_{N_REPLICATES}runs_{RUN_TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUTPUTS_FILE = OUTPUT_DIR / f"fir2024_{SAFE_MODEL_NAME}_textonly_itemwise_{N_REPLICATES}runs_{RUN_TIMESTAMP}_raw_outputs.jsonl"
RESULTS_FILE = OUTPUT_DIR / f"fir2024_{SAFE_MODEL_NAME}_textonly_itemwise_{N_REPLICATES}runs_{RUN_TIMESTAMP}_scored_results.csv"
SUMMARY_FILE = OUTPUT_DIR / f"fir2024_{SAFE_MODEL_NAME}_textonly_itemwise_{N_REPLICATES}runs_{RUN_TIMESTAMP}_summary_by_replicate.csv"
METADATA_FILE = OUTPUT_DIR / f"fir2024_{SAFE_MODEL_NAME}_textonly_itemwise_{N_REPLICATES}runs_{RUN_TIMESTAMP}_run_metadata.json"
SUMMARY_BY_IMAGE_FILE = OUTPUT_DIR / f"fir2024_{SAFE_MODEL_NAME}_textonly_itemwise_{N_REPLICATES}runs_{RUN_TIMESTAMP}_summary_by_image_status.csv"


# =========================
# HELPER FUNCTIONS
# =========================

def build_prompt(row: pd.Series) -> str:
    """
    Builds the exact prompt shown to the model.
    IMPORTANT: the correct answer is never included here.
    IMPORTANT: no image is sent to DeepSeek. For has_image=True questions, this is
    intentionally an image-blind/text-only prompt, because the official DeepSeek
    V4 Pro API did not support image inputs in our probes.
    """
    image_note = ""
    if str(row.get("has_image", "")).strip().lower() in ["true", "1", "yes"]:
        image_note = (
            "\nNote: The original exam item contains an image/figure, but no image is provided "
            "in this API run. Answer using only the available text and options.\n"
        )

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


def normalize_answer(answer):
    if answer is None:
        return None
    answer = str(answer).strip().lower()
    if answer in ["1", "2", "3", "4", "blank"]:
        return answer
    return answer


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

    # DeepSeek/OpenAI-compatible Chat Completions typically use these names.
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    # Keep possible provider-specific cache fields if present.
    prompt_cache_hit_tokens = usage.get("prompt_cache_hit_tokens")
    prompt_cache_miss_tokens = usage.get("prompt_cache_miss_tokens")

    # Some providers may expose extra completion details.
    completion_details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = completion_details.get("reasoning_tokens")

    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_input_tokens": prompt_cache_hit_tokens,
        "cache_miss_input_tokens": prompt_cache_miss_tokens,
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
    if not has_official_answer:
        return "no_official_answer", None, parsed_answer == "blank", None

    if parsed_answer is None:
        return "missing_model_answer", None, False, None

    if parsed_answer == "blank":
        return "blank", False, True, 0.0

    if str(parsed_answer).strip() == str(correct_answer).strip():
        return "correct", True, False, 1.0

    return "incorrect", False, False, -1.0 / 3.0


def nonempty_string(value) -> bool:
    if pd.isna(value):
        return False
    value = str(value).strip()
    return value not in ["", "nan", "None", "NA"]


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


def summarize_by_image_status(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Stratified summary by replicate and has_image flag.
    This is important because has_image=True items are run image-blind in this script.
    """
    rows = []

    df = scored.copy()
    df["has_image_bool"] = df["has_image"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

    for (rep, has_img), group in df.groupby(["replicate_id", "has_image_bool"]):
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
            "has_image": bool(has_img),
            "image_handling": "image_blind_text_only" if has_img else "text_only_original_item",
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

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("No se ha encontrado DEEPSEEK_API_KEY. Revisa el archivo .env.")

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"No existe {QUESTIONS_FILE}")

    if not ANSWER_KEY_FILE.exists():
        raise FileNotFoundError(f"No existe {ANSWER_KEY_FILE}")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    questions = pd.read_csv(QUESTIONS_FILE)
    answer_key = pd.read_csv(ANSWER_KEY_FILE)

    required_question_cols = {
        "exam",
        "question_id",
        "original_question_number",
        "is_reserve",
        "subject",
        "has_image",
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

    # Prepare metadata
    metadata = {
        "provider": PROVIDER,
        "interface": INTERFACE,
        "api_endpoint": API_ENDPOINT,
        "base_url": BASE_URL,
        "model_requested": MODEL_REQUESTED,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "generation_parameters": GENERATION_PARAMETERS,
        "n_replicates": N_REPLICATES,
        "questions_file": str(QUESTIONS_FILE),
        "answer_key_file": str(ANSWER_KEY_FILE),
        "run_timestamp_utc": RUN_TIMESTAMP,
        "truststore_enabled": TRUSTSTORE_ENABLED,
        "important_note": (
            "DeepSeek V4 thinking mode defaults to enabled. This script explicitly disables thinking mode "
            "so that temperature and top_p are effective in the controlled primary benchmark. "
            "FIR items with has_image=True are deliberately run without image input; the has_image flag is preserved "
            "for stratified analysis. This is an image-blind/text-only DeepSeek V4 Pro API run."
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
            system_prompt = (
                "You are a biomedical multiple-choice exam solver. "
                "Return only valid JSON. Do not include markdown."
            )

            run_id = (
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
                f"rep{replicate_id}_{row['exam']}_{row['question_id']}_{SAFE_MODEL_NAME}"
            )

            print(
                f"[{call_counter}/{total_calls}] "
                f"rep={replicate_id} {row['exam']} {row['question_id']} "
                f"(original {row['original_question_number']})"
            )

            request_time_utc = datetime.now(timezone.utc).isoformat()
            start = time.perf_counter()

            error_type = None
            error_message = None
            raw_text = None
            reasoning_content = None
            model_returned = None
            finish_reason = None
            response_id = None
            usage_dict = {}
            response_dump = None

            try:
                response = client.chat.completions.create(
                    model=MODEL_REQUESTED,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=GENERATION_PARAMETERS["temperature"],
                    top_p=GENERATION_PARAMETERS["top_p"],
                    max_tokens=GENERATION_PARAMETERS["max_tokens"],
                    response_format=GENERATION_PARAMETERS["response_format"],
                    stream=GENERATION_PARAMETERS["stream"],
                    extra_body={"thinking": {"type": GENERATION_PARAMETERS["thinking"]}},
                )

                response_id = getattr(response, "id", None)
                model_returned = getattr(response, "model", None)
                usage_dict = get_usage_dict(response)

                try:
                    choice = response.choices[0]
                    message = choice.message
                    raw_text = message.content
                    reasoning_content = getattr(message, "reasoning_content", None)
                    finish_reason = getattr(choice, "finish_reason", None)
                except Exception as e:
                    error_type = "response_extraction_error"
                    error_message = str(e)

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

            raw_record = {
                "run_id": run_id,
                "replicate_id": replicate_id,
                "provider": PROVIDER,
                "interface": INTERFACE,
                "api_endpoint": API_ENDPOINT,
                "base_url": BASE_URL,
                "model_requested": MODEL_REQUESTED,
                "model_returned": model_returned,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "generation_parameters": GENERATION_PARAMETERS,
                "exam": row["exam"],
                "question_id": row["question_id"],
                "original_question_number": row["original_question_number"],
                "is_reserve": row["is_reserve"],
                "subject": row["subject"],
                "has_image": row["has_image"],
                "prompt": prompt,
                "system_prompt": system_prompt,
                "prompt_sha256": sha256_text(system_prompt + "\n" + prompt),
                "request_time_utc": request_time_utc,
                "response_time_utc": response_time_utc,
                "latency_seconds": latency_seconds,
                "response_id": response_id,
                "finish_reason": finish_reason,
                "raw_output": raw_text,
                "reasoning_content": reasoning_content,
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
                "base_url": BASE_URL,
                "model_requested": MODEL_REQUESTED,
                "model_returned": model_returned,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "exam": row["exam"],
                "question_id": row["question_id"],
                "original_question_number": row["original_question_number"],
                "is_reserve": row["is_reserve"],
                "subject": row["subject"],
                "has_image": row["has_image"],
                "prompt_sha256": sha256_text(system_prompt + "\n" + prompt),
                "request_time_utc": request_time_utc,
                "response_time_utc": response_time_utc,
                "latency_seconds": round(latency_seconds, 4),
                "response_id": response_id,
                "finish_reason": finish_reason,
                "raw_output": raw_text,
                "reasoning_content": reasoning_content,
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
                "max_tokens": GENERATION_PARAMETERS["max_tokens"],
                "thinking": GENERATION_PARAMETERS["thinking"],
                "reasoning_effort": GENERATION_PARAMETERS["reasoning_effort"],
                "stream": GENERATION_PARAMETERS["stream"],
                "store": GENERATION_PARAMETERS["store"],
                "tools_enabled": GENERATION_PARAMETERS["tools_enabled"],
                "web_search_enabled": GENERATION_PARAMETERS["web_search_enabled"],
                "image_input_enabled": GENERATION_PARAMETERS["image_input_enabled"],
                "image_handling": GENERATION_PARAMETERS["image_handling"],
            })

    results_df = pd.DataFrame(results)

    scored = results_df.merge(
        answer_key,
        on=["exam", "question_id"],
        how="left"
    )

    scored["parsed_answer"] = scored["parsed_answer"].astype("string").str.strip().str.lower()
    scored["correct_answer"] = scored["correct_answer"].astype("string").str.strip()

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

    # API and parse errors override the scoring status
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

    summary_by_image = summarize_by_image_status(scored)
    summary_by_image.to_csv(
        SUMMARY_BY_IMAGE_FILE,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding="utf-8-sig"
    )

    print("\nDone.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Raw outputs saved to: {RAW_OUTPUTS_FILE}")
    print(f"Scored results saved to: {RESULTS_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print(f"Summary by image status saved to: {SUMMARY_BY_IMAGE_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")

    print("\nSummary by replicate:")
    print(summary)

    print("\nSummary by image status:")
    print(summary_by_image)


if __name__ == "__main__":
    main()

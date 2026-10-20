import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Windows / corporate SSL compatibility, harmless if not needed.
try:
    import truststore
    truststore.inject_into_ssl()
    TRUSTSTORE_ENABLED = True
except Exception:
    TRUSTSTORE_ENABLED = False


MODEL = "deepseek-v4-pro"
N_REPLICATES = 2
TEMPERATURE = 0
TOP_P = 1
MAX_TOKENS = 30000

PROMPT_TEMPLATE_VERSION = "deepseek_v4pro_whole_extracted_text_examlike_json_210answers_v1"


EXAMS = {
    "bir": {
        "exam_label": "BIR2024",
        "exam_name": "Spanish Biology residency access examination",
        "questions_csv": Path("bir2024_questions_210.csv"),
        "answer_key_csv": Path("bir2024_answer_key_210.csv"),
        "output_prefix": "bir2024_deepseek_v4_pro_whole_text_2runs",
    },
    "fir": {
        "exam_label": "FIR2024",
        "exam_name": "Spanish Pharmacy residency access examination",
        "questions_csv": Path("fir2024_questions_210.csv"),
        "answer_key_csv": Path("fir2024_answer_key_210.csv"),
        "output_prefix": "fir2024_deepseek_v4_pro_whole_text_imageblind_2runs",
    },
}


# =========================
# HELPERS
# =========================

def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_answer(x: Any) -> Optional[str]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip()
    if not s:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    s_lower = s.lower()
    if s_lower in {"nan", "none", "null", "na", "n/a", "annulled", "anulada", "anulado", "-", ""}:
        return None
    if s_lower in {"blank", "blanco", "en blanco"}:
        return "blank"
    if s in {"1", "2", "3", "4"}:
        return s
    return None


def normalize_question_number(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip().upper().replace("Q", "")
    if s.endswith(".0"):
        s = s[:-2]
    try:
        return int(s)
    except Exception:
        return None


def bool_like(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes", "y", "si", "sí"}


def detect_question_col(df: pd.DataFrame) -> str:
    candidates = [
        "original_question_number", "question_number", "question", "q_number",
        "qnum", "number", "question_id", "id", "pregunta", "n_pregunta"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return df.columns[0]


def detect_answer_col(df: pd.DataFrame) -> str:
    candidates = [
        "correct_answer", "official_answer", "answer", "answer_key",
        "rc", "respuesta_correcta", "correcta"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    raise ValueError(f"Could not detect answer column in {list(df.columns)}")


def load_questions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing questions CSV: {path}")

    df = pd.read_csv(path)
    q_col = detect_question_col(df)

    required = ["question_text", "option_1", "option_2", "option_3", "option_4"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in questions CSV: {missing}")

    out = df.copy()
    out["question_number"] = out[q_col].apply(normalize_question_number)
    out = out.dropna(subset=["question_number"]).copy()
    out["question_number"] = out["question_number"].astype(int)

    if "question_id" not in out.columns:
        out["question_id"] = out["question_number"].apply(lambda x: f"Q{x:03d}")

    if "is_reserve" not in out.columns:
        out["is_reserve"] = out["question_number"] > 200

    if "has_image" not in out.columns:
        out["has_image"] = False
    else:
        out["has_image"] = out["has_image"].apply(bool_like)

    out = out.sort_values("question_number")
    if len(out) != 210:
        print(f"WARNING: expected 210 questions, found {len(out)}")
    return out


def load_answer_key(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing answer key CSV: {path}")

    df = pd.read_csv(path)
    q_col = detect_question_col(df)
    a_col = detect_answer_col(df)

    out = pd.DataFrame({
        "question_number": df[q_col].apply(normalize_question_number),
        "correct_answer_raw": df[a_col],
        "correct_answer": df[a_col].apply(normalize_answer),
    })

    # Respect official availability flags when present.
    availability = pd.Series([True] * len(df))
    if "official_answer_available" in df.columns:
        availability = df["official_answer_available"].apply(bool_like)
    if "has_official_answer" in df.columns:
        availability = df["has_official_answer"].apply(bool_like)
    if "is_annulled_or_without_official_answer" in df.columns:
        availability = availability & (~df["is_annulled_or_without_official_answer"].apply(bool_like))

    out["has_official_answer"] = availability.values
    out.loc[~out["has_official_answer"], "correct_answer"] = None

    out = out.dropna(subset=["question_number"]).copy()
    out["question_number"] = out["question_number"].astype(int)
    out = out.drop_duplicates(subset=["question_number"], keep="first")
    return out[["question_number", "correct_answer", "has_official_answer"]]


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def build_exam_text(questions: pd.DataFrame) -> str:
    blocks = []
    for _, row in questions.iterrows():
        qn = int(row["question_number"])
        prefix = f"Question {qn}"
        if bool(row.get("has_image", False)):
            prefix += " [IMAGE-CONTAINING ITEM; IMAGE NOT PROVIDED IN THIS TEXT-ONLY RUN]"
        if bool(row.get("is_reserve", False)):
            prefix += " [RESERVE QUESTION]"

        block = [
            prefix,
            f"Stem: {clean_text(row['question_text'])}",
            f"1. {clean_text(row['option_1'])}",
            f"2. {clean_text(row['option_2'])}",
            f"3. {clean_text(row['option_3'])}",
            f"4. {clean_text(row['option_4'])}",
        ]
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def build_prompt(exam_name: str, exam_label: str, questions: pd.DataFrame) -> str:
    n_images = int(questions["has_image"].sum()) if "has_image" in questions.columns else 0
    exam_text = build_exam_text(questions)

    image_note = ""
    if n_images:
        image_note = (
            f"\n- This extracted-text version contains {n_images} image-containing items. "
            "Their image content is not provided in this DeepSeek API run. "
            "For those items, answer only if the textual stem/options are sufficient; otherwise use \"blank\"."
        )

    return f"""You are taking the {exam_name}.

Below is the complete extracted text of the official examination booklet, version 0. It contains 200 main questions plus 10 reserve questions, for a total of 210 multiple-choice questions.

Instructions:
- Answer all 210 questions based only on the extracted exam text below.
- Do not use web search, tools, external references, or an answer key.
- Each question has four options numbered 1 to 4, and only one option is correct.
- In the official exam, correct answers score +1 point, incorrect answers score -1/3 points, and blank answers score 0 points.
- Act as a human candidate and try to maximize the final score.
- If you are not sufficiently confident, use "blank".
- Use question_number 1 to 210, corresponding to the order and numbering in the extracted exam text.{image_note}

Return only valid JSON.
Return exactly this JSON object shape:
{{
  "answers": [
    {{"question_number": 1, "answer": "1|2|3|4|blank", "rationale": "brief explanation"}},
    ...
    {{"question_number": 210, "answer": "1|2|3|4|blank", "rationale": "brief explanation"}}
  ],
  "notes": "brief notes, if any"
}}

Return exactly 210 answer objects, one for each question_number from 1 to 210.
Keep each rationale brief, ideally under 20 words.

EXAM_LABEL: {exam_label}

EXTRACTED_EXAM_TEXT:
{exam_text}
"""


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def score_answer(model_answer: Optional[str], correct_answer: Optional[str], has_official_answer: bool) -> Dict[str, Any]:
    if not has_official_answer or correct_answer is None:
        return {
            "has_official_answer": False,
            "is_correct": None,
            "score": None,
            "answer_status": "not_evaluable",
        }

    ans = normalize_answer(model_answer)
    if ans is None:
        return {
            "has_official_answer": True,
            "is_correct": False,
            "score": 0.0,
            "answer_status": "missing_or_invalid",
        }

    if ans == "blank":
        return {
            "has_official_answer": True,
            "is_correct": False,
            "score": 0.0,
            "answer_status": "blank",
        }

    if ans == correct_answer:
        return {
            "has_official_answer": True,
            "is_correct": True,
            "score": 1.0,
            "answer_status": "correct",
        }

    return {
        "has_official_answer": True,
        "is_correct": False,
        "score": -1.0 / 3.0,
        "answer_status": "incorrect",
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rep, g in df.groupby("replicate"):
        eval_g = g[g["has_official_answer"] == True]
        n_eval = len(eval_g)
        correct = int((eval_g["is_correct"] == True).sum())
        rows.append({
            "replicate": rep,
            "n_questions_total": int(len(g)),
            "n_evaluable": int(n_eval),
            "correct": correct,
            "incorrect": int((eval_g["answer_status"] == "incorrect").sum()),
            "blank": int((eval_g["answer_status"] == "blank").sum()),
            "missing_or_invalid": int((eval_g["answer_status"] == "missing_or_invalid").sum()),
            "not_evaluable": int((g["answer_status"] == "not_evaluable").sum()),
            "accuracy_evaluable": correct / n_eval if n_eval else None,
            "score_evaluable": float(eval_g["score"].sum()) if n_eval else None,
            "api_error": bool(g["api_error"].iloc[0]) if len(g) else None,
            "parse_error": bool(g["parse_error"].iloc[0]) if len(g) else None,
            "finish_reason": g["finish_reason"].iloc[0] if len(g) else None,
            "latency_seconds": float(g["latency_seconds"].iloc[0]) if len(g) else None,
            "prompt_tokens": g["prompt_tokens"].iloc[0] if len(g) else None,
            "completion_tokens": g["completion_tokens"].iloc[0] if len(g) else None,
            "reasoning_tokens": g["reasoning_tokens"].iloc[0] if len(g) else None,
            "total_tokens": g["total_tokens"].iloc[0] if len(g) else None,
            "n_answers_returned_raw": g["n_answers_returned_raw"].iloc[0] if len(g) else None,
            "n_unique_questions_returned": g["n_unique_questions_returned"].iloc[0] if len(g) else None,
        })
    return pd.DataFrame(rows)


def summarize_by_image_status(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rep, has_image), g in df.groupby(["replicate", "has_image"], dropna=False):
        eval_g = g[g["has_official_answer"] == True]
        n_eval = len(eval_g)
        correct = int((eval_g["is_correct"] == True).sum())
        rows.append({
            "replicate": rep,
            "has_image": has_image,
            "n_total": int(len(g)),
            "n_evaluable": int(n_eval),
            "correct": correct,
            "incorrect": int((eval_g["answer_status"] == "incorrect").sum()),
            "blank": int((eval_g["answer_status"] == "blank").sum()),
            "missing_or_invalid": int((eval_g["answer_status"] == "missing_or_invalid").sum()),
            "not_evaluable": int((g["answer_status"] == "not_evaluable").sum()),
            "accuracy_evaluable": correct / n_eval if n_eval else None,
            "score_evaluable": float(eval_g["score"].sum()) if n_eval else None,
        })
    return pd.DataFrame(rows)


def run_exam(exam_key: str):
    cfg = EXAMS[exam_key]

    questions = load_questions(cfg["questions_csv"])
    answer_key = load_answer_key(cfg["answer_key_csv"])
    prompt = build_prompt(cfg["exam_name"], cfg["exam_label"], questions)

    run_id = f"{cfg['output_prefix']}_{utc_stamp()}"
    out_dir = Path("outputs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        timeout=600.0,
        max_retries=0,
    )

    metadata = {
        "run_id": run_id,
        "created_utc": utc_stamp(),
        "provider": "DeepSeek",
        "api_endpoint": "OpenAI-compatible Chat Completions",
        "base_url": "https://api.deepseek.com",
        "model_requested": MODEL,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "exam": exam_key,
        "questions_csv": str(cfg["questions_csv"]),
        "answer_key_csv": str(cfg["answer_key_csv"]),
        "n_replicates": N_REPLICATES,
        "n_questions": int(len(questions)),
        "n_image_containing_items_text_only": int(questions["has_image"].sum()),
        "generation_parameters": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "thinking": "disabled",
            "response_format": {"type": "json_object"},
            "stream": False,
            "tools_enabled": False,
            "web_search_enabled": False,
            "file_input_enabled": False,
            "input_type": "complete extracted exam text from CSV",
        },
        "scoring": {
            "correct": 1,
            "incorrect": -1/3,
            "blank": 0,
            "missing_or_invalid": 0,
            "questions_without_official_answer_excluded": True,
        },
    }

    all_rows: List[Dict[str, Any]] = []
    raw_jsonl_path = out_dir / f"{run_id}_raw_outputs.jsonl"

    print(f"\nRun ID: {run_id}")
    print(f"Exam: {exam_key.upper()}")
    print(f"Questions CSV: {cfg['questions_csv']}")
    print(f"Answer key CSV: {cfg['answer_key_csv']}")
    print(f"Output folder: {out_dir}")
    print(f"Truststore enabled: {TRUSTSTORE_ENABLED}")
    print(f"Prompt length characters: {len(prompt):,}")

    with raw_jsonl_path.open("w", encoding="utf-8") as raw_f:
        for rep in range(1, N_REPLICATES + 1):
            print(f"\n=== {exam_key.upper()} replicate {rep}/{N_REPLICATES} ===")
            start = time.time()
            api_error = False
            parse_error = False
            error_message = None
            raw_text = ""
            parsed = None
            response_dict = None
            model_returned = None
            finish_reason = None
            usage = {}
            answers_list = []

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    extra_body={"thinking": {"type": "disabled"}},
                )

                latency = time.time() - start
                response_dict = response.model_dump()
                model_returned = response_dict.get("model")
                choice = (response_dict.get("choices") or [{}])[0]
                finish_reason = choice.get("finish_reason")
                message = choice.get("message") or {}
                raw_text = message.get("content") or ""

                usage = response_dict.get("usage") or {}

                try:
                    parsed = extract_json_object(raw_text)
                    answers_list = parsed.get("answers", [])
                    if not isinstance(answers_list, list):
                        raise ValueError("'answers' is not a list")
                except Exception as e:
                    parse_error = True
                    error_message = f"Parse error: {repr(e)}"
                    answers_list = []

            except Exception as e:
                latency = time.time() - start
                api_error = True
                error_message = repr(e)

            answer_map: Dict[int, Dict[str, Any]] = {}
            duplicate_counts: Dict[int, int] = {}
            for item in answers_list:
                if not isinstance(item, dict):
                    continue
                qn = normalize_question_number(item.get("question_number"))
                if qn is None:
                    continue
                duplicate_counts[qn] = duplicate_counts.get(qn, 0) + 1
                if qn not in answer_map:
                    answer_map[qn] = item

            n_unique = len(answer_map)

            raw_record = {
                "replicate": rep,
                "api_error": api_error,
                "parse_error": parse_error,
                "error_message": error_message,
                "latency_seconds": latency,
                "model_requested": MODEL,
                "model_returned": model_returned,
                "finish_reason": finish_reason,
                "usage": usage,
                "n_answers_returned_raw": len(answers_list) if isinstance(answers_list, list) else None,
                "n_unique_questions_returned": n_unique,
                "raw_text": raw_text,
                "parsed": parsed,
                "response": response_dict,
            }
            raw_f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
            raw_f.flush()

            print(f"Latency: {latency:.1f}s")
            print(f"API error: {api_error}; parse error: {parse_error}; finish_reason: {finish_reason}; returned answers: {len(answers_list)}; unique Q: {n_unique}")
            if error_message:
                print(f"Error: {error_message}")

            for qn in range(1, 211):
                item = answer_map.get(qn)
                model_answer = normalize_answer(item.get("answer")) if item else None
                rationale = item.get("rationale") if item else None

                correct_row = answer_key[answer_key["question_number"] == qn]
                if len(correct_row):
                    correct_answer = correct_row["correct_answer"].iloc[0]
                    has_official = bool(correct_row["has_official_answer"].iloc[0])
                else:
                    correct_answer = None
                    has_official = False

                qrow = questions[questions["question_number"] == qn]
                if len(qrow):
                    has_image = bool(qrow["has_image"].iloc[0])
                    is_reserve = bool(qrow["is_reserve"].iloc[0])
                else:
                    has_image = False
                    is_reserve = qn > 200

                score_info = score_answer(model_answer, correct_answer, has_official)

                row = {
                    "replicate": rep,
                    "question_number": qn,
                    "question_id": f"Q{qn:03d}",
                    "model_answer": model_answer,
                    "rationale": rationale,
                    "correct_answer": correct_answer,
                    "has_image": has_image,
                    "is_reserve": is_reserve,
                    "duplicate_count_in_response": duplicate_counts.get(qn, 0),
                    "missing_from_response": item is None,
                    "api_error": api_error,
                    "parse_error": parse_error,
                    "error_message": error_message,
                    "finish_reason": finish_reason,
                    "latency_seconds": latency,
                    "n_answers_returned_raw": len(answers_list) if isinstance(answers_list, list) else None,
                    "n_unique_questions_returned": n_unique,
                    "model_requested": MODEL,
                    "model_returned": model_returned,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "reasoning_tokens": (
                        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                        if isinstance(usage.get("completion_tokens_details"), dict)
                        else None
                    ),
                    **score_info,
                }
                all_rows.append(row)

    results = pd.DataFrame(all_rows)

    scored_path = out_dir / f"{run_id}_scored_results.csv"
    summary_path = out_dir / f"{run_id}_summary_by_replicate.csv"
    image_summary_path = out_dir / f"{run_id}_summary_by_image_status.csv"
    metadata_path = out_dir / f"{run_id}_run_metadata.json"

    results.to_csv(scored_path, index=False, encoding="utf-8")

    summary = summarize(results)
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    image_summary = summarize_by_image_status(results)
    image_summary.to_csv(image_summary_path, index=False, encoding="utf-8")

    metadata["completed_utc"] = utc_stamp()
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nDONE")
    print(f"Raw outputs: {raw_jsonl_path}")
    print(f"Scored results: {scored_path}")
    print(f"Summary: {summary_path}")
    print(f"Summary by image status: {image_summary_path}")
    print(f"Metadata: {metadata_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nSummary by image status:")
    print(image_summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exam",
        choices=["bir", "fir", "both"],
        required=True,
        help="Which extracted-text whole-exam run to execute.",
    )
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("Missing DEEPSEEK_API_KEY in .env")

    exams_to_run = ["bir", "fir"] if args.exam == "both" else [args.exam]
    for exam_key in exams_to_run:
        run_exam(exam_key)


if __name__ == "__main__":
    main()

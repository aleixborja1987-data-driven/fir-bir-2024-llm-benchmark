# Request message roles and historical deviations

| Model/workflow | Explicit system message | User message and evidence |
|---|---|---|
| GPT-5.2, Grok 4.3, Qwen3-VL, all selected workflows | None | Exam instructions and question/full exam in user content; FIR image blocks where applicable. Original scripts establish request roles. |
| Claude Opus 4.8, all selected workflows | None | Instructions and question/full exam in user content; output forced through `record_answer` or `record_all_answers` tool schemas. These are output-format schemas, not externally executed tools. |
| DeepSeek item-wise | `You are a biomedical multiple-choice exam solver. Return only valid JSON. Do not include markdown.` | Exam instruction and individual question; FIR image-blind. |
| DeepSeek whole-exam | None | Full exam and brief rationales requested in both BIR and FIR. Pure-function reconstructed prompts are labelled; no full captured request payload is claimed. |
| Kimi item-wise | Original system instruction in metadata/script | Question text and FIR image blocks; FIR system literally says **Clinical Physics residency access**. This historical error is preserved. |
| Kimi BIR whole-exam | `PROMPT_PREAMBLE` | Full prompt with that preamble removed using `removeprefix(...).lstrip()` in the original script. |
| Kimi FIR whole-exam | None | User content blocks contain instruction, questions and images; the preamble also uses Clinical Physics residency access, as preserved in Appendix 2. |

The source scripts are the authoritative reference for role assignment, schema and request construction. `data/prompt_index.csv` distinguishes logged text, extracted metadata and reconstructed text. A preamble alone is not a full user payload; an unlabelled historical `prompt` field is not proof of a system role.

DeepSeek received no FIR images in either workflow. Its whole-exam instruction permits blank when the text is insufficient, and its observed 20-image subset is entirely blank. Other whole-exam workflows request answers without rationales. Whole-exam DeepSeek results therefore do not isolate aggregation with identical output requirements.

The original Grok BIR item-wise metadata records `definitive_run_completed_as_prespecified=false`. Three scored recoveries and the logged retry counters are retained. Full responses for every discarded attempt are not available. See `data/request_records.csv` and the original logs; do not describe all 48 runs as deviation-free.

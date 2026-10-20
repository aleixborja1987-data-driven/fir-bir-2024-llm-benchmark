# Methods implemented by the offline reconstruction

## Basis and scope

The package contains the 24 conditions specified in config/run_manifest.json (48 replicate runs). Selected acquisition files and the 20 original FIR images were reconciled with the original archive using SHA256. Source mappings, original/public hashes and redactions are in provenance/. Historical derived summaries are excluded; scores are reconstructed directly from raw outputs.

`provenance/original_environment.json` records the archived venv's declared Python 3.14.6 and package metadata. It establishes the environment present in this archive, not the runtime used for each historical command. `analysis_environment.json` records the independently tested offline runtime.

## Answer extraction precedes scoring

The adapter reads emitted JSON strings or emitted Anthropic tool-input objects. Tool schemas here describe output formatting; the presence of a tool-input object is not treated as evidence of web search or external retrieval. Whole-exam outputs must contain each question number 1–210 exactly once. For per-attempt logs, the final record for a replicate/question is retained; the entire original log and all attempt rows remain accessible. This matches the final successful recorded requests; tests and cross-checks must be reviewed before release.

A unique explicit selected answer is retained when only an auxiliary rationale violated a length constraint or was truncated. For the truncated DeepSeek record, recovery accepts only an explicit leading JSON `answer` field followed by the rationale field. It does not guess an answer from free prose. Explicit multiple options are recoded as blank. Original answer values, original parse status, source file, line number, normalized answer, rule and score are preserved. Reference answers are joined only after extraction. An incorrect answer is -1/3, a correct answer +1, and blank or missing/invalid 0.

The 43 scored recoveries comprise 1 DeepSeek BIR, 3 Grok BIR, 15 Grok FIR, 5 Opus BIR and 19 Opus FIR responses. There is one additional Opus FIR recovery on unused reserve Q210. The 11 multiple-option outputs comprise 1 Qwen BIR and 10 Qwen FIR responses. These counts describe reconstructed rule application; the package does not claim that a human auditor was blinded or that an undocumented historical adjudication process occurred.

## Scoring selection

The first 200 questions in numerical order with an available official answer are selected, using the selection algorithm implemented in `analysis/reproduce.py`.

In the supplied FIR key, main items 53, 169, 172 and 190 have no official answer. Reserves 201–204 are included; 205–210 are unused.

In the supplied BIR key, main items 198 and 200 and reserve items 202 and 206 have no official answer. Reserves 201 and 203 are included; valid reserves 204, 205 and 207–210 are unused. Thus, two main questions are replaced by reserves 201 and 203, skipping reserve 202 because it is unkeyed. The author confirmed this selection.

No Ministry final-answer-key PDF is included. The question booklets and supplied key CSVs are included; the selected set has been confirmed by the corresponding author. The selected set reproduces the previously reported 24 mean accuracies and scores, but that agreement is a consistency check, not proof that the key is official.

## Bootstrap

The supplied `bootstrap_ci.py` is executed through its pure analysis function. Within each question the two replicate outcomes are averaged; question clusters are sampled with replacement. There are 10,000 percentile-bootstrap draws with NumPy default_rng seed 2026, reinitialized for each model/workflow/stratum. Intervals are 2.5th/97.5th percentiles. Accuracy is the mean of item outcomes times 100, while examination score is the sum of item scores for the stated stratum. FIR text-only scores have maximum 180, image-stratum scores maximum 20; these are not rescaled to 200.

These intervals quantify item-level variation conditional on the observed two runs; they do not fully quantify variability across future calls, future model versions, servers, or examinations. The image-blind whole-exam DeepSeek subset has an empirical zero-width bootstrap interval when all 20 observed responses are blank; this does not mean zero uncertainty for other future experiments.

## Agreement

Replicate agreement is exact final-answer agreement over 200 items, including blank as a category. Between-workflow agreement is binary correctness, with matching question and replicate IDs (400 pairs). For two binary raters, pooled correct prevalence is p, the Gwet AC1 chance term is 2p(1-p), and AC1 = (observed agreement - chance term)/(1 - chance term). It is a descriptive agreement measure, not a superiority test. Supplemental model-level comparisons are generated in `paired_sign_tests_and_reductions.csv`: exact two-sided binomial sign tests on whole-exam minus item-wise condition means, excluding ties, with six paired model observations per examination. These do not establish equivalence and are distinct from the descriptive agreement calculations.

## Operational metrics

Original usage dictionaries are retained. Input/output/reasoning/cache counts are exported separately. Missing reasoning-token fields remain missing rather than being assumed zero. Kimi records of one reasoning token are retained as reported. Reasoning tokens already included in the provider's completion count are not added again.

Separate fields retain final-attempt latency, source-logged latency and retry-inclusive elapsed time when present. A sum of request latencies is not necessarily wall-clock execution duration. The original acquisition scripts use serial request loops; some responses are streamed, which is not concurrent dispatch. Failed-attempt counts and sleep/wait overhead are not uniformly logged. At least 5,099 attempted requests are represented by counters/logs across the 48 runs, versus 5,064 nominal planned calls. This is not a certified billing ledger or complete history of all preliminary experiments.

Estimated list-price cost is only computed when both historical unit prices and `status=author_verified` are entered in `config/api_prices.csv`. The formula is (input tokens × input price + output tokens × output price) / 1,000,000. It is a standard uncached-price estimate and does not establish actual invoiced expenditure, cache discounts, failed-call charges or energy consumption. The rate table has been completed following author confirmation and source checks. The DeepSeek verification scope is explicitly recorded in the table; sources accessed now are not represented as archived snapshots from the historical consultation date.

## Prompts and workflow comparison

`data/prompt_index.csv` identifies verbatim captured prompts and metadata preambles. DeepSeek full-exam prompts were not captured as full text in the raw log, so they are materialized from the supplied script's pure formatting functions and supplied question CSVs; they are explicitly labelled reconstructed, not independently captured requests.

DeepSeek whole-exam prompting requested a brief rationale for every answer and explicitly stated that the images were not provided. The other saved whole-exam workflows are answer-only. The Kimi FIR item-wise system prompt literally says “Clinical Physics residency access”; it has not been silently rewritten as Pharmacy. These source-level differences are preserved in the exact prompt templates supplied in Multimedia Appendix 2.

The two workflows jointly vary aggregation, repeated instructions, number of calls, output length and some prompt details. They do not isolate a causal effect of context length alone. The 180 text-only FIR scoring subset provides common text items, but other images were still present elsewhere in the multimodal models' whole-exam requests.

## Input integrity and prompt reconstruction additions

All 20 image files are checked against original source hashes. Nine multimodal conditions provide 180 request/manifest hash comparisons, all required to match. Kimi FIR item-wise contributes 20 explicitly unlogged request hashes: its original files and detection evidence are available, but independent hash confirmation of the request payload is not claimed. All ten multimodal conditions appear in the coverage table; DeepSeek's image-blind conditions are excluded from that table by design.

The full Qwen BIR whole-exam user prompt is reconstructed from its formatting functions and matches its recorded full-prompt SHA256. Kimi FIR item-wise user text and image references are reconstructed from its original functions/inputs, without claiming a recorded payload hash. See `docs/PROMPT_ROLES.md`.

Question-latency quartiles pool the 420 retained item-wise requests per condition, including reserve questions; they are not averages of per-replicate quantiles. Whole-exam request latency cannot be allocated empirically to individual questions; normalization by 200 is a descriptive per-scored-question ratio only.

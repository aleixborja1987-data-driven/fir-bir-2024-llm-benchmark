# FIR/BIR 2024 reproducibility package — release candidate

Offline reconstruction of the 48 runs selected for the current JMIR AI manuscript: six models × two examinations × two workflows × two replicates. This package contains the recorded responses, 9,600 scored responses, scoring adapter, original bootstrap script, and original question/image inputs. It is not yet a published or DOI-assigned release. Read `docs/OPEN_ITEMS.md` before publication.

## Reproduce

Use Python 3.12 (tested: 3.12.14). In this directory:

```bash
python -m venv .venv
# Activate the environment using your operating system's activation command.
python -m pip install -r requirements.txt
python verify_checksums.py
python reproduce.py
python -m unittest discover -s tests -v
python analysis/independent_verify.py . . results/independent
```

`python reproduce.py --no-figures` runs the same scoring, bootstrap, prompt reconstruction and tables without plotting. No API account, key, network access or paid model calls are needed after installing dependencies. The current verification used the installed pinned versions; installation from a new dependency download and Windows execution have not been tested.

`verify_checksums.py` verifies the distributed snapshot **before** rerunning. Regeneration may change generated-file hashes, especially SVG metadata. Input integrity is also checked within the analysis. A missing/corrupt image or incomplete experiment fails the command. See `results/validation.json`; it certifies only the checks implemented, not manuscript agreement, price verification or publication readiness.

## File roles

| Location | Role |
|---|---|
| `config/run_manifest.json` | Explicit list of the 24 selected conditions and their raw/metadata files. |
| `source_data/inputs/` | Supplied question booklets, 210-question/key CSVs, and 20 original FIR PNGs. |
| `source_data/runs/<condition>/raw.jsonl` | Sanitized original acquisition records. Answer text and scientific parameters are preserved. |
| Other files in `source_data/runs/` | Acquisition metadata, prompt logs, image manifests and input manifests. Historical derived result/summary CSVs have been removed; canonical scores are regenerated in `data/`. |
| `source_scripts/acquisition/` | Historical acquisition scripts; retained for inspecting actual prompts/settings. Never executed by the offline workflow; paths/dependencies need adaptation for new API runs. |
| `source_scripts/analysis/` | Original bootstrap implementation called by the portable scoring adapter. Use `python reproduce.py` for the complete workflow. |
| `analysis/` and `reproduce.py` | Canonical portable reanalysis, documentation, figures and independent verification. These are reconstruction code, not a claim about the date of original analysis. |
| `data/official_200_scored_items.csv` | Definitive 9,600 scored responses, each linked to its raw file and line. |
| `data/all_210_responses_rescored.csv` | All 10,080 answer positions, including unscored reserves/annulled items. |
| `data/reclassification_audit_*.csv` | Original answer, extraction/recode rule, final scored answer and source pointer. |
| `data/prompts/`, `data/prompt_index.csv` | Logged and reconstructed prompts, explicitly distinguished. Roles: `docs/PROMPT_ROLES.md`. |
| `results/tables/` | Regenerated scores, 10,000-draw cluster-bootstrap intervals, efficiency and paired descriptive sign tests. |
| `results/figures/` | New reproducible plots; filenames are not the manuscript's figure numbering. See `docs/FIGURE_MAP.md`. |
| `provenance/` | Hashes/redactions of deposited sources, original environment metadata, and tested reanalysis environment. |

Scoring selects the first 200 numerically ordered items with answers in the supplied key, then applies correct − incorrect/3 (blank/missing-invalid: zero). Question clusters retain both replicates for each bootstrap draw; 10,000 draws, NumPy default_rng seed 2026 reset for every condition/subset, percentile 95% intervals. See `docs/METHODS_AND_PROVENANCE.md` and `docs/DATA_DICTIONARY.md`.

Estimated costs use the standard input/output rates in `config/api_prices.csv`, without cache discounts or charges for unrecorded discarded attempts. Sources and verification scope are recorded in that table. Current results use consistent final-attempt latency; retry-inclusive and source-logged fields remain separate. Supplemental sign tests are exact two-sided paired tests over six models, with ties excluded; they do not establish equivalence.

Only the 24 manuscript conditions and their 48 definitive replicate runs are distributed. Auxiliary pilots and unrelated experiments are outside this package. Within the selected runs, original retry records and extraction/reclassification evidence are preserved.

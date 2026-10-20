#!/usr/bin/env python3
"""Bootstrap 95% CIs for accuracy and examination score.

Reproduces the bootstrap analysis used in the manuscript revision:
- 10,000 non-parametric bootstrap resamples by default;
- question/item is the resampling unit;
- the two replicate responses for a question remain together as a cluster;
- percentile 95% confidence intervals;
- fixed random seed = 2026.

Operationally, the two replicate outcomes are first averaged within each
question. Questions are then sampled with replacement. This is equivalent to
resampling question clusters while retaining both replicate responses together.
The RNG is re-initialized with the same fixed seed for every model-condition,
which reproduces the reported revision outputs and makes each CI independent of
group iteration order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scored", type=Path, required=True, help="official_200_scored_items.csv from score_responses.py")
    p.add_argument("--output-dir", type=Path, default=Path("derived"))
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def bootstrap_group(g: pd.DataFrame, n_bootstrap: int, seed: int) -> dict[str, float]:
    # Retain the two replicates together by collapsing to one cluster row/question.
    item = (
        g.groupby("question_number", sort=True)
        .agg(mean_correct=("is_correct", "mean"), mean_score=("score", "mean"), n_replicates=("replicate", "nunique"))
        .reset_index()
    )
    if not (item["n_replicates"] == 2).all():
        bad = item.loc[item["n_replicates"] != 2, ["question_number", "n_replicates"]]
        raise ValueError(f"Each question must retain exactly two replicates. Problem rows:\n{bad}")

    n_items = len(item)
    correct = item["mean_correct"].to_numpy(dtype=float)
    score = item["mean_score"].to_numpy(dtype=float)

    # This reset-per-condition seed reproduces the analysis used for the revision.
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, n_items, size=(n_bootstrap, n_items))

    boot_accuracy_pct = correct[sample_idx].mean(axis=1) * 100.0
    boot_score = score[sample_idx].sum(axis=1)

    acc_low, acc_high = np.percentile(boot_accuracy_pct, [2.5, 97.5])
    score_low, score_high = np.percentile(boot_score, [2.5, 97.5])

    return {
        "n_items": n_items,
        "mean_correct": correct.sum(),
        "mean_accuracy_pct": correct.mean() * 100.0,
        "mean_score": score.sum(),
        "acc_ci_low_pct": acc_low,
        "acc_ci_high_pct": acc_high,
        "score_ci_low": score_low,
        "score_ci_high": score_high,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def run_subset(df: pd.DataFrame, subset_name: str, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    for (model, exam, strategy), g in df.groupby(["model", "exam", "strategy"], sort=False):
        out = bootstrap_group(g, n_bootstrap=n_bootstrap, seed=seed)
        rows.append({"model": model, "exam": exam, "strategy": strategy, "subset": subset_name, **out})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scored)
    required = {"model", "exam", "strategy", "replicate", "question_number", "is_correct", "score", "has_image"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Scored file is missing columns: {sorted(missing)}")

    # Main endpoint: 200 official questions per examination.
    main_200 = run_subset(df, "official_200", args.n_bootstrap, args.seed)
    if not (main_200["n_items"] == 200).all():
        raise ValueError("Main bootstrap expected exactly 200 items per condition.")
    main_200.to_csv(outdir / "bootstrap_official_200.csv", index=False)

    # Common-input FIR sensitivity analysis: exclude all 20 image-containing items.
    fir = df[df["exam"].astype(str).str.upper().str.startswith("FIR")].copy()
    fir_text = fir[~fir["has_image"].astype(bool)].copy()
    text_180 = run_subset(fir_text, "fir_text_only_180", args.n_bootstrap, args.seed)
    if not (text_180["n_items"] == 180).all():
        raise ValueError("FIR text-only bootstrap expected exactly 180 items per condition.")
    text_180.to_csv(outdir / "bootstrap_fir_text_only_180.csv", index=False)

    # Descriptive multimodal subgroup.
    fir_img = fir[fir["has_image"].astype(bool)].copy()
    image_20 = run_subset(fir_img, "fir_image_20", args.n_bootstrap, args.seed)
    if not (image_20["n_items"] == 20).all():
        raise ValueError("FIR image bootstrap expected exactly 20 items per condition.")
    image_20.to_csv(outdir / "bootstrap_fir_image_20.csv", index=False)

    print(f"Bootstrap resamples per model-condition: {args.n_bootstrap:,}")
    print(f"Seed: {args.seed}")
    print(f"Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute synchronized row-bootstrap CIs for Table 4 mean F1 values."""

import argparse

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = (
    SCRIPT_DIR.parents[1]
    / "0. Dataset"
    / "modeling_outputs"
    / "06_AspectLabel"
    / "test_predictions_aspect.csv"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "aspect_mean_f1_bootstrap.csv"
ASPECTS = [
    "depression",
    "anxiety",
    "suicidal",
    "stress",
    "bipolar",
    "personality_disorder",
]
MODELS = ["albert", "biobert"]
AVERAGES = ["weighted", "macro"]
N_RESAMPLES = 1_000
SEED = 2025


def mean_f1(frame: pd.DataFrame, model: str, average: str, indices=None) -> float:
    scores = []
    for aspect in ASPECTS:
        truth = frame[f"{aspect}_true"].to_numpy()
        pred = frame[f"{model}_{aspect}_pred"].to_numpy()
        if indices is not None:
            truth = truth[indices]
            pred = pred[indices]
        scores.append(f1_score(truth, pred, average=average))
    return float(np.mean(scores))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.predictions)
    rng = np.random.RandomState(SEED)
    samples = [rng.randint(0, len(frame), len(frame)) for _ in range(N_RESAMPLES)]
    rows = []
    for model in MODELS:
        for average in AVERAGES:
            observed = mean_f1(frame, model, average)
            replicates = np.asarray(
                [mean_f1(frame, model, average, indices) for indices in samples]
            )
            low, high = np.percentile(replicates, [2.5, 97.5])
            rows.append(
                {
                    "model": model,
                    "metric": f"mean_{average}_f1_across_six_heads",
                    "estimate": observed,
                    "ci_low": low,
                    "ci_high": high,
                    "confidence_level": 0.95,
                    "bootstrap_resamples": N_RESAMPLES,
                    "bootstrap_seed": SEED,
                    "bootstrap_unit": "held-out test post shared across six heads",
                    "n_test_posts": len(frame),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

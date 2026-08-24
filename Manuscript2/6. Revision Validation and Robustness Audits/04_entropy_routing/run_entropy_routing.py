#!/usr/bin/env python3
"""Review-budget analysis for released-label versus AI-label disagreement.

The analysis ranks full-corpus records for manual review using four observable
signals plus an expected random baseline.  The category baseline is cross-fit:
each record is scored by its released category's disagreement rate estimated
from the other four text-group folds, so its own outcome is never used in its
priority score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = (
    Path(__file__).resolve().parents[2]
    / "0. Dataset"
    / "analysis_ready"
    / "mental_health_unified_labels_final.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"

PROB_COLS = [
    "u_p_normal",
    "u_p_depression",
    "u_p_anxiety",
    "u_p_suicidal",
    "u_p_stress",
    "u_p_bipolar",
    "u_p_personality_disorder",
]
KEY_BUDGETS = [1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0]
N_FOLDS = 5
SEED = 42


def clean_text(text: object) -> str:
    """Match the R2 training notebooks' statement normalization."""
    if pd.isna(text):
        return ""
    value = str(text).lower()
    value = re.sub(r"http\S+", " urltoken ", value)
    value = re.sub(r"@\w+", " usertoken ", value)
    value = re.sub(r"#(\w+)", r" hashtag_\1 ", value)
    value = re.sub(r"[^\w\s]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def text_fold(text: str, n_folds: int = N_FOLDS) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % n_folds


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_released_label(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", "_", regex=True)
    )


def make_category_oof_score(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Cross-fit disagreement prevalence by released category and text group."""
    scores = np.empty(len(df), dtype=float)
    rows: list[dict[str, object]] = []
    for fold in range(N_FOLDS):
        train = df.loc[df["group_fold"] != fold]
        held_out = df.loc[df["group_fold"] == fold]
        global_rate = float(train["disagreement"].mean())
        rates = train.groupby("released_label", observed=False)["disagreement"].mean()
        held_scores = held_out["released_label"].map(rates).fillna(global_rate)
        scores[held_out.index.to_numpy()] = held_scores.to_numpy(dtype=float)
        for category, rate in rates.items():
            rows.append(
                {
                    "held_out_fold": fold,
                    "released_label": category,
                    "training_n": int((train["released_label"] == category).sum()),
                    "training_disagreement_rate": float(rate),
                    "held_out_n": int((held_out["released_label"] == category).sum()),
                }
            )
    return scores, pd.DataFrame(rows)


def ranked_curve(
    outcome: np.ndarray,
    priority_score: np.ndarray,
    budgets: np.ndarray,
    method: str,
    tie_breaker: np.ndarray,
) -> pd.DataFrame:
    # Lexicographic sorting gives high score first and uses a seeded,
    # outcome-independent tie breaker for the many rounded score-vector ties.
    order = np.lexsort((tie_breaker, -priority_score))
    ranked = outcome[order]
    cumulative = np.cumsum(ranked)
    total_events = int(outcome.sum())
    global_rate = float(outcome.mean())
    records: list[dict[str, object]] = []
    for budget in budgets:
        reviewed_n = min(len(outcome), max(1, math.ceil(len(outcome) * budget / 100.0)))
        captured_n = int(cumulative[reviewed_n - 1])
        yield_rate = captured_n / reviewed_n
        records.append(
            {
                "method": method,
                "budget_percent": float(budget),
                "reviewed_n": reviewed_n,
                "captured_disagreements_n": captured_n,
                "disagreement_capture": captured_n / total_events,
                "routed_disagreement_rate": yield_rate,
                "lift_over_random": yield_rate / global_rate,
            }
        )
    return pd.DataFrame(records)


def expected_random_curve(
    outcome: np.ndarray, budgets: np.ndarray, method: str = "Random (expected)"
) -> pd.DataFrame:
    total_events = int(outcome.sum())
    n = len(outcome)
    global_rate = total_events / n
    rows: list[dict[str, object]] = []
    for budget in budgets:
        reviewed_n = min(n, max(1, math.ceil(n * budget / 100.0)))
        expected_events = reviewed_n * global_rate
        rows.append(
            {
                "method": method,
                "budget_percent": float(budget),
                "reviewed_n": reviewed_n,
                "captured_disagreements_n": expected_events,
                "disagreement_capture": reviewed_n / n,
                "routed_disagreement_rate": global_rate,
                "lift_over_random": 1.0,
            }
        )
    return pd.DataFrame(rows)


def plot_curves(curves: pd.DataFrame, output_dir: Path) -> None:
    labels = [
        "Entropy",
        "Maximum score",
        "Top-two margin",
        "Released category (OOF)",
        "Random (expected)",
    ]
    colors = {
        "Entropy": "#0072B2",
        "Maximum score": "#D55E00",
        "Top-two margin": "#009E73",
        "Released category (OOF)": "#CC79A7",
        "Random (expected)": "#666666",
    }
    linestyles = {label: ("--" if label == "Random (expected)" else "-") for label in labels}

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.15), constrained_layout=True)
    for label in labels:
        part = curves.loc[curves["method"] == label]
        axes[0].plot(
            part["budget_percent"],
            100 * part["disagreement_capture"],
            label=label,
            color=colors[label],
            linestyle=linestyles[label],
            linewidth=2,
        )
        axes[1].plot(
            part["budget_percent"],
            100 * part["routed_disagreement_rate"],
            label=label,
            color=colors[label],
            linestyle=linestyles[label],
            linewidth=2,
        )

    axes[0].set_title("A. Disagreements captured")
    axes[0].set_ylabel("Share of all disagreements (%)")
    axes[1].set_title("B. Disagreement yield")
    axes[1].set_ylabel("Disagreements among routed posts (%)")
    for axis in axes:
        axis.set_xlabel("Review budget (% of corpus)")
        axis.set_xlim(0, 50)
        axis.grid(alpha=0.22, linewidth=0.8)
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(0, 100)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "routing_review_budget_curves.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "routing_review_budget_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["statement", "status", "u_label", *PROB_COLS]
    df = pd.read_csv(args.input, usecols=columns)
    if len(df) != 53_043:
        raise ValueError(f"Expected 53,043 records, found {len(df):,}")

    probs = df[PROB_COLS].to_numpy(dtype=float)
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("Score vectors contain non-finite or negative values")
    raw_sums = probs.sum(axis=1)
    if (raw_sums <= 0).any():
        raise ValueError(
            "At least one seven-class score vector has a non-positive row sum"
        )
    # The stored seven class scores are non-negative evidence weights, not
    # guaranteed to sum to one.  Match the R2 soft-label notebooks by
    # row-normalizing before entropy, maximum-score, and margin calculations.
    probs = probs / raw_sums[:, None]
    sums = probs.sum(axis=1)

    df["released_label"] = normalize_released_label(df["status"])
    df["ai_hard_label"] = df["u_label"].astype(str).str.strip().str.upper()
    df["disagreement"] = (df["released_label"] != df["ai_hard_label"]).astype(int)
    df["clean_statement"] = df["statement"].map(clean_text)
    df["group_fold"] = df["clean_statement"].map(text_fold)

    # Evaluate the Shannon convention 0 log(0) = 0 exactly.  Clipping zero
    # probabilities to epsilon would create artificial differences around
    # 1e-14 and silently break genuinely tied entropy values.
    entropy_terms = np.zeros_like(probs)
    positive = probs > 0
    entropy_terms[positive] = -probs[positive] * np.log(probs[positive])
    entropy = entropy_terms.sum(axis=1) / np.log(probs.shape[1])
    sorted_probs = np.sort(probs, axis=1)
    maximum_score = sorted_probs[:, -1]
    top_two_margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    category_score, category_folds = make_category_oof_score(df)

    priority_scores = {
        "Entropy": entropy,
        "Maximum score": 1.0 - maximum_score,
        "Top-two margin": 1.0 - top_two_margin,
        "Released category (OOF)": category_score,
    }
    outcome = df["disagreement"].to_numpy(dtype=int)
    rng = np.random.default_rng(SEED)
    tie_breaker = rng.random(len(df))
    budgets = np.arange(0.5, 100.0 + 0.5, 0.5)
    curves = pd.concat(
        [
            *[
                ranked_curve(outcome, score, budgets, name, tie_breaker)
                for name, score in priority_scores.items()
            ],
            expected_random_curve(outcome, budgets),
        ],
        ignore_index=True,
    )
    key = curves.loc[curves["budget_percent"].isin(KEY_BUDGETS)].copy()

    auc_rows = [
        {"method": name, "disagreement_auroc": roc_auc_score(outcome, score)}
        for name, score in priority_scores.items()
    ]
    auc_rows.append({"method": "Random (expected)", "disagreement_auroc": 0.5})
    auc_table = pd.DataFrame(auc_rows)

    threshold_mask = entropy >= 0.50
    threshold_summary = {
        "rule": "normalized_entropy_ge_0.50",
        "reviewed_n": int(threshold_mask.sum()),
        "review_share": float(threshold_mask.mean()),
        "captured_disagreements_n": int(outcome[threshold_mask].sum()),
        "disagreement_capture": float(outcome[threshold_mask].sum() / outcome.sum()),
        "routed_disagreement_rate": float(outcome[threshold_mask].mean()),
    }

    score_table = pd.DataFrame(
        {
            "row_id": np.arange(len(df), dtype=int),
            "released_label": df["released_label"],
            "ai_hard_label": df["ai_hard_label"],
            "disagreement": outcome,
            "normalized_entropy": entropy,
            "maximum_score": maximum_score,
            "top_two_margin": top_two_margin,
            "released_category_oof_disagreement_rate": category_score,
            "text_group_fold": df["group_fold"],
        }
    )

    curves.to_csv(output_dir / "routing_review_budget_curves.csv", index=False)
    key.to_csv(output_dir / "routing_review_budget_key_points.csv", index=False)
    auc_table.to_csv(output_dir / "routing_disagreement_auroc.csv", index=False)
    category_folds.to_csv(output_dir / "routing_category_oof_rates.csv", index=False)
    score_table.to_csv(output_dir / "routing_row_scores.csv", index=False)
    plot_curves(curves, output_dir)

    input_hash = sha256_file(args.input)
    manifest = {
        "input_file": args.input.name,
        "input_sha256": input_hash,
        "n_records": int(len(df)),
        "n_unique_clean_text_groups": int(df["clean_statement"].nunique(dropna=False)),
        "n_disagreements": int(outcome.sum()),
        "global_disagreement_rate": float(outcome.mean()),
        "raw_score_sum_min": float(raw_sums.min()),
        "raw_score_sum_max": float(raw_sums.max()),
        "normalized_probability_sum_min": float(sums.min()),
        "normalized_probability_sum_max": float(sums.max()),
        "seed_for_ties": SEED,
        "category_baseline_folds": N_FOLDS,
        "budget_grid_percent": {"start": 0.5, "stop": 100.0, "step": 0.5},
        "reviewed_n_rule": "ceil(n_records * budget_percent / 100)",
        "entropy_zero_rule": "0*log(0)=0 exactly; no epsilon tie breaking",
        "threshold_check": threshold_summary,
    }
    (output_dir / "routing_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2))
    print("\nAUROC")
    print(auc_table.to_string(index=False))
    print("\nKey review budgets")
    print(
        key.loc[key["budget_percent"].isin([2.5, 5.0, 10.0, 20.0])].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()

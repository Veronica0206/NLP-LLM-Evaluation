#!/usr/bin/env python3
"""Recompute chance-corrected human-audit aspect agreement.

The bootstrap resampling unit is the sampled post.  Each replicate therefore
keeps all six aspect ratings for a sampled post together and resamples all 300
posts with replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MANUSCRIPT_ROOT = BASE_DIR.parents[1]
DEFAULT_INPUT_DIR = MANUSCRIPT_ROOT / "0. Dataset" / "human_audit"
DEFAULT_AUDIT_MATRIX = DEFAULT_INPUT_DIR / "human_validation_aspect_matrix_300.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"

ASPECTS = (
    "depression",
    "anxiety",
    "suicidal",
    "stress",
    "bipolar",
    "personality_disorder",
)
ASPECT_DISPLAY = {
    "depression": "Depression",
    "anxiety": "Anxiety",
    "suicidal": "Suicidal",
    "stress": "Stress",
    "bipolar": "Bipolar",
    "personality_disorder": "Personality Disorder",
}
LEVEL_TO_INT = {"none": 0, "weak": 1, "clear": 2}
METRIC_NAMES = (
    "three_level_exact_agreement",
    "three_level_nominal_kappa",
    "three_level_linear_weighted_kappa",
    "binary_exact_agreement",
    "binary_nominal_kappa",
)
COMPARISONS = (
    ("Annotator 1 vs. Annotator 2", "a1", "a2"),
    ("Annotator 1 vs. AI aspects", "a1", "ai"),
    ("Annotator 2 vs. AI aspects", "a2", "ai"),
)
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 2025


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_levels(series: pd.Series, field_name: str) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    invalid = normalized.isna() | ~normalized.isin(LEVEL_TO_INT)
    if invalid.any():
        values = sorted(normalized.loc[invalid].astype(str).unique().tolist())
        raise AssertionError(f"Invalid values in {field_name}: {values}")
    return normalized


def normalize_boolean(series: pd.Series, field_name: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
    }
    normalized = series.map(lambda value: mapping.get(value, mapping.get(str(value).strip().lower())))
    if normalized.isna().any():
        values = sorted(series.loc[normalized.isna()].astype(str).unique().tolist())
        raise AssertionError(f"Invalid boolean values in {field_name}: {values}")
    return normalized.astype(bool)


def read_inputs(audit_matrix_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    merged = pd.read_csv(audit_matrix_path)
    required = {"sample_id"}
    required.update(
        f"{source}_{aspect}"
        for source in ("a1", "a2", "ai")
        for aspect in ASPECTS
    )
    missing = sorted(required.difference(merged.columns))
    if missing:
        raise AssertionError(f"Audit matrix is missing required columns: {missing}")
    if len(merged) != 300:
        raise AssertionError(f"Audit matrix has {len(merged)} rows, expected 300")
    if merged["sample_id"].isna().any() or merged["sample_id"].nunique() != 300:
        raise AssertionError("Audit matrix does not contain 300 unique, nonmissing sample IDs")

    merged = merged.sort_values("sample_id", kind="stable").reset_index(drop=True)
    for source in ("a1", "a2", "ai"):
        for aspect in ASPECTS:
            column = f"{source}_{aspect}"
            merged[column] = normalize_levels(merged[column], column)

    presence_columns = [f"ai_{aspect}_present" for aspect in ASPECTS]
    presence_check_available = all(column in merged.columns for column in presence_columns)
    ai_presence_consistent: bool | None = None
    if presence_check_available:
        ai_presence_consistent = True
        for aspect, present_column in zip(ASPECTS, presence_columns):
            present = normalize_boolean(merged[present_column], present_column)
            expected_present = merged[f"ai_{aspect}"].ne("none")
            if not np.array_equal(present.to_numpy(dtype=bool), expected_present.to_numpy(dtype=bool)):
                ai_presence_consistent = False
                break
        if not ai_presence_consistent:
            raise AssertionError("AI strength and presence fields are inconsistent")

    validation = {
        "audit_matrix_rows": int(len(merged)),
        "unique_sample_ids": int(merged["sample_id"].nunique()),
        "all_strength_values_valid": True,
        "ai_presence_check_available": presence_check_available,
        "ai_strength_presence_consistent": ai_presence_consistent,
        "contains_post_text": bool(
            any(column.lower() in {"statement", "text", "post"} for column in merged.columns)
        ),
    }
    if validation["contains_post_text"]:
        raise AssertionError("Public audit matrix must not contain post text")
    return merged, validation


def cohen_kappa(first: np.ndarray, second: np.ndarray, n_levels: int, *, linear: bool) -> float:
    first = np.asarray(first, dtype=np.int8)
    second = np.asarray(second, dtype=np.int8)
    if first.shape != second.shape or first.size == 0:
        raise ValueError("Kappa inputs must be nonempty arrays with identical shapes")

    confusion = np.zeros((n_levels, n_levels), dtype=np.float64)
    np.add.at(confusion, (first, second), 1.0)
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / first.size
    if linear:
        positions = np.arange(n_levels, dtype=np.float64)
        weights = np.abs(positions[:, None] - positions[None, :]) / (n_levels - 1)
    else:
        weights = np.ones((n_levels, n_levels), dtype=np.float64) - np.eye(n_levels)

    observed_disagreement = float(np.sum(weights * confusion))
    expected_disagreement = float(np.sum(weights * expected))
    if np.isclose(expected_disagreement, 0.0):
        return float("nan")
    return 1.0 - observed_disagreement / expected_disagreement


def metrics(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.int8)
    second = np.asarray(second, dtype=np.int8)
    first_binary = (first > 0).astype(np.int8)
    second_binary = (second > 0).astype(np.int8)
    return np.asarray(
        [
            np.mean(first == second),
            cohen_kappa(first, second, 3, linear=False),
            cohen_kappa(first, second, 3, linear=True),
            np.mean(first_binary == second_binary),
            cohen_kappa(first_binary, second_binary, 2, linear=False),
        ],
        dtype=np.float64,
    )


def metric_summary(point: np.ndarray, bootstrap: np.ndarray) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for metric_index, metric_name in enumerate(METRIC_NAMES):
        values = bootstrap[:, metric_index]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise AssertionError(f"No finite bootstrap estimates for {metric_name}")
        result[metric_name] = float(point[metric_index])
        result[f"{metric_name}_ci_low"] = float(np.percentile(finite, 2.5))
        result[f"{metric_name}_ci_high"] = float(np.percentile(finite, 97.5))
        result[f"{metric_name}_bootstrap_valid_n"] = int(finite.size)
    return result


def calculate_agreement(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoded: dict[str, np.ndarray] = {}
    for source in ("a1", "a2", "ai"):
        encoded[source] = np.column_stack(
            [merged[f"{source}_{aspect}"].map(LEVEL_TO_INT).to_numpy(dtype=np.int8) for aspect in ASPECTS]
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(0, len(merged), size=(BOOTSTRAP_REPLICATES, len(merged)))
    overall_rows: list[dict[str, object]] = []
    aspect_rows: list[dict[str, object]] = []

    for comparison, first_source, second_source in COMPARISONS:
        first = encoded[first_source]
        second = encoded[second_source]
        point_by_aspect = np.vstack([metrics(first[:, index], second[:, index]) for index in range(len(ASPECTS))])
        point_macro = np.mean(point_by_aspect, axis=0)
        point_micro = metrics(first.reshape(-1), second.reshape(-1))

        bootstrap_by_aspect = np.empty((BOOTSTRAP_REPLICATES, len(ASPECTS), len(METRIC_NAMES)))
        bootstrap_micro = np.empty((BOOTSTRAP_REPLICATES, len(METRIC_NAMES)))
        for replicate, indices in enumerate(bootstrap_indices):
            first_sample = first[indices]
            second_sample = second[indices]
            for aspect_index in range(len(ASPECTS)):
                bootstrap_by_aspect[replicate, aspect_index] = metrics(
                    first_sample[:, aspect_index], second_sample[:, aspect_index]
                )
            bootstrap_micro[replicate] = metrics(first_sample.reshape(-1), second_sample.reshape(-1))
        bootstrap_macro = np.mean(bootstrap_by_aspect, axis=1)

        for aspect_index, aspect in enumerate(ASPECTS):
            row: dict[str, object] = {
                "comparison": comparison,
                "aspect": ASPECT_DISPLAY[aspect],
                "n_items": int(len(merged)),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
            row.update(metric_summary(point_by_aspect[aspect_index], bootstrap_by_aspect[:, aspect_index]))
            aspect_rows.append(row)

        for aggregation, point, bootstrap in (
            ("macro_across_six_aspects", point_macro, bootstrap_macro),
            ("micro_across_1800_cells", point_micro, bootstrap_micro),
        ):
            row = {
                "comparison": comparison,
                "aggregation": aggregation,
                "n_items": int(len(merged)),
                "n_aspects": int(len(ASPECTS)),
                "n_cells": int(len(merged) * len(ASPECTS)),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
            row.update(metric_summary(point, bootstrap))
            overall_rows.append(row)

    return pd.DataFrame(overall_rows), pd.DataFrame(aspect_rows)


def assert_acceptance_values(overall: pd.DataFrame) -> dict[str, object]:
    expected_macro = {
        "Annotator 1 vs. Annotator 2": (0.77166667, 0.48746057, 0.55652075, 0.81722222, 0.55451895),
        "Annotator 1 vs. AI aspects": (0.75222222, 0.49689069, 0.55762968, 0.81666667, 0.58846506),
        "Annotator 2 vs. AI aspects": (0.72555556, 0.40927647, 0.49071026, 0.79833333, 0.51235876),
    }
    expected_exact_counts = {
        "Annotator 1 vs. Annotator 2": (1389, 1471),
        "Annotator 1 vs. AI aspects": (1354, 1470),
        "Annotator 2 vs. AI aspects": (1306, 1437),
    }

    macro = overall.loc[overall["aggregation"].eq("macro_across_six_aspects")].set_index("comparison")
    for comparison, expected in expected_macro.items():
        observed = macro.loc[comparison, list(METRIC_NAMES)].to_numpy(dtype=float)
        if not np.allclose(observed, np.asarray(expected), rtol=0.0, atol=5e-8):
            raise AssertionError(f"Acceptance metrics changed for {comparison}: {observed}")
        exact_count = int(round(float(observed[0]) * 1800))
        binary_count = int(round(float(observed[3]) * 1800))
        if (exact_count, binary_count) != expected_exact_counts[comparison]:
            raise AssertionError(f"Acceptance counts changed for {comparison}")

    return {
        "status": "passed",
        "tolerance": 5e-8,
        "expected_macro_metric_order": list(METRIC_NAMES),
        "expected_macro_values": {key: list(value) for key, value in expected_macro.items()},
        "expected_exact_cell_counts": {
            key: {"three_level": value[0], "binary": value[1]}
            for key, value in expected_exact_counts.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-matrix", type=Path, default=DEFAULT_AUDIT_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged, validation = read_inputs(args.audit_matrix)
    overall, by_aspect = calculate_agreement(merged)
    acceptance = assert_acceptance_values(overall)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall_output_path = args.output_dir / "human_validation_aspect_agreement_kappa_overall.csv"
    by_aspect_output_path = args.output_dir / "human_validation_aspect_agreement_kappa_by_aspect.csv"
    manifest_output_path = args.output_dir / "human_validation_aspect_agreement_kappa_manifest.json"
    overall.to_csv(overall_output_path, index=False, float_format="%.6f")
    by_aspect.to_csv(by_aspect_output_path, index=False, float_format="%.6f")

    manifest = {
        "analysis": "Human agreement audit: chance-corrected aspect agreement",
        "input_files": {
            args.audit_matrix.name: sha256(args.audit_matrix),
        },
        "validation": validation,
        "comparisons": [comparison for comparison, _, _ in COMPARISONS],
        "aspects": [ASPECT_DISPLAY[aspect] for aspect in ASPECTS],
        "definitions": {
            "three_level_scale": {"none": 0, "weak": 1, "clear": 2},
            "binary_presence": "none=0; weak or clear=1",
            "three_level_exact_agreement": "identical NONE/WEAK/CLEAR ratings",
            "three_level_nominal_kappa": "unweighted Cohen's kappa on three levels",
            "three_level_linear_weighted_kappa": "linear-weighted Cohen's kappa on the ordered three-level scale",
            "binary_nominal_kappa": "unweighted Cohen's kappa after the binary presence mapping",
            "macro": "arithmetic mean of the six aspect-specific estimates",
            "micro": "estimate after pooling all 300 x 6 aspect cells",
        },
        "bootstrap": {
            "unit": "sampled post",
            "scheme": "sample 300 posts with replacement and retain all six aspect cells per selected post",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "2.5th and 97.5th percentile item-cluster bootstrap interval",
        },
        "acceptance_assertions": acceptance,
        "outputs": {
            overall_output_path.name: int(len(overall)),
            by_aspect_output_path.name: int(len(by_aspect)),
        },
    }
    manifest_output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {overall_output_path}")
    print(f"Wrote {by_aspect_output_path}")
    print(f"Wrote {manifest_output_path}")


if __name__ == "__main__":
    main()

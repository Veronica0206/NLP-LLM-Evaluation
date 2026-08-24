#!/usr/bin/env python3
"""Reproduce the multi-LLM protocol-sensitivity analyses.

This script is analysis-only. It reconstructs the 100-item entropy-stratified
sample from the current 53,043-row corpus, validates the 21,600 crossed outputs,
and exports sample-composition, factor-level, entropy, aspect-co-occurrence,
three-level aspect-strength, and binary aspect-presence summaries. It does not
call any model API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.metrics import cohen_kappa_score


PAPER_LABEL_ORDER = [
    "NORMAL",
    "DEPRESSION",
    "SUICIDAL",
    "ANXIETY",
    "STRESS",
    "BIPOLAR",
    "PERSONALITY_DISORDER",
]
ARGMAX_TIE_ORDER = [
    "NORMAL",
    "ANXIETY",
    "DEPRESSION",
    "SUICIDAL",
    "STRESS",
    "BIPOLAR",
    "PERSONALITY_DISORDER",
]
SCREENING_PROB_COLS = [
    "u_p_normal",
    "u_p_depression",
    "u_p_anxiety",
    "u_p_suicidal",
    "u_p_stress",
    "u_p_bipolar",
    "u_p_personality_disorder",
]
OUTPUT_PROB_COLS = [f"prob_{label.lower()}" for label in ARGMAX_TIE_ORDER]
ASPECT_COLS = [
    "depression_present",
    "anxiety_present",
    "suicidal_present",
    "stress_present",
    "bipolar_present",
    "personality_disorder_present",
]
RAW_ASPECT_NAMES = [column.removesuffix("_present").upper() for column in ASPECT_COLS]
VALID_STRENGTHS = {"none", "weak", "clear"}
MODEL_ORDER = [
    "gpt-4o-mini",
    "gemini-2.5-flash",
    "llama-70b",
    "mistral-small",
]
PROMPT_ORDER = ["minimal", "rubric", "cot"]
TEMPERATURE_ORDER = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEED_ORDER = [1001, 2001, 3001]
RARE_LABELS = {"STRESS", "BIPOLAR", "PERSONALITY_DISORDER"}
RANDOM_STATE = 42
SAMPLE_SIZE = 100
BOOTSTRAP_SEED = 2025
STRENGTH_ORDER = ["none", "weak", "clear"]
STRENGTH_TO_CODE = {label: index for index, label in enumerate(STRENGTH_ORDER)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_label(value: object) -> str:
    return str(value).strip().upper().replace(" ", "_")


def shannon_entropy(matrix: np.ndarray, log_base: str = "e") -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    safe = np.where(matrix > 0, matrix, 1.0)
    if log_base == "2":
        logs = np.log2(safe)
    else:
        logs = np.log(safe)
    return -(np.where(matrix > 0, matrix * logs, 0.0)).sum(axis=1)


def coerce_bool_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if not pd.api.types.is_bool_dtype(frame[column]):
            mapped = frame[column].astype(str).str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )
            if mapped.isna().any():
                raise ValueError(f"Cannot coerce every value in {column} to boolean")
            frame[column] = mapped.astype(bool)
    return frame


def validate_raw_json_aspects(frame: pd.DataFrame) -> dict[str, int | None]:
    """Audit archived strength/cue fields before the agreement analyses."""
    if "raw_json" not in frame:
        return {
            "raw_json_parse_failures": None,
            "raw_json_aspect_schema_failures": None,
            "raw_json_invalid_strength_values": None,
            "raw_json_valid_strength_present_mismatches": None,
            "raw_json_flattened_present_mismatches": None,
        }

    parse_failures = 0
    aspect_schema_failures = 0
    invalid_strength_values = 0
    valid_strength_present_mismatches = 0
    flattened_present_mismatches = 0

    for row in frame.itertuples(index=False):
        try:
            payload = json.loads(row.raw_json)
        except (TypeError, json.JSONDecodeError):
            parse_failures += 1
            continue
        aspects = payload.get("aspects")
        if not isinstance(aspects, dict) or set(aspects) != set(RAW_ASPECT_NAMES):
            aspect_schema_failures += 1
            continue
        for aspect_name, flat_column in zip(RAW_ASPECT_NAMES, ASPECT_COLS):
            aspect = aspects.get(aspect_name)
            if not isinstance(aspect, dict) or not {"present", "strength", "cues"}.issubset(aspect):
                aspect_schema_failures += 1
                continue
            present = aspect["present"]
            if not isinstance(present, bool):
                aspect_schema_failures += 1
                continue
            strength = str(aspect["strength"]).strip().lower()
            if strength not in VALID_STRENGTHS:
                invalid_strength_values += 1
            elif present != (strength != "none"):
                valid_strength_present_mismatches += 1
            if present != bool(getattr(row, flat_column)):
                flattened_present_mismatches += 1

    return {
        "raw_json_parse_failures": parse_failures,
        "raw_json_aspect_schema_failures": aspect_schema_failures,
        "raw_json_invalid_strength_values": invalid_strength_values,
        "raw_json_valid_strength_present_mismatches": valid_strength_present_mismatches,
        "raw_json_flattened_present_mismatches": flattened_present_mismatches,
    }


def extract_raw_aspect_long(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract the six raw aspect objects from every crossed-design record."""
    if "raw_json" not in frame:
        raise ValueError("Multi-LLM output has no raw_json column")

    rows: list[dict[str, object]] = []
    keys = ["item_id", "prompt_type", "temperature", "seed", "evaluator"]
    for record in frame.itertuples(index=False):
        try:
            payload = json.loads(record.raw_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot parse raw_json for {record.item_id}") from exc
        aspects = payload.get("aspects")
        if not isinstance(aspects, dict) or set(aspects) != set(RAW_ASPECT_NAMES):
            raise ValueError(f"Invalid aspect schema for {record.item_id}")
        base = {key: getattr(record, key) for key in keys}
        for aspect_name, flat_column in zip(RAW_ASPECT_NAMES, ASPECT_COLS):
            aspect = aspects.get(aspect_name)
            if not isinstance(aspect, dict) or not {"present", "strength", "cues"}.issubset(aspect):
                raise ValueError(f"Invalid {aspect_name} object for {record.item_id}")
            present = aspect["present"]
            if not isinstance(present, bool):
                raise ValueError(f"Non-boolean {aspect_name}.present for {record.item_id}")
            raw_value = aspect["strength"]
            strength_raw = "" if raw_value is None else str(raw_value).strip().lower()
            strength_valid = strength_raw in STRENGTH_TO_CODE
            strength_code = STRENGTH_TO_CODE.get(strength_raw, -1)
            flattened_present = bool(getattr(record, flat_column))
            rows.append(
                {
                    **base,
                    "aspect": aspect_name,
                    "strength_raw": strength_raw,
                    "strength_code": strength_code,
                    "strength_valid": strength_valid,
                    "present_raw": present,
                    "flattened_present": flattened_present,
                    "schema_consistent": bool(
                        strength_valid and present == (strength_code > 0)
                    ),
                }
            )

    result = pd.DataFrame(rows)
    expected = len(frame) * len(RAW_ASPECT_NAMES)
    if len(result) != expected:
        raise ValueError(f"Expected {expected:,} raw aspect entries, found {len(result):,}")
    if (result["present_raw"] != result["flattened_present"]).any():
        raise ValueError("Raw and flattened aspect-presence fields disagree")
    return result


def contingency_metrics(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact, nominal-kappa, and linear-kappa arrays from count tables."""
    counts = np.asarray(counts, dtype=float)
    n_levels = counts.shape[-1]
    totals = counts.sum(axis=(-2, -1))
    diagonal = np.trace(counts, axis1=-2, axis2=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        exact = diagonal / totals
        row_margins = counts.sum(axis=-1)
        column_margins = counts.sum(axis=-2)
        expected_agreement = (row_margins * column_margins).sum(axis=-1) / totals**2
        nominal = (exact - expected_agreement) / (1.0 - expected_agreement)

        disagreement_weights = (
            np.abs(np.arange(n_levels)[:, None] - np.arange(n_levels)[None, :])
            / max(n_levels - 1, 1)
        )
        observed_disagreement = (
            counts * disagreement_weights
        ).sum(axis=(-2, -1)) / totals
        expected_tables = (
            row_margins[..., :, None] * column_margins[..., None, :] / totals[..., None, None]
        )
        expected_disagreement = (
            expected_tables * disagreement_weights
        ).sum(axis=(-2, -1)) / totals
        linear = 1.0 - observed_disagreement / expected_disagreement

    exact = np.where(totals > 0, exact, np.nan)
    nominal = np.where((totals > 0) & (np.abs(1.0 - expected_agreement) > 1e-15), nominal, np.nan)
    linear = np.where((totals > 0) & (expected_disagreement > 1e-15), linear, np.nan)
    return exact, nominal, linear


def percentile_ci(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan, 0
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high), int(len(finite))


def build_aspect_item_contingencies(
    raw_aspects: pd.DataFrame,
    model_pairs: list[tuple[str, str]],
) -> tuple[dict[str, np.ndarray], list[dict[str, str]], dict[str, object]]:
    """Build item-cluster contingency counts for all pair-by-aspect cells."""
    item_ids = sorted(raw_aspects["item_id"].unique())
    item_lookup = {item_id: index for index, item_id in enumerate(item_ids)}
    keys = ["item_id", "prompt_type", "temperature", "seed", "aspect"]
    cell_metadata: list[dict[str, str]] = []
    raw_counts: list[np.ndarray] = []
    schema_counts: list[np.ndarray] = []
    binary_counts: list[np.ndarray] = []
    both_present_counts: list[np.ndarray] = []

    def add_counts(
        item_index: np.ndarray,
        value_a: np.ndarray,
        value_b: np.ndarray,
        mask: np.ndarray,
        n_levels: int,
    ) -> np.ndarray:
        counts = np.zeros((len(item_ids), n_levels, n_levels), dtype=np.int32)
        np.add.at(
            counts,
            (item_index[mask], value_a[mask], value_b[mask]),
            1,
        )
        return counts

    for model_a, model_b in model_pairs:
        columns = keys + [
            "strength_code",
            "strength_valid",
            "present_raw",
            "schema_consistent",
        ]
        left = raw_aspects.loc[raw_aspects["evaluator"] == model_a, columns].rename(
            columns={
                "strength_code": "strength_code_a",
                "strength_valid": "strength_valid_a",
                "present_raw": "present_raw_a",
                "schema_consistent": "schema_consistent_a",
            }
        )
        right = raw_aspects.loc[raw_aspects["evaluator"] == model_b, columns].rename(
            columns={
                "strength_code": "strength_code_b",
                "strength_valid": "strength_valid_b",
                "present_raw": "present_raw_b",
                "schema_consistent": "schema_consistent_b",
            }
        )
        paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
        for aspect in RAW_ASPECT_NAMES:
            cell = paired.loc[paired["aspect"] == aspect].sort_values(
                ["item_id", "prompt_type", "temperature", "seed"]
            )
            if len(cell) != SAMPLE_SIZE * len(PROMPT_ORDER) * len(TEMPERATURE_ORDER) * len(SEED_ORDER):
                raise ValueError(
                    f"Unexpected matched cell size for {model_a}, {model_b}, {aspect}: {len(cell)}"
                )
            item_index = cell["item_id"].map(item_lookup).to_numpy(dtype=int)
            strength_a = cell["strength_code_a"].to_numpy(dtype=int)
            strength_b = cell["strength_code_b"].to_numpy(dtype=int)
            valid = (
                cell["strength_valid_a"].to_numpy(dtype=bool)
                & cell["strength_valid_b"].to_numpy(dtype=bool)
            )
            consistent = (
                cell["schema_consistent_a"].to_numpy(dtype=bool)
                & cell["schema_consistent_b"].to_numpy(dtype=bool)
            )
            present_a = cell["present_raw_a"].to_numpy(dtype=bool)
            present_b = cell["present_raw_b"].to_numpy(dtype=bool)
            both_present = consistent & present_a & present_b

            raw_counts.append(add_counts(item_index, strength_a, strength_b, valid, 3))
            schema_counts.append(add_counts(item_index, strength_a, strength_b, consistent, 3))
            binary_counts.append(
                add_counts(
                    item_index,
                    present_a.astype(int),
                    present_b.astype(int),
                    np.ones(len(cell), dtype=bool),
                    2,
                )
            )
            # In the schema-consistent both-present subset, strength codes are 1 or 2.
            both_present_counts.append(
                add_counts(
                    item_index,
                    strength_a - 1,
                    strength_b - 1,
                    both_present,
                    2,
                )
            )
            cell_metadata.append(
                {"model_a": model_a, "model_b": model_b, "aspect": aspect}
            )

    arrays = {
        "strength_raw_valid": np.stack(raw_counts),
        "strength_schema_consistent": np.stack(schema_counts),
        "binary_raw_present": np.stack(binary_counts),
        "both_present_schema_consistent": np.stack(both_present_counts),
    }
    invalid_counts = (
        raw_aspects.loc[~raw_aspects["strength_valid"], "strength_raw"]
        .value_counts(dropna=False)
        .sort_index()
    )
    mismatch = raw_aspects["strength_valid"] & ~raw_aspects["schema_consistent"]
    mismatch_direction = {
        "present_false_with_weak_or_clear": int(
            (mismatch & ~raw_aspects["present_raw"] & (raw_aspects["strength_code"] > 0)).sum()
        ),
        "present_true_with_none": int(
            (mismatch & raw_aspects["present_raw"] & (raw_aspects["strength_code"] == 0)).sum()
        ),
    }
    overlap_index = ["item_id", "prompt_type", "temperature", "seed", "aspect"]
    overlap = raw_aspects.assign(
        invalid_entry=~raw_aspects["strength_valid"],
        mismatched_entry=mismatch,
    ).pivot(
        index=overlap_index,
        columns="evaluator",
        values=["invalid_entry", "mismatched_entry"],
    )
    pairs_with_two_mismatches = 0
    mismatch_with_invalid = 0
    for model_a, model_b in model_pairs:
        mismatch_a = overlap[("mismatched_entry", model_a)].to_numpy(dtype=bool)
        mismatch_b = overlap[("mismatched_entry", model_b)].to_numpy(dtype=bool)
        invalid_a = overlap[("invalid_entry", model_a)].to_numpy(dtype=bool)
        invalid_b = overlap[("invalid_entry", model_b)].to_numpy(dtype=bool)
        pairs_with_two_mismatches += int((mismatch_a & mismatch_b).sum())
        mismatch_with_invalid += int(
            ((mismatch_a & invalid_b) | (mismatch_b & invalid_a)).sum()
        )
    qc = {
        "total_strength_entries": int(len(raw_aspects)),
        "valid_strength_entries": int(raw_aspects["strength_valid"].sum()),
        "invalid_strength_entries": int((~raw_aspects["strength_valid"]).sum()),
        "schema_consistent_strength_entries": int(raw_aspects["schema_consistent"].sum()),
        "valid_strength_present_mismatch_entries": int(mismatch.sum()),
        "total_pair_comparisons": int(arrays["binary_raw_present"].sum()),
        "raw_valid_pair_comparisons": int(arrays["strength_raw_valid"].sum()),
        "schema_consistent_pair_comparisons": int(
            arrays["strength_schema_consistent"].sum()
        ),
        "invalid_strength_pair_exclusions": int(
            arrays["binary_raw_present"].sum() - arrays["strength_raw_valid"].sum()
        ),
        "additional_schema_mismatch_pair_exclusions": int(
            arrays["strength_raw_valid"].sum()
            - arrays["strength_schema_consistent"].sum()
        ),
        "pair_cells_with_two_mismatched_entries": pairs_with_two_mismatches,
        "mismatched_entries_paired_with_invalid_entries": mismatch_with_invalid,
        "invalid_strength_literal_counts": {
            str(key): int(value) for key, value in invalid_counts.items()
        },
        "strength_present_mismatch_direction_counts": mismatch_direction,
    }
    return arrays, cell_metadata, qc


def aspect_agreement_outputs(
    raw_aspects: pd.DataFrame,
    model_pairs: list[tuple[str, str]],
    bootstrap_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Calculate three-level, binary, and schema-sensitivity agreement outputs."""
    arrays, metadata, qc = build_aspect_item_contingencies(raw_aspects, model_pairs)
    bootstrap_weights = np.zeros(
        (len(bootstrap_indices), SAMPLE_SIZE), dtype=np.int16
    )
    for row, indices in enumerate(bootstrap_indices):
        bootstrap_weights[row] = np.bincount(indices, minlength=SAMPLE_SIZE)

    variants = [
        ("strength_3level", "raw_valid", "strength_raw_valid"),
        ("strength_3level", "schema_consistent", "strength_schema_consistent"),
        ("binary_presence", "raw_present", "binary_raw_present"),
    ]
    cached: dict[str, dict[str, np.ndarray]] = {}
    for _, _, key in variants:
        counts = arrays[key]
        point_cell_counts = counts.sum(axis=1)
        boot_cell_counts = np.einsum(
            "bi,cijk->bcjk", bootstrap_weights, counts, optimize=True
        )
        point_exact, point_nominal, point_linear = contingency_metrics(point_cell_counts)
        boot_exact, boot_nominal, boot_linear = contingency_metrics(boot_cell_counts)
        cached[key] = {
            "counts": counts,
            "point_cell_counts": point_cell_counts,
            "boot_cell_counts": boot_cell_counts,
            "point_exact": point_exact,
            "point_nominal": point_nominal,
            "point_linear": point_linear,
            "boot_exact": boot_exact,
            "boot_nominal": boot_nominal,
            "boot_linear": boot_linear,
        }

    pair_indices: dict[tuple[str, str], list[int]] = {}
    aspect_indices: dict[str, list[int]] = {}
    for index, cell in enumerate(metadata):
        pair_indices.setdefault((cell["model_a"], cell["model_b"]), []).append(index)
        aspect_indices.setdefault(cell["aspect"], []).append(index)

    scopes: list[dict[str, object]] = []
    for index, cell in enumerate(metadata):
        scopes.append(
            {
                "scope": "pair_aspect",
                "indices": [index],
                "model_a": cell["model_a"],
                "model_b": cell["model_b"],
                "aspect": cell["aspect"],
                "aggregation_rule": "pooled_contingency",
            }
        )
    for pair in model_pairs:
        scopes.append(
            {
                "scope": "model_pair",
                "indices": pair_indices[pair],
                "model_a": pair[0],
                "model_b": pair[1],
                "aspect": "",
                "aggregation_rule": "pooled_contingency",
            }
        )
    for aspect in RAW_ASPECT_NAMES:
        scopes.append(
            {
                "scope": "aspect",
                "indices": aspect_indices[aspect],
                "model_a": "",
                "model_b": "",
                "aspect": aspect,
                "aggregation_rule": "pooled_contingency",
            }
        )
    scopes.append(
        {
            "scope": "overall_micro",
            "indices": list(range(len(metadata))),
            "model_a": "",
            "model_b": "",
            "aspect": "",
            "aggregation_rule": "pooled_contingency",
        }
    )
    scopes.append(
        {
            "scope": "overall_macro_pair_aspect",
            "indices": list(range(len(metadata))),
            "model_a": "",
            "model_b": "",
            "aspect": "",
            "aggregation_rule": "equal_mean_36_pair_aspect_cells",
        }
    )

    def scope_metrics(key: str, scope: dict[str, object]) -> dict[str, object]:
        indices = np.asarray(scope["indices"], dtype=int)
        data = cached[key]
        if scope["scope"] == "overall_macro_pair_aspect":
            point = (
                float(np.nanmean(data["point_exact"][indices])),
                float(np.nanmean(data["point_nominal"][indices])),
                float(np.nanmean(data["point_linear"][indices])),
            )
            boot = (
                np.nanmean(data["boot_exact"][:, indices], axis=1),
                np.nanmean(data["boot_nominal"][:, indices], axis=1),
                np.nanmean(data["boot_linear"][:, indices], axis=1),
            )
        else:
            point_counts = data["point_cell_counts"][indices].sum(axis=0)
            boot_counts = data["boot_cell_counts"][:, indices].sum(axis=1)
            point_arrays = contingency_metrics(point_counts)
            boot = contingency_metrics(boot_counts)
            point = tuple(float(np.asarray(value)) for value in point_arrays)
        exact_ci = percentile_ci(boot[0])
        nominal_ci = percentile_ci(boot[1])
        linear_ci = percentile_ci(boot[2])
        used_per_item = data["counts"][indices].sum(axis=(0, 2, 3))
        return {
            "n_used": int(data["counts"][indices].sum()),
            "n_items": int((used_per_item > 0).sum()),
            "exact": point[0],
            "exact_low": exact_ci[0],
            "exact_high": exact_ci[1],
            "nominal": point[1],
            "nominal_low": nominal_ci[0],
            "nominal_high": nominal_ci[1],
            "linear": point[2],
            "linear_low": linear_ci[0],
            "linear_high": linear_ci[1],
            "valid_replicates": min(
                count
                for count in [exact_ci[2], nominal_ci[2], linear_ci[2]]
                if count > 0
            ),
            "boot_exact": boot[0],
            "boot_nominal": boot[1],
            "boot_linear": boot[2],
        }

    rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    for scope in scopes:
        indices = np.asarray(scope["indices"], dtype=int)
        total_matched = int(arrays["binary_raw_present"][indices].sum())
        raw_metrics = scope_metrics("strength_raw_valid", scope)
        schema_metrics = scope_metrics("strength_schema_consistent", scope)
        for representation, inclusion, key in variants:
            metrics = scope_metrics(key, scope)
            if representation == "strength_3level":
                invalid_exclusions = total_matched - raw_metrics["n_used"]
                mismatch_exclusions = (
                    0
                    if inclusion == "raw_valid"
                    else raw_metrics["n_used"] - schema_metrics["n_used"]
                )
                linear_values = (
                    metrics["linear"],
                    metrics["linear_low"],
                    metrics["linear_high"],
                )
            else:
                invalid_exclusions = 0
                mismatch_exclusions = 0
                linear_values = (np.nan, np.nan, np.nan)
            rows.append(
                {
                    "scope": scope["scope"],
                    "representation": representation,
                    "inclusion_definition": inclusion,
                    "aggregation_rule": scope["aggregation_rule"],
                    "model_a": scope["model_a"],
                    "model_b": scope["model_b"],
                    "aspect": scope["aspect"],
                    "n_pair_aspect_cells": int(len(indices)),
                    "n_items": metrics["n_items"],
                    "n_conditions_per_item": 54,
                    "n_total_matched": total_matched,
                    "n_used": metrics["n_used"],
                    "n_excluded_invalid_strength": int(invalid_exclusions),
                    "n_excluded_strength_present_mismatch": int(mismatch_exclusions),
                    "exact_agreement": metrics["exact"],
                    "exact_agreement_ci_low": metrics["exact_low"],
                    "exact_agreement_ci_high": metrics["exact_high"],
                    "nominal_kappa": metrics["nominal"],
                    "nominal_kappa_ci_low": metrics["nominal_low"],
                    "nominal_kappa_ci_high": metrics["nominal_high"],
                    "linear_weighted_kappa": linear_values[0],
                    "linear_weighted_kappa_ci_low": linear_values[1],
                    "linear_weighted_kappa_ci_high": linear_values[2],
                    "bootstrap_resamples": int(len(bootstrap_indices)),
                    "bootstrap_valid_replicates": metrics["valid_replicates"],
                }
            )

        deltas = {
            "exact": schema_metrics["boot_exact"] - raw_metrics["boot_exact"],
            "nominal": schema_metrics["boot_nominal"] - raw_metrics["boot_nominal"],
            "linear": schema_metrics["boot_linear"] - raw_metrics["boot_linear"],
        }
        exact_delta_ci = percentile_ci(deltas["exact"])
        nominal_delta_ci = percentile_ci(deltas["nominal"])
        linear_delta_ci = percentile_ci(deltas["linear"])
        sensitivity_rows.append(
            {
                "scope": scope["scope"],
                "aggregation_rule": scope["aggregation_rule"],
                "model_a": scope["model_a"],
                "model_b": scope["model_b"],
                "aspect": scope["aspect"],
                "n_pair_aspect_cells": int(len(indices)),
                "n_total_matched": total_matched,
                "n_raw_valid": raw_metrics["n_used"],
                "n_schema_consistent": schema_metrics["n_used"],
                "n_additionally_excluded": int(
                    raw_metrics["n_used"] - schema_metrics["n_used"]
                ),
                "exact_agreement_raw": raw_metrics["exact"],
                "exact_agreement_schema": schema_metrics["exact"],
                "exact_agreement_delta": schema_metrics["exact"] - raw_metrics["exact"],
                "exact_agreement_delta_ci_low": exact_delta_ci[0],
                "exact_agreement_delta_ci_high": exact_delta_ci[1],
                "nominal_kappa_raw": raw_metrics["nominal"],
                "nominal_kappa_schema": schema_metrics["nominal"],
                "nominal_kappa_delta": schema_metrics["nominal"] - raw_metrics["nominal"],
                "nominal_kappa_delta_ci_low": nominal_delta_ci[0],
                "nominal_kappa_delta_ci_high": nominal_delta_ci[1],
                "linear_weighted_kappa_raw": raw_metrics["linear"],
                "linear_weighted_kappa_schema": schema_metrics["linear"],
                "linear_weighted_kappa_delta": schema_metrics["linear"] - raw_metrics["linear"],
                "linear_weighted_kappa_delta_ci_low": linear_delta_ci[0],
                "linear_weighted_kappa_delta_ci_high": linear_delta_ci[1],
                "bootstrap_resamples": int(len(bootstrap_indices)),
                "bootstrap_valid_replicates": min(
                    count
                    for count in [exact_delta_ci[2], nominal_delta_ci[2], linear_delta_ci[2]]
                    if count > 0
                ),
            }
        )

    both_counts = arrays["both_present_schema_consistent"]
    both_point_cells = both_counts.sum(axis=1)
    both_boot_cells = np.einsum(
        "bi,cijk->bcjk", bootstrap_weights, both_counts, optimize=True
    )
    both_rows: list[dict[str, object]] = []
    for scope in scopes:
        indices = np.asarray(scope["indices"], dtype=int)
        if scope["scope"] == "overall_macro_pair_aspect":
            cell_point_exact = contingency_metrics(both_point_cells)[0]
            cell_boot_exact = contingency_metrics(both_boot_cells)[0]
            exact = float(np.nanmean(cell_point_exact[indices]))
            boot_exact = np.nanmean(cell_boot_exact[:, indices], axis=1)
        else:
            exact = float(
                np.asarray(
                    contingency_metrics(both_point_cells[indices].sum(axis=0))[0]
                )
            )
            boot_exact = contingency_metrics(both_boot_cells[:, indices].sum(axis=1))[0]
        low, high, valid_replicates = percentile_ci(boot_exact)
        used_per_item = both_counts[indices].sum(axis=(0, 2, 3))
        both_rows.append(
            {
                "scope": scope["scope"],
                "aggregation_rule": scope["aggregation_rule"],
                "model_a": scope["model_a"],
                "model_b": scope["model_b"],
                "aspect": scope["aspect"],
                "n_pair_aspect_cells": int(len(indices)),
                "n_items": int((used_per_item > 0).sum()),
                "n_total_matched": int(arrays["binary_raw_present"][indices].sum()),
                "n_both_present_schema_consistent": int(both_counts[indices].sum()),
                "exact_agreement": exact,
                "exact_agreement_ci_low": low,
                "exact_agreement_ci_high": high,
                "bootstrap_resamples": int(len(bootstrap_indices)),
                "bootstrap_valid_replicates": valid_replicates,
            }
        )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(sensitivity_rows),
        pd.DataFrame(both_rows),
        qc,
    )


def reconstruct_sample(full: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    probs = full[SCREENING_PROB_COLS].to_numpy(dtype=float)
    full = full.copy()
    full["sampling_score_sum"] = probs.sum(axis=1)
    if (full["sampling_score_sum"] <= 0).any():
        raise ValueError("At least one screening score vector has non-positive row sum")
    # Historical sampling notebook: applied -sum(s*log2(s)) directly to stored
    # u_p_* scores, without row normalization. Preserve that exact statistic only
    # for reconstructing sample membership and strata.
    full["sampling_score_dispersion_bits"] = shannon_entropy(probs, log_base="2")
    q25, q50, q75 = full["sampling_score_dispersion_bits"].quantile(
        [0.25, 0.50, 0.75]
    ).to_numpy()
    # Manuscript-facing entropy: row-normalize first, then compute Shannon entropy.
    normalized_probs = probs / full["sampling_score_sum"].to_numpy()[:, None]
    full["analysis_entropy_nats"] = shannon_entropy(normalized_probs, log_base="e")
    full["analysis_normalized_entropy"] = full["analysis_entropy_nats"] / np.log(7.0)

    def assign_stratum(entropy: float) -> str:
        if entropy <= q25:
            return "Q1"
        if entropy <= q50:
            return "Q2"
        if entropy <= q75:
            return "Q3"
        return "Q4"

    full["entropy_stratum"] = full["sampling_score_dispersion_bits"].map(assign_stratum)
    stratum_counts = (
        full["entropy_stratum"].value_counts(normalize=True).sort_index() * SAMPLE_SIZE
    ).round().astype(int)
    difference = SAMPLE_SIZE - int(stratum_counts.sum())
    if difference:
        stratum_counts.loc[stratum_counts.idxmax()] += difference

    pieces = []
    for stratum, count in stratum_counts.items():
        pieces.append(
            full.loc[full["entropy_stratum"] == stratum].sample(
                n=int(count), random_state=RANDOM_STATE
            )
        )
    sample = pd.concat(pieces).reset_index(drop=True)
    sample["item_id"] = [f"mh_{index:04d}" for index in range(len(sample))]
    sample["screening_label"] = sample["u_label"].map(canonical_label)
    sample["released_category"] = sample["status"].map(canonical_label)
    sample["released_screening_disagreement"] = (
        sample["released_category"] != sample["screening_label"]
    )
    full["screening_label"] = full["u_label"].map(canonical_label)
    full["released_category"] = full["status"].map(canonical_label)
    full["released_screening_disagreement"] = (
        full["released_category"] != full["screening_label"]
    )
    thresholds = {
        "sampling_raw_score_q25_bits": float(q25),
        "sampling_raw_score_q50_bits": float(q50),
        "sampling_raw_score_q75_bits": float(q75),
    }
    return full, sample, thresholds


def add_distribution_rows(
    rows: list[dict[str, object]],
    section: str,
    levels: list[object],
    full_values: pd.Series,
    sample_values: pd.Series,
) -> None:
    full_counts = full_values.value_counts(dropna=False)
    sample_counts = sample_values.value_counts(dropna=False)
    for level in levels:
        full_n = int(full_counts.get(level, 0))
        sample_n = int(sample_counts.get(level, 0))
        rows.append(
            {
                "section": section,
                "level": str(level),
                "full_n": full_n,
                "full_pct": 100.0 * full_n / len(full_values),
                "sample_n": sample_n,
                "sample_pct": 100.0 * sample_n / len(sample_values),
            }
        )


def sample_characteristics(full: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    add_distribution_rows(
        rows,
        "entropy_stratum",
        ["Q1", "Q2", "Q3", "Q4"],
        full["entropy_stratum"],
        sample["entropy_stratum"],
    )
    add_distribution_rows(
        rows,
        "screening_label",
        PAPER_LABEL_ORDER,
        full["screening_label"],
        sample["screening_label"],
    )
    add_distribution_rows(
        rows,
        "released_category",
        PAPER_LABEL_ORDER,
        full["released_category"],
        sample["released_category"],
    )
    add_distribution_rows(
        rows,
        "released_vs_screening",
        ["agree", "disagree"],
        full["released_screening_disagreement"].map({False: "agree", True: "disagree"}),
        sample["released_screening_disagreement"].map({False: "agree", True: "disagree"}),
    )
    add_distribution_rows(
        rows,
        "rare_screening_label",
        ["not_rare", "rare"],
        full["screening_label"].isin(RARE_LABELS).map({False: "not_rare", True: "rare"}),
        sample["screening_label"].isin(RARE_LABELS).map({False: "not_rare", True: "rare"}),
    )
    add_distribution_rows(
        rows,
        "rare_released_category",
        ["not_rare", "rare"],
        full["released_category"].isin(RARE_LABELS).map({False: "not_rare", True: "rare"}),
        sample["released_category"].isin(RARE_LABELS).map({False: "not_rare", True: "rare"}),
    )
    return pd.DataFrame(rows)


def entropy_summary(full: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_name, frame in [("full_corpus", full), ("audit_sample", sample)]:
        for scale, column in [
            ("sampling_raw_score_sum", "sampling_score_sum"),
            ("sampling_score_dispersion_bits", "sampling_score_dispersion_bits"),
            ("analysis_entropy_nats", "analysis_entropy_nats"),
            ("analysis_normalized_entropy", "analysis_normalized_entropy"),
        ]:
            values = frame[column]
            rows.append(
                {
                    "source": source_name,
                    "scale": scale,
                    "n": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "sd": float(values.std()),
                    "min": float(values.min()),
                    "q25": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q75": float(values.quantile(0.75)),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def validate_and_enrich_outputs(
    outputs: pd.DataFrame, sample: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {
        "item_id",
        "evaluator",
        "prompt_type",
        "temperature",
        "seed",
        "label",
        "text_hash",
        *OUTPUT_PROB_COLS,
        *ASPECT_COLS,
    }
    missing = sorted(required.difference(outputs.columns))
    if missing:
        raise ValueError(f"Multi-LLM output is missing required columns: {missing}")

    outputs = coerce_bool_columns(outputs.copy(), ASPECT_COLS)
    keys = ["item_id", "evaluator", "prompt_type", "temperature", "seed"]
    duplicate_keys = int(outputs.duplicated(keys).sum())
    probability_sums = outputs[OUTPUT_PROB_COLS].sum(axis=1)
    argmax = np.asarray(ARGMAX_TIE_ORDER)[
        np.argmax(outputs[OUTPUT_PROB_COLS].to_numpy(dtype=float), axis=1)
    ]

    item_hashes = sample.set_index("item_id")["statement"].map(
        lambda text: hashlib.md5(str(text).encode()).hexdigest()
    )
    observed_hashes = outputs[["item_id", "text_hash"]].drop_duplicates("item_id").set_index("item_id")
    hash_match = item_hashes.equals(observed_hashes.loc[item_hashes.index, "text_hash"])

    design = {
        "rows": int(len(outputs)),
        "unique_items": int(outputs["item_id"].nunique()),
        "models": sorted(outputs["evaluator"].unique().tolist()),
        "prompts": sorted(outputs["prompt_type"].unique().tolist()),
        "temperatures": sorted(float(x) for x in outputs["temperature"].unique()),
        "seeds": sorted(int(x) for x in outputs["seed"].unique()),
        "duplicate_design_keys": duplicate_keys,
        "probability_sum_min": float(probability_sums.min()),
        "probability_sum_max": float(probability_sums.max()),
        "labels_equal_forced_argmax_n": int((argmax == outputs["label"].to_numpy()).sum()),
        "missing_probability_values": int(outputs[OUTPUT_PROB_COLS].isna().sum().sum()),
        "missing_aspect_values": int(outputs[ASPECT_COLS].isna().sum().sum()),
        "raw_json_missing": int(outputs["raw_json"].isna().sum()) if "raw_json" in outputs else None,
        "reasoning_missing": int(outputs["reasoning"].isna().sum()) if "reasoning" in outputs else None,
        "sample_text_hashes_match": bool(hash_match),
        **validate_raw_json_aspects(outputs),
    }
    if design["rows"] != 21600:
        raise ValueError(f"Expected 21,600 output rows, found {design['rows']}")
    if design["unique_items"] != 100:
        raise ValueError(f"Expected 100 items, found {design['unique_items']}")
    if design["models"] != sorted(MODEL_ORDER):
        raise ValueError(f"Unexpected evaluator set: {design['models']}")
    if design["prompts"] != sorted(PROMPT_ORDER):
        raise ValueError(f"Unexpected prompt set: {design['prompts']}")
    if design["temperatures"] != TEMPERATURE_ORDER:
        raise ValueError(f"Unexpected temperature set: {design['temperatures']}")
    if design["seeds"] != SEED_ORDER:
        raise ValueError(f"Unexpected seed set: {design['seeds']}")
    if duplicate_keys:
        raise ValueError(f"Found {duplicate_keys} duplicate crossed-design keys")
    if not np.allclose(probability_sums, 1.0, atol=1e-9):
        raise ValueError("At least one class-score vector does not sum to one")
    if design["labels_equal_forced_argmax_n"] != len(outputs):
        raise ValueError("At least one hard label differs from the forced score-vector argmax")
    if design["missing_probability_values"] or design["missing_aspect_values"]:
        raise ValueError("Missing score-vector or aspect-present output fields")
    if not hash_match:
        raise ValueError("Crossed-output item hashes do not match the reconstructed sample")

    metadata = sample[
        [
            "item_id",
            "screening_label",
            "released_category",
            "released_screening_disagreement",
            "entropy_stratum",
            "analysis_entropy_nats",
            "analysis_normalized_entropy",
        ]
    ]
    outputs = outputs.merge(metadata, on="item_id", how="left", validate="many_to_one")
    output_probs = outputs[OUTPUT_PROB_COLS].to_numpy(dtype=float)
    outputs["score_entropy_nats"] = shannon_entropy(output_probs, log_base="e")
    outputs["normalized_score_entropy"] = outputs["score_entropy_nats"] / np.log(7.0)
    outputs["aspect_count"] = outputs[ASPECT_COLS].sum(axis=1)
    outputs["multi_aspect"] = outputs["aspect_count"] >= 2
    outputs["screening_label_match"] = outputs["label"] == outputs["screening_label"]
    outputs["released_label_match"] = outputs["label"] == outputs["released_category"]
    return outputs, design


def matched_condition_wide(outputs: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    index = [
        "item_id",
        "screening_label",
        "released_category",
        "entropy_stratum",
        "prompt_type",
        "temperature",
        "seed",
    ]
    wide = outputs.pivot(index=index, columns="evaluator", values="label").reset_index()
    model_pairs = list(combinations(sorted(outputs["evaluator"].unique()), 2))
    agreement_columns = []
    for model_a, model_b in model_pairs:
        column = f"agree__{model_a}__{model_b}"
        wide[column] = wide[model_a] == wide[model_b]
        agreement_columns.append(column)
    wide["matched_cross_llm_pair_agreement"] = wide[agreement_columns].mean(axis=1)
    return wide, model_pairs


def factor_level_summary(outputs: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    factor_columns = {
        "prompt": "prompt_type",
        "temperature": "temperature",
        "seed": "seed",
        "released_category": "released_category",
        "screening_label": "screening_label",
        "entropy_stratum": "entropy_stratum",
    }
    for factor_name, column in factor_columns.items():
        long_summary = outputs.groupby(column, observed=False).agg(
            n_items=("item_id", "nunique"),
            n_outputs=("label", "size"),
            screening_label_concordance=("screening_label_match", "mean"),
            released_label_concordance=("released_label_match", "mean"),
            mean_entropy_nats=("score_entropy_nats", "mean"),
            mean_normalized_entropy=("normalized_score_entropy", "mean"),
            multi_aspect_rate=("multi_aspect", "mean"),
            mean_aspect_count=("aspect_count", "mean"),
        )
        cross_model = wide.groupby(column, observed=False)[
            "matched_cross_llm_pair_agreement"
        ].mean()
        joined = long_summary.join(cross_model)
        for level, values in joined.iterrows():
            rows.append({"factor": factor_name, "level": str(level), **values.to_dict()})

    model_pair_rates: dict[str, list[float]] = {model: [] for model in MODEL_ORDER}
    _, model_pairs = matched_condition_wide(outputs)
    key = ["item_id", "prompt_type", "temperature", "seed"]
    model_wide = outputs.pivot(index=key, columns="evaluator", values="label")
    for model_a, model_b in model_pairs:
        rate = float((model_wide[model_a] == model_wide[model_b]).mean())
        model_pair_rates[model_a].append(rate)
        model_pair_rates[model_b].append(rate)

    model_summary = outputs.groupby("evaluator").agg(
        n_items=("item_id", "nunique"),
        n_outputs=("label", "size"),
        screening_label_concordance=("screening_label_match", "mean"),
        released_label_concordance=("released_label_match", "mean"),
        mean_entropy_nats=("score_entropy_nats", "mean"),
        mean_normalized_entropy=("normalized_score_entropy", "mean"),
        multi_aspect_rate=("multi_aspect", "mean"),
        mean_aspect_count=("aspect_count", "mean"),
    )
    for model, values in model_summary.iterrows():
        rows.append(
            {
                "factor": "model",
                "level": model,
                **values.to_dict(),
                "matched_cross_llm_pair_agreement": float(np.mean(model_pair_rates[model])),
            }
        )
    result = pd.DataFrame(rows)
    result["n_items"] = result["n_items"].astype(int)
    result["n_outputs"] = result["n_outputs"].astype(int)
    factor_rank = {
        "model": 0,
        "prompt": 1,
        "temperature": 2,
        "seed": 3,
        "released_category": 4,
        "screening_label": 5,
        "entropy_stratum": 6,
    }
    result["_factor_rank"] = result["factor"].map(factor_rank)
    return (
        result.sort_values(["_factor_rank", "level"])
        .drop(columns="_factor_rank")
        .reset_index(drop=True)
    )


def model_level_summary(outputs: pd.DataFrame, hard_pairs: pd.DataFrame) -> pd.DataFrame:
    modal = (
        outputs.groupby(["evaluator", "item_id"])["label"]
        .agg(lambda labels: labels.value_counts().iloc[0] / len(labels))
        .groupby("evaluator")
        .agg([("median_modal_share", "median"), ("mean_modal_share", "mean")])
    )
    base = outputs.groupby("evaluator").agg(
        n_items=("item_id", "nunique"),
        n_outputs=("label", "size"),
        screening_label_concordance=("screening_label_match", "mean"),
        released_label_concordance=("released_label_match", "mean"),
        mean_entropy_nats=("score_entropy_nats", "mean"),
        mean_normalized_entropy=("normalized_score_entropy", "mean"),
        multi_aspect_rate=("multi_aspect", "mean"),
        mean_aspect_count=("aspect_count", "mean"),
    )
    all_condition_pairs = hard_pairs.loc[hard_pairs["scope"] == "all_matched_conditions"]
    pair_values: dict[str, list[float]] = {model: [] for model in MODEL_ORDER}
    for row in all_condition_pairs.itertuples(index=False):
        pair_values[row.model_a].append(float(row.agreement))
        pair_values[row.model_b].append(float(row.agreement))
    base["mean_pairwise_agreement"] = [np.mean(pair_values[model]) for model in base.index]
    base["minimum_pairwise_agreement"] = [np.min(pair_values[model]) for model in base.index]
    base["maximum_pairwise_agreement"] = [np.max(pair_values[model]) for model in base.index]
    return base.join(modal).reset_index().rename(columns={"evaluator": "model"})


def aspect_prevalence_by_model(outputs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, frame in outputs.groupby("evaluator"):
        for aspect in ASPECT_COLS:
            rows.append(
                {
                    "model": model,
                    "aspect": aspect.removesuffix("_present"),
                    "n_outputs": int(len(frame)),
                    "present_rate": float(frame[aspect].mean()),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    estimates = values[indices].mean(axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def hard_label_pairwise(
    outputs: pd.DataFrame, model_pairs: list[tuple[str, str]], bootstrap_indices: np.ndarray
) -> pd.DataFrame:
    rows = []
    all_index = ["item_id", "prompt_type", "temperature", "seed"]
    all_wide = outputs.pivot(index=all_index, columns="evaluator", values="label").reset_index()
    fixed = outputs.loc[
        (outputs["prompt_type"] == "rubric")
        & (outputs["temperature"] == 0.0)
        & (outputs["seed"] == 1001)
    ].pivot(index="item_id", columns="evaluator", values="label")

    for scope, frame in [("all_matched_conditions", all_wide), ("rubric_t0_seed1001", fixed.reset_index())]:
        for model_a, model_b in model_pairs:
            agreement_indicator = (frame[model_a] == frame[model_b]).astype(float)
            if scope == "all_matched_conditions":
                per_item = agreement_indicator.groupby(frame["item_id"]).mean().sort_index().to_numpy()
            else:
                per_item = agreement_indicator.to_numpy()
            low, high = bootstrap_mean_ci(per_item, bootstrap_indices)
            rows.append(
                {
                    "scope": scope,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_matched_outputs": int(len(frame)),
                    "agreement": float(agreement_indicator.mean()),
                    "agreement_ci_low": low,
                    "agreement_ci_high": high,
                    "cohen_kappa_nominal": float(cohen_kappa_score(frame[model_a], frame[model_b])),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_spearman_ci(
    values_a: np.ndarray, values_b: np.ndarray, bootstrap_indices: np.ndarray
) -> tuple[float, float]:
    estimates = np.array(
        [spearmanr(values_a[index], values_b[index]).statistic for index in bootstrap_indices]
    )
    low, high = np.nanpercentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def pairwise_item_mean_correlations(
    outputs: pd.DataFrame,
    value_column: str,
    measure: str,
    model_pairs: list[tuple[str, str]],
    bootstrap_indices: np.ndarray,
) -> pd.DataFrame:
    per_item = (
        outputs.groupby(["item_id", "evaluator"])[value_column]
        .mean()
        .unstack()
        .sort_index()
    )
    rows = []
    for model_a, model_b in model_pairs:
        values_a = per_item[model_a].to_numpy(dtype=float)
        values_b = per_item[model_b].to_numpy(dtype=float)
        result = spearmanr(values_a, values_b)
        low, high = bootstrap_spearman_ci(values_a, values_b, bootstrap_indices)
        rows.append(
            {
                "measure": measure,
                "aggregation": "item_mean_across_54_conditions",
                "model_a": model_a,
                "model_b": model_b,
                "n_items": int(len(per_item)),
                "spearman_rho": float(result.statistic),
                "p_value": float(result.pvalue),
                "rho_ci_low": low,
                "rho_ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def cooccurrence_robustness(
    outputs: pd.DataFrame,
    model_pairs: list[tuple[str, str]],
    bootstrap_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aspect_pairs = list(combinations(ASPECT_COLS, 2))
    item_ids = sorted(outputs["item_id"].unique())
    per_item: dict[str, np.ndarray] = {}
    rate_rows = []
    for model, model_frame in outputs.groupby("evaluator"):
        item_rows = []
        for item_id, item_frame in model_frame.groupby("item_id"):
            item_rows.append(
                (
                    item_id,
                    np.array(
                        [(item_frame[a] & item_frame[b]).mean() for a, b in aspect_pairs],
                        dtype=float,
                    ),
                )
            )
        item_map = dict(item_rows)
        per_item[model] = np.vstack([item_map[item_id] for item_id in item_ids])
        pooled = per_item[model].mean(axis=0)
        for (aspect_a, aspect_b), rate in zip(aspect_pairs, pooled):
            rate_rows.append(
                {
                    "model": model,
                    "aspect_a": aspect_a.removesuffix("_present"),
                    "aspect_b": aspect_b.removesuffix("_present"),
                    "joint_present_rate": float(rate),
                }
            )

    rows = []
    for model_a, model_b in model_pairs:
        pooled_a = per_item[model_a].mean(axis=0)
        pooled_b = per_item[model_b].mean(axis=0)
        result = spearmanr(pooled_a, pooled_b)
        boot = []
        for index in bootstrap_indices:
            boot_a = per_item[model_a][index].mean(axis=0)
            boot_b = per_item[model_b][index].mean(axis=0)
            boot.append(spearmanr(boot_a, boot_b).statistic)
        low, high = np.nanpercentile(np.asarray(boot), [2.5, 97.5])
        rows.append(
            {
                "measure": "fifteen_aspect_pair_joint_present_rates",
                "aggregation": "pooled_across_100_items_and_54_conditions",
                "model_a": model_a,
                "model_b": model_b,
                "n_aspect_pairs": len(aspect_pairs),
                "spearman_rho": float(result.statistic),
                "p_value": float(result.pvalue),
                "rho_ci_low": float(low),
                "rho_ci_high": float(high),
                "mean_absolute_rate_difference": float(np.mean(np.abs(pooled_a - pooled_b))),
                "maximum_absolute_rate_difference": float(np.max(np.abs(pooled_a - pooled_b))),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(rate_rows)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    manuscript_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-corpus",
        type=Path,
        default=manuscript_root
        / "0. Dataset"
        / "analysis_ready"
        / "mental_health_unified_labels_final.csv",
    )
    parser.add_argument(
        "--multillm-outputs",
        type=Path,
        default=manuscript_root
        / "0. Dataset"
        / "multi_llm"
        / "mh_labeling_final.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.full_corpus.exists():
        raise FileNotFoundError(args.full_corpus)
    if not args.multillm_outputs.exists():
        raise FileNotFoundError(args.multillm_outputs)

    full = pd.read_csv(args.full_corpus, low_memory=False)
    if "Unnamed: 0" in full.columns:
        full = full.drop(columns=["Unnamed: 0"])
    full, sample, thresholds = reconstruct_sample(full)

    outputs = pd.read_csv(args.multillm_outputs, low_memory=False)
    outputs, design = validate_and_enrich_outputs(outputs, sample)
    raw_aspects = extract_raw_aspect_long(outputs)
    wide, model_pairs = matched_condition_wide(outputs)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(
        0, SAMPLE_SIZE, size=(args.bootstrap_resamples, SAMPLE_SIZE)
    )

    characteristics = sample_characteristics(full, sample)
    entropy_stats = entropy_summary(full, sample)
    hard_pairs = hard_label_pairwise(outputs, model_pairs, bootstrap_indices)
    factors = factor_level_summary(outputs, wide)
    models = model_level_summary(outputs, hard_pairs)
    aspect_prevalence = aspect_prevalence_by_model(outputs)
    entropy_pairs = pairwise_item_mean_correlations(
        outputs,
        "normalized_score_entropy",
        "normalized_score_entropy",
        model_pairs,
        bootstrap_indices,
    )
    aspect_count_pairs = pairwise_item_mean_correlations(
        outputs,
        "aspect_count",
        "aspect_count",
        model_pairs,
        bootstrap_indices,
    )
    cooccurrence_pairs, cooccurrence_rates = cooccurrence_robustness(
        outputs, model_pairs, bootstrap_indices
    )
    (
        aspect_agreement,
        aspect_schema_sensitivity,
        both_present_agreement,
        aspect_agreement_qc,
    ) = aspect_agreement_outputs(raw_aspects, model_pairs, bootstrap_indices)

    exports = {
        "sample_characteristics.csv": characteristics,
        "sample_entropy_summary.csv": entropy_stats,
        "factor_level_summary.csv": factors,
        "model_level_summary.csv": models,
        "aspect_prevalence_by_model.csv": aspect_prevalence,
        "hard_label_pairwise_robustness.csv": hard_pairs,
        "entropy_pairwise_robustness.csv": entropy_pairs,
        "aspect_count_pairwise_robustness.csv": aspect_count_pairs,
        "cooccurrence_pattern_pairwise_robustness.csv": cooccurrence_pairs,
        "cooccurrence_rates_by_model.csv": cooccurrence_rates,
        "aspect_level_agreement_robustness.csv": aspect_agreement,
        "aspect_strength_schema_sensitivity.csv": aspect_schema_sensitivity,
        "aspect_strength_both_present_agreement.csv": both_present_agreement,
    }
    for name, frame in exports.items():
        frame.to_csv(args.output_dir / name, index=False)

    manifest = {
        "analysis": "Multi-LLM protocol-sensitivity documentation and robustness",
        "api_calls_performed": False,
        "input_files": {
            "full_corpus": {
                "path": args.full_corpus.name,
                "sha256": sha256_file(args.full_corpus),
                "rows": int(len(full)),
            },
            "multillm_outputs": {
                "path": args.multillm_outputs.name,
                "sha256": sha256_file(args.multillm_outputs),
                "rows": int(len(outputs)),
            },
        },
        "sampling": {
            "sample_size": SAMPLE_SIZE,
            "random_state": RANDOM_STATE,
            "strategy": "proportional strata from the historical raw-score dispersion statistic",
            "historical_sampling_statistic": "-sum(s_k*log2(s_k)) applied directly to stored u_p_* scores without row normalization",
            "quartile_thresholds": thresholds,
            "realized_stratum_counts": {
                str(key): int(value)
                for key, value in sample["entropy_stratum"].value_counts().sort_index().items()
            },
            "stored_score_sum_check": {
                "full_min": float(full["sampling_score_sum"].min()),
                "full_max": float(full["sampling_score_sum"].max()),
                "full_rows_not_equal_one": int((~np.isclose(full["sampling_score_sum"], 1.0)).sum()),
                "sample_min": float(sample["sampling_score_sum"].min()),
                "sample_max": float(sample["sampling_score_sum"].max()),
                "sample_rows_not_equal_one": int((~np.isclose(sample["sampling_score_sum"], 1.0)).sum()),
            },
            "note": "Q2 is empty because the historical raw-score 25th and 50th percentile cut points are both zero. Manuscript-facing entropy summaries row-normalize scores first.",
        },
        "crossed_design_validation": design,
        "aspect_strength_agreement_validation": {
            key: value
            for key, value in aspect_agreement_qc.items()
            if key
            not in {
                "invalid_strength_literal_counts",
                "strength_present_mismatch_direction_counts",
            }
        },
        "invalid_strength_literal_counts": aspect_agreement_qc[
            "invalid_strength_literal_counts"
        ],
        "strength_present_mismatch_direction_counts": aspect_agreement_qc[
            "strength_present_mismatch_direction_counts"
        ],
        "definitions": {
            "screening_label": "u_label from the full-corpus gpt-4o-mini screening annotation",
            "released_category": "canonicalized status from the released corpus",
            "analysis_entropy_nats": "-sum(p*ln(p)) after row-normalizing the stored screening u_p_* scores",
            "analysis_normalized_entropy": "analysis_entropy_nats/ln(7)",
            "score_entropy_nats": "-sum(p*ln(p)) for each crossed-model score vector",
            "normalized_score_entropy": "-sum(p*ln(p))/ln(7) for each crossed-model score vector",
            "multi_aspect": "at least two of the six aspect-present fields are true",
            "matched_cross_llm_pair_agreement": "mean of the six pairwise hard-label agreement indicators within the same item, prompt, temperature, and seed condition",
            "cooccurrence_pattern": "the 15 pooled joint-positive rates among the six binary aspect-present fields",
            "strength_3level_order": STRENGTH_ORDER,
            "raw_valid_strength_analysis": "Includes valid three-level strength strings regardless of agreement with the archived present flag.",
            "schema_consistent_strength_analysis": "Includes valid strength strings only when present equals strength != none; entries are excluded rather than recoded.",
            "binary_presence_analysis": "Uses the archived raw-JSON present flag and does not derive presence from strength.",
            "both_present_strength_analysis": "Among schema-consistent entries for which both models marked the aspect present, compares weak versus clear exactly.",
            "nominal_kappa": "Unweighted Cohen's kappa.",
            "linear_weighted_kappa": "Cohen's kappa with linear weights over none=0, weak=1, clear=2.",
            "micro_agreement": "Metrics calculated after pooling contingency counts across the included pair-aspect cells.",
            "macro_pair_aspect_agreement": "Equal mean of the 36 model-pair-by-aspect metrics.",
            "rare_labels": sorted(RARE_LABELS),
        },
        "bootstrap": {
            "resamples": int(args.bootstrap_resamples),
            "seed": BOOTSTRAP_SEED,
            "unit": "item cluster",
            "interval": "percentile 95%",
            "paired_across_strength_definitions": True,
            "aspect_agreement_resamples": int(args.bootstrap_resamples),
            "aspect_agreement_unit": "item_id cluster",
            "aspect_agreement_fixed_grid": "4 models x 3 prompts x 6 temperatures x 3 seeds x 6 aspects",
            "aspect_agreement_interval": "percentile 95%",
        },
        "limitations": [
            "The audit is a separately sampled subset of the same corpus, not an external validation dataset.",
            "The sample was stratified only by screening-score entropy; label and disagreement composition were not quota controlled.",
            "Rare-category summaries are descriptive because the 100-item sample contains few rare-label items.",
            "The final output CSV has no explicit per-item parse-fallback flag, so a validation-failure rate cannot be reconstructed from this file alone.",
            "Twenty-one invalid strength strings are excluded without semantic recoding.",
            "The raw-valid strength analysis retains 96 valid strength/presence inconsistencies; the schema-consistent sensitivity analysis excludes them rather than reconciling them.",
            "Item-cluster confidence intervals quantify item-sampling uncertainty conditional on the fixed models, prompts, temperatures, and seeds.",
            "Model-pair rows share model ratings and do not support naive cell-level inference or generalization to unseen LLMs.",
            "Cross-model correlations assess relative item ordering or co-occurrence structure; absolute entropy and aspect prevalence remain model dependent.",
        ],
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "output_files": sorted(exports),
    }
    with (args.output_dir / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(f"Validated {len(outputs):,} crossed outputs on {sample['item_id'].nunique()} items.")
    print(f"Wrote {len(exports) + 1} files to {args.output_dir}")


if __name__ == "__main__":
    main()

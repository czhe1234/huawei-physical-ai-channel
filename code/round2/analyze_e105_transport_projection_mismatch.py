#!/usr/bin/env python3
"""Locate where the E104 endpoint PAS gain is lost in joint projection."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

import analyze_e100_locked_energy_angle_correction as e100
import analyze_e86_surface_endpoint_v11_blend as e86
import train_surface_interaction_energy_model as e85
from analyze_v4_exact_joint_projection import (
    exact_pas_projection,
    exact_pdp_projection,
    pas_distribution,
    pdp_distribution,
    preserve_point_power,
)
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import endpoint_cosine, point_metrics


VERSION = "Round2-E105-transport-projection-mismatch"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "transport_projection_mismatch"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_transport_projection_mismatch"
REPORT = ANALYSIS_DIR / "transport_projection_mismatch.json"
METRICS = ARTIFACT_DIR / "transport_projection_mismatch_metrics.npz"
E104_DIR = PATHS.artifacts / "round2" / "locked_pdp_angle_transport"
E104_PAS = E104_DIR / "strict_oof_locked_pdp_transport_pas_f16.npy"
E104_METRICS = E104_DIR / "strict_oof_locked_pdp_transport_metrics.npz"
OLD_PAS = e85.ARTIFACT_DIR / "e86_geometry_sequence_rendered_pas_f16.npy"
OLD_PDP = e85.ARTIFACT_DIR / "e86_geometry_sequence_rendered_pdp_f16.npy"
COMPLEX_BANK = (
    PATHS.artifacts / "round2" / "complex_carrier_bank" / "oof_complex_carrier_bank.npy"
)
STACKED_CARRIER = (
    PATHS.artifacts / "round2" / "complex_carrier_stacking" / "oof_stacked_complex_channel.npy"
)
TINY = 1e-30


def finite_correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    x = left[valid]
    y = right[valid]
    if len(x) < 3 or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return {"count": int(len(x)), "pearson": 0.0, "spearman": 0.0}
    return {
        "count": int(len(x)),
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def sign_summary(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    x = left[valid]
    y = right[valid]
    return {
        "count": int(len(x)),
        "same_sign_fraction": float(np.mean(np.signbit(x) == np.signbit(y))),
        "left_positive_fraction": float(np.mean(x > 0.0)),
        "right_positive_fraction": float(np.mean(y > 0.0)),
        "both_positive_fraction": float(np.mean((x > 0.0) & (y > 0.0))),
        "left_positive_right_nonpositive_fraction": float(
            np.mean((x > 0.0) & (y <= 0.0))
        ),
    }


def fold_means(values: np.ndarray, fold_id: np.ndarray) -> list[dict[str, float | int]]:
    return [
        {
            "fold": outer,
            "count": int(np.sum(fold_id == outer)),
            "mean": float(np.mean(values[fold_id == outer], dtype=np.float64)),
        }
        for outer in range(5)
    ]


def compare_metrics(
    selected: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fold_id: np.ndarray,
) -> dict[str, object]:
    return e100.metric_comparison(selected, baseline, fold_id)


def channel_metrics(
    channel: torch.Tensor,
    truth: torch.Tensor,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.inference_mode():
        return point_metrics(channel, truth, truth_pas, truth_pdp)


def pas_target_metrics(
    target: torch.Tensor, truth_pas: torch.Tensor
) -> np.ndarray:
    return endpoint_cosine(target, truth_pas, "pas")


def stage_record(
    name: str,
    old_channel: torch.Tensor,
    new_channel: torch.Tensor,
    truth: torch.Tensor,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
    fold_id: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dict[str, object],
]:
    old_metrics = channel_metrics(old_channel, truth, truth_pas, truth_pdp)
    new_metrics = channel_metrics(new_channel, truth, truth_pas, truth_pdp)
    arrays[f"{name}_pas_delta"] = (new_metrics[0] - old_metrics[0]).astype(np.float32)
    arrays[f"{name}_pdp_delta"] = (new_metrics[1] - old_metrics[1]).astype(np.float32)
    arrays[f"{name}_error_delta"] = (new_metrics[2] - old_metrics[2]).astype(np.float64)
    comparison = compare_metrics(new_metrics, old_metrics, fold_id)
    return old_metrics, new_metrics, comparison


def array_shape(path: object) -> list[int] | None:
    path = PATHS.root / path if isinstance(path, str) else path
    if not path.is_file():
        return None
    return list(np.load(path, mmap_mode="r").shape)


def main() -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    required = (
        E104_PAS,
        E104_METRICS,
        OLD_PAS,
        OLD_PDP,
        e86.V11_CHANNEL,
        e86.TRAIN_CHANNEL,
        e100.V12_CHANNEL,
        e85.OOF_META,
    )
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    metadata = np.load(e85.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    e104_metrics = np.load(E104_METRICS, allow_pickle=False)
    if not np.array_equal(query_ids, e104_metrics["query_ids"].astype(np.int64)):
        raise ValueError("E104 query order does not match canonical 500 points")

    base = torch.as_tensor(
        np.array(np.load(e86.V11_CHANNEL, mmap_mode="r"), copy=True), device=device
    )
    truth = torch.as_tensor(
        np.asarray(
            np.load(e86.TRAIN_CHANNEL, mmap_mode="r")[query_ids],
            dtype=np.complex64,
        ),
        device=device,
    )
    v12 = torch.as_tensor(
        np.array(np.load(e100.V12_CHANNEL, mmap_mode="r"), copy=True), device=device
    )
    old_endpoint_pas = torch.as_tensor(
        np.asarray(np.load(OLD_PAS, mmap_mode="r"), dtype=np.float32), device=device
    )
    new_endpoint_collapsed = torch.as_tensor(
        np.asarray(np.load(E104_PAS, mmap_mode="r"), dtype=np.float32), device=device
    )
    new_endpoint_pas = new_endpoint_collapsed[..., None].expand_as(old_endpoint_pas)
    endpoint_pdp = torch.as_tensor(
        np.asarray(np.load(OLD_PDP, mmap_mode="r"), dtype=np.float32), device=device
    )

    with torch.inference_mode():
        base_pas = pas_distribution(base)
        base_pdp = pdp_distribution(base)
        truth_pas = pas_distribution(truth)
        truth_pdp = pdp_distribution(truth)
        old_desired_pas = e86.endpoint_target(
            base_pas, old_endpoint_pas, 0.20, "sqrt", (1, 2, 3)
        )
        new_desired_pas = e86.endpoint_target(
            base_pas, new_endpoint_pas, 0.20, "sqrt", (1, 2, 3)
        )
        desired_pdp = e86.endpoint_target(
            base_pdp, endpoint_pdp, 0.20, "sqrt", (3,)
        )

    old_endpoint_score = pas_target_metrics(old_endpoint_pas, truth_pas)
    new_endpoint_score = pas_target_metrics(new_endpoint_pas, truth_pas)
    endpoint_delta = new_endpoint_score - old_endpoint_score
    old_desired_score = pas_target_metrics(old_desired_pas, truth_pas)
    new_desired_score = pas_target_metrics(new_desired_pas, truth_pas)
    desired_delta = new_desired_score - old_desired_score

    saved_endpoint_delta = np.asarray(e104_metrics["endpoint_delta"], dtype=np.float64)
    endpoint_reproduction = {
        "old_mean": float(np.mean(old_endpoint_score)),
        "new_mean": float(np.mean(new_endpoint_score)),
        "gain": float(np.mean(endpoint_delta)),
        "saved_gain": float(np.mean(saved_endpoint_delta)),
        "point_delta_max_abs_difference": float(
            np.max(np.abs(endpoint_delta - saved_endpoint_delta))
        ),
        "old_frequency_max_abs_deviation": float(
            torch.max(
                torch.abs(old_endpoint_pas - old_endpoint_pas.mean(dim=5, keepdim=True))
            ).item()
        ),
    }
    desired_summary = {
        "old_mean": float(np.mean(old_desired_score)),
        "new_mean": float(np.mean(new_desired_score)),
        "gain": float(np.mean(desired_delta)),
        "fold_gain": fold_means(desired_delta, fold_id),
        "endpoint_to_desired_correlation": finite_correlation(
            endpoint_delta, desired_delta
        ),
        "endpoint_to_desired_sign": sign_summary(endpoint_delta, desired_delta),
    }

    arrays: dict[str, np.ndarray] = {
        "query_ids": query_ids.astype(np.int32),
        "fold_id": fold_id,
        "endpoint_delta": endpoint_delta.astype(np.float32),
        "desired_pas_delta": desired_delta.astype(np.float32),
    }
    stages: list[dict[str, object]] = []
    old_output = base.clone()
    new_output = base.clone()
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    old_previous = channel_metrics(base, truth, truth_pas, truth_pdp)
    new_previous = old_previous

    for cycle in range(1, 5):
        with torch.inference_mode():
            old_output = exact_pas_projection(old_output, old_desired_pas)
            new_output = exact_pas_projection(new_output, new_desired_pas)
        old_values, new_values, comparison = stage_record(
            f"cycle{cycle}_pas",
            old_output,
            new_output,
            truth,
            truth_pas,
            truth_pdp,
            fold_id,
            arrays,
        )
        stages.append(
            {
                "stage": f"cycle{cycle}_pas",
                "new_vs_old": comparison,
                "old_vs_previous": compare_metrics(old_values, old_previous, fold_id),
                "new_vs_previous": compare_metrics(new_values, new_previous, fold_id),
            }
        )
        old_previous, new_previous = old_values, new_values

        with torch.inference_mode():
            old_output = exact_pdp_projection(old_output, desired_pdp)
            new_output = exact_pdp_projection(new_output, desired_pdp)
        old_values, new_values, comparison = stage_record(
            f"cycle{cycle}_pdp",
            old_output,
            new_output,
            truth,
            truth_pas,
            truth_pdp,
            fold_id,
            arrays,
        )
        stages.append(
            {
                "stage": f"cycle{cycle}_pdp",
                "new_vs_old": comparison,
                "old_vs_previous": compare_metrics(old_values, old_previous, fold_id),
                "new_vs_previous": compare_metrics(new_values, new_previous, fold_id),
            }
        )
        old_previous, new_previous = old_values, new_values

        with torch.inference_mode():
            old_output = preserve_point_power(old_output, reference_power)
            new_output = preserve_point_power(new_output, reference_power)
        old_values, new_values, comparison = stage_record(
            f"cycle{cycle}_power",
            old_output,
            new_output,
            truth,
            truth_pas,
            truth_pdp,
            fold_id,
            arrays,
        )
        stages.append(
            {
                "stage": f"cycle{cycle}_power",
                "new_vs_old": comparison,
                "old_vs_previous": compare_metrics(old_values, old_previous, fold_id),
                "new_vs_previous": compare_metrics(new_values, new_previous, fold_id),
            }
        )
        old_previous, new_previous = old_values, new_values

    final_old_metrics = channel_metrics(old_output, truth, truth_pas, truth_pdp)
    final_new_metrics = channel_metrics(new_output, truth, truth_pas, truth_pdp)
    v12_metrics = channel_metrics(v12, truth, truth_pas, truth_pdp)
    final_pas_delta = final_new_metrics[0] - final_old_metrics[0]
    final_pdp_delta = final_new_metrics[1] - final_old_metrics[1]
    final_error_improvement = final_old_metrics[2] - final_new_metrics[2]

    correlation = {
        "endpoint_to_final_pas": finite_correlation(endpoint_delta, final_pas_delta),
        "desired_to_final_pas": finite_correlation(desired_delta, final_pas_delta),
        "endpoint_to_final_error_improvement": finite_correlation(
            endpoint_delta, final_error_improvement
        ),
        "desired_to_final_error_improvement": finite_correlation(
            desired_delta, final_error_improvement
        ),
        "final_pas_to_error_improvement": finite_correlation(
            final_pas_delta, final_error_improvement
        ),
        "endpoint_to_final_pas_sign": sign_summary(endpoint_delta, final_pas_delta),
        "desired_to_final_pas_sign": sign_summary(desired_delta, final_pas_delta),
    }
    fold_attribution = []
    for outer in range(5):
        mask = fold_id == outer
        fold_attribution.append(
            {
                "fold": outer,
                "count": int(np.sum(mask)),
                "endpoint_gain": float(np.mean(endpoint_delta[mask])),
                "desired_pas_gain": float(np.mean(desired_delta[mask])),
                "final_pas_gain": float(np.mean(final_pas_delta[mask])),
                "final_pdp_gain": float(np.mean(final_pdp_delta[mask])),
                "final_error_improvement": float(
                    np.sum(final_error_improvement[mask], dtype=np.float64)
                ),
                "final_score_gain": float(
                    statistics(*final_new_metrics, mask)["score"]
                    - statistics(*final_old_metrics, mask)["score"]
                ),
            }
        )

    old_replay_vs_v12 = compare_metrics(final_old_metrics, v12_metrics, fold_id)
    new_vs_old = compare_metrics(final_new_metrics, final_old_metrics, fold_id)
    arrays["final_pas_delta"] = final_pas_delta.astype(np.float32)
    arrays["final_pdp_delta"] = final_pdp_delta.astype(np.float32)
    arrays["final_error_improvement"] = final_error_improvement.astype(np.float64)
    np.savez_compressed(METRICS, **arrays)

    carrier_shapes = {
        "canonical_v11": array_shape(e86.V11_CHANNEL),
        "complex_carrier_bank": array_shape(COMPLEX_BANK),
        "stacked_complex_carrier": array_shape(STACKED_CARRIER),
    }
    carrier_audit = {
        "known_shapes": carrier_shapes,
        "full_4000_v11_equivalent_oof_available": any(
            shape is not None and 4000 in shape[:2] for shape in carrier_shapes.values()
        ),
    }

    report = {
        "experiment": VERSION,
        "status": "diagnostic_complete",
        "primary_hypothesis": (
            "E104 improves a standalone frequency-invariant endpoint objective, but "
            "the gain is either rotated by the V11-relative sqrt blend or erased by "
            "the locked-PDP alternating projection."
        ),
        "validation": (
            "Canonical 500 contiguous holes; no labels are used for model fitting in "
            "this diagnostic. Existing strict E104 predictions are replayed only."
        ),
        "endpoint_reproduction": endpoint_reproduction,
        "desired_pas_before_projection": desired_summary,
        "projection_stages": stages,
        "final": {
            "old_replay_vs_stored_v12": old_replay_vs_v12,
            "transport_vs_old_replay": new_vs_old,
            "fold_attribution": fold_attribution,
            "correlation": correlation,
        },
        "carrier_audit": carrier_audit,
        "outputs": {
            "metrics": PATHS.relative(METRICS),
            "array_keys": sorted(arrays),
            "finite": bool(all(np.isfinite(value).all() for value in arrays.values())),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "endpoint_reproduction": endpoint_reproduction,
                "desired_pas_before_projection": desired_summary,
                "stage_score_delta": [
                    {
                        "stage": row["stage"],
                        "score_delta": row["new_vs_old"]["delta"],
                        "pas_delta": row["new_vs_old"]["selected"]["pas"]
                        - row["new_vs_old"]["baseline"]["pas"],
                        "pdp_delta": row["new_vs_old"]["selected"]["pdp"]
                        - row["new_vs_old"]["baseline"]["pdp"],
                    }
                    for row in stages
                ],
                "final": report["final"],
                "carrier_audit": carrier_audit,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

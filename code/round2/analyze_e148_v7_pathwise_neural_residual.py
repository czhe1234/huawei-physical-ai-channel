#!/usr/bin/env python3
"""Test a zero-initialized pathwise neural residual on the strict V7 carrier."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import time

import numpy as np
import torch

import analyze_e119_v12_unique_path_complement as e119
import analyze_e123_complete_path_variable_system as e123
import train_e115_complete_path_state_operator as e115
import train_e118_hierarchical_unique_path_operator as e118
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import statistics


VERSION = "Round2-E148-v7-pathwise-neural-residual"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v7_pathwise_neural_residual"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v7_pathwise_neural_residual"
METRICS = ARTIFACT_DIR / "v7_pathwise_neural_residual_metrics.npz"
REPORT = ANALYSIS_DIR / "v7_pathwise_neural_residual.json"
V7_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "full_sionna_migration"
    / "oof_v7_nested_channel.npy"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
PRIMARY_VARIANT = "mass_coordinate"
VARIANTS = ("mass", "mass_coordinate", "full_state")
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    alpha_grid: tuple[float, ...] = (
        0.0,
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.35,
        0.50,
        0.75,
        1.00,
    )
    metric_batch_size: int = 12
    render_batch_size: int = 4
    required_score_gain: float = 0.005
    required_positive_folds: int = 4
    minimum_geometry_margin: float = 0.002
    maximum_worst_fold_loss: float = 0.001
    control_distance_strata: int = 4
    seed: int = 20_260_809_148


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--render-batch-size", type=int, default=4)
    parser.add_argument("--metric-batch-size", type=int, default=12)
    return parser.parse_args()


def checkpoint_path(outer: int):
    return (
        e118.ARTIFACT_DIR / f"e120_outer{outer}_hierarchical_scene.pt"
    )


def load_fold_model(
    outer: int,
    device: torch.device,
) -> tuple[
    e118.HierarchicalUniquePathOperator,
    e118.ClusterTensorStore,
    dict[str, torch.Tensor],
    dict[str, object],
]:
    directory = e118.cache_dir(e123.cache_tag(outer))
    store = e118.ClusterTensorStore(directory, device)
    path = checkpoint_path(outer)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = e118.Config(**checkpoint["config"])
    variant = e115.Variant(**checkpoint["variant"])
    model = e118.HierarchicalUniquePathOperator(
        store.raw.candidate_feature.shape[-1],
        store.cluster_feature.shape[-1],
        store.raw.context.shape[-1],
        store.graph_feature.shape[-1],
        config,
        variant,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    normalization = {
        key: value.to(device) for key, value in checkpoint["normalization"].items()
    }
    return model, store, normalization, checkpoint


def state_variant(
    raw: dict[str, torch.Tensor],
    learned: dict[str, torch.Tensor],
    name: str,
) -> dict[str, torch.Tensor]:
    output = dict(raw)
    output["mass"] = learned["mass"]
    output["mass_logits"] = learned["mass_logits"]
    if name in ("mass_coordinate", "full_state"):
        output["coordinate"] = learned["coordinate"]
        output["coordinate_residual"] = learned["coordinate_residual"]
    if name == "full_state":
        output["port"] = learned["port"]
        output["port_residual"] = learned["port_residual"]
        output["phase"] = learned["phase"]
        output["phase_residual"] = learned["phase_residual"]
        output["log_power"] = learned["log_power"]
    return output


@torch.inference_mode()
def render_path_residuals(
    model: e118.HierarchicalUniquePathOperator,
    store: e118.ClusterTensorStore,
    normalization: dict[str, torch.Tensor],
    point_ids: np.ndarray,
    batch_size: int,
) -> dict[str, np.ndarray]:
    shape = (len(point_ids), e115.MP * e115.MH * e115.MV, e115.N_UE, e115.S_SC)
    output = {name: np.empty(shape, dtype=np.complex64) for name in VARIANTS}
    for start in range(0, len(point_ids), batch_size):
        end = min(start + batch_size, len(point_ids))
        rows = torch.as_tensor(point_ids[start:end], device=store.device)
        batch = store.batch(rows, normalization)
        raw = e115.raw_output(batch)
        learned = model(batch)
        raw_channel = e115.render_complex_channel(raw)
        for name in VARIANTS:
            candidate = e115.render_complex_channel(state_variant(raw, learned, name))
            output[name][start:end] = (candidate - raw_channel).cpu().numpy().astype(
                np.complex64
            )
    return output


@torch.inference_mode()
def metric_arrays(
    prediction: np.ndarray,
    truth: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(prediction)
    pas = np.empty(count, dtype=np.float64)
    pdp = np.empty(count, dtype=np.float64)
    error = np.empty(count, dtype=np.float64)
    energy = np.empty(count, dtype=np.float64)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        predicted = torch.tensor(
            np.asarray(prediction[start:end], dtype=np.complex64), device=device
        )
        target = torch.tensor(
            np.asarray(truth[start:end], dtype=np.complex64), device=device
        )
        values = e115.torch_official_metrics(predicted, target)
        for destination, value in zip((pas, pdp, error, energy), values, strict=True):
            destination[start:end] = value.cpu().numpy()
    return pas, pdp, error, energy


def choose_alpha(
    base: np.ndarray,
    residual: np.ndarray,
    truth: np.ndarray,
    config: Config,
    device: torch.device,
) -> tuple[float, list[dict[str, object]]]:
    rows = []
    best_alpha = 0.0
    best_score = -np.inf
    for alpha in config.alpha_grid:
        values = metric_arrays(
            base + np.float32(alpha) * residual,
            truth,
            device,
            config.metric_batch_size,
        )
        summary = statistics(*values)
        rows.append({"alpha": float(alpha), "metrics": summary})
        score = float(summary["score"])
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and alpha < best_alpha
        ):
            best_score = score
            best_alpha = float(alpha)
    return best_alpha, rows


def mismatch_permutation(
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    seed: int,
    strata: int,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    radial = np.linalg.norm(positions - bs_positions[site], axis=1)
    permutation = np.arange(len(positions), dtype=np.int64)
    for site_value in np.unique(site):
        site_rows = np.flatnonzero(site == site_value)
        edges = np.quantile(radial[site_rows], np.linspace(0.0, 1.0, strata + 1))
        bins = np.clip(np.searchsorted(edges[1:-1], radial[site_rows]), 0, strata - 1)
        for bin_value in range(strata):
            rows = site_rows[bins == bin_value]
            if len(rows) <= 1:
                continue
            shuffled = rows[rng.permutation(len(rows))]
            if np.any(shuffled == rows):
                shuffled = np.roll(shuffled, 1)
            permutation[rows] = shuffled
    fixed = permutation == np.arange(len(permutation))
    if np.any(fixed):
        for site_value in np.unique(site[fixed]):
            rows = np.flatnonzero((site == site_value) & fixed)
            pool = np.flatnonzero(site == site_value)
            if len(pool) > 1:
                for row in rows:
                    choices = pool[pool != row]
                    permutation[row] = choices[
                        int(np.argmin(np.abs(radial[choices] - radial[row])))
                    ]
    separation = np.linalg.norm(positions - positions[permutation], axis=1)
    radial_delta = np.abs(radial - radial[permutation])
    return permutation, {
        "self_fraction": float(np.mean(permutation == np.arange(len(permutation)))),
        "median_spatial_separation_m": float(np.median(separation)),
        "median_bs_distance_difference_m": float(np.median(radial_delta)),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config(
        metric_batch_size=args.metric_batch_size,
        render_batch_size=args.render_batch_size,
    )
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed % (2**32 - 1))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    for path in (V7_CHANNEL, TRAIN_CHANNEL, e119.OOF_META, e115.FULL_FOLD):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = np.load(e119.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    full_fold_all = np.load(e115.FULL_FOLD).astype(np.int8)
    query_fold = full_fold_all[query_ids]
    setup, positions, _, site_all, _ = load_positions()
    query_positions = positions[query_ids]
    query_site = site_all[query_ids].astype(np.int8)
    v7 = np.load(V7_CHANNEL, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    if v7.shape != (len(query_ids), 256, 4, 192):
        raise RuntimeError(f"unexpected V7 shape: {v7.shape}")

    baseline_values = metric_arrays(
        np.asarray(v7, dtype=np.complex64),
        np.asarray(truth_all[query_ids], dtype=np.complex64),
        device,
        config.metric_batch_size,
    )
    baseline = statistics(*baseline_values)
    expected_v7 = 0.644456345600551
    if abs(float(baseline["score"]) - expected_v7) > 1e-9:
        raise RuntimeError(
            f"V7 baseline mismatch: {baseline['score']} != {expected_v7}"
        )

    selected_values = {
        name: tuple(np.empty(len(query_ids), dtype=np.float64) for _ in range(4))
        for name in VARIANTS
    }
    shuffled_values = tuple(np.empty(len(query_ids), dtype=np.float64) for _ in range(4))
    selected_alpha = {name: np.zeros(5, dtype=np.float32) for name in VARIANTS}
    relative_residual = {name: np.zeros(len(query_ids), dtype=np.float32) for name in VARIANTS}
    fold_records = []

    for outer in range(5):
        inner = (outer + 1) % 5
        validation_rows = np.flatnonzero(query_fold == inner)
        held_rows = np.flatnonzero(query_fold == outer)
        combined_rows = np.concatenate((validation_rows, held_rows))
        point_ids = query_ids[combined_rows]
        model, store, normalization, checkpoint = load_fold_model(outer, device)
        residuals = render_path_residuals(
            model,
            store,
            normalization,
            point_ids,
            config.render_batch_size,
        )
        validation_count = len(validation_rows)
        fold_variants = {}
        for name in VARIANTS:
            validation_residual = residuals[name][:validation_count]
            held_residual = residuals[name][validation_count:]
            validation_base = np.asarray(v7[validation_rows], dtype=np.complex64)
            validation_truth = np.asarray(
                truth_all[query_ids[validation_rows]], dtype=np.complex64
            )
            alpha, curve = choose_alpha(
                validation_base,
                validation_residual,
                validation_truth,
                config,
                device,
            )
            selected_alpha[name][outer] = np.float32(alpha)
            held_base = np.asarray(v7[held_rows], dtype=np.complex64)
            held_truth = np.asarray(truth_all[query_ids[held_rows]], dtype=np.complex64)
            active = (
                np.sum(np.abs(held_base) ** 2, axis=(1, 2, 3), dtype=np.float64)
                > TINY
            )
            held_residual = held_residual.copy()
            held_residual[~active] = 0.0
            prediction = held_base + np.float32(alpha) * held_residual
            values = metric_arrays(
                prediction,
                held_truth,
                device,
                config.metric_batch_size,
            )
            for destination, value in zip(
                selected_values[name], values, strict=True
            ):
                destination[held_rows] = value
            base_norm = np.sqrt(
                np.sum(np.abs(held_base) ** 2, axis=(1, 2, 3), dtype=np.float64)
            )
            residual_norm = np.sqrt(
                np.sum(np.abs(held_residual) ** 2, axis=(1, 2, 3), dtype=np.float64)
            )
            relative_residual[name][held_rows] = np.divide(
                residual_norm,
                np.maximum(base_norm, TINY),
            ).astype(np.float32)
            held_baseline = statistics(
                *(value[held_rows] for value in baseline_values)
            )
            held_summary = statistics(*values)
            fold_variants[name] = {
                "selected_alpha": float(alpha),
                "validation_curve": curve,
                "held_baseline": held_baseline,
                "held_selected": held_summary,
                "held_delta": float(held_summary["score"] - held_baseline["score"]),
                "held_relative_residual_median": float(
                    np.median(relative_residual[name][held_rows])
                ),
            }

            if name == PRIMARY_VARIANT:
                permutation, control = mismatch_permutation(
                    query_positions[held_rows],
                    query_site[held_rows],
                    setup.bs_positions,
                    config.seed + outer * 1009,
                    config.control_distance_strata,
                )
                shuffled_prediction = held_base + np.float32(alpha) * held_residual[
                    permutation
                ]
                control_values = metric_arrays(
                    shuffled_prediction,
                    held_truth,
                    device,
                    config.metric_batch_size,
                )
                for destination, value in zip(
                    shuffled_values, control_values, strict=True
                ):
                    destination[held_rows] = value
                fold_variants[name]["mismatch_control"] = {
                    **control,
                    "metrics": statistics(*control_values),
                }

        fold_records.append(
            {
                "outer": outer,
                "inner": inner,
                "fit_folds": [fold for fold in range(5) if fold not in (outer, inner)],
                "validation_count": int(len(validation_rows)),
                "held_count": int(len(held_rows)),
                "checkpoint": PATHS.relative(checkpoint_path(outer)),
                "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
                "variants": fold_variants,
            }
        )
        print(
            f"outer={outer} primary_alpha={selected_alpha[PRIMARY_VARIANT][outer]:.3f} "
            f"primary_delta={fold_variants[PRIMARY_VARIANT]['held_delta']:+.6f}",
            flush=True,
        )
        del model, store, residuals
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summaries = {}
    for name in VARIANTS:
        summary = statistics(*selected_values[name])
        fold_delta = [
            float(record["variants"][name]["held_delta"]) for record in fold_records
        ]
        summaries[name] = {
            "metrics": summary,
            "delta_to_v7": float(summary["score"] - baseline["score"]),
            "positive_folds": int(sum(value > 0.0 for value in fold_delta)),
            "worst_fold_delta": float(min(fold_delta)),
            "fold_delta": fold_delta,
            "selected_alpha": selected_alpha[name].tolist(),
            "relative_residual_median": float(np.median(relative_residual[name])),
        }
    shuffled = statistics(*shuffled_values)
    primary = summaries[PRIMARY_VARIANT]
    geometry_margin = float(primary["metrics"]["score"] - shuffled["score"])
    gate = {
        "primary_variant": PRIMARY_VARIANT,
        "required_score_gain": config.required_score_gain,
        "required_positive_folds": config.required_positive_folds,
        "minimum_geometry_margin": config.minimum_geometry_margin,
        "maximum_worst_fold_loss": config.maximum_worst_fold_loss,
        "score_gain": primary["delta_to_v7"],
        "positive_folds": primary["positive_folds"],
        "worst_fold_delta": primary["worst_fold_delta"],
        "geometry_margin": geometry_margin,
        "pas_non_worse": bool(primary["metrics"]["pas"] >= baseline["pas"]),
        "pdp_non_worse": bool(primary["metrics"]["pdp"] >= baseline["pdp"]),
        "nmse_non_worse": bool(primary["metrics"]["nmse"] <= baseline["nmse"]),
    }
    gate["passed"] = bool(
        gate["score_gain"] >= gate["required_score_gain"]
        and gate["positive_folds"] >= gate["required_positive_folds"]
        and gate["worst_fold_delta"] >= -gate["maximum_worst_fold_loss"]
        and gate["geometry_margin"] >= gate["minimum_geometry_margin"]
        and gate["pas_non_worse"]
        and gate["pdp_non_worse"]
        and gate["nmse_non_worse"]
    )

    report = {
        "experiment": VERSION,
        "status": "fivefold_information_screen_complete",
        "final_goal": (
            "Official-input-only Round 2 model above V12 local 0.677521 and online 0.70"
        ),
        "single_experimental_goal": (
            "Determine whether learned path-state changes are complementary to V7 when "
            "used as a zero-initialized physical residual instead of an absolute replacement."
        ),
        "primary_hypothesis": (
            "E115-E126 failed mainly because their absolute path renderer replaced the "
            "coherent V7 carrier. Subtracting the raw transported-path channel from the "
            "learned transported-path channel and adding only that change to V7 should "
            "preserve the carrier while exposing learned path birth, death, and coordinate shifts."
        ),
        "protocol": {
            "canonical_fixed500": True,
            "outer_folds": 5,
            "inner_rule": "(outer + 1) % 5",
            "alpha_selected_on_inner_only": True,
            "outer_used_for_final_scoring_only": True,
            "v7_exact_skip_connection": True,
            "zero_residual_equals_v7": True,
            "official_h_only": True,
            "sionna_labels_or_test_outputs_used": False,
            "v12_used": False,
            "submission_generated": False,
            "primary_variant": PRIMARY_VARIANT,
            "diagnostic_variants": [name for name in VARIANTS if name != PRIMARY_VARIANT],
        },
        "config": asdict(config),
        "baseline_v7": baseline,
        "variants": summaries,
        "primary_mismatch_control": shuffled,
        "primary_geometry_margin": geometry_margin,
        "folds": fold_records,
        "gate": gate,
        "decision": {
            "scale_up_authorized": bool(gate["passed"]),
            "next_if_passed": (
                "Fine-tune a bounded residual head directly on the official score while "
                "keeping the V7 skip and pathwise analytic renderer frozen."
            ),
            "next_if_failed": (
                "Do not enlarge the network. The existing official path-state direction is "
                "not sufficiently complementary to V7 through this physical residual interface."
            ),
        },
        "outputs": {
            "metrics": PATHS.relative(METRICS),
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        METRICS,
        query_ids=query_ids.astype(np.int32),
        query_fold=query_fold,
        baseline_pas=baseline_values[0],
        baseline_pdp=baseline_values[1],
        baseline_error=baseline_values[2],
        baseline_energy=baseline_values[3],
        primary_pas=selected_values[PRIMARY_VARIANT][0],
        primary_pdp=selected_values[PRIMARY_VARIANT][1],
        primary_error=selected_values[PRIMARY_VARIANT][2],
        primary_energy=selected_values[PRIMARY_VARIANT][3],
        shuffled_pas=shuffled_values[0],
        shuffled_pdp=shuffled_values[1],
        shuffled_error=shuffled_values[2],
        shuffled_energy=shuffled_values[3],
        primary_alpha=selected_alpha[PRIMARY_VARIANT],
        primary_relative_residual=relative_residual[PRIMARY_VARIANT],
    )
    print(
        json.dumps(
            {
                "baseline_v7": baseline,
                "variants": summaries,
                "primary_mismatch_control": shuffled,
                "gate": gate,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test Hellinger-space Sionna PAS corrections on the Round 2 V10 carrier."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import time

import numpy as np
import torch

from analyze_joint_path_token_oracle import OOF_META
from analyze_v11_explicit_path_pas_upper_bound import v10_oof_channel
from analyze_v11_low_rank_path_pas_residual import (
    SOURCE_NAMES,
    load_sources,
    normalize_pas,
    pas_from_slices,
    pas_slices,
    slice_cosine,
)
from analyze_v4_exact_joint_projection import (
    pas_distribution,
    pdp_distribution,
    truth_views,
)
from prepare_sionna_calibration_dataset import TRUTH_PAS
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics
from train_full_path_pas_slice_gate import project_joint


VERSION = "Round2-E75-sqrt-PAS-source-consensus"
PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n469_s24_k128_b096_d064_scatter1_sel106"
)
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_sqrt_pas_source_consensus"
REPORT = ANALYSIS_DIR / "sqrt_pas_source_consensus.json"
TINY = 1e-30


@dataclass(frozen=True)
class Candidate:
    name: str
    source: str | None = None
    geometry: str = "identity"
    alpha: float = 0.0
    order: str = "pas_pdp"
    cycles: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--alpha", default="0.01,0.02,0.035,0.05,0.075,0.10,0.15,0.20"
    )
    parser.add_argument("--cycles", default="1,2")
    parser.add_argument("--orders", default="pas_pdp,pdp_pas")
    return parser.parse_args()


def parse_float_grid(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("empty float grid")
    return result


def parse_int_grid(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("empty integer grid")
    return result


def build_consensus(
    base: torch.Tensor, sources: list[torch.Tensor], indices: tuple[int, ...]
) -> torch.Tensor:
    residual = torch.stack([sources[index] - base for index in indices])
    norm = residual.square().sum(dim=4, keepdim=True).sqrt().clamp_min(1e-8)
    unit = residual / norm
    mean_direction = unit.mean(dim=0)
    agreement = mean_direction.square().sum(dim=3, keepdim=True).sqrt()
    direction = mean_direction / agreement.clamp_min(1e-8)
    median_scale = residual.square().sum(dim=4).sqrt().median(dim=0).values
    return base + direction * median_scale[..., None] * agreement


def desired_slices(
    base: torch.Tensor,
    source: torch.Tensor,
    geometry: str,
    alpha: float,
) -> torch.Tensor:
    if geometry == "sqrt":
        amplitude = ((1.0 - alpha) * base + alpha * source).clamp_min(0.0)
        power = amplitude.square()
    elif geometry == "power":
        power = (1.0 - alpha) * base.square() + alpha * source.square()
    else:
        raise ValueError(geometry)
    return power / power.sum(dim=3, keepdim=True).clamp_min(TINY)


def fold_summary(
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fold_id: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    overall = statistics(*values)
    baseline_overall = statistics(*baseline)
    folds = []
    for fold in range(5):
        mask = fold_id == fold
        selected = statistics(*values, mask)
        anchor = statistics(*baseline, mask)
        folds.append(
            {
                "fold": fold,
                "baseline": anchor,
                "selected": selected,
                "delta": float(selected["score"] - anchor["score"]),
            }
        )
    return (
        {
            "baseline": baseline_overall,
            "selected": overall,
            "delta": float(overall["score"] - baseline_overall["score"]),
            "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
            "worst_fold_delta": float(min(row["delta"] for row in folds)),
        },
        folds,
    )


def choose_nested(
    candidates: list[Candidate],
    metrics: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fold_id: np.ndarray,
    geometry: str,
) -> dict[str, object]:
    available = [candidate for candidate in candidates if candidate.geometry == geometry]
    stitched = tuple(np.array(value, copy=True) for value in baseline)
    decisions = []
    for outer in range(5):
        fit_mask = fold_id != outer
        held_mask = fold_id == outer
        base_fit = statistics(*baseline, fit_mask)
        rows = []
        for candidate in available:
            fit = statistics(*metrics[candidate.name], fit_mask)
            inner_deltas = []
            for inner in range(5):
                if inner == outer:
                    continue
                mask = fold_id == inner
                result = statistics(*metrics[candidate.name], mask)
                anchor = statistics(*baseline, mask)
                inner_deltas.append(float(result["score"] - anchor["score"]))
            rows.append(
                {
                    "candidate": candidate.name,
                    "fit_score": float(fit["score"]),
                    "fit_delta": float(fit["score"] - base_fit["score"]),
                    "positive_inner_folds": int(sum(value > 0.0 for value in inner_deltas)),
                    "worst_inner_delta": float(min(inner_deltas)),
                    "inner_deltas": inner_deltas,
                }
            )
        eligible = [
            row
            for row in rows
            if row["fit_delta"] > 0.0 and row["positive_inner_folds"] >= 3
        ]
        winner = max(
            eligible,
            key=lambda row: (row["fit_score"], row["worst_inner_delta"]),
            default=None,
        )
        if winner is None:
            name = "identity"
        else:
            name = str(winner["candidate"])
            selected = metrics[name]
            for output, source in zip(stitched, selected):
                output[held_mask] = source[held_mask]
        held = statistics(*stitched, held_mask)
        anchor = statistics(*baseline, held_mask)
        decisions.append(
            {
                "fold": outer,
                "candidate": name,
                "held_delta": float(held["score"] - anchor["score"]),
                "selection": winner,
            }
        )
    result = statistics(*stitched)
    anchor = statistics(*baseline)
    return {
        "geometry": geometry,
        "baseline": anchor,
        "selected": result,
        "delta": float(result["score"] - anchor["score"]),
        "positive_folds": int(sum(row["held_delta"] > 0.0 for row in decisions)),
        "decisions": decisions,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    alphas = parse_float_grid(args.alpha)
    cycles_grid = parse_int_grid(args.cycles)
    orders = tuple(item.strip() for item in args.orders.split(",") if item.strip())
    if any(alpha <= 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("alpha values must be in (0, 1]")
    if any(cycles <= 0 for cycles in cycles_grid):
        raise ValueError("cycles must be positive")
    if not set(orders).issubset({"pas_pdp", "pdp_pas"}):
        raise ValueError(orders)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (OOF_META, PATH_CACHE, TRUTH_PAS)
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    metadata = np.load(OOF_META, allow_pickle=False)
    rows = np.load(PATH_CACHE / "rows.npy").astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)[rows]
    global_ids = metadata["query_ids"].astype(np.int64)[rows]
    base_channel = torch.as_tensor(v10_oof_channel(device), device=device)
    base_pas = normalize_pas(pas_distribution(base_channel))
    base_amplitude = torch.sqrt(pas_slices(base_pas).clamp_min(0.0))
    base_pdp = pdp_distribution(base_channel)
    truth_channel_all = np.load(
        PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r"
    )
    truth_channel = torch.as_tensor(
        np.asarray(truth_channel_all[global_ids], dtype=np.complex64), device=device
    )
    truth_pas, truth_pdp = truth_views(truth_channel)
    truth_pas_normalized = normalize_pas(
        torch.as_tensor(
            np.asarray(np.load(TRUTH_PAS, mmap_mode="r")[global_ids], dtype=np.float32),
            device=device,
        )
    )
    truth_slices = pas_slices(truth_pas_normalized)
    baseline_metrics = point_metrics(
        base_channel, truth_channel, truth_pas, truth_pdp
    )

    source_args = argparse.Namespace(
        batch_size=args.batch_size, force_incoherent=False
    )
    source_values = load_sources(rows, global_ids, source_args, device)
    source_map = dict(zip(SOURCE_NAMES, source_values))
    source_map.update(
        {
            "consensus_endpoint": build_consensus(
                base_amplitude, source_values, (0, 1, 2, 3)
            ),
            "consensus_path": build_consensus(base_amplitude, source_values, (4, 5)),
            "consensus_all": build_consensus(
                base_amplitude, source_values, tuple(range(len(source_values)))
            ),
        }
    )

    candidates = [Candidate("identity")]
    metrics = {"identity": baseline_metrics}
    endpoint_rows = []
    result_rows = []
    for source_name, source in source_map.items():
        for geometry in ("sqrt", "power"):
            for alpha in alphas:
                slices = desired_slices(base_amplitude, source, geometry, alpha)
                endpoint = slice_cosine(slices, truth_slices).mean(dim=(1, 2)).cpu().numpy()
                endpoint_rows.append(
                    {
                        "source": source_name,
                        "geometry": geometry,
                        "alpha": alpha,
                        "pas": float(np.mean(endpoint)),
                        "folds": [
                            float(np.mean(endpoint[fold_id == fold])) for fold in range(5)
                        ],
                    }
                )
                desired_pas = pas_from_slices(slices)
                for order in orders:
                    for cycles in cycles_grid:
                        name = (
                            f"{geometry}_{source_name}_a{alpha:.3f}_"
                            f"{order}_c{cycles}"
                        )
                        candidate = Candidate(
                            name=name,
                            source=source_name,
                            geometry=geometry,
                            alpha=alpha,
                            order=order,
                            cycles=cycles,
                        )
                        with torch.inference_mode():
                            channel = project_joint(
                                base_channel, desired_pas, base_pdp, cycles, order
                            )
                            values = point_metrics(
                                channel, truth_channel, truth_pas, truth_pdp
                            )
                        metrics[name] = values
                        summary, folds = fold_summary(
                            values, baseline_metrics, fold_id
                        )
                        candidates.append(candidate)
                        result_rows.append(
                            {
                                **candidate.__dict__,
                                **summary,
                                "folds": folds,
                            }
                        )
                del desired_pas, slices
        print(f"evaluated PAS source {source_name}", flush=True)

    result_rows.sort(key=lambda row: float(row["selected"]["score"]), reverse=True)
    endpoint_rows.sort(key=lambda row: float(row["pas"]), reverse=True)
    nested = {
        geometry: choose_nested(
            candidates, metrics, baseline_metrics, fold_id, geometry
        )
        for geometry in ("sqrt", "power")
    }
    deployable = [
        row
        for row in result_rows
        if row["geometry"] == "sqrt"
        and int(row["positive_folds"]) >= 4
        and float(row["delta"]) > 0.0
    ]
    selected = deployable[0] if deployable else result_rows[0]
    selected_source = str(selected["source"])
    selected_alpha = float(selected["alpha"])
    comparable_power = [
        row
        for row in result_rows
        if row["geometry"] == "power"
        and row["source"] == selected_source
        and float(row["alpha"]) == selected_alpha
        and row["order"] == selected["order"]
        and int(row["cycles"]) == int(selected["cycles"])
    ][0]
    passed = bool(
        nested["sqrt"]["delta"] >= 3e-4
        and nested["sqrt"]["positive_folds"] >= 4
        and float(selected["delta"]) >= 3e-4
        and int(selected["positive_folds"]) >= 4
        and float(selected["delta"]) > float(comparable_power["delta"]) + 5e-5
    )
    payload = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Sionna PAS corrections are better behaved along square-root probability "
            "geometry than under the previously tested linear power mixture."
        ),
        "configuration": {
            **vars(args),
            "alphas": alphas,
            "cycles_grid": cycles_grid,
            "orders": orders,
            "source_names": list(source_map),
        },
        "point_count": int(len(rows)),
        "baseline": statistics(*baseline_metrics),
        "nested_model_selection": nested,
        "selected": selected,
        "comparable_power_control": comparable_power,
        "top_candidates": result_rows[:30],
        "top_endpoint_candidates": endpoint_rows[:30],
        "deployment_gate": {
            "minimum_nested_delta": 3e-4,
            "minimum_nested_positive_folds": 4,
            "minimum_selected_delta": 3e-4,
            "minimum_selected_positive_folds": 4,
            "minimum_margin_over_power_control": 5e-5,
            "passed": passed,
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "nested": nested,
                "selected": selected,
                "power_control": comparable_power,
                "deployment_gate": payload["deployment_gate"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

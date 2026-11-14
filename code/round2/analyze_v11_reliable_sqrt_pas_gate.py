#!/usr/bin/env python3
"""Validate an observable reliability gate for square-root PAS consensus."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import time

import numpy as np
import torch

from analyze_joint_path_token_oracle import OOF_META
from analyze_v11_explicit_path_pas_upper_bound import v10_oof_channel
from analyze_v11_low_rank_path_pas_residual import (
    load_sources,
    normalize_pas,
    pas_from_slices,
    pas_slices,
)
from analyze_v11_sqrt_pas_source_consensus import build_consensus
from analyze_v4_exact_joint_projection import (
    pas_distribution,
    pdp_distribution,
    truth_views,
)
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics
from train_full_path_pas_slice_gate import project_joint


VERSION = "Round2-E76-reliable-sqrt-PAS-consensus"
PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n469_s24_k128_b096_d064_scatter1_sel106"
)
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_reliable_sqrt_pas_gate"
REPORT = ANALYSIS_DIR / "reliable_sqrt_pas_gate.json"
TINY = 1e-30


@dataclass(frozen=True)
class Candidate:
    name: str
    gate: str
    quantile: float
    alpha: float
    order: str
    cycles: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def percentile_rank(value: torch.Tensor) -> torch.Tensor:
    flat = value.reshape(-1)
    order = torch.argsort(flat, stable=True)
    ranks = torch.empty_like(flat)
    ranks[order] = torch.arange(
        1, len(flat) + 1, device=value.device, dtype=value.dtype
    ) / float(len(flat))
    return ranks.reshape_as(value)


def reliability_features(
    base: torch.Tensor, sources: list[torch.Tensor]
) -> dict[str, torch.Tensor]:
    endpoint = sources[:4]
    residual = torch.stack([source - base for source in endpoint])
    residual_norm = residual.square().sum(dim=4).sqrt()
    unit = residual / residual_norm[..., None].clamp_min(1e-8)
    agreement = unit.mean(dim=0).square().sum(dim=3).sqrt()
    pair_spread = torch.stack([source.square() for source in endpoint]).std(
        dim=0
    ).mean(dim=3)
    median_norm = residual_norm.median(dim=0).values
    base_power = base.square()
    entropy = -(
        base_power * torch.log(base_power.clamp_min(1e-12))
    ).sum(dim=3) / np.log(256.0)
    pair_rank = percentile_rank(pair_spread)
    norm_rank = percentile_rank(median_norm)
    entropy_rank = percentile_rank(entropy)
    disagreement_rank = percentile_rank(-agreement)
    return {
        "pair": pair_rank,
        "rnorm": norm_rank,
        "pair_rnorm": 0.5 * (pair_rank + norm_rank),
        "risk3": (pair_rank + norm_rank + entropy_rank) / 3.0,
        "risk": (
            pair_rank + norm_rank + entropy_rank + disagreement_rank
        ) / 4.0,
    }


def candidate_grid() -> list[Candidate]:
    gates = {
        "risk": (0.45, 0.50, 0.55),
        "rnorm": (0.55, 0.60, 0.65, 0.70),
        "pair_rnorm": (0.65, 0.70, 0.75),
    }
    output = []
    for gate, quantiles in gates.items():
        for quantile in quantiles:
            for alpha in (0.15, 0.20, 0.25):
                for order in ("pas_pdp", "pdp_pas"):
                    for cycles in (2, 4):
                        output.append(
                            Candidate(
                                name=(
                                    f"{gate}_q{quantile:.2f}_a{alpha:.2f}_"
                                    f"{order}_c{cycles}"
                                ),
                                gate=gate,
                                quantile=quantile,
                                alpha=alpha,
                                order=order,
                                cycles=cycles,
                            )
                        )
    return output


def metric_summary(
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fold_id: np.ndarray,
) -> dict[str, object]:
    selected = statistics(*values)
    anchor = statistics(*baseline)
    folds = []
    for fold in range(5):
        mask = fold_id == fold
        local = statistics(*values, mask)
        local_anchor = statistics(*baseline, mask)
        folds.append(
            {
                "fold": fold,
                "selected": local,
                "baseline": local_anchor,
                "delta": float(local["score"] - local_anchor["score"]),
            }
        )
    return {
        "baseline": anchor,
        "selected": selected,
        "delta": float(selected["score"] - anchor["score"]),
        "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
        "worst_fold_delta": float(min(row["delta"] for row in folds)),
        "folds": folds,
    }


def nested_selection(
    specs: list[Candidate],
    metrics: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fold_id: np.ndarray,
) -> dict[str, object]:
    stitched = tuple(np.array(value, copy=True) for value in baseline)
    decisions = []
    for outer in range(5):
        fit_mask = fold_id != outer
        held_mask = fold_id == outer
        fit_anchor = statistics(*baseline, fit_mask)
        rows = []
        for spec in specs:
            fit = statistics(*metrics[spec.name], fit_mask)
            inner_deltas = []
            for inner in range(5):
                if inner == outer:
                    continue
                mask = fold_id == inner
                result = statistics(*metrics[spec.name], mask)
                anchor = statistics(*baseline, mask)
                inner_deltas.append(float(result["score"] - anchor["score"]))
            rows.append(
                {
                    "candidate": spec.name,
                    "fit_score": float(fit["score"]),
                    "fit_delta": float(fit["score"] - fit_anchor["score"]),
                    "positive_inner_folds": int(sum(value > 0.0 for value in inner_deltas)),
                    "worst_inner_delta": float(min(inner_deltas)),
                    "inner_deltas": inner_deltas,
                }
            )
        eligible = [row for row in rows if row["positive_inner_folds"] == 4]
        winner = max(
            eligible,
            key=lambda row: (row["fit_score"], row["worst_inner_delta"]),
            default=None,
        )
        name = "identity" if winner is None else str(winner["candidate"])
        if winner is not None:
            for output, source in zip(stitched, metrics[name]):
                output[held_mask] = source[held_mask]
        selected = statistics(*stitched, held_mask)
        anchor = statistics(*baseline, held_mask)
        decisions.append(
            {
                "fold": outer,
                "candidate": name,
                "held_delta": float(selected["score"] - anchor["score"]),
                "selection": winner,
            }
        )
    selected = statistics(*stitched)
    anchor = statistics(*baseline)
    return {
        "baseline": anchor,
        "selected": selected,
        "delta": float(selected["score"] - anchor["score"]),
        "positive_folds": int(sum(row["held_delta"] > 0.0 for row in decisions)),
        "decisions": decisions,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
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
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth_channel = torch.as_tensor(
        np.asarray(truth_all[global_ids], dtype=np.complex64), device=device
    )
    truth_pas, truth_pdp = truth_views(truth_channel)
    baseline_metrics = point_metrics(
        base_channel, truth_channel, truth_pas, truth_pdp
    )
    source_args = argparse.Namespace(
        batch_size=args.batch_size, force_incoherent=False
    )
    sources = load_sources(rows, global_ids, source_args, device)
    consensus = build_consensus(base_amplitude, sources, (0, 1, 2, 3))
    feature = reliability_features(base_amplitude, sources)

    specs = candidate_grid()
    metrics = {}
    rows_out = []
    for index, spec in enumerate(specs):
        gate = (feature[spec.gate] <= spec.quantile).float()
        amplitude = (
            base_amplitude
            + spec.alpha * gate[..., None] * (consensus - base_amplitude)
        ).clamp_min(0.0)
        desired_slices = amplitude.square()
        desired_slices /= desired_slices.sum(dim=3, keepdim=True).clamp_min(TINY)
        desired_pas = pas_from_slices(desired_slices)
        with torch.inference_mode():
            candidate = project_joint(
                base_channel, desired_pas, base_pdp, spec.cycles, spec.order
            )
            values = point_metrics(
                candidate, truth_channel, truth_pas, truth_pdp
            )
        metrics[spec.name] = values
        rows_out.append(
            {
                **asdict(spec),
                "active_fraction": float(gate.mean().item()),
                **metric_summary(values, baseline_metrics, fold_id),
            }
        )
        if (index + 1) % 20 == 0 or index + 1 == len(specs):
            print(f"reliable sqrt PAS {index + 1}/{len(specs)}", flush=True)

    rows_out.sort(key=lambda row: float(row["selected"]["score"]), reverse=True)
    deployable = [row for row in rows_out if int(row["positive_folds"]) == 5]
    selected = deployable[0] if deployable else rows_out[0]
    nested = nested_selection(specs, metrics, baseline_metrics, fold_id)

    selected_spec = next(spec for spec in specs if spec.name == selected["name"])
    selected_gate = (feature[selected_spec.gate] <= selected_spec.quantile).float()
    power = (
        (1.0 - selected_spec.alpha * selected_gate[..., None])
        * base_amplitude.square()
        + selected_spec.alpha
        * selected_gate[..., None]
        * consensus.clamp_min(0.0).square()
    )
    power /= power.sum(dim=3, keepdim=True).clamp_min(TINY)
    with torch.inference_mode():
        power_channel = project_joint(
            base_channel,
            pas_from_slices(power),
            base_pdp,
            selected_spec.cycles,
            selected_spec.order,
        )
        power_values = point_metrics(
            power_channel, truth_channel, truth_pas, truth_pdp
        )
    power_control = metric_summary(
        power_values, baseline_metrics, fold_id
    )
    passed = bool(
        float(selected["delta"]) >= 8e-4
        and int(selected["positive_folds"]) == 5
        and float(selected["worst_fold_delta"]) >= 2e-4
        and float(nested["delta"]) >= 5e-4
        and int(nested["positive_folds"]) >= 4
        and float(selected["delta"]) > float(power_control["delta"]) + 5e-5
    )
    payload = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Only absorb the square-root PAS correction where four independent "
            "Sionna endpoints agree and the requested movement is low-risk."
        ),
        "point_count": int(len(rows)),
        "candidate_count": int(len(specs)),
        "selected": selected,
        "nested_selection": nested,
        "power_geometry_control": power_control,
        "top_candidates": rows_out[:30],
        "deployment_gate": {
            "minimum_selected_delta": 8e-4,
            "minimum_selected_positive_folds": 5,
            "minimum_worst_fold_delta": 2e-4,
            "minimum_nested_delta": 5e-4,
            "minimum_nested_positive_folds": 4,
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
                "selected": selected,
                "nested_selection": nested,
                "power_geometry_control": power_control,
                "deployment_gate": payload["deployment_gate"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

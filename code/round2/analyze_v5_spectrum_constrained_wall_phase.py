#!/usr/bin/env python3
"""Reproject deployable wall-phase actions to the validated V4 spectra."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from analyze_v4_exact_joint_projection import (
    exact_pas_projection,
    exact_pdp_projection,
    pas_distribution,
    pdp_distribution,
)
from analyze_v5_deployable_wall_phase_transport import (
    PHASE_CACHE,
    TRAIN_CHANNEL,
    V4_OOF_CHANNEL,
    gate_masks,
    mix_full_phase,
)
from analyze_v5_path_relative_port_phase import (
    fixed_coordinate_decomposition,
    ports_to_channel,
    preserve_power,
)
from official_metrics import point_metric_statistics
from project_paths import PATHS
from run_rbf_spectral_field import statistics


VERSION = "Round2-E32-spectrum-constrained-wall-phase"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v5_spectrum_constrained_wall_phase"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_v5_spectrum_constrained_wall_phase"
)
METRIC_CACHE = ARTIFACT_DIR / "oof_spectrum_constrained_wall_phase_metrics.npz"
REPORT = ANALYSIS_DIR / "spectrum_constrained_wall_phase.json"


@dataclass(frozen=True)
class Config:
    phase_alpha: float = 0.10
    decomposition_ridge: float = 0.003
    orders: tuple[str, ...] = ("pas_pdp", "pdp_pas")
    checkpoints: tuple[int, ...] = (1, 2, 4)


def candidate_names(gates: list[str], config: Config) -> list[str]:
    return ["identity"] + [
        f"{gate}_{order}_c{cycle}"
        for gate in gates
        for order in config.orders
        for cycle in config.checkpoints
    ]


def projection_cycle(
    channel: torch.Tensor,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
    reference: torch.Tensor,
    order: str,
) -> torch.Tensor:
    if order == "pas_pdp":
        channel = exact_pas_projection(channel, target_pas)
        channel = exact_pdp_projection(channel, target_pdp)
    elif order == "pdp_pas":
        channel = exact_pdp_projection(channel, target_pdp)
        channel = exact_pas_projection(channel, target_pas)
    else:
        raise ValueError(order)
    return preserve_power(channel, reference)


@torch.inference_mode()
def evaluate(
    arrays: dict[str, np.ndarray],
    config: Config,
    device: torch.device,
    batch_size: int,
    rebuild: bool,
) -> dict[str, object]:
    gates, masks = gate_masks(arrays)
    names = candidate_names(gates, config)
    if METRIC_CACHE.is_file() and not rebuild:
        cached = np.load(METRIC_CACHE)
        metric = {key: cached[key] for key in ("pas", "pdp", "error", "energy")}
        return summarize(names, metric, arrays)

    query_ids = arrays["query_ids"].astype(np.int64)
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    reference_all = np.load(V4_OOF_CHANNEL, mmap_mode="r")
    metric = {
        key: np.zeros((len(names), len(query_ids)), dtype=np.float64)
        for key in ("pas", "pdp", "error", "energy")
    }
    index_by_name = {name: index for index, name in enumerate(names)}
    maximum_cycle = max(config.checkpoints)

    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        reference = torch.as_tensor(
            np.array(reference_all[start:end], dtype=np.complex64, copy=True),
            device=device,
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        target_pas = pas_distribution(reference)
        target_pdp = pdp_distribution(reference)
        coordinates = arrays["coordinates"][start:end]
        base_coefficient, residual, atoms = fixed_coordinate_decomposition(
            reference, coordinates, config.decomposition_ridge, device
        )
        source = torch.as_tensor(arrays["source"][start:end], device=device)
        variants: dict[str, torch.Tensor] = {"identity": reference}
        for gate_index, gate in enumerate(gates):
            gate_mask = torch.as_tensor(masks[gate_index, start:end], device=device)
            coefficient = mix_full_phase(
                base_coefficient, source, gate_mask, config.phase_alpha
            )
            ports = residual + torch.einsum(
                "bkc,bkhvs->bchvs", coefficient, atoms
            )
            phase_channel = preserve_power(ports_to_channel(ports), reference)
            for order in config.orders:
                candidate = phase_channel
                for cycle in range(1, maximum_cycle + 1):
                    candidate = projection_cycle(
                        candidate, target_pas, target_pdp, reference, order
                    )
                    if cycle in config.checkpoints:
                        variants[f"{gate}_{order}_c{cycle}"] = candidate

        truth_numpy = truth.cpu().numpy()
        for name, candidate in variants.items():
            point = point_metric_statistics(candidate.cpu().numpy(), truth_numpy)
            index = index_by_name[name]
            metric["pas"][index, start:end] = point.pas
            metric["pdp"][index, start:end] = point.pdp
            metric["error"][index, start:end] = point.squared_error
            metric["energy"][index, start:end] = point.truth_energy
        if end % 20 == 0 or end == len(query_ids):
            print(f"spectrum-constrained wall phase {end}/{len(query_ids)}", flush=True)

    np.savez_compressed(
        METRIC_CACHE,
        **metric,
        variant_names=np.asarray(names),
        fold_id=arrays["fold_id"],
        site=arrays["site"],
    )
    return summarize(names, metric, arrays)


def summarize(
    names: list[str],
    metric: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
) -> dict[str, object]:
    fold_id = arrays["fold_id"]
    site = arrays["site"]
    candidate_rows = []
    for index, name in enumerate(names):
        overall = statistics(
            metric["pas"][index],
            metric["pdp"][index],
            metric["error"][index],
            metric["energy"][index],
        )
        candidate_rows.append(
            {
                "name": name,
                **overall,
                "fold_score": [
                    statistics(
                        metric["pas"][index],
                        metric["pdp"][index],
                        metric["error"][index],
                        metric["energy"][index],
                        fold_id == fold,
                    )["score"]
                    for fold in range(5)
                ],
            }
        )
    candidate_rows.sort(key=lambda row: float(row["score"]), reverse=True)

    selected = np.zeros(len(fold_id), dtype=np.int64)
    decisions = []
    for held_fold in range(5):
        for site_value in range(2):
            train = (fold_id != held_fold) & (site == site_value)
            held = (fold_id == held_fold) & (site == site_value)
            scores = np.asarray(
                [
                    statistics(
                        metric["pas"][index],
                        metric["pdp"][index],
                        metric["error"][index],
                        metric["energy"][index],
                        train,
                    )["score"]
                    for index in range(len(names))
                ]
            )
            chosen = int(np.argmax(scores))
            selected[held] = chosen
            decisions.append(
                {
                    "held_fold": held_fold,
                    "site": site_value,
                    "candidate": names[chosen],
                    "training_score": float(scores[chosen]),
                }
            )
    point = np.arange(len(fold_id))
    nested = statistics(
        metric["pas"][selected, point],
        metric["pdp"][selected, point],
        metric["error"][selected, point],
        metric["energy"][selected, point],
    )
    baseline = statistics(
        metric["pas"][0], metric["pdp"][0], metric["error"][0], metric["energy"][0]
    )
    nested["delta_vs_identity"] = float(nested["score"] - baseline["score"])
    fold_delta = []
    for fold in range(5):
        mask = fold_id == fold
        current = statistics(
            metric["pas"][selected, point],
            metric["pdp"][selected, point],
            metric["error"][selected, point],
            metric["energy"][selected, point],
            mask,
        )
        old = statistics(
            metric["pas"][0],
            metric["pdp"][0],
            metric["error"][0],
            metric["energy"][0],
            mask,
        )
        fold_delta.append(float(current["score"] - old["score"]))
    nested["fold_delta_vs_identity"] = fold_delta
    nested["positive_fold_count"] = int(np.sum(np.asarray(fold_delta) > 0.0))
    nested["decisions"] = decisions
    return {
        "baseline": baseline,
        "top_candidates": candidate_rows[:20],
        "nested_selection": nested,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in np.load(PHASE_CACHE).items()}
    evaluation = evaluate(
        arrays,
        config,
        torch.device(args.device),
        args.batch_size,
        args.rebuild,
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_only",
        "primary_hypothesis": (
            "Alternating exact projections to V4's own PAS/PDP distributions preserve "
            "deployable wall-phase NMSE gains while removing E31's spectral damage."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "config": asdict(config),
        "evaluation": evaluation,
        "outputs": {"metrics": PATHS.relative(METRIC_CACHE)},
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

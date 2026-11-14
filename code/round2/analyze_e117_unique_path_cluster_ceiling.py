#!/usr/bin/env python3
"""Audit unique-path clustering before replacing E115 token-level outputs."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import analyze_global_local_path_lifecycle as e79
import train_e115_complete_path_state_operator as e115
from project_paths import PATHS


VERSION = "Round2-E117-unique-path-cluster-ceiling"
REPORT = (
    PATHS.analysis
    / "results"
    / "round2_unique_path_cluster_ceiling"
    / "unique_path_cluster_ceiling.json"
)
PERIODS = e115.PERIODS.astype(np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-clusters", type=int, default=64)
    return parser.parse_args()


def circular_mean(coordinate: np.ndarray, weight: np.ndarray) -> np.ndarray:
    total = max(float(np.sum(weight, dtype=np.float64)), e115.TINY)
    angle = 2.0 * np.pi * coordinate.astype(np.float64) / PERIODS
    sine = np.sum(weight[:, None] * np.sin(angle), axis=0, dtype=np.float64) / total
    cosine = np.sum(weight[:, None] * np.cos(angle), axis=0, dtype=np.float64) / total
    output = np.mod(np.arctan2(sine, cosine), 2.0 * np.pi)
    return (output * PERIODS / (2.0 * np.pi)).astype(np.float32)


def cluster_point(
    coordinate: np.ndarray,
    base_mass: np.ndarray,
    base_port: np.ndarray,
    target_mass: np.ndarray,
    target_delta: np.ndarray,
    target_port: np.ndarray,
    threshold: float,
    max_clusters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    order = np.argsort(-base_mass, kind="stable")
    cost = e79.support_cost(coordinate, coordinate)
    assigned = np.zeros(len(coordinate), dtype=bool)
    groups: list[np.ndarray] = []
    for leader in order:
        if assigned[leader]:
            continue
        group = np.flatnonzero((cost[leader] <= threshold) & ~assigned)
        assigned[group] = True
        groups.append(group)
    group_mass = np.asarray(
        [np.sum(base_mass[group], dtype=np.float64) for group in groups]
    )
    selected = np.argsort(-group_mass, kind="stable")[:max_clusters]
    cluster_count = len(groups)

    output_coordinate = np.zeros((max_clusters, 3), np.float32)
    output_base_mass = np.zeros(max_clusters, np.float32)
    output_base_port = np.full(
        (max_clusters, e115.PORTS), 1.0 / e115.PORTS, np.float32
    )
    output_target_mass = np.zeros(max_clusters, np.float32)
    output_target_coordinate = np.zeros((max_clusters, 3), np.float32)
    output_target_port = np.full(
        (max_clusters, e115.PORTS), 1.0 / e115.PORTS, np.float32
    )
    corrected = np.mod(coordinate + target_delta, e115.PERIODS)
    for output_index, group_index in enumerate(selected):
        group = groups[int(group_index)]
        local_base_mass = base_mass[group]
        local_target_mass = target_mass[group]
        base_total = float(np.sum(local_base_mass, dtype=np.float64))
        target_total = float(np.sum(local_target_mass, dtype=np.float64))
        output_coordinate[output_index] = circular_mean(
            coordinate[group], local_base_mass
        )
        output_base_mass[output_index] = base_total
        output_base_port[output_index] = np.sum(
            local_base_mass[:, None] * base_port[group], axis=0, dtype=np.float64
        ) / max(base_total, e115.TINY)
        output_target_mass[output_index] = target_total
        if target_total > 0.0:
            output_target_coordinate[output_index] = circular_mean(
                corrected[group], local_target_mass
            )
            output_target_port[output_index] = np.sum(
                local_target_mass[:, None] * target_port[group],
                axis=0,
                dtype=np.float64,
            ) / max(target_total, e115.TINY)
        else:
            output_target_coordinate[output_index] = output_coordinate[output_index]
            output_target_port[output_index] = output_base_port[output_index]
    return (
        output_coordinate,
        output_base_mass,
        output_base_port,
        output_target_mass,
        output_target_coordinate,
        output_target_port,
        cluster_count,
    )


@torch.inference_mode()
def endpoint_scores(
    coordinate: np.ndarray,
    mass: np.ndarray,
    port: np.ndarray,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    pas_parts = []
    pdp_parts = []
    for start in range(0, len(coordinate), batch_size):
        end = min(start + batch_size, len(coordinate))
        local_mass = torch.as_tensor(mass[start:end], device=device)
        output = {
            "mass": local_mass
            / local_mass.sum(dim=1, keepdim=True).clamp_min(e115.TINY),
            "coordinate": torch.as_tensor(coordinate[start:end], device=device),
            "port": torch.as_tensor(port[start:end], device=device),
        }
        batch = {
            "target_pas": target_pas[start:end],
            "target_pdp": target_pdp[start:end],
        }
        pas, pdp, _, _ = e115.sufficient_scores(output, batch)
        pas_parts.append(pas.cpu().numpy())
        pdp_parts.append(pdp.cpu().numpy())
    pas = np.concatenate(pas_parts)
    pdp = np.concatenate(pdp_parts)
    return {
        "pas": float(np.mean(pas)),
        "pdp": float(np.mean(pdp)),
        "pas_pdp_mean": float(np.mean(0.5 * (pas + pdp))),
    }


def audit_threshold(
    rows: np.ndarray,
    coordinate: np.ndarray,
    base_mass: np.ndarray,
    base_port: np.ndarray,
    target_mass: np.ndarray,
    target_delta: np.ndarray,
    target_port: np.ndarray,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
    threshold: float,
    max_clusters: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    count = len(rows)
    cluster_coordinate = np.zeros((count, max_clusters, 3), np.float32)
    cluster_base_mass = np.zeros((count, max_clusters), np.float32)
    cluster_base_port = np.zeros((count, max_clusters, e115.PORTS), np.float32)
    cluster_target_mass = np.zeros((count, max_clusters), np.float32)
    cluster_target_coordinate = np.zeros((count, max_clusters, 3), np.float32)
    cluster_target_port = np.zeros((count, max_clusters, e115.PORTS), np.float32)
    cluster_counts = np.zeros(count, np.int32)
    for local_row, point_id in enumerate(rows):
        values = cluster_point(
            coordinate[point_id],
            base_mass[point_id],
            base_port[point_id],
            target_mass[point_id],
            target_delta[point_id],
            target_port[point_id],
            threshold,
            max_clusters,
        )
        cluster_coordinate[local_row] = values[0]
        cluster_base_mass[local_row] = values[1]
        cluster_base_port[local_row] = values[2]
        cluster_target_mass[local_row] = values[3]
        cluster_target_coordinate[local_row] = values[4]
        cluster_target_port[local_row] = values[5]
        cluster_counts[local_row] = values[6]
    normalized_target = cluster_target_mass / np.maximum(
        cluster_target_mass.sum(axis=1, keepdims=True), e115.TINY
    )
    ordered = np.sort(normalized_target, axis=1)[:, ::-1]
    entropy = -np.sum(
        np.where(
            normalized_target > 0.0,
            normalized_target * np.log(np.maximum(normalized_target, 1e-30)),
            0.0,
        ),
        axis=1,
    )
    base_score = endpoint_scores(
        cluster_coordinate,
        cluster_base_mass,
        cluster_base_port,
        target_pas,
        target_pdp,
        device,
        batch_size,
    )
    mass_score = endpoint_scores(
        cluster_coordinate,
        cluster_target_mass,
        cluster_base_port,
        target_pas,
        target_pdp,
        device,
        batch_size,
    )
    oracle_score = endpoint_scores(
        cluster_target_coordinate,
        cluster_target_mass,
        cluster_target_port,
        target_pas,
        target_pdp,
        device,
        batch_size,
    )
    return {
        "threshold": threshold,
        "max_clusters": max_clusters,
        "cluster_count_median": float(np.median(cluster_counts)),
        "cluster_count_p90": float(np.quantile(cluster_counts, 0.90)),
        "truncated_point_fraction": float(np.mean(cluster_counts > max_clusters)),
        "positive_cluster_count_median": float(
            np.median((cluster_target_mass > 0.0).sum(axis=1))
        ),
        "effective_target_count_median": float(np.median(np.exp(entropy))),
        "top24_target_mass_median": float(np.median(ordered[:, :24].sum(axis=1))),
        "retained_target_mass_median": float(
            np.median(cluster_target_mass.sum(axis=1) / np.maximum(target_mass[rows].sum(axis=1), e115.TINY))
        ),
        "base": base_score,
        "mass_oracle": mass_score,
        "complete_state_oracle": oracle_score,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    directory = e115.cache_dir("screen_block01")
    coordinate = np.asarray(
        np.load(e115.array_path(directory, "coordinate"), mmap_mode="r"),
        dtype=np.float32,
    )
    base_log_mass = np.asarray(
        np.load(e115.array_path(directory, "base_log_mass"), mmap_mode="r"),
        dtype=np.float32,
    )
    base_mass = np.exp(base_log_mass)
    base_port = torch.softmax(
        torch.as_tensor(
            np.asarray(
                np.load(e115.array_path(directory, "base_log_port"), mmap_mode="r")
            ),
            device=device,
        ),
        dim=2,
    ).cpu().numpy()
    target_mass = np.asarray(
        np.load(e115.array_path(directory, "target_mass"), mmap_mode="r"),
        dtype=np.float32,
    )
    target_delta = np.asarray(
        np.load(e115.array_path(directory, "target_delta"), mmap_mode="r"),
        dtype=np.float32,
    )
    target_port = np.asarray(
        np.load(e115.array_path(directory, "target_port"), mmap_mode="r"),
        dtype=np.float32,
    )
    fold_id = np.load(e115.array_path(directory, "fold_id"))
    point_power = np.load(e115.array_path(directory, "point_power"))
    target_pas_all = torch.as_tensor(np.load(e115.TARGET_PAS, mmap_mode="r"), device=device)
    target_pdp_all = torch.as_tensor(np.load(e115.TARGET_PDP, mmap_mode="r"), device=device)
    splits: dict[str, object] = {}
    for split_name, fold in (("validation_fold1", 1), ("held_fold0", 0)):
        rows = np.flatnonzero((fold_id == fold) & (point_power > 0.0))
        split_target_pas = target_pas_all[torch.as_tensor(rows, device=device)]
        split_target_pdp = target_pdp_all[torch.as_tensor(rows, device=device)]
        values = [
            audit_threshold(
                rows,
                coordinate,
                base_mass,
                base_port,
                target_mass,
                target_delta,
                target_port,
                split_target_pas,
                split_target_pdp,
                threshold,
                args.max_clusters,
                device,
                args.batch_size,
            )
            for threshold in (0.05, 0.10, 0.15, 0.20, 0.25)
        ]
        splits[split_name] = {
            "count": int(len(rows)),
            "thresholds": values,
            "selected_threshold_by_oracle_retention": max(
                values,
                key=lambda row: (
                    float(row["top24_target_mass_median"]),
                    float(row["complete_state_oracle"]["pas_pdp_mean"]),
                ),
            )["threshold"],
        }
        print(f"finished split={split_name}", flush=True)
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The 176 raw tokens contain sufficient evidence but duplicate physical "
            "paths, causing diffuse mass labels and an inconsistent top-24 complex "
            "renderer. Unique-path clusters should preserve the state oracle while "
            "concentrating target mass into deployable output paths."
        ),
        "official_h_only": True,
        "sionna_used": False,
        "max_clusters": args.max_clusters,
        "splits": splits,
        "seconds": float(time.time() - started),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

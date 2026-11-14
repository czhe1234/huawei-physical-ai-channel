#!/usr/bin/env python3
"""Build a strict official-H complex stack and add frozen unique-path states."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import analyze_e86_surface_endpoint_v11_blend as e86
import analyze_e121_standalone_unique_path_carrier as e121
import analyze_v7_complex_carrier_stacking as stack
import train_e115_complete_path_state_operator as e115
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from build_v7_complex_carrier_bank import carrier_configs
from carrier_calibration import estimate_carrier
from complex_baselines import predict_one
from data import load_positions
from project_paths import PATHS
from run_v9_joint_sionna_migration import point_metrics
from run_rbf_spectral_field import statistics
from train_full_path_pas_slice_gate import project_joint


VERSION = "Round2-E122-strict-complex-stack-unique-path"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "strict_complex_stack_unique_path"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_strict_complex_stack_unique_path"
STACKED_CHANNEL = ARTIFACT_DIR / "strict_stacked_official_channel.npy"
PRIMARY_CHANNEL = ARTIFACT_DIR / "strict_stacked_unique_path_channel.npy"
PROGRESS = ARTIFACT_DIR / "build_progress.json"
REPORT = ANALYSIS_DIR / "strict_complex_stack_unique_path.json"
ENDPOINT = e121.ENDPOINT
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"


@dataclass(frozen=True)
class Action:
    mode: str
    pas_beta: float
    pdp_beta: float
    order: str
    cycles: int

    @property
    def name(self) -> str:
        return (
            f"{self.mode}_p{int(round(1000 * self.pas_beta)):03d}_"
            f"d{int(round(1000 * self.pdp_beta)):03d}_{self.order}_c{self.cycles}"
        )


PRIMARY_ACTION = Action("sqrt", 0.10, 0.10, "pas_pdp", 4)
PAS_CONTROL = Action("sqrt", 0.10, 0.0, "pas_pdp", 4)
PDP_CONTROL = Action("sqrt", 0.0, 0.10, "pas_pdp", 4)
STACK_CONFIG = stack.StackConfig("site", "port8", 1.0, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--fit-samples-per-site", type=int, default=320)
    parser.add_argument("--carrier-pairs", type=int, default=500)
    parser.add_argument("--gram-batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def farthest_sample(
    ids: np.ndarray,
    positions: np.ndarray,
    count: int,
) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) <= count:
        return ids
    xy = positions[ids, :2].astype(np.float64)
    center = xy.mean(axis=0)
    first = int(np.argmin(np.sum(np.square(xy - center), axis=1)))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = first
    minimum = np.sum(np.square(xy - xy[first]), axis=1)
    minimum[first] = -1.0
    for index in range(1, count):
        next_row = int(np.argmax(minimum))
        selected[index] = next_row
        distance = np.sum(np.square(xy - xy[next_row]), axis=1)
        minimum = np.minimum(minimum, distance)
        minimum[selected[: index + 1]] = -1.0
    return ids[selected]


def neighbor_table(
    fit: np.ndarray,
    query_ids: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    distance = np.empty((len(query_ids), max_k), dtype=np.float64)
    neighbors = np.empty((len(query_ids), max_k), dtype=np.int64)
    for site_value in np.unique(site[query_ids]):
        rows = np.flatnonzero(site[query_ids] == site_value)
        bank = fit[site[fit] == site_value]
        take = min(max_k + 1, len(bank))
        local_distance, local = cKDTree(positions[bank, :2]).query(
            positions[query_ids[rows], :2], k=take, workers=-1
        )
        local_distance = np.asarray(local_distance)
        local = np.asarray(local)
        if take == 1:
            local_distance = local_distance[:, None]
            local = local[:, None]
        local_ids = bank[local.astype(np.int64)]
        for local_row, destination in enumerate(rows):
            query_id = query_ids[destination]
            keep = local_ids[local_row] != query_id
            selected_ids = local_ids[local_row, keep][:max_k]
            selected_distance = local_distance[local_row, keep][:max_k]
            if len(selected_ids) < max_k:
                if len(selected_ids) == 0:
                    raise RuntimeError(f"no donors for query {int(query_id)}")
                pad = max_k - len(selected_ids)
                selected_ids = np.pad(selected_ids, (0, pad), mode="edge")
                selected_distance = np.pad(
                    selected_distance, (0, pad), mode="edge"
                )
            neighbors[destination] = selected_ids
            distance[destination] = selected_distance
    return distance, neighbors


def generate_carriers(
    query_id: int,
    donor_ids: np.ndarray,
    distance: np.ndarray,
    templates,
    carrier,
    train_h: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
) -> np.ndarray:
    site_value = int(site[query_id])
    donor_h = np.asarray(train_h[donor_ids], dtype=np.complex64)
    output = []
    for template in templates:
        config = replace(
            template,
            beta=carrier[site_value].beta,
            gamma=carrier[site_value].gamma,
        )
        prediction, _ = predict_one(
            donor_h,
            positions[donor_ids],
            positions[query_id],
            distance,
            bs_positions[site_value],
            config,
        )
        output.append(prediction)
    return np.stack(output).astype(np.complex64)


def gram_block(
    candidates: np.ndarray,
    truth: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    batch, carriers = candidates.shape[:2]
    candidate = torch.as_tensor(candidates, device=device).reshape(
        batch, carriers, 2, 16, 8, 4, 192
    )
    candidate = candidate.permute(0, 2, 5, 1, 3, 4, 6).reshape(
        batch, 8, carriers, -1
    )
    target = torch.as_tensor(truth, device=device).reshape(
        batch, 2, 16, 8, 4, 192
    )
    target = target.permute(0, 1, 4, 2, 3, 5).reshape(batch, 8, -1)
    with torch.inference_mode():
        gram = torch.einsum("bgcd,bged->bgce", candidate.conj(), candidate)
        cross = torch.einsum("bgcd,bgd->bgc", candidate.conj(), target)
    return gram.cpu().numpy(), cross.cpu().numpy()


def combine_carriers(candidates: np.ndarray, weights: np.ndarray) -> np.ndarray:
    carriers = len(candidates)
    candidate = (
        candidates.reshape(carriers, 2, 16, 8, 4, 192)
        .transpose(1, 4, 0, 2, 3, 5)
        .reshape(8, carriers, 16, 8, 192)
    )
    combined = np.einsum("gcvhd,gc->gvhd", candidate, weights, optimize=True)
    return (
        combined.reshape(2, 4, 16, 8, 192)
        .transpose(0, 2, 3, 1, 4)
        .reshape(256, 4, 192)
        .astype(np.complex64)
    )


def fit_fold_weights(
    outer: int,
    fit: np.ndarray,
    templates,
    carrier,
    train_h: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    samples_per_site: int,
    gram_batch_size: int,
    device: torch.device,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    selected_parts = []
    for site_value in np.unique(site):
        local = fit[site[fit] == site_value]
        selected_parts.append(
            farthest_sample(local, positions, min(samples_per_site, len(local)))
        )
    selected = np.concatenate(selected_parts)
    max_k = max(config.k for config in templates)
    distance, neighbors = neighbor_table(
        fit, selected, positions, site, max_k
    )
    carriers = len(templates)
    gram = np.zeros((2, 8, carriers, carriers), dtype=np.complex128)
    cross = np.zeros((2, 8, carriers), dtype=np.complex128)
    for start in range(0, len(selected), gram_batch_size):
        stop = min(start + gram_batch_size, len(selected))
        candidate = np.stack(
            [
                generate_carriers(
                    int(selected[row]),
                    neighbors[row],
                    distance[row],
                    templates,
                    carrier,
                    train_h,
                    positions,
                    site,
                    bs_positions,
                )
                for row in range(start, stop)
            ]
        )
        target = np.asarray(train_h[selected[start:stop]], dtype=np.complex64)
        local_gram, local_cross = gram_block(candidate, target, device)
        local_site = site[selected[start:stop]]
        for site_value in np.unique(local_site):
            mask = local_site == site_value
            gram[int(site_value)] += local_gram[mask].sum(
                axis=0, dtype=np.complex128
            )
            cross[int(site_value)] += local_cross[mask].sum(
                axis=0, dtype=np.complex128
            )
        if stop % 40 == 0 or stop == len(selected):
            print(
                f"outer={outer} Gram {stop}/{len(selected)}",
                flush=True,
            )
    weights: dict[int, np.ndarray] = {}
    for site_value in range(2):
        weights[site_value] = np.stack(
            [
                stack._solve_weights(
                    gram[site_value, port],
                    cross[site_value, port],
                    STACK_CONFIG,
                )
                for port in range(8)
            ]
        )
    return weights, {
        "selected_count": int(len(selected)),
        "selected_site_count": {
            str(site_value): int(np.sum(site[selected] == site_value))
            for site_value in np.unique(site)
        },
        "selected_ids": selected.tolist(),
    }


def save_weights(path: Path, weights: dict[int, np.ndarray]) -> None:
    np.savez_compressed(
        path,
        site0=weights[0].astype(np.complex64),
        site1=weights[1].astype(np.complex64),
    )


def build_strict_stack(args: argparse.Namespace, device: torch.device):
    endpoint = np.load(ENDPOINT, allow_pickle=False)
    query_ids = endpoint["query_ids"].astype(np.int64)
    full_fold = endpoint["full_fold"].astype(np.int8)
    expected = (len(query_ids), 256, 4, 192)
    if args.force or not PROGRESS.is_file():
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        output = np.lib.format.open_memmap(
            STACKED_CHANNEL, mode="w+", dtype=np.complex64, shape=expected
        )
        output[:] = 0.0
        output.flush()
        progress: dict[str, object] = {"completed": [], "folds": {}}
    else:
        progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
        output = np.lib.format.open_memmap(STACKED_CHANNEL, mode="r+")
        if output.shape != expected or output.dtype != np.complex64:
            raise RuntimeError("stale strict stack output; rerun with --force")

    setup, positions, _, site, _ = load_positions()
    train_h = np.load(TRAIN_CHANNEL, mmap_mode="r")
    all_fold = np.load(e115.FULL_FOLD).astype(np.int8)
    point_power = np.load(e115.COMPONENTS, allow_pickle=False)["point_power"].astype(
        np.float64
    )
    templates = carrier_configs()
    max_k = max(config.k for config in templates)
    completed = {int(value) for value in progress["completed"]}
    for outer in range(5):
        if outer in completed:
            print(f"outer={outer} reused", flush=True)
            continue
        inner = (outer + 1) % 5
        fit = np.flatnonzero((all_fold != outer) & (all_fold != inner))
        carrier = estimate_carrier(
            train_h,
            positions,
            site,
            setup.bs_positions,
            fit,
            point_power,
            max_pairs_per_site=args.carrier_pairs,
            seed=20_260_808 + 101 * outer,
        )
        weights, fit_record = fit_fold_weights(
            outer,
            fit,
            templates,
            carrier,
            train_h,
            positions,
            site,
            setup.bs_positions,
            args.fit_samples_per_site,
            args.gram_batch_size,
            device,
        )
        save_weights(ARTIFACT_DIR / f"outer{outer}_weights.npz", weights)
        rows = np.flatnonzero(full_fold == outer)
        held_ids = query_ids[rows]
        distance, neighbors = neighbor_table(
            fit, held_ids, positions, site, max_k
        )
        for local_row, destination in enumerate(rows):
            query_id = int(held_ids[local_row])
            candidates = generate_carriers(
                query_id,
                neighbors[local_row],
                distance[local_row],
                templates,
                carrier,
                train_h,
                positions,
                site,
                setup.bs_positions,
            )
            output[destination] = combine_carriers(
                candidates, weights[int(site[query_id])]
            )
            if (local_row + 1) % 20 == 0 or local_row + 1 == len(rows):
                output.flush()
                print(
                    f"outer={outer} held {local_row + 1}/{len(rows)}",
                    flush=True,
                )
        progress["folds"][str(outer)] = {
            "inner": int(inner),
            "fit_count": int(len(fit)),
            "held_count": int(len(rows)),
            "fit_sampling": fit_record,
            "carrier": {
                str(site_value): value.to_dict()
                for site_value, value in carrier.items()
            },
            "weights": PATHS.relative(ARTIFACT_DIR / f"outer{outer}_weights.npz"),
        }
        completed.add(outer)
        progress["completed"] = sorted(completed)
        PROGRESS.write_text(
            json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    output.flush()
    return np.load(STACKED_CHANNEL, mmap_mode="r"), progress


def expand_endpoint(
    pas: np.ndarray,
    pdp: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pas_tensor = torch.as_tensor(pas, device=device)
    pas_tensor = pas_tensor[..., None].expand(-1, -1, -1, -1, -1, 192)
    pdp_tensor = torch.as_tensor(pdp, device=device)
    pdp_tensor = pdp_tensor[:, :, None, None, :, :].expand(
        -1, 2, 16, 8, 4, 192
    )
    return pas_tensor, pdp_tensor.reshape(-1, 256, 4, 192)


def make_candidate(
    base: torch.Tensor,
    base_pas: torch.Tensor,
    base_pdp: torch.Tensor,
    endpoint_pas: torch.Tensor,
    endpoint_pdp: torch.Tensor,
    action: Action,
) -> torch.Tensor:
    desired_pas = e86.endpoint_target(
        base_pas, endpoint_pas, action.pas_beta, action.mode, (1, 2, 3)
    )
    desired_pdp = e86.endpoint_target(
        base_pdp, endpoint_pdp, action.pdp_beta, action.mode, (3,)
    )
    return project_joint(
        base, desired_pas, desired_pdp, action.cycles, action.order
    )


def metric_rows(
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    full_fold: np.ndarray,
) -> list[dict[str, object]]:
    output = []
    for fold in range(5):
        mask = full_fold == fold
        selected = statistics(*values, mask)
        anchor = statistics(*baseline, mask)
        output.append(
            {
                "fold": fold,
                "count": int(np.sum(mask)),
                "selected": selected,
                "baseline": anchor,
                "delta": float(selected["score"] - anchor["score"]),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    endpoint = np.load(ENDPOINT, allow_pickle=False)
    query_ids = endpoint["query_ids"].astype(np.int64)
    full_fold = endpoint["full_fold"].astype(np.int8)
    endpoint_pas = endpoint["pas"].astype(np.float32)
    endpoint_pdp = endpoint["pdp"].astype(np.float32)
    stacked_np, progress = build_strict_stack(args, device)
    train_h = np.load(TRAIN_CHANNEL, mmap_mode="r")
    base = torch.as_tensor(
        np.array(stacked_np, dtype=np.complex64, copy=True), device=device
    )
    truth = torch.as_tensor(
        np.array(train_h[query_ids], dtype=np.complex64, copy=True), device=device
    )
    truth_pas = pas_distribution(truth)
    truth_pdp = pdp_distribution(truth)
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    expanded = expand_endpoint(endpoint_pas, endpoint_pdp, device)
    baseline = point_metrics(base, truth, truth_pas, truth_pdp)
    baseline_summary = statistics(*baseline)
    action_rows = []
    primary_prediction = None
    for action in (PRIMARY_ACTION, PAS_CONTROL, PDP_CONTROL):
        prediction = make_candidate(base, base_pas, base_pdp, *expanded, action)
        values = point_metrics(prediction, truth, truth_pas, truth_pdp)
        selected = statistics(*values)
        folds = metric_rows(values, baseline, full_fold)
        action_rows.append(
            {
                "name": action.name,
                "action": asdict(action),
                "selected": selected,
                "baseline": baseline_summary,
                "delta": float(selected["score"] - baseline_summary["score"]),
                "positive_folds": int(
                    sum(item["delta"] > 0.0 for item in folds)
                ),
                "worst_fold_delta": float(min(item["delta"] for item in folds)),
                "folds": folds,
            }
        )
        if action == PRIMARY_ACTION:
            primary_prediction = prediction
    if primary_prediction is None:
        raise RuntimeError("primary prediction was not generated")
    np.save(PRIMARY_CHANNEL, primary_prediction.cpu().numpy().astype(np.complex64))
    primary = action_rows[0]
    gate = {
        "minimum_delta": 0.001,
        "minimum_positive_folds": 4,
        "minimum_worst_fold_delta": -0.001,
        "delta": primary["delta"],
        "positive_folds": primary["positive_folds"],
        "worst_fold_delta": primary["worst_fold_delta"],
    }
    gate["passed"] = bool(
        gate["delta"] >= gate["minimum_delta"]
        and gate["positive_folds"] >= gate["minimum_positive_folds"]
        and gate["worst_fold_delta"] >= gate["minimum_worst_fold_delta"]
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "A fit-only complex stack supplies coherent phase and port organization "
            "that composes with the already validated official-H unique-path states."
        ),
        "protocol": {
            "outer_folds": 5,
            "inner_rule": "(outer + 1) % 5",
            "all_donors_exclude_outer_and_inner": True,
            "fit_samples_per_site": int(args.fit_samples_per_site),
            "carrier_count": len(carrier_configs()),
            "stack_config": asdict(STACK_CONFIG),
            "path_action_frozen": asdict(PRIMARY_ACTION),
            "v12_used": False,
            "sionna_used": False,
        },
        "build": progress,
        "baseline": baseline_summary,
        "actions": action_rows,
        "primary": primary,
        "deployment_gate": gate,
        "comparison": {
            "e121_radial_base": 0.5292381279506954,
            "e121_radial_path": 0.5382320257827342,
            "stack_gain_over_e121_radial_base": float(
                baseline_summary["score"] - 0.5292381279506954
            ),
            "path_gain_change_vs_e121": float(
                primary["delta"] - 0.008993897832038855
            ),
        },
        "outputs": {
            "stacked_channel": PATHS.relative(STACKED_CHANNEL),
            "primary_channel": PATHS.relative(PRIMARY_CHANNEL),
            "endpoint": PATHS.relative(ENDPOINT),
        },
        "seconds": float(time.time() - started),
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "baseline": baseline_summary,
                "actions": action_rows,
                "gate": gate,
                "comparison": report["comparison"],
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test unique-path states on a strictly aligned official-H complex carrier."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import time

import numpy as np
import torch

import analyze_e119_v12_unique_path_complement as e119
import run_e120_fivefold_unique_path_v12 as e120
import train_e115_complete_path_state_operator as e115
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from build_v7_complex_carrier_bank import carrier_configs
from carrier_calibration import estimate_carrier
from complex_baselines import predict_one, query_site_neighbors
from data import load_positions
from project_paths import PATHS
from run_v9_joint_sionna_migration import point_metrics
from run_rbf_spectral_field import statistics


VERSION = "Round2-E121-standalone-unique-path-carrier"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "standalone_unique_path_carrier"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_standalone_unique_path_carrier"
BASE_CHANNEL = ARTIFACT_DIR / "strict_radial_idw_k24_channel.npy"
PRIMARY_CHANNEL = ARTIFACT_DIR / "strict_unique_path_projected_channel.npy"
REPORT = ANALYSIS_DIR / "standalone_unique_path_carrier.json"
ENDPOINT = (
    PATHS.artifacts
    / "round2"
    / "fivefold_unique_path_v12"
    / "strict_oof_mass_coordinate_endpoint.npz"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"

PRIMARY_ACTION = e119.Action(
    endpoint="scene_mass_coordinate",
    mode="sqrt",
    pas_beta=0.10,
    pdp_beta=0.10,
    order="pas_pdp",
    cycles=4,
)
PAS_CONTROL = e119.Action(
    endpoint="scene_mass_coordinate",
    mode="sqrt",
    pas_beta=0.10,
    pdp_beta=0.0,
    order="pas_pdp",
    cycles=4,
)
PDP_CONTROL = e119.Action(
    endpoint="scene_mass_coordinate",
    mode="sqrt",
    pas_beta=0.0,
    pdp_beta=0.10,
    order="pas_pdp",
    cycles=4,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--carrier-pairs", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def radial_idw_config():
    return next(config for config in carrier_configs() if config.name == "radial_idw_k24")


def build_strict_carrier(
    query_ids: np.ndarray,
    full_fold: np.ndarray,
    carrier_pairs: int,
    force: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    expected = (len(query_ids), 256, 4, 192)
    if BASE_CHANNEL.is_file() and not force:
        cached = np.load(BASE_CHANNEL, mmap_mode="r")
        if cached.shape == expected and cached.dtype == np.complex64:
            return cached, {"reused": True}

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    setup, positions, _, site, _ = load_positions()
    train_h = np.load(TRAIN_CHANNEL, mmap_mode="r")
    component = np.load(e115.COMPONENTS, allow_pickle=False)
    point_power = component["point_power"].astype(np.float64)
    all_fold = np.load(e115.FULL_FOLD).astype(np.int8)
    output = np.lib.format.open_memmap(
        BASE_CHANNEL, mode="w+", dtype=np.complex64, shape=expected
    )
    config_template = radial_idw_config()
    records: dict[str, object] = {}

    for outer in range(5):
        inner = (outer + 1) % 5
        fit = np.flatnonzero((all_fold != outer) & (all_fold != inner))
        rows = np.flatnonzero(full_fold == outer)
        fold_queries = query_ids[rows]
        carrier = estimate_carrier(
            train_h,
            positions,
            site,
            setup.bs_positions,
            fit,
            point_power,
            max_pairs_per_site=carrier_pairs,
            seed=20_260_808 + 101 * outer,
        )
        distance, neighbors = query_site_neighbors(
            positions, site, fit, fold_queries, max_k=config_template.k
        )
        for local_row, destination in enumerate(rows):
            query_id = int(fold_queries[local_row])
            donor_ids = neighbors[local_row]
            site_value = int(site[query_id])
            config = type(config_template)(
                **{
                    **asdict(config_template),
                    "beta": carrier[site_value].beta,
                    "gamma": carrier[site_value].gamma,
                }
            )
            prediction, _ = predict_one(
                np.asarray(train_h[donor_ids], dtype=np.complex64),
                positions[donor_ids],
                positions[query_id],
                distance[local_row],
                setup.bs_positions[site_value],
                config,
            )
            output[destination] = prediction
        output.flush()
        records[str(outer)] = {
            "inner": int(inner),
            "fit_count": int(len(fit)),
            "query_count": int(len(rows)),
            "carrier": {
                str(site_value): value.to_dict()
                for site_value, value in carrier.items()
            },
        }
        print(
            f"strict carrier outer={outer} inner={inner} queries={len(rows)}",
            flush=True,
        )
    output.flush()
    return np.load(BASE_CHANNEL, mmap_mode="r"), {
        "reused": False,
        "folds": records,
    }


def fold_rows(
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    full_fold: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for fold in range(5):
        mask = full_fold == fold
        selected = statistics(*values, mask)
        anchor = statistics(*baseline, mask)
        rows.append(
            {
                "fold": fold,
                "count": int(np.sum(mask)),
                "selected": selected,
                "baseline": anchor,
                "delta": float(selected["score"] - anchor["score"]),
            }
        )
    return rows


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
    base_np, carrier_record = build_strict_carrier(
        query_ids, full_fold, args.carrier_pairs, args.force
    )

    train_h = np.load(TRAIN_CHANNEL, mmap_mode="r")
    base = torch.as_tensor(np.asarray(base_np, np.complex64), device=device)
    truth = torch.as_tensor(
        np.asarray(train_h[query_ids], dtype=np.complex64), device=device
    )
    truth_pas = pas_distribution(truth)
    truth_pdp = pdp_distribution(truth)
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    expanded_endpoint = e119.expand_endpoint(endpoint_pas, endpoint_pdp, device)
    baseline = point_metrics(base, truth, truth_pas, truth_pdp)
    baseline_summary = statistics(*baseline)

    action_rows = []
    primary_prediction: torch.Tensor | None = None
    for action in (PRIMARY_ACTION, PAS_CONTROL, PDP_CONTROL):
        prediction = e119.make_candidate(
            base,
            base_pas,
            base_pdp,
            *expanded_endpoint,
            action,
        )
        values = point_metrics(prediction, truth, truth_pas, truth_pdp)
        summary = statistics(*values)
        folds = fold_rows(values, baseline, full_fold)
        row = {
            "name": action.name,
            "action": asdict(action),
            "selected": summary,
            "baseline": baseline_summary,
            "delta": float(summary["score"] - baseline_summary["score"]),
            "positive_folds": int(sum(item["delta"] > 0.0 for item in folds)),
            "worst_fold_delta": float(min(item["delta"] for item in folds)),
            "folds": folds,
        }
        action_rows.append(row)
        if action == PRIMARY_ACTION:
            primary_prediction = prediction

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
    if primary_prediction is None:
        raise RuntimeError("primary prediction was not generated")
    np.save(PRIMARY_CHANNEL, primary_prediction.cpu().numpy().astype(np.complex64))

    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Unique-path mass and coordinate states learned only from official H "
            "improve a strictly donor-aligned official-H complex carrier without "
            "V12 or pointwise Sionna outputs."
        ),
        "protocol": {
            "query_count": int(len(query_ids)),
            "outer_folds": 5,
            "inner_rule": "(outer + 1) % 5",
            "carrier_donors_exclude_outer_and_inner": True,
            "endpoint_donors_exclude_outer_and_inner": True,
            "carrier": "radial_idw_k24 with fit-only beta/gamma",
            "sionna_used": False,
            "v12_used": False,
        },
        "carrier_generation": carrier_record,
        "baseline": baseline_summary,
        "actions": action_rows,
        "primary": primary,
        "deployment_gate": gate,
        "outputs": {
            "base_channel": PATHS.relative(BASE_CHANNEL),
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
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

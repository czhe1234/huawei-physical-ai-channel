#!/usr/bin/env python3
"""Isolate the PDP value of cached explicit Sionna path tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze_joint_path_token_oracle import (
    ARTIFACT_DIR as PATH_ARTIFACT_DIR,
    OOF_META,
    REGIME_CACHE,
    cache_paths,
    grouped_results,
    normalized_cosine_torch,
    optimize_path_amplitudes,
    selected_rows,
    v9_channel,
)
from analyze_v4_exact_joint_projection import pas_distribution, truth_views
from project_paths import PATHS
from round2.data import load_positions
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import Candidate, apply_candidate, point_metrics


VERSION = "Round2-E61-explicit-path-pdp-isolation"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_joint_path_token_pdp_isolation"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "joint_path_token_pdp_isolation"
DEFAULT_CACHE_TAG = "n100_s24_k128_b096_d064_scatter1_sel106"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--path-cache-tag", default=DEFAULT_CACHE_TAG)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--selection-seed", type=int, default=106)
    parser.add_argument("--seed-count", type=int, default=24)
    parser.add_argument("--max-paths", type=int, default=128)
    parser.add_argument("--base-reserve", type=int, default=48)
    parser.add_argument("--base-per-seed", type=int, default=96)
    parser.add_argument("--scatter-per-seed", type=int, default=64)
    parser.add_argument("--scatter-pool", type=int, default=256)
    parser.add_argument("--include-scatter", action="store_true")
    parser.add_argument("--reuse-path-cache", action="store_true")
    return parser.parse_args()


def load_path_cache(tag: str) -> dict[str, np.ndarray]:
    cache_dir = PATH_ARTIFACT_DIR / tag
    names = ("coefficient", "tau", "mask", "rows")
    paths = {name: cache_dir / f"{name}.npy" for name in names}
    missing = [PATHS.relative(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing cached path tensors: {missing}")
    return {name: np.load(path, mmap_mode="r") for name, path in paths.items()}


def projection_specs() -> list[Candidate]:
    specs = [Candidate(name="v9_identity")]
    for order in ("pas_pdp", "pdp_pas"):
        for cycles in (1, 2, 4, 8):
            for alpha_pdp in (0.10, 0.20, 0.35, 0.50, 0.75, 1.00):
                specs.append(
                    Candidate(
                        name=(
                            f"path_pdp_{order}_c{cycles}_"
                            f"ad{int(round(100 * alpha_pdp)):03d}"
                        ),
                        alpha_pas=0.0,
                        alpha_pdp=alpha_pdp,
                        order=order,
                        cycles=cycles,
                    )
                )
    return specs


def evaluate_grid(
    base: np.ndarray,
    truth: np.ndarray,
    path_pdp: np.ndarray,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    specs = projection_specs()
    base_tensor = torch.as_tensor(base, device=device)
    truth_tensor = torch.as_tensor(truth, device=device)
    path_pdp_tensor = torch.as_tensor(path_pdp, device=device)
    base_pas_tensor = pas_distribution(base_tensor)
    truth_pas, truth_pdp = truth_views(truth_tensor)
    result = {
        key: np.empty((len(specs) + 2, len(base)), dtype=np.float64)
        for key in ("pas", "pdp", "error", "energy")
    }
    names: list[str] = []
    for index, spec in enumerate(specs):
        with torch.inference_mode():
            if spec.name == "v9_identity":
                prediction = base_tensor
            else:
                prediction = apply_candidate(
                    base_tensor,
                    base_pas_tensor,
                    base_pas_tensor,
                    path_pdp_tensor,
                    path_pdp_tensor,
                    spec,
                )
            values = point_metrics(prediction, truth_tensor, truth_pas, truth_pdp)
        for key, value in zip(result, values):
            result[key][index] = value
        names.append(spec.name)

    for offset, order in enumerate(("pas_pdp", "pdp_pas"), start=len(specs)):
        spec = Candidate(
            name=f"truth_pdp_only_{order}",
            alpha_pas=0.0,
            alpha_pdp=1.0,
            order=order,
            cycles=8,
        )
        with torch.inference_mode():
            prediction = apply_candidate(
                base_tensor,
                base_pas_tensor,
                base_pas_tensor,
                truth_pdp,
                truth_pdp,
                spec,
            )
            values = point_metrics(prediction, truth_tensor, truth_pas, truth_pdp)
        for key, value in zip(result, values):
            result[key][offset] = value
        names.append(spec.name)

    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "candidate": name,
                **statistics(
                    result["pas"][index],
                    result["pdp"][index],
                    result["error"][index],
                    result["energy"][index],
                ),
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    result["candidate_names"] = np.asarray(names, dtype="U64")
    return rows, result


def main() -> None:
    args = parse_args()
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)
    train_h = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    _, _, _, train_site, _ = load_positions()
    if args.max_points > 0:
        truth_power = np.sum(
            np.abs(train_h[query_ids]) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        rows = selected_rows(
            query_ids,
            fold_id,
            regime,
            truth_power,
            args.max_points,
            args.selection_seed,
        )
        paths = cache_paths(args, rows, query_ids, train_site)
        cache_tag = (
            f"n{len(rows):03d}_s{args.seed_count:02d}_k{args.max_paths:03d}_"
            f"b{args.base_per_seed:03d}_d{args.scatter_per_seed:03d}_"
            f"scatter{int(args.include_scatter)}_sel{args.selection_seed}"
        )
    else:
        paths = load_path_cache(args.path_cache_tag)
        rows = np.asarray(paths["rows"], dtype=np.int64)
        cache_tag = args.path_cache_tag
    truth = np.asarray(train_h[query_ids[rows]], dtype=np.complex64)
    base = v9_channel(rows, device, args.batch_size)

    path_channel_blocks = []
    path_pdp_blocks = []
    amplitude_blocks = []
    for start in range(0, len(rows), args.batch_size):
        end = min(start + args.batch_size, len(rows))
        values = optimize_path_amplitudes(
            np.asarray(paths["coefficient"][start:end]),
            np.asarray(paths["tau"][start:end]),
            np.asarray(paths["mask"][start:end]),
            truth[start:end],
            device,
            args.steps,
            args.learning_rate,
            pas_weight=0.0,
            pdp_weight=1.0,
        )
        path_channel_blocks.append(values[0])
        path_pdp_blocks.append(values[2])
        amplitude_blocks.append(values[3])
        print(f"PDP-only path oracle optimized {end}/{len(rows)}", flush=True)
    path_channel = np.concatenate(path_channel_blocks)
    path_pdp = np.concatenate(path_pdp_blocks)
    amplitudes = np.concatenate(amplitude_blocks)

    projection_rows, projection_metrics = evaluate_grid(base, truth, path_pdp, device)
    local_fold = fold_id[rows]
    local_regime = regime[rows]
    grouped = grouped_results(projection_metrics, local_fold, local_regime)
    baseline = grouped["v9_identity"]["overall"]
    path_rows = [
        row for row in projection_rows if str(row["candidate"]).startswith("path_pdp_")
    ]
    selected = path_rows[0]
    selected_grouped = grouped[str(selected["candidate"])]
    fold_deltas = [
        float(current["score"] - anchor["score"])
        for current, anchor in zip(
            selected_grouped["folds"], grouped["v9_identity"]["folds"]
        )
    ]
    truth_tensor = torch.as_tensor(truth, device=device)
    truth_pas, truth_pdp = truth_views(truth_tensor)
    path_direct = statistics(
        *point_metrics(
            torch.as_tensor(path_channel, device=device),
            truth_tensor,
            truth_pas,
            truth_pdp,
        )
    )
    endpoint_pdp = float(
        normalized_cosine_torch(
            torch.as_tensor(path_pdp, device=device), truth_pdp, "pdp"
        ).mean().item()
    )
    passed = bool(
        float(selected["score"]) >= 0.70
        and sum(delta > 0.0 for delta in fold_deltas) >= 4
    )
    metrics_path = ARTIFACT_DIR / f"pdp_isolation_metrics_n{len(rows):03d}.npz"
    np.savez_compressed(
        metrics_path,
        rows=rows,
        query_ids=query_ids[rows],
        fold_id=local_fold,
        regime=local_regime,
        amplitudes=amplitudes.astype(np.float16),
        path_pdp=path_pdp.astype(np.float16),
        **projection_metrics,
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The explicit path library contains useful hidden PDP structure, while its PAS "
            "angles are too inaccurate to move. Optimize only path amplitudes against PDP "
            "and force every projection to retain the V9 PAS endpoint."
        ),
        "configuration": vars(args),
        "point_count": int(len(rows)),
        "site_counts": {
            str(site): int(np.sum(train_site[query_ids[rows]] == site)) for site in (0, 1)
        },
        "v9_baseline": baseline,
        "pdp_only_path_channel": path_direct,
        "pdp_only_endpoint_cosine": endpoint_pdp,
        "projection_grid": projection_rows,
        "grouped": grouped,
        "selected": selected,
        "selected_fold_deltas_vs_v9": fold_deltas,
        "gate": {
            "required_score": 0.70,
            "required_positive_folds": 4,
            "passed": passed,
            "decision": (
                "Expand the cached path oracle to all 500 OOF points and train a deployable "
                "path-energy predictor."
                if passed
                else "Reject PDP-only explicit-path correction at this interface."
            ),
        },
        "artifacts": {
            "metrics": PATHS.relative(metrics_path),
            "path_cache": PATHS.relative(PATH_ARTIFACT_DIR / cache_tag),
        },
        "seconds": float(time.time() - started),
    }
    report_path = ANALYSIS_DIR / f"joint_path_token_pdp_isolation_n{len(rows):03d}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

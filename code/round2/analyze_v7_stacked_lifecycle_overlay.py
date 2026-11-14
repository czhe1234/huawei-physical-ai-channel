#!/usr/bin/env python3
"""Test whether the V6 lifecycle support action composes with the V7 carrier stack."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META
from run_v6_regime_component_lifecycle import (
    REGIME_CACHE,
    TARGET_PATHS,
    _candidate_from_name,
    apply_candidate,
    candidate_list,
    channel_ad_support,
    point_metrics,
    stage_selection,
)
from run_v7_complex_carrier_exact_projection import (
    OOF_PROJECTED,
    REPORT as STACK_REPORT,
    TEST_PROJECTED,
)


VERSION = "Round2-E55-stacked-carrier-lifecycle-overlay"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "complex_carrier_lifecycle_overlay"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_complex_carrier_lifecycle_overlay"
OOF_SCALED = ARTIFACT_DIR / "oof_stacked_v4_projection_nested_scaled.npy"
OOF_CURRENT = ARTIFACT_DIR / "oof_stacked_current_ad_f16.npy"
TEST_CURRENT = ARTIFACT_DIR / "test_stacked_current_ad_f16.npy"
METRICS = ARTIFACT_DIR / "stacked_lifecycle_metrics.npz"
OUTPUT = ARTIFACT_DIR / "Round2_V7_Stacked_Lifecycle_Test_Channel.npy"
SELECTION = ARTIFACT_DIR / "stacked_lifecycle_selection.json"
REPORT = ANALYSIS_DIR / "stacked_lifecycle_overlay.json"


def nested_scale_lookup(
    report: dict[str, object], fold_id: np.ndarray, site: np.ndarray
) -> np.ndarray:
    output = np.empty(len(fold_id), dtype=np.float32)
    decisions = report["projected_score"]["nested"]["scale_decisions"]
    assigned = np.zeros(len(fold_id), dtype=bool)
    for row in decisions:
        mask = (fold_id == int(row["fold"])) & (site == int(row["site"]))
        output[mask] = np.float32(row["scale"])
        assigned[mask] = True
    if not bool(assigned.all()):
        raise RuntimeError("nested carrier scales do not cover every OOF row")
    return output


def build_scaled_oof(
    fold_id: np.ndarray, site: np.ndarray, report: dict[str, object]
) -> np.ndarray:
    base = np.load(OOF_PROJECTED, mmap_mode="r")
    scales = nested_scale_lookup(report, fold_id, site)
    output = np.lib.format.open_memmap(
        OOF_SCALED, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start in range(0, len(base), 8):
        end = min(start + 8, len(base))
        output[start:end] = (
            np.asarray(base[start:end], dtype=np.complex64)
            * scales[start:end, None, None, None]
        ).astype(np.complex64)
        output.flush()
    return np.load(OOF_SCALED, mmap_mode="r")


def evaluate(
    base: np.ndarray,
    current: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
    regime: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    candidates = candidate_list()
    names = ["identity"] + [candidate.name for candidate in candidates]
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    target_arrays = {
        name: np.load(paths[0], mmap_mode="r") for name, paths in TARGET_PATHS.items()
    }
    result = {
        key: np.empty((len(names), len(query_ids)), dtype=np.float64)
        for key in ("pas", "pdp", "error", "energy")
    }
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        current_tensor = torch.as_tensor(
            np.asarray(current[start:end], dtype=np.float32), device=device
        )
        with torch.inference_mode():
            values = point_metrics(base_tensor, truth)
            for key, value in zip(("pas", "pdp", "error", "energy"), values):
                result[key][0, start:end] = value
            for candidate_index, candidate in enumerate(candidates, start=1):
                target = torch.as_tensor(
                    np.asarray(
                        target_arrays[candidate.target][start:end], dtype=np.float32
                    ),
                    device=device,
                )
                prediction = apply_candidate(
                    base_tensor.clone(),
                    current_tensor,
                    target,
                    regime[start:end],
                    site[start:end],
                    candidate,
                )
                values = point_metrics(prediction, truth)
                for key, value in zip(("pas", "pdp", "error", "energy"), values):
                    result[key][candidate_index, start:end] = value
        print(f"stacked lifecycle grid {end}/{len(query_ids)}", flush=True)
    result["candidate_names"] = np.asarray(names, dtype="U72")
    result["query_ids"] = query_ids
    result["fold_id"] = fold_id
    result["site"] = site
    result["regime"] = regime
    np.savez_compressed(METRICS, **result)
    return result


def select(metrics: dict[str, np.ndarray]) -> dict[str, object]:
    stages = [stage_selection(metrics, stage) for stage in TARGET_PATHS]
    passed = [
        row
        for row in stages
        if int(row["positive_folds"]) >= 4
        and float(row["nested"]["score"]) > float(row["baseline"]["score"])
    ]
    winner = max(passed, key=lambda row: float(row["nested"]["score"])) if passed else None
    payload = {
        "experiment": VERSION,
        "baseline": stages[0]["baseline"],
        "stages": stages,
        "winner": winner,
    }
    SELECTION.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def generate_test(
    selection: dict[str, object], device: torch.device, batch_size: int
) -> dict[str, object] | None:
    winner = selection["winner"]
    if winner is None:
        return None
    base = np.load(TEST_PROJECTED, mmap_mode="r")
    current = channel_ad_support(base, TEST_CURRENT, device, batch_size)
    regimes = np.load(REGIME_CACHE, allow_pickle=False)["test_regime"].astype(np.int8)
    _, _, _, _, site = load_positions()
    target = np.load(TARGET_PATHS[str(winner["stage"])][1], mmap_mode="r")
    policy = {
        (int(row["site"]), int(row["regime_index"])): str(row["candidate"])
        for row in winner["final_policy"]
    }
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        block = np.array(base[start:end], dtype=np.complex64, copy=True)
        block_site = site[start:end]
        block_regime = regimes[start:end]
        for site_value in np.unique(block_site):
            for regime_value in np.unique(block_regime):
                rows = np.flatnonzero(
                    (block_site == site_value) & (block_regime == regime_value)
                )
                if len(rows) == 0:
                    continue
                name = policy[(int(site_value), int(regime_value))]
                if name == "identity":
                    continue
                candidate = _candidate_from_name(name)
                with torch.inference_mode():
                    prediction = apply_candidate(
                        torch.as_tensor(block[rows], device=device),
                        torch.as_tensor(
                            np.asarray(current[start:end][rows], dtype=np.float32),
                            device=device,
                        ),
                        torch.as_tensor(
                            np.asarray(target[start:end][rows], dtype=np.float32),
                            device=device,
                        ),
                        block_regime[rows],
                        block_site[rows],
                        candidate,
                    )
                block[rows] = prediction.cpu().numpy().astype(np.complex64)
        output[start:end] = block
        output.flush()
        print(f"stacked lifecycle test {end}/{len(base)}", flush=True)
    finite = True
    for start in range(0, len(output), 8):
        value = np.asarray(output[start : start + 8])
        finite = finite and bool(
            np.isfinite(value.real).all() and np.isfinite(value.imag).all()
        )
    if not finite:
        raise ValueError("stacked lifecycle output contains non-finite values")
    return {
        "path": PATHS.relative(OUTPUT),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "bytes": int(OUTPUT.stat().st_size),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)
    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)
    stack_report = json.loads(STACK_REPORT.read_text(encoding="utf-8"))
    base = build_scaled_oof(fold_id, site, stack_report)
    current = channel_ad_support(
        base, OOF_CURRENT, torch.device(args.device), args.batch_size
    )
    metrics = evaluate(
        base,
        current,
        query_ids,
        fold_id,
        site,
        regime,
        torch.device(args.device),
        args.batch_size,
    )
    selection = select(metrics)
    output = generate_test(selection, torch.device(args.device), args.batch_size)
    payload = {
        "experiment": VERSION,
        "status": "candidate_ready" if output is not None else "analysis_only",
        "official_inputs_only": True,
        "sionna_used": False,
        "primary_hypothesis": (
            "The V6 geodesic lifecycle support shift provides information complementary "
            "to the E46 complex carrier stack and should retain a positive nested gain."
        ),
        "selection": selection,
        "output": output,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate E134 candidate selection in the complete canonical V12 pipeline."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict
import json
import time

import numpy as np
import torch

import analyze_e119_v12_unique_path_complement as e119
import analyze_e123_complete_path_variable_system as e123
import analyze_e131_v7_path_selection_evidence as e131
import analyze_e133_v7_deformable_peak_alignment as e133
import analyze_e134_sionna_candidate_selection_evidence as e134
import train_e115_complete_path_state_operator as e115
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics
from train_surface_interaction_energy_model import render_spectra


VERSION = "Round2-E135-sionna-selection-v12-complete-score"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "sionna_selection_v12_complete_score"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_sionna_selection_v12_complete_score"
REPORT = ANALYSIS_DIR / "sionna_selection_v12_complete_score.json"
METRICS = ARTIFACT_DIR / "sionna_selection_v12_complete_score_metrics.npz"

PAS_BETA = 0.10
PDP_BETA = 0.10
PROJECTION_ORDER = "pas_pdp"
PROJECTION_CYCLES = 4
MINIMUM_COMPLETE_SCORE_GAIN = 0.001
MINIMUM_POSITIVE_FOLDS = 4
MINIMUM_WORST_FOLD_GAIN = -0.001

SELECTION_KEYS = OrderedDict(
    (
        ("e131_base_plus_v7", "selected_base_plus_v7"),
        (
            "e134_raw_full_one_to_one",
            "selected_base_v7_plus_raw_full_one_to_one",
        ),
        (
            "e134_corrected_full_one_to_one",
            "selected_base_v7_plus_corrected_full_one_to_one",
        ),
        (
            "e134_corrected_full_soft",
            "selected_base_v7_plus_corrected_full_soft",
        ),
    )
)
PRIMARY = "e134_corrected_full_one_to_one"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def render_candidate_endpoint(
    coordinate: np.ndarray,
    mass: np.ndarray,
    port: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    pas_parts = []
    pdp_parts = []
    for start in range(0, len(coordinate), batch_size):
        end = min(start + batch_size, len(coordinate))
        output = {
            "coordinate": torch.as_tensor(coordinate[start:end], device=device),
            "mass": torch.as_tensor(mass[start:end], device=device),
            "port": torch.as_tensor(port[start:end], device=device),
        }
        with torch.inference_mode():
            pas, pdp = render_spectra(output)
        pas_parts.append(pas.cpu().numpy().astype(np.float32))
        pdp_parts.append(pdp.cpu().numpy().astype(np.float32))
    return np.concatenate(pas_parts), np.concatenate(pdp_parts)


def fold_rows(
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fold: np.ndarray,
) -> list[dict[str, object]]:
    output = []
    for fold_value in range(5):
        selected = fold == fold_value
        current = statistics(*values, selected)
        anchor = statistics(*baseline, selected)
        output.append(
            {
                "fold": fold_value,
                "count": int(np.sum(selected)),
                "metrics": current,
                "baseline": anchor,
                "delta": float(current["score"] - anchor["score"]),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    evidence = np.load(e134.METRICS, allow_pickle=False)
    query_ids = evidence["query_ids"].astype(np.int64)
    candidate_fold = evidence["candidate_fold"].astype(np.int8)
    candidate_mask = evidence["candidate_mask"].astype(bool)
    candidate_cache = e131.gather_candidate_cache(query_ids, candidate_fold)
    coordinate = candidate_cache["coordinate"].astype(np.float32)
    full_cache = e123.collect_cache_arrays(query_ids, candidate_fold)
    base_mass = e123.normalize_mass(
        np.exp(
            np.clip(full_cache["base_log_mass"].astype(np.float32), -30.0, 20.0)
        ),
        candidate_mask,
    )
    base_port = e123.normalize_port(
        np.exp(
            np.clip(full_cache["base_log_port"].astype(np.float32), -20.0, 10.0)
        )
    ).reshape(len(query_ids), -1, e115.MP, e115.N_UE)

    endpoints = OrderedDict()
    selections = {}
    for name, cache_key in SELECTION_KEYS.items():
        selection = evidence[cache_key].astype(bool)
        if np.any(selection.sum(axis=1) != e131.TOP_K):
            raise RuntimeError(f"{name} does not select exactly {e131.TOP_K} paths")
        selections[name] = selection
        mass = e133.selected_mass(base_mass, selection, candidate_mask)
        endpoints[name] = render_candidate_endpoint(
            coordinate,
            mass,
            base_port,
            device,
            args.batch_size,
        )

    base_array = np.load(e119.V12_CHANNEL, mmap_mode="r")
    truth_array = np.load(e119.TRAIN_CHANNEL, mmap_mode="r")
    base = torch.as_tensor(
        np.array(base_array, dtype=np.complex64, copy=True), device=device
    )
    truth = torch.as_tensor(
        np.asarray(truth_array[query_ids], dtype=np.complex64), device=device
    )
    if base.shape != truth.shape:
        raise RuntimeError(f"V12/truth shape mismatch: {base.shape} vs {truth.shape}")
    truth_pas = pas_distribution(truth)
    truth_pdp = pdp_distribution(truth)
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    baseline_point = point_metrics(base, truth, truth_pas, truth_pdp)
    baseline = statistics(*baseline_point)
    if abs(float(baseline["score"]) - 0.677521) > 2e-6:
        raise RuntimeError(f"unexpected V12 canonical score: {baseline['score']}")

    action_rows = OrderedDict()
    point_cache = {}
    for name, (endpoint_pas, endpoint_pdp) in endpoints.items():
        action = e119.Action(
            endpoint=name,
            mode="sqrt",
            pas_beta=PAS_BETA,
            pdp_beta=PDP_BETA,
            order=PROJECTION_ORDER,
            cycles=PROJECTION_CYCLES,
        )
        expanded = e119.expand_endpoint(endpoint_pas, endpoint_pdp, device)
        with torch.inference_mode():
            prediction = e119.make_candidate(
                base,
                base_pas,
                base_pdp,
                *expanded,
                action,
            )
            values = point_metrics(prediction, truth, truth_pas, truth_pdp)
        metrics = statistics(*values)
        folds = fold_rows(values, baseline_point, candidate_fold)
        action_rows[name] = {
            "action": asdict(action),
            "metrics": metrics,
            "delta_to_v12": float(metrics["score"] - baseline["score"]),
            "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
            "worst_fold_delta": float(min(row["delta"] for row in folds)),
            "folds": folds,
        }
        point_cache[name] = tuple(np.asarray(value) for value in values)
        print(
            f"{name}: score={metrics['score']:.9f} "
            f"delta={metrics['score'] - baseline['score']:+.9f}",
            flush=True,
        )

    primary = action_rows[PRIMARY]
    e131_row = action_rows["e131_base_plus_v7"]
    gate = {
        "minimum_complete_score_gain": MINIMUM_COMPLETE_SCORE_GAIN,
        "minimum_positive_folds": MINIMUM_POSITIVE_FOLDS,
        "minimum_worst_fold_gain": MINIMUM_WORST_FOLD_GAIN,
        "primary_variant": PRIMARY,
        "v12_score": float(baseline["score"]),
        "primary_score": float(primary["metrics"]["score"]),
        "delta_to_v12": float(primary["delta_to_v12"]),
        "delta_to_e131_endpoint": float(
            primary["metrics"]["score"] - e131_row["metrics"]["score"]
        ),
        "positive_folds": int(primary["positive_folds"]),
        "worst_fold_delta": float(primary["worst_fold_delta"]),
    }
    gate["passed"] = bool(
        gate["delta_to_v12"] >= MINIMUM_COMPLETE_SCORE_GAIN
        and gate["positive_folds"] >= MINIMUM_POSITIVE_FOLDS
        and gate["worst_fold_delta"] >= MINIMUM_WORST_FOLD_GAIN
        and float(primary["metrics"]["pas"]) >= float(baseline["pas"])
        and float(primary["metrics"]["pdp"]) >= float(baseline["pdp"])
        and float(primary["metrics"]["nmse"]) <= float(baseline["nmse"])
    )
    payload = {
        "experiment": VERSION,
        "status": "canonical_complete_score_audit",
        "primary_hypothesis": (
            "E134 Sionna-assisted candidate selection remains beneficial after its "
            "path endpoint is inserted into the complete canonical V12 channel."
        ),
        "protocol": {
            "canonical_fixed500": True,
            "complete_v12_baseline": True,
            "official_score_components": ["pas", "pdp", "nmse"],
            "selection_frozen_from_e134": True,
            "projection_action_frozen_from_e120": True,
            "parameter_scan": False,
            "test_sionna_role": "diagnostic direct candidate evidence only",
        },
        "baseline_v12": baseline,
        "actions": action_rows,
        "gate": gate,
        "decision": {
            "overall_method_success": bool(gate["passed"]),
            "candidate_or_zip_allowed": False,
            "next_action": (
                "distill_sionna_candidate_teacher"
                if gate["passed"]
                else "do_not_call_candidate_selection_an_overall_success"
            ),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        METRICS,
        query_ids=query_ids.astype(np.int32),
        candidate_fold=candidate_fold,
        **{
            f"endpoint_{name}_{kind}": value[index].astype(np.float16)
            for name, value in endpoints.items()
            for index, kind in enumerate(("pas", "pdp"))
        },
        **{
            f"point_{name}_{kind}": value[index].astype(np.float32)
            for name, value in point_cache.items()
            for index, kind in enumerate(("pas", "pdp", "error", "energy"))
        },
    )
    print(
        json.dumps(
            {
                "baseline_v12": baseline,
                "actions": {
                    name: {
                        "metrics": value["metrics"],
                        "delta_to_v12": value["delta_to_v12"],
                        "positive_folds": value["positive_folds"],
                        "worst_fold_delta": value["worst_fold_delta"],
                    }
                    for name, value in action_rows.items()
                },
                "gate": gate,
                "decision": payload["decision"],
                "seconds": payload["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

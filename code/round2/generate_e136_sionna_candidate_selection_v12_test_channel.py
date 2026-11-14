#!/usr/bin/env python3
"""Generate the E134 candidate-selection endpoint on Round 2 test points."""

from __future__ import annotations

import argparse
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
import analyze_e135_sionna_selection_v12_complete_score as e135
import generate_e120_unique_path_v12_test_channel as e120_test
import train_e115_complete_path_state_operator as e115
import train_e118_hierarchical_unique_path_operator as e118
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import component_energy
from run_path_track_spectral_field import valid_components


VERSION = "Round2-E136-sionna-candidate-selection-v12-test"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "sionna_candidate_selection_v12_test"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_sionna_candidate_selection_v12_test"
)
ENDPOINT = ARTIFACT_DIR / "test_sionna_candidate_selection_endpoint.npz"
OUTPUT = ARTIFACT_DIR / "Round2_E136_Sionna_Candidate_Selection_V12_Test_Channel.npy"
REPORT = ANALYSIS_DIR / "sionna_candidate_selection_v12_test.json"
V7_SUPPORT = (
    PATHS.artifacts
    / "round2"
    / "complex_carrier_lifecycle_overlay"
    / "test_stacked_current_ad_f16.npy"
)
SIONNA_PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "test_n500_s24_k128_b096_d064_scatter1_sel106"
)
ACTION = e119.Action(
    endpoint="e134_corrected_full_one_to_one",
    mode="sqrt",
    pas_beta=e135.PAS_BETA,
    pdp_beta=e135.PDP_BETA,
    order=e135.PROJECTION_ORDER,
    cycles=e135.PROJECTION_CYCLES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=24)
    return parser.parse_args()


def load_sionna_paths(test_count: int) -> dict[str, np.ndarray]:
    names = ("rows", "tau", "angles", "mask", "presence", "energy_fraction")
    values = {
        name: np.asarray(np.load(SIONNA_PATH_CACHE / f"{name}.npy", mmap_mode="r"))
        for name in names
    }
    rows = values.pop("rows").astype(np.int64)
    if len(rows) != test_count or not np.array_equal(np.sort(rows), np.arange(test_count)):
        raise RuntimeError("test Sionna rows are not a permutation of all test points")
    order = np.argsort(rows)
    output = {name: value[order] for name, value in values.items()}
    mask = output["mask"].astype(bool)
    for name in ("tau", "angles", "energy_fraction"):
        value = np.asarray(output[name])
        finite = np.isfinite(value)
        if finite.ndim > mask.ndim:
            finite = finite.reshape(*mask.shape, -1).all(axis=2)
        if not finite[mask].all():
            raise RuntimeError(f"non-finite valid Sionna path field: {name}")
    return output


def endpoint_agreement(stack: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return e120_test.mean_endpoint_cosine(stack, mean).astype(np.float32)


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    required = (
        e115.COMPONENTS,
        e115.FULL_FOLD,
        e115.GEODESIC_CACHE,
        V7_SUPPORT,
        e120_test.V12_TEST,
        e134.e98.PARAMETERS,
        *[
            SIONNA_PATH_CACHE / f"{name}.npy"
            for name in (
                "rows",
                "tau",
                "angles",
                "mask",
                "presence",
                "energy_fraction",
            )
        ],
    )
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    setup, train_positions, test_positions, train_site, test_site = load_positions()
    train_count = len(train_positions)
    positions = np.concatenate((train_positions, test_positions), axis=0)
    site = np.concatenate((train_site, test_site), axis=0)
    fold_id = np.load(e115.FULL_FOLD).astype(np.int8)
    component = np.load(e115.COMPONENTS, allow_pickle=False)
    coordinate = component["coordinates"].astype(np.float32)
    coefficient = component["coefficients"].astype(np.complex64)
    point_power = component["point_power"].astype(np.float64)
    residual_fraction = component["residual_fraction"].astype(np.float32)
    raw_config = e115.Config(batch_size=args.batch_size)
    unique_config = e118.Config(batch_size=args.batch_size)
    valid, fraction = valid_components(
        component_energy(coefficient),
        point_power,
        raw_config.donor_components,
        raw_config.minimum_fraction,
    )
    geodesic_cache = np.load(e115.GEODESIC_CACHE, allow_pickle=False)
    geodesic = np.concatenate(
        (
            geodesic_cache["train_features"].astype(np.float32),
            geodesic_cache["test_features"].astype(np.float32),
        ),
        axis=0,
    )
    v7_support = np.load(V7_SUPPORT, mmap_mode="r")
    if v7_support.shape != (setup.test_count, e115.MH, e115.MV, e115.S_SC):
        raise RuntimeError(f"unexpected V7 test support shape: {v7_support.shape}")

    path = load_sionna_paths(setup.test_count)
    path_mask = path["mask"].astype(bool)
    path_mass = np.maximum(path["energy_fraction"].astype(np.float32), 0.0)
    path_mass *= np.sqrt(
        np.maximum(path["presence"].astype(np.float32), 1.0) / 24.0
    )
    path_mass *= path_mask

    pas_models = []
    pdp_models = []
    selected_models = []
    baseline_selected_models = []
    evidence_models = []
    fold_rows = []
    for outer in range(5):
        raw, donor_summary = e120_test.build_raw_evidence(
            outer,
            raw_config,
            positions,
            site,
            train_count,
            fold_id,
            setup.bs_positions,
            coordinate,
            coefficient,
            fraction,
            valid,
            point_power,
            residual_fraction,
            geodesic,
        )
        cluster, cluster_summary = e120_test.cluster_evidence(raw, unique_config)
        candidate_coordinate = cluster["coordinate"].astype(np.float32)
        candidate_mask = cluster["mask"].astype(bool)
        base_mass = e123.normalize_mass(
            np.exp(np.clip(cluster["base_log_mass"].astype(np.float32), -30.0, 20.0)),
            candidate_mask,
        )
        base_port = e123.normalize_port(
            np.exp(np.clip(cluster["base_log_port"].astype(np.float32), -20.0, 10.0))
        ).reshape(setup.test_count, unique_config.candidate_count, e115.MP, e115.N_UE)

        v7_energy = e131.patch_energy(v7_support, candidate_coordinate, candidate_mask)
        corrected_coordinate = e134.corrected_sionna_coordinates(
            path["tau"],
            path["angles"],
            test_site,
            np.full(setup.test_count, outer, dtype=np.int8),
        )
        sionna_evidence, evidence_summary = e134.transfer_sionna_evidence(
            candidate_coordinate,
            candidate_mask,
            corrected_coordinate,
            path_mask,
            path_mass,
            (0, 1, 2),
            True,
        )
        base_rank = e131.row_ranks(base_mass, candidate_mask)
        v7_rank = e131.row_ranks(v7_energy, candidate_mask)
        sionna_rank = e131.row_ranks(sionna_evidence, candidate_mask)
        baseline_score = 0.5 * (base_rank + v7_rank)
        score = (base_rank + v7_rank + sionna_rank) / 3.0
        baseline_selected = e131.top_k_mask(baseline_score, candidate_mask)
        selected = e131.top_k_mask(score, candidate_mask)
        if np.any(selected.sum(axis=1) != e131.TOP_K):
            raise RuntimeError(f"outer {outer} does not select exactly {e131.TOP_K} paths")
        selected_mass = e133.selected_mass(base_mass, selected, candidate_mask)
        endpoint_pas, endpoint_pdp = e135.render_candidate_endpoint(
            candidate_coordinate,
            selected_mass,
            base_port,
            device,
            args.batch_size,
        )
        pas_models.append(endpoint_pas)
        pdp_models.append(endpoint_pdp)
        selected_models.append(selected)
        baseline_selected_models.append(baseline_selected)
        evidence_models.append(sionna_evidence.astype(np.float16))
        overlap = np.sum(selected & baseline_selected, axis=1)
        fold_rows.append(
            {
                **donor_summary,
                **cluster_summary,
                "angle_mapping_fold": outer,
                "valid_candidate_count_mean": float(candidate_mask.sum(axis=1).mean()),
                "selected_count": e131.TOP_K,
                "mean_selected_overlap_with_base_v7": float(np.mean(overlap)),
                "changed_points": int(np.sum(overlap < e131.TOP_K)),
                "sionna_evidence": evidence_summary,
            }
        )
        print(f"rendered E134 test endpoint outer={outer}", flush=True)

    pas_stack = np.stack(pas_models)
    pdp_stack = np.stack(pdp_models)
    selected_stack = np.stack(selected_models)
    baseline_selected_stack = np.stack(baseline_selected_models)
    evidence_stack = np.stack(evidence_models)
    endpoint_pas = np.mean(pas_stack, axis=0, dtype=np.float64).astype(np.float32)
    endpoint_pdp = np.mean(pdp_stack, axis=0, dtype=np.float64).astype(np.float32)
    pas_agreement = endpoint_agreement(pas_stack, endpoint_pas)
    pdp_agreement = endpoint_agreement(pdp_stack, endpoint_pdp)
    np.savez_compressed(
        ENDPOINT,
        pas=endpoint_pas.astype(np.float16),
        pdp=endpoint_pdp.astype(np.float16),
        pas_model_cosine=pas_agreement,
        pdp_model_cosine=pdp_agreement,
        selected=selected_stack,
        baseline_selected=baseline_selected_stack,
        sionna_evidence=evidence_stack,
    )

    base_np = np.load(e120_test.V12_TEST, mmap_mode="r")
    base = torch.as_tensor(np.array(base_np, dtype=np.complex64, copy=True), device=device)
    with torch.inference_mode():
        base_pas = pas_distribution(base)
        base_pdp = pdp_distribution(base)
        expanded = e119.expand_endpoint(endpoint_pas, endpoint_pdp, device)
        output = e119.make_candidate(base, base_pas, base_pdp, *expanded, ACTION)
    output_np = output.cpu().numpy().astype(np.complex64)
    expected = (
        setup.test_count,
        setup.tx_antennas,
        setup.rx_antennas,
        setup.subcarriers,
    )
    if output_np.shape != expected:
        raise RuntimeError(f"unexpected output shape: {output_np.shape}")
    if not np.isfinite(output_np.real).all() or not np.isfinite(output_np.imag).all():
        raise RuntimeError("E136 test output contains non-finite values")
    np.save(OUTPUT, output_np)

    base_array = np.asarray(base_np)
    base_power = np.sum(np.abs(base_array) ** 2, axis=(1, 2, 3), dtype=np.float64)
    output_power = np.sum(np.abs(output_np) ** 2, axis=(1, 2, 3), dtype=np.float64)
    power_ratio = np.divide(
        output_power,
        np.maximum(base_power, e115.TINY),
        out=np.ones_like(output_power),
    )
    relative_nmse = float(
        np.sum(np.abs(output_np - base_array) ** 2, dtype=np.float64)
        / np.maximum(np.sum(np.abs(base_array) ** 2, dtype=np.float64), e115.TINY)
    )
    report = {
        "experiment": VERSION,
        "status": "test_channel_generated",
        "source_local_audit": PATHS.relative(e135.REPORT),
        "local_fixed500": {
            "v12_score": 0.677520807,
            "e134_score": 0.678225636,
            "delta": 0.000704828,
            "deployment_gate_passed": False,
            "online_candidate_generated_at_user_request": True,
        },
        "deployment_policy": {
            "candidate_source": "official train-H neighbor path components",
            "candidate_count": unique_config.candidate_count,
            "selected_count": e131.TOP_K,
            "selection_evidence": [
                "candidate base mass rank",
                "V7 local angle-delay energy rank",
                "same-position E98-aligned Sionna H/V/delay rank",
            ],
            "selection_fusion": "equal-rank mean",
            "fold_ensemble": "mean of five independently constructed endpoints",
            "action": asdict(ACTION),
            "v12_complex_carrier_preserved": True,
            "v12_point_power_preserved": True,
        },
        "fold_models": fold_rows,
        "ensemble_agreement": {
            "pas_mean": float(np.mean(pas_agreement)),
            "pas_p10": float(np.quantile(pas_agreement, 0.10)),
            "pdp_mean": float(np.mean(pdp_agreement)),
            "pdp_p10": float(np.quantile(pdp_agreement, 0.10)),
            "mean_pairwise_selected_overlap": float(
                np.mean(
                    np.sum(
                        selected_stack.sum(axis=0)
                        * (selected_stack.sum(axis=0) - 1),
                        axis=1,
                    )
                    / (5.0 * 4.0 * e131.TOP_K)
                )
            ),
        },
        "endpoint": {
            "path": PATHS.relative(ENDPOINT),
            "pas_shape": list(endpoint_pas.shape),
            "pdp_shape": list(endpoint_pdp.shape),
            "finite": bool(np.isfinite(endpoint_pas).all() and np.isfinite(endpoint_pdp).all()),
        },
        "output": {
            "path": PATHS.relative(OUTPUT),
            "shape": list(output_np.shape),
            "dtype": str(output_np.dtype),
            "finite": True,
            "bytes": int(OUTPUT.stat().st_size),
            "relative_nmse_to_v12": relative_nmse,
            "power_ratio_mean": float(np.mean(power_ratio)),
            "power_ratio_max_abs_error": float(np.max(np.abs(power_ratio - 1.0))),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

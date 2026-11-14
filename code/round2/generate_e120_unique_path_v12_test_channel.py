#!/usr/bin/env python3
"""Generate the E120 unique-path correction on the 500 Round 2 test points."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

import analyze_global_local_path_lifecycle as e79
import analyze_e119_v12_unique_path_complement as e119
import run_e120_fivefold_unique_path_v12 as e120
import train_e115_complete_path_state_operator as e115
import train_e118_hierarchical_unique_path_operator as e118
import train_path_conditioned_scene_gnn as e83
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import component_energy
from run_path_track_spectral_field import valid_components
from train_surface_interaction_energy_model import render_spectra


VERSION = "Round2-E120-unique-path-v12-test"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "fivefold_unique_path_v12"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_fivefold_unique_path_v12"
ENDPOINT = ARTIFACT_DIR / "test_fivefold_mass_coordinate_endpoint.npz"
OUTPUT = ARTIFACT_DIR / "Round2_E120_Unique_Path_V12_Test_Channel.npy"
REPORT = ANALYSIS_DIR / "fivefold_unique_path_v12_test.json"
V12_TEST = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "Round2_E86_Surface_Endpoint_V11_Blend_Test_Channel.npy"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--endpoint-only",
        action="store_true",
        help="Generate the official-H path endpoint without reading or modifying V12.",
    )
    return parser.parse_args()


def build_raw_evidence(
    outer: int,
    raw_config: e115.Config,
    positions: np.ndarray,
    site: np.ndarray,
    train_count: int,
    fold_id: np.ndarray,
    bs_positions: np.ndarray,
    coordinate: np.ndarray,
    coefficient: np.ndarray,
    fraction: np.ndarray,
    valid: np.ndarray,
    point_power: np.ndarray,
    residual_fraction: np.ndarray,
    geodesic: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    test_count = len(positions) - train_count
    query_ids = train_count + np.arange(test_count, dtype=np.int64)
    raw_tokens = raw_config.candidate_count
    probe_feature = e115.candidate_features(
        query_ids[:1],
        np.zeros((1, raw_tokens), np.int64),
        np.zeros((1, raw_tokens), np.int64),
        np.ones((1, raw_tokens), np.float32),
        np.ones((1, raw_tokens), np.float32),
        positions,
        site,
        bs_positions,
        coordinate,
        coefficient,
        fraction,
        point_power,
        residual_fraction,
        raw_config,
    )[0]
    raw = {
        "candidate_feature": np.zeros(
            (test_count, raw_tokens, probe_feature.shape[-1]), np.float32
        ),
        "coordinate": np.zeros((test_count, raw_tokens, 3), np.float32),
        "base_log_mass": np.full((test_count, raw_tokens), -30.0, np.float32),
        "base_log_port": np.full(
            (test_count, raw_tokens, e115.PORTS),
            np.log(1.0 / e115.PORTS),
            np.float32,
        ),
        "base_phase": np.zeros(
            (test_count, raw_tokens, 2 * e115.PORTS), np.float32
        ),
        "mask": np.zeros((test_count, raw_tokens), bool),
        "context": np.zeros(
            (test_count, e79.POINT_CONTEXT_DIM + geodesic.shape[1]), np.float32
        ),
        "base_log_power": np.zeros(test_count, np.float32),
    }
    raw["base_phase"][:, :, : e115.PORTS] = 1.0
    blocked = np.asarray((outer, (outer + 1) % 5), np.int8)
    donor_counts: dict[str, int] = {}
    nearest_distances = []
    for site_value in np.unique(site[train_count:]):
        local_rows = np.flatnonzero(site[train_count:] == site_value)
        local_queries = query_ids[local_rows]
        all_bank = np.flatnonzero(
            (site[:train_count] == site_value) & ~np.isin(fold_id, blocked)
        )
        nonzero_bank = all_bank[point_power[all_bank] > e115.TINY]
        take = min(raw_config.donor_count, len(nonzero_bank))
        distance, local = cKDTree(positions[nonzero_bank, :2]).query(
            positions[local_queries, :2], k=take, workers=-1
        )
        distance = np.asarray(distance)
        local = np.asarray(local)
        if take == 1:
            distance = distance[:, None]
            local = local[:, None]
        nearest_distances.extend(distance[:, 0].tolist())
        donors = nonzero_bank[local.astype(np.int64)]
        point_weight = np.exp(
            -0.5 * np.square(distance / raw_config.spatial_scale_m)
        ) / np.sqrt(np.maximum(distance, 1.0))
        point_weight /= np.maximum(point_weight.sum(axis=1, keepdims=True), e115.TINY)
        donor_table = np.repeat(donors, raw_config.donor_components, axis=1)
        component_table = np.broadcast_to(
            np.arange(raw_config.donor_components, dtype=np.int64)[None, None],
            (len(local_queries), take, raw_config.donor_components),
        ).reshape(len(local_queries), -1)
        token_weight = np.repeat(point_weight, raw_config.donor_components, axis=1)
        token_valid = valid[donor_table, component_table]
        score = fraction[donor_table, component_table] * np.sqrt(
            np.maximum(token_weight, 1e-12)
        )
        score[~token_valid] = -np.inf
        available = token_valid.sum(axis=1)
        if np.min(available) < raw_tokens:
            raise RuntimeError(
                f"insufficient test candidates outer={outer} site={site_value}: "
                f"minimum={int(np.min(available))}"
            )
        selected = np.argpartition(-score, raw_tokens - 1, axis=1)[:, :raw_tokens]
        selected_score = np.take_along_axis(score, selected, axis=1)
        order = np.argsort(-selected_score, axis=1, kind="stable")
        selected = np.take_along_axis(selected, order, axis=1)
        selected_score = np.take_along_axis(selected_score, order, axis=1)
        selected_donor = np.take_along_axis(donor_table, selected, axis=1)
        selected_component = np.take_along_axis(component_table, selected, axis=1)
        donor_rank = selected // raw_config.donor_components
        selected_distance = np.take_along_axis(distance, donor_rank, axis=1)
        feature, selected_coordinate, base_port, base_phase = e115.candidate_features(
            local_queries,
            selected_donor,
            selected_component,
            selected_distance.astype(np.float32),
            selected_score.astype(np.float32),
            positions,
            site,
            bs_positions,
            coordinate,
            coefficient,
            fraction,
            point_power,
            residual_fraction,
            raw_config,
        )
        raw["candidate_feature"][local_rows] = feature
        raw["coordinate"][local_rows] = selected_coordinate
        raw["base_log_mass"][local_rows] = np.log(
            np.maximum(selected_score, 1e-12)
        )
        raw["base_log_port"][local_rows] = np.log(np.maximum(base_port, 1e-8))
        raw["base_phase"][local_rows] = base_phase
        raw["mask"][local_rows] = True
        context, base_power = e115.context_for_group(
            local_queries,
            all_bank,
            positions,
            site,
            bs_positions,
            point_power,
            geodesic,
        )
        raw["context"][local_rows] = context
        raw["base_log_power"][local_rows] = base_power
        donor_counts[str(int(site_value))] = int(len(all_bank))
    if not np.all(raw["mask"]):
        raise RuntimeError(f"unpopulated raw test tokens for outer={outer}")
    return raw, {
        "outer": outer,
        "inner": int((outer + 1) % 5),
        "blocked_folds": blocked.tolist(),
        "donor_counts": donor_counts,
        "nearest_donor_distance_median": float(np.median(nearest_distances)),
        "nearest_donor_distance_p90": float(np.quantile(nearest_distances, 0.90)),
    }


def cluster_evidence(
    raw: dict[str, np.ndarray], config: e118.Config
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    raw_mass = np.exp(raw["base_log_mass"])
    raw_port = np.exp(raw["base_log_port"])
    raw_phase = (
        raw["base_phase"][..., : e115.PORTS]
        + 1j * raw["base_phase"][..., e115.PORTS :]
    )
    count, raw_tokens = raw_mass.shape
    feature_dim = 6 + 3 + 3 + 3 + 1 + e115.PORTS + 2 * e115.PORTS
    cluster = {
        "token_cluster": np.full((count, raw_tokens), -1, np.int64),
        "mask": np.zeros((count, config.candidate_count), bool),
        "coordinate": np.zeros((count, config.candidate_count, 3), np.float32),
        "base_log_mass": np.full(
            (count, config.candidate_count), -30.0, np.float32
        ),
        "base_log_port": np.full(
            (count, config.candidate_count, e115.PORTS),
            np.log(1.0 / e115.PORTS),
            np.float32,
        ),
        "base_phase": np.zeros(
            (count, config.candidate_count, 2 * e115.PORTS), np.float32
        ),
        "cluster_feature": np.zeros(
            (count, config.candidate_count, feature_dim), np.float32
        ),
        "member_count": np.zeros((count, config.candidate_count), np.int16),
    }
    cluster["base_phase"][:, :, : e115.PORTS] = 1.0
    cluster_counts = []
    retained_base_mass = []
    for point_id in range(count):
        groups, selected_mass, total_clusters = e118.cluster_groups(
            raw["coordinate"][point_id],
            raw_mass[point_id],
            config.cluster_threshold,
            config.candidate_count,
        )
        cluster_counts.append(total_clusters)
        retained_base_mass.append(
            float(np.sum(selected_mass, dtype=np.float64))
            / max(float(np.sum(raw_mass[point_id], dtype=np.float64)), e115.TINY)
        )
        for cluster_id, group in enumerate(groups):
            cluster["token_cluster"][point_id, group] = cluster_id
            cluster["mask"][point_id, cluster_id] = True
            cluster["member_count"][point_id, cluster_id] = len(group)
            local_mass = raw_mass[point_id, group]
            total_mass = float(np.sum(local_mass, dtype=np.float64))
            local_coordinate = raw["coordinate"][point_id, group]
            base_coordinate = e117_circular_mean(local_coordinate, local_mass)
            base_port = np.sum(
                local_mass[:, None] * raw_port[point_id, group],
                axis=0,
                dtype=np.float64,
            ) / max(total_mass, e115.TINY)
            base_port /= max(float(np.sum(base_port)), e115.TINY)
            base_direction = e118.normalized_complex_average(
                raw_phase[point_id, group],
                local_mass[:, None] * raw_port[point_id, group],
            )
            cluster["coordinate"][point_id, cluster_id] = base_coordinate
            cluster["base_log_mass"][point_id, cluster_id] = np.log(
                max(total_mass, 1e-12)
            )
            cluster["base_log_port"][point_id, cluster_id] = np.log(
                np.maximum(base_port, 1e-8)
            )
            cluster["base_phase"][point_id, cluster_id] = np.concatenate(
                (base_direction.real, base_direction.imag)
            )
            cluster["cluster_feature"][point_id, cluster_id] = e118.cluster_feature(
                base_coordinate,
                total_mass,
                base_port,
                base_direction,
                local_coordinate,
                local_mass,
                raw_tokens,
            )
    return cluster, {
        "cluster_count_median": float(np.median(cluster_counts)),
        "cluster_count_p90": float(np.quantile(cluster_counts, 0.90)),
        "retained_base_mass_mean": float(np.mean(retained_base_mass)),
        "retained_base_mass_min": float(np.min(retained_base_mass)),
    }


def e117_circular_mean(coordinate: np.ndarray, weight: np.ndarray) -> np.ndarray:
    phase = 2.0 * np.pi * coordinate / e115.PERIODS
    sine = np.sum(weight[:, None] * np.sin(phase), axis=0)
    cosine = np.sum(weight[:, None] * np.cos(phase), axis=0)
    angle = np.mod(np.arctan2(sine, cosine), 2.0 * np.pi)
    return (angle * e115.PERIODS / (2.0 * np.pi)).astype(np.float32)


def scene_features(
    test_positions: np.ndarray,
    test_site: np.ndarray,
    bs_positions: np.ndarray,
) -> dict[str, np.ndarray]:
    ids = np.arange(len(test_positions), dtype=np.int64)
    route, panorama = e83.route_node_features(
        ids, test_positions, test_site, bs_positions, split="test"
    )
    raster = np.load(e115.RASTER_CACHE, allow_pickle=False)
    return {
        "route": route,
        "panorama": panorama,
        "polar": raster["test_polar"].astype(np.float32),
        "corridor": raster["test_corridor"].astype(np.float32),
    }


def model_batch(
    rows: slice,
    raw: dict[str, np.ndarray],
    cluster: dict[str, np.ndarray],
    scene: dict[str, np.ndarray],
    normalization: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    def tensor(value: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(value[rows], dtype=dtype, device=device)

    candidate = (
        tensor(raw["candidate_feature"]) - normalization["candidate_mean"]
    ) / normalization["candidate_std"]
    context = (tensor(raw["context"]) - normalization["context_mean"]) / normalization[
        "context_std"
    ]
    cluster_feature = (
        tensor(cluster["cluster_feature"]) - normalization["cluster_mean"]
    ) / normalization["cluster_std"]
    return {
        "token_feature": candidate.clamp(-8.0, 8.0),
        "token_mask": tensor(raw["mask"], torch.bool),
        "token_base_log_mass": tensor(raw["base_log_mass"]),
        "token_cluster": tensor(cluster["token_cluster"], torch.int64),
        "context": context.clamp(-8.0, 8.0),
        "base_log_power": tensor(raw["base_log_power"]),
        "cluster_feature": cluster_feature.clamp(-8.0, 8.0),
        "coordinate": tensor(cluster["coordinate"]),
        "base_log_mass": tensor(cluster["base_log_mass"]),
        "base_log_port": tensor(cluster["base_log_port"]),
        "base_phase": tensor(cluster["base_phase"]),
        "mask": tensor(cluster["mask"], torch.bool),
        "route": tensor(scene["route"]).permute(0, 3, 1, 2),
        "panorama": tensor(scene["panorama"]).permute(0, 3, 1, 2),
        "polar": tensor(scene["polar"]),
        "corridor": tensor(scene["corridor"]),
    }


@torch.inference_mode()
def predict_endpoint(
    checkpoint_path,
    raw: dict[str, np.ndarray],
    cluster: dict[str, np.ndarray],
    scene: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = e118.Config(**checkpoint["config"])
    variant = e115.Variant(**checkpoint["variant"])
    if not variant.use_scene or variant.use_graph:
        raise RuntimeError(f"unexpected E120 deployment variant: {variant}")
    normalization = {
        key: value.to(device=device, dtype=torch.float32)
        for key, value in checkpoint["normalization"].items()
    }
    model = e118.HierarchicalUniquePathOperator(
        raw["candidate_feature"].shape[-1],
        cluster["cluster_feature"].shape[-1],
        raw["context"].shape[-1],
        e118.GRAPH_DIM,
        config,
        variant,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    pas_parts = []
    pdp_parts = []
    for start in range(0, len(raw["context"]), batch_size):
        rows = slice(start, min(start + batch_size, len(raw["context"])))
        batch = model_batch(rows, raw, cluster, scene, normalization, device)
        learned = model(batch)
        rendered = {
            "mass": learned["mass"],
            "coordinate": learned["coordinate"],
            "port": torch.softmax(batch["base_log_port"], dim=2).reshape(
                len(learned["mass"]), config.candidate_count, e115.MP, e115.N_UE
            ),
        }
        pas, pdp = render_spectra(rendered)
        pas_parts.append(pas.cpu().numpy().astype(np.float32))
        pdp_parts.append(pdp.cpu().numpy().astype(np.float32))
    return (
        np.concatenate(pas_parts),
        np.concatenate(pdp_parts),
        {
            "checkpoint": PATHS.relative(checkpoint_path),
            "best_epoch": int(checkpoint["best_epoch"]),
            "variant": asdict(variant),
        },
    )


def mean_endpoint_cosine(stack: np.ndarray, mean: np.ndarray) -> np.ndarray:
    flat = stack.reshape(stack.shape[0], stack.shape[1], -1).astype(np.float64)
    center = mean.reshape(mean.shape[0], -1).astype(np.float64)
    numerator = np.sum(flat * center[None], axis=2)
    denominator = np.sqrt(np.sum(np.square(flat), axis=2)) * np.sqrt(
        np.sum(np.square(center), axis=1)
    )[None]
    cosine = np.divide(
        numerator,
        np.maximum(denominator, e115.TINY),
        out=np.zeros_like(numerator),
    )
    return cosine.mean(axis=0)


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (e115.COMPONENTS, e115.FULL_FOLD, e115.GEODESIC_CACHE, e115.RASTER_CACHE)
    if not args.endpoint_only:
        required = (*required, V12_TEST)
    missing = [PATHS.relative(path) for path in required if not path.exists()]
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
    scene = scene_features(test_positions, test_site, setup.bs_positions)

    pas_models = []
    pdp_models = []
    fold_rows = []
    for outer in range(5):
        raw, donor_summary = build_raw_evidence(
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
        cluster, cluster_summary = cluster_evidence(raw, unique_config)
        checkpoint = (
            e118.ARTIFACT_DIR / f"e120_outer{outer}_hierarchical_scene.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        pas, pdp, model_summary = predict_endpoint(
            checkpoint, raw, cluster, scene, device, args.batch_size
        )
        pas_models.append(pas)
        pdp_models.append(pdp)
        fold_rows.append({**donor_summary, **cluster_summary, **model_summary})
        print(f"predicted test endpoint outer={outer}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pas_stack = np.stack(pas_models)
    pdp_stack = np.stack(pdp_models)
    endpoint_pas = np.mean(pas_stack, axis=0, dtype=np.float64).astype(np.float32)
    endpoint_pdp = np.mean(pdp_stack, axis=0, dtype=np.float64).astype(np.float32)
    pas_agreement = mean_endpoint_cosine(pas_stack, endpoint_pas)
    pdp_agreement = mean_endpoint_cosine(pdp_stack, endpoint_pdp)
    np.savez_compressed(
        ENDPOINT,
        pas=endpoint_pas.astype(np.float16),
        pdp=endpoint_pdp.astype(np.float16),
        pas_model_cosine=pas_agreement.astype(np.float32),
        pdp_model_cosine=pdp_agreement.astype(np.float32),
    )

    if args.endpoint_only:
        report = {
            "experiment": VERSION,
            "status": "clean_test_endpoint_generated",
            "official_inputs_only": True,
            "sionna_used": False,
            "v12_used": False,
            "deployment_policy": {
                "candidate_source": "official train-H continuous path components only",
                "model_specific_donors": "three fit folds; outer and inner folds excluded",
                "ensemble": "arithmetic mean of five independently rendered PAS/PDP endpoints",
                "endpoint_state": "learned mass + learned coordinate + raw cluster port",
            },
            "fold_models": fold_rows,
            "ensemble_agreement": {
                "pas_mean": float(np.mean(pas_agreement)),
                "pas_p10": float(np.quantile(pas_agreement, 0.10)),
                "pdp_mean": float(np.mean(pdp_agreement)),
                "pdp_p10": float(np.quantile(pdp_agreement, 0.10)),
            },
            "endpoint": {
                "path": PATHS.relative(ENDPOINT),
                "pas_shape": list(endpoint_pas.shape),
                "pdp_shape": list(endpoint_pdp.shape),
                "finite": bool(
                    np.isfinite(endpoint_pas).all()
                    and np.isfinite(endpoint_pdp).all()
                ),
            },
            "seconds": float(time.time() - started),
        }
        REPORT.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return

    base_np = np.load(V12_TEST, mmap_mode="r")
    base = torch.as_tensor(np.array(base_np, copy=True), device=device)
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    expanded = e119.expand_endpoint(endpoint_pas, endpoint_pdp, device)
    output = e119.make_candidate(
        base, base_pas, base_pdp, *expanded, e120.PRIMARY_ACTION
    )
    output_np = output.cpu().numpy().astype(np.complex64)
    if output_np.shape != (setup.test_count, setup.tx_antennas, setup.rx_antennas, setup.subcarriers):
        raise RuntimeError(f"unexpected output shape: {output_np.shape}")
    if not np.isfinite(output_np.real).all() or not np.isfinite(output_np.imag).all():
        raise RuntimeError("E120 test output contains non-finite values")
    np.save(OUTPUT, output_np)
    point_power_base = np.sum(np.abs(np.asarray(base_np)) ** 2, axis=(1, 2, 3))
    point_power_output = np.sum(np.abs(output_np) ** 2, axis=(1, 2, 3))
    power_ratio = np.divide(
        point_power_output,
        np.maximum(point_power_base, e115.TINY),
        out=np.ones_like(point_power_output),
    )
    relative_nmse = float(
        np.sum(np.abs(output_np - np.asarray(base_np)) ** 2, dtype=np.float64)
        / np.maximum(np.sum(np.abs(np.asarray(base_np)) ** 2, dtype=np.float64), e115.TINY)
    )
    report = {
        "experiment": VERSION,
        "status": "test_channel_generated",
        "source_gate_report": PATHS.relative(e120.REPORT),
        "deployment_policy": {
            "candidate_source": "official train-H continuous path components only",
            "model_specific_donors": "three fit folds; outer and inner folds excluded",
            "ensemble": "arithmetic mean of five independently rendered PAS/PDP endpoints",
            "endpoint_state": "learned mass + learned coordinate + raw cluster port",
            "action": asdict(e120.PRIMARY_ACTION),
            "v12_phase_preserved": True,
            "v12_point_power_preserved": True,
        },
        "fold_models": fold_rows,
        "ensemble_agreement": {
            "pas_mean": float(np.mean(pas_agreement)),
            "pas_p10": float(np.quantile(pas_agreement, 0.10)),
            "pdp_mean": float(np.mean(pdp_agreement)),
            "pdp_p10": float(np.quantile(pdp_agreement, 0.10)),
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

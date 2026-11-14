#!/usr/bin/env python3
"""Run and score one single-seed Round 2 Sionna configuration."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback
from typing import Any

import numpy as np

from e200_config_search import SearchConfig, portable_payload
from generate_multiseed_sionna_dataset import chunked_cfr, convert_cfr
from project_paths import PATHS
from round2.data import load_positions


NAMESPACE = os.environ.get("HW_E200_NAMESPACE", "e200").strip().lower()
if NAMESPACE not in {"e200", "e201"}:
    raise ValueError(f"unsupported E200-compatible namespace: {NAMESPACE}")
VERSION = f"Round2-{NAMESPACE.upper()}-single-seed-config-screen"
ARTIFACT_DIR = PATHS.artifacts / "round2" / f"{NAMESPACE}_config_search"
ANALYSIS_DIR = PATHS.analysis / "results" / f"round2_{NAMESPACE}_config_search"
SCENE_DIR = ARTIFACT_DIR / "scenes"
GROUND_OBJ = SCENE_DIR / "ground.obj"
ETOILE_SCENE = (
    PATHS.artifacts
    / "external"
    / "restored_sionna_official_etoile"
    / "etoile.xml"
)
TRUTH_PAS = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "official_pas_cosine_sufficient_target_f32.npy"
)
TRUTH_PDP = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "official_pdp_cosine_sufficient_target_f32.npy"
)
TRUTH_AD = (
    PATHS.artifacts / "round2" / "sionna_official_calibration" / "official_train_ad_f16.npy"
)
TRUTH_POWER = (
    PATHS.artifacts
    / "round2"
    / "sionna_official_calibration"
    / "official_train_point_power_f64.npy"
)
FULL_FOLD = PATHS.artifacts / "round2" / "full_path_pdp_link_gate" / "train_fold_id.npy"
FULL_REGIME = (
    PATHS.artifacts / "round2" / "explicit_path_pdp_predictor" / "path_regime_train_n4000.npy"
)
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-per-site", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260813200)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(portable_payload(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def result_paths(config: SearchConfig) -> tuple[Path, Path]:
    root = ARTIFACT_DIR / "screen" / safe_name(config.name)
    return root / "point_metrics.npz", ANALYSIS_DIR / "configs" / f"{safe_name(config.name)}.json"


def prepare_scene_files(config: SearchConfig) -> Path:
    if NAMESPACE == "e201":
        if not ETOILE_SCENE.is_file():
            raise FileNotFoundError(ETOILE_SCENE)
        return ETOILE_SCENE
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    positions = np.load(PATHS.round2 / "Round2_Train_Pos.npy", mmap_mode="r")
    pad = 120.0
    x0 = float(positions[:, 0].min() - pad)
    x1 = float(positions[:, 0].max() + pad)
    y0 = float(positions[:, 1].min() - pad)
    y1 = float(positions[:, 1].max() + pad)
    if not GROUND_OBJ.is_file():
        temporary = GROUND_OBJ.with_name(f".{GROUND_OBJ.name}.{os.getpid()}.tmp")
        temporary.write_text(
            "\n".join(
                (
                    "# Round 2 E200 generated ground",
                    f"v {x0:.6f} {y0:.6f} 0.0",
                    f"v {x1:.6f} {y0:.6f} 0.0",
                    f"v {x1:.6f} {y1:.6f} 0.0",
                    f"v {x0:.6f} {y1:.6f} 0.0",
                    "f 1 2 3 4",
                    "",
                )
            ),
            encoding="utf-8",
        )
        os.replace(temporary, GROUND_OBJ)
    scene_path = SCENE_DIR / f"{safe_name(config.name)}.xml"
    map_path = PATHS.round2 / "Round2_Map.ply"
    relative_map = os.path.relpath(map_path, SCENE_DIR)
    payload = "\n".join(
        (
            '<scene version="2.1.0">',
            '  <bsdf type="itu-radio-material" id="building_material">',
            f'    <string name="type" value="{config.building_material}"/>',
            '    <float name="thickness" value="0.1"/>',
            "  </bsdf>",
            '  <bsdf type="itu-radio-material" id="ground_material">',
            f'    <string name="type" value="{config.ground_material}"/>',
            '    <float name="thickness" value="0.1"/>',
            "  </bsdf>",
            '  <shape type="ply" id="round2-map">',
            f'    <string name="filename" value="{relative_map}"/>',
            '    <boolean name="face_normals" value="true"/>',
            '    <ref id="building_material" name="bsdf"/>',
            "  </shape>",
            '  <shape type="obj" id="ground">',
            '    <string name="filename" value="ground.obj"/>',
            '    <boolean name="face_normals" value="true"/>',
            '    <ref id="ground_material" name="bsdf"/>',
            "  </shape>",
            "</scene>",
            "",
        )
    )
    if not scene_path.is_file() or scene_path.read_text(encoding="utf-8") != payload:
        temporary = scene_path.with_name(f".{scene_path.name}.{os.getpid()}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, scene_path)
    return scene_path


def select_drjit_cuda_device(dr, device: int) -> None:
    library_path = Path(dr.__file__).with_name("libdrjit-core.so")
    library = ctypes.CDLL(str(library_path))
    setter = getattr(library, "_Z19jit_cuda_set_devicei")
    setter.argtypes = [ctypes.c_int]
    setter.restype = None
    setter(int(device))


def tx_target(position: np.ndarray, yaw_deg: float, tilt_deg: float, mi):
    radius = 140.0
    yaw = math.radians(yaw_deg)
    tilt = math.radians(tilt_deg)
    return mi.Point3f(
        float(position[0] + radius * math.cos(yaw)),
        float(position[1] + radius * math.sin(yaw)),
        float(position[2] + radius * math.tan(tilt)),
    )


def normalize(value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    total = np.sum(value, axis=axes, keepdims=True, dtype=np.float64)
    return np.divide(
        value,
        np.maximum(total, TINY),
        out=np.zeros_like(value, dtype=np.float32),
        where=total > 0.0,
    )


def compact_views(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(channel)
    shaped = channel.reshape(count, 2, 16, 8, 4, 192)
    spatial = np.fft.fft2(shaped, axes=(2, 3))
    pas = normalize(np.square(np.abs(spatial), dtype=np.float32), (1, 2, 3)).mean(
        axis=5, dtype=np.float64
    ).astype(np.float32)
    cir = np.fft.ifft(shaped, axis=5)
    pdp = normalize(np.square(np.abs(cir), dtype=np.float32), (5,)).mean(
        axis=(2, 3), dtype=np.float64
    ).astype(np.float32)
    joint = np.fft.ifft(spatial, axis=5)
    ad = normalize(
        np.square(np.abs(joint), dtype=np.float32).sum(axis=(1, 4), dtype=np.float64).astype(np.float32),
        (1, 2, 3),
    )
    power = np.sum(np.abs(channel) ** 2, axis=(1, 2, 3), dtype=np.float64)
    return pas, pdp, ad, power


def angle_score_grid(
    prediction: np.ndarray, truth: np.ndarray, kind: str
) -> np.ndarray:
    if kind == "pas":
        pred_fft = np.fft.fft2(prediction, axes=(2, 3))
        truth_fft = np.fft.fft2(truth, axes=(2, 3))
        correlation = np.fft.ifft2(
            np.conj(pred_fft) * truth_fft, axes=(2, 3)
        ).real
        numerator = correlation.sum(axis=1, dtype=np.float64)
        pred_norm = np.sqrt(
            np.sum(prediction * prediction, axis=(1, 2, 3), dtype=np.float64)
        )
        truth_norm = np.sqrt(
            np.sum(truth * truth, axis=(1, 2, 3), dtype=np.float64)
        )
        denominator = pred_norm * truth_norm
        score = numerator / np.maximum(denominator[:, None, None, :], TINY)
        return score.mean(axis=3, dtype=np.float64).astype(np.float32)
    if kind == "ad":
        pred_fft = np.fft.fft2(prediction, axes=(1, 2))
        truth_fft = np.fft.fft2(truth, axes=(1, 2))
        numerator = np.fft.ifft2(
            np.conj(pred_fft) * truth_fft, axes=(1, 2)
        ).real.sum(axis=3, dtype=np.float64)
        denominator = np.sqrt(
            np.sum(prediction * prediction, axis=(1, 2, 3), dtype=np.float64)
            * np.sum(truth * truth, axis=(1, 2, 3), dtype=np.float64)
        )
        return (numerator / np.maximum(denominator[:, None, None], TINY)).astype(np.float32)
    raise ValueError(kind)


def pdp_score(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    numerator = np.sum(prediction * truth, axis=3, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(prediction * prediction, axis=3, dtype=np.float64)
        * np.sum(truth * truth, axis=3, dtype=np.float64)
    )
    return (numerator / np.maximum(denominator, TINY)).mean(
        axis=(1, 2), dtype=np.float64
    ).astype(np.float32)


def sample_ids(ids: np.ndarray, site: np.ndarray, count_per_site: int, seed: int) -> np.ndarray:
    if count_per_site <= 0:
        return np.asarray(ids, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = []
    for site_value in np.unique(site[ids]):
        local = ids[site[ids] == site_value]
        selected.append(
            np.sort(rng.choice(local, size=min(count_per_site, len(local)), replace=False))
        )
    return np.sort(np.concatenate(selected)).astype(np.int64)


def summarize(
    config: SearchConfig,
    ids: np.ndarray,
    site: np.ndarray,
    fold: np.ndarray,
    regime: np.ndarray,
    truth_power: np.ndarray,
    pas_grid: np.ndarray,
    ad_grid: np.ndarray,
    pdp: np.ndarray,
    predicted_power: np.ndarray,
    path_count: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    valid = truth_power[ids] > TINY
    shifts: dict[int, tuple[int, int]] = {}
    pas = np.zeros(len(ids), dtype=np.float32)
    ad = np.zeros(len(ids), dtype=np.float32)
    for site_value in (0, 1):
        local = valid & (site[ids] == site_value)
        if not np.any(local):
            shifts[site_value] = (0, 0)
            continue
        objective = (
            0.70 * pas_grid[local].mean(axis=0, dtype=np.float64)
            + 0.30 * ad_grid[local].mean(axis=0, dtype=np.float64)
        )
        h_shift, v_shift = np.unravel_index(int(np.argmax(objective)), objective.shape)
        shifts[site_value] = (int(h_shift), int(v_shift))
        pas[local] = pas_grid[local, h_shift, v_shift]
        ad[local] = ad_grid[local, h_shift, v_shift]
    power = np.zeros(len(ids), dtype=np.float32)
    both = valid & (predicted_power > TINY)
    power[both] = np.exp(
        -np.abs(np.log(predicted_power[both]) - np.log(truth_power[ids[both]])) / 4.0
    ).astype(np.float32)
    raw_pas = pas_grid[:, 0, 0].astype(np.float32)
    raw_ad = ad_grid[:, 0, 0].astype(np.float32)
    combined = 0.45 * pas + 0.40 * pdp + 0.15 * ad
    raw_combined = 0.45 * raw_pas + 0.40 * pdp + 0.15 * raw_ad
    selection_values = raw_combined if NAMESPACE == "e201" else combined
    fold_scores = [
        float(np.mean(selection_values[valid & (fold[ids] == value)]))
        for value in range(5)
        if np.any(valid & (fold[ids] == value))
    ]
    robust = float(np.mean(fold_scores) - 0.25 * np.std(fold_scores))
    difficult = valid & (regime[ids] >= 2)
    difficult_score = (
        float(np.mean(selection_values[difficult])) if np.any(difficult) else robust
    )
    selection = 0.70 * robust + 0.30 * difficult_score
    by_site: dict[str, Any] = {}
    for site_value in (0, 1):
        local = valid & (site[ids] == site_value)
        site_folds = []
        for fold_value in range(5):
            region = local & (fold[ids] == fold_value)
            if not np.any(region):
                continue
            site_folds.append(
                {
                    "fold": fold_value,
                    "count": int(np.sum(region)),
                    "raw_pas": float(np.mean(raw_pas[region])),
                    "raw_ad": float(np.mean(raw_ad[region])),
                    "pdp": float(np.mean(pdp[region])),
                    "raw_combined": float(np.mean(raw_combined[region])),
                    "aligned_pas": float(np.mean(pas[region])),
                    "aligned_ad": float(np.mean(ad[region])),
                    "aligned_combined": float(np.mean(combined[region])),
                }
            )
        if np.any(local):
            by_site[f"site{site_value}"] = {
                "count": int(np.sum(local)),
                "raw_pas": float(np.mean(raw_pas[local])),
                "raw_ad": float(np.mean(raw_ad[local])),
                "pdp": float(np.mean(pdp[local])),
                "raw_combined": float(np.mean(raw_combined[local])),
                "aligned_pas": float(np.mean(pas[local])),
                "aligned_ad": float(np.mean(ad[local])),
                "aligned_combined": float(np.mean(combined[local])),
                "folds": site_folds,
            }
    report = {
        "experiment": VERSION,
        "status": "complete",
        "configuration": config.to_dict(),
        "point_count": int(len(ids)),
        "valid_point_count": int(np.sum(valid)),
        "angle_alignment": {
            f"site{key}": {"horizontal_shift": value[0], "vertical_shift": value[1]}
            for key, value in shifts.items()
        },
        "metrics": {
            "pas": float(np.mean(pas[valid])),
            "pdp": float(np.mean(pdp[valid])),
            "ad": float(np.mean(ad[valid])),
            "power_similarity": float(np.mean(power[valid])),
            "combined": float(np.mean(combined[valid])),
            "fold_scores": fold_scores,
            "fold_mean": float(np.mean(fold_scores)),
            "fold_std": float(np.std(fold_scores)),
            "robust": robust,
            "difficult": difficult_score,
            "selection": selection,
            "raw_pas": float(np.mean(raw_pas[valid])),
            "raw_ad": float(np.mean(raw_ad[valid])),
            "raw_combined": float(np.mean(raw_combined[valid])),
            "selection_uses_raw_alignment": NAMESPACE == "e201",
        },
        "by_site": by_site,
        "paths": {
            "mean": float(np.mean(path_count)),
            "median": float(np.median(path_count)),
            "p90": float(np.quantile(path_count, 0.90)),
            "zero_fraction": float(np.mean(path_count == 0)),
        },
    }
    arrays = {
        "ids": ids.astype(np.int32),
        "pas": pas,
        "pdp": pdp.astype(np.float32),
        "ad": ad,
        "power_similarity": power,
        "combined": combined.astype(np.float32),
        "raw_pas": raw_pas,
        "raw_ad": raw_ad,
        "raw_combined": raw_combined.astype(np.float32),
        "pas_grid": pas_grid.astype(np.float16),
        "ad_grid": ad_grid.astype(np.float16),
        "predicted_power": predicted_power.astype(np.float64),
        "path_count": path_count.astype(np.int32),
    }
    return report, arrays


def gpu_snapshot(physical_index: int) -> dict[str, Any]:
    import subprocess

    command = [
        "nvidia-smi",
        f"--id={physical_index}",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {"physical_index": physical_index, "nvidia_smi": completed.stdout.strip()}


def main() -> None:
    args = parse_args()
    started = time.time()
    config = SearchConfig.from_dict(json.loads(args.config_json.read_text(encoding="utf-8")))
    metrics_path, report_path = result_paths(config)
    if report_path.is_file() and metrics_path.is_file() and not args.force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(json.dumps(existing, ensure_ascii=False, indent=2), flush=True)
            return
    required = (args.ids_file, TRUTH_PAS, TRUTH_PDP, TRUTH_AD, TRUTH_POWER, FULL_FOLD, FULL_REGIME)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing E200 inputs: {missing}")
    scene_path = prepare_scene_files(config)
    setup, train_positions, _, train_site, _ = load_positions()
    ids = np.load(args.ids_file).astype(np.int64)
    ids = sample_ids(ids, train_site, args.limit_per_site, args.sample_seed)
    truth_pas = np.load(TRUTH_PAS, mmap_mode="r")
    truth_pdp = np.load(TRUTH_PDP, mmap_mode="r")
    truth_ad = np.load(TRUTH_AD, mmap_mode="r")
    truth_power = np.load(TRUTH_POWER, mmap_mode="r")
    fold = np.load(FULL_FOLD).astype(np.int8)
    regime = np.load(FULL_REGIME).astype(np.int8)
    pas_grid = np.zeros((len(ids), 16, 8), dtype=np.float16)
    ad_grid = np.zeros((len(ids), 16, 8), dtype=np.float16)
    pdp = np.zeros(len(ids), dtype=np.float32)
    predicted_power = np.zeros(len(ids), dtype=np.float64)
    path_count = np.zeros(len(ids), dtype=np.int32)
    status = "failed"
    error = None
    phase_seconds: dict[str, float] = {}
    effective_batch_size = min(args.batch_size, 8) if not config.synthetic_array else args.batch_size
    try:
        import drjit as dr
        import mitsuba as mi

        if mi.variant() is None:
            mi.set_variant(args.mitsuba_variant)
        select_drjit_cuda_device(dr, args.cuda_device)
        import sionna
        from sionna.phy.channel import subcarrier_frequencies
        from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

        if sionna.__version__ != "2.0.1":
            raise RuntimeError(f"expected Sionna 2.0.1, got {sionna.__version__}")
        frequencies = np.asarray(subcarrier_frequencies(192, 30_000.0)).astype(np.float32)
        solve_started = time.time()
        for site_value in (0, 1):
            local_rows = np.flatnonzero(train_site[ids] == site_value)
            if not len(local_rows):
                continue
            scene = load_scene(
                str(scene_path), merge_shapes=True, remove_duplicate_vertices=True
            )
            scene.frequency = config.frequency_hz
            scene.tx_array = PlanarArray(
                num_rows=8,
                num_cols=16,
                vertical_spacing=config.tx_vertical_spacing,
                horizontal_spacing=config.tx_horizontal_spacing,
                pattern="tr38901",
                polarization="VH",
            )
            scene.rx_array = PlanarArray(
                num_rows=2,
                num_cols=1,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="dipole",
                polarization="cross",
            )
            if NAMESPACE == "e201":
                for material in scene.radio_materials.values():
                    material.scattering_coefficient = 0.0
            else:
                scene.radio_materials["building_material"].scattering_coefficient = (
                    config.building_scattering
                )
                scene.radio_materials["ground_material"].scattering_coefficient = (
                    config.ground_scattering
                )
            bs_position = setup.bs_positions[site_value]
            transmitter = Transmitter(
                name=f"tx_site{site_value}",
                position=mi.Point3f(*map(float, bs_position)),
                look_at=tx_target(
                    bs_position,
                    config.site_yaw_deg[site_value],
                    config.site_tilt_deg[site_value],
                    mi,
                ),
                power_dbm=44.0,
            )
            scene.add(transmitter)
            solver = PathSolver()
            receiver_names: list[str] = []
            for offset in range(0, len(local_rows), effective_batch_size):
                rows = local_rows[offset : offset + effective_batch_size]
                for name in receiver_names:
                    scene.remove(name)
                receiver_names = []
                for row in rows:
                    point_id = int(ids[row])
                    name = f"rx_{point_id:05d}"
                    receiver_args: dict[str, Any] = {
                        "name": name,
                        "position": mi.Point3f(*map(float, train_positions[point_id])),
                    }
                    if config.rx_orientation == "track_bs":
                        receiver_args["look_at"] = transmitter
                    else:
                        receiver_args["orientation"] = mi.Point3f(0.0, 0.0, 0.0)
                    scene.add(Receiver(**receiver_args))
                    receiver_names.append(name)
                select_drjit_cuda_device(dr, args.cuda_device)
                traced = solver(
                    scene,
                    max_depth=config.max_depth,
                    max_num_paths_per_src=config.max_num_paths_per_src,
                    samples_per_src=config.samples_per_src,
                    synthetic_array=config.synthetic_array,
                    los=config.los,
                    specular_reflection=config.specular_reflection,
                    diffuse_reflection=config.diffuse_reflection,
                    refraction=config.refraction,
                    diffraction=config.diffraction,
                    edge_diffraction=config.edge_diffraction,
                    diffraction_lit_region=True,
                    seed=config.seed + 100_000 * site_value + int(ids[rows[0]]),
                )
                path_count[rows] = np.asarray(traced.valid.numpy(), dtype=bool).reshape(
                    len(rows), -1
                ).sum(axis=1)
                raw = chunked_cfr(traced, frequencies, mi)
                channel = convert_cfr(np.asarray(raw), len(rows))
                local_pas, local_pdp, local_ad, local_power = compact_views(channel)
                batch_truth_pas = np.asarray(truth_pas[ids[rows]], dtype=np.float32)
                batch_truth_pdp = np.asarray(truth_pdp[ids[rows]], dtype=np.float32)
                batch_truth_ad = np.asarray(truth_ad[ids[rows]], dtype=np.float32)
                pas_grid[rows] = angle_score_grid(local_pas, batch_truth_pas, "pas").astype(
                    np.float16
                )
                ad_grid[rows] = angle_score_grid(local_ad, batch_truth_ad, "ad").astype(
                    np.float16
                )
                pdp[rows] = pdp_score(local_pdp, batch_truth_pdp)
                predicted_power[rows] = local_power
                del traced, raw, channel, local_pas, local_pdp, local_ad
                dr.sync_thread()
                completed = offset + len(rows)
                if completed == len(local_rows) or completed % 320 < len(rows):
                    print(
                        f"E200 {config.name} site={site_value} "
                        f"{completed}/{len(local_rows)} elapsed={time.time() - solve_started:.1f}s",
                        flush=True,
                    )
            for name in receiver_names:
                scene.remove(name)
            del scene, solver
            dr.sync_thread()
        phase_seconds["sionna_and_metrics"] = float(time.time() - solve_started)
        report, arrays = summarize(
            config,
            ids,
            train_site,
            fold,
            regime,
            truth_power,
            np.asarray(pas_grid, dtype=np.float32),
            np.asarray(ad_grid, dtype=np.float32),
            pdp,
            predicted_power,
            path_count,
        )
        report["data_boundary"] = {
            "official_train_ids": PATHS.relative(args.ids_file),
            "canonical_fixed500_excluded_by_runner": True,
            "official_test_h_used": False,
            "seed_count": 1,
            "scene": (
                PATHS.relative(ETOILE_SCENE)
                if NAMESPACE == "e201"
                else "generated Round2_Map.ply plus planar ground"
            ),
            "scene_default_materials_preserved": NAMESPACE == "e201",
        }
        report["execution"] = {
            "requested_batch_size": int(args.batch_size),
            "effective_batch_size": int(effective_batch_size),
            "physical_array_batch_reduction": bool(not config.synthetic_array),
        }
        report["outputs"] = {"point_metrics": PATHS.relative(metrics_path)}
        status = "complete"
        atomic_npz(metrics_path, **arrays)
    except BaseException:
        error = traceback.format_exc()
        report = {
            "experiment": VERSION,
            "status": "failed",
            "configuration": config.to_dict(),
            "traceback": error,
        }
        raise
    finally:
        report["status"] = status
        report["performance"] = {
            "gpu": gpu_snapshot(
                int(
                    os.environ.get(
                        f"{NAMESPACE.upper()}_PHYSICAL_GPU",
                        os.environ.get("E200_PHYSICAL_GPU", args.cuda_device),
                    )
                )
            ),
            "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),
            "phase_seconds": phase_seconds,
            "wall_seconds": float(time.time() - started),
            "measured_bottleneck": "Sionna PathSolver plus scorer-aligned FFTs",
        }
        report["traceback"] = error
        atomic_json(report_path, report)
        print(json.dumps(portable_payload(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

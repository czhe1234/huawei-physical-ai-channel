#!/usr/bin/env python3
"""Generate and package the official-only confocal normalized-D candidate."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor

import analyze_v279_joint_component_trajectory as v279
import analyze_v393_confocal_pointcloud_pdp as v393
import analyze_v395_mass_preserving_confocal_pdp as v395
from project_paths import PATHS


VERSION_NUMBER = 396
VERSION = "v396_confocal_mass_preserving_d_projection"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = OUT_DIR / "v396_Round1_Test_Channel.npy"
TARGETS = OUT_DIR / "v396_confocal_targets.npz"
MODEL = CHECKPOINT_DIR / "confocal_regressor.joblib"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
BASE = PATHS.cache_dir(284) / "v284_Round1_Test_Channel.npy"
EFFECTIVE_ALPHA = 0.5
FINAL_SEED = 396_500
BATCH_SIZE = 8

SOURCE_FILES = [
    "generate_v396_confocal_mass_preserving_d_projection.py",
    "analyze_v395_mass_preserving_confocal_pdp.py",
    "analyze_v394_confirmatory_confocal_transport.py",
    "analyze_v393_confocal_pointcloud_pdp.py",
    "analyze_v315_full_sequence_confirmatory.py",
    "analyze_v284_planar_geometry_residual_field.py",
    "analyze_v283_joint_geometry_path_renderer.py",
    "analyze_v279_joint_component_trajectory.py",
    "analyze_v274_planar_barrier_attention.py",
    "analyze_v265_scatterer_delay_angle_oracle.py",
    "analyze_v264_patch_scatterer_transformer.py",
    "analyze_v257_neural_pairwise_spectral_transport.py",
    "analyze_v254_route_topology_spectral_field.py",
    "analyze_v247_delaunay_spectral_field.py",
    "analyze_v239_nnls_path_energy_decoder.py",
    "analyze_v208_strict_historical_replay.py",
    "generate_v218_official_multispectrum_physics_ai.py",
    "generate_v215_pointcloud_physics_neural_field.py",
    "generate_v187_leaked_pas_lifecycle_projection.py",
    "generate_v87_easy_field_projection.py",
    "build_official_pointcloud_scene.py",
    "lifecycle_geometry_features.py",
    "powerfield_common.py",
    "project_paths.py",
]


def summarize(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64)
    return {
        "min": float(np.min(value)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "mean": float(np.mean(value)),
        "p90": float(np.quantile(value, 0.90)),
        "max": float(np.max(value)),
    }


def digest(path: Path, algorithm: str) -> str:
    current = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            current.update(block)
    return current.hexdigest()


def test_anchor_d(
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    valid_ids: np.ndarray,
    truth_sh: np.ndarray,
) -> np.ndarray:
    distance, local = cKDTree(train_pos[valid_ids, :2]).query(
        test_pos[:, :2], k=16
    )
    weight = 1.0 / np.maximum(distance, 0.05) ** 2
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-30)
    sh = v279.normalize(
        np.sum(
            weight[:, :, None, None] * truth_sh[valid_ids[local]],
            axis=1,
            dtype=np.float64,
        ).astype(np.float32)
    )
    return v393.normalize(v279.derive_d_numpy(sh))


def train_final_target() -> tuple[np.ndarray, dict[str, object]]:
    confocal_all, feature_meta = v393.load_or_build_features()
    target = np.load(v393.TARGET_CACHE)
    truth_sh = v279.normalize(target["sh"].astype(np.float32))
    truth_d = v393.normalize(target["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    all_pos = np.vstack([train_pos, test_pos])
    scene = np.load(v393.SCENE_CACHE)
    scene_feature = np.vstack(
        [scene["train_features"], scene["test_features"]]
    ).astype(np.float32)
    permutation = v393.distance_stratified_permutation(all_pos, 39602)
    fit_anchor = v393.anchor_d(valid_ids, valid_ids, train_pos, truth_sh)
    test_anchor = test_anchor_d(train_pos, test_pos, valid_ids, truth_sh)
    test_ids = np.arange(len(train_pos), len(all_pos), dtype=np.int64)
    x_fit = v393.model_matrix(
        valid_ids,
        fit_anchor,
        all_pos,
        scene_feature,
        confocal_all,
        valid_ids,
        "full",
        permutation,
    )
    x_test = v393.model_matrix(
        test_ids,
        test_anchor,
        all_pos,
        scene_feature,
        confocal_all,
        valid_ids,
        "full",
        permutation,
    )
    residual_target = np.log(
        (truth_d[valid_ids, : v393.ACTIVE_BINS] + v393.EPS)
        / (fit_anchor[:, : v393.ACTIVE_BINS] + v393.EPS)
    )
    residual_target = np.clip(residual_target, -4.0, 4.0).reshape(-1)
    importance = 0.10 + 10.0 * (
        truth_d[valid_ids, : v393.ACTIVE_BINS]
        + fit_anchor[:, : v393.ACTIVE_BINS]
    )
    importance = np.clip(importance, 0.10, 5.0).reshape(-1)
    model = ExtraTreesRegressor(
        n_estimators=v393.TREE_COUNT,
        max_depth=v393.MAX_DEPTH,
        min_samples_leaf=v393.MIN_LEAF,
        max_features=v393.MAX_FEATURES,
        n_jobs=-1,
        random_state=FINAL_SEED,
    )
    model.fit(x_fit, residual_target, sample_weight=importance)
    joblib.dump(
        {
            "version": VERSION,
            "seed": FINAL_SEED,
            "active_bins": v393.ACTIVE_BINS,
            "effective_alpha": EFFECTIVE_ALPHA,
            "model": model,
        },
        MODEL,
        compress=3,
    )
    residual = np.clip(
        model.predict(x_test).reshape(len(test_pos), v393.ACTIVE_BINS),
        -4.0,
        4.0,
    ).astype(np.float32)
    target_d = v395.apply_mass_preserving_residual(
        test_anchor, residual, EFFECTIVE_ALPHA
    )
    active_mass_error = np.max(
        np.abs(
            target_d[:, : v393.ACTIVE_BINS].sum(axis=1)
            - test_anchor[:, : v393.ACTIVE_BINS].sum(axis=1)
        )
    )
    np.savez_compressed(
        TARGETS,
        target_d=target_d,
        anchor_d=test_anchor,
        residual=residual,
        effective_alpha=np.asarray(EFFECTIVE_ALPHA, dtype=np.float32),
        valid_ids=valid_ids,
    )
    diagnostics = {
        "feature_cache": feature_meta,
        "fit_count": int(len(valid_ids)),
        "test_count": int(len(test_pos)),
        "input_dimension": int(x_fit.shape[1]),
        "fit_bin_rows": int(len(x_fit)),
        "active_mass_max_abs_error": float(active_mass_error),
        "target_anchor_l1": summarize(
            np.sum(np.abs(target_d - test_anchor), axis=1)
        ),
        "residual": summarize(residual),
        "target_finite": bool(np.isfinite(target_d).all()),
        "target_row_sum_max_abs_error": float(
            np.max(np.abs(target_d.sum(axis=1) - 1.0))
        ),
    }
    return target_d, diagnostics


def project_d_only(target_d: np.ndarray) -> dict[str, object]:
    source = np.load(BASE, mmap_mode="r")
    if source.shape != (500, 256, 4, 192) or source.dtype != np.complex64:
        raise ValueError(f"invalid v284 base: {source.shape} {source.dtype}")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    d_cosine = []
    d_l1 = []
    power_ratio = []
    allocation_error = []
    gain_min = []
    gain_max = []
    zero_current_bins = 0
    for start in range(0, len(source), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        cir0 = np.fft.ifft(h0, axis=-1).astype(np.complex64)
        power0 = np.abs(cir0).astype(np.float32) ** 2
        delay_power0 = power0.sum(axis=(1, 2), dtype=np.float64).astype(np.float32)
        total_delay0 = np.maximum(delay_power0.sum(axis=1, keepdims=True), 1e-30)
        current_d = delay_power0 / total_delay0
        zero = current_d <= 1e-30
        zero_current_bins += int(np.sum(zero & (target_d[start:end] > 1e-12)))
        ratio = target_d[start:end] / np.maximum(current_d, 1e-30)
        gain = np.sqrt(np.maximum(ratio, 0.0)).astype(np.float32)
        cir1 = cir0 * gain[:, None, None, :]
        h1 = np.fft.fft(cir1, axis=-1).astype(np.complex64)
        original_power = np.maximum(
            np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), 1e-30
        )
        final_power = np.maximum(
            np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), 1e-30
        )
        h1 *= np.sqrt(original_power / final_power)[:, None, None, None].astype(
            np.float32
        )
        if not np.isfinite(h1).all():
            raise FloatingPointError(f"non-finite projection block {start}:{end}")
        output[start:end] = h1
        output.flush()

        cir_final = np.fft.ifft(h1, axis=-1).astype(np.complex64)
        final_link_power = np.abs(cir_final).astype(np.float32) ** 2
        final_delay_power = final_link_power.sum(axis=(1, 2), dtype=np.float64)
        final_d = v393.normalize(final_delay_power.astype(np.float32))
        d_cosine.append(v393.row_cos(target_d[start:end], final_d))
        d_l1.append(np.sum(np.abs(target_d[start:end] - final_d), axis=1))
        observed_power = np.sum(
            np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        power_ratio.append(observed_power / original_power)
        allocation0 = power0 / np.maximum(
            delay_power0[:, None, None, :], 1e-30
        )
        allocation1 = final_link_power / np.maximum(
            final_delay_power[:, None, None, :], 1e-30
        )
        active = (delay_power0 > 1e-20)[:, None, None, :]
        allocation_error.append(
            np.max(
                np.where(active, np.abs(allocation1 - allocation0), 0.0),
                axis=(1, 2, 3),
            )
        )
        gain_min.append(np.min(gain, axis=1))
        gain_max.append(np.max(gain, axis=1))
        print(f"v396 D-only projection {end}/{len(source)}", flush=True)
    output.flush()
    return {
        "final_d_cosine_target": summarize(np.concatenate(d_cosine)),
        "final_d_l1_target": summarize(np.concatenate(d_l1)),
        "point_power_ratio": summarize(np.concatenate(power_ratio)),
        "delay_bin_allocation_max_abs_error": summarize(
            np.concatenate(allocation_error)
        ),
        "gain_min": summarize(np.concatenate(gain_min)),
        "gain_max": summarize(np.concatenate(gain_max)),
        "zero_current_nonzero_target_bins": zero_current_bins,
    }


def validate_output() -> dict[str, object]:
    prediction = np.load(OUTPUT, mmap_mode="r")
    expected = (500, 256, 4, 192)
    if prediction.shape != expected or prediction.dtype != np.complex64:
        raise ValueError(f"invalid output: {prediction.shape} {prediction.dtype}")
    finite = True
    for start in range(0, len(prediction), 12):
        finite = finite and bool(
            np.isfinite(np.asarray(prediction[start : start + 12])).all()
        )
    if not finite:
        raise FloatingPointError("v396 output contains non-finite values")
    return {"shape": list(prediction.shape), "dtype": str(prediction.dtype), "finite": finite}


def package(output_check: dict[str, object]) -> dict[str, object]:
    for filename in SOURCE_FILES:
        if not (PATHS.code / filename).is_file():
            raise FileNotFoundError(PATHS.code / filename)
    model_files = [
        MODEL,
        TARGETS,
        v393.FEATURE_CACHE,
        v393.SCENE_CACHE,
        PATHS.checkpoints / "v284_planar_geometry_residual_field" / "seed_28401.pt",
        PATHS.checkpoints / "v284_planar_geometry_residual_field" / "seed_28431.pt",
        PATHS.checkpoints / "v284_planar_geometry_residual_field" / "seed_28461.pt",
        PATHS.cache_dir(284) / "v284_planar_geometry_targets.npz",
        PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz",
        PATHS.cache_dir(264) / "v264_test_wall_reflection_candidates.npz",
        PATHS.version_report_dir(265) / "v265_scatterer_delay_angle_oracle.json",
        PATHS.version_report_dir(395) / "v395_mass_preserving_confocal_pdp.json",
    ]
    for path in model_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for filename in SOURCE_FILES:
            archive.write(PATHS.code / filename, filename)
        for path in model_files:
            archive.write(path, path.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise ValueError(f"ZIP CRC failure: {bad_member}")
    nested = [name for name in members if "/" in name.strip("/")]
    if nested:
        raise ValueError(f"submission ZIP must be flat: {nested}")
    return {
        **output_check,
        "zip_crc": True,
        "zip_member_count": len(members),
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }


def main() -> None:
    for path in (BASE, v393.TARGET_CACHE, v393.FEATURE_CACHE, v393.SCENE_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    target_d, target_diagnostics = train_final_target()
    projection = project_d_only(target_d)
    output_check = validate_output()
    integrity = package(output_check)
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The independently confirmed confocal point-cloud model improves test normalized "
            "D when applied as an active-mass-preserving, phase/power/allocation-preserving "
            "CIR projection over the strict v284 carrier."
        ),
        "official_oof_evidence": {
            "source": "v395 third-fold confirmation",
            "full_vs_anchor_cosine": 0.0038580202203924996,
            "full_vs_context_cosine": 0.0013109778722687703,
            "full_vs_shuffled_cosine": 0.0021191536511334873,
            "circular_emd_reduction": 0.04357906044801792,
            "joint_fold_wins": 5,
        },
        "parameters": {
            "final_seed": FINAL_SEED,
            "effective_alpha": EFFECTIVE_ALPHA,
            "active_bins": v393.ACTIVE_BINS,
            "subshells": v393.SUBSHELLS,
            "tree_count": v393.TREE_COUNT,
            "base": PATHS.relative(BASE),
            "projection": "one exact positive real gain per CIR delay bin",
        },
        "target_diagnostics": target_diagnostics,
        "projection_diagnostics": projection,
        "compliance_boundary": {
            "uses": [
                "official train H normalized D labels",
                "official train/test coordinates and Round1_Map point cloud",
                "supervised ExtraTrees AI residual model",
                "strict official-only v284 full-H carrier",
            ],
            "does_not_use": [
                "Sionna/RT test target or path coefficient",
                "restored mesh/material/object labels",
                "v212/synthetic teacher",
                "hidden, local-replay, or online truth",
            ],
        },
        "preserved_quantities": {
            "complex_phase_within_each_delay_bin": True,
            "point_total_power": True,
            "per_delay_bin_antenna_pol_ue_allocation": True,
            "v284_dimensions_changed": ["normalized global PDP D only"],
        },
        "integrity": integrity,
        "outputs": {
            "prediction": PATHS.relative(OUTPUT),
            "targets": PATHS.relative(TARGETS),
            "model": PATHS.relative(MODEL),
            "zip": PATHS.relative(ZIP_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

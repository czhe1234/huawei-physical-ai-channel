#!/usr/bin/env python3
"""v266: learned wall-scatterer weights with joint PDP/leaked-SH projection."""

from __future__ import annotations

import json

import numpy as np
import torch

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v254_route_topology_spectral_field as v254
import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v264_patch_scatterer_transformer as v264
import analyze_v265_scatterer_delay_angle_oracle as v265
import generate_v187_leaked_pas_lifecycle_projection as leak
from generate_v87_easy_field_projection import S_SC, summarize
from project_paths import PATHS


VERSION_NUMBER = 266
VERSION = "v266_learned_scatterer_joint_projection"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = OUT_DIR / "v266_Round1_Test_Channel.npy"
TARGETS = OUT_DIR / "v266_joint_targets.npz"
ACTION = OUT_DIR / "v266_joint_projection_actions.npz"

TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
TRAIN_CANDIDATES = PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz"
NNLS_CACHE = PATHS.cache_dir(264) / "v264_train_scatterer_nnls.npz"
TEST_CANDIDATES = PATHS.cache_dir(264) / "v264_test_wall_reflection_candidates.npz"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
V265_REPORT = PATHS.version_report_dir(265) / "v265_scatterer_delay_angle_oracle.json"
CHECKPOINT_DIR = PATHS.artifacts / "checkpoints" / v264.VERSION

VAL_COUNT = 320
VAL_SEED = 254
TRAIN_EPOCHS = 80
ALPHA_GRID = np.linspace(0.0, 0.60, 13, dtype=np.float32)
MIN_JOINT_GAIN = 2e-4
LOCAL_GATE = 0.645604
PROJECTION_STEPS = 4
PROJECTION_BATCH = 10


def normalize(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 1e-12)
    out /= np.maximum(out.sum(axis=tuple(range(1, out.ndim)), keepdims=True), 1e-30)
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def current_sh(h: np.ndarray) -> np.ndarray:
    spec = leak.leaked_forward(h)
    power = leak.leaked_cell_power(spec)
    return normalize(power.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32))


def alternating_project(
    source_path,
    output_path,
    target_sh: np.ndarray,
    target_d: np.ndarray,
) -> dict[str, np.ndarray]:
    source = np.load(source_path, mmap_mode="r")
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.complex64, shape=source.shape
    )
    diagnostics: dict[str, list[np.ndarray]] = {
        "sh_cos_target": [],
        "d_cos_target": [],
        "sh_l1": [],
        "d_l1": [],
        "power_factor": [],
        "gain_min": [],
        "gain_max": [],
    }
    for start in range(0, len(source), PROJECTION_BATCH):
        end = min(start + PROJECTION_BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        h = h0.copy()
        base_power = np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64)
        all_gain = []
        for _ in range(PROJECTION_STEPS):
            spec = leak.leaked_forward(h)
            power = leak.leaked_cell_power(spec)
            sh_now = normalize(power.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32))
            ratio_sh = target_sh[start:end] / np.maximum(sh_now, 1e-12)
            gain_sh = np.sqrt(np.clip(ratio_sh, 0.0625, 16.0)).astype(np.float32)
            spec *= gain_sh[:, None, :, :, None, None]
            h = leak.leaked_inverse(spec)

            cir = np.fft.ifft(h, axis=-1)
            delay_power = np.abs(cir).astype(np.float32) ** 2
            d_now = normalize(delay_power.sum(axis=(1, 2), dtype=np.float64).astype(np.float32))
            ratio_d = target_d[start:end] / np.maximum(d_now, 1e-12)
            gain_d = np.sqrt(np.clip(ratio_d, 0.0625, 16.0)).astype(np.float32)
            cir *= gain_d[:, None, None, :]
            h = np.fft.fft(cir, axis=-1).astype(np.complex64)
            power_now = np.sum(np.abs(h) ** 2, axis=(1, 2, 3), dtype=np.float64)
            h *= np.sqrt(base_power / np.maximum(power_now, 1e-30))[:, None, None, None].astype(
                np.float32
            )
            all_gain.append(gain_sh.reshape(end - start, -1))
            all_gain.append(gain_d.reshape(end - start, -1))
        output[start:end] = h
        output.flush()
        final_sh = current_sh(h)
        final_cir = np.fft.ifft(h, axis=-1)
        final_d = normalize(
            (np.abs(final_cir).astype(np.float32) ** 2).sum(
                axis=(1, 2), dtype=np.float64
            ).astype(np.float32)
        )
        initial_sh = current_sh(h0)
        initial_cir = np.fft.ifft(h0, axis=-1)
        initial_d = normalize(
            (np.abs(initial_cir).astype(np.float32) ** 2).sum(
                axis=(1, 2), dtype=np.float64
            ).astype(np.float32)
        )
        gains = np.concatenate(all_gain, axis=1)
        diagnostics["sh_cos_target"].append(row_cos(final_sh, target_sh[start:end]))
        diagnostics["d_cos_target"].append(row_cos(final_d, target_d[start:end]))
        diagnostics["sh_l1"].append(np.sum(np.abs(final_sh - initial_sh), axis=(1, 2)))
        diagnostics["d_l1"].append(np.sum(np.abs(final_d - initial_d), axis=1))
        diagnostics["power_factor"].append(
            (
                np.sum(np.abs(h) ** 2, axis=(1, 2, 3), dtype=np.float64)
                / np.maximum(base_power, 1e-30)
            ).astype(np.float32)
        )
        diagnostics["gain_min"].append(gains.min(axis=1).astype(np.float32))
        diagnostics["gain_max"].append(gains.max(axis=1).astype(np.float32))
        if end == len(source) or end % 100 == 0:
            print(f"v266 joint projection {end}/{len(source)}", flush=True)
    return {key: np.concatenate(value).astype(np.float32) for key, value in diagnostics.items()}


def load_model_predictions(
    feature: v264.FeatureSet,
    patches: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    pred_d = []
    pred_weight = []
    for seed in v264.SEEDS:
        checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = v264.ScattererTransformer(
            feature.global_features.shape[1], feature.path_numeric.shape[2], len(patches)
        ).to(device)
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["state_dict"])
        current_d, current_weight = v264.predict(model, feature, device)
        pred_d.append(current_d)
        pred_weight.append(current_weight)
        del model
    return normalize(np.mean(pred_d, axis=0)), normalize(np.mean(pred_weight, axis=0))


def main() -> None:
    for path in (
        TARGET_CACHE,
        TRAIN_CANDIDATES,
        NNLS_CACHE,
        TEST_CANDIDATES,
        SCENE_CACHE,
        BASE,
        V265_REPORT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    targets = np.load(TARGET_CACHE)
    truth_d = normalize(targets["d"].astype(np.float32))
    truth_sh = normalize(targets["sh"].astype(np.float32))
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    matched = v254.choose_seeded_matched_indices(
        train_pos[:, :2], test_pos[:, :2], VAL_COUNT, VAL_SEED
    )
    val_ids = np.intersect1d(matched, valid_ids)
    fit_ids = np.setdiff1d(valid_ids, val_ids)
    nnls_weights = np.load(NNLS_CACHE)["weights"].astype(np.float32)
    patches = np.load(SCENE_CACHE)["wall_patches"].astype(np.float32)
    train_candidates = dict(np.load(TRAIN_CANDIDATES))
    test_candidates = dict(np.load(TEST_CANDIDATES))
    train_node, test_node = v257.build_node_features(train_pos[:, :2], test_pos[:, :2])
    train_raw = v264.build_features(train_pos, train_candidates, patches, train_node)
    test_raw = v264.build_features(test_pos, test_candidates, patches, test_node)
    fit_raw = train_raw.take(fit_ids)
    fit_feature, full_feature, _unused_test, _split_scaling = v264.standardize_features(
        fit_raw, train_raw, test_raw
    )
    val_feature = full_feature.take(val_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_model, validation_training = v264.train_model(
        fit_feature,
        truth_d[fit_ids],
        nnls_weights[fit_ids],
        v264.SEEDS[0],
        TRAIN_EPOCHS,
        device,
        len(patches),
        None,
    )
    val_pred_d, val_pred_weights = v264.predict(validation_model, val_feature, device)
    del validation_model

    with V265_REPORT.open("r", encoding="utf-8") as handle:
        best_spec = json.load(handle)["selection"]["best"]
    train_azimuth, train_path_valid, train_patch_id = v265.departure_azimuth(
        train_pos, train_candidates
    )
    train_h_center = v265.h_center_for(
        train_azimuth,
        float(best_spec["orientation"]),
        str(best_spec["h_mode"]),
        float(best_spec["h_sign"]),
        float(best_spec["h_shift"]),
    )
    true_fit_path_weight, true_fit_delay = v265.collapse_path_weights(
        nnls_weights, train_raw.delay
    )
    split_patch_offset, split_patch_report = v265.estimate_patch_offsets(
        truth_sh,
        fit_ids,
        true_fit_path_weight,
        true_fit_delay,
        train_h_center,
        float(best_spec["s_sign"]),
        train_path_valid,
        train_patch_id,
        len(patches),
    )
    val_path_weight, val_delay = v265.collapse_path_weights(
        val_pred_weights, train_raw.delay[val_ids]
    )
    val_h_center = v265.apply_patch_offsets(
        train_h_center[val_ids], train_patch_id[val_ids], split_patch_offset
    )
    val_pred_sh = v265.render_sh(
        val_path_weight,
        val_delay,
        val_h_center,
        float(best_spec["s_sign"]),
        train_path_valid[val_ids],
        float(best_spec["s_sigma"]),
        float(best_spec["h_sigma"]),
    )
    anchor_sh = v265.knn_anchor(
        train_pos[fit_ids, :2], truth_sh[fit_ids], train_pos[val_ids, :2]
    )
    anchor_d = v264.knn_anchor(
        train_pos[fit_ids, :2], truth_d[fit_ids], train_pos[val_ids, :2]
    )
    grid_results = []
    for alpha_sh in ALPHA_GRID:
        blended_sh = normalize((1.0 - alpha_sh) * anchor_sh + alpha_sh * val_pred_sh)
        sh_score = float(np.mean(row_cos(blended_sh, truth_sh[val_ids])))
        for alpha_d in ALPHA_GRID:
            blended_d = normalize((1.0 - alpha_d) * anchor_d + alpha_d * val_pred_d)
            d_score = float(np.mean(row_cos(blended_d, truth_d[val_ids])))
            grid_results.append(
                {
                    "alpha_sh": float(alpha_sh),
                    "alpha_d": float(alpha_d),
                    "sh_cos": sh_score,
                    "d_cos": d_score,
                    "joint": 0.5 * (sh_score + d_score),
                }
            )
    grid_results.sort(key=lambda item: float(item["joint"]), reverse=True)
    selected = grid_results[0]
    anchor_joint = 0.5 * (
        float(np.mean(row_cos(anchor_sh, truth_sh[val_ids])))
        + float(np.mean(row_cos(anchor_d, truth_d[val_ids])))
    )
    official_gain = float(selected["joint"] - anchor_joint)
    official_selected = bool(
        (selected["alpha_sh"] > 0.0 or selected["alpha_d"] > 0.0)
        and official_gain >= MIN_JOINT_GAIN
    )

    actions = None
    local_score = None
    final_patch_report = None
    if official_selected:
        all_raw = train_raw.take(valid_ids)
        all_feature, _full_train, final_test_feature, _final_scaling = v264.standardize_features(
            all_raw, train_raw, test_raw
        )
        del all_feature
        test_pred_d, test_pred_weights = load_model_predictions(
            final_test_feature, patches, device
        )
        final_patch_offset, final_patch_report = v265.estimate_patch_offsets(
            truth_sh,
            valid_ids,
            true_fit_path_weight,
            true_fit_delay,
            train_h_center,
            float(best_spec["s_sign"]),
            train_path_valid,
            train_patch_id,
            len(patches),
        )
        test_azimuth, test_path_valid, test_patch_id = v265.departure_azimuth(
            test_pos, test_candidates
        )
        test_h_center = v265.h_center_for(
            test_azimuth,
            float(best_spec["orientation"]),
            str(best_spec["h_mode"]),
            float(best_spec["h_sign"]),
            float(best_spec["h_shift"]),
        )
        test_h_center = v265.apply_patch_offsets(
            test_h_center, test_patch_id, final_patch_offset
        )
        test_path_weight, test_delay = v265.collapse_path_weights(
            test_pred_weights, test_raw.delay
        )
        test_pred_sh = v265.render_sh(
            test_path_weight,
            test_delay,
            test_h_center,
            float(best_spec["s_sign"]),
            test_path_valid,
            float(best_spec["s_sigma"]),
            float(best_spec["h_sigma"]),
        )
        base = np.load(BASE, mmap_mode="r")
        base_array = np.asarray(base, dtype=np.complex64)
        base_sh = current_sh(base_array)
        _base_unused_sh, base_d = v247.extract_current(base_array)
        target_test_sh = normalize(
            (1.0 - float(selected["alpha_sh"])) * base_sh
            + float(selected["alpha_sh"]) * test_pred_sh
        )
        target_test_d = normalize(
            (1.0 - float(selected["alpha_d"])) * base_d
            + float(selected["alpha_d"]) * test_pred_d
        )
        np.savez_compressed(
            TARGETS,
            sh=target_test_sh,
            d=target_test_d,
            predicted_sh=test_pred_sh,
            predicted_d=test_pred_d,
            alpha_sh=np.asarray(selected["alpha_sh"]),
            alpha_d=np.asarray(selected["alpha_d"]),
        )
        actions = alternating_project(BASE, OUTPUT, target_test_sh, target_test_d)
        np.savez_compressed(ACTION, **actions)
        local_score = v208.score_prediction(OUTPUT)

    report = {
        "method": VERSION,
        "status": (
            "gate_passed_needs_packaging"
            if local_score is not None
            and float(local_score["score_traceback_leaked"]) > LOCAL_GATE
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "Officially supervised patch-weight prediction plus train-only patch angle "
            "calibration can transfer the v263-v265 PDP and leaked-SH gains to unseen points."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H NNLS path weights and S-H targets",
                "official train/test coordinates",
                "official point-cloud wall patches and analytic reflections",
                "PyTorch patch-ID Transformer",
                "strict v239 carrier",
            ],
            "does_not_use": [
                "Sionna or synthetic labels",
                "teacher output",
                "local replay truth for training or official selection",
                "hidden labels",
                "raw RT H",
            ],
        },
        "parameters": {
            "validation_epochs": TRAIN_EPOCHS,
            "projection_steps": PROJECTION_STEPS,
            "alpha_grid": [float(value) for value in ALPHA_GRID],
            "device": str(device),
        },
        "validation": {
            "fit_count": int(len(fit_ids)),
            "validation_count": int(len(val_ids)),
            "training": validation_training,
            "anchor_sh_cos": float(np.mean(row_cos(anchor_sh, truth_sh[val_ids]))),
            "model_sh_cos": float(np.mean(row_cos(val_pred_sh, truth_sh[val_ids]))),
            "anchor_d_cos": float(np.mean(row_cos(anchor_d, truth_d[val_ids]))),
            "model_d_cos": float(np.mean(row_cos(val_pred_d, truth_d[val_ids]))),
            "anchor_joint": anchor_joint,
            "selected": selected,
            "official_gain": official_gain,
            "official_selected": official_selected,
            "top_grid": grid_results[:20],
            "split_patch_calibration": split_patch_report,
        },
        "final_patch_calibration": final_patch_report,
        "projection_diagnostics": (
            {key: summarize(value) for key, value in actions.items()}
            if actions is not None
            else None
        ),
        "local_gate": (
            {
                "required": LOCAL_GATE,
                "score": local_score,
                "passed": bool(float(local_score["score_traceback_leaked"]) > LOCAL_GATE),
                "gap": float(local_score["score_traceback_leaked"]) - LOCAL_GATE,
            }
            if local_score is not None
            else None
        ),
        "outputs": {
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "targets": PATHS.relative(TARGETS) if TARGETS.is_file() else None,
            "actions": PATHS.relative(ACTION) if ACTION.is_file() else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

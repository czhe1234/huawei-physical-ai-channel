#!/usr/bin/env python3
"""Retrain the validated v282 path selector on scorer-aligned linkwise PDP."""

from __future__ import annotations

import json

import numpy as np
import torch

import analyze_v225_official_graph_residual_field as v225
import analyze_v233_official_joint_ad_manifold as v233
import analyze_v236_official_path_pdp_decoder as v236
import analyze_v238_raw_path_conditional_angle as v238
import analyze_v239_nnls_path_energy_decoder as v239
import analyze_v281_v239_crossfit_pdp_current as v281
import analyze_v282_nested_v239_alpha_selector as v282
import analyze_v344_linkwise_delay_attention as v344
import generate_v215_pointcloud_physics_neural_field as v215
import generate_v218_official_multispectrum_physics_ai as v218
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 345
VERSION = "v345_scorer_aligned_path_selector"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

ALIGNED_SUPPORT = OUT_DIR / "v345_train_link_aligned_support.npy"
PATH_TARGET = OUT_DIR / "v345_test_path_d.npy"
V281_CACHE = OUT_DIR / "v345_crossfit_path_current.npz"
V281_REPORT = REPORT_DIR / "v345_crossfit_path_current.json"
ACTION_OUT = OUT_DIR / "v345_scorer_aligned_actions.npz"
OUTPUT = OUT_DIR / "v345_Round1_Test_Channel.npy"
REPORT = REPORT_DIR / f"{VERSION}.json"

OUTER_SEEDS = (345, 1345, 2345, 3345, 4345)
FINAL_SEEDS = (34501, 34531, 34561)
FINAL_EPOCHS = 60
BATCH = 8
FLOOR = 1e-12

_original_load_targets = v215.load_targets
_original_build_raw_support = v238.build_raw_support


def prepare_aligned_support(target_d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw, valid, log_power = _original_build_raw_support()
    if not ALIGNED_SUPPORT.is_file():
        output = np.lib.format.open_memmap(
            ALIGNED_SUPPORT, mode="w+", dtype=np.float32, shape=raw.shape
        )
        for start in range(0, len(raw), 16):
            end = min(start + 16, len(raw))
            block = v238.normalize_support(np.asarray(raw[start:end], dtype=np.float32))
            current_d = block.sum(axis=(1, 2), dtype=np.float64).astype(np.float32)
            ratio = np.maximum(target_d[start:end], FLOOR) / np.maximum(current_d, FLOOR)
            block *= ratio[:, None, None, :]
            output[start:end] = v238.normalize_support(block)
            output.flush()
    return np.load(ALIGNED_SUPPORT, mmap_mode="r"), valid, log_power


def patched_load_targets():
    clean, raw, log_power, valid = _original_load_targets()
    target_d = v344.link_global_d(PATHS.round1 / "Round1_Train_Channel.npy")
    clean = dict(clean); raw = dict(raw)
    clean["d"] = target_d; raw["d"] = target_d
    valid = valid & (target_d.sum(axis=1) > 1e-20)
    return clean, raw, log_power, valid


def patched_build_raw_support():
    target_d = v344.link_global_d(PATHS.round1 / "Round1_Train_Channel.npy")
    return prepare_aligned_support(target_d)


def test_anchor(
    support: np.ndarray,
    valid_ids: np.ndarray,
    log_power: np.ndarray,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    train_blocked: np.ndarray,
    test_blocked: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices, weights = v218.neighbor_index_weights(
        test_pos,
        test_blocked,
        train_pos[valid_ids],
        train_blocked[valid_ids],
        k=v238.ANCHOR_K,
    )
    anchor = v281.weighted_support(support[valid_ids], indices, weights)
    anchor_power = np.sum(log_power[valid_ids][indices] * weights, axis=1).astype(np.float32)
    return anchor, anchor_power, weights


def train_final_path_target() -> dict[str, object]:
    clean, _raw, _power, valid_delay = patched_load_targets()
    target_d = v239.normalize_d(clean["d"])
    support, valid_support, log_power = patched_build_raw_support()
    valid_ids = np.flatnonzero(valid_delay & valid_support).astype(np.int64)
    nnls_weights, _reconstruction, nnls_report = v239.build_nnls_labels(target_d, valid_ids)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(v238.GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    train_anchor, train_power, train_weights = v281.make_anchor(
        valid_ids, valid_ids, support, log_power, train_pos, train_blocked, True
    )
    test_support, test_power, test_weights = test_anchor(
        support, valid_ids, log_power, train_pos, test_pos, train_blocked, test_blocked
    )
    train_anchor_d = v239.normalize_d(train_anchor.sum(axis=(1, 2)))
    test_anchor_d = v239.normalize_d(test_support.sum(axis=(1, 2)))
    train_raw = v281.features_for(
        valid_ids, train_anchor, train_power, train_weights, train_pos, train_blocked, train_anchor_d
    )
    test_raw = v236.augment_raw_features(
        v233.raw_features(
            "test", None, test_pos, test_blocked, test_support, test_power, test_weights
        ),
        test_anchor_d,
    )
    train_features, [test_features], stats = v233.standardize_feature_sets(train_raw, [test_raw])
    train_delay = v236.raw_path_delay("train", valid_ids)
    test_delay = v236.raw_path_delay("test", None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    training = []
    for seed in FINAL_SEEDS:
        model, info = v239.train_model(
            train_features,
            train_delay,
            target_d[valid_ids],
            nnls_weights[valid_ids],
            seed,
            FINAL_EPOCHS,
            device,
            None,
        )
        torch.save(
            {"state_dict": model.state_dict(), "feature_stats": stats, "seed": seed, "epochs": FINAL_EPOCHS},
            CHECKPOINT_DIR / f"seed_{seed}.pt",
        )
        predictions.append(v239.predict_model(model, test_features, test_delay, device).d)
        training.append(info)
    prediction = v239.normalize_d(np.mean(predictions, axis=0))
    np.save(PATH_TARGET, prediction.astype(np.float32))
    return {
        "valid_count": int(len(valid_ids)),
        "nnls": nnls_report,
        "training": training,
        "seed_agreement": summarize(
            np.mean(
                np.stack(
                    [v239.row_cos(predictions[i], predictions[j]) for i in range(3) for j in range(i + 1, 3)]
                ),
                axis=0,
            )
        ),
    }


def project_linkwise(source_path, output_path, targets, regimes, gain_clip_scale):
    source = np.load(source_path, mmap_mode="r")
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.complex64, shape=source.shape
    )
    target_d = np.asarray(targets["d"], dtype=np.float32)
    diagnostics = {"target_cos": [], "d_l1_actual": [], "power_factor": []}
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        cir = np.fft.ifft(h0, axis=-1)
        link_power = np.sum(np.abs(cir) ** 2, axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
        initial = None
        for _ in range(8):
            power = np.abs(cir).astype(np.float32) ** 2
            normalized = power / np.maximum(power.sum(axis=-1, keepdims=True), 1e-30)
            current = v239.normalize_d(normalized.sum(axis=(1, 2), dtype=np.float64).astype(np.float32))
            if initial is None: initial = current.copy()
            ratio = np.maximum(target_d[start:end], FLOOR) / np.maximum(current, FLOOR)
            cir *= np.sqrt(ratio)[:, None, None, :]
            now = np.sum(np.abs(cir) ** 2, axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
            cir *= np.sqrt(link_power / np.maximum(now, 1e-30))
        h1 = np.fft.fft(cir, axis=-1).astype(np.complex64)
        output[start:end] = h1; output.flush()
        power = np.abs(cir).astype(np.float32) ** 2
        normalized = power / np.maximum(power.sum(axis=-1, keepdims=True), 1e-30)
        final = v239.normalize_d(normalized.sum(axis=(1, 2), dtype=np.float64).astype(np.float32))
        diagnostics["target_cos"].append(v239.row_cos(final, target_d[start:end]))
        diagnostics["d_l1_actual"].append(np.sum(np.abs(final - initial), axis=1).astype(np.float32))
        diagnostics["power_factor"].append(
            (
                np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64)
                / np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), 1e-30)
            ).astype(np.float32)
        )
    return {key: np.concatenate(value) for key, value in diagnostics.items()}


def configure() -> None:
    v215.load_targets = patched_load_targets
    v238.build_raw_support = patched_build_raw_support
    v281.OUTER_SEEDS = OUTER_SEEDS
    v281.OUT_DIR = OUT_DIR
    v281.REPORT_DIR = REPORT_DIR
    v281.OOF_OUT = V281_CACHE
    v281.REPORT = V281_REPORT
    v282.OUTER_SEEDS = OUTER_SEEDS
    v282.VERSION_NUMBER = VERSION_NUMBER
    v282.VERSION = VERSION
    v282.OUT_DIR = OUT_DIR
    v282.REPORT_DIR = REPORT_DIR
    v282.REPORT = REPORT
    v282.ACTION_OUT = ACTION_OUT
    v282.OUTPUT = OUTPUT
    v282.V281_CACHE = V281_CACHE
    v239.TARGET_OUT = PATH_TARGET
    v225.apply_delay_targets = project_linkwise


def main() -> None:
    configure()
    v281.main()
    stage1 = json.loads(V281_REPORT.read_text(encoding="utf-8"))
    stage1_delta = float(stage1["selection"]["mean_delta"])
    stage1_wins = int(stage1["selection"]["wins"])
    if stage1_delta <= 0.005 or stage1_wins < 4:
        report = {
            "method": VERSION,
            "status": "analysis_only_stage1_rejected",
            "stage1": stage1["selection"],
            "decision": "nested selector and final target not run",
        }
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    final_training = train_final_path_target()
    v282.main()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report["method"] = VERSION
    report["scorer_alignment"] = {
        "label": "1024 links normalized independently before global D aggregation",
        "support_anchor": "raw angle-delay support reweighted to scorer-aligned D",
        "nnls": "recomputed from scorer-aligned D",
        "projection": "per-link power preserving",
        "stage1": stage1["selection"],
        "final_path_training": final_training,
    }
    report["outputs"]["report"] = PATHS.relative(REPORT)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

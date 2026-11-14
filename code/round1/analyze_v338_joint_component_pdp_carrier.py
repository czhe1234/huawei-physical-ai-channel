#!/usr/bin/env python3
"""Combine independently validated official-only PDP and conditional-H experts."""

from __future__ import annotations

import json

import numpy as np
import torch

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v274_planar_barrier_attention as v274
import analyze_v337_componentwise_deformable_attention as v337
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 338
VERSION = "v338_joint_component_pdp_carrier"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = OUT_DIR / "v338_Round1_Test_Channel.npy"
TARGETS_OUT = OUT_DIR / "v338_joint_component_pdp_targets.npz"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
V282_ACTIONS = PATHS.cache_dir(282) / "v282_v239_alpha_actions.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

COMPONENT_ALPHA = 0.625
FINAL_EPOCH = 1
FINAL_SEEDS = (33801, 33831, 33861)


def impose_s_marginal(sh: np.ndarray, target_s: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(sh, dtype=np.float32), 0.0)
    marginal = out.sum(axis=2)
    conditional = out / np.maximum(marginal[:, :, None], 1e-20)
    empty = marginal <= 1e-20
    if np.any(empty):
        conditional[empty] = 1.0 / out.shape[2]
    out = conditional * target_s[:, :, None]
    return v337.normalize(out)


def main() -> None:
    for path in (BASE, V282_ACTIONS, TARGET_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    target_data = np.load(TARGET_CACHE)
    sh = v337.normalize(target_data["sh"].astype(np.float32))
    d = target_data["d"].astype(np.float32)
    d /= np.maximum(d.sum(axis=1, keepdims=True), 1e-20)
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    geometry = v274.load_geometry()
    all_pos, signatures, blocker_count, scene_features = geometry[:4]
    node_raw = v274.build_node_features(
        all_pos, signatures, blocker_count, scene_features
    )
    test_ids = np.arange(len(sh), len(all_pos), dtype=np.int64)
    test_set, stats = v337.build_set(test_ids, valid_ids, geometry, node_raw)
    train_set = v337.training_set(valid_ids, geometry, node_raw, stats)
    test_anchor = v337.base_prediction(test_set, sh)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_predictions = []
    training_rows = []
    for seed in FINAL_SEEDS:
        model, training = v337.train_model(
            train_set,
            node_raw,
            stats,
            sh,
            d,
            seed,
            device,
            FINAL_EPOCH,
        )
        prediction, gates, centers = v337.predict_model(
            model, test_set, node_raw, stats, sh, device
        )
        model_predictions.append(prediction)
        checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
        torch.save(
            {
                "version": VERSION,
                "seed": seed,
                "epochs": FINAL_EPOCH,
                "model_state": model.state_dict(),
            },
            checkpoint,
        )
        training_rows.append(
            {
                "seed": seed,
                "training": training,
                "gate": summarize(gates),
                "center_probability": summarize(centers),
                "checkpoint": PATHS.relative(checkpoint),
            }
        )

    test_model = v337.normalize(np.mean(model_predictions, axis=0))
    current_sh, _ = v274.extract_current_marginals(BASE)
    component_sh = v337.normalize(
        current_sh + COMPONENT_ALPHA * (test_model - test_anchor)
    )
    v282_actions = np.load(V282_ACTIONS)
    target_d = np.asarray(v282_actions["target_d"], dtype=np.float32)
    reverse = (-np.arange(v337.S_SIZE)) % v337.S_SIZE
    target_s = target_d[:, reverse]
    target_sh = impose_s_marginal(component_sh, target_s)
    derived_d = v337.derive_d_numpy(target_sh)
    pdp_identity_l1 = np.sum(np.abs(derived_d - target_d), axis=1)

    projection = v247.project(target_sh, target_d, OUTPUT)
    local_score = v208.score_prediction(OUTPUT)
    np.savez_compressed(
        TARGETS_OUT,
        anchor_sh=test_anchor,
        model_sh=test_model,
        current_sh=current_sh,
        component_sh=component_sh,
        target_sh=target_sh,
        target_d=target_d,
        component_alpha=np.asarray(COMPONENT_ALPHA, dtype=np.float32),
    )

    report = {
        "method": VERSION,
        "status": "strict_complete_local_diagnostic",
        "primary_hypothesis": (
            "The v282 path-current expert should own the absolute S/PDP marginal, while "
            "v337 component-wise attention should only correct H-angle conditionals within each S bin."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H S-H/PDP labels",
                "official train/test coordinates",
                "official point-cloud scene/barrier features",
                "v282 official-only adaptive path PDP target",
                "strict v239 official-only complex carrier",
            ],
            "does_not_use": [
                "v211/v212 outputs or teacher labels",
                "synthetic labels during training",
                "restored scene mesh/material/object data",
                "RT H/path coefficients",
                "online-score subspace operations",
            ],
            "local_truth_usage": "one frozen diagnostic after both expert rules were fixed",
        },
        "frozen_evidence": {
            "v282_pdp_oof_delta": 0.0353250867408993,
            "v282_pdp_oof_wins": 5,
            "v337_sh_oof_delta": 0.004551708698272705,
            "v337_weighted_oof_delta": 0.0035606102943420924,
            "v337_oof_wins": 5,
        },
        "parameters": {
            "component_alpha": COMPONENT_ALPHA,
            "component_epochs": FINAL_EPOCH,
            "component_seeds": list(FINAL_SEEDS),
            "pdp_target": "frozen v282 target_d",
        },
        "training": training_rows,
        "diagnostics": {
            "pdp_identity_l1": summarize(pdp_identity_l1),
            "component_residual_l1": summarize(
                np.sum(np.abs(component_sh - current_sh), axis=(1, 2))
            ),
            "projection": {key: summarize(value) for key, value in projection.items()},
        },
        "local_score": local_score,
        "references": {
            "v239_local": 0.5900571130783295,
            "v282_local": 0.5858171973760586,
            "v284_local": 0.5925112281386167,
            "v211_local": 0.6350959054536596,
            "v212_local": 0.6456036433098431,
        },
        "outputs": {
            "prediction": PATHS.relative(OUTPUT),
            "targets": PATHS.relative(TARGETS_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(local_score, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

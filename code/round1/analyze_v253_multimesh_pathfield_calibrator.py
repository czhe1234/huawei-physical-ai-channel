#!/usr/bin/env python3
"""v253: official-H calibration of wall-patch and Poisson path-field experts."""

from __future__ import annotations

import json

import numpy as np
import torch

import analyze_v252_pointcloud_pathfield_calibrator as v252
import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
from evaluate_derotated_loo_grid import choose_val_indices
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 253
VERSION = "v253_multimesh_pathfield_calibrator"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = OUT_DIR / "v253_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v253_test_calibrated_pdp.npy"
ACTION_OUT = OUT_DIR / "v253_multimesh_actions.npz"

WALL_CACHE = PATHS.cache_dir(252) / "v252_pointcloud_sionna_path_fields.npz"
POISSON_CACHE = OUT_DIR / "v253_poisson_sionna_path_fields.npz"
EPOCHS = 60
SEEDS = (25301, 25331, 25361)


def combined_features(
    wall: np.lib.npyio.NpzFile,
    poisson: np.lib.npyio.NpzFile,
    prefix: str,
    indices: np.ndarray | None,
    pos: np.ndarray,
    blocked: np.ndarray,
    anchor: np.ndarray,
    weights: np.ndarray,
) -> v252.Features:
    wall_f = v252.field_features(wall, prefix, indices, pos, blocked, anchor, weights)
    poisson_f = v252.field_features(poisson, prefix, indices, pos, blocked, anchor, weights)
    wall_source = np.zeros(wall_f.path_numeric.shape[:2] + (1,), dtype=np.float32)
    poisson_source = np.ones(poisson_f.path_numeric.shape[:2] + (1,), dtype=np.float32)
    path_numeric = np.concatenate(
        [
            np.concatenate([wall_f.path_numeric, wall_source], axis=2),
            np.concatenate([poisson_f.path_numeric, poisson_source], axis=2),
        ],
        axis=1,
    ).astype(np.float32)
    path_valid = np.concatenate([wall_f.path_valid, poisson_f.path_valid], axis=1)
    path_type = np.concatenate([wall_f.path_type, poisson_f.path_type], axis=1)
    raw_delay = np.concatenate([wall_f.raw_delay, poisson_f.raw_delay], axis=1)
    log_field_power = np.concatenate([wall_f.log_field_power, poisson_f.log_field_power], axis=1)

    def coverage(cache: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
        valid = cache[f"{prefix}_valid"]
        power = cache[f"{prefix}_path_power"]
        if indices is not None:
            valid = valid[indices]
            power = power[indices]
        return valid.mean(axis=1).astype(np.float32), np.log10(np.maximum(power.sum(axis=1), 1e-40)).astype(np.float32)

    wall_cov, wall_power = coverage(wall)
    poisson_cov, poisson_power = coverage(poisson)
    extra = np.stack(
        [
            poisson_cov,
            poisson_power,
            poisson_cov - wall_cov,
            np.clip(poisson_power - wall_power, -12.0, 12.0),
        ],
        axis=1,
    ).astype(np.float32)
    global_features = np.concatenate([wall_f.global_features, extra], axis=1).astype(np.float32)
    return v252.Features(
        global_features,
        path_numeric,
        path_type,
        path_valid,
        raw_delay,
        log_field_power,
    )


def main() -> None:
    for path in [WALL_CACHE, POISSON_CACHE, v252.TARGET_CACHE, v252.GEOM_CACHE, v252.BASE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wall = np.load(WALL_CACHE)
    poisson = np.load(POISSON_CACHE)
    targets = np.load(v252.TARGET_CACHE)
    d_all = v252.normalize(targets["d"].astype(np.float32))
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(v252.GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    matched = choose_val_indices(train_pos[:, :2], test_pos[:, :2], v252.VAL_COUNT, "matched-test", VERSION_NUMBER)
    val_ids = np.intersect1d(matched, valid_ids)
    train_ids = np.setdiff1d(valid_ids, val_ids)

    train_anchor, train_w = v252.make_anchor(
        train_pos[train_ids], train_blocked[train_ids], train_pos[train_ids], train_blocked[train_ids], d_all[train_ids],
        query_ids=train_ids, bank_ids=train_ids,
    )
    val_anchor, val_w = v252.make_anchor(
        train_pos[val_ids], train_blocked[val_ids], train_pos[train_ids], train_blocked[train_ids], d_all[train_ids]
    )
    train_raw = combined_features(wall, poisson, "train", train_ids, train_pos[train_ids], train_blocked[train_ids], train_anchor, train_w)
    val_raw = combined_features(wall, poisson, "train", val_ids, train_pos[val_ids], train_blocked[val_ids], val_anchor, val_w)
    train_features, [val_features], _ = v252.standardize(train_raw, [val_raw])
    model, validation = v252.train_model(
        train_features, train_anchor, d_all[train_ids], SEEDS[0], EPOCHS, device,
        (val_features, val_anchor, d_all[val_ids]),
    )
    val_pred, val_path, val_gate = v252.predict(model, val_features, val_anchor, device)
    val_regimes = regime_of(train_blocked[val_ids]).astype(str)
    alpha_by_regime, calibration = v252.calibrate_alpha(
        val_anchor, val_pred, d_all[val_ids], val_regimes
    )
    val_alpha = np.asarray([alpha_by_regime[r] for r in val_regimes], dtype=np.float32)
    calibrated_val = v252.normalize(
        (1.0 - val_alpha[:, None]) * val_anchor + val_alpha[:, None] * val_pred
    )
    validation_metrics = {
        "anchor_cos": float(np.mean(v252.row_cos(val_anchor, d_all[val_ids]))),
        "model_cos": float(np.mean(v252.row_cos(val_pred, d_all[val_ids]))),
        "calibrated_cos": float(np.mean(v252.row_cos(calibrated_val, d_all[val_ids]))),
        "path_cos": float(np.mean(v252.row_cos(val_path, d_all[val_ids]))),
        "gate": summarize(val_gate),
    }
    print(json.dumps({"validation": validation_metrics, "alpha": alpha_by_regime}, indent=2), flush=True)

    if validation_metrics["calibrated_cos"] <= validation_metrics["anchor_cos"] + 1e-5:
        report = {
            "method": VERSION,
            "status": "analysis_only_no_candidate",
            "primary_hypothesis": "A learned two-mesh expert can use accurate Poisson paths where available and wall-patch paths for coverage.",
            "validation": validation,
            "validation_metrics": validation_metrics,
            "calibration": calibration,
            "local_gate": None,
            "outputs": {"report": PATHS.relative(REPORT), "zip": None},
        }
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        return

    best_epoch = max(10, int(validation["best_epoch"]))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    final_anchor, final_w = v252.make_anchor(
        train_pos[valid_ids], train_blocked[valid_ids], train_pos[valid_ids], train_blocked[valid_ids], d_all[valid_ids],
        query_ids=valid_ids, bank_ids=valid_ids,
    )
    test_anchor, test_w = v252.make_anchor(
        test_pos, test_blocked, train_pos[valid_ids], train_blocked[valid_ids], d_all[valid_ids]
    )
    final_raw = combined_features(wall, poisson, "train", valid_ids, train_pos[valid_ids], train_blocked[valid_ids], final_anchor, final_w)
    test_raw = combined_features(wall, poisson, "test", None, test_pos, test_blocked, test_anchor, test_w)
    final_features, [test_features], feature_stats = v252.standardize(final_raw, [test_raw])
    seed_pred = []
    seed_path = []
    seed_gate = []
    final_training = []
    for seed in SEEDS:
        final_model, info = v252.train_model(
            final_features, final_anchor, d_all[valid_ids], seed, best_epoch, device, None
        )
        torch.save(
            {"state_dict": final_model.state_dict(), "feature_stats": feature_stats, "seed": seed, "epochs": best_epoch},
            CHECKPOINT_DIR / f"seed_{seed}.pt",
        )
        pred, path_d, gate = v252.predict(final_model, test_features, test_anchor, device)
        seed_pred.append(pred)
        seed_path.append(path_d)
        seed_gate.append(gate)
        final_training.append(info)
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    test_pred = v252.normalize(np.mean(np.stack(seed_pred), axis=0))
    test_path = v252.normalize(np.mean(np.stack(seed_path), axis=0))
    test_gate = np.mean(np.stack(seed_gate), axis=0).astype(np.float32)
    regimes = regime_of(test_blocked).astype(str)
    alpha = np.asarray([alpha_by_regime[r] for r in regimes], dtype=np.float32)
    current = v225.current_delay_marginals(v252.BASE)
    target_d = v252.normalize(
        (1.0 - alpha[:, None]) * current["d"] + alpha[:, None] * test_pred
    )
    np.save(TARGET_OUT, target_d)
    projection = v225.apply_delay_targets(
        v252.BASE,
        OUTPUT,
        {"d": target_d, "du": current["du"], "dh": current["dh"]},
        regimes,
        1.05,
    )
    score = v208.score_prediction(OUTPUT)
    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        alpha=alpha,
        test_gate=test_gate,
        model_vs_anchor=v252.row_cos(test_pred, test_anchor),
        path_vs_anchor=v252.row_cos(test_path, test_anchor),
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate" if score["score_traceback_leaked"] < v252.LOCAL_GATE else "gate_passed_no_zip_yet",
        "primary_hypothesis": "A learned two-mesh expert can use accurate Poisson paths where available and wall-patch paths for coverage.",
        "compliance_boundary": {
            "generation_uses": [
                "official train H PDP",
                "official train/test coordinates",
                "official point-cloud wall-patch and Poisson meshes",
                "scalar Sionna path fields as physics features",
                "official-H-supervised neural expert",
                "strict v239 carrier",
            ],
            "generation_does_not_use": [
                "Sionna CFR or MIMO RT H",
                "restored Etoile mesh/object/material labels",
                "synthetic channel labels",
                "v212/v217 teacher outputs",
                "local replay truth during generation or selection",
                "hidden online labels",
            ],
            "local_truth_usage": "one final diagnostic score after output generation",
        },
        "training": {
            "device": str(device),
            "best_epoch": best_epoch,
            "validation": validation,
            "validation_metrics": validation_metrics,
            "calibration": calibration,
            "alpha_by_regime": alpha_by_regime,
            "final_training": final_training,
        },
        "test_diagnostics": {
            "gate": summarize(test_gate),
            "alpha": summarize(alpha),
            "model_vs_anchor": summarize(v252.row_cos(test_pred, test_anchor)),
            "path_vs_anchor": summarize(v252.row_cos(test_path, test_anchor)),
        },
        "projection": {key: summarize(value) for key, value in projection.items()},
        "local_gate": {
            "required": v252.LOCAL_GATE,
            "score": score,
            "passed": bool(score["score_traceback_leaked"] >= v252.LOCAL_GATE),
            "gap": float(score["score_traceback_leaked"] - v252.LOCAL_GATE),
        },
        "outputs": {
            "prediction": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET_OUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

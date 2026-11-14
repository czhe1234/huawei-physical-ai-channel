#!/usr/bin/env python3
"""Deploy the transferable v445 path residual over the v559 channel carrier."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import torch

import analyze_v205_sionna_config_calibration as calibration
import analyze_v451_official_path_representation_probe as v451
import analyze_v452_v445_representation_over_strong_anchor as v452
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v424_sionna_pas_full_pdp_lock_diagnostic as v424
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
import generate_v435_train_calibrated_sionna_joint_sh_diagnostic as v435
import generate_v558_external_pas_v549_pdp_fusion as v558
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 561
VERSION = "v561_pretrained_path_residual_v559_pdp_lock"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUTPUT = CACHE_DIR / "v561_Round1_Test_Channel.npy"
ACTIONS = CACHE_DIR / "v561_pretrained_path_actions.npz"
CHECKPOINT = PATHS.checkpoints / VERSION / "v561_path_adapter.pt"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(559) / "v559_Round1_Test_Channel.npy"
REFERENCE = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
COMPATIBILITY_REPORT = (
    PATHS.version_report_dir(560) / "v560_strong_carrier_compatibility.json"
)
TARGET_SOURCE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
POSITION_SOURCE = PATHS.cache_dir(475) / "positions.npz"
FEATURE_SOURCE = (
    PATHS.cache_dir(560) / "official" / "embedding_legacy_full_k128.npy"
)
PATH_VALID_SOURCE = PATHS.cache_dir(477) / "v477_path_candidate_valid.npy"
GEOMETRY_SOURCE = PATHS.shared_features / "height_geometry_cache.npz"
PRETRAINED_CHECKPOINT = (
    PATHS.checkpoints / "v445_candidate_path_token_model" / "component_route_best.pt"
)

TRAIN_COUNT = 2000
TEST_COUNT = 500
MAX_PATHS = 128
PCA_DIM = 64
RIDGE = 100.0
STRENGTH = 0.30
SEED = 56_101
BATCH = 2
FLOOR = 1e-7
ACTION_CLIP = 6.0
ACTIVE_REGIMES = ("light", "medium", "heavy")
METRICS = v558.METRICS


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), FLOOR)
    value /= np.maximum(
        np.sum(value, axis=tuple(range(1, value.ndim)), keepdims=True),
        1e-30,
    )
    return value.astype(np.float32)


def summarize(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64)
    return {
        "mean": float(np.mean(value)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
        "max": float(np.max(value)),
    }


def action_summary(value: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    absolute = np.abs(value[mask]).reshape(int(np.sum(mask)), -1)
    return {
        "count": int(np.sum(mask)),
        "mean_abs": float(np.mean(absolute)),
        "p90_abs": float(np.quantile(absolute, 0.90)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "max_abs": float(np.max(absolute)),
    }


def fit_adapter() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    target = np.load(TARGET_SOURCE)
    positions = np.load(POSITION_SOURCE)["positions"].astype(np.float32)
    feature = np.load(FEATURE_SOURCE).astype(np.float32)
    path_valid = np.load(PATH_VALID_SOURCE, mmap_mode="r")[:, :MAX_PATHS]
    geometry = np.load(GEOMETRY_SOURCE)
    blocked = np.concatenate(
        [geometry["train_blocked"], geometry["test_blocked"]]
    ).astype(np.float32)
    if positions.shape[0] != TRAIN_COUNT + TEST_COUNT:
        raise ValueError(f"unexpected position count: {positions.shape}")
    if feature.shape[0] != len(positions):
        raise ValueError(f"embedding/position mismatch: {feature.shape}/{positions.shape}")

    path_count = np.sum(np.asarray(path_valid, dtype=np.int16), axis=1)
    complete = path_count < MAX_PATHS
    valid_ids = np.flatnonzero(target["valid"].astype(bool) & complete[:TRAIN_COUNT])
    test_ids = np.arange(TRAIN_COUNT, TRAIN_COUNT + TEST_COUNT, dtype=np.int64)
    truth_sh = target["sh"].astype(np.float32)
    train_anchor = v452.knn_anchor(
        valid_ids,
        valid_ids,
        positions[:, :2],
        blocked,
        truth_sh,
        exclude_self=True,
    )
    test_anchor = v452.knn_anchor(
        test_ids,
        valid_ids,
        positions[:, :2],
        blocked,
        truth_sh,
        exclude_self=False,
    )
    residual = v452.residual_target(truth_sh[valid_ids], train_anchor)
    local_ids = np.arange(len(valid_ids), dtype=np.int64)
    latent, pca = v451.prepare_latent_target(residual, local_ids, SEED)
    if pca.n_components_ != PCA_DIM:
        raise ValueError(f"unexpected PCA rank: {pca.n_components_}")

    fit_feature = feature[valid_ids]
    test_feature = feature[test_ids]
    feature_mean = np.mean(fit_feature, axis=0, dtype=np.float64).astype(np.float32)
    feature_std = np.std(fit_feature, axis=0, dtype=np.float64).astype(np.float32)
    feature_std = np.maximum(feature_std, 1e-4)
    fit_x = ((fit_feature - feature_mean) / feature_std).astype(np.float32)
    test_x = ((test_feature - feature_mean) / feature_std).astype(np.float32)
    coefficient, intercept = v451.ridge_fit(fit_x, latent, RIDGE)
    test_latent = test_x @ coefficient + intercept
    test_delta = pca.inverse_transform(test_latent).astype(np.float32)
    weak_model = v451.apply_delta(test_anchor, test_delta, STRENGTH)
    log_action = np.clip(
        np.log(np.maximum(weak_model, FLOOR))
        - np.log(np.maximum(test_anchor, FLOOR)),
        -ACTION_CLIP,
        ACTION_CLIP,
    ).reshape(TEST_COUNT, 192, 16).astype(np.float32)

    test_regimes = regime_of(geometry["test_blocked"]).astype("U8")
    active = np.isin(test_regimes, ACTIVE_REGIMES) & complete[TRAIN_COUNT:]
    checkpoint = {
        "version": VERSION,
        "pretrained_encoder_source": PATHS.relative(PRETRAINED_CHECKPOINT),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "ridge_coefficient": coefficient,
        "ridge_intercept": intercept,
        "pca_components": pca.components_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "test_path_embedding": test_feature.astype(np.float32),
        "test_knn_anchor": test_anchor.astype(np.float32),
        "test_log_action": log_action.astype(np.float32),
        "test_active": active.astype(np.bool_),
        "parameters": {
            "ridge": RIDGE,
            "strength": STRENGTH,
            "pca_dim": PCA_DIM,
            "seed": SEED,
            "active_regimes": ACTIVE_REGIMES,
            "max_paths": MAX_PATHS,
        },
    }
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, CHECKPOINT)
    adapter_report = {
        "fit_rows": int(len(valid_ids)),
        "feature_dim": int(feature.shape[1]),
        "pca_dim": int(pca.n_components_),
        "ridge": RIDGE,
        "strength": STRENGTH,
        "active_count": int(np.sum(active)),
        "fallback_count": int(np.sum(~active)),
        "saturated_test_count": int(np.sum(~complete[TRAIN_COUNT:])),
        "active_by_regime": {
            regime: int(np.sum(active & (test_regimes == regime)))
            for regime in np.unique(test_regimes)
        },
        "fallback_by_regime": {
            regime: int(np.sum((~active) & (test_regimes == regime)))
            for regime in np.unique(test_regimes)
        },
        "action": action_summary(log_action, active),
    }
    return log_action, active, test_regimes, adapter_report


def render(
    log_action: np.ndarray, active: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    base = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    diagnostics = {
        "target_l1": np.zeros(TEST_COUNT, dtype=np.float32),
        "achieved_l1": np.zeros(TEST_COUNT, dtype=np.float32),
        "target_error_l1": np.zeros(TEST_COUNT, dtype=np.float32),
        "pdp_lock_error": np.zeros(TEST_COUNT, dtype=np.float32),
        "power_ratio": np.ones(TEST_COUNT, dtype=np.float32),
        "action_nmse": np.zeros(TEST_COUNT, dtype=np.float32),
    }
    for start in range(0, TEST_COUNT, BATCH):
        end = min(start + BATCH, TEST_COUNT)
        base_h = np.asarray(base[start:end], dtype=np.complex64)
        candidate = base_h.copy()
        local_active = active[start:end]
        if np.any(local_active):
            ids = np.flatnonzero(local_active)
            active_h = base_h[ids]
            base_support, _ = v423.leaked_support(active_h)
            base_sh = v423.leaked_marginals(base_support)["sh"]
            target_sh = normalize_rows(
                base_sh * np.exp(log_action[start:end][ids]).astype(np.float32)
            )
            desired, _, _ = v435.build_desired_support(
                base_support,
                target_sh,
                np.ones(len(ids), dtype=np.float32),
            )
            rendered = v425.project_full_support(active_h, desired)
            rendered = v424.restore_full_linkwise_pdp(rendered, active_h)
            candidate[ids] = rendered
            actual_support, _ = v423.leaked_support(rendered)
            actual_sh = v423.leaked_marginals(actual_support)["sh"]
            global_ids = start + ids
            diagnostics["target_l1"][global_ids] = v423.l1_rows(target_sh, base_sh)
            diagnostics["achieved_l1"][global_ids] = v423.l1_rows(actual_sh, base_sh)
            diagnostics["target_error_l1"][global_ids] = v423.l1_rows(
                actual_sh, target_sh
            )
            diagnostics["pdp_lock_error"][global_ids] = v424.full_linkwise_pdp_error(
                rendered, active_h
            )
            base_power = np.sum(np.abs(active_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
            candidate_power = np.sum(
                np.abs(rendered) ** 2, axis=(1, 2, 3), dtype=np.float64
            )
            diagnostics["power_ratio"][global_ids] = (
                candidate_power / np.maximum(base_power, 1e-30)
            ).astype(np.float32)
            diagnostics["action_nmse"][global_ids] = (
                np.sum(np.abs(rendered - active_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
                / np.maximum(base_power, 1e-30)
            ).astype(np.float32)
        output[start:end] = candidate
        if end % 20 == 0 or end == TEST_COUNT:
            output.flush()
            print(f"v561 render {end}/{TEST_COUNT}", flush=True)
    output.flush()
    del output

    base_check = np.load(BASE, mmap_mode="r")
    candidate_check = np.load(OUTPUT, mmap_mode="r")
    inactive_error = 0.0
    for start in range(0, TEST_COUNT, 10):
        end = min(start + 10, TEST_COUNT)
        local = ~active[start:end]
        if np.any(local):
            inactive_error = max(
                inactive_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(candidate_check[start:end])[local]
                            - np.asarray(base_check[start:end])[local]
                        )
                    )
                ),
            )
    report = {
        "inactive_max_abs_error": inactive_error,
        "active_target_l1": summarize(diagnostics["target_l1"][active]),
        "active_achieved_l1": summarize(diagnostics["achieved_l1"][active]),
        "active_target_error_l1": summarize(diagnostics["target_error_l1"][active]),
        "pdp_lock_max": float(np.max(diagnostics["pdp_lock_error"])),
        "power_ratio": summarize(diagnostics["power_ratio"]),
        "active_action_nmse": summarize(diagnostics["action_nmse"][active]),
    }
    return diagnostics, report


def reference_audit() -> dict[str, object]:
    base = np.load(BASE, mmap_mode="r")
    candidate = np.load(OUTPUT, mmap_mode="r")
    reference = np.load(REFERENCE, mmap_mode="r")
    metrics = {
        name: {metric: np.empty(TEST_COUNT, dtype=np.float32) for metric in METRICS}
        for name in ("v559", "v561")
    }
    errors = {"v559": 0.0, "v561": 0.0}
    truth_energy = 0.0
    for start in range(0, TEST_COUNT, BATCH):
        end = min(start + BATCH, TEST_COUNT)
        truth_h = np.asarray(reference[start:end], dtype=np.complex64)
        values = {
            "v559": np.asarray(base[start:end], dtype=np.complex64),
            "v561": np.asarray(candidate[start:end], dtype=np.complex64),
        }
        views = calibration.spectral_views(truth_h)
        truth_energy += float(np.sum(np.abs(truth_h) ** 2, dtype=np.float64))
        for name, value in values.items():
            score = calibration.spectral_scores(value, views)
            for metric in METRICS:
                metrics[name][metric][start:end] = score[metric]
            errors[name] += float(np.sum(np.abs(value - truth_h) ** 2, dtype=np.float64))
    scores = {
        name: v558.finalize_score(metrics[name], errors[name], truth_energy)
        for name in ("v559", "v561")
    }
    return {
        "warning": (
            "v207 same-coordinate Sionna is diagnostic only and is not used to select "
            "the adapter, its strength, active regimes, or renderer."
        ),
        "scores": scores,
        "delta": {
            key: float(scores["v561"][key] - scores["v559"][key])
            for key in scores["v559"]
        },
    }


def package(report: dict[str, object]) -> dict[str, object]:
    dependencies = (
        "generate_v561_pretrained_path_residual_v559.py",
        "analyze_v560_strong_carrier_compatibility.py",
        "analyze_v451_official_path_representation_probe.py",
        "analyze_v452_v445_representation_over_strong_anchor.py",
        "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
        "generate_v424_sionna_pas_full_pdp_lock_diagnostic.py",
        "generate_v425_sionna_full_slice_pas_completion_diagnostic.py",
        "generate_v435_train_calibrated_sionna_joint_sh_diagnostic.py",
        "powerfield_common.py",
        "project_paths.py",
    )
    for name in dependencies:
        require(PATHS.code / name)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.writestr("model/", b"")
        archive.write(CHECKPOINT, "model/v561_path_adapter.pt")
        archive.write(REPORT, "model/v561_report.json")
        for name in dependencies:
            archive.write(PATHS.code / name, name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise ValueError(f"ZIP CRC failure: {bad_member}")
    channel = np.load(OUTPUT, mmap_mode="r")
    finite = all(
        np.isfinite(np.asarray(channel[start : start + 5])).all()
        for start in range(0, len(channel), 5)
    )
    return {
        "shape": list(channel.shape),
        "dtype": str(channel.dtype),
        "finite": bool(finite),
        "zip_crc": True,
        "zip_members": members,
        "zip_bytes": int(ZIP_OUT.stat().st_size),
        "checkpoint_bytes": int(CHECKPOINT.stat().st_size),
    }


def main() -> None:
    required = (
        BASE,
        REFERENCE,
        COMPATIBILITY_REPORT,
        TARGET_SOURCE,
        POSITION_SOURCE,
        FEATURE_SOURCE,
        PATH_VALID_SOURCE,
        GEOMETRY_SOURCE,
        PRETRAINED_CHECKPOINT,
    )
    for path in required:
        require(path)
    compatibility = json.loads(COMPATIBILITY_REPORT.read_text(encoding="utf-8"))
    if not compatibility["decision"]["passed"]:
        raise RuntimeError("v560 strong-carrier gate did not pass")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log_action, active, regimes, adapter = fit_adapter()
    diagnostics, renderer = render(log_action, active)
    np.savez_compressed(
        ACTIONS,
        log_action=log_action.astype(np.float16),
        active=active,
        regimes=regimes,
        **diagnostics,
    )
    audit = reference_audit()
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The v445 encoder learned transferable Sionna path-existence and energy "
            "structure. Its low-rank official-H residual can correct the remaining "
            "joint delay-angle spectrum of v559 without changing v559 linkwise PDP or power."
        ),
        "new_information": [
            "v445 encoder pretrained on 200k synthetic Sionna path-token examples",
            "official-H low-rank residual calibration on 1944 complete training rows",
            "v560 strong-carrier compatibility evidence with matched negative controls",
        ],
        "parameters": {
            "pca_dim": PCA_DIM,
            "ridge": RIDGE,
            "strength": STRENGTH,
            "active_regimes": list(ACTIVE_REGIMES),
            "saturated_path_fallback": f"path_count>={MAX_PATHS}",
            "projection_cycles": 1,
            "pdp_and_power_lock": "exact v559 per-link CIR magnitude restoration",
        },
        "compatibility_decision": compatibility["decision"],
        "adapter": adapter,
        "renderer": renderer,
        "reference_audit": audit,
        "outputs": {
            "channel": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTIONS),
            "checkpoint": PATHS.relative(CHECKPOINT),
            "report": PATHS.relative(REPORT),
            "archive": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report["integrity"] = package(report)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "archive": report["outputs"]["archive"],
                "adapter": adapter,
                "renderer": renderer,
                "reference_delta": audit["delta"],
                "integrity": report["integrity"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

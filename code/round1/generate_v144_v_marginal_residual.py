#!/usr/bin/env python3
"""Generate v144 with a learned continuous V-marginal residual correction.

The only new variable relative to the online-best v140 is a low-strength
continuous ratio over the eight V-angle bins.  The ratio model uses the
matched train/test v129 residual probabilities established after v143.
H-angle overlap, delay, phase, total power, and polarization/UE allocation
remain governed by the v140 target and projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from analyze_post_v143_hidden_information import normalize_without_floor, support_score
from analyze_v143_matched_residual_probability import basic_features
from generate_v112_kernel_moe_path_support import normalize_support
from generate_v140_balanced_hminus2_overdrive import build_risk, row_cos, support_features
from generate_v143_transductive_geodesic_risk import (
    apply_risk_shift,
    digest,
    output_comparison,
)
from generate_v87_easy_field_projection import summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v144_v_marginal_residual"
OUT_DIR = PATHS.cache_dir(144)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(144)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v144_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v144_v_marginal_residual_target.npy"
PREDICTION_OUT = OUT_DIR / "v144_v_marginal_log_ratio.npy"
REPORT = REPORT_DIR / "v144_v_marginal_residual_generate.json"
ZIP_OUT = PATHS.round1_versions / "v144_v_marginal_residual_candidate_pending.zip"

RANDOM_STATE = 14403
STRENGTH = 0.50
N_ESTIMATORS = 280
MIN_SAMPLES_LEAF = 5


def make_features(
    prefix: str,
    positions: np.ndarray,
    blocked: np.ndarray,
    current: np.ndarray,
    geodesic: np.lib.npyio.NpzFile,
    pointcloud: np.lib.npyio.NpzFile,
    residual_probability: np.ndarray,
) -> np.ndarray:
    basic, _ = basic_features(positions, blocked)
    values = np.concatenate(
        [
            basic,
            geodesic[f"{prefix}_features"].astype(np.float32),
            pointcloud[f"{prefix}_features"].astype(np.float32),
            residual_probability.astype(np.float32),
            support_features(current),
        ],
        axis=1,
    )
    return np.nan_to_num(values, nan=0.0, posinf=8.0, neginf=-8.0).astype(
        np.float32
    )


def v_marginal(support: np.ndarray) -> np.ndarray:
    return support.sum(axis=(1, 3))


def apply_v_log_ratio(
    base: np.ndarray, prediction: np.ndarray, strength: float
) -> np.ndarray:
    ratio = np.exp(
        np.clip(prediction, -3.0, 3.0) * float(strength)
    ).astype(np.float32)
    output = (base * ratio[:, None, :, None]).astype(np.float32)
    output /= np.maximum(output.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return output


def summarize_array(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "p10": float(np.quantile(flat, 0.10)),
        "median": float(np.quantile(flat, 0.50)),
        "p90": float(np.quantile(flat, 0.90)),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


def file_digest(path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    positions_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(
        np.float32
    )
    positions_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(
        np.float32
    )
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked_train = height["train_blocked"].astype(np.float32)
    blocked_test = height["test_blocked"].astype(np.float32)
    regimes_train = regime_of(blocked_train)
    regimes_test = regime_of(blocked_test)
    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    pointcloud = np.load(
        PATHS.shared_features / "official_pointcloud_scene_cache.npz"
    )
    geodesic_names = [str(value) for value in geodesic["feature_names"]]
    probability_train = np.load(
        PATHS.shared_features / "train_v129_crossfit_residual_probs.npy"
    ).astype(np.float32)
    probability_test = np.load(
        PATHS.cache_dir(129) / "v129_apex_badpoint_support_residual_probs.npy"
    ).astype(np.float32)

    train_source = normalize_support(
        np.load(PATHS.cache_dir(128) / "train_v125eq_collapsed_support.npy"),
        1e-12,
    )
    train_risk = build_risk(
        regimes_train,
        probability_train,
        geodesic["train_features"].astype(np.float32),
        geodesic_names,
    )
    train_current, _ = apply_risk_shift(train_source, train_risk)
    test_current = normalize_support(
        np.load(
            PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_target.npy"
        ),
        1e-12,
    )

    truth_raw = np.load(PATHS.shared_features / "truth_coll_power.npy").astype(
        np.float32
    )
    valid = truth_raw.sum(axis=(1, 2, 3)) > 0
    truth = normalize_without_floor(truth_raw)
    log_ratio = np.log(np.maximum(v_marginal(truth), 1e-10)) - np.log(
        np.maximum(v_marginal(train_current), 1e-10)
    )
    log_ratio = np.clip(log_ratio, -3.0, 3.0).astype(np.float32)

    features_train = make_features(
        "train",
        positions_train,
        blocked_train,
        train_current,
        geodesic,
        pointcloud,
        probability_train,
    )
    features_test = make_features(
        "test",
        positions_test,
        blocked_test,
        test_current,
        geodesic,
        pointcloud,
        probability_test,
    )
    if features_train.shape[1] != features_test.shape[1]:
        raise ValueError("train/test feature width mismatch")

    selected = np.flatnonzero(valid)
    groups = KMeans(
        n_clusters=25, random_state=RANDOM_STATE, n_init=20
    ).fit_predict(positions_train[:, :2])
    oof_prediction = np.zeros_like(log_ratio, dtype=np.float32)
    fold_reports: list[dict[str, int]] = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train_local, validation_local) in enumerate(
        splitter.split(features_train[selected], groups=groups[selected])
    ):
        train_index = selected[train_local]
        validation_index = selected[validation_local]
        model = ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS,
            max_features="sqrt",
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
        )
        model.fit(features_train[train_index], log_ratio[train_index])
        oof_prediction[validation_index] = model.predict(
            features_train[validation_index]
        ).astype(np.float32)
        fold_reports.append(
            {
                "fold": fold,
                "train_count": int(len(train_index)),
                "validation_count": int(len(validation_index)),
            }
        )

    current_score = support_score(train_current, truth)
    oof_adjusted = apply_v_log_ratio(train_current, oof_prediction, STRENGTH)
    oof_gain = support_score(oof_adjusted, truth) - current_score

    final_model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_model.fit(features_train[selected], log_ratio[selected])
    test_prediction = final_model.predict(features_test).astype(np.float32)
    final_target = apply_v_log_ratio(test_current, test_prediction, STRENGTH)
    np.save(PREDICTION_OUT, test_prediction)
    np.save(TARGET_OUT, final_target)

    weights = np.load(
        PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_weights.npy"
    ).astype(np.float32)
    args = argparse.Namespace(
        base=str(PATHS.cache_dir(129) / "v129_Round1_Test_Channel.npy"),
        output=str(OUTPUT),
        beta=140.3,
        gamma=0.10,
        eta="clear:0.0,light:0.028,medium:0.470,heavy:0.780",
        min_gate=0.12,
        max_gate=2.95,
        conf_center=0.22,
        conf_width=0.38,
        entropy_low=0.02,
        entropy_high=0.90,
        gain_clip="0.34,3.20",
        top_gain_clip="0.48,2.35",
        top_components=5632,
        target_floor=1e-12,
        project_batch=20,
    )
    projection_report = projection(
        final_target, weights, regimes_test, positions_test.astype(np.float64), args
    )

    output = np.load(OUTPUT, mmap_mode="r")
    v140_output = np.load(
        PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", mmap_mode="r"
    )
    comparison = output_comparison(output, v140_output, regimes_test)
    if output.shape != v140_output.shape or output.dtype != np.complex64:
        raise ValueError(
            f"invalid output layout: shape={output.shape} dtype={output.dtype}"
        )
    if not comparison["finite"]:
        raise ValueError("v144 output contains non-finite values")

    target_cos = row_cos(final_target, test_current)
    v_before = v_marginal(test_current)
    v_after = v_marginal(final_target)
    report_args = dict(vars(args))
    report_args["base"] = PATHS.relative(args.base)
    report_args["output"] = PATHS.relative(args.output)
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "base": "v140 online 0.501352; complex base remains v129",
        "hypothesis": (
            "the dominant learnable post-v140 residual is continuous V-angle "
            "marginal allocation, not an integer V/delay shift"
        ),
        "only_primary_change": (
            "apply a learned eight-bin V-marginal log-ratio at strength 0.50 "
            "to the v140 support target"
        ),
        "preserved_from_v140": {
            "h_log_overlap": True,
            "delay_structure_within_each_v_bin": True,
            "conditional_h_delay_structure_within_each_v_bin": True,
            "projection_weights": PATHS.relative(
                PATHS.cache_dir(140)
                / "v140_balanced_hminus2_overdrive_weights.npy"
            ),
            "projection_args": report_args,
            "complex_phase": True,
            "total_power": True,
            "pol_ue_allocation": True,
        },
        "model": {
            "feature_dim": int(features_train.shape[1]),
            "features": (
                "basic position/blockage + geodesic + official pointcloud + "
                "matched v129 residual probabilities + current support signature"
            ),
            "target": "clipped log truth/current V-marginal ratio",
            "target_bins": int(log_ratio.shape[1]),
            "strength": STRENGTH,
            "n_estimators": N_ESTIMATORS,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "random_state": RANDOM_STATE,
            "valid_train_points": int(valid.sum()),
            "train_probability": PATHS.relative(
                PATHS.shared_features / "train_v129_crossfit_residual_probs.npy"
            ),
            "test_probability": PATHS.relative(
                PATHS.cache_dir(129)
                / "v129_apex_badpoint_support_residual_probs.npy"
            ),
            "train_target_log_ratio": summarize_array(log_ratio[valid]),
            "oof_prediction": summarize_array(oof_prediction[valid]),
            "test_prediction": summarize_array(test_prediction),
            "folds": fold_reports,
        },
        "spatial_oof": {
            "groups": 25,
            "folds": 5,
            "gain": summarize(oof_gain[valid]),
            "positive_fraction": float(np.mean(oof_gain[valid] > 0)),
        },
        "target_diagnostics": {
            "target_to_v140_target": summarize(target_cos),
            "v_marginal_l1_change": summarize(
                np.sum(np.abs(v_after - v_before), axis=1)
            ),
            "v_marginal_max_ratio": summarize(
                np.max(v_after / np.maximum(v_before, 1e-12), axis=1)
            ),
        },
        "projection": projection_report,
        "comparison_to_v140": comparison,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET_OUT),
            "prediction": PATHS.relative(PREDICTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
        "integrity": {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": comparison["finite"],
            "output_bytes": int(output.nbytes),
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source_path, name in (
            (__file__, "generate_v144_v_marginal_residual.py"),
            (
                PATHS.code / "generate_v140_balanced_hminus2_overdrive.py",
                "generate_v140_balanced_hminus2_overdrive.py",
            ),
            (
                PATHS.code / "analyze_v143_marginal_residual_learnability.py",
                "analyze_v143_marginal_residual_learnability.py",
            ),
            (
                PATHS.code / "generate_v98_path_ensemble_graph_attention_joint_ad.py",
                "generate_v98_path_ensemble_graph_attention_joint_ad.py",
            ),
            (PATHS.code / "powerfield_common.py", "powerfield_common.py"),
            (PATHS.code_root / "project_paths.py", "project_paths.py"),
        ):
            archive.write(source_path, name)

    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise ValueError(f"ZIP CRC failure: {bad_member}")
    report["integrity"].update(
        {
            "zip_crc": True,
            "zip_members": members,
            "sha256": digest(ZIP_OUT, "sha256"),
            "md5": file_digest(ZIP_OUT, "md5"),
        }
    )
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {ZIP_OUT}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

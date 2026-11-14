#!/usr/bin/env python3
"""Generate v152 with a reverse-source-matched local H residual field."""

from __future__ import annotations

import argparse
import json
import zipfile

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from analyze_post_v143_hidden_information import normalize_without_floor, support_score
from analyze_v145_gated_local_h_field import local_h_predictions
from generate_v112_kernel_moe_path_support import normalize_support
from generate_v140_balanced_hminus2_overdrive import build_risk, row_cos, support_features
from generate_v143_transductive_geodesic_risk import (
    apply_risk_shift,
    digest,
    output_comparison,
)
from generate_v145_local_h_residual_field import local_h_field
from generate_v87_easy_field_projection import summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v152_reverse_source_local_h_field"
OUT_DIR = PATHS.cache_dir(152)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(152)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v152_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v152_reverse_source_local_h_field_target.npy"
FIELD_OUT = OUT_DIR / "v152_reverse_source_local_h_field.npz"
REPORT = REPORT_DIR / "v152_reverse_source_local_h_field_generate.json"
ZIP_OUT = PATHS.round1_versions / "v152_reverse_source_local_h_field_candidate_pending.zip"

RANDOM_STATE = 15201
REVERSE_K = 8
LOCAL_K = 8
DISTANCE_FLOOR = 0.25
MAX_DISTANCE = 20.0
STRENGTH = 0.10
STRENGTH_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.35)
REGIMES = ("clear", "light", "medium", "heavy")


def normalize_h(values: np.ndarray) -> np.ndarray:
    return (
        values / np.maximum(values.sum(axis=1, keepdims=True), TINY)
    ).astype(np.float32)


def interpolate_h_marginal(
    positions_source: np.ndarray,
    h_source: np.ndarray,
    positions_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = NearestNeighbors(n_neighbors=REVERSE_K, n_jobs=-1)
    model.fit(positions_source[:, :2])
    distance, local = model.kneighbors(positions_query[:, :2])
    inverse = 1.0 / np.maximum(distance, DISTANCE_FLOOR)
    weight = inverse / np.maximum(inverse.sum(axis=1, keepdims=True), 1e-12)
    prediction = np.sum(weight[..., None] * h_source[local], axis=1)
    return (
        normalize_h(prediction),
        local.astype(np.int32),
        distance.astype(np.float32),
        weight.astype(np.float32),
    )


def replace_h_marginal(current: np.ndarray, h_target: np.ndarray) -> np.ndarray:
    current_h = current.sum(axis=(2, 3))
    ratio = h_target / np.maximum(current_h, 1e-10)
    output = (current * ratio[:, :, None, None]).astype(np.float32)
    output /= np.maximum(output.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return output


def domain_auc(train_support: np.ndarray, test_support: np.ndarray) -> dict[str, float]:
    x = np.concatenate(
        [support_features(train_support), support_features(test_support)], axis=0
    )
    y = np.concatenate(
        [
            np.zeros(len(train_support), dtype=np.int32),
            np.ones(len(test_support), dtype=np.int32),
        ]
    )
    prediction = np.empty(len(y), dtype=np.float32)
    for train_index, validation_index in StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    ).split(x, y):
        model = ExtraTreesClassifier(
            n_estimators=180,
            max_features="sqrt",
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x[train_index], y[train_index])
        prediction[validation_index] = model.predict_proba(
            x[validation_index]
        )[:, 1]
    return {"auc": float(roc_auc_score(y, prediction))}


def adjust_support(
    current: np.ndarray,
    prediction: np.ndarray,
    gate: np.ndarray,
    strength: float,
) -> np.ndarray:
    ratio = np.exp(
        np.clip(prediction, -3.0, 3.0)
        * (strength * gate.astype(np.float32))[:, None]
    ).astype(np.float32)
    output = (current * ratio[:, :, None, None]).astype(np.float32)
    output /= np.maximum(output.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return output


def adjust_h(
    current_h: np.ndarray,
    prediction: np.ndarray,
    gate: np.ndarray,
    strength: float,
) -> np.ndarray:
    ratio = np.exp(
        np.clip(prediction, -3.0, 3.0)
        * (strength * gate.astype(np.float32))[:, None]
    ).astype(np.float32)
    output = current_h * ratio
    return normalize_h(output)


def nested_oof_audit(
    positions_train: np.ndarray,
    reverse_current: np.ndarray,
    truth: np.ndarray,
    reverse_h: np.ndarray,
    truth_h: np.ndarray,
    h_target: np.ndarray,
    valid: np.ndarray,
    regimes_train: np.ndarray,
) -> dict[str, object]:
    groups = KMeans(
        n_clusters=25, random_state=RANDOM_STATE, n_init=20
    ).fit_predict(positions_train[:, :2])
    selected = np.flatnonzero(valid)
    outer = GroupKFold(n_splits=5)
    gain_by_strength = {
        strength: np.zeros(len(positions_train), dtype=np.float32)
        for strength in STRENGTH_GRID
    }
    h_gain_by_strength = {
        strength: np.zeros(len(positions_train), dtype=np.float32)
        for strength in STRENGTH_GRID
    }
    active_by_strength = {
        strength: np.zeros(len(positions_train), dtype=bool)
        for strength in STRENGTH_GRID
    }
    fold_reports = []

    for fold, (train_local, validation_local) in enumerate(
        outer.split(selected, groups=groups[selected])
    ):
        train_index = selected[train_local]
        validation_index = selected[validation_local]
        prediction, _, distance = local_h_predictions(
            positions_train, h_target, train_index, validation_index
        )
        pred = prediction[LOCAL_K]
        gate = distance[LOCAL_K] <= MAX_DISTANCE
        strength_rows = []
        for strength in STRENGTH_GRID:
            adjusted = adjust_support(
                reverse_current[validation_index], pred, gate, strength
            )
            gain = (
                support_score(adjusted, truth[validation_index])
                - support_score(reverse_current[validation_index], truth[validation_index])
            )
            adjusted_h = adjust_h(reverse_h[validation_index], pred, gate, strength)
            h_gain = (
                row_cos(adjusted_h, truth_h[validation_index])
                - row_cos(reverse_h[validation_index], truth_h[validation_index])
            )
            gain_by_strength[strength][validation_index] = gain
            h_gain_by_strength[strength][validation_index] = h_gain
            active_by_strength[strength][validation_index] = gate
            strength_rows.append(
                {
                    "strength": strength,
                    "full_gain": summarize(gain),
                    "h_gain": summarize(h_gain),
                    "active_fraction": float(np.mean(gate)),
                    "positive_fraction": float(np.mean(gain > 0.0)),
                }
            )
        fold_reports.append(
            {
                "fold": fold,
                "train_count": int(len(train_index)),
                "validation_count": int(len(validation_index)),
                "strengths": strength_rows,
            }
        )

    strength_summary = {}
    for strength in STRENGTH_GRID:
        gain = gain_by_strength[strength][valid]
        h_gain = h_gain_by_strength[strength][valid]
        active = active_by_strength[strength][valid]
        fold_means = []
        for fold in fold_reports:
            row = next(
                item
                for item in fold["strengths"]
                if item["strength"] == strength
            )
            fold_means.append(row["full_gain"]["mean"])
        by_regime = {}
        for regime in REGIMES:
            mask = valid & (regimes_train == regime)
            by_regime[regime] = {
                "count": int(mask.sum()),
                "gain": summarize(gain_by_strength[strength][mask]),
                "active_fraction": float(
                    np.mean(active_by_strength[strength][mask])
                ),
            }
        strength_summary[f"{strength:.2f}"] = {
            "full_gain": summarize(gain),
            "h_gain": summarize(h_gain),
            "positive_fraction": float(np.mean(gain > 0.0)),
            "active_fraction": float(np.mean(active)),
            "active_gain": summarize(gain[active]),
            "fold_means": fold_means,
            "all_folds_positive": bool(np.all(np.asarray(fold_means) > 0.0)),
            "by_regime": by_regime,
        }
    return {
        "outer_folds": 5,
        "local_k": LOCAL_K,
        "maximum_distance": MAX_DISTANCE,
        "strength_grid": list(STRENGTH_GRID),
        "selected_strength": STRENGTH,
        "strengths": strength_summary,
        "folds": fold_reports,
    }


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
    geodesic_names = [str(value) for value in geodesic["feature_names"]]
    probability_train = np.load(
        PATHS.shared_features / "train_v129_crossfit_residual_probs.npy"
    ).astype(np.float32)

    source = normalize_support(
        np.load(PATHS.cache_dir(128) / "train_v125eq_collapsed_support.npy"),
        1e-12,
    )
    train_risk = build_risk(
        regimes_train,
        probability_train,
        geodesic["train_features"].astype(np.float32),
        geodesic_names,
    )
    train_current, _ = apply_risk_shift(source, train_risk)
    truth_raw = np.load(PATHS.shared_features / "truth_coll_power.npy").astype(
        np.float32
    )
    valid = truth_raw.sum(axis=(1, 2, 3)) > 0
    truth = normalize_without_floor(truth_raw)
    truth_h = truth.sum(axis=(2, 3))

    v140_target = normalize_support(
        np.load(
            PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_target.npy"
        ),
        1e-12,
    )
    test_h = v140_target.sum(axis=(2, 3))
    reverse_h, reverse_neighbors, reverse_distances, reverse_weights = (
        interpolate_h_marginal(positions_test, test_h, positions_train)
    )
    reverse_current = replace_h_marginal(train_current, reverse_h)
    h_target = np.clip(
        np.log(np.maximum(truth_h, 1e-10))
        - np.log(np.maximum(reverse_h, 1e-10)),
        -3.0,
        3.0,
    ).astype(np.float32)

    audit = nested_oof_audit(
        positions_train,
        reverse_current,
        truth,
        reverse_h,
        truth_h,
        h_target,
        valid,
        regimes_train,
    )
    selected_summary = audit["strengths"][f"{STRENGTH:.2f}"]
    accepted_local = bool(
        selected_summary["full_gain"]["mean"] > 0.0
        and selected_summary["all_folds_positive"]
    )

    prediction, dispersion, gate, neighbors, graph_values = local_h_field(
        positions_train,
        positions_test,
        h_target,
        valid,
    )
    ratio = np.exp(
        np.clip(prediction, -3.0, 3.0)
        * (STRENGTH * gate.astype(np.float32))[:, None]
    ).astype(np.float32)
    final_target = (v140_target * ratio[:, :, None, None]).astype(np.float32)
    final_target /= np.maximum(
        final_target.sum(axis=(1, 2, 3), keepdims=True), TINY
    )
    np.save(TARGET_OUT, final_target)
    np.savez_compressed(
        FIELD_OUT,
        reverse_h=reverse_h,
        reverse_neighbors=reverse_neighbors,
        reverse_distances=reverse_distances,
        reverse_weights=reverse_weights,
        train_h_residual=h_target,
        prediction=prediction,
        dispersion=dispersion,
        gate=gate,
        neighbor_indices=neighbors,
        neighbor_distances=graph_values[:, :LOCAL_K],
        neighbor_weights=graph_values[:, LOCAL_K:],
        selected_strength=np.full(len(positions_test), STRENGTH, dtype=np.float32)
        * gate.astype(np.float32),
    )

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
    baseline = np.load(
        PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", mmap_mode="r"
    )
    comparison = output_comparison(output, baseline, regimes_test)
    if output.shape != baseline.shape or output.dtype != np.complex64:
        raise ValueError(
            f"invalid output layout: shape={output.shape} dtype={output.dtype}"
        )
    if not comparison["finite"]:
        raise ValueError("v152 output contains non-finite values")

    v145_target = normalize_support(
        np.load(PATHS.cache_dir(145) / "v145_local_h_residual_field_target.npy"),
        1e-12,
    )
    final_h = final_target.sum(axis=(2, 3))
    v145_h = v145_target.sum(axis=(2, 3))
    report_args = dict(vars(args))
    report_args["base"] = PATHS.relative(args.base)
    report_args["output"] = PATHS.relative(args.output)
    report = {
        "method": VERSION,
        "status": "candidate_pending" if accepted_local else "rejected_local",
        "base": "v140 online 0.501352; v145 action family with reverse-source residuals",
        "primary_hypothesis": (
            "v150/v151 failed mainly because train residual supervision used a "
            "source that was not homologous with the test v140 generator; using "
            "the observed test v140 H marginal as a transductive reverse source "
            "should produce a more relevant local H residual field"
        ),
        "only_primary_change": (
            "replace the train current H marginal used to form residual labels "
            "with an inverse-distance interpolation of the test v140 H marginal; "
            "project the resulting local H field at strength 0.10"
        ),
        "reverse_source": {
            "reverse_k": REVERSE_K,
            "distance_floor": DISTANCE_FLOOR,
            "train_current_vs_truth": summarize(
                support_score(train_current[valid], truth[valid])
            ),
            "reverse_current_vs_truth": summarize(
                support_score(reverse_current[valid], truth[valid])
            ),
            "train_current_to_test_v140_domain": domain_auc(train_current, v140_target),
            "reverse_current_to_test_v140_domain": domain_auc(
                reverse_current, v140_target
            ),
            "nearest_test_distance": summarize(reverse_distances[:, 0]),
            "kth_test_distance": summarize(reverse_distances[:, REVERSE_K - 1]),
        },
        "nested_spatial_oof": audit,
        "accepted_local": accepted_local,
        "test_decision": {
            "local_k": LOCAL_K,
            "strength": STRENGTH,
            "maximum_distance": MAX_DISTANCE,
            "active_fraction": float(np.mean(gate > 0.0)),
            "active_by_regime": {
                regime: float(np.mean(gate[regimes_test == regime] > 0.0))
                for regime in REGIMES
            },
            "prediction": summarize(prediction),
            "dispersion": summarize(dispersion),
            "eighth_neighbor_distance": summarize(graph_values[:, LOCAL_K - 1]),
        },
        "target_diagnostics": {
            "target_to_v140_target": summarize(row_cos(final_target, v140_target)),
            "h_marginal_l1_change": summarize(np.sum(np.abs(final_h - test_h), axis=1)),
            "h_marginal_max_ratio": summarize(
                np.max(final_h / np.maximum(test_h, 1e-12), axis=1)
            ),
            "v145_target_to_v140_target": summarize(row_cos(v145_target, v140_target)),
            "v145_h_marginal_l1_change": summarize(
                np.sum(np.abs(v145_h - test_h), axis=1)
            ),
        },
        "projection": projection_report,
        "comparison_to_v140": comparison,
        "preserved_from_v140": {
            "geodesic_risk_and_h_shift": True,
            "projection_weights": PATHS.relative(
                PATHS.cache_dir(140)
                / "v140_balanced_hminus2_overdrive_weights.npy"
            ),
            "projection_args": report_args,
            "complex_phase": True,
            "total_power": True,
            "pol_ue_allocation": True,
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET_OUT),
            "field": PATHS.relative(FIELD_OUT),
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
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if accepted_local:
        with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.write(OUTPUT, "Round1_Test_Channel.npy")
            for source_path, name in (
                (__file__, "generate_v152_reverse_source_local_h_field.py"),
                (
                    PATHS.code / "generate_v145_local_h_residual_field.py",
                    "generate_v145_local_h_residual_field.py",
                ),
                (
                    PATHS.code / "analyze_v145_gated_local_h_field.py",
                    "analyze_v145_gated_local_h_field.py",
                ),
                (
                    PATHS.code / "generate_v140_balanced_hminus2_overdrive.py",
                    "generate_v140_balanced_hminus2_overdrive.py",
                ),
                (
                    PATHS.code / "generate_v143_transductive_geodesic_risk.py",
                    "generate_v143_transductive_geodesic_risk.py",
                ),
                (
                    PATHS.code / "generate_v112_kernel_moe_path_support.py",
                    "generate_v112_kernel_moe_path_support.py",
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
                "md5": digest(ZIP_OUT, "md5"),
            }
        )
    else:
        if ZIP_OUT.exists():
            ZIP_OUT.unlink()
        report["integrity"].update({"zip_crc": None, "zip_members": []})

    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {REPORT}")
    if accepted_local:
        print(f"wrote {ZIP_OUT}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

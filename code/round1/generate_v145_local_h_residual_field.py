#!/usr/bin/env python3
"""Generate v145 with a local spatial H-marginal residual field over v140."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile

import numpy as np
from sklearn.neighbors import NearestNeighbors

from analyze_post_v143_hidden_information import normalize_without_floor
from generate_v112_kernel_moe_path_support import normalize_support
from generate_v140_balanced_hminus2_overdrive import build_risk, row_cos
from generate_v143_transductive_geodesic_risk import (
    apply_risk_shift,
    digest,
    output_comparison,
)
from generate_v87_easy_field_projection import summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v145_local_h_residual_field"
OUT_DIR = PATHS.cache_dir(145)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(145)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v145_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v145_local_h_residual_field_target.npy"
FIELD_OUT = OUT_DIR / "v145_local_h_residual_field.npz"
REPORT = REPORT_DIR / "v145_local_h_residual_field_generate.json"
AUDIT = REPORT_DIR / "v145_gated_local_h_field_audit.json"
ZIP_OUT = PATHS.round1_versions / "v145_local_h_residual_field_candidate_pending.zip"

K = 8
STRENGTH = 0.25
DISTANCE_FLOOR = 0.25
MAX_DISTANCE = 20.0


def file_digest(path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def local_h_field(
    positions_train: np.ndarray,
    positions_test: np.ndarray,
    train_log_ratio: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_index = np.flatnonzero(valid)
    model = NearestNeighbors(n_neighbors=K, n_jobs=-1)
    model.fit(positions_train[train_index, :2])
    distance, local = model.kneighbors(positions_test[:, :2])
    neighbors = train_index[local]
    inverse = 1.0 / np.maximum(distance, DISTANCE_FLOOR)
    weight = inverse / np.maximum(inverse.sum(axis=1, keepdims=True), 1e-12)
    values = train_log_ratio[neighbors]
    prediction = np.sum(weight[..., None] * values, axis=1).astype(np.float32)
    variance = np.sum(
        weight[..., None] * (values - prediction[:, None]) ** 2, axis=1
    )
    dispersion = np.sqrt(np.mean(variance, axis=1)).astype(np.float32)
    gate = (distance[:, -1] <= MAX_DISTANCE).astype(np.float32)
    return (
        prediction,
        dispersion,
        gate,
        neighbors.astype(np.int32),
        np.column_stack([distance, weight]).astype(np.float32),
    )


def main() -> None:
    positions_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(
        np.float32
    )
    positions_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(
        np.float64
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
    train_current_h = train_current.sum(axis=(2, 3))
    truth_h = truth.sum(axis=(2, 3))
    train_log_ratio = np.clip(
        np.log(np.maximum(truth_h, 1e-10))
        - np.log(np.maximum(train_current_h, 1e-10)),
        -3.0,
        3.0,
    ).astype(np.float32)

    v140_target = normalize_support(
        np.load(
            PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_target.npy"
        ),
        1e-12,
    )
    prediction, dispersion, gate, neighbors, graph_values = local_h_field(
        positions_train,
        positions_test.astype(np.float32),
        train_log_ratio,
        valid,
    )
    ratio = np.exp(
        np.clip(prediction, -3.0, 3.0)
        * (STRENGTH * gate)[:, None]
    ).astype(np.float32)
    final_target = (v140_target * ratio[:, :, None, None]).astype(np.float32)
    final_target /= np.maximum(
        final_target.sum(axis=(1, 2, 3), keepdims=True), TINY
    )
    np.save(TARGET_OUT, final_target)
    np.savez_compressed(
        FIELD_OUT,
        prediction=prediction,
        dispersion=dispersion,
        gate=gate,
        neighbor_indices=neighbors,
        neighbor_distances=graph_values[:, :K],
        neighbor_weights=graph_values[:, K:],
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
        final_target, weights, regimes_test, positions_test, args
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
        raise ValueError("v145 output contains non-finite values")

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    v140_h = v140_target.sum(axis=(2, 3))
    final_h = final_target.sum(axis=(2, 3))
    report_args = dict(vars(args))
    report_args["base"] = PATHS.relative(args.base)
    report_args["output"] = PATHS.relative(args.output)
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "base": "v140 online 0.501352; complex base remains v129",
        "primary_hypothesis": (
            "post-v140 H-marginal residuals form a local spatial field within "
            "20 m and can be transferred without the v144 probability calibration gap"
        ),
        "only_primary_change": (
            "multiply exact v140 support target by an inverse-distance XY k=8 "
            "train H log-ratio field at strength 0.25"
        ),
        "excluded": {
            "v144_v_correction": True,
            "learned_badness_gate": True,
            "residual_probability_as_neighbor_feature": True,
            "delay_correction": True,
            "complex_phase_correction": True,
        },
        "field": {
            "k": K,
            "strength": STRENGTH,
            "distance_floor": DISTANCE_FLOOR,
            "maximum_distance": MAX_DISTANCE,
            "active_fraction": float(np.mean(gate > 0)),
            "prediction": summarize(prediction),
            "dispersion": summarize(dispersion),
            "eighth_neighbor_distance": summarize(graph_values[:, K - 1]),
            "nested_oof_full_support_gain": audit["nested_oof"][
                "full_support_gain"
            ],
            "nested_oof_active_gain": audit["nested_oof"]["active_gain"],
            "nested_oof_active_positive_fraction": audit["nested_oof"][
                "active_positive_fraction"
            ],
            "nested_oof_by_regime": audit["nested_oof"]["by_regime"],
            "modal_fold_count": audit["config_stability"]["modal_fold_count"],
        },
        "preserved_from_v140": {
            "geodesic_risk_and_h_shift": True,
            "conditional_v_delay_given_h": True,
            "projection_weights": PATHS.relative(
                PATHS.cache_dir(140)
                / "v140_balanced_hminus2_overdrive_weights.npy"
            ),
            "projection_args": report_args,
            "complex_phase": True,
            "total_power": True,
            "pol_ue_allocation": True,
        },
        "target_diagnostics": {
            "target_to_v140_target": summarize(row_cos(final_target, v140_target)),
            "h_marginal_l1_change": summarize(
                np.sum(np.abs(final_h - v140_h), axis=1)
            ),
            "h_marginal_max_ratio": summarize(
                np.max(final_h / np.maximum(v140_h, 1e-12), axis=1)
            ),
        },
        "projection": projection_report,
        "comparison_to_v140": comparison,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET_OUT),
            "field": PATHS.relative(FIELD_OUT),
            "audit": PATHS.relative(AUDIT),
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
            (__file__, "generate_v145_local_h_residual_field.py"),
            (
                PATHS.code / "analyze_v145_gated_local_h_field.py",
                "analyze_v145_gated_local_h_field.py",
            ),
            (
                PATHS.code / "generate_v140_balanced_hminus2_overdrive.py",
                "generate_v140_balanced_hminus2_overdrive.py",
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

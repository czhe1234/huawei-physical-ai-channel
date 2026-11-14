#!/usr/bin/env python3
"""Generate v143 with transductively diffused geodesic risk.

v140's online gain came entirely from its hand-built geodesic fallback: the
learned shift classifier predicted zero for every test point.  v143 keeps the
validated v140 log-overlap operator and projection unchanged, but makes the
risk selector locally consistent over an official-feature test-test graph.
Only normalized joint angle-delay support changes; v129 complex phase, total
power, and polarization/UE allocation are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile

import numpy as np

from generate_v112_kernel_moe_path_support import normalize_support
from generate_v140_balanced_hminus2_overdrive import (
    build_risk,
    row_cos,
    shifted_log_blend,
)
from generate_v87_easy_field_projection import summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v143_transductive_geodesic_risk"
OUT_DIR = PATHS.cache_dir(143)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(143)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v143_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v143_transductive_geodesic_risk_target.npy"
WEIGHTS_OUT = OUT_DIR / "v143_transductive_geodesic_risk_weights.npy"
RISK_OUT = OUT_DIR / "v143_transductive_geodesic_risk.npy"
GRAPH_OUT = OUT_DIR / "v143_transductive_geodesic_graph.npz"
REPORT = REPORT_DIR / "v143_transductive_geodesic_risk_generate.json"
ZIP_OUT = PATHS.round1_versions / "v143_transductive_geodesic_risk_candidate_pending.zip"

K_TRAIN = 24
K_TEST = 10
DIST_POWER = 2.2
DIST_FLOOR = 0.55
RAW_WEIGHT = 0.50


def robust_standardize(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([train, test], axis=0)
    median = np.median(combined, axis=0, keepdims=True).astype(np.float32)
    iqr = (
        np.quantile(combined, 0.75, axis=0, keepdims=True)
        - np.quantile(combined, 0.25, axis=0, keepdims=True)
    ).astype(np.float32)
    scale = np.maximum(iqr, 1e-3)
    return (
        np.clip((train - median) / scale, -6.0, 6.0).astype(np.float32),
        np.clip((test - median) / scale, -6.0, 6.0).astype(np.float32),
    )


def official_graph_embedding(
    pos: np.ndarray,
    geodesic: np.ndarray,
    pointcloud: np.ndarray,
    template_probs: np.ndarray,
) -> np.ndarray:
    values = np.concatenate(
        [
            pos[:, :2].astype(np.float32) / 100.0,
            geodesic.astype(np.float32),
            pointcloud.astype(np.float32),
            template_probs.astype(np.float32),
        ],
        axis=1,
    )
    return np.nan_to_num(values, nan=0.0, posinf=8.0, neginf=-8.0).astype(
        np.float32
    )


def graph_average(
    embedding: np.ndarray, values: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    squared_norm = np.sum(embedding * embedding, axis=1)
    distance = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * (embedding @ embedding.T)
    )
    distance = np.maximum(distance, 0.0).astype(np.float32)
    np.fill_diagonal(distance, np.inf)
    indices = np.argpartition(distance, k - 1, axis=1)[:, :k]
    selected = np.take_along_axis(distance, indices, axis=1)
    order = np.argsort(selected, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    selected = np.take_along_axis(selected, order, axis=1)

    weights = 1.0 / np.maximum(
        np.sqrt(np.maximum(selected, 0.0)), DIST_FLOOR
    ) ** DIST_POWER
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    average = np.sum(weights * values[indices], axis=1).astype(np.float32)
    disagreement = np.sum(
        weights * np.abs(values[:, None] - values[indices]), axis=1
    ).astype(np.float32)
    return average, disagreement, indices.astype(np.int32), weights.astype(np.float32)


def diffuse_risk(
    raw: np.ndarray, neighbor_average: np.ndarray, regimes: np.ndarray
) -> np.ndarray:
    result = RAW_WEIGHT * raw + (1.0 - RAW_WEIGHT) * neighbor_average
    eligible = np.isin(regimes, ["medium", "heavy"])
    return np.where(eligible, np.clip(result, 0.0, 1.0), 0.0).astype(np.float32)


def apply_risk_shift(
    source: np.ndarray, risk: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    empty_shift = np.zeros(len(source), dtype=np.int32)
    empty_weight = np.zeros(len(source), dtype=np.float32)
    return shifted_log_blend(source, empty_shift, empty_weight, risk)


def output_comparison(
    output: np.ndarray, baseline: np.ndarray, regimes: np.ndarray
) -> dict[str, object]:
    numerator = 0.0
    denominator = 0.0
    by_regime = {
        name: {"numerator": 0.0, "denominator": 0.0}
        for name in ("clear", "light", "medium", "heavy")
    }
    ratios: list[np.ndarray] = []
    finite = True
    for start in range(0, len(output), 10):
        end = min(start + 10, len(output))
        current = np.asarray(output[start:end])
        reference = np.asarray(baseline[start:end])
        diff_power = np.sum(np.abs(current - reference) ** 2, axis=(1, 2, 3))
        ref_power = np.sum(np.abs(reference) ** 2, axis=(1, 2, 3))
        out_power = np.sum(np.abs(current) ** 2, axis=(1, 2, 3))
        numerator += float(diff_power.sum())
        denominator += float(ref_power.sum())
        ratios.append((out_power / np.maximum(ref_power, 1e-30)).astype(np.float64))
        finite = finite and bool(
            np.isfinite(current.real).all() and np.isfinite(current.imag).all()
        )
        block_regimes = regimes[start:end]
        for name in by_regime:
            mask = block_regimes == name
            by_regime[name]["numerator"] += float(diff_power[mask].sum())
            by_regime[name]["denominator"] += float(ref_power[mask].sum())

    ratio = np.concatenate(ratios)
    return {
        "finite": finite,
        "nmse_to_v140": numerator / max(denominator, 1e-30),
        "nmse_to_v140_by_regime": {
            name: values["numerator"] / max(values["denominator"], 1e-30)
            for name, values in by_regime.items()
        },
        "power_ratio_to_v140": summarize(ratio),
    }


def digest(path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    pos_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    height = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes_train = regime_of(height["train_blocked"].astype(np.float32))
    regimes_test = regime_of(height["test_blocked"].astype(np.float32))
    geodesic = np.load(PATHS.shared_features / "geodesic_support_features.npz")
    pointcloud = np.load(
        PATHS.shared_features / "official_pointcloud_scene_cache.npz"
    )
    feature_names = [str(value) for value in geodesic["feature_names"]]
    probs_train = np.load(
        PATHS.shared_features / "train_v125eq_crossfit_template_probs.npy"
    ).astype(np.float32)
    probs_test = np.load(
        PATHS.cache_dir(129) / "v129_apex_badpoint_support_residual_probs.npy"
    ).astype(np.float32)

    raw_train = build_risk(
        regimes_train,
        probs_train,
        geodesic["train_features"].astype(np.float32),
        feature_names,
    )
    raw_test = build_risk(
        regimes_test,
        probs_test,
        geodesic["test_features"].astype(np.float32),
        feature_names,
    )
    embedding_train = official_graph_embedding(
        pos_train,
        geodesic["train_features"],
        pointcloud["train_features"],
        probs_train,
    )
    embedding_test = official_graph_embedding(
        pos_test,
        geodesic["test_features"],
        pointcloud["test_features"],
        probs_test,
    )
    embedding_train, embedding_test = robust_standardize(
        embedding_train, embedding_test
    )
    avg_train, disagreement_train, _, _ = graph_average(
        embedding_train, raw_train, K_TRAIN
    )
    avg_test, disagreement_test, test_indices, test_weights = graph_average(
        embedding_test, raw_test, K_TEST
    )
    risk_train = diffuse_risk(raw_train, avg_train, regimes_train)
    risk_test = diffuse_risk(raw_test, avg_test, regimes_test)

    np.save(RISK_OUT, np.column_stack([raw_test, avg_test, risk_test, disagreement_test]))
    np.savez_compressed(
        GRAPH_OUT,
        neighbor_indices=test_indices,
        neighbor_weights=test_weights,
        embedding=embedding_test,
    )

    train_source = normalize_support(
        np.load(PATHS.cache_dir(128) / "train_v125eq_collapsed_support.npy"),
        1e-12,
    )
    truth = normalize_support(
        np.load(PATHS.shared_features / "truth_coll_power.npy"), 1e-12
    )
    raw_train_target, raw_train_shift = apply_risk_shift(train_source, raw_train)
    graph_train_target, graph_train_shift = apply_risk_shift(train_source, risk_train)
    base_train_cos = row_cos(train_source, truth)
    raw_train_gain = row_cos(raw_train_target, truth) - base_train_cos
    graph_train_gain = row_cos(graph_train_target, truth) - base_train_cos

    source = normalize_support(
        np.load(
            PATHS.cache_dir(129)
            / "v129_apex_badpoint_support_residual_target.npy"
        ),
        1e-12,
    )
    target, shift_report = apply_risk_shift(source, risk_test)
    np.save(TARGET_OUT, target)

    weights = np.empty((len(risk_test), 8), dtype=np.float32)
    weights[:, 0] = 0.10 + 0.90 * risk_test
    weights[:, 1:] = (1.0 - weights[:, :1]) / 7.0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    np.save(WEIGHTS_OUT, weights)

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
    projection_report = projection(target, weights, regimes_test, pos_test, args)

    v140_target = normalize_support(
        np.load(
            PATHS.cache_dir(140) / "v140_balanced_hminus2_overdrive_target.npy"
        ),
        1e-12,
    )
    output = np.load(OUTPUT, mmap_mode="r")
    v140_output = np.load(
        PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", mmap_mode="r"
    )
    comparison = output_comparison(output, v140_output, regimes_test)
    if not comparison["finite"]:
        raise ValueError("v143 output contains non-finite values")

    report_args = dict(vars(args))
    report_args["base"] = PATHS.relative(args.base)
    report_args["output"] = PATHS.relative(args.output)
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "base": "v140 online 0.501352; complex base remains v129",
        "hypothesis": (
            "v140's pointwise geodesic risk over-selects isolated locations; "
            "test-test consistency in official propagation-feature space can "
            "redistribute the same risk toward coherent corridors"
        ),
        "only_primary_change": "v140 raw risk -> 50% raw + 50% official-graph neighbor risk",
        "preserved_from_v140": {
            "shift_operator": "geometric log-overlap blend",
            "fallback_threshold": 0.64,
            "h1_weight_formula": "0.165 * risk",
            "extreme_threshold": 0.90,
            "h2_weight_formula": "0.060 * risk",
            "h3": False,
            "projection_args": report_args,
            "complex_phase": True,
            "total_power": True,
            "pol_ue_allocation": True,
        },
        "graph": {
            "features": "xy + 15 geodesic + 73 official-pointcloud + 96 template probabilities",
            "embedding_dim": int(embedding_test.shape[1]),
            "k_train": K_TRAIN,
            "k_test": K_TEST,
            "distance_power": DIST_POWER,
            "distance_floor": DIST_FLOOR,
            "raw_weight": RAW_WEIGHT,
            "clear_light_forced_zero": True,
            "train_disagreement": summarize(disagreement_train),
            "test_disagreement": summarize(disagreement_test),
        },
        "risk": {
            "raw_train": summarize(raw_train),
            "graph_train": summarize(risk_train),
            "raw_test": summarize(raw_test),
            "graph_test": summarize(risk_test),
            "raw_test_active_frac": float(np.mean(raw_test > 0.64)),
            "graph_test_active_frac": float(np.mean(risk_test > 0.64)),
            "raw_test_extreme_frac": float(np.mean(raw_test > 0.90)),
            "graph_test_extreme_frac": float(np.mean(risk_test > 0.90)),
            "graph_test_active_by_regime": {
                name: int(np.sum((regimes_test == name) & (risk_test > 0.64)))
                for name in ("clear", "light", "medium", "heavy")
            },
        },
        "train_diagnostics": {
            "raw_shift": raw_train_shift,
            "graph_shift": graph_train_shift,
            "raw_gain": summarize(raw_train_gain),
            "graph_gain": summarize(graph_train_gain),
            "raw_positive_fraction": float(np.mean(raw_train_gain > 0)),
            "graph_positive_fraction": float(np.mean(graph_train_gain > 0)),
        },
        "shift_application": shift_report,
        "target_diagnostics": {
            "target_to_v129_source": summarize(row_cos(target, source)),
            "v140_target_to_v129_source": summarize(row_cos(v140_target, source)),
            "target_to_v140_target": summarize(row_cos(target, v140_target)),
        },
        "projection": projection_report,
        "comparison_to_v140": comparison,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET_OUT),
            "weights": PATHS.relative(WEIGHTS_OUT),
            "risk": PATHS.relative(RISK_OUT),
            "graph": PATHS.relative(GRAPH_OUT),
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
            (__file__, "generate_v143_transductive_geodesic_risk.py"),
            (
                PATHS.code / "generate_v140_balanced_hminus2_overdrive.py",
                "generate_v140_balanced_hminus2_overdrive.py",
            ),
            (
                PATHS.code / "generate_v98_path_ensemble_graph_attention_joint_ad.py",
                "generate_v98_path_ensemble_graph_attention_joint_ad.py",
            ),
            (
                PATHS.code / "generate_v87_easy_field_projection.py",
                "generate_v87_easy_field_projection.py",
            ),
            (
                PATHS.code / "generate_v112_kernel_moe_path_support.py",
                "generate_v112_kernel_moe_path_support.py",
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
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {ZIP_OUT}")
    print(json.dumps(report["risk"], indent=2))
    print(json.dumps(report["train_diagnostics"], indent=2))
    print(json.dumps(report["comparison_to_v140"], indent=2))
    print(json.dumps(report["integrity"], indent=2))


if __name__ == "__main__":
    main()

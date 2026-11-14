#!/usr/bin/env python3
"""Generate v198: power-calibrated reliability gate on v196.

v196 confirmed that low-dimensional delay/PDP mouths are a real scoring axis,
but it still preserved v189's per-point total power.  This version challenges
that ceiling without continuing v197's failed mouth expansion.

The candidate keeps v196 as the base and applies two bounded changes:

* unreliable v196 delay-support moves are partially rolled back toward v189;
* per-point total power is weakly recalibrated from train true power, a
  geometry/regime loss model, and test-test power consensus.

No phase transport is attempted.  The output keeps v196 delay-domain phase and
changes only support amplitude plus a scalar power factor.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import generate_v178_test_track_complex_consensus as track
import generate_v196_train_guided_delay_pdp_mouth as v196
from generate_v87_easy_field_projection import MH, MP, N_ANT, S_SC, summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import BS_POS, TINY, regime_of
from project_paths import PATHS


VERSION = "v198_power_calibrated_delay_gate"
OUT_DIR = PATHS.cache_dir(198)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(198)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v198_Round1_Test_Channel.npy"
BASE_SUPPORT_OUT = OUT_DIR / "v198_v196_delay_support.npy"
BASE_TOTAL_OUT = OUT_DIR / "v198_v196_delay_total.npy"
ANCHOR_SUPPORT_OUT = OUT_DIR / "v198_v189_delay_support.npy"
TARGET_SUPPORT_OUT = OUT_DIR / "v198_delay_gate_target.npy"
ACTION_OUT = OUT_DIR / "v198_power_calibrated_delay_gate_actions.npz"
REPORT = REPORT_DIR / "v198_power_calibrated_delay_gate_generate.json"
ZIP_OUT = PATHS.round1_versions / "v198_power_calibrated_delay_gate_candidate_pending.zip"

BASE = PATHS.cache_dir(196) / "v196_Round1_Test_Channel.npy"
ANCHOR = PATHS.cache_dir(189) / "v189_Round1_Test_Channel.npy"
V197_BASE = PATHS.cache_dir(197) / "v197_Round1_Test_Channel.npy"
V189_BASE = PATHS.cache_dir(189) / "v189_Round1_Test_Channel.npy"

MV = 8
FLOOR = 1e-12
BATCH = 8
TRAIN_K = 72
TEST_K = 96

ROLLBACK_MAX = {"clear": 0.0, "light": 0.035, "medium": 0.13, "heavy": 0.09}
POWER_ALPHA = {"clear": 0.08, "light": 0.12, "medium": 0.36, "heavy": 0.52}
POWER_MIN_DELTA = {"clear": -0.04, "light": -0.08, "medium": -0.22, "heavy": -0.38}
POWER_MAX_DELTA = {"clear": 0.08, "light": 0.12, "medium": 0.16, "heavy": 0.10}


def normalize_support(x: np.ndarray) -> np.ndarray:
    flat = np.maximum(np.asarray(x, dtype=np.float32).reshape(len(x), -1), FLOOR)
    flat /= np.maximum(flat.sum(axis=1, keepdims=True), TINY)
    return flat.reshape(x.shape).astype(np.float32)


def build_delay_support_and_total(
    source_path,
    support_path,
    total_path=None,
    label: str = "support",
) -> tuple[np.ndarray, np.ndarray | None]:
    if support_path.is_file() and (total_path is None or total_path.is_file()):
        support = np.load(support_path, mmap_mode="r")
        total = np.load(total_path, mmap_mode="r") if total_path is not None else None
        return support, total

    source = np.load(source_path, mmap_mode="r")
    shape = (len(source), MP, MH, MV, N_ANT, S_SC)
    support = np.lib.format.open_memmap(support_path, mode="w+", dtype=np.float32, shape=shape)
    total = None
    if total_path is not None:
        total = np.lib.format.open_memmap(total_path, mode="w+", dtype=np.float32, shape=(len(source),))

    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        power = v196.delay_power_from_h(np.asarray(source[start:end], dtype=np.complex64))
        block_total = np.maximum(power.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        support[start:end] = (power / block_total[:, None, None, None, None, None]).astype(np.float32)
        support.flush()
        if total is not None:
            total[start:end] = block_total
            total.flush()
        print(f"v198 {label} {end}/{len(source)}", flush=True)

    loaded_support = np.load(support_path, mmap_mode="r")
    loaded_total = np.load(total_path, mmap_mode="r") if total_path is not None else None
    return loaded_support, loaded_total


def feature_matrix(pos: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    p = np.asarray(pos, dtype=np.float64)
    b = np.asarray(blocked, dtype=np.float64)
    rel = p - BS_POS[None, :]
    dist = np.linalg.norm(rel, axis=1)
    az = np.arctan2(rel[:, 1], rel[:, 0])
    regimes = regime_of(blocked)
    one_hot = np.stack(
        [regimes == "clear", regimes == "light", regimes == "medium", regimes == "heavy"],
        axis=1,
    ).astype(np.float64)
    return np.column_stack(
        [
            p / 220.0,
            rel / 220.0,
            dist / 320.0,
            np.log1p(dist),
            np.sin(az),
            np.cos(az),
            b / 220.0,
            np.log1p(b),
            one_hot,
        ]
    ).astype(np.float32)


def test_log_power_consensus(
    base_log_power: np.ndarray,
    pos: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    idx, raw_w = track.neighbor_graph(pos, blocked, regimes, k=TEST_K)
    out = np.empty(len(pos), dtype=np.float32)
    agreement = np.empty(len(pos), dtype=np.float32)
    effective = np.empty(len(pos), dtype=np.float32)
    for i in range(len(pos)):
        nb = idx[i].astype(np.int64)
        same = (regimes[nb] == regimes[i]).astype(np.float32)
        w = raw_w[i].astype(np.float32) * (0.50 + 0.50 * same)
        w /= max(float(w.sum()), 1e-12)
        out[i] = np.float32(np.sum(w * base_log_power[nb]))
        agreement[i] = np.float32(np.sum(raw_w[i] * same))
        effective[i] = np.float32(1.0 / max(float(np.sum(w * w)), 1e-12))
    return out, {
        "power_test_neighbor_idx": idx.astype(np.int32),
        "power_test_neighbor_weights": raw_w.astype(np.float32),
        "power_test_neighbor_agreement": agreement,
        "power_test_effective_neighbor_count": effective,
    }


def fit_power_model(
    pos_train: np.ndarray,
    blocked_train: np.ndarray,
    total_power: np.ndarray,
    valid: np.ndarray,
) -> ExtraTreesRegressor:
    x = feature_matrix(pos_train, blocked_train)
    y = np.log(np.maximum(total_power, TINY)).astype(np.float32)
    model = ExtraTreesRegressor(
        n_estimators=260,
        min_samples_leaf=5,
        max_features=0.85,
        random_state=198,
        n_jobs=-1,
    )
    model.fit(x[valid], y[valid])
    return model


def build_power_scale(
    base_total: np.ndarray,
    regimes: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    pos_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked)

    train_marginals = v196.build_train_delay_marginals()
    valid = train_marginals["valid"].astype(bool)
    train_idx, train_w, train_conf = v196.geometry_train_weights(
        pos_train,
        pos_test,
        train_blocked,
        test_blocked,
        train_regimes,
        regimes,
        valid.astype(np.float32),
    )

    train_log = np.log(np.maximum(train_marginals["total_power"], TINY)).astype(np.float32)
    knn_log = np.sum(train_w * train_log[train_idx], axis=1).astype(np.float32)

    model = fit_power_model(pos_train, train_blocked, train_marginals["total_power"], valid)
    model_log = model.predict(feature_matrix(pos_test, test_blocked)).astype(np.float32)

    base_log = np.log(np.maximum(np.asarray(base_total, dtype=np.float32), TINY)).astype(np.float32)
    test_log, graph_diag = test_log_power_consensus(base_log, pos_test, test_blocked, regimes)

    combined_log = (0.50 * knn_log + 0.32 * model_log + 0.18 * test_log).astype(np.float32)
    raw_delta = (combined_log - base_log).astype(np.float32)
    model_agreement = np.exp(-np.abs(knn_log - model_log) / 1.15).astype(np.float32)
    reliability = np.clip(
        0.45 * train_conf + 0.35 * graph_diag["power_test_neighbor_agreement"] + 0.20 * model_agreement,
        0.0,
        1.0,
    ).astype(np.float32)

    min_delta = np.asarray([POWER_MIN_DELTA[str(r)] for r in regimes], dtype=np.float32)
    max_delta = np.asarray([POWER_MAX_DELTA[str(r)] for r in regimes], dtype=np.float32)
    alpha = np.asarray([POWER_ALPHA[str(r)] for r in regimes], dtype=np.float32)

    same_direction = np.sign(knn_log - base_log) == np.sign(model_log - base_log)
    small_disagree = np.abs(knn_log - model_log) < 0.65
    direction_gate = np.where(same_direction | small_disagree, 1.0, 0.42).astype(np.float32)
    effective_alpha = alpha * (0.30 + 0.70 * reliability) * direction_gate
    clipped_delta = np.clip(raw_delta, min_delta, max_delta).astype(np.float32)
    log_power_delta = (clipped_delta * effective_alpha).astype(np.float32)
    power_scale = np.exp(log_power_delta).astype(np.float32)

    return power_scale, {
        **graph_diag,
        "power_train_neighbor_idx": train_idx,
        "power_train_neighbor_weights": train_w,
        "power_train_confidence": train_conf,
        "power_knn_log": knn_log,
        "power_model_log": model_log,
        "power_test_consensus_log": test_log,
        "power_base_log": base_log,
        "power_raw_delta": raw_delta,
        "power_clipped_delta": clipped_delta,
        "power_model_agreement": model_agreement,
        "power_reliability": reliability,
        "power_effective_alpha": effective_alpha,
        "power_log_delta": log_power_delta,
        "power_scale": power_scale,
    }


def build_target_support(
    base_support: np.ndarray,
    anchor_support: np.ndarray,
    power_diag: dict[str, np.ndarray],
    regimes: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    reliability = power_diag["power_reliability"]
    model_gap = np.abs(power_diag["power_knn_log"] - power_diag["power_model_log"]).astype(np.float32)
    low_rel = np.clip((0.72 - reliability) / 0.44, 0.0, 1.0).astype(np.float32)
    conflict = np.clip(model_gap / 2.20, 0.0, 1.0).astype(np.float32)
    rollback_cap = np.asarray([ROLLBACK_MAX[str(r)] for r in regimes], dtype=np.float32)
    rollback = (rollback_cap * low_rel * (0.55 + 0.45 * conflict)).astype(np.float32)

    shape = (len(regimes),) + (1,) * (base_support.ndim - 1)
    work = (1.0 - rollback.reshape(shape)) * np.asarray(base_support, dtype=np.float32)
    work += rollback.reshape(shape) * np.asarray(anchor_support, dtype=np.float32)
    work = normalize_support(work)

    target = np.lib.format.open_memmap(TARGET_SUPPORT_OUT, mode="w+", dtype=np.float32, shape=base_support.shape)
    target[:] = work.astype(np.float32)
    target.flush()

    flat0 = np.asarray(base_support).reshape(len(regimes), -1)
    flat1 = work.reshape(len(regimes), -1)
    flat_anchor = np.asarray(anchor_support).reshape(len(regimes), -1)
    diag = {
        "rollback_strength": rollback,
        "support_l1_to_v196": np.sum(np.abs(flat1 - flat0), axis=1).astype(np.float32),
        "support_l1_v196_to_v189": np.sum(np.abs(flat0 - flat_anchor), axis=1).astype(np.float32),
        "target_d_l1_to_v196": np.sum(
            np.abs(
                work.sum(axis=(1, 2, 3, 4), dtype=np.float64)
                - np.asarray(base_support).sum(axis=(1, 2, 3, 4), dtype=np.float64)
            ),
            axis=1,
        ).astype(np.float32),
    }
    return np.load(TARGET_SUPPORT_OUT, mmap_mode="r"), diag


def apply_target(
    target: np.ndarray,
    power_scale: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    diag: dict[str, list[np.ndarray]] = {
        "actual_power_factor": [],
        "full_l1_actual": [],
        "d_l1_actual": [],
        "gain_min": [],
        "gain_p10": [],
        "gain_p90": [],
        "gain_max": [],
    }

    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        cir0 = np.fft.ifft(h0, axis=-1)
        arr0 = cir0.reshape(end - start, MP, MH, MV, N_ANT, S_SC)
        power0 = (np.abs(arr0).astype(np.float32) ** 2).astype(np.float32)
        total0 = np.maximum(power0.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        support0 = power0 / total0[:, None, None, None, None, None]
        h_power0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)

        target_power = target[start:end] * total0[:, None, None, None, None, None]
        gain = np.sqrt(target_power / np.maximum(power0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.20, 4.80).astype(np.float32)
        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.58, 1.78)
        gain[strong1024] = np.clip(gain[strong1024], 0.68, 1.46)
        gain[strong128] = np.clip(gain[strong128], 0.82, 1.22)

        for local, regime in enumerate(regimes[start:end].astype(str)):
            if regime == "clear":
                gain[local] = 1.0 + 0.12 * (gain[local] - 1.0)
            elif regime == "light":
                gain[local] = 1.0 + 0.30 * (gain[local] - 1.0)

        arr1 = arr0 * gain
        h1 = np.fft.fft(arr1.reshape(end - start, 256, N_ANT, S_SC), axis=-1)
        h_power1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        desired_power = h_power0 * power_scale[start:end].astype(np.float32)
        h1 *= np.sqrt(desired_power / h_power1)[:, None, None, None].astype(np.float32)

        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()

        power1 = v196.delay_power_from_h(h1.astype(np.complex64))
        support1 = power1 / np.maximum(power1.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        diag["actual_power_factor"].append(
            (np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64) / h_power0).astype(np.float32)
        )
        diag["full_l1_actual"].append(np.sum(np.abs(support1 - support0), axis=(1, 2, 3, 4, 5)).astype(np.float32))
        diag["d_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(1, 2, 3, 4), dtype=np.float64)
                    - support0.sum(axis=(1, 2, 3, 4), dtype=np.float64)
                ),
                axis=1,
            ).astype(np.float32)
        )
        gflat = gain.reshape(end - start, -1)
        diag["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diag["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["gain_max"].append(gflat.max(axis=1).astype(np.float32))
        print(f"v198 projection {end}/{len(source)}", flush=True)

    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def compare_output_to(output_path, base_path, regimes: np.ndarray) -> dict[str, object]:
    return track.compare_output_to(output_path, base_path, regimes)


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for src in [
            PATHS.code / "generate_v198_power_calibrated_delay_gate.py",
            PATHS.code / "generate_v196_train_guided_delay_pdp_mouth.py",
            PATHS.code / "analyze_v195_scorer_mouth_reverse_inference.py",
            PATHS.code / "generate_v189_leaked_multimarginal_overdrive.py",
            PATHS.code / "generate_v178_test_track_complex_consensus.py",
            PATHS.code / "generate_v87_easy_field_projection.py",
            PATHS.code / "generate_v171_full_complex_ad_tail_denoising.py",
            PATHS.code / "generate_v169_power_scaled_tail_denoising.py",
            PATHS.code / "generate_v159_frontier_consensus_support_soup.py",
            PATHS.code / "powerfield_common.py",
            PATHS.code_root / "project_paths.py",
        ]:
            archive.write(src, src.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    output = np.load(OUTPUT, mmap_mode="r")
    finite = True
    for start in range(0, len(output), 20):
        block = np.asarray(output[start : start + 20], dtype=np.complex64)
        finite = finite and bool(np.isfinite(block.real).all() and np.isfinite(block.imag).all())
    report["integrity"] = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geom["test_blocked"].astype(np.float32))

    base_support, base_total = build_delay_support_and_total(
        BASE,
        BASE_SUPPORT_OUT,
        BASE_TOTAL_OUT,
        "v196 delay support",
    )
    if base_total is None:
        raise RuntimeError("base total cache was not built")
    anchor_support, _ = build_delay_support_and_total(
        ANCHOR,
        ANCHOR_SUPPORT_OUT,
        None,
        "v189 anchor support",
    )

    power_scale, power_diag = build_power_scale(base_total, regimes)
    target_support, support_diag = build_target_support(base_support, anchor_support, power_diag, regimes)
    apply_diag = apply_target(target_support, power_scale, regimes)
    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        **power_diag,
        **support_diag,
        **apply_diag,
    )

    comparison_to_v196 = compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v189 = compare_output_to(OUTPUT, V189_BASE, regimes)
    comparison_to_v197 = compare_output_to(OUTPUT, V197_BASE, regimes)
    if comparison_to_v196["shape"] != [500, 256, 4, 192] or comparison_to_v196["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v196['shape']} {comparison_to_v196['dtype']}")
    if not comparison_to_v196["finite"]:
        raise ValueError("v198 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v196 is correct about the low-dimensional delay/PDP mouth, but its "
            "hard preservation of v189 per-point total power may cap the score. "
            "v198 tests a bounded propagation-loss power calibration and only "
            "rolls back v196 support where train/model/test power evidence is "
            "low-confidence or inconsistent."
        ),
        "new_information_used": {
            "train_truth_total_power": "train true delay-domain total power is used as a log-power target, excluding undefined zero-power spectra",
            "propagation_loss_model": "ExtraTrees model over official coordinates, BS distance, blocked count and regime predicts log total power",
            "test_test_power_consensus": "v196 test log power is smoothed over the same local/regime graph to avoid isolated train extrapolation",
            "v197_boundary": "does not add P-D/V-D/H-U-D mouths; v197 showed that direction over-refines target shape",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "anchor_for_low_confidence_support": PATHS.relative(ANCHOR),
            "train_k": TRAIN_K,
            "test_k": TEST_K,
            "rollback_max": ROLLBACK_MAX,
            "power_alpha": POWER_ALPHA,
            "power_min_log_delta": POWER_MIN_DELTA,
            "power_max_log_delta": POWER_MAX_DELTA,
            "power_target_mix": {
                "train_knn_log_power": 0.50,
                "extra_trees_log_power": 0.32,
                "test_consensus_log_power": 0.18,
            },
            "power_policy": "bounded per-point power scaling; no global normalization back to v196 power",
            "phase_policy": "preserve v196 delay-domain phase; no absolute phase transport",
            "support_policy": "only low-confidence v196 support is partially rolled back toward v189; no extra delay mouth is added",
        },
        "diagnostics": {
            "overall": {
                **{
                    key: summarize(value)
                    for key, value in power_diag.items()
                    if key not in {"power_train_neighbor_idx", "power_train_neighbor_weights", "power_test_neighbor_idx", "power_test_neighbor_weights"}
                },
                **{key: summarize(value) for key, value in support_diag.items()},
                **{key: summarize(value) for key, value in apply_diag.items()},
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "power_scale": power_diag["power_scale"],
                    "power_raw_delta": power_diag["power_raw_delta"],
                    "power_clipped_delta": power_diag["power_clipped_delta"],
                    "power_reliability": power_diag["power_reliability"],
                    "power_effective_alpha": power_diag["power_effective_alpha"],
                    "rollback_strength": support_diag["rollback_strength"],
                    "support_l1_to_v196": support_diag["support_l1_to_v196"],
                    "full_l1_actual": apply_diag["full_l1_actual"],
                    "d_l1_actual": apply_diag["d_l1_actual"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                    "gain_max": apply_diag["gain_max"],
                },
            ),
        },
        "comparison_to_v196": comparison_to_v196,
        "comparison_to_v189": comparison_to_v189,
        "comparison_to_v197": comparison_to_v197,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "base_delay_support": PATHS.relative(BASE_SUPPORT_OUT),
            "base_delay_total": PATHS.relative(BASE_TOTAL_OUT),
            "anchor_delay_support": PATHS.relative(ANCHOR_SUPPORT_OUT),
            "target_support": PATHS.relative(TARGET_SUPPORT_OUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report = write_zip(report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

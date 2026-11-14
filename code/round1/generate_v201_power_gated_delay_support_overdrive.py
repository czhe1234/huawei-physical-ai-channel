#!/usr/bin/env python3
"""Generate v201: power-gated delay support overdrive on top of v200.

v200 proved online that v199's high-confidence power labels are useful.  This
version uses that new online feedback to unlock a second action: only where
the power label is reliable, move the normalized delay-domain support toward
train/test D, U-D and H-D targets again.

The key difference from a plain v200 scale-up is that v201 keeps v200's total
power scalar and changes only support shape.  It also opens a small set of
"false conflict" points where train/model power evidence strongly agrees but
test-local power consensus caused v199 to mark conflict; these points get
support-only correction, not scalar power.
"""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

import generate_v178_test_track_complex_consensus as track
import generate_v196_train_guided_delay_pdp_mouth as v196
from generate_v87_easy_field_projection import MH, MP, N_ANT, S_SC, summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v201_power_gated_delay_support_overdrive"
OUT_DIR = PATHS.cache_dir(201)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(201)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v201_Round1_Test_Channel.npy"
SUPPORT_OUT = OUT_DIR / "v201_v200_delay_support.npy"
TARGET_OUT = OUT_DIR / "v201_power_gated_delay_target.npy"
ACTION_OUT = OUT_DIR / "v201_power_gated_delay_support_actions.npz"
REPORT = REPORT_DIR / "v201_power_gated_delay_support_overdrive_generate.json"
ZIP_OUT = PATHS.round1_versions / "v201_power_gated_delay_support_overdrive_candidate_pending.zip"

BASE = PATHS.cache_dir(200) / "v200_Round1_Test_Channel.npy"
V196_BASE = PATHS.cache_dir(196) / "v196_Round1_Test_Channel.npy"
V189_BASE = PATHS.cache_dir(189) / "v189_Round1_Test_Channel.npy"
V199_DIAG = PATHS.cache_dir(199) / "v199_test_power_diagnostics.npz"
V200_ACTIONS = PATHS.cache_dir(200) / "v200_selective_twosided_power_gate_actions.npz"

MV = 8
BATCH = 8
FLOOR = 1e-12
IPF_ITERS = 5


def normalize_rows(x: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=axis, keepdims=True), TINY)
    return out.astype(np.float32)


def delay_support_from_output(path: Path) -> np.ndarray:
    if SUPPORT_OUT.is_file():
        return np.load(SUPPORT_OUT, mmap_mode="r")
    source = np.load(path, mmap_mode="r")
    shape = (len(source), MP, MH, MV, N_ANT, S_SC)
    support = np.lib.format.open_memmap(SUPPORT_OUT, mode="w+", dtype=np.float32, shape=shape)
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        power = v196.delay_power_from_h(np.asarray(source[start:end], dtype=np.complex64))
        total = np.maximum(power.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        support[start:end] = (power / total).astype(np.float32)
        support.flush()
        print(f"v201 v200 delay support {end}/{len(source)}", flush=True)
    return np.load(SUPPORT_OUT, mmap_mode="r")


def build_gates(regimes: np.ndarray) -> dict[str, np.ndarray]:
    diag = np.load(V199_DIAG)
    v200_actions = np.load(V200_ACTIONS)
    labels = diag["power_label"].astype(str)
    confidence = diag["power_confidence"].astype(np.float32)
    d_knn = diag["delta_knn"].astype(np.float32)
    d_model = diag["delta_model"].astype(np.float32)
    d_consensus = diag["delta_consensus"].astype(np.float32)
    active_power = v200_actions["active"].astype(bool)

    false_conflict = (
        (labels == "conflict")
        & (confidence >= 0.82)
        & (np.abs(d_knn) >= 0.50)
        & (np.abs(d_model) >= 0.50)
        & (np.sign(d_knn) == np.sign(d_model))
    )

    action = np.full(len(regimes), "none", dtype="U16")
    action[active_power & (labels == "power_up")] = "power_up"
    action[active_power & (labels == "power_down")] = "power_down"
    action[false_conflict & (d_knn > 0) & (d_model > 0)] = "false_up"
    action[false_conflict & (d_knn < 0) & (d_model < 0)] = "false_down"

    alpha_d = np.zeros(len(regimes), dtype=np.float32)
    alpha_du = np.zeros(len(regimes), dtype=np.float32)
    alpha_dh = np.zeros(len(regimes), dtype=np.float32)
    train_mix = np.zeros(len(regimes), dtype=np.float32)
    test_mix = np.zeros(len(regimes), dtype=np.float32)

    base_params = {
        ("power_up", "clear"): (0.32, 0.20, 0.14, 0.54, 0.46),
        ("power_up", "light"): (0.08, 0.05, 0.03, 0.40, 0.60),
        ("power_up", "medium"): (0.34, 0.23, 0.16, 0.58, 0.42),
        ("power_up", "heavy"): (0.18, 0.12, 0.08, 0.66, 0.34),
        ("power_down", "clear"): (0.14, 0.08, 0.06, 0.50, 0.50),
        ("power_down", "medium"): (0.24, 0.16, 0.11, 0.58, 0.42),
        ("power_down", "heavy"): (0.26, 0.17, 0.12, 0.70, 0.30),
        ("false_up", "clear"): (0.18, 0.10, 0.07, 0.48, 0.52),
        ("false_down", "clear"): (0.12, 0.07, 0.05, 0.48, 0.52),
        ("false_down", "heavy"): (0.16, 0.10, 0.07, 0.62, 0.38),
    }
    for i, regime in enumerate(regimes.astype(str)):
        key = (str(action[i]), regime)
        if key not in base_params:
            continue
        d, du, dh, tm, xm = base_params[key]
        conf_gain = float(np.clip(0.70 + 0.46 * (confidence[i] - 0.62) / 0.34, 0.68, 1.16))
        evidence_gain = float(np.clip(0.86 + 0.10 * min(abs(d_knn[i]), abs(d_model[i])), 0.86, 1.12))
        gain = conf_gain * evidence_gain
        alpha_d[i] = np.float32(d * gain)
        alpha_du[i] = np.float32(du * gain)
        alpha_dh[i] = np.float32(dh * gain)
        train_mix[i] = np.float32(tm)
        test_mix[i] = np.float32(xm)

    alpha_d = np.clip(alpha_d, 0.0, 0.40).astype(np.float32)
    alpha_du = np.clip(alpha_du, 0.0, 0.26).astype(np.float32)
    alpha_dh = np.clip(alpha_dh, 0.0, 0.18).astype(np.float32)
    return {
        "power_label": labels.astype("U16"),
        "power_confidence": confidence,
        "delta_knn": d_knn,
        "delta_model": d_model,
        "delta_consensus": d_consensus,
        "v200_power_active": active_power,
        "false_conflict": false_conflict,
        "support_action": action,
        "alpha_d": alpha_d,
        "alpha_du": alpha_du,
        "alpha_dh": alpha_dh,
        "train_mix": train_mix,
        "test_mix": test_mix,
    }


def custom_mix_target(
    current: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
    alpha: np.ndarray,
    train_mix: np.ndarray,
    test_mix: np.ndarray,
) -> np.ndarray:
    shape = (len(alpha),) + (1,) * (current.ndim - 1)
    mix_total = np.maximum(train_mix + test_mix, FLOOR)
    tm = (train_mix / mix_total).reshape(shape)
    xm = (test_mix / mix_total).reshape(shape)
    combined = tm * train_target + xm * test_target
    desired = (1.0 - alpha.reshape(shape)) * current + alpha.reshape(shape) * combined
    return normalize_rows(desired, axis=tuple(range(1, desired.ndim)))


def apply_delay_ratio(work: np.ndarray, desired: np.ndarray, mode: str, exponent: float) -> None:
    if mode == "d":
        current = work.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)
        ratio = (np.maximum(desired, FLOOR) / np.maximum(current, FLOOR)).astype(np.float32)
        work *= (ratio**np.float32(exponent))[:, None, None, None, None, :]
    elif mode == "du":
        current = work.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
        ratio = (np.maximum(desired, FLOOR) / np.maximum(current, FLOOR)).astype(np.float32)
        work *= (ratio**np.float32(exponent))[:, None, None, None, :, :]
    elif mode == "dh":
        current = work.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)
        ratio = (np.maximum(desired, FLOOR) / np.maximum(current, FLOOR)).astype(np.float32)
        work *= (ratio**np.float32(exponent))[:, None, :, None, None, :]
    else:
        raise ValueError(mode)


def build_target(support: np.ndarray, regimes: np.ndarray, gates: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    pos_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked)

    train_marginals = v196.build_train_delay_marginals()
    train_idx, train_w, train_conf = v196.geometry_train_weights(
        pos_train,
        pos_test,
        train_blocked,
        test_blocked,
        train_regimes,
        regimes,
        train_marginals["valid"],
    )
    train_pred = v196.predict_train_marginals(train_marginals, train_idx, train_w)
    test_consensus, graph_diag = v196.build_test_consensus(support, pos_test, test_blocked, regimes)

    d0 = np.asarray(support).sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)
    du0 = np.asarray(support).sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    dh0 = np.asarray(support).sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)
    dc = test_consensus.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)
    duc = test_consensus.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    dhc = test_consensus.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)

    alpha_d = gates["alpha_d"].astype(np.float32)
    alpha_du = gates["alpha_du"].astype(np.float32)
    alpha_dh = gates["alpha_dh"].astype(np.float32)
    # Keep weak-neighbor areas from taking full support moves.
    stability = np.clip(0.50 * train_conf + 0.50 * graph_diag["test_neighbor_agreement"], 0.0, 1.0).astype(np.float32)
    stabilizer = np.clip(0.72 + 0.38 * stability, 0.72, 1.10).astype(np.float32)
    alpha_d *= stabilizer
    alpha_du *= stabilizer
    alpha_dh *= stabilizer

    desired_d = custom_mix_target(d0, train_pred["d"], dc, alpha_d, gates["train_mix"], gates["test_mix"])
    desired_du = custom_mix_target(du0, train_pred["du"], duc, alpha_du, gates["train_mix"], gates["test_mix"])
    desired_dh = custom_mix_target(dh0, train_pred["dh"], dhc, alpha_dh, gates["train_mix"], gates["test_mix"])

    n = len(regimes)
    work = np.asarray(support, dtype=np.float32).copy()
    for _ in range(IPF_ITERS):
        apply_delay_ratio(work, desired_d, "d", 0.72)
        apply_delay_ratio(work, desired_du, "du", 0.48)
        apply_delay_ratio(work, desired_dh, "dh", 0.36)
        work = normalize_rows(work.reshape(n, -1), axis=1).reshape(support.shape)

    target = np.lib.format.open_memmap(TARGET_OUT, mode="w+", dtype=np.float32, shape=support.shape)
    target[:] = work.astype(np.float32)
    target.flush()

    d1 = work.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)
    du1 = work.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    dh1 = work.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)
    diag = {
        **graph_diag,
        "train_neighbor_idx": train_idx,
        "train_neighbor_weights": train_w,
        "train_confidence": train_conf,
        "stability": stability,
        "final_alpha_d": alpha_d,
        "final_alpha_du": alpha_du,
        "final_alpha_dh": alpha_dh,
        "target_full_l1": np.sum(np.abs(work.reshape(n, -1) - np.asarray(support).reshape(n, -1)), axis=1).astype(np.float32),
        "target_d_l1": np.sum(np.abs(d1 - d0), axis=1).astype(np.float32),
        "target_du_l1": np.sum(np.abs(du1 - du0), axis=(1, 2)).astype(np.float32),
        "target_dh_l1": np.sum(np.abs(dh1 - dh0), axis=(1, 2)).astype(np.float32),
        "train_d_l1_to_current": np.sum(np.abs(train_pred["d"] - d0), axis=1).astype(np.float32),
        "test_d_l1_to_current": np.sum(np.abs(dc - d0), axis=1).astype(np.float32),
    }
    return np.load(TARGET_OUT, mmap_mode="r"), diag


def apply_target(target: np.ndarray, regimes: np.ndarray, gates: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    action = gates["support_action"].astype(str)
    diag: dict[str, list[np.ndarray]] = {
        "actual_power_factor": [],
        "full_l1_actual": [],
        "d_l1_actual": [],
        "du_l1_actual": [],
        "dh_l1_actual": [],
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
        gain = np.clip(gain, 0.30, 3.20).astype(np.float32)
        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.62, 1.70)
        gain[strong1024] = np.clip(gain[strong1024], 0.72, 1.44)
        gain[strong128] = np.clip(gain[strong128], 0.86, 1.20)

        for local, (regime, act) in enumerate(zip(regimes[start:end].astype(str), action[start:end])):
            if act == "none":
                gain[local] = 1.0
            elif regime == "clear":
                gain[local] = 1.0 + 0.55 * (gain[local] - 1.0)
            elif regime == "light":
                gain[local] = 1.0 + 0.35 * (gain[local] - 1.0)
            elif regime == "medium":
                gain[local] = 1.0 + 0.72 * (gain[local] - 1.0)
            else:
                gain[local] = 1.0 + 0.78 * (gain[local] - 1.0)

        arr1 = arr0 * gain
        h1 = np.fft.fft(arr1.reshape(end - start, 256, N_ANT, S_SC), axis=-1)
        h_power1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        h1 *= np.sqrt(h_power0 / h_power1)[:, None, None, None].astype(np.float32)

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
        diag["du_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(1, 2, 3), dtype=np.float64)
                    - support0.sum(axis=(1, 2, 3), dtype=np.float64)
                ),
                axis=(1, 2),
            ).astype(np.float32)
        )
        diag["dh_l1_actual"].append(
            np.sum(
                np.abs(
                    support1.sum(axis=(1, 3, 4), dtype=np.float64)
                    - support0.sum(axis=(1, 3, 4), dtype=np.float64)
                ),
                axis=(1, 2),
            ).astype(np.float32)
        )
        gflat = gain.reshape(end - start, -1)
        diag["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diag["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["gain_max"].append(gflat.max(axis=1).astype(np.float32))
        print(f"v201 support overdrive {end}/{len(source)}", flush=True)

    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def action_counts(action: np.ndarray, regimes: np.ndarray) -> dict[str, object]:
    out: dict[str, object] = {"total_active": int(np.sum(action.astype(str) != "none"))}
    for name, count in Counter(action.astype(str).tolist()).items():
        out[name] = int(count)
    by_regime: dict[str, object] = {}
    for regime in ["clear", "light", "medium", "heavy"]:
        mask = regimes.astype(str) == regime
        by_regime[regime] = {name: int(np.sum(mask & (action.astype(str) == name))) for name in ["power_up", "power_down", "false_up", "false_down", "none"]}
    out["by_regime"] = by_regime
    return out


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for src in [
            PATHS.code / "generate_v201_power_gated_delay_support_overdrive.py",
            PATHS.code / "generate_v200_selective_twosided_power_gate.py",
            PATHS.code / "analyze_v199_power_field_reliability.py",
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
    support = delay_support_from_output(BASE)
    gates = build_gates(regimes)
    target, target_diag = build_target(support, regimes, gates)
    apply_diag = apply_target(target, regimes, gates)
    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        **gates,
        **target_diag,
        **apply_diag,
    )

    comparison_to_v200 = track.compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v196 = track.compare_output_to(OUTPUT, V196_BASE, regimes)
    comparison_to_v189 = track.compare_output_to(OUTPUT, V189_BASE, regimes)
    if comparison_to_v200["shape"] != [500, 256, 4, 192] or comparison_to_v200["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v200['shape']} {comparison_to_v200['dtype']}")
    if not comparison_to_v200["finite"]:
        raise ValueError("v201 output contains non-finite values")

    active = gates["support_action"].astype(str) != "none"
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v200 online improvement proves v199 power labels are useful.  v201 "
            "keeps v200 total power and uses those labels as a gate for a second "
            "D/U-D/H-D support correction, including a small train/model-agree "
            "false-conflict set where scalar power remains unsafe."
        ),
        "new_information_used": {
            "v200_online_feedback": "selective two-sided power labels are no longer just diagnostic; they have online value",
            "power_gated_support": "support/PDP movement is enabled only by v200-active power labels or train/model-agree false conflicts",
            "false_conflict_split": "some v199 conflict points have train/model power evidence strongly aligned and are used for support-only correction",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "power_diagnostics": PATHS.relative(V199_DIAG),
            "v200_actions": PATHS.relative(V200_ACTIONS),
            "ipf_iters": IPF_ITERS,
            "power_policy": "inherit v200 total power; support-only overdrive preserves per-point power",
            "phase_policy": "preserve v200 delay-domain phase; no absolute phase transport",
            "marginals": ["D", "U-D", "H-D"],
        },
        "action_counts": action_counts(gates["support_action"], regimes),
        "diagnostics": {
            "overall": {
                **{
                    key: summarize(value)
                    for key, value in gates.items()
                    if key not in {"power_label", "support_action"}
                },
                **{
                    key: summarize(value)
                    for key, value in target_diag.items()
                    if key not in {"train_neighbor_idx", "train_neighbor_weights", "test_neighbor_idx", "test_neighbor_weights"}
                },
                **{key: summarize(value) for key, value in apply_diag.items()},
                "active_full_l1_actual": summarize(apply_diag["full_l1_actual"][active]),
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "active": active.astype(np.float32),
                    "final_alpha_d": target_diag["final_alpha_d"],
                    "final_alpha_du": target_diag["final_alpha_du"],
                    "final_alpha_dh": target_diag["final_alpha_dh"],
                    "target_full_l1": target_diag["target_full_l1"],
                    "full_l1_actual": apply_diag["full_l1_actual"],
                    "d_l1_actual": apply_diag["d_l1_actual"],
                    "du_l1_actual": apply_diag["du_l1_actual"],
                    "dh_l1_actual": apply_diag["dh_l1_actual"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                    "gain_max": apply_diag["gain_max"],
                },
            ),
        },
        "comparison_to_v200": comparison_to_v200,
        "comparison_to_v196": comparison_to_v196,
        "comparison_to_v189": comparison_to_v189,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "support": PATHS.relative(SUPPORT_OUT),
            "target": PATHS.relative(TARGET_OUT),
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

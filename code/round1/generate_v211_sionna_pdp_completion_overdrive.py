#!/usr/bin/env python3
"""Generate v211: second-stage completion of the v210 Sionna PDP signal.

v210 produced a large online jump by using only high-budget Sionna normalized
PDP/delay shape, anchored by the validated v196 delay target.  The post-v210
diagnostic still shows a large remaining heavy PDP gap to the same Sionna
reference, so v211 continues that specific signal instead of changing phase,
raw RT power, or unrelated support machinery.

The design is intentionally simpler than v210:

* base is v210, not v209;
* raw Sionna phase and raw Sionna total power are still never copied;
* point total power is preserved exactly;
* D/PDP is the main action, while U-D/H-D are weaker consistency constraints.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

import generate_v178_test_track_complex_consensus as track
import generate_v196_train_guided_delay_pdp_mouth as v196
import generate_v210_sionna_pdp_prior_projection as v210
from generate_v87_easy_field_projection import MH, MP, N_ANT, S_SC, summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v211_sionna_pdp_completion_overdrive"

OUT_DIR = PATHS.cache_dir(211)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(211)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v211_Round1_Test_Channel.npy"
MARGINALS = OUT_DIR / "v211_sionna_completion_marginals.npz"
ACTION_OUT = OUT_DIR / "v211_sionna_pdp_completion_actions.npz"
REPORT = REPORT_DIR / "v211_sionna_pdp_completion_overdrive_generate.json"
ZIP_OUT = PATHS.round1_versions / "v211_sionna_pdp_completion_overdrive_candidate_pending.zip"

BASE = PATHS.cache_dir(210) / "v210_Round1_Test_Channel.npy"
V207_REFERENCE = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
V196_TARGET = PATHS.cache_dir(196) / "v196_train_guided_delay_target.npy"
V210_ACTIONS = PATHS.cache_dir(210) / "v210_sionna_pdp_prior_actions.npz"
V210_BASE = PATHS.cache_dir(210) / "v210_Round1_Test_Channel.npy"
V209_BASE = PATHS.cache_dir(209) / "v209_Round1_Test_Channel.npy"
V200_BASE = PATHS.cache_dir(200) / "v200_Round1_Test_Channel.npy"

MV = 8
FLOOR = 1e-12
BATCH = 8
IPF_ITERS = 6

D_ALPHA_BASE = {"clear": 0.0, "light": 0.018, "medium": 0.30, "heavy": 0.44}
D_ALPHA_CAP = {"clear": 0.0, "light": 0.035, "medium": 0.50, "heavy": 0.68}
DU_ALPHA_BASE = {"clear": 0.0, "light": 0.006, "medium": 0.075, "heavy": 0.135}
DU_ALPHA_CAP = {"clear": 0.0, "light": 0.014, "medium": 0.145, "heavy": 0.240}
DH_ALPHA_BASE = {"clear": 0.0, "light": 0.005, "medium": 0.065, "heavy": 0.120}
DH_ALPHA_CAP = {"clear": 0.0, "light": 0.012, "medium": 0.130, "heavy": 0.220}
SIONNA_MIX_BASE = {"clear": 0.0, "light": 0.36, "medium": 0.72, "heavy": 0.84}


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize(x: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=axis, keepdims=True), TINY)
    return out.astype(np.float32)


def safe_ratio(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    return (np.maximum(desired, FLOOR) / np.maximum(current, FLOOR)).astype(np.float32)


def support_marginals(support: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = support.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)
    du = support.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    dh = support.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)
    return normalize(d, axis=1), normalize(du, axis=(1, 2)), normalize(dh, axis=(1, 2))


def delay_support_from_h(h: np.ndarray) -> np.ndarray:
    power = v196.delay_power_from_h(h)
    return (power / np.maximum(power.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)).astype(np.float32)


def l1_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(a.reshape(len(a), -1) - b.reshape(len(b), -1)), axis=1).astype(np.float32)


def build_marginals() -> dict[str, np.ndarray]:
    if MARGINALS.is_file():
        data = np.load(MARGINALS)
        return {key: data[key] for key in data.files}

    base = np.load(BASE, mmap_mode="r")
    sionna = np.load(V207_REFERENCE, mmap_mode="r")
    v196_target = np.load(V196_TARGET, mmap_mode="r")
    n = len(base)
    arrays = {
        "current_d": np.empty((n, S_SC), dtype=np.float32),
        "current_du": np.empty((n, N_ANT, S_SC), dtype=np.float32),
        "current_dh": np.empty((n, MH, S_SC), dtype=np.float32),
        "sionna_d": np.empty((n, S_SC), dtype=np.float32),
        "sionna_du": np.empty((n, N_ANT, S_SC), dtype=np.float32),
        "sionna_dh": np.empty((n, MH, S_SC), dtype=np.float32),
        "v196_d": np.empty((n, S_SC), dtype=np.float32),
        "v196_du": np.empty((n, N_ANT, S_SC), dtype=np.float32),
        "v196_dh": np.empty((n, MH, S_SC), dtype=np.float32),
        "sionna_total_power": np.empty(n, dtype=np.float32),
    }

    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        current_support = delay_support_from_h(np.asarray(base[start:end], dtype=np.complex64))
        sionna_power = v196.delay_power_from_h(np.asarray(sionna[start:end], dtype=np.complex64))
        sionna_total = sionna_power.sum(axis=(1, 2, 3, 4, 5), dtype=np.float64).astype(np.float32)
        sionna_support = sionna_power / np.maximum(sionna_total[:, None, None, None, None, None], TINY)
        zero = sionna_total <= 1e-20
        if np.any(zero):
            sionna_support[zero] = current_support[zero]
        target_support = np.asarray(v196_target[start:end], dtype=np.float32)

        arrays["current_d"][start:end], arrays["current_du"][start:end], arrays["current_dh"][start:end] = support_marginals(current_support)
        arrays["sionna_d"][start:end], arrays["sionna_du"][start:end], arrays["sionna_dh"][start:end] = support_marginals(sionna_support)
        arrays["v196_d"][start:end], arrays["v196_du"][start:end], arrays["v196_dh"][start:end] = support_marginals(target_support)
        arrays["sionna_total_power"][start:end] = sionna_total
        print(f"v211 marginals {end}/{n}", flush=True)

    np.savez_compressed(MARGINALS, **arrays)
    data = np.load(MARGINALS)
    return {key: data[key] for key in data.files}


def build_actions(regimes: np.ndarray, marginals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    prior = np.load(V210_ACTIONS)
    current_d = marginals["current_d"].astype(np.float32)
    current_du = marginals["current_du"].astype(np.float32)
    current_dh = marginals["current_dh"].astype(np.float32)
    sionna_d = marginals["sionna_d"].astype(np.float32)
    sionna_du = marginals["sionna_du"].astype(np.float32)
    sionna_dh = marginals["sionna_dh"].astype(np.float32)
    v196_d = marginals["v196_d"].astype(np.float32)
    v196_du = marginals["v196_du"].astype(np.float32)
    v196_dh = marginals["v196_dh"].astype(np.float32)
    sionna_zero = marginals["sionna_total_power"].astype(np.float32) <= 1e-20

    d_gap = l1_rows(sionna_d, current_d)
    du_gap = l1_rows(sionna_du, current_du)
    dh_gap = l1_rows(sionna_dh, current_dh)
    d_anchor_gap = l1_rows(sionna_d, v196_d)
    du_anchor_gap = l1_rows(sionna_du, v196_du)
    dh_anchor_gap = l1_rows(sionna_dh, v196_dh)

    prev_alpha = prior["alpha_d"].astype(np.float32)
    prev_mix = prior["sionna_mix"].astype(np.float32)
    prev_agreement = prior["agreement"].astype(np.float32)
    prev_novelty = prior["novelty"].astype(np.float32)
    prev_actual = prior["d_l1_actual"].astype(np.float32)

    agreement = np.clip(
        1.02 - 0.28 * d_anchor_gap - 0.09 * du_anchor_gap - 0.08 * dh_anchor_gap,
        0.12,
        1.0,
    ).astype(np.float32)
    remaining = np.clip((d_gap - 0.34) / 0.86, 0.0, 1.0).astype(np.float32)
    continuation = np.clip(0.55 + 0.45 * prev_alpha / 0.66 + 0.30 * prev_novelty + 0.22 * prev_agreement, 0.0, 1.34)
    overpush_guard = np.clip((prev_actual - 0.26) / 0.18, 0.0, 1.0).astype(np.float32)
    quality = np.clip(0.38 + 0.48 * agreement + 0.40 * remaining + 0.20 * continuation - 0.22 * overpush_guard, 0.0, 1.25)

    alpha_d = np.zeros(len(regimes), dtype=np.float32)
    alpha_du = np.zeros(len(regimes), dtype=np.float32)
    alpha_dh = np.zeros(len(regimes), dtype=np.float32)
    sionna_mix = np.zeros(len(regimes), dtype=np.float32)
    for regime in ["clear", "light", "medium", "heavy"]:
        mask = regimes.astype(str) == regime
        if not np.any(mask):
            continue
        alpha_d[mask] = D_ALPHA_BASE[regime] * quality[mask]
        alpha_du[mask] = DU_ALPHA_BASE[regime] * quality[mask]
        alpha_dh[mask] = DH_ALPHA_BASE[regime] * quality[mask]
        sionna_mix[mask] = np.maximum(
            prev_mix[mask],
            SIONNA_MIX_BASE[regime] * np.clip(0.68 + 0.36 * agreement[mask] + 0.18 * remaining[mask], 0.45, 1.12),
        )
        alpha_d[mask] = np.minimum(alpha_d[mask], D_ALPHA_CAP[regime])
        alpha_du[mask] = np.minimum(alpha_du[mask], DU_ALPHA_CAP[regime])
        alpha_dh[mask] = np.minimum(alpha_dh[mask], DH_ALPHA_CAP[regime])

    high_remaining_heavy = (
        (regimes.astype(str) == "heavy")
        & (d_gap > 0.86)
        & (agreement > 0.28)
        & (prev_alpha > 0.25)
        & (~sionna_zero)
    )
    alpha_d[high_remaining_heavy] = np.minimum(alpha_d[high_remaining_heavy] + 0.08, D_ALPHA_CAP["heavy"])
    sionna_mix[high_remaining_heavy] = np.minimum(np.maximum(sionna_mix[high_remaining_heavy], 0.82), 0.94)

    block = (regimes.astype(str) == "clear") | sionna_zero | ((agreement < 0.16) & (d_gap < 0.70))
    alpha_d[block] = 0.0
    alpha_du[block] = 0.0
    alpha_dh[block] = 0.0
    sionna_mix[block] = 0.0

    target_d = normalize((1.0 - sionna_mix[:, None]) * v196_d + sionna_mix[:, None] * sionna_d, axis=1)
    target_du = normalize((1.0 - sionna_mix[:, None, None]) * v196_du + sionna_mix[:, None, None] * sionna_du, axis=(1, 2))
    target_dh = normalize((1.0 - sionna_mix[:, None, None]) * v196_dh + sionna_mix[:, None, None] * sionna_dh, axis=(1, 2))
    desired_d = normalize((1.0 - alpha_d[:, None]) * current_d + alpha_d[:, None] * target_d, axis=1)
    desired_du = normalize((1.0 - alpha_du[:, None, None]) * current_du + alpha_du[:, None, None] * target_du, axis=(1, 2))
    desired_dh = normalize((1.0 - alpha_dh[:, None, None]) * current_dh + alpha_dh[:, None, None] * target_dh, axis=(1, 2))

    return {
        "regimes": regimes.astype("U8"),
        "alpha_d": alpha_d.astype(np.float32),
        "alpha_du": alpha_du.astype(np.float32),
        "alpha_dh": alpha_dh.astype(np.float32),
        "sionna_mix": sionna_mix.astype(np.float32),
        "agreement": agreement,
        "remaining": remaining,
        "continuation": continuation.astype(np.float32),
        "overpush_guard": overpush_guard,
        "sionna_zero": sionna_zero.astype(bool),
        "d_gap_to_sionna": d_gap,
        "du_gap_to_sionna": du_gap,
        "dh_gap_to_sionna": dh_gap,
        "d_anchor_gap": d_anchor_gap,
        "prev_alpha_d": prev_alpha,
        "prev_sionna_mix": prev_mix,
        "desired_d": desired_d.astype(np.float32),
        "desired_du": desired_du.astype(np.float32),
        "desired_dh": desired_dh.astype(np.float32),
        "target_d": target_d.astype(np.float32),
        "target_du": target_du.astype(np.float32),
        "target_dh": target_dh.astype(np.float32),
    }


def apply_projection(actions: dict[str, np.ndarray], regimes: np.ndarray) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    desired_d = actions["desired_d"].astype(np.float32)
    desired_du = actions["desired_du"].astype(np.float32)
    desired_dh = actions["desired_dh"].astype(np.float32)
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

        work = support0.copy()
        for _ in range(IPF_ITERS):
            v196.apply_delay_ratio(
                work,
                safe_ratio(desired_d[start:end], work.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)),
                "d",
                0.98,
            )
            v196.apply_delay_ratio(
                work,
                safe_ratio(desired_du[start:end], work.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)),
                "du",
                0.30,
            )
            v196.apply_delay_ratio(
                work,
                safe_ratio(desired_dh[start:end], work.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)),
                "dh",
                0.28,
            )
            work = normalize(work.reshape(end - start, -1), axis=1).reshape(support0.shape)

        gain = np.sqrt((work * total0[:, None, None, None, None, None]) / np.maximum(power0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.20, 5.40).astype(np.float32)
        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.46, 2.34)
        gain[strong1024] = np.clip(gain[strong1024], 0.62, 1.64)
        gain[strong128] = np.clip(gain[strong128], 0.82, 1.28)

        for local, regime in enumerate(regimes[start:end].astype(str)):
            if regime == "clear":
                gain[local] = 1.0
            elif regime == "light":
                gain[local] = 1.0 + 0.22 * (gain[local] - 1.0)

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
        print(f"v211 completion projection {end}/{len(source)}", flush=True)

    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for src in [
            PATHS.code / "generate_v211_sionna_pdp_completion_overdrive.py",
            PATHS.code / "generate_v210_sionna_pdp_prior_projection.py",
            PATHS.code / "generate_v207_sionna_test_reference.py",
            PATHS.code / "generate_v196_train_guided_delay_pdp_mouth.py",
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
    for path in [BASE, V207_REFERENCE, V196_TARGET, V210_ACTIONS]:
        require_file(path)
    geom = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geom["test_blocked"].astype(np.float32))
    marginals = build_marginals()
    actions = build_actions(regimes, marginals)
    apply_diag = apply_projection(actions, regimes)
    action_diag = {**actions, **apply_diag}
    np.savez_compressed(ACTION_OUT, **action_diag)

    comparison_to_v210 = track.compare_output_to(OUTPUT, V210_BASE, regimes)
    comparison_to_v209 = track.compare_output_to(OUTPUT, V209_BASE, regimes)
    comparison_to_v200 = track.compare_output_to(OUTPUT, V200_BASE, regimes)
    if comparison_to_v210["shape"] != [500, 256, 4, 192] or comparison_to_v210["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v210['shape']} {comparison_to_v210['dtype']}")
    if not comparison_to_v210["finite"]:
        raise ValueError("v211 output contains non-finite values")

    support_active = actions["alpha_d"] > 1e-7
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v210's large online jump validates high-budget Sionna normalized PDP as a real target source. "
            "Post-v210 diagnostics still show a large heavy D/PDP gap to the same reference, so v211 completes "
            "that specific delay-shape movement on top of v210 while preserving v210 phase and point power."
        ),
        "new_information_used": {
            "v210_online_feedback": "v210=0.534149, +0.022215 vs v209, validates the Sionna PDP target source",
            "post_v210_residual": "heavy D/PDP gap to v207 Sionna reference remains large after v210",
            "failure_boundary": "raw Sionna phase and raw Sionna total power are still not copied; action is normalized delay shape only",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "sionna_reference": PATHS.relative(V207_REFERENCE),
            "v196_delay_target": PATHS.relative(V196_TARGET),
            "ipf_iters": IPF_ITERS,
            "d_alpha_base": D_ALPHA_BASE,
            "d_alpha_cap": D_ALPHA_CAP,
            "du_alpha_base": DU_ALPHA_BASE,
            "du_alpha_cap": DU_ALPHA_CAP,
            "dh_alpha_base": DH_ALPHA_BASE,
            "dh_alpha_cap": DH_ALPHA_CAP,
            "sionna_mix_base": SIONNA_MIX_BASE,
            "power_policy": "preserve v210 point power exactly",
            "phase_policy": "preserve v210 delay-domain phase; no raw Sionna phase transport",
        },
        "action_counts": {
            "support_active": int(np.sum(support_active)),
            "sionna_zero_count": int(np.sum(actions["sionna_zero"])),
        },
        "diagnostics": {
            "overall": {
                **{
                    key: summarize(value)
                    for key, value in action_diag.items()
                    if key
                    not in {
                        "regimes",
                        "desired_d",
                        "desired_du",
                        "desired_dh",
                        "target_d",
                        "target_du",
                        "target_dh",
                    }
                },
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "alpha_d": actions["alpha_d"],
                    "alpha_du": actions["alpha_du"],
                    "alpha_dh": actions["alpha_dh"],
                    "sionna_mix": actions["sionna_mix"],
                    "agreement": actions["agreement"],
                    "remaining": actions["remaining"],
                    "d_gap_to_sionna": actions["d_gap_to_sionna"],
                    "full_l1_actual": apply_diag["full_l1_actual"],
                    "d_l1_actual": apply_diag["d_l1_actual"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                },
            ),
        },
        "comparison_to_v210": comparison_to_v210,
        "comparison_to_v209": comparison_to_v209,
        "comparison_to_v200": comparison_to_v200,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "marginals": PATHS.relative(MARGINALS),
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

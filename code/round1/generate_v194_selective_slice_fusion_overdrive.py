#!/usr/bin/env python3
"""Generate v194: selective v189 + v192 full-slice conditional fusion.

v189 is still the online best because its low-dimensional leaked marginals are
well balanced.  v192 almost tied v189 while changing only full pol/UE/V slice
conditionals and anchoring those low marginals back to v189.  This version uses
that new information aggressively but selectively:

* base complex channel is v189;
* target starts from v189 leaked full support;
* v192 conditional ratios are over-driven in medium/heavy high-confidence
  points;
* v189 low marginals S-H, pol-S-H, S-H-UE, S-H-V and S are restored by IPF;
* directions too similar to the failed v190 hyper-marginal target are damped.

The goal is to capture v192's slice-allocation benefit without losing v189's
validated multi-marginal backbone.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np

import generate_v178_test_track_complex_consensus as track
import generate_v187_leaked_pas_lifecycle_projection as leak
from generate_v87_easy_field_projection import row_cos, summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY
from project_paths import PATHS


VERSION = "v194_selective_slice_fusion_overdrive"
OUT_DIR = PATHS.cache_dir(194)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(194)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v194_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v194_selective_slice_fusion_target.npy"
ACTION_OUT = OUT_DIR / "v194_selective_slice_fusion_actions.npz"
REPORT = REPORT_DIR / "v194_selective_slice_fusion_overdrive_generate.json"
ZIP_OUT = PATHS.round1_versions / "v194_selective_slice_fusion_overdrive_candidate_pending.zip"

BASE = PATHS.cache_dir(189) / "v189_Round1_Test_Channel.npy"
V189_SUPPORT = PATHS.cache_dir(190) / "v190_v189_leaked_full_support.npy"
V189_ACTIONS = PATHS.cache_dir(189) / "v189_leaked_multimarginal_actions.npz"
V192_TARGET = PATHS.cache_dir(192) / "v192_probe_slice_conditionals_target.npy"
V192_ACTIONS = PATHS.cache_dir(192) / "v192_probe_slice_conditionals_actions.npz"
V190_TARGET = PATHS.cache_dir(190) / "v190_leaked_hypermarginal_target.npy"
V192_BASE = PATHS.cache_dir(192) / "v192_Round1_Test_Channel.npy"
V190_BASE = PATHS.cache_dir(190) / "v190_Round1_Test_Channel.npy"
V188_BASE = PATHS.cache_dir(188) / "v188_Round1_Test_Channel.npy"

FLOOR = 1e-12
BATCH = 8
IPF_ITERS = 7

BASE_ALPHA = {"clear": 0.0, "light": 0.22, "medium": 1.38, "heavy": 1.88}
MAX_ALPHA = {"clear": 0.0, "light": 0.28, "medium": 1.55, "heavy": 2.10}


def normalize_rows(flat: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(flat, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=1, keepdims=True), TINY)
    return out.astype(np.float32)


def safe_ratio(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    return (np.maximum(desired, FLOOR) / np.maximum(current, FLOOR)).astype(np.float32)


def apply_ratio(work: np.ndarray, ratio: np.ndarray, axes: str, exponent: float) -> None:
    ratio = np.maximum(ratio, FLOOR).astype(np.float32) ** np.float32(exponent)
    if axes == "sh":
        work *= ratio[:, None, :, :, None, None]
    elif axes == "psh":
        work *= ratio[:, :, :, :, None, None]
    elif axes == "shue":
        work *= ratio[:, None, :, :, :, None]
    elif axes == "shv":
        work *= ratio[:, None, :, :, None, :]
    elif axes == "s":
        work *= ratio[:, None, :, None, None, None]
    else:
        raise ValueError(axes)


def delta_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    af = a.reshape(a.shape[0], -1).astype(np.float32)
    bf = b.reshape(b.shape[0], -1).astype(np.float32)
    num = np.sum(af * bf, axis=1, dtype=np.float64)
    den = np.sqrt(np.sum(af * af, axis=1, dtype=np.float64) * np.sum(bf * bf, axis=1, dtype=np.float64))
    return (num / np.maximum(den, 1e-12)).astype(np.float32)


def build_alpha(
    regimes: np.ndarray,
    v192_actions: np.lib.npyio.NpzFile,
    v189_actions: np.lib.npyio.NpzFile,
    cos_v190: np.ndarray,
    l1_v192: np.ndarray,
) -> dict[str, np.ndarray]:
    agreement = v192_actions["neighbor_agreement"].astype(np.float32)
    effective = v192_actions["effective_neighbor_count"].astype(np.float32)
    v189_full_l1 = v189_actions["full_l1_actual"].astype(np.float32)
    v189_collapsed = v189_actions["collapsed_sh_l1_actual"].astype(np.float32)

    alpha = np.zeros(len(regimes), dtype=np.float32)
    confidence = np.zeros(len(regimes), dtype=np.float32)
    v190_safety = np.zeros(len(regimes), dtype=np.float32)
    l1_gate = np.zeros(len(regimes), dtype=np.float32)
    saturation_gate = np.zeros(len(regimes), dtype=np.float32)

    for i, regime in enumerate(regimes.astype(str)):
        base = BASE_ALPHA[regime]
        if base <= 0:
            continue
        if regime == "light":
            agree_gate = np.clip((agreement[i] - 0.18) / 0.22, 0.0, 1.0)
            eff_gate = np.clip((effective[i] - 14.0) / 24.0, 0.0, 1.0)
        elif regime == "medium":
            agree_gate = np.clip((agreement[i] - 0.22) / 0.50, 0.0, 1.0)
            eff_gate = np.clip((effective[i] - 13.0) / 30.0, 0.0, 1.0)
        else:
            agree_gate = np.clip((agreement[i] - 0.62) / 0.34, 0.0, 1.0)
            eff_gate = np.clip((effective[i] - 22.0) / 34.0, 0.0, 1.0)
        confidence[i] = np.float32(0.56 + 0.30 * agree_gate + 0.14 * eff_gate)

        # v190 was a failed global hyper-marginal overdrive.  A very high
        # alignment with that direction is treated as risk, not as a hard veto.
        v190_risk = np.clip((cos_v190[i] - 0.60) / 0.18, 0.0, 1.0)
        v190_safety[i] = np.float32(1.0 - 0.42 * v190_risk)

        # Very tiny v192 corrections have little upside; extremely large ones
        # are more likely unstable.  Medium/heavy normal range is 0.04-0.09.
        low = np.clip((l1_v192[i] - 0.025) / 0.040, 0.0, 1.0)
        high = 1.0 - 0.28 * np.clip((l1_v192[i] - 0.090) / 0.035, 0.0, 1.0)
        l1_gate[i] = np.float32(0.72 + 0.28 * low * high)

        # If v189 already made a strong low-marginal correction, the conditionals
        # are more likely missing detail inside a correct broad support.
        sat = np.clip((v189_full_l1[i] - 0.18) / 0.26, 0.0, 1.0)
        broad = np.clip((v189_collapsed[i] - 0.12) / 0.24, 0.0, 1.0)
        saturation_gate[i] = np.float32(0.78 + 0.14 * sat + 0.08 * broad)

        value = base * confidence[i] * v190_safety[i] * l1_gate[i] * saturation_gate[i]
        alpha[i] = np.float32(np.clip(value, 0.0, MAX_ALPHA[regime]))

    return {
        "alpha_conditional": alpha,
        "confidence_gate": confidence,
        "v190_safety": v190_safety,
        "l1_gate": l1_gate,
        "saturation_gate": saturation_gate,
    }


def enforce_v189_marginals(work: np.ndarray, current: np.ndarray) -> np.ndarray:
    n = work.shape[0]
    desired_sh = current.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
    desired_psh = current.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)
    desired_shue = current.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)
    desired_shv = current.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
    desired_s = current.sum(axis=(1, 3, 4, 5), dtype=np.float64).astype(np.float32)
    for _ in range(IPF_ITERS):
        apply_ratio(work, safe_ratio(desired_sh, work.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)), "sh", 0.92)
        apply_ratio(work, safe_ratio(desired_psh, work.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)), "psh", 0.82)
        apply_ratio(work, safe_ratio(desired_shue, work.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)), "shue", 0.84)
        apply_ratio(work, safe_ratio(desired_shv, work.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)), "shv", 0.82)
        apply_ratio(work, safe_ratio(desired_s, work.sum(axis=(1, 3, 4, 5), dtype=np.float64).astype(np.float32)), "s", 0.72)
        work = normalize_rows(work.reshape(n, -1)).reshape(current.shape)
    return work.astype(np.float32)


def build_target() -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    current = np.asarray(np.load(V189_SUPPORT, mmap_mode="r"), dtype=np.float32)
    target192 = np.asarray(np.load(V192_TARGET, mmap_mode="r"), dtype=np.float32)
    target190 = np.asarray(np.load(V190_TARGET, mmap_mode="r"), dtype=np.float32)
    v192_actions = np.load(V192_ACTIONS)
    v189_actions = np.load(V189_ACTIONS)
    regimes = v192_actions["regimes"].astype("U8")

    delta192 = target192 - current
    delta190 = target190 - current
    cos_v190 = delta_cosine(delta192, delta190)
    l1_v192 = np.sum(np.abs(delta192.reshape(delta192.shape[0], -1)), axis=1).astype(np.float32)
    alpha_diag = build_alpha(regimes, v192_actions, v189_actions, cos_v190, l1_v192)
    alpha = alpha_diag["alpha_conditional"]

    ratio = np.maximum(target192, FLOOR) / np.maximum(current, FLOOR)
    work = current * (ratio ** alpha[:, None, None, None, None, None]).astype(np.float32)
    work = normalize_rows(work.reshape(current.shape[0], -1)).reshape(current.shape)
    work = enforce_v189_marginals(work, current)

    target = np.lib.format.open_memmap(TARGET_OUT, mode="w+", dtype=np.float32, shape=current.shape)
    target[:] = work
    target.flush()

    flat0 = current.reshape(current.shape[0], -1)
    flat1 = work.reshape(work.shape[0], -1)
    sh0 = current.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
    sh1 = work.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
    psh0 = current.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)
    psh1 = work.sum(axis=(4, 5), dtype=np.float64).astype(np.float32)
    shue0 = current.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)
    shue1 = work.sum(axis=(1, 5), dtype=np.float64).astype(np.float32)
    shv0 = current.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
    shv1 = work.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
    s0 = current.sum(axis=(1, 3, 4, 5), dtype=np.float64).astype(np.float32)
    s1 = work.sum(axis=(1, 3, 4, 5), dtype=np.float64).astype(np.float32)

    diag = {
        **alpha_diag,
        "v190_delta_cos": cos_v190,
        "v192_conditional_l1": l1_v192,
        "target_full_l1": np.sum(np.abs(flat1 - flat0), axis=1).astype(np.float32),
        "target_full_cos": row_cos(flat1, flat0).astype(np.float32),
        "margin_sh_l1": np.sum(np.abs(sh1 - sh0), axis=(1, 2)).astype(np.float32),
        "margin_psh_l1": np.sum(np.abs(psh1 - psh0), axis=(1, 2, 3)).astype(np.float32),
        "margin_shue_l1": np.sum(np.abs(shue1 - shue0), axis=(1, 2, 3)).astype(np.float32),
        "margin_shv_l1": np.sum(np.abs(shv1 - shv0), axis=(1, 2, 3)).astype(np.float32),
        "margin_s_l1": np.sum(np.abs(s1 - s0), axis=1).astype(np.float32),
    }
    return np.load(TARGET_OUT, mmap_mode="r"), diag, regimes


def apply_target(target: np.ndarray, regimes: np.ndarray) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    diag: dict[str, list[np.ndarray]] = {
        "actual_power_factor": [],
        "full_l1_actual": [],
        "collapsed_sh_l1_actual": [],
        "s_l1_actual": [],
        "gain_min": [],
        "gain_p10": [],
        "gain_p90": [],
        "gain_max": [],
    }
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        spec0 = leak.leaked_forward(h0)
        power0 = leak.leaked_cell_power(spec0)
        total0 = np.maximum(power0.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        support0 = power0 / total0[:, None, None, None, None, None]
        h_power0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)

        target_power = target[start:end] * total0[:, None, None, None, None, None]
        gain = np.sqrt(target_power / np.maximum(power0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.10, 8.80).astype(np.float32)
        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.34, 3.25)
        gain[strong1024] = np.clip(gain[strong1024], 0.52, 2.18)
        gain[strong128] = np.clip(gain[strong128], 0.72, 1.55)
        for local, regime in enumerate(regimes[start:end].astype(str)):
            if regime == "clear":
                gain[local] = 1.0
            elif regime == "light":
                gain[local] = 1.0 + 0.42 * (gain[local] - 1.0)

        spec1 = spec0 * gain
        h1 = leak.leaked_inverse(spec1)
        h_power1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY).astype(np.float32)
        h1 *= np.sqrt(h_power0 / h_power1)[:, None, None, None].astype(np.float32)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()

        spec_after = leak.leaked_forward(h1.astype(np.complex64))
        power1 = leak.leaked_cell_power(spec_after)
        support1 = power1 / np.maximum(power1.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        collapsed0 = support0.sum(axis=(1, 4, 5))
        collapsed1 = support1.sum(axis=(1, 4, 5))
        s0 = support0.sum(axis=(1, 3, 4, 5))
        s1 = support1.sum(axis=(1, 3, 4, 5))
        diag["actual_power_factor"].append(
            (np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64) / h_power0).astype(np.float32)
        )
        diag["full_l1_actual"].append(np.sum(np.abs(support1 - support0), axis=(1, 2, 3, 4, 5)).astype(np.float32))
        diag["collapsed_sh_l1_actual"].append(np.sum(np.abs(collapsed1 - collapsed0), axis=(1, 2)).astype(np.float32))
        diag["s_l1_actual"].append(np.sum(np.abs(s1 - s0), axis=1).astype(np.float32))
        gflat = gain.reshape(end - start, -1)
        diag["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diag["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["gain_max"].append(gflat.max(axis=1).astype(np.float32))
        print(f"v194 selective slice fusion projection {end}/{len(source)}", flush=True)
    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def compare_output_to(output_path, base_path, regimes: np.ndarray) -> dict[str, object]:
    return track.compare_output_to(output_path, base_path, regimes)


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for src in [
            PATHS.code / "generate_v194_selective_slice_fusion_overdrive.py",
            PATHS.code / "generate_v191_v193_metric_probe_suite.py",
            PATHS.code / "generate_v190_leaked_hypermarginal_overdrive.py",
            PATHS.code / "generate_v189_leaked_multimarginal_overdrive.py",
            PATHS.code / "generate_v188_leaked_slice_overdrive.py",
            PATHS.code / "generate_v187_leaked_pas_lifecycle_projection.py",
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
    target, target_diag, regimes = build_target()
    apply_diag = apply_target(target, regimes)
    np.savez_compressed(ACTION_OUT, regimes=regimes.astype("U8"), **target_diag, **apply_diag)

    comparison_to_v189 = compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v192 = compare_output_to(OUTPUT, V192_BASE, regimes)
    comparison_to_v190 = compare_output_to(OUTPUT, V190_BASE, regimes)
    comparison_to_v188 = compare_output_to(OUTPUT, V188_BASE, regimes)
    if comparison_to_v189["shape"] != [500, 256, 4, 192] or comparison_to_v189["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v189['shape']} {comparison_to_v189['dtype']}")
    if not comparison_to_v189["finite"]:
        raise ValueError("v194 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v192 nearly tied v189 while changing only full pol/UE/V slice conditionals. "
            "v194 tests whether an aggressive selective conditional overdrive can add "
            "that fine slice detail while preserving v189's validated low marginals."
        ),
        "new_information_used": {
            "v189": "current online best 0.510038; low multi-marginal backbone remains best",
            "v192": "online 0.510018; full slice conditionals are the strongest probe",
            "v190": "online 0.506173; high alignment with failed hyper-marginal overdrive is damped",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "support_base": PATHS.relative(V189_SUPPORT),
            "v192_target": PATHS.relative(V192_TARGET),
            "v190_failed_target": PATHS.relative(V190_TARGET),
            "base_alpha": BASE_ALPHA,
            "max_alpha": MAX_ALPHA,
            "ipf_iters": IPF_ITERS,
            "preserved_marginals": ["S-H", "pol-S-H", "S-H-UE", "S-H-V", "S"],
            "gain_clip": {
                "all": [0.10, 8.80],
                "top8192_full_slices": [0.34, 3.25],
                "top1024_full_slices": [0.52, 2.18],
                "top128_full_slices": [0.72, 1.55],
                "light_blend": "42% of raw gain",
                "clear": "identity",
            },
            "power_policy": "preserve v189 per-point total power",
            "phase_policy": "preserve leaked-domain phase; no absolute phase transport",
        },
        "diagnostics": {
            "overall": {
                **{key: summarize(value) for key, value in target_diag.items()},
                **{key: summarize(value) for key, value in apply_diag.items()},
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "alpha_conditional": target_diag["alpha_conditional"],
                    "v190_delta_cos": target_diag["v190_delta_cos"],
                    "v192_conditional_l1": target_diag["v192_conditional_l1"],
                    "target_full_l1": target_diag["target_full_l1"],
                    "margin_sh_l1": target_diag["margin_sh_l1"],
                    "margin_psh_l1": target_diag["margin_psh_l1"],
                    "margin_shue_l1": target_diag["margin_shue_l1"],
                    "margin_shv_l1": target_diag["margin_shv_l1"],
                    "margin_s_l1": target_diag["margin_s_l1"],
                    "full_l1_actual": apply_diag["full_l1_actual"],
                    "collapsed_sh_l1_actual": apply_diag["collapsed_sh_l1_actual"],
                    "s_l1_actual": apply_diag["s_l1_actual"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                    "gain_max": apply_diag["gain_max"],
                },
            ),
        },
        "comparison_to_v189": comparison_to_v189,
        "comparison_to_v192": comparison_to_v192,
        "comparison_to_v190": comparison_to_v190,
        "comparison_to_v188": comparison_to_v188,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
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

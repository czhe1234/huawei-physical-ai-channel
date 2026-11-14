#!/usr/bin/env python3
"""Generate v177: exact-zero clean support delta on current best v173.

User request:
    Re-check the effect of removing the 16 exact-zero training channels on the
    current best version.

Controlled interpretation:
    v173 itself no longer retrains residual templates from train points.  The
    strict historical deletion experiment is v137 versus v133:

    * v133: geodesic shift overdrive, no exact-zero deletion.
    * v137: the same method, but residual target/probs and shift classifier are
      rebuilt after removing exactly the 16 train samples whose channel power is
      exactly zero.

    v177 therefore transfers only the measured collapsed AD-support delta
    (v137 support / v133 support) onto v173.  It preserves v173 complex phase,
    total power, and pol/UE allocation, applying only positive AD-cell gains.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import (
    BS_POS,
    MH,
    MV,
    S_SC,
    ad_transform,
    beta_subcarriers,
    inverse_ad,
    row_cos,
    summarize,
)
from generate_v159_frontier_consensus_support_soup import REGIMES, digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from generate_v171_full_complex_ad_tail_denoising import top_mask
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v177_exact_zero_support_delta_on_v173"
OUT_DIR = PATHS.cache_dir(177)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(177)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v177_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v177_exact_zero_support_delta_target.npy"
ACTION_OUT = OUT_DIR / "v177_exact_zero_support_delta_actions.npz"
REPORT = REPORT_DIR / "v177_exact_zero_support_delta_on_v173_generate.json"
ZIP_OUT = PATHS.round1_versions / "v177_exact_zero_support_delta_on_v173_candidate_pending.zip"

BASE = PATHS.cache_dir(173) / "v173_Round1_Test_Channel.npy"
DIRTY_REF = PATHS.cache_dir(133) / "v133_Round1_Test_Channel.npy"
CLEAN_REF = PATHS.cache_dir(137) / "v137_Round1_Test_Channel.npy"

FLOOR = 1e-12


def normalize_support(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return out.astype(np.float32)


def collapsed_support_from_output(
    path: Path,
    positions: np.ndarray,
    cache_path: Path,
    batch: int = 20,
) -> np.ndarray:
    if cache_path.is_file():
        print(f"reuse support cache {PATHS.relative(cache_path)}", flush=True)
        return normalize_support(np.load(cache_path).astype(np.float32))

    source = np.load(path, mmap_mode="r")
    beta_s = beta_subcarriers(140.3, 0.10)
    distance = np.linalg.norm(positions - BS_POS[None, :], axis=1)
    support = np.empty((source.shape[0], MH, MV, S_SC), dtype=np.float32)
    for start in range(0, source.shape[0], batch):
        end = min(start + batch, source.shape[0])
        h = np.asarray(source[start:end], dtype=np.complex64)
        derotated = h * np.exp(
            1j * distance[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        ad = ad_transform(derotated)
        power = (np.abs(ad) ** 2).astype(np.float32)
        collapsed = power.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
        support[start:end] = collapsed / np.maximum(
            collapsed.sum(axis=(1, 2, 3), keepdims=True),
            TINY,
        )
        print(f"{path.name} support {end}/{source.shape[0]}", flush=True)

    support = normalize_support(support)
    np.save(cache_path, support)
    return support


def build_target_support(
    base_support: np.ndarray,
    dirty_support: np.ndarray,
    clean_support: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Move v173 support by the exact-zero-clean ratio measured in v137/v133."""

    raw_log_delta = (
        np.log(np.maximum(clean_support, FLOOR))
        - np.log(np.maximum(dirty_support, FLOOR))
    ).astype(np.float32)
    # Avoid letting tiny reference cells create unbounded gains.  The ratio
    # still covers roughly 0.26x to 3.9x support changes before projection.
    clipped_log_delta = np.clip(raw_log_delta, -1.35, 1.35).astype(np.float32)
    target = normalize_support(
        np.exp(np.log(np.maximum(base_support, FLOOR)) + clipped_log_delta).astype(np.float32)
    )

    dirty_to_clean_l1 = np.sum(
        np.abs(clean_support - dirty_support),
        axis=(1, 2, 3),
    ).astype(np.float32)
    base_to_target_l1 = np.sum(np.abs(target - base_support), axis=(1, 2, 3)).astype(np.float32)
    diagnostics = {
        "dirty_to_clean_support_l1": dirty_to_clean_l1,
        "base_to_target_support_l1": base_to_target_l1,
        "dirty_clean_support_cos": row_cos(dirty_support, clean_support).astype(np.float32),
        "base_target_support_cos": row_cos(base_support, target).astype(np.float32),
        "raw_log_delta_abs_mean": np.mean(np.abs(raw_log_delta), axis=(1, 2, 3)).astype(np.float32),
        "clipped_log_delta_abs_mean": np.mean(np.abs(clipped_log_delta), axis=(1, 2, 3)).astype(np.float32),
    }
    return target, diagnostics


def apply_target_support(
    positions: np.ndarray,
    target: np.ndarray,
    batch: int = 20,
) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT,
        mode="w+",
        dtype=np.complex64,
        shape=source.shape,
    )
    beta_s = beta_subcarriers(140.3, 0.10)
    distance = np.linalg.norm(positions - BS_POS[None, :], axis=1)

    diagnostics: dict[str, list[np.ndarray]] = {
        "actual_power_factor": [],
        "support_l1_actual": [],
        "gain_min": [],
        "gain_p10": [],
        "gain_p90": [],
        "gain_max": [],
    }

    for start in range(0, len(source), batch):
        end = min(start + batch, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        derotated = h0 * np.exp(
            1j * distance[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        ad0 = ad_transform(derotated)
        cell0 = (np.abs(ad0) ** 2).astype(np.float32).sum(
            axis=(1, 4),
            dtype=np.float64,
        ).astype(np.float32)
        total0 = np.maximum(cell0.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
        support0 = cell0 / total0[:, None, None, None]

        target_cell = target[start:end] * total0[:, None, None, None]
        gain_cell = np.sqrt(target_cell / np.maximum(cell0, 1e-30)).astype(np.float32)
        gain_cell = np.clip(gain_cell, 0.25, 3.80).astype(np.float32)

        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(end - start, MH, MV, S_SC)
        strong512 = top_mask(flat, 512).reshape(end - start, MH, MV, S_SC)
        gain_cell[strong512] = np.clip(gain_cell[strong512], 0.58, 1.75)
        gain_cell[strong128] = np.clip(gain_cell[strong128], 0.78, 1.30)

        ad1 = ad0 * gain_cell[:, None, :, :, None, :]
        total_after = np.maximum(
            np.sum(np.abs(ad1) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64),
            TINY,
        ).astype(np.float32)
        # This experiment is about support, not power.  Preserve v173 point
        # total power exactly after AD-cell gain clipping.
        scale = np.sqrt(total0 / total_after).astype(np.float32)
        ad1 *= scale[:, None, None, None, None, None]

        h1 = inverse_ad(ad1).reshape(end - start, 256, 4, 192)
        h1 = h1 * np.exp(
            -1j * distance[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()

        cell1 = (np.abs(ad1) ** 2).astype(np.float32).sum(
            axis=(1, 4),
            dtype=np.float64,
        ).astype(np.float32)
        support1 = cell1 / np.maximum(cell1.sum(axis=(1, 2, 3), keepdims=True), TINY)
        diagnostics["actual_power_factor"].append(
            (cell1.sum(axis=(1, 2, 3)) / total0).astype(np.float32)
        )
        diagnostics["support_l1_actual"].append(
            np.sum(np.abs(support1 - support0), axis=(1, 2, 3)).astype(np.float32)
        )
        gflat = gain_cell.reshape(end - start, -1)
        diagnostics["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diagnostics["gain_p10"].append(np.quantile(gflat, 0.10, axis=1).astype(np.float32))
        diagnostics["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diagnostics["gain_max"].append(gflat.max(axis=1).astype(np.float32))
        print(f"exact-zero support delta applied {end}/500", flush=True)

    return {key: np.concatenate(value).astype(np.float32) for key, value in diagnostics.items()}


def compare_output_to(output_path: Path, base_path: Path, regimes: np.ndarray) -> dict[str, object]:
    output = np.load(output_path, mmap_mode="r")
    base = np.load(base_path, mmap_mode="r")
    numerator = np.zeros(len(output), dtype=np.float64)
    denominator = np.zeros(len(output), dtype=np.float64)
    out_power = np.zeros(len(output), dtype=np.float64)
    base_power = np.zeros(len(output), dtype=np.float64)
    finite = True
    for start in range(0, len(output), 20):
        end = min(start + 20, len(output))
        current = np.asarray(output[start:end], dtype=np.complex64)
        reference = np.asarray(base[start:end], dtype=np.complex64)
        numerator[start:end] = np.sum(
            np.abs(current - reference) ** 2,
            axis=(1, 2, 3),
            dtype=np.float64,
        )
        denominator[start:end] = np.sum(
            np.abs(reference) ** 2,
            axis=(1, 2, 3),
            dtype=np.float64,
        )
        out_power[start:end] = np.sum(
            np.abs(current) ** 2,
            axis=(1, 2, 3),
            dtype=np.float64,
        )
        base_power[start:end] = denominator[start:end]
        finite = finite and bool(
            np.isfinite(current.real).all() and np.isfinite(current.imag).all()
        )
    point_nmse = numerator / np.maximum(denominator, TINY)
    return {
        "finite": finite,
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "nmse": float(numerator.sum() / max(denominator.sum(), TINY)),
        "nmse_by_regime": {
            regime: float(
                numerator[regimes == regime].sum()
                / max(denominator[regimes == regime].sum(), TINY)
            )
            for regime in REGIMES
            if np.any(regimes == regime)
        },
        "point_nmse": summarize(point_nmse),
        "power_ratio": summarize(out_power / np.maximum(base_power, TINY)),
    }


def load_exact_zero_count() -> int:
    power_path = PATHS.shared_features / "train_total_power.npy"
    if not power_path.is_file():
        return -1
    power = np.load(power_path)
    return int(np.sum(power == 0.0))


def generate() -> dict[str, object]:
    for required in (BASE, DIRTY_REF, CLEAN_REF):
        if not required.is_file():
            raise FileNotFoundError(required)

    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked_test = np.load(PATHS.shared_features / "height_geometry_cache.npz")[
        "test_blocked"
    ].astype(np.float32)
    regimes = regime_of(blocked_test)

    support_base = collapsed_support_from_output(
        BASE,
        pos_test,
        OUT_DIR / "v173_collapsed_support.npy",
    )
    support_dirty = collapsed_support_from_output(
        DIRTY_REF,
        pos_test,
        OUT_DIR / "v133_collapsed_support.npy",
    )
    support_clean = collapsed_support_from_output(
        CLEAN_REF,
        pos_test,
        OUT_DIR / "v137_collapsed_support.npy",
    )

    target, target_diag = build_target_support(support_base, support_dirty, support_clean)
    np.save(TARGET_OUT, target)
    apply_diag = apply_target_support(pos_test, target)

    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        **target_diag,
        **apply_diag,
    )

    comparison_to_v173 = compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v137 = compare_output_to(OUTPUT, CLEAN_REF, regimes)
    comparison_to_v133 = compare_output_to(OUTPUT, DIRTY_REF, regimes)
    if comparison_to_v173["shape"] != [500, 256, 4, 192] or comparison_to_v173["dtype"] != "complex64":
        raise ValueError(
            f"invalid output layout: {comparison_to_v173['shape']} {comparison_to_v173['dtype']}"
        )
    if not comparison_to_v173["finite"]:
        raise ValueError("v177 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "Re-test whether removing the 16 exact-zero train channels helps "
            "after the current best v173 support/noise branch.  Since v173 no "
            "longer directly trains exact-zero-sensitive templates, v177 "
            "transfers the observed v137-v133 exact-zero-clean collapsed AD "
            "support delta onto v173."
        ),
        "new_information_used": {
            "v133_online": "0.499218, no exact-zero deletion",
            "v137_online": "0.498983, strict exact-zero-clean rerun of v133",
            "current_best": "v173 online 0.503667",
            "exact_zero_train_count": load_exact_zero_count(),
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "dirty_reference": PATHS.relative(DIRTY_REF),
            "clean_reference": PATHS.relative(CLEAN_REF),
            "support_delta": "clipped log(clean_support / dirty_support), clip [-1.35, 1.35]",
            "target": "normalize(v173_support * exp(clipped_log_delta))",
            "gain_clip": [0.25, 3.80],
            "top512_gain_clip": [0.58, 1.75],
            "top128_gain_clip": [0.78, 1.30],
            "phase_policy": "preserve v173 phase; positive AD-cell gains only",
            "power_policy": "preserve v173 per-point total power",
            "pol_ue_policy": "preserve v173 cell-internal 2-pol x 4-UE allocation up to common cell gain",
        },
        "diagnostics": {
            "overall": {
                **{key: summarize(value) for key, value in target_diag.items()},
                **{key: summarize(value) for key, value in apply_diag.items()},
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "dirty_to_clean_support_l1": target_diag["dirty_to_clean_support_l1"],
                    "base_to_target_support_l1": target_diag["base_to_target_support_l1"],
                    "dirty_clean_support_cos": target_diag["dirty_clean_support_cos"],
                    "base_target_support_cos": target_diag["base_target_support_cos"],
                    "support_l1_actual": apply_diag["support_l1_actual"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                },
            ),
        },
        "comparison_to_v173": comparison_to_v173,
        "comparison_to_v137": comparison_to_v137,
        "comparison_to_v133": comparison_to_v133,
        "preserved": {
            "base_complex": "v173",
            "phase": "preserved",
            "point_total_power": "preserved",
            "pol_ue_allocation": "preserved inside each AD cell",
            "randomness": "none",
        },
        "changed": {
            "collapsed_ad_support": "nudged by the strict exact-zero-clean v137-v133 support ratio",
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "target_support": PATHS.relative(TARGET_OUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source, name in (
            (__file__, "generate_v177_exact_zero_support_delta_on_v173.py"),
            (PATHS.code / "generate_v137_exact_zero_clean_rerun_v133.py", "generate_v137_exact_zero_clean_rerun_v133.py"),
            (PATHS.code / "generate_v173_spatial_coherence_tail_denoising.py", "generate_v173_spatial_coherence_tail_denoising.py"),
            (PATHS.code / "generate_v171_full_complex_ad_tail_denoising.py", "generate_v171_full_complex_ad_tail_denoising.py"),
            (PATHS.code / "generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
            (PATHS.code / "generate_v159_frontier_consensus_support_soup.py", "generate_v159_frontier_consensus_support_soup.py"),
            (PATHS.code / "generate_v169_power_scaled_tail_denoising.py", "generate_v169_power_scaled_tail_denoising.py"),
            (PATHS.code / "powerfield_common.py", "powerfield_common.py"),
            (PATHS.code_root / "project_paths.py", "project_paths.py"),
        ):
            archive.write(source, name)

    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise ValueError(f"ZIP CRC failure: {bad_member}")
    output = np.load(OUTPUT, mmap_mode="r")
    integrity = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": bool(np.isfinite(output.real).all() and np.isfinite(output.imag).all()),
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    report["integrity"] = integrity
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    report = generate()
    report = write_zip(report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

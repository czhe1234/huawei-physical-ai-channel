#!/usr/bin/env python3
"""Generate v171: full-complex AD tail denoising on top of v170.

Hypothesis:
    v169/v170 validated the weak-tail reliability branch, but both versions
    still modify only collapsed joint AD support plus point total power.  If
    released train labels contain complex Gaussian-like background, then part
    of the unreliable tail also lives inside each AD cell's 2-pol x 4-UE
    complex coefficients.  v171 therefore denoises the full complex AD tensor:

    * protect strong ranked AD cells;
    * estimate each point's AD floor outside top512 cells;
    * shrink low-SNR coefficients inside weak cells with an aggressive
      Wiener-like power-floor subtraction;
    * keep total power shrink bounded by the removed weak-tail mass.

This is the first candidate in this branch that changes weak-tail phase and
pol/UE allocation implicitly by suppressing unreliable complex coefficients.
Strong paths remain protected.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import (
    BS_POS,
    MH,
    MP,
    MV,
    N_ANT,
    S_SC,
    ad_transform,
    beta_subcarriers,
    inverse_ad,
    summarize,
)
from generate_v159_frontier_consensus_support_soup import REGIMES, compare_outputs, digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v171_full_complex_ad_tail_denoising"
OUT_DIR = PATHS.cache_dir(171)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(171)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v171_Round1_Test_Channel.npy"
ACTION_OUT = OUT_DIR / "v171_full_complex_ad_tail_denoising_actions.npz"
REPORT = REPORT_DIR / "v171_full_complex_ad_tail_denoising_generate.json"
ZIP_OUT = PATHS.round1_versions / "v171_full_complex_ad_tail_denoising_candidate_pending.zip"

BASE = PATHS.cache_dir(170) / "v170_Round1_Test_Channel.npy"
BASE_ACTIONS = PATHS.cache_dir(170) / "v170_adaptive_floor_subtraction_actions.npz"

COMP_PER_CELL = MP * N_ANT
CELL_COUNT = MH * MV * S_SC
FLOOR = 1e-12


def top_mask(flat: np.ndarray, k: int) -> np.ndarray:
    k = min(k, flat.shape[1])
    ids = np.argpartition(flat, -k, axis=1)[:, -k:]
    mask = np.zeros(flat.shape, dtype=bool)
    mask[np.arange(flat.shape[0])[:, None], ids] = True
    return mask


def support_metrics_from_cell(cell: np.ndarray) -> dict[str, np.ndarray]:
    total = np.maximum(cell.sum(axis=(1, 2, 3)), TINY)
    support = cell / total[:, None, None, None]
    flat = support.reshape(len(cell), -1).astype(np.float32)
    sorted_flat = np.sort(flat, axis=1)
    entropy = -np.sum(flat * np.log(np.maximum(flat, FLOOR)), axis=1) / np.log(CELL_COUNT)
    return {
        "support": support.astype(np.float32),
        "flat_support": flat,
        "cell_total_power": total.astype(np.float32),
        "top64_fraction": sorted_flat[:, -64:].sum(axis=1).astype(np.float32),
        "top128_fraction": sorted_flat[:, -128:].sum(axis=1).astype(np.float32),
        "top512_fraction": sorted_flat[:, -512:].sum(axis=1).astype(np.float32),
        "tail512_floor": sorted_flat[:, :-512].mean(axis=1).astype(np.float32),
        "tail512_fraction": (sorted_flat[:, :-512].mean(axis=1) * CELL_COUNT).astype(np.float32),
        "entropy": entropy.astype(np.float32),
    }


def action_from_v170(regimes: np.ndarray) -> dict[str, np.ndarray]:
    data = np.load(BASE_ACTIONS)
    base = data["action"].astype(np.float32)
    scale = np.ones(len(base), dtype=np.float32)
    scale[regimes == "clear"] = 0.35
    scale[regimes == "light"] = 0.45
    scale[regimes == "medium"] = 1.12
    scale[regimes == "heavy"] = 1.22
    action = np.clip(base * scale, 0.0, 1.0).astype(np.float32)
    action[regimes == "clear"] = np.minimum(action[regimes == "clear"], 0.020)
    action[regimes == "light"] = np.minimum(action[regimes == "light"], 0.040)
    return {
        "v170_action": base,
        "action_scale": scale,
        "action": action,
        "v170_removed_mass": data["removed_mass"].astype(np.float32),
        "v170_power_factor": data["power_factor"].astype(np.float32),
        "v170_multibounce_protect": data["multibounce_protect"].astype(np.float32),
    }


def lower_power_bounds(regimes: np.ndarray) -> np.ndarray:
    lower = np.ones(len(regimes), dtype=np.float32)
    lower[regimes == "clear"] = 0.997
    lower[regimes == "light"] = 0.994
    lower[regimes == "medium"] = 0.958
    lower[regimes == "heavy"] = 0.930
    return lower


def denoise_batch(
    ad: np.ndarray,
    action: np.ndarray,
    lower_bound: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    coeff_power = (np.abs(ad) ** 2).astype(np.float32)
    cell = coeff_power.sum(axis=(1, 4), dtype=np.float64).astype(np.float32)
    metrics = support_metrics_from_cell(cell)
    flat = metrics["flat_support"]

    mask128 = top_mask(flat, 128)
    mask512 = top_mask(flat, 512)
    mask2048 = top_mask(flat, 2048)
    mask4096 = top_mask(flat, 4096)

    rank_multiplier = np.ones(flat.shape, dtype=np.float32)
    rank_multiplier[mask4096] = 1.15
    rank_multiplier[mask2048] = 0.88
    rank_multiplier[mask512] = 0.40
    rank_multiplier[mask128] = 0.0

    min_keep = np.full(flat.shape, 0.008, dtype=np.float32)
    min_keep[mask4096] = 0.045
    min_keep[mask2048] = 0.12
    min_keep[mask512] = 0.50
    min_keep[mask128] = 1.0

    floor_cell_power = (
        np.maximum(metrics["tail512_floor"], FLOOR)
        * metrics["cell_total_power"]
    ).astype(np.float32)
    coeff_floor = (floor_cell_power / float(COMP_PER_CELL)).astype(np.float32)
    coeff_lambda = (4.50 * action).astype(np.float32)

    rank = rank_multiplier.reshape(len(ad), 1, MH, MV, 1, S_SC)
    keep = min_keep.reshape(len(ad), 1, MH, MV, 1, S_SC)
    subtract = (
        coeff_lambda[:, None, None, None, None, None]
        * coeff_floor[:, None, None, None, None, None]
        * rank
    )
    denoised_power = np.maximum(coeff_power - subtract, coeff_power * keep)

    total0 = np.maximum(coeff_power.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
    total_direct = np.maximum(denoised_power.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
    removed_fraction = np.clip(1.0 - total_direct / total0, 0.0, 1.0).astype(np.float32)
    target_power_factor = np.clip(
        1.0 - 1.05 * removed_fraction,
        lower_bound,
        1.0,
    ).astype(np.float32)

    gain = np.sqrt(denoised_power / np.maximum(coeff_power, 1e-30)).astype(np.float32)
    ad1 = (ad * gain).astype(np.complex64)
    total_after_gain = np.maximum(
        np.sum(np.abs(ad1) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64),
        TINY,
    ).astype(np.float32)
    scale = np.sqrt(target_power_factor * total0 / total_after_gain).astype(np.float32)
    ad1 *= scale[:, None, None, None, None, None]

    cell1 = (np.abs(ad1) ** 2).astype(np.float32).sum(axis=(1, 4))
    support0 = metrics["support"]
    support1 = cell1 / np.maximum(cell1.sum(axis=(1, 2, 3), keepdims=True), TINY)
    support_l1 = np.sum(np.abs(support1 - support0), axis=(1, 2, 3)).astype(np.float32)

    diag = {
        "top64_fraction": metrics["top64_fraction"],
        "top128_fraction": metrics["top128_fraction"],
        "top512_fraction": metrics["top512_fraction"],
        "tail512_fraction": metrics["tail512_fraction"],
        "entropy": metrics["entropy"],
        "floor_cell_power": floor_cell_power,
        "coeff_floor": coeff_floor,
        "coeff_lambda": coeff_lambda,
        "removed_fraction_direct": removed_fraction,
        "target_power_factor": target_power_factor,
        "support_l1_to_base": support_l1,
        "actual_power_factor": (
            np.sum(np.abs(ad1) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64)
            / np.maximum(total0, TINY)
        ).astype(np.float32),
    }
    return ad1.astype(np.complex64), diag


def generate() -> dict[str, object]:
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked_test = np.load(PATHS.shared_features / "height_geometry_cache.npz")[
        "test_blocked"
    ].astype(np.float32)
    regimes = regime_of(blocked_test)
    actions = action_from_v170(regimes)
    lower = lower_power_bounds(regimes)

    beta_s = beta_subcarriers(140.3, 0.10)
    distance = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )

    diagnostics: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "top64_fraction",
            "top128_fraction",
            "top512_fraction",
            "tail512_fraction",
            "entropy",
            "floor_cell_power",
            "coeff_floor",
            "coeff_lambda",
            "removed_fraction_direct",
            "target_power_factor",
            "actual_power_factor",
            "support_l1_to_base",
        )
    }

    for start in range(0, len(source), 20):
        end = min(start + 20, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        derot = h0 * np.exp(
            1j * distance[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        ad0 = ad_transform(derot)
        ad1, diag = denoise_batch(
            ad0,
            actions["action"][start:end],
            lower[start:end],
        )
        h1 = inverse_ad(ad1).reshape(end - start, 256, 4, 192)
        h1 = h1 * np.exp(
            -1j * distance[start:end, None, None, None] * beta_s
        ).astype(np.complex64)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()
        for key, value in diag.items():
            diagnostics[key].append(value.astype(np.float32))
        print(f"full-complex denoised {end}/500", flush=True)

    diag_arrays = {key: np.concatenate(values).astype(np.float32) for key, values in diagnostics.items()}
    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        **actions,
        **diag_arrays,
    )

    comparison_to_v170 = compare_outputs(OUTPUT, BASE, regimes)
    comparison_to_v169 = compare_outputs(
        OUTPUT, PATHS.cache_dir(169) / "v169_Round1_Test_Channel.npy", regimes
    )
    comparison_to_v163 = compare_outputs(
        OUTPUT, PATHS.cache_dir(163) / "v163_Round1_Test_Channel.npy", regimes
    )
    comparison_to_v140 = compare_outputs(
        OUTPUT, PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", regimes
    )
    if comparison_to_v170["shape"] != [500, 256, 4, 192] or comparison_to_v170["dtype"] != "complex64":
        raise ValueError(
            f"invalid output layout: {comparison_to_v170['shape']} {comparison_to_v170['dtype']}"
        )
    if not comparison_to_v170["finite"]:
        raise ValueError("v171 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v170 confirmed adaptive collapsed AD floor subtraction. v171 tests "
            "the larger missed layer: released-label complex noise may also live "
            "inside weak AD cells' 2-pol x 4-UE complex coefficients. Therefore "
            "full complex AD tail coefficients should be shrunk directly while "
            "strong cells remain protected."
        ),
        "new_information_used": {
            "v170_online": "0.503511 confirms adaptive AD floor-subtraction improves v169",
            "v168": "tail coefficients are circular-complex-like and power-dependent",
            "v170_actions": "reuse proven floor-subtraction selector and strengthen only medium/heavy",
        },
        "excluded": {
            "random_noise_injection": True,
            "negative_common_subspace": True,
            "broad_sionna_delay_controller": True,
            "exact_zero_deletion": True,
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "action": "v170 action scaled by regime: clear .35, light .45, medium 1.12, heavy 1.22",
            "coeff_floor": "tail512_floor * point_total_power / 8",
            "coeff_lambda": "4.50 * action",
            "rank_multiplier": {
                "top128": 0.0,
                "top512": 0.40,
                "top2048": 0.88,
                "top4096": 1.15,
                "tail": 1.0,
            },
            "min_keep": {
                "top128": 1.0,
                "top512": 0.50,
                "top2048": 0.12,
                "top4096": 0.045,
                "tail": 0.008,
            },
            "target_power_factor": "clip(1 - 1.05*direct_removed_fraction, regime lower bound, 1)",
            "lower_bounds": {"clear": 0.997, "light": 0.994, "medium": 0.958, "heavy": 0.930},
        },
        "diagnostics": {
            "overall": {
                **{key: summarize(value) for key, value in actions.items() if value.ndim == 1},
                **{key: summarize(value) for key, value in diag_arrays.items()},
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "action": actions["action"],
                    "v170_action": actions["v170_action"],
                    "removed_fraction_direct": diag_arrays["removed_fraction_direct"],
                    "target_power_factor": diag_arrays["target_power_factor"],
                    "actual_power_factor": diag_arrays["actual_power_factor"],
                    "support_l1_to_base": diag_arrays["support_l1_to_base"],
                    "tail512_fraction": diag_arrays["tail512_fraction"],
                    "entropy": diag_arrays["entropy"],
                    "coeff_lambda": diag_arrays["coeff_lambda"],
                },
            ),
        },
        "comparison_to_v170": comparison_to_v170,
        "comparison_to_v169": comparison_to_v169,
        "comparison_to_v163": comparison_to_v163,
        "comparison_to_v140": comparison_to_v140,
        "preserved": {
            "base_complex": "v170",
            "strong_cells": "top128 AD cells are not shrunk; top512 has high minimum keep",
            "randomness": "none",
        },
        "changed": {
            "full_complex_ad_tail": "coefficient-level floor subtraction in AD domain",
            "weak_tail_phase_and_pol_ue": "attenuated indirectly through complex coefficient shrink",
            "total_power": "bounded shrink from direct removed weak-tail power",
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
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
            (__file__, "generate_v171_full_complex_ad_tail_denoising.py"),
            (PATHS.code / "generate_v170_adaptive_floor_subtraction_denoising.py", "generate_v170_adaptive_floor_subtraction_denoising.py"),
            (PATHS.code / "generate_v169_power_scaled_tail_denoising.py", "generate_v169_power_scaled_tail_denoising.py"),
            (PATHS.code / "analyze_v168_channel_awgn_power_audit.py", "analyze_v168_channel_awgn_power_audit.py"),
            (PATHS.code / "generate_v159_frontier_consensus_support_soup.py", "generate_v159_frontier_consensus_support_soup.py"),
            (PATHS.code / "generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
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

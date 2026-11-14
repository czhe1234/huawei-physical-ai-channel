#!/usr/bin/env python3
"""Generate v169: power-scaled weak-tail denoising on top of v163.

Hypothesis:
    The official train channels may be noisy observations of a cleaner Sionna
    field.  v168 showed that the AD tail has circular-complex background
    structure, while the tail floor strongly follows channel power.  If the
    best current generator has learned part of that weak train-label tail,
    then a cleaner hidden target should prefer:

    * preserving strong joint AD paths;
    * suppressing only low-SNR support tail;
    * slightly reducing total power by the amount interpreted as tail noise.

This is not random noise injection.  It is deterministic denoising of v163.

Preserved:
    v163 phase source, strong AD components, and 2-pol x 4-UE allocation.
    The only changed objects are collapsed joint AD support and a bounded
    point-level amplitude scale.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from generate_v87_easy_field_projection import summarize
from generate_v98_path_ensemble_graph_attention_joint_ad import projection
from generate_v159_frontier_consensus_support_soup import (
    FLOOR,
    REGIMES,
    collapsed_support_from_output,
    compare_outputs,
    digest,
    normalize_support,
)
from generate_v161_hminus3_tail_distillation import l1_h_change, row_cosine
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v169_power_scaled_tail_denoising"
OUT_DIR = PATHS.cache_dir(169)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(169)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v169_Round1_Test_Channel.npy"
TARGET_OUT = OUT_DIR / "v169_power_scaled_tail_denoising_target.npy"
WEIGHTS_OUT = OUT_DIR / "v169_power_scaled_tail_denoising_weights.npy"
ACTION_OUT = OUT_DIR / "v169_power_scaled_tail_denoising_actions.npz"
REPORT = REPORT_DIR / "v169_power_scaled_tail_denoising_generate.json"
ZIP_OUT = PATHS.round1_versions / "v169_power_scaled_tail_denoising_candidate_pending.zip"


def support_cache() -> Path:
    candidates = [
        OUT_DIR / "v163_collapsed_support.npy",
        PATHS.cache_dir(163) / "probe_v163_collapsed_support.npy",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return OUT_DIR / "v163_collapsed_support.npy"


def support_metrics(support: np.ndarray) -> dict[str, np.ndarray]:
    flat = support.reshape(len(support), -1)
    cells = flat.shape[1]
    sorted_flat = np.sort(flat, axis=1)
    entropy = (
        -np.sum(flat * np.log(np.maximum(flat, FLOOR)), axis=1) / np.log(cells)
    ).astype(np.float32)
    return {
        "top8_fraction": sorted_flat[:, -8:].sum(axis=1).astype(np.float32),
        "top64_fraction": sorted_flat[:, -64:].sum(axis=1).astype(np.float32),
        "top128_fraction": sorted_flat[:, -128:].sum(axis=1).astype(np.float32),
        "top512_fraction": sorted_flat[:, -512:].sum(axis=1).astype(np.float32),
        "tail128_floor": sorted_flat[:, :-128].mean(axis=1).astype(np.float32),
        "tail128_fraction": (sorted_flat[:, :-128].mean(axis=1) * cells).astype(
            np.float32
        ),
        "tail512_floor": sorted_flat[:, :-512].mean(axis=1).astype(np.float32),
        "tail512_fraction": (sorted_flat[:, :-512].mean(axis=1) * cells).astype(
            np.float32
        ),
        "tail1024_fraction": (sorted_flat[:, :-1024].mean(axis=1) * cells).astype(
            np.float32
        ),
        "entropy": entropy,
    }


def top_mask(flat: np.ndarray, k: int) -> np.ndarray:
    k = min(k, flat.shape[1])
    ids = np.argpartition(flat, -k, axis=1)[:, -k:]
    mask = np.zeros(flat.shape, dtype=bool)
    mask[np.arange(flat.shape[0])[:, None], ids] = True
    return mask


def build_target(
    anchor: np.ndarray,
    regimes: np.ndarray,
    risk: np.ndarray,
    previous_action: np.ndarray,
    family: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    metrics = support_metrics(anchor)
    flat = anchor.reshape(len(anchor), -1)
    cells = flat.shape[1]

    regime_scale = np.zeros(len(regimes), dtype=np.float32)
    regime_scale[regimes == "clear"] = 0.20
    regime_scale[regimes == "light"] = 0.28
    regime_scale[regimes == "medium"] = 0.66
    regime_scale[regimes == "heavy"] = 1.00

    tail_gate = np.clip((metrics["tail128_fraction"] - 0.080) / 0.200, 0.0, 1.0)
    entropy_gate = np.clip((metrics["entropy"] - 0.32) / 0.26, 0.0, 1.0)
    top64_loose_gate = np.clip((0.94 - metrics["top64_fraction"]) / 0.38, 0.0, 1.0)
    risk_gate = np.clip((risk - 0.38) / 0.54, 0.0, 1.0).astype(np.float32)
    prior_gate = np.clip((previous_action - 0.04) / 0.66, 0.0, 1.0).astype(
        np.float32
    )

    top1_los = family["test_top1_los"].astype(np.float32)
    top1_one = family["test_top1_one"].astype(np.float32)
    top1_two = family["test_top1_two"].astype(np.float32)
    two_power = family["test_two_power_frac"].astype(np.float32)
    assigned_power = family["test_assigned_power_frac"].astype(np.float32)

    simple_path_gate = np.clip(0.45 * top1_los + 0.35 * top1_one, 0.0, 1.0)
    multibounce_protect = (
        1.0
        - 0.45 * np.clip((top1_two - 0.68) / 0.24, 0.0, 1.0)
        - 0.22 * np.clip((two_power - 0.42) / 0.28, 0.0, 1.0)
    ).astype(np.float32)
    multibounce_protect = np.clip(multibounce_protect, 0.34, 1.0)

    geometry_confidence = np.clip((assigned_power - 0.18) / 0.52, 0.0, 1.0)

    base_gate = (
        regime_scale
        * tail_gate
        * np.clip(0.35 + 0.65 * entropy_gate, 0.0, 1.0)
        * np.clip(0.45 + 0.55 * top64_loose_gate, 0.0, 1.0)
        * np.clip(0.55 + 0.45 * geometry_confidence, 0.0, 1.0)
    ).astype(np.float32)

    # For medium/heavy, geodesic risk and the previous online-validated action
    # say whether the point is in the hard region where template-derived tails
    # were historically used.  For clear/light, a simple LOS/reflection path
    # raises denoising confidence instead.
    hard_mask = (regimes == "medium") | (regimes == "heavy")
    easy_mask = ~hard_mask
    contextual_gate = np.zeros(len(regimes), dtype=np.float32)
    contextual_gate[hard_mask] = np.clip(
        0.36 + 0.40 * risk_gate[hard_mask] + 0.24 * prior_gate[hard_mask], 0.0, 1.0
    )
    contextual_gate[easy_mask] = np.clip(
        0.44 + 0.56 * simple_path_gate[easy_mask], 0.0, 1.0
    )

    action = np.clip(base_gate * contextual_gate * multibounce_protect, 0.0, 1.0)

    tail_floor = np.maximum(metrics["tail128_floor"], FLOOR).astype(np.float32)
    snr = flat / tail_floor[:, None]
    log_snr = np.log(np.maximum(snr, 1e-6))
    keep = np.clip(
        (log_snr - np.log(1.20)) / (np.log(9.0) - np.log(1.20)),
        0.0,
        1.0,
    ).astype(np.float32)
    keep = (0.06 + 0.94 * keep).astype(np.float32)

    mask128 = top_mask(flat, 128)
    mask512 = top_mask(flat, 512)
    mask2048 = top_mask(flat, 2048)
    mask4096 = top_mask(flat, 4096)
    keep[mask4096] = np.maximum(keep[mask4096], 0.22)
    keep[mask2048] = np.maximum(keep[mask2048], 0.36)
    keep[mask512] = np.maximum(keep[mask512], 0.62)
    keep[mask128] = 1.0

    shrink_strength = (0.96 * action).astype(np.float32)
    attenuation = 1.0 - shrink_strength[:, None] * (1.0 - keep)
    attenuation = np.clip(attenuation, 0.12, 1.0).astype(np.float32)
    denoised_flat = (flat * attenuation).astype(np.float32)

    removed_mass = np.sum(flat - denoised_flat, axis=1).astype(np.float32)
    target = normalize_support(denoised_flat.reshape(anchor.shape))

    # Interpret only part of the removed tail as measurement-noise power.
    # The clip limits keep this from becoming a blind total-power retune.
    power_factor = (1.0 - 0.46 * removed_mass).astype(np.float32)
    lower = np.ones(len(regimes), dtype=np.float32)
    lower[regimes == "clear"] = 0.990
    lower[regimes == "light"] = 0.986
    lower[regimes == "medium"] = 0.958
    lower[regimes == "heavy"] = 0.932
    power_factor = np.clip(power_factor, lower, 1.0).astype(np.float32)

    diagnostics = {
        **metrics,
        "risk": risk.astype(np.float32),
        "previous_action": previous_action.astype(np.float32),
        "regime_scale": regime_scale,
        "tail_gate": tail_gate.astype(np.float32),
        "entropy_gate": entropy_gate.astype(np.float32),
        "top64_loose_gate": top64_loose_gate.astype(np.float32),
        "risk_gate": risk_gate.astype(np.float32),
        "prior_gate": prior_gate,
        "simple_path_gate": simple_path_gate.astype(np.float32),
        "multibounce_protect": multibounce_protect,
        "geometry_confidence": geometry_confidence.astype(np.float32),
        "contextual_gate": contextual_gate,
        "action": action,
        "shrink_strength": shrink_strength,
        "removed_mass": removed_mass,
        "power_factor": power_factor,
        "top1_los": top1_los,
        "top1_one": top1_one,
        "top1_two": top1_two,
        "two_power_frac": two_power,
        "assigned_power_frac": assigned_power,
        "cell_count": np.full(len(regimes), cells, dtype=np.float32),
    }
    return target, power_factor, diagnostics


def apply_power_scale(path: Path, power_factor: np.ndarray, batch: int = 20) -> None:
    output = np.load(path, mmap_mode="r+")
    amp = np.sqrt(power_factor).astype(np.float32)
    for start in range(0, len(output), batch):
        end = min(start + batch, len(output))
        output[start:end] = (output[start:end] * amp[start:end, None, None, None]).astype(
            np.complex64
        )
        output.flush()


def by_regime_summary(regimes: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for regime in REGIMES:
        mask = regimes == regime
        if not np.any(mask):
            continue
        payload[regime] = {
            "count": int(np.sum(mask)),
            **{name: summarize(value[mask]) for name, value in arrays.items()},
        }
    return payload


def main() -> None:
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked_test = np.load(PATHS.shared_features / "height_geometry_cache.npz")[
        "test_blocked"
    ].astype(np.float32)
    regimes = regime_of(blocked_test)
    risk = np.load(PATHS.cache_dir(140) / "v140_geodesic_risk.npy").astype(np.float32)

    anchor = collapsed_support_from_output(
        PATHS.cache_dir(163) / "v163_Round1_Test_Channel.npy",
        pos_test,
        support_cache(),
    )
    v163_actions = np.load(
        PATHS.cache_dir(163) / "v163_validated_momentum_overdrive_actions.npz"
    )
    previous_action = v163_actions["action"].astype(np.float32)
    family = np.load(PATHS.shared_features / "v166_path_family_predictability_predictions.npz")

    target, power_factor, diag = build_target(
        anchor, regimes, risk, previous_action, family
    )
    np.save(TARGET_OUT, target)

    action = diag["action"].astype(np.float32)
    max_weight = np.clip(0.70 + 0.28 * action, 0.70, 0.98).astype(np.float32)
    weights = np.empty((len(action), 8), dtype=np.float32)
    weights[:, 0] = max_weight
    weights[:, 1:] = (1.0 - max_weight[:, None]) / 7.0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)
    np.save(WEIGHTS_OUT, weights)

    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        weights=weights,
        **diag,
    )

    args = argparse.Namespace(
        base=str(PATHS.cache_dir(163) / "v163_Round1_Test_Channel.npy"),
        output=str(OUTPUT),
        beta=140.3,
        gamma=0.10,
        eta="clear:0.48,light:0.62,medium:0.92,heavy:1.16",
        min_gate=0.05,
        max_gate=2.94,
        conf_center=0.32,
        conf_width=0.32,
        entropy_low=0.04,
        entropy_high=0.94,
        gain_clip="0.18,3.48",
        top_gain_clip="0.62,1.92",
        top_components=512,
        target_floor=FLOOR,
        project_batch=20,
    )
    projection_report = projection(target, weights, regimes, pos_test, args)
    apply_power_scale(OUTPUT, power_factor)

    comparison_to_v163 = compare_outputs(
        OUTPUT, PATHS.cache_dir(163) / "v163_Round1_Test_Channel.npy", regimes
    )
    comparison_to_v162 = compare_outputs(
        OUTPUT, PATHS.cache_dir(162) / "v162_Round1_Test_Channel.npy", regimes
    )
    comparison_to_v140 = compare_outputs(
        OUTPUT, PATHS.cache_dir(140) / "v140_Round1_Test_Channel.npy", regimes
    )
    if comparison_to_v163["shape"] != [500, 256, 4, 192] or comparison_to_v163["dtype"] != "complex64":
        raise ValueError(
            f"invalid output layout: {comparison_to_v163['shape']} {comparison_to_v163['dtype']}"
        )
    if not comparison_to_v163["finite"]:
        raise ValueError("v169 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "Official train labels may contain power-dependent complex noise; "
            "therefore v163's weak AD tail is less reliable than its strong "
            "paths. v169 deterministically denoises low-SNR tail support and "
            "applies a bounded total-power shrink equal to part of the removed "
            "tail mass."
        ),
        "new_information_used": {
            "v168": "AD tail is circular-complex-like, and tail floor has Spearman 0.971 with total power",
            "user_clarification": "official may add noise to released train labels after withholding test labels",
            "v166": "high-confidence two-bounce/path-family signals are used only as protection against deleting real multipath",
            "v163": "current online best and phase/pol-UE base",
        },
        "excluded": {
            "random_noise_injection": True,
            "away_from_negative_common_subspace": True,
            "full_phase_relearning": True,
            "full_path_template_generation": True,
            "exact_zero_deletion": True,
        },
        "parameters": {
            "support_tail_floor": "mean of all cells excluding top128",
            "snr_keep": "0.06 + 0.94*clip((log(snr)-log(1.20))/(log(9)-log(1.20)))",
            "rank_protection": {
                "top128": "keep=1.0",
                "top512": "keep>=0.62",
                "top2048": "keep>=0.36",
                "top4096": "keep>=0.22",
            },
            "action": "regime_scale * tail_gate * entropy/top64 gates * context gate * multibounce protection",
            "regime_scale": {"clear": 0.20, "light": 0.28, "medium": 0.66, "heavy": 1.00},
            "shrink_strength": "0.96 * action",
            "power_factor": "clip(1 - 0.46*removed_tail_mass, regime lower bound, 1.0)",
            "projection_args": {
                key: (PATHS.relative(value) if key in {"base", "output"} else value)
                for key, value in vars(args).items()
            },
        },
        "diagnostics": {
            "overall": {
                key: summarize(value)
                for key, value in diag.items()
                if isinstance(value, np.ndarray) and value.ndim == 1 and key != "cell_count"
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "action": action,
                    "removed_mass": diag["removed_mass"],
                    "power_factor": power_factor,
                    "tail512_fraction": diag["tail512_fraction"],
                    "tail128_fraction": diag["tail128_fraction"],
                    "top64_fraction": diag["top64_fraction"],
                    "entropy": diag["entropy"],
                    "multibounce_protect": diag["multibounce_protect"],
                },
            ),
            "active_fraction": float(np.mean(action > 1e-6)),
            "power_shrink_fraction": float(np.mean(power_factor < 0.999999)),
        },
        "target_diagnostics": {
            "target_to_v163_support": summarize(row_cosine(target, anchor)),
            "target_h_l1_change_to_v163": summarize(l1_h_change(target, anchor)),
        },
        "projection": projection_report,
        "comparison_to_v163": comparison_to_v163,
        "comparison_to_v162": comparison_to_v162,
        "comparison_to_v140": comparison_to_v140,
        "preserved": {
            "base_complex": "v163",
            "strong_paths": "top512 joint AD cells are never attenuated in target construction",
            "phase": "inherits v163 phase through positive AD amplitude projection",
            "pol_ue_allocation": "projection scales each collapsed H/V/delay component uniformly over pol/UE",
        },
        "changed": {
            "collapsed_joint_ad_support": "low-SNR tail attenuation and renormalization",
            "total_power": "bounded deterministic shrink from removed tail mass",
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET_OUT),
            "weights": PATHS.relative(WEIGHTS_OUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source, name in (
            (__file__, "generate_v169_power_scaled_tail_denoising.py"),
            (PATHS.code / "analyze_v168_channel_awgn_power_audit.py", "analyze_v168_channel_awgn_power_audit.py"),
            (PATHS.code / "generate_v163_validated_momentum_overdrive.py", "generate_v163_validated_momentum_overdrive.py"),
            (PATHS.code / "generate_v159_frontier_consensus_support_soup.py", "generate_v159_frontier_consensus_support_soup.py"),
            (PATHS.code / "generate_v98_path_ensemble_graph_attention_joint_ad.py", "generate_v98_path_ensemble_graph_attention_joint_ad.py"),
            (PATHS.code / "generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
            (PATHS.code / "generate_v161_hminus3_tail_distillation.py", "generate_v161_hminus3_tail_distillation.py"),
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
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

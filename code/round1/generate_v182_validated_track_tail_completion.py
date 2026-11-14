#!/usr/bin/env python3
"""Generate v182: validated track-tail completion on top of v181.

v178-v181 form an online-positive chain, but v181 only improved v180 by a very
small margin.  v182 does not build a fifth-pass track target.  Instead it treats
the actual v178->v179->v180->v181 support movement as a validated direction and
continues only the points where the last movement is still aligned with the
previous two online-positive movements.

Preserved:
    v181 complex phase/backbone, point total power, clear/light points, and no
    train-truth peak prior.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np

import generate_v178_test_track_complex_consensus as track
from generate_v159_frontier_consensus_support_soup import (
    FLOOR,
    REGIMES,
    collapsed_support_from_output,
    compare_outputs,
    digest,
    normalize_support,
)
from generate_v161_hminus3_tail_distillation import row_cosine, weighted_log_direction
from generate_v87_easy_field_projection import summarize
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v182_validated_track_tail_completion"
OUT_DIR = PATHS.cache_dir(182)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(182)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v182_Round1_Test_Channel.npy"
SUPPORT_181 = OUT_DIR / "v182_v181_collapsed_support.npy"
TARGET_OUT = OUT_DIR / "v182_validated_track_tail_completion_target.npy"
ACTION_OUT = OUT_DIR / "v182_validated_track_tail_completion_actions.npz"
REPORT = REPORT_DIR / "v182_validated_track_tail_completion_generate.json"
ZIP_OUT = PATHS.round1_versions / "v182_validated_track_tail_completion_candidate_pending.zip"

BASE = PATHS.cache_dir(181) / "v181_Round1_Test_Channel.npy"
V180_BASE = PATHS.cache_dir(180) / "v180_Round1_Test_Channel.npy"
V179_BASE = PATHS.cache_dir(179) / "v179_Round1_Test_Channel.npy"
V178_BASE = PATHS.cache_dir(178) / "v178_Round1_Test_Channel.npy"
V173_BASE = PATHS.cache_dir(173) / "v173_Round1_Test_Channel.npy"

SUPPORT_178 = PATHS.cache_dir(179) / "v179_v178_collapsed_support.npy"
SUPPORT_179 = PATHS.cache_dir(180) / "v180_v179_collapsed_support.npy"
SUPPORT_180 = PATHS.cache_dir(181) / "v181_v180_collapsed_support.npy"
ACTION_181 = PATHS.cache_dir(181) / "v181_multihop_track_field_actions.npz"

MAX_TRACKS = 32


def rms_direction(direction: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.mean(direction.reshape(len(direction), -1).astype(np.float64) ** 2, axis=1)
    ).astype(np.float32)


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def configure_track_module() -> None:
    track.BASE = BASE
    track.OUTPUT = OUTPUT
    track.MAX_TRACKS = MAX_TRACKS


def load_supports(pos_test: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "v178": normalize_support(np.load(SUPPORT_178).astype(np.float32)),
        "v179": normalize_support(np.load(SUPPORT_179).astype(np.float32)),
        "v180": normalize_support(np.load(SUPPORT_180).astype(np.float32)),
        "v181": collapsed_support_from_output(BASE, pos_test, SUPPORT_181),
    }


def build_target(
    supports: dict[str, np.ndarray],
    regimes: np.ndarray,
    action_181: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    s178 = supports["v178"]
    s179 = supports["v179"]
    s180 = supports["v180"]
    s181 = supports["v181"]

    d_178_179 = weighted_log_direction(s178, s179)
    d_179_180 = weighted_log_direction(s179, s180)
    d_180_181 = weighted_log_direction(s180, s181)
    previous_blend = 0.65 * d_179_180 + 0.35 * d_178_179

    align_181_180 = row_cosine(d_180_181, d_179_180)
    align_180_179 = row_cosine(d_179_180, d_178_179)
    align_181_prev = row_cosine(d_180_181, previous_blend)
    mag_180_181 = rms_direction(d_180_181)
    mag_179_180 = rms_direction(d_179_180)

    track_conf = action_181["track_conf"].astype(np.float32)
    route = action_181["mean_route_evidence"].astype(np.float32)
    stable_energy = action_181["stable_energy_fraction"].astype(np.float32)
    persistence = action_181["mean_persistence"].astype(np.float32)
    mean_shift = action_181["mean_shift_abs"].astype(np.float32)
    net_alpha = action_181["net_alpha"].astype(np.float32)
    rollback_alpha = action_181["rollback_alpha"].astype(np.float32)
    support_l1 = action_181["support_l1_actual"].astype(np.float32)
    gain_max = action_181["gain_max"].astype(np.float32)

    heavy = regimes == "heavy"
    medium = regimes == "medium"
    movable = heavy | medium

    conf_gate = np.zeros(len(regimes), dtype=np.float32)
    conf_gate[heavy] = clip01((track_conf[heavy] - 0.55) / 0.20)
    conf_gate[medium] = clip01((track_conf[medium] - 0.53) / 0.18)

    route_gate = np.zeros(len(regimes), dtype=np.float32)
    route_gate[heavy] = clip01((route[heavy] - 0.075) / 0.105)
    route_gate[medium] = clip01((route[medium] - 0.035) / 0.095)

    energy_gate = np.zeros(len(regimes), dtype=np.float32)
    energy_gate[heavy] = clip01((stable_energy[heavy] - 0.28) / 0.48)
    energy_gate[medium] = clip01((stable_energy[medium] - 0.24) / 0.48)

    persistence_gate = np.zeros(len(regimes), dtype=np.float32)
    persistence_gate[heavy] = clip01((persistence[heavy] - 0.66) / 0.20)
    persistence_gate[medium] = clip01((persistence[medium] - 0.64) / 0.22)

    shift_gate = np.clip((1.12 - mean_shift) / 0.55, 0.22, 1.0).astype(np.float32)
    align_gate = (
        clip01((align_181_prev - 0.25) / 0.70)
        * (0.70 + 0.30 * clip01((align_181_180 + 0.05) / 0.80))
        * (0.80 + 0.20 * clip01((align_180_179 + 0.05) / 0.75))
    ).astype(np.float32)
    positive_gate = clip01((net_alpha + 0.006) / 0.22)
    magnitude_gate = clip01((np.maximum(mag_180_181, 0.72 * mag_179_180) - 0.00002) / 0.00036)
    not_overpushed = clip01(
        1.0
        - 0.55 * clip01((support_l1 - 0.082) / 0.040)
        - 0.35 * clip01((gain_max - 4.05) / 0.20)
        - 0.25 * clip01((rollback_alpha - 0.010) / 0.008)
    )

    regime_scale = np.zeros(len(regimes), dtype=np.float32)
    regime_scale[heavy] = 1.00
    regime_scale[medium] = 0.38

    action = (
        regime_scale
        * conf_gate
        * route_gate
        * energy_gate
        * persistence_gate
        * shift_gate
        * align_gate
        * positive_gate
        * magnitude_gate
        * not_overpushed
    ).astype(np.float32)

    # This is intentionally stronger than v181, but the gate is far narrower:
    # it only acts where three consecutive online-positive support directions
    # are mutually consistent.
    alpha = np.zeros(len(regimes), dtype=np.float32)
    alpha[heavy] = 2.80 * action[heavy]
    alpha[medium] = 0.90 * action[medium]

    raw_direction = (
        0.58 * (np.log(np.maximum(s181, FLOOR)) - np.log(np.maximum(s180, FLOOR)))
        + 0.30 * (np.log(np.maximum(s180, FLOOR)) - np.log(np.maximum(s179, FLOOR)))
        + 0.12 * (np.log(np.maximum(s179, FLOOR)) - np.log(np.maximum(s178, FLOOR)))
    ).astype(np.float32)
    raw_direction = np.clip(raw_direction, -1.80, 1.80).astype(np.float32)

    target = np.exp(
        np.log(np.maximum(s181, FLOOR)).astype(np.float32)
        + alpha[:, None, None, None] * raw_direction
    ).astype(np.float32)
    target = normalize_support(target)
    target[~movable] = s181[~movable]

    track_weights = action_181["track_weights"].astype(np.float32).copy()
    direction_scale = np.clip(0.35 + 0.75 * action, 0.0, 1.05).astype(np.float32)
    track_weights *= direction_scale[:, None]
    track_weights[~movable] = 0.0

    diagnostics = {
        "align_181_180": align_181_180,
        "align_180_179": align_180_179,
        "align_181_prev": align_181_prev,
        "mag_180_181": mag_180_181,
        "mag_179_180": mag_179_180,
        "track_conf": track_conf,
        "route": route,
        "stable_energy_fraction": stable_energy,
        "mean_persistence": persistence,
        "mean_shift_abs": mean_shift,
        "net_alpha_v181": net_alpha,
        "rollback_alpha_v181": rollback_alpha,
        "support_l1_v181": support_l1,
        "gain_max_v181": gain_max,
        "conf_gate": conf_gate,
        "route_gate": route_gate,
        "energy_gate": energy_gate,
        "persistence_gate": persistence_gate,
        "shift_gate": shift_gate,
        "align_gate": align_gate,
        "positive_gate": positive_gate,
        "magnitude_gate": magnitude_gate,
        "not_overpushed": not_overpushed,
        "action": action,
        "alpha": alpha,
        "target_support_l1": np.sum(np.abs(target - s181), axis=(1, 2, 3)).astype(np.float32),
        "track_weights": track_weights,
    }
    return target, diagnostics


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source, name in (
            (__file__, "generate_v182_validated_track_tail_completion.py"),
            (PATHS.code / "generate_v181_multihop_track_field.py", "generate_v181_multihop_track_field.py"),
            (PATHS.code / "generate_v180_track_saturation_bifurcation.py", "generate_v180_track_saturation_bifurcation.py"),
            (PATHS.code / "generate_v179_track_confidence_overdrive.py", "generate_v179_track_confidence_overdrive.py"),
            (PATHS.code / "generate_v178_test_track_complex_consensus.py", "generate_v178_test_track_complex_consensus.py"),
            (PATHS.code / "generate_v159_frontier_consensus_support_soup.py", "generate_v159_frontier_consensus_support_soup.py"),
            (PATHS.code / "generate_v161_hminus3_tail_distillation.py", "generate_v161_hminus3_tail_distillation.py"),
            (PATHS.code / "generate_v169_power_scaled_tail_denoising.py", "generate_v169_power_scaled_tail_denoising.py"),
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
    report["integrity"] = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": bool(np.isfinite(output.real).all() and np.isfinite(output.imag).all()),
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def generate() -> dict[str, object]:
    configure_track_module()
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    action_181 = np.load(ACTION_181)
    supports = load_supports(pos_test)
    target, diagnostics = build_target(supports, regimes, action_181)
    np.save(TARGET_OUT, target)

    apply_diag = track.apply_track_target(
        pos_test,
        target,
        action_181["track_cells"].astype(np.int16),
        action_181["track_dirs"].astype(np.complex64),
        diagnostics["track_weights"],
    )

    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        **{key: value for key, value in diagnostics.items() if key != "track_weights"},
        track_weights=diagnostics["track_weights"],
        **apply_diag,
    )

    comparison_to_v181 = compare_outputs(OUTPUT, BASE, regimes)
    comparison_to_v180 = compare_outputs(OUTPUT, V180_BASE, regimes)
    comparison_to_v179 = compare_outputs(OUTPUT, V179_BASE, regimes)
    comparison_to_v178 = compare_outputs(OUTPUT, V178_BASE, regimes)
    comparison_to_v173 = compare_outputs(OUTPUT, V173_BASE, regimes)
    if comparison_to_v181["shape"] != [500, 256, 4, 192] or comparison_to_v181["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v181['shape']} {comparison_to_v181['dtype']}")
    if not comparison_to_v181["finite"]:
        raise ValueError("v182 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v178-v181 are an online-positive component-track chain.  v182 "
            "continues only the remaining support tail where v180->v181 is "
            "still aligned with v179->v180 and v178->v179, so the aggressive "
            "step is tied to online-validated direction rather than another "
            "unfiltered track pass."
        ),
        "new_information_used": {
            "current_best": "v181 online 0.504284",
            "validated_chain": "v178->v179->v180->v181 online-positive support movement",
            "new_selector": "three-step support-direction alignment plus v181 route/confidence/not-overpushed gates",
            "guardrails": "clear/light untouched, per-point total power preserved, no train-truth peak prior",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "support_chain": {
                "v178": PATHS.relative(SUPPORT_178),
                "v179": PATHS.relative(SUPPORT_179),
                "v180": PATHS.relative(SUPPORT_180),
                "v181": PATHS.relative(SUPPORT_181),
            },
            "raw_direction_blend": "0.58*(v180->v181)+0.30*(v179->v180)+0.12*(v178->v179), clipped to [-1.80,1.80]",
            "alpha_scale": {"clear": 0.0, "light": 0.0, "medium": 0.90, "heavy": 2.80},
            "gate_inputs": [
                "align_181_prev",
                "align_181_180",
                "align_180_179",
                "track_conf",
                "mean_route_evidence",
                "stable_energy_fraction",
                "mean_persistence",
                "mean_shift_abs",
                "net_alpha_v181",
                "support_l1_v181",
                "gain_max_v181",
                "rollback_alpha_v181",
            ],
            "power_policy": "preserve v181 per-point total power",
            "direction_policy": "reuse v181 track directions with action-scaled weak weights",
        },
        "diagnostics": {
            "overall": {
                **{
                    key: summarize(value)
                    for key, value in diagnostics.items()
                    if isinstance(value, np.ndarray)
                    and value.ndim == 1
                    and np.issubdtype(value.dtype, np.number)
                },
                **{key: summarize(value) for key, value in apply_diag.items()},
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    key: value
                    for key, value in diagnostics.items()
                    if isinstance(value, np.ndarray)
                    and value.ndim == 1
                    and np.issubdtype(value.dtype, np.number)
                }
                | {key: value for key, value in apply_diag.items()},
            ),
        },
        "comparison_to_v181": comparison_to_v181,
        "comparison_to_v180": comparison_to_v180,
        "comparison_to_v179": comparison_to_v179,
        "comparison_to_v178": comparison_to_v178,
        "comparison_to_v173": comparison_to_v173,
        "preserved": {
            "base_complex": "v181",
            "point_total_power": "v181",
            "global_phase": "not learned; not globally regressed",
            "clear_light": "no support action",
            "randomness": "none",
        },
        "changed": {
            "support_tail": "medium/heavy support moves further along online-positive track-chain direction only where gates agree",
            "cell_internal_direction": "v181 track-direction smoothing is retained only in proportion to v182 action",
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "support_cache": PATHS.relative(SUPPORT_181),
            "target_support": PATHS.relative(TARGET_OUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    report = generate()
    report = write_zip(report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

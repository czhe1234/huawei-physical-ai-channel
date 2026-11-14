#!/usr/bin/env python3
"""Generate v180: bifurcated component-track saturation control.

v179 proved that a second-pass test-test component track overdrive can still
improve the online score.  v180 keeps that information source, but stops treating
all strong tracks as equivalent:

* continue only high-confidence, high-persistence, low-shift tracks;
* locally roll back a small part of points whose v179 support movement looks
  saturated or inconsistent;
* keep clear/light untouched and preserve v179 per-point total power.
"""

from __future__ import annotations

import json
import math
import zipfile

import numpy as np

import generate_v178_test_track_complex_consensus as track
from generate_v87_easy_field_projection import MH, MP, MV, N_ANT, S_SC, row_cos, summarize
from generate_v159_frontier_consensus_support_soup import REGIMES, digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v180_track_saturation_bifurcation"
OUT_DIR = PATHS.cache_dir(180)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(180)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v180_Round1_Test_Channel.npy"
SUPPORT_OUT = OUT_DIR / "v180_v179_collapsed_support.npy"
PEAKS_OUT = OUT_DIR / "v180_v179_top_peaks.npz"
TARGET_OUT = OUT_DIR / "v180_track_saturation_target_support.npy"
ACTION_OUT = OUT_DIR / "v180_track_saturation_bifurcation_actions.npz"
REPORT = REPORT_DIR / "v180_track_saturation_bifurcation_generate.json"
ZIP_OUT = PATHS.round1_versions / "v180_track_saturation_bifurcation_candidate_pending.zip"

BASE = PATHS.cache_dir(179) / "v179_Round1_Test_Channel.npy"
V178_BASE = PATHS.cache_dir(178) / "v178_Round1_Test_Channel.npy"
V173_BASE = PATHS.cache_dir(173) / "v173_Round1_Test_Channel.npy"
V178_SUPPORT = PATHS.cache_dir(179) / "v179_v178_collapsed_support.npy"
V173_ACTIONS = PATHS.cache_dir(173) / "v173_spatial_coherence_tail_denoising_actions.npz"
V178_ACTIONS = PATHS.cache_dir(178) / "v178_test_track_complex_consensus_actions.npz"
V179_ACTIONS = PATHS.cache_dir(179) / "v179_track_confidence_overdrive_actions.npz"

FLOOR = 1e-12
PEAK_COUNT = 64
TRACK_PEAKS = 28
MAX_TRACKS = 28
PEAK_CANDIDATES = 2560


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def configure_track_module() -> None:
    """Point v178 utilities at v180 paths and v179 as the base output."""
    track.BASE = BASE
    track.OUTPUT = OUTPUT
    track.SUPPORT_OUT = SUPPORT_OUT
    track.PEAKS_OUT = PEAKS_OUT
    track.TARGET_OUT = TARGET_OUT
    track.ACTION_OUT = ACTION_OUT
    track.REPORT = REPORT
    track.ZIP_OUT = ZIP_OUT
    track.PEAK_COUNT = PEAK_COUNT
    track.TRACK_PEAKS = TRACK_PEAKS
    track.PEAK_CANDIDATES = PEAK_CANDIDATES
    track.MAX_TRACKS = MAX_TRACKS


def continuation_gate(
    regime: str,
    track_conf: float,
    stable_energy_fraction: float,
    mean_persistence: float,
    mean_shift_abs: float,
    prior_cos: float,
    v179_alpha: float,
    v178_alpha: float,
) -> float:
    """Confidence gate for tracks that can safely continue past v179."""
    if regime == "heavy":
        conf = clip01((track_conf - 0.675) / 0.225)
        energy = clip01((stable_energy_fraction - 0.485) / 0.385)
        persistence = clip01((mean_persistence - 0.675) / 0.220)
        shift = clip01((1.020 - mean_shift_abs) / 0.550)
        cos_gate = clip01((prior_cos - 0.925) / 0.060)
        previous = clip01(v179_alpha / 0.34)
        not_saturated = 0.90 + 0.10 * clip01((0.40 - v179_alpha) / 0.18)
    elif regime == "medium":
        conf = clip01((track_conf - 0.650) / 0.230)
        energy = clip01((stable_energy_fraction - 0.470) / 0.360)
        persistence = clip01((mean_persistence - 0.660) / 0.240)
        shift = clip01((1.050 - mean_shift_abs) / 0.560)
        cos_gate = clip01((prior_cos - 0.905) / 0.070)
        previous = clip01(v179_alpha / 0.10)
        not_saturated = 0.92 + 0.08 * clip01((0.12 - v179_alpha) / 0.06)
    else:
        return 0.0

    # v179 already proved that the previous action is informative, but v180
    # should not require v179 to be maximal; a modest floor keeps coherent
    # newly-visible tracks alive.
    previous_gate = 0.55 + 0.45 * math.sqrt(previous)
    return clip01(conf * energy * persistence * shift * cos_gate * previous_gate * not_saturated)


def rollback_gate(
    regime: str,
    track_conf: float,
    mean_shift_abs: float,
    prior_cos: float,
    v179_alpha: float,
    support_l1: float,
    gain_max: float,
    continue_gate: float,
) -> float:
    """Small rollback gate for points where v179 likely over-moved support."""
    if regime == "heavy":
        cos_bad = clip01((0.966 - prior_cos) / 0.040)
        shift_bad = clip01((mean_shift_abs - 0.835) / 0.360)
        action_bad = clip01((support_l1 - 0.086) / 0.060)
        gain_bad = clip01((gain_max - 3.15) / 1.10)
        weak_conf = clip01((0.740 - track_conf) / 0.180)
        active = clip01((v179_alpha - 0.150) / 0.180)
        scale = 0.090
    elif regime == "medium":
        cos_bad = clip01((0.944 - prior_cos) / 0.055)
        shift_bad = clip01((mean_shift_abs - 0.910) / 0.400)
        action_bad = clip01((support_l1 - 0.040) / 0.036)
        gain_bad = clip01((gain_max - 1.270) / 0.320)
        weak_conf = clip01((0.710 - track_conf) / 0.185)
        active = clip01((v179_alpha - 0.045) / 0.055)
        scale = 0.040
    else:
        return 0.0

    risk = max(0.55 * cos_bad + 0.45 * shift_bad, 0.55 * action_bad + 0.45 * gain_bad)
    risk *= 0.45 + 0.55 * max(weak_conf, active)
    risk *= 1.0 - 0.75 * continue_gate
    return float(scale * clip01(risk))


def build_track_target(
    support: np.ndarray,
    support_v178: np.ndarray,
    peaks: np.ndarray,
    energy: np.ndarray,
    vectors: np.ndarray,
    neighbor_idx: np.ndarray,
    neighbor_weight: np.ndarray,
    regimes: np.ndarray,
    v173_action: np.ndarray,
    v178_alpha: np.ndarray,
    v179_alpha: np.ndarray,
    v179_support_l1: np.ndarray,
    v179_gain_max: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    target = np.empty_like(support, dtype=np.float32)
    track_cells = np.full((len(support), MAX_TRACKS, 3), -1, dtype=np.int16)
    track_dirs = np.zeros((len(support), MAX_TRACKS, MP, N_ANT), dtype=np.complex64)
    track_weights = np.zeros((len(support), MAX_TRACKS), dtype=np.float32)

    track_conf = np.zeros(len(support), dtype=np.float32)
    stable_track_count = np.zeros(len(support), dtype=np.float32)
    stable_energy_fraction = np.zeros(len(support), dtype=np.float32)
    mean_persistence = np.zeros(len(support), dtype=np.float32)
    mean_shift_abs = np.zeros(len(support), dtype=np.float32)
    prior_cos = np.ones(len(support), dtype=np.float32)
    continue_gate_arr = np.zeros(len(support), dtype=np.float32)
    rollback_gate_arr = np.zeros(len(support), dtype=np.float32)
    continue_alpha = np.zeros(len(support), dtype=np.float32)
    rollback_alpha = np.zeros(len(support), dtype=np.float32)

    for i in range(len(support)):
        field = (0.22 * support[i]).astype(np.float32)
        total_top_energy = float(np.sum(energy[i, :18])) + FLOOR
        stable_energy = 0.0
        persistence_acc = 0.0
        shift_acc = 0.0
        track_slot = 0

        for pidx in range(TRACK_PEAKS):
            h0, v0, d0 = peaks[i, pidx]
            if h0 < 0:
                break
            e0 = float(energy[i, pidx])
            if e0 < 5.5e-5:
                continue
            current_dir = track.unit_vec(vectors[i, pidx])

            weighted_score = 0.0
            weighted_presence = 0.0
            dh_sum = 0.0
            dv_sum = 0.0
            dd_sum = 0.0
            neighbor_energy_sum = 0.0
            dir_sum = (0.66 * current_dir).astype(np.complex64)
            dir_weight = 0.66
            matched_neighbors = 0

            for nslot, nb in enumerate(neighbor_idx[i]):
                qslot, score, delta = track.match_neighbor_peak(
                    (int(h0), int(v0), int(d0)),
                    peaks[nb],
                    energy[nb],
                )
                if qslot < 0:
                    continue
                w = float(neighbor_weight[i, nslot]) * score / math.sqrt(max(e0, FLOOR))
                if w <= 0:
                    continue
                matched_neighbors += 1
                presence = float(neighbor_weight[i, nslot]) * min(1.0, score / math.sqrt(max(e0, FLOOR)))
                weighted_presence += presence
                weighted_score += w
                dh_sum += w * delta[0]
                dv_sum += w * delta[1]
                dd_sum += w * delta[2]
                neighbor_energy_sum += w * float(energy[nb, qslot])

                ndir = track.unit_vec(vectors[nb, qslot])
                inner = np.sum(np.conj(ndir) * current_dir)
                phase = inner / max(abs(inner), 1e-12)
                dir_sum += np.complex64(w) * (ndir * phase).astype(np.complex64)
                dir_weight += w

            persistence = float(np.clip(weighted_presence, 0.0, 1.0))
            if matched_neighbors < 7 or persistence < 0.43 or weighted_score <= 0:
                continue

            mean_dh = float(np.clip(dh_sum / weighted_score, -1.55, 1.55))
            mean_dv = float(np.clip(dv_sum / weighted_score, -1.00, 1.00))
            mean_dd = float(np.clip(dd_sum / weighted_score, -3.45, 3.45))
            h_track = (int(h0) + int(round(0.52 * mean_dh))) % MH
            v_track = (int(v0) + int(round(0.52 * mean_dv))) % MV
            d_track = int(np.clip(int(d0) + int(round(0.52 * mean_dd)), 0, S_SC - 1))

            neighbor_mean_energy = neighbor_energy_sum / weighted_score
            amp_ratio = float(np.clip(math.sqrt(max(neighbor_mean_energy, FLOOR) / max(e0, FLOOR)), 0.80, 1.24))
            mass = e0 * (0.62 + 0.80 * persistence) * amp_ratio
            track.add_peak_kernel(field, h_track, v_track, d_track, mass, h_sigma=0.84, v_sigma=0.70, d_sigma=2.05)

            stable_energy += e0 * persistence
            persistence_acc += persistence
            shift_acc += abs(mean_dh) + 0.65 * abs(mean_dv) + 0.24 * abs(mean_dd)
            stable_track_count[i] += 1.0

            if track_slot < MAX_TRACKS:
                target_dir = track.unit_vec(dir_sum / np.complex64(max(dir_weight, 1e-12)))
                track_cells[i, track_slot] = (h_track, v_track, d_track)
                track_dirs[i, track_slot] = target_dir
                if str(regimes[i]) == "medium":
                    direction_scale = 0.32
                elif str(regimes[i]) == "heavy":
                    direction_scale = 0.58
                else:
                    direction_scale = 0.00
                track_weights[i, track_slot] = np.float32(
                    direction_scale * min(0.125, 0.014 + 0.130 * persistence)
                )
                track_slot += 1

        if stable_track_count[i] > 0:
            field = np.maximum(field, FLOOR)
            prior = field / np.maximum(field.sum(), TINY)
            stable_energy_fraction[i] = np.float32(np.clip(stable_energy / total_top_energy, 0.0, 1.0))
            mean_persistence[i] = np.float32(persistence_acc / stable_track_count[i])
            mean_shift_abs[i] = np.float32(shift_acc / stable_track_count[i])
            prior_cos[i] = float(row_cos(prior[None], support[i : i + 1])[0])
        else:
            prior = support[i]
            prior_cos[i] = 1.0

        conf = np.clip(
            0.49 * stable_energy_fraction[i]
            + 0.42 * mean_persistence[i]
            + 0.09 * np.clip(stable_track_count[i] / 12.0, 0.0, 1.0),
            0.0,
            1.0,
        )
        track_conf[i] = np.float32(conf)

        regime = str(regimes[i])
        cont_gate = continuation_gate(
            regime,
            float(track_conf[i]),
            float(stable_energy_fraction[i]),
            float(mean_persistence[i]),
            float(mean_shift_abs[i]),
            float(prior_cos[i]),
            float(v179_alpha[i]),
            float(v178_alpha[i]),
        )
        rb_gate = rollback_gate(
            regime,
            float(track_conf[i]),
            float(mean_shift_abs[i]),
            float(prior_cos[i]),
            float(v179_alpha[i]),
            float(v179_support_l1[i]),
            float(v179_gain_max[i]),
            cont_gate,
        )

        if regime == "heavy":
            cont_scale = 0.70
        elif regime == "medium":
            cont_scale = 0.30
        else:
            cont_scale = 0.0

        spatial_gate = 0.72 + 0.28 * math.sqrt(float(np.clip(v173_action[i], 0.0, 1.0)))
        cont_alpha = cont_scale * cont_gate * spatial_gate
        rb_alpha = rb_gate
        if cont_alpha + rb_alpha > 0.58:
            shrink = 0.58 / (cont_alpha + rb_alpha)
            cont_alpha *= shrink
            rb_alpha *= shrink

        continue_gate_arr[i] = np.float32(cont_gate)
        rollback_gate_arr[i] = np.float32(rb_gate)
        continue_alpha[i] = np.float32(cont_alpha)
        rollback_alpha[i] = np.float32(rb_alpha)

        mixed = (
            (1.0 - continue_alpha[i] - rollback_alpha[i]) * support[i]
            + continue_alpha[i] * prior
            + rollback_alpha[i] * support_v178[i]
        )
        target[i] = track.normalize_support(mixed[None])[0]

        if rollback_alpha[i] > continue_alpha[i]:
            track_weights[i, :] *= np.float32(0.35)
        else:
            track_weights[i, :] *= np.float32(0.65 + 0.35 * cont_gate)

        if (i + 1) % 100 == 0:
            print(f"v180 saturation target {i + 1}/{len(support)}", flush=True)

    diagnostics = {
        "track_cells": track_cells,
        "track_dirs": track_dirs,
        "track_weights": track_weights,
        "track_conf": track_conf,
        "stable_track_count": stable_track_count,
        "stable_energy_fraction": stable_energy_fraction,
        "mean_persistence": mean_persistence,
        "mean_shift_abs": mean_shift_abs,
        "prior_cos": prior_cos,
        "continue_gate": continue_gate_arr,
        "rollback_gate": rollback_gate_arr,
        "continue_alpha": continue_alpha,
        "rollback_alpha": rollback_alpha,
        "net_alpha": (continue_alpha - rollback_alpha).astype(np.float32),
        "target_support_l1": np.sum(np.abs(target - support), axis=(1, 2, 3)).astype(np.float32),
        "target_support_cos": row_cos(target, support).astype(np.float32),
        "rollback_to_v178_l1": np.sum(np.abs(target - support_v178), axis=(1, 2, 3)).astype(np.float32),
    }
    return track.normalize_support(target), diagnostics


def generate() -> dict[str, object]:
    configure_track_module()
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    v173_action = np.load(V173_ACTIONS)["action"].astype(np.float32)
    v178_data = np.load(V178_ACTIONS)
    v179_data = np.load(V179_ACTIONS)
    v178_alpha = v178_data["alpha"].astype(np.float32)
    v179_alpha = v179_data["alpha"].astype(np.float32)
    v179_support_l1 = v179_data["support_l1_actual"].astype(np.float32)
    v179_gain_max = v179_data["gain_max"].astype(np.float32)
    support_v178 = track.normalize_support(np.load(V178_SUPPORT).astype(np.float32))

    support, peaks, energy, vectors = track.extract_v173_support_and_peaks(pos_test)
    idx, weights = track.neighbor_graph(pos_test, blocked, regimes)
    target, track_diag = build_track_target(
        support,
        support_v178,
        peaks,
        energy,
        vectors,
        idx,
        weights,
        regimes,
        v173_action,
        v178_alpha,
        v179_alpha,
        v179_support_l1,
        v179_gain_max,
    )
    np.save(TARGET_OUT, target)
    apply_diag = track.apply_track_target(
        pos_test,
        target,
        track_diag["track_cells"],
        track_diag["track_dirs"],
        track_diag["track_weights"],
    )

    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        neighbor_idx=idx,
        neighbor_weights=weights,
        v173_action=v173_action,
        v178_alpha=v178_alpha,
        v179_alpha=v179_alpha,
        v179_support_l1=v179_support_l1,
        v179_gain_max=v179_gain_max,
        **track_diag,
        **apply_diag,
    )

    comparison_to_v179 = track.compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v178 = track.compare_output_to(OUTPUT, V178_BASE, regimes)
    comparison_to_v173 = track.compare_output_to(OUTPUT, V173_BASE, regimes)
    if comparison_to_v179["shape"] != [500, 256, 4, 192] or comparison_to_v179["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v179['shape']} {comparison_to_v179['dtype']}")
    if not comparison_to_v179["finite"]:
        raise ValueError("v180 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v179 proved second-pass test-test component tracks still improve online. "
            "v180 adds per-track saturation control: continue only coherent low-shift "
            "tracks, and softly roll back likely over-moved points toward v178 support."
        ),
        "new_information_used": {
            "current_best": "v179 online 0.504125",
            "third_pass_base": PATHS.relative(BASE),
            "validated_direction": "v178->v179 online-positive track overdrive",
            "saturation_signals": "track confidence/persistence/stable energy/mean shift/prior cosine/v179 support_l1/gain_max",
            "guardrails": "clear/light untouched, per-point total power preserved, no train-truth peak prior",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "rollback_support": PATHS.relative(V178_SUPPORT),
            "peak_count": PEAK_COUNT,
            "track_peaks_per_point": TRACK_PEAKS,
            "peak_candidates": PEAK_CANDIDATES,
            "neighbor_k": 18,
            "neighbor_metric": "XY + 0.18*blocked_delta + 4.0*regime_mismatch",
            "peak_energy_floor": 5.5e-5,
            "matched_neighbors_min": 7,
            "persistence_min": 0.43,
            "prior_floor": 0.22,
            "continue_alpha_scale": {"clear": 0.0, "light": 0.0, "medium": 0.30, "heavy": 0.70},
            "rollback_alpha_scale": {"clear": 0.0, "light": 0.0, "medium": 0.040, "heavy": 0.090},
            "max_combined_alpha": 0.58,
            "power_policy": "preserve v179 per-point total power",
        },
        "diagnostics": {
            "overall": {
                "v173_action": summarize(v173_action),
                "v178_alpha": summarize(v178_alpha),
                "v179_alpha": summarize(v179_alpha),
                "v179_support_l1": summarize(v179_support_l1),
                "v179_gain_max": summarize(v179_gain_max),
                **{
                    key: summarize(value)
                    for key, value in track_diag.items()
                    if isinstance(value, np.ndarray)
                    and value.ndim == 1
                    and np.issubdtype(value.dtype, np.number)
                },
                **{key: summarize(value) for key, value in apply_diag.items()},
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "v173_action": v173_action,
                    "v178_alpha": v178_alpha,
                    "v179_alpha": v179_alpha,
                    "track_conf": track_diag["track_conf"],
                    "stable_track_count": track_diag["stable_track_count"],
                    "stable_energy_fraction": track_diag["stable_energy_fraction"],
                    "mean_persistence": track_diag["mean_persistence"],
                    "mean_shift_abs": track_diag["mean_shift_abs"],
                    "prior_cos": track_diag["prior_cos"],
                    "continue_gate": track_diag["continue_gate"],
                    "rollback_gate": track_diag["rollback_gate"],
                    "continue_alpha": track_diag["continue_alpha"],
                    "rollback_alpha": track_diag["rollback_alpha"],
                    "net_alpha": track_diag["net_alpha"],
                    "target_support_l1": track_diag["target_support_l1"],
                    "support_l1_actual": apply_diag["support_l1_actual"],
                    "direction_update_energy": apply_diag["direction_update_energy"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                },
            ),
        },
        "comparison_to_v179": comparison_to_v179,
        "comparison_to_v178": comparison_to_v178,
        "comparison_to_v173": comparison_to_v173,
        "preserved": {
            "base_complex": "v179",
            "point_total_power": "v179",
            "global_phase": "not learned; not globally regressed",
            "clear_light": "no support action",
            "randomness": "none",
        },
        "changed": {
            "tracked_support": "medium/heavy high-confidence tracks get third-pass continuation",
            "saturation_control": "likely over-moved points softly blend back toward v178 collapsed support",
            "cell_internal_direction": "tracked cells receive gated phase-aligned 2x4 direction smoothing",
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "support_cache": PATHS.relative(SUPPORT_OUT),
            "peaks": PATHS.relative(PEAKS_OUT),
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
            (__file__, "generate_v180_track_saturation_bifurcation.py"),
            (PATHS.code / "generate_v179_track_confidence_overdrive.py", "generate_v179_track_confidence_overdrive.py"),
            (PATHS.code / "generate_v178_test_track_complex_consensus.py", "generate_v178_test_track_complex_consensus.py"),
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

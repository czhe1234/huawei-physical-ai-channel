#!/usr/bin/env python3
"""Generate v181: multi-hop test-test component-track field.

v178-v180 all used local one-hop component tracks.  v181 keeps the same
train-truth-free guardrails, but asks for a stronger signal: a path component
should remain trackable not only in immediate neighbors, but also through
neighbors-of-neighbors along the test receiver graph.

Preserved:
    v180 point total power, clear/light, no global phase regression, no train
    truth peak prior.
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


VERSION = "v181_multihop_track_field"
OUT_DIR = PATHS.cache_dir(181)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(181)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v181_Round1_Test_Channel.npy"
SUPPORT_OUT = OUT_DIR / "v181_v180_collapsed_support.npy"
PEAKS_OUT = OUT_DIR / "v181_v180_top_peaks.npz"
TARGET_OUT = OUT_DIR / "v181_multihop_track_target_support.npy"
ACTION_OUT = OUT_DIR / "v181_multihop_track_field_actions.npz"
REPORT = REPORT_DIR / "v181_multihop_track_field_generate.json"
ZIP_OUT = PATHS.round1_versions / "v181_multihop_track_field_candidate_pending.zip"

BASE = PATHS.cache_dir(180) / "v180_Round1_Test_Channel.npy"
V179_BASE = PATHS.cache_dir(179) / "v179_Round1_Test_Channel.npy"
V178_BASE = PATHS.cache_dir(178) / "v178_Round1_Test_Channel.npy"
V173_BASE = PATHS.cache_dir(173) / "v173_Round1_Test_Channel.npy"
V179_SUPPORT = PATHS.cache_dir(180) / "v180_v179_collapsed_support.npy"
V173_ACTIONS = PATHS.cache_dir(173) / "v173_spatial_coherence_tail_denoising_actions.npz"
V180_ACTIONS = PATHS.cache_dir(180) / "v180_track_saturation_bifurcation_actions.npz"

FLOOR = 1e-12
PEAK_COUNT = 72
TRACK_PEAKS = 32
MAX_TRACKS = 32
PEAK_CANDIDATES = 3072
DIRECT_K = 18
EXTENDED_K = 44


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def configure_track_module() -> None:
    """Point shared v178 utilities at v181 paths and v180 as the base."""
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


def multihop_neighbor_graph(
    pos: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    direct_idx, direct_w = track.neighbor_graph(pos, blocked, regimes, k=DIRECT_K)
    extended_idx = np.full((len(pos), EXTENDED_K), -1, dtype=np.int32)
    extended_w = np.zeros((len(pos), EXTENDED_K), dtype=np.float32)
    twohop_flag = np.zeros((len(pos), EXTENDED_K), dtype=np.float32)
    direct_mass = np.zeros(len(pos), dtype=np.float32)
    twohop_mass = np.zeros(len(pos), dtype=np.float32)

    for i in range(len(pos)):
        weights: dict[int, float] = {}
        direct_nodes: set[int] = set()
        for slot, nb in enumerate(direct_idx[i]):
            node = int(nb)
            direct_nodes.add(node)
            weights[node] = weights.get(node, 0.0) + 0.74 * float(direct_w[i, slot])

        for slot, nb in enumerate(direct_idx[i]):
            first = int(nb)
            first_w = float(direct_w[i, slot])
            for slot2, nb2 in enumerate(direct_idx[first]):
                node = int(nb2)
                if node == i:
                    continue
                # Two-hop evidence is lower weight, but can reveal corridor
                # components that are weak in a single immediate neighbor.
                penalty = 0.88 if regimes[node] == regimes[i] else 0.48
                weights[node] = weights.get(node, 0.0) + 0.46 * first_w * float(direct_w[first, slot2]) * penalty

        if not weights:
            continue
        ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:EXTENDED_K]
        total = sum(value for _, value in ordered)
        if total <= 0:
            continue
        for out_slot, (node, value) in enumerate(ordered):
            w = float(value / total)
            extended_idx[i, out_slot] = int(node)
            extended_w[i, out_slot] = np.float32(w)
            if node in direct_nodes:
                direct_mass[i] += np.float32(w)
            else:
                twohop_flag[i, out_slot] = 1.0
                twohop_mass[i] += np.float32(w)

    return direct_idx, direct_w, extended_idx, extended_w, twohop_flag, direct_mass, twohop_mass


def continuation_gate(
    regime: str,
    track_conf: float,
    stable_energy_fraction: float,
    mean_persistence: float,
    mean_shift_abs: float,
    mean_route_evidence: float,
    prior_cos: float,
    v180_continue_alpha: float,
    v180_rollback_alpha: float,
    v180_support_l1: float,
) -> float:
    if regime == "heavy":
        conf = clip01((track_conf - 0.555) / 0.230)
        energy = clip01((stable_energy_fraction - 0.370) / 0.420)
        persistence = clip01((mean_persistence - 0.640) / 0.240)
        route = 0.70 + 0.30 * clip01((mean_route_evidence - 0.045) / 0.190)
        shift = clip01((1.090 - mean_shift_abs) / 0.620)
        cos_gate = clip01((prior_cos - 0.920) / 0.060)
        previous = clip01((v180_continue_alpha - v180_rollback_alpha + 0.055) / 0.210)
        not_overwide = 0.86 + 0.14 * clip01((0.115 - v180_support_l1) / 0.085)
    elif regime == "medium":
        conf = clip01((track_conf - 0.540) / 0.240)
        energy = clip01((stable_energy_fraction - 0.360) / 0.420)
        persistence = clip01((mean_persistence - 0.625) / 0.260)
        route = 0.72 + 0.28 * clip01((mean_route_evidence - 0.035) / 0.170)
        shift = clip01((1.080 - mean_shift_abs) / 0.600)
        cos_gate = clip01((prior_cos - 0.905) / 0.070)
        previous = clip01((v180_continue_alpha - v180_rollback_alpha + 0.030) / 0.090)
        not_overwide = 0.90 + 0.10 * clip01((0.045 - v180_support_l1) / 0.045)
    else:
        return 0.0

    previous_gate = 0.70 + 0.30 * math.sqrt(previous)
    return clip01(conf * energy * persistence * route * shift * cos_gate * previous_gate * not_overwide)


def rollback_gate(
    regime: str,
    track_conf: float,
    mean_route_evidence: float,
    mean_shift_abs: float,
    prior_cos: float,
    v180_continue_alpha: float,
    v180_rollback_alpha: float,
    v180_support_l1: float,
    v180_gain_max: float,
    continue_gate_value: float,
) -> float:
    if regime == "heavy":
        weak_route = clip01((0.135 - mean_route_evidence) / 0.135)
        weak_conf = clip01((0.720 - track_conf) / 0.170)
        cos_bad = clip01((0.956 - prior_cos) / 0.045)
        shift_bad = clip01((mean_shift_abs - 0.860) / 0.360)
        action_bad = clip01((v180_support_l1 - 0.060) / 0.060)
        gain_bad = clip01((v180_gain_max - 2.80) / 1.20)
        active = clip01((v180_continue_alpha - v180_rollback_alpha - 0.040) / 0.220)
        scale = 0.030
    elif regime == "medium":
        weak_route = clip01((0.105 - mean_route_evidence) / 0.105)
        weak_conf = clip01((0.705 - track_conf) / 0.170)
        cos_bad = clip01((0.938 - prior_cos) / 0.055)
        shift_bad = clip01((mean_shift_abs - 0.930) / 0.390)
        action_bad = clip01((v180_support_l1 - 0.024) / 0.030)
        gain_bad = clip01((v180_gain_max - 1.200) / 0.420)
        active = clip01((v180_continue_alpha - v180_rollback_alpha - 0.010) / 0.070)
        scale = 0.012
    else:
        return 0.0

    risk = max(0.60 * weak_route + 0.40 * weak_conf, 0.55 * cos_bad + 0.45 * shift_bad)
    risk = max(risk, 0.55 * action_bad + 0.45 * gain_bad)
    risk *= 0.55 + 0.45 * active
    risk *= 1.0 - 0.80 * continue_gate_value
    return float(scale * clip01(risk))


def build_multihop_target(
    support: np.ndarray,
    support_v179: np.ndarray,
    peaks: np.ndarray,
    energy: np.ndarray,
    vectors: np.ndarray,
    extended_idx: np.ndarray,
    extended_w: np.ndarray,
    twohop_flag: np.ndarray,
    regimes: np.ndarray,
    v173_action: np.ndarray,
    v180_continue_alpha: np.ndarray,
    v180_rollback_alpha: np.ndarray,
    v180_support_l1: np.ndarray,
    v180_gain_max: np.ndarray,
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
    mean_route_evidence = np.zeros(len(support), dtype=np.float32)
    prior_cos = np.ones(len(support), dtype=np.float32)
    continue_gate_arr = np.zeros(len(support), dtype=np.float32)
    rollback_gate_arr = np.zeros(len(support), dtype=np.float32)
    continue_alpha = np.zeros(len(support), dtype=np.float32)
    rollback_alpha = np.zeros(len(support), dtype=np.float32)

    for i in range(len(support)):
        field = (0.20 * support[i]).astype(np.float32)
        total_top_energy = float(np.sum(energy[i, :20])) + FLOOR
        stable_energy = 0.0
        persistence_acc = 0.0
        route_acc = 0.0
        shift_acc = 0.0
        track_slot = 0

        for pidx in range(TRACK_PEAKS):
            h0, v0, d0 = peaks[i, pidx]
            if h0 < 0:
                break
            e0 = float(energy[i, pidx])
            if e0 < 4.5e-5:
                continue
            current_dir = track.unit_vec(vectors[i, pidx])

            weighted_score = 0.0
            weighted_presence = 0.0
            direct_presence = 0.0
            twohop_presence = 0.0
            dh_sum = 0.0
            dv_sum = 0.0
            dd_sum = 0.0
            neighbor_energy_sum = 0.0
            dir_sum = (0.62 * current_dir).astype(np.complex64)
            dir_weight = 0.62
            matched_direct = 0
            matched_total = 0

            for nslot, nb in enumerate(extended_idx[i]):
                if nb < 0:
                    break
                qslot, score, delta = track.match_neighbor_peak(
                    (int(h0), int(v0), int(d0)),
                    peaks[int(nb)],
                    energy[int(nb)],
                )
                if qslot < 0:
                    continue
                is_twohop = float(twohop_flag[i, nslot]) > 0.5
                hop_scale = 0.72 if is_twohop else 1.0
                w = float(extended_w[i, nslot]) * hop_scale * score / math.sqrt(max(e0, FLOOR))
                if w <= 0:
                    continue
                matched_total += 1
                if not is_twohop:
                    matched_direct += 1
                presence = float(extended_w[i, nslot]) * hop_scale * min(1.0, score / math.sqrt(max(e0, FLOOR)))
                weighted_presence += presence
                if is_twohop:
                    twohop_presence += presence
                else:
                    direct_presence += presence
                weighted_score += w
                dh_sum += w * delta[0]
                dv_sum += w * delta[1]
                dd_sum += w * delta[2]
                neighbor_energy_sum += w * float(energy[int(nb), qslot])

                ndir = track.unit_vec(vectors[int(nb), qslot])
                inner = np.sum(np.conj(ndir) * current_dir)
                phase = inner / max(abs(inner), 1e-12)
                dir_sum += np.complex64(w) * (ndir * phase).astype(np.complex64)
                dir_weight += w

            persistence = float(np.clip(direct_presence + 0.88 * twohop_presence, 0.0, 1.0))
            route_evidence = float(np.clip(twohop_presence * 2.65, 0.0, 1.0))
            if matched_direct < 4 or matched_total < 9 or persistence < 0.44 or weighted_score <= 0:
                continue

            mean_dh = float(np.clip(dh_sum / weighted_score, -1.65, 1.65))
            mean_dv = float(np.clip(dv_sum / weighted_score, -1.05, 1.05))
            mean_dd = float(np.clip(dd_sum / weighted_score, -3.65, 3.65))
            h_track = (int(h0) + int(round(0.50 * mean_dh))) % MH
            v_track = (int(v0) + int(round(0.50 * mean_dv))) % MV
            d_track = int(np.clip(int(d0) + int(round(0.50 * mean_dd)), 0, S_SC - 1))

            neighbor_mean_energy = neighbor_energy_sum / weighted_score
            amp_ratio = float(np.clip(math.sqrt(max(neighbor_mean_energy, FLOOR) / max(e0, FLOOR)), 0.78, 1.28))
            mass = e0 * (0.54 + 0.70 * persistence + 0.34 * route_evidence) * amp_ratio
            track.add_peak_kernel(field, h_track, v_track, d_track, mass, h_sigma=0.90, v_sigma=0.74, d_sigma=2.28)

            stable_energy += e0 * persistence
            persistence_acc += persistence
            route_acc += route_evidence
            shift_acc += abs(mean_dh) + 0.65 * abs(mean_dv) + 0.24 * abs(mean_dd)
            stable_track_count[i] += 1.0

            if track_slot < MAX_TRACKS:
                target_dir = track.unit_vec(dir_sum / np.complex64(max(dir_weight, 1e-12)))
                track_cells[i, track_slot] = (h_track, v_track, d_track)
                track_dirs[i, track_slot] = target_dir
                if str(regimes[i]) == "medium":
                    direction_scale = 0.24
                elif str(regimes[i]) == "heavy":
                    direction_scale = 0.42
                else:
                    direction_scale = 0.0
                track_weights[i, track_slot] = np.float32(
                    direction_scale * min(0.105, 0.010 + 0.105 * persistence + 0.030 * route_evidence)
                )
                track_slot += 1

        if stable_track_count[i] > 0:
            field = np.maximum(field, FLOOR)
            prior = field / np.maximum(field.sum(), TINY)
            stable_energy_fraction[i] = np.float32(np.clip(stable_energy / total_top_energy, 0.0, 1.0))
            mean_persistence[i] = np.float32(persistence_acc / stable_track_count[i])
            mean_route_evidence[i] = np.float32(route_acc / stable_track_count[i])
            mean_shift_abs[i] = np.float32(shift_acc / stable_track_count[i])
            prior_cos[i] = float(row_cos(prior[None], support[i : i + 1])[0])
        else:
            prior = support[i]
            prior_cos[i] = 1.0

        conf = np.clip(
            0.40 * stable_energy_fraction[i]
            + 0.33 * mean_persistence[i]
            + 0.17 * mean_route_evidence[i]
            + 0.10 * np.clip(stable_track_count[i] / 14.0, 0.0, 1.0),
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
            float(mean_route_evidence[i]),
            float(prior_cos[i]),
            float(v180_continue_alpha[i]),
            float(v180_rollback_alpha[i]),
            float(v180_support_l1[i]),
        )
        rb_gate = rollback_gate(
            regime,
            float(track_conf[i]),
            float(mean_route_evidence[i]),
            float(mean_shift_abs[i]),
            float(prior_cos[i]),
            float(v180_continue_alpha[i]),
            float(v180_rollback_alpha[i]),
            float(v180_support_l1[i]),
            float(v180_gain_max[i]),
            cont_gate,
        )

        if regime == "heavy":
            cont_scale = 0.92
        elif regime == "medium":
            cont_scale = 0.38
        else:
            cont_scale = 0.0

        spatial_gate = 0.66 + 0.34 * math.sqrt(float(np.clip(v173_action[i], 0.0, 1.0)))
        cont_alpha = cont_scale * cont_gate * spatial_gate
        rb_alpha = rb_gate
        if cont_alpha + rb_alpha > 0.66:
            shrink = 0.66 / (cont_alpha + rb_alpha)
            cont_alpha *= shrink
            rb_alpha *= shrink

        continue_gate_arr[i] = np.float32(cont_gate)
        rollback_gate_arr[i] = np.float32(rb_gate)
        continue_alpha[i] = np.float32(cont_alpha)
        rollback_alpha[i] = np.float32(rb_alpha)

        mixed = (
            (1.0 - continue_alpha[i] - rollback_alpha[i]) * support[i]
            + continue_alpha[i] * prior
            + rollback_alpha[i] * support_v179[i]
        )
        target[i] = track.normalize_support(mixed[None])[0]

        if rollback_alpha[i] > continue_alpha[i]:
            track_weights[i, :] *= np.float32(0.35)
        else:
            track_weights[i, :] *= np.float32(0.60 + 0.40 * cont_gate)

        if (i + 1) % 100 == 0:
            print(f"v181 multihop target {i + 1}/{len(support)}", flush=True)

    diagnostics = {
        "track_cells": track_cells,
        "track_dirs": track_dirs,
        "track_weights": track_weights,
        "track_conf": track_conf,
        "stable_track_count": stable_track_count,
        "stable_energy_fraction": stable_energy_fraction,
        "mean_persistence": mean_persistence,
        "mean_route_evidence": mean_route_evidence,
        "mean_shift_abs": mean_shift_abs,
        "prior_cos": prior_cos,
        "continue_gate": continue_gate_arr,
        "rollback_gate": rollback_gate_arr,
        "continue_alpha": continue_alpha,
        "rollback_alpha": rollback_alpha,
        "net_alpha": (continue_alpha - rollback_alpha).astype(np.float32),
        "target_support_l1": np.sum(np.abs(target - support), axis=(1, 2, 3)).astype(np.float32),
        "target_support_cos": row_cos(target, support).astype(np.float32),
        "rollback_to_v179_l1": np.sum(np.abs(target - support_v179), axis=(1, 2, 3)).astype(np.float32),
    }
    return track.normalize_support(target), diagnostics


def generate() -> dict[str, object]:
    configure_track_module()
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    v173_action = np.load(V173_ACTIONS)["action"].astype(np.float32)
    v180_data = np.load(V180_ACTIONS)
    v180_continue_alpha = v180_data["continue_alpha"].astype(np.float32)
    v180_rollback_alpha = v180_data["rollback_alpha"].astype(np.float32)
    v180_support_l1 = v180_data["support_l1_actual"].astype(np.float32)
    v180_gain_max = v180_data["gain_max"].astype(np.float32)
    support_v179 = track.normalize_support(np.load(V179_SUPPORT).astype(np.float32))

    support, peaks, energy, vectors = track.extract_v173_support_and_peaks(pos_test)
    direct_idx, direct_w, extended_idx, extended_w, twohop_flag, direct_mass, twohop_mass = multihop_neighbor_graph(
        pos_test,
        blocked,
        regimes,
    )
    target, track_diag = build_multihop_target(
        support,
        support_v179,
        peaks,
        energy,
        vectors,
        extended_idx,
        extended_w,
        twohop_flag,
        regimes,
        v173_action,
        v180_continue_alpha,
        v180_rollback_alpha,
        v180_support_l1,
        v180_gain_max,
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
        direct_neighbor_idx=direct_idx,
        direct_neighbor_weights=direct_w,
        extended_neighbor_idx=extended_idx,
        extended_neighbor_weights=extended_w,
        twohop_flag=twohop_flag,
        direct_mass=direct_mass,
        twohop_mass=twohop_mass,
        v173_action=v173_action,
        v180_continue_alpha=v180_continue_alpha,
        v180_rollback_alpha=v180_rollback_alpha,
        v180_support_l1=v180_support_l1,
        v180_gain_max=v180_gain_max,
        **track_diag,
        **apply_diag,
    )

    comparison_to_v180 = track.compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v179 = track.compare_output_to(OUTPUT, V179_BASE, regimes)
    comparison_to_v178 = track.compare_output_to(OUTPUT, V178_BASE, regimes)
    comparison_to_v173 = track.compare_output_to(OUTPUT, V173_BASE, regimes)
    if comparison_to_v180["shape"] != [500, 256, 4, 192] or comparison_to_v180["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v180['shape']} {comparison_to_v180['dtype']}")
    if not comparison_to_v180["finite"]:
        raise ValueError("v181 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v180 confirmed test-test component tracks still help. v181 extends "
            "the evidence from one-hop local neighbors to a two-hop multi-hop "
            "track field, so only components that remain coherent along the "
            "test graph get another support move."
        ),
        "new_information_used": {
            "current_best": "v180 online 0.504248",
            "fourth_pass_base": PATHS.relative(BASE),
            "validated_direction": "v178->v179->v180 online-positive component-track progression",
            "new_signal": "two-hop test-test route evidence for AD component persistence",
            "guardrails": "clear/light untouched, per-point total power preserved, no train-truth peak prior",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "rollback_support": PATHS.relative(V179_SUPPORT),
            "peak_count": PEAK_COUNT,
            "track_peaks_per_point": TRACK_PEAKS,
            "peak_candidates": PEAK_CANDIDATES,
            "direct_neighbor_k": DIRECT_K,
            "extended_neighbor_k": EXTENDED_K,
            "neighbor_metric": "direct XY + 0.18*blocked_delta + 4.0*regime_mismatch, plus weighted two-hop closure",
            "peak_energy_floor": 4.5e-5,
            "matched_direct_min": 4,
            "matched_total_min": 9,
            "persistence_min": 0.44,
            "prior_floor": 0.20,
            "continue_alpha_scale": {"clear": 0.0, "light": 0.0, "medium": 0.38, "heavy": 0.92},
            "rollback_alpha_scale": {"clear": 0.0, "light": 0.0, "medium": 0.012, "heavy": 0.030},
            "max_combined_alpha": 0.66,
            "power_policy": "preserve v180 per-point total power",
        },
        "diagnostics": {
            "overall": {
                "v173_action": summarize(v173_action),
                "v180_continue_alpha": summarize(v180_continue_alpha),
                "v180_rollback_alpha": summarize(v180_rollback_alpha),
                "v180_support_l1": summarize(v180_support_l1),
                "v180_gain_max": summarize(v180_gain_max),
                "direct_mass": summarize(direct_mass),
                "twohop_mass": summarize(twohop_mass),
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
                    "v180_continue_alpha": v180_continue_alpha,
                    "v180_rollback_alpha": v180_rollback_alpha,
                    "v180_support_l1": v180_support_l1,
                    "track_conf": track_diag["track_conf"],
                    "stable_track_count": track_diag["stable_track_count"],
                    "stable_energy_fraction": track_diag["stable_energy_fraction"],
                    "mean_persistence": track_diag["mean_persistence"],
                    "mean_route_evidence": track_diag["mean_route_evidence"],
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
        "comparison_to_v180": comparison_to_v180,
        "comparison_to_v179": comparison_to_v179,
        "comparison_to_v178": comparison_to_v178,
        "comparison_to_v173": comparison_to_v173,
        "preserved": {
            "base_complex": "v180",
            "point_total_power": "v180",
            "global_phase": "not learned; not globally regressed",
            "clear_light": "no support action",
            "randomness": "none",
        },
        "changed": {
            "tracked_support": "medium/heavy components with one-hop and two-hop route evidence get another support movement",
            "saturation_control": "low-route or over-wide v180 points softly blend back toward v179 collapsed support",
            "cell_internal_direction": "tracked cells receive weak phase-aligned 2x4 direction smoothing",
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
            (__file__, "generate_v181_multihop_track_field.py"),
            (PATHS.code / "generate_v180_track_saturation_bifurcation.py", "generate_v180_track_saturation_bifurcation.py"),
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

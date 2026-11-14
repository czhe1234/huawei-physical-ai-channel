#!/usr/bin/env python3
"""Generate v179: confidence-stratified second-pass component-track overdrive.

v178 validated that test-test component tracks are useful.  v179 keeps that
new information source, but runs a second pass from the v178 output itself:

* re-extract top AD peaks from v178, not from train truth;
* rebuild local component tracks among nearby test receivers;
* move support harder only when track confidence, persistence, and v178's
  previous action agree;
* keep clear/light untouched and preserve v178 per-point total power.
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


VERSION = "v179_track_confidence_overdrive"
OUT_DIR = PATHS.cache_dir(179)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(179)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v179_Round1_Test_Channel.npy"
SUPPORT_OUT = OUT_DIR / "v179_v178_collapsed_support.npy"
PEAKS_OUT = OUT_DIR / "v179_v178_top_peaks.npz"
TARGET_OUT = OUT_DIR / "v179_track_confidence_target_support.npy"
ACTION_OUT = OUT_DIR / "v179_track_confidence_overdrive_actions.npz"
REPORT = REPORT_DIR / "v179_track_confidence_overdrive_generate.json"
ZIP_OUT = PATHS.round1_versions / "v179_track_confidence_overdrive_candidate_pending.zip"

BASE = PATHS.cache_dir(178) / "v178_Round1_Test_Channel.npy"
V173_BASE = PATHS.cache_dir(173) / "v173_Round1_Test_Channel.npy"
V178_ACTIONS = PATHS.cache_dir(178) / "v178_test_track_complex_consensus_actions.npz"
V173_ACTIONS = PATHS.cache_dir(173) / "v173_spatial_coherence_tail_denoising_actions.npz"

FLOOR = 1e-12
PEAK_COUNT = 56
TRACK_PEAKS = 24
MAX_TRACKS = 24
PEAK_CANDIDATES = 2048


def configure_track_module() -> None:
    """Point the v178 utilities at v179 paths and v178 as the base output."""
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


def confidence_gate(track_conf: float, mean_shift_abs: float) -> float:
    conf_gate = np.clip((track_conf - 0.56) / 0.31, 0.0, 1.0)
    shift_gate = np.clip((1.12 - mean_shift_abs) / 0.82, 0.22, 1.0)
    return float(conf_gate * (0.60 + 0.40 * shift_gate))


def build_track_target(
    support: np.ndarray,
    peaks: np.ndarray,
    energy: np.ndarray,
    vectors: np.ndarray,
    neighbor_idx: np.ndarray,
    neighbor_weight: np.ndarray,
    regimes: np.ndarray,
    v173_action: np.ndarray,
    v178_alpha: np.ndarray,
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
    overdrive_gate = np.zeros(len(support), dtype=np.float32)
    alpha = np.zeros(len(support), dtype=np.float32)
    prior_cos = np.ones(len(support), dtype=np.float32)

    for i in range(len(support)):
        field = (0.26 * support[i]).astype(np.float32)
        total_top_energy = float(np.sum(energy[i, :16])) + FLOOR
        stable_energy = 0.0
        persistence_acc = 0.0
        shift_acc = 0.0
        track_slot = 0

        for pidx in range(TRACK_PEAKS):
            h0, v0, d0 = peaks[i, pidx]
            if h0 < 0:
                break
            e0 = float(energy[i, pidx])
            if e0 < 7.5e-5:
                continue
            current_dir = track.unit_vec(vectors[i, pidx])

            weighted_score = 0.0
            weighted_presence = 0.0
            dh_sum = 0.0
            dv_sum = 0.0
            dd_sum = 0.0
            neighbor_energy_sum = 0.0
            dir_sum = (0.60 * current_dir).astype(np.complex64)
            dir_weight = 0.60
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
            if matched_neighbors < 6 or persistence < 0.39 or weighted_score <= 0:
                continue

            mean_dh = float(np.clip(dh_sum / weighted_score, -1.7, 1.7))
            mean_dv = float(np.clip(dv_sum / weighted_score, -1.1, 1.1))
            mean_dd = float(np.clip(dd_sum / weighted_score, -3.8, 3.8))
            h_track = (int(h0) + int(round(0.64 * mean_dh))) % MH
            v_track = (int(v0) + int(round(0.64 * mean_dv))) % MV
            d_track = int(np.clip(int(d0) + int(round(0.64 * mean_dd)), 0, S_SC - 1))

            neighbor_mean_energy = neighbor_energy_sum / weighted_score
            amp_ratio = float(np.clip(math.sqrt(max(neighbor_mean_energy, FLOOR) / max(e0, FLOOR)), 0.76, 1.30))
            mass = e0 * (0.70 + 0.88 * persistence) * amp_ratio
            track.add_peak_kernel(field, h_track, v_track, d_track, mass, h_sigma=0.90, v_sigma=0.74, d_sigma=2.25)

            stable_energy += e0 * persistence
            persistence_acc += persistence
            shift_acc += abs(mean_dh) + 0.65 * abs(mean_dv) + 0.24 * abs(mean_dd)
            stable_track_count[i] += 1.0

            if track_slot < MAX_TRACKS:
                target_dir = track.unit_vec(dir_sum / np.complex64(max(dir_weight, 1e-12)))
                track_cells[i, track_slot] = (h_track, v_track, d_track)
                track_dirs[i, track_slot] = target_dir
                if str(regimes[i]) == "medium":
                    direction_scale = 0.45
                elif str(regimes[i]) == "heavy":
                    direction_scale = 0.80
                else:
                    direction_scale = 0.00
                track_weights[i, track_slot] = np.float32(
                    direction_scale * min(0.15, 0.020 + 0.155 * persistence)
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
            0.50 * stable_energy_fraction[i]
            + 0.42 * mean_persistence[i]
            + 0.08 * np.clip(stable_track_count[i] / 10.0, 0.0, 1.0),
            0.0,
            1.0,
        )
        track_conf[i] = np.float32(conf)
        gate = confidence_gate(float(conf), float(mean_shift_abs[i]))
        overdrive_gate[i] = np.float32(gate)

        regime = str(regimes[i])
        if regime == "medium":
            scale = 0.18
            previous_norm = float(np.clip(v178_alpha[i] / 0.18, 0.0, 1.0))
        elif regime == "heavy":
            scale = 0.42
            previous_norm = float(np.clip(v178_alpha[i] / 0.32, 0.0, 1.0))
        else:
            scale = 0.0
            previous_norm = 0.0

        previous_gate = 0.28 + 0.72 * math.sqrt(previous_norm)
        spatial_gate = 0.70 + 0.30 * math.sqrt(float(np.clip(v173_action[i], 0.0, 1.0)))
        point_alpha = scale * gate * previous_gate * spatial_gate
        alpha[i] = np.float32(np.clip(point_alpha, 0.0, scale))
        target[i] = track.normalize_support(
            ((1.0 - alpha[i]) * support[i] + alpha[i] * prior)[None]
        )[0]

        if (i + 1) % 100 == 0:
            print(f"v179 track target {i + 1}/{len(support)}", flush=True)

    diagnostics = {
        "track_cells": track_cells,
        "track_dirs": track_dirs,
        "track_weights": track_weights,
        "track_conf": track_conf,
        "stable_track_count": stable_track_count,
        "stable_energy_fraction": stable_energy_fraction,
        "mean_persistence": mean_persistence,
        "mean_shift_abs": mean_shift_abs,
        "overdrive_gate": overdrive_gate,
        "alpha": alpha,
        "prior_cos": prior_cos,
        "target_support_l1": np.sum(np.abs(target - support), axis=(1, 2, 3)).astype(np.float32),
        "target_support_cos": row_cos(target, support).astype(np.float32),
    }
    return track.normalize_support(target), diagnostics


def generate() -> dict[str, object]:
    configure_track_module()
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    blocked = np.load(PATHS.shared_features / "height_geometry_cache.npz")["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked)
    v173_action = np.load(V173_ACTIONS)["action"].astype(np.float32)
    v178_data = np.load(V178_ACTIONS)
    v178_alpha = v178_data["alpha"].astype(np.float32)

    support, peaks, energy, vectors = track.extract_v173_support_and_peaks(pos_test)
    idx, weights = track.neighbor_graph(pos_test, blocked, regimes)
    target, track_diag = build_track_target(
        support,
        peaks,
        energy,
        vectors,
        idx,
        weights,
        regimes,
        v173_action,
        v178_alpha,
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
        **track_diag,
        **apply_diag,
    )

    comparison_to_v178 = track.compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v173 = track.compare_output_to(OUTPUT, V173_BASE, regimes)
    comparison_to_v171 = track.compare_output_to(OUTPUT, PATHS.cache_dir(171) / "v171_Round1_Test_Channel.npy", regimes)
    if comparison_to_v178["shape"] != [500, 256, 4, 192] or comparison_to_v178["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v178['shape']} {comparison_to_v178['dtype']}")
    if not comparison_to_v178["finite"]:
        raise ValueError("v179 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v178 proved test-test component tracks are useful.  v179 uses v178 "
            "as the base and applies a second-pass, confidence-stratified track "
            "overdrive only on medium/heavy high-persistence components."
        ),
        "new_information_used": {
            "current_best": "v178 online 0.503876",
            "second_pass_base": PATHS.relative(BASE),
            "guardrails": "clear/light untouched, per-point total power preserved, no train-truth peak prior",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "peak_count": PEAK_COUNT,
            "track_peaks_per_point": TRACK_PEAKS,
            "peak_candidates": PEAK_CANDIDATES,
            "neighbor_k": 18,
            "neighbor_metric": "XY + 0.18*blocked_delta + 4.0*regime_mismatch",
            "peak_energy_floor": 7.5e-5,
            "matched_neighbors_min": 6,
            "persistence_min": 0.39,
            "prior_floor": 0.26,
            "alpha_scale": {"clear": 0.0, "light": 0.0, "medium": 0.18, "heavy": 0.42},
            "confidence_gate": "clip((track_conf-.56)/.31) * (0.60 + 0.40*shift_gate)",
            "max_direction_weight": 0.15,
            "power_policy": "preserve v178 per-point total power",
        },
        "diagnostics": {
            "overall": {
                "v173_action": summarize(v173_action),
                "v178_alpha": summarize(v178_alpha),
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
                    "track_conf": track_diag["track_conf"],
                    "stable_track_count": track_diag["stable_track_count"],
                    "stable_energy_fraction": track_diag["stable_energy_fraction"],
                    "mean_persistence": track_diag["mean_persistence"],
                    "mean_shift_abs": track_diag["mean_shift_abs"],
                    "overdrive_gate": track_diag["overdrive_gate"],
                    "alpha": track_diag["alpha"],
                    "target_support_l1": track_diag["target_support_l1"],
                    "support_l1_actual": apply_diag["support_l1_actual"],
                    "direction_update_energy": apply_diag["direction_update_energy"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                },
            ),
        },
        "comparison_to_v178": comparison_to_v178,
        "comparison_to_v173": comparison_to_v173,
        "comparison_to_v171": comparison_to_v171,
        "preserved": {
            "base_complex": "v178",
            "point_total_power": "v178",
            "global_phase": "not learned; not globally regressed",
            "clear_light": "no support action",
            "randomness": "none",
        },
        "changed": {
            "tracked_support": "medium/heavy high-confidence tracks receive second-pass support movement",
            "cell_internal_direction": "tracked cells get weaker phase-aligned 2x4 direction smoothing than v178",
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
            (__file__, "generate_v179_track_confidence_overdrive.py"),
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

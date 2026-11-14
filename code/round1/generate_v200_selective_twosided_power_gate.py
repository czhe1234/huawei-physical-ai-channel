#!/usr/bin/env python3
"""Generate v200: selective two-sided power gate on top of v196.

v199 showed that v196 did not really fix per-point total power: it inherited
v189 power almost exactly.  It also showed that train/model/test evidence is
not reliable enough for a global replacement, because many test points are
conflict/no-action.

This candidate therefore uses only the high-confidence v199 power labels:

* power_up points are scaled up;
* power_down points are scaled down;
* conflict and no_action points keep v196 unchanged.

No support projection, phase transport, or delay-mouth expansion is attempted.
The candidate is an isolated test of whether v199's pointwise power evidence
can release part of the v196 ceiling without repeating v198's heavy-wide
downscale.
"""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

import generate_v178_test_track_complex_consensus as track
from generate_v87_easy_field_projection import summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from project_paths import PATHS


VERSION = "v200_selective_twosided_power_gate"

OUT_DIR = PATHS.cache_dir(200)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(200)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v200_Round1_Test_Channel.npy"
ACTION_OUT = OUT_DIR / "v200_selective_twosided_power_gate_actions.npz"
REPORT = REPORT_DIR / "v200_selective_twosided_power_gate_generate.json"
ZIP_OUT = PATHS.round1_versions / "v200_selective_twosided_power_gate_candidate_pending.zip"

BASE = PATHS.cache_dir(196) / "v196_Round1_Test_Channel.npy"
V189_BASE = PATHS.cache_dir(189) / "v189_Round1_Test_Channel.npy"
V198_BASE = PATHS.cache_dir(198) / "v198_Round1_Test_Channel.npy"
V199_DIAG = PATHS.cache_dir(199) / "v199_test_power_diagnostics.npz"
V196_ACTIONS = PATHS.cache_dir(196) / "v196_train_guided_delay_actions.npz"

BATCH = 20

UP_CAP = {
    "clear": 0.075,
    "light": 0.040,
    "medium": 0.120,
    "heavy": 0.070,
}
DOWN_CAP = {
    "clear": -0.045,
    "light": -0.035,
    "medium": -0.100,
    "heavy": -0.125,
}
UP_SCALE = {
    "clear": 1.35,
    "light": 0.75,
    "medium": 1.25,
    "heavy": 0.65,
}
DOWN_SCALE = {
    "clear": 0.80,
    "light": 0.65,
    "medium": 0.95,
    "heavy": 1.15,
}
CONF_MIN_UP = {
    "clear": 0.76,
    "light": 0.58,
    "medium": 0.68,
    "heavy": 0.82,
}
CONF_MIN_DOWN = {
    "clear": 0.76,
    "light": 0.58,
    "medium": 0.66,
    "heavy": 0.70,
}


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def build_log_delta() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    require_file(V199_DIAG)
    require_file(V196_ACTIONS)
    diag = np.load(V199_DIAG)
    v196_actions = np.load(V196_ACTIONS)

    regimes = diag["regimes"].astype(str)
    labels = diag["power_label"].astype(str)
    suggested = diag["suggested_log_delta"].astype(np.float32)
    confidence = diag["power_confidence"].astype(np.float32)
    d_knn = diag["delta_knn"].astype(np.float32)
    d_model = diag["delta_model"].astype(np.float32)
    d_consensus = diag["delta_consensus"].astype(np.float32)
    test_agreement = diag["test_neighbor_agreement"].astype(np.float32)
    v196_agreement = v196_actions["test_neighbor_agreement"].astype(np.float32)
    v196_gain_max = v196_actions["gain_max"].astype(np.float32)

    final = np.zeros(len(regimes), dtype=np.float32)
    active = np.zeros(len(regimes), dtype=bool)
    strict_gate = np.zeros(len(regimes), dtype=bool)
    strong_agree = np.zeros(len(regimes), dtype=bool)

    for regime in ["clear", "light", "medium", "heavy"]:
        rmask = regimes == regime
        up = (
            rmask
            & (labels == "power_up")
            & (confidence >= CONF_MIN_UP[regime])
            & (d_knn > 0.06)
            & (d_model > 0.06)
            & (d_consensus > -0.85)
        )
        down = (
            rmask
            & (labels == "power_down")
            & (confidence >= CONF_MIN_DOWN[regime])
            & (d_knn < -0.06)
            & (d_model < -0.06)
            & (d_consensus < 0.85)
        )

        conf_gain = np.clip(0.75 + 0.40 * (confidence - 0.62) / 0.33, 0.75, 1.15)
        stable = np.clip(0.50 * test_agreement + 0.50 * v196_agreement, 0.0, 1.0)
        support_stress = np.clip((v196_gain_max - 4.50) / 2.20, 0.0, 1.0)
        stability_gain = np.clip(0.84 + 0.30 * stable - 0.14 * support_stress, 0.75, 1.10)

        up_value = suggested * UP_SCALE[regime] * conf_gain * stability_gain
        down_value = suggested * DOWN_SCALE[regime] * conf_gain * stability_gain
        up_value = np.clip(up_value, 0.0, UP_CAP[regime]).astype(np.float32)
        down_value = np.clip(down_value, DOWN_CAP[regime], 0.0).astype(np.float32)

        final[up] = up_value[up]
        final[down] = down_value[down]
        strict_gate |= up | down
        active |= np.abs(final) > 1e-7

    strong_agree = active & (confidence >= 0.80) & (np.abs(d_knn) >= 1.0) & (np.abs(d_model) >= 1.0)
    strong_agree &= np.sign(d_knn) == np.sign(d_model)
    final[strong_agree] *= 1.08
    for regime in ["clear", "light", "medium", "heavy"]:
        rmask = regimes == regime
        final[rmask & (final > 0)] = np.minimum(final[rmask & (final > 0)], UP_CAP[regime])
        final[rmask & (final < 0)] = np.maximum(final[rmask & (final < 0)], DOWN_CAP[regime])

    power_factor = np.exp(final).astype(np.float32)
    amp_factor = np.exp(0.5 * final).astype(np.float32)

    return final.astype(np.float32), {
        "regimes": regimes.astype("U8"),
        "power_label": labels.astype("U16"),
        "suggested_log_delta": suggested,
        "final_log_delta": final.astype(np.float32),
        "power_factor": power_factor,
        "amp_factor": amp_factor,
        "active": active,
        "strict_gate": strict_gate,
        "strong_agree": strong_agree,
        "power_confidence": confidence,
        "delta_knn": d_knn,
        "delta_model": d_model,
        "delta_consensus": d_consensus,
        "test_neighbor_agreement": test_agreement,
        "v196_test_neighbor_agreement": v196_agreement,
        "v196_gain_max": v196_gain_max,
    }


def apply_power_scale(amp_factor: np.ndarray) -> dict[str, np.ndarray]:
    require_file(BASE)
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    actual_power_factor = np.empty(len(source), dtype=np.float32)

    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        h1 = h0 * amp_factor[start:end, None, None, None].astype(np.float32)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()
        p0 = np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64)
        p1 = np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64)
        actual_power_factor[start:end] = (p1 / np.maximum(p0, 1e-30)).astype(np.float32)
        print(f"v200 power scale {end}/{len(source)}", flush=True)

    return {"actual_power_factor": actual_power_factor}


def label_action_summary(labels: np.ndarray, regimes: np.ndarray, active: np.ndarray) -> dict[str, object]:
    out: dict[str, object] = {}
    for label in ["power_up", "power_down", "conflict", "no_action"]:
        lmask = labels.astype(str) == label
        item: dict[str, object] = {
            "label_count": int(np.sum(lmask)),
            "active_count": int(np.sum(lmask & active)),
        }
        for regime in ["clear", "light", "medium", "heavy"]:
            rmask = regimes.astype(str) == regime
            item[f"{regime}_label_count"] = int(np.sum(lmask & rmask))
            item[f"{regime}_active_count"] = int(np.sum(lmask & rmask & active))
        out[label] = item
    return out


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for src in [
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
    for start in range(0, len(output), BATCH):
        block = np.asarray(output[start : start + BATCH], dtype=np.complex64)
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
    log_delta, action_diag = build_log_delta()
    apply_diag = apply_power_scale(action_diag["amp_factor"])
    regimes = action_diag["regimes"].astype(str)
    labels = action_diag["power_label"].astype(str)
    active = action_diag["active"].astype(bool)

    np.savez_compressed(
        ACTION_OUT,
        **action_diag,
        **apply_diag,
    )

    comparison_to_v196 = track.compare_output_to(OUTPUT, BASE, regimes)
    comparison_to_v189 = track.compare_output_to(OUTPUT, V189_BASE, regimes)
    comparison_to_v198 = track.compare_output_to(OUTPUT, V198_BASE, regimes)
    if comparison_to_v196["shape"] != [500, 256, 4, 192] or comparison_to_v196["dtype"] != "complex64":
        raise ValueError(f"invalid output layout: {comparison_to_v196['shape']} {comparison_to_v196['dtype']}")
    if not comparison_to_v196["finite"]:
        raise ValueError("v200 output contains non-finite values")

    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v196 improved delay/PDP shape but inherited v189 total power almost exactly. "
            "v199 found high-confidence power-up and power-down test points. v200 tests "
            "whether a selective two-sided scalar correction can release that ceiling "
            "while leaving conflict/no-action points, support and phase untouched."
        ),
        "new_information_used": {
            "v199_power_labels": "power_up/power_down/conflict/no_action derived from train-kNN log power, geometry model log power and test-test v196 power consensus",
            "v196_delay_reliability": "v196 neighbor agreement and gain_max only modulate action confidence; they do not create a new support target",
            "v198_negative_boundary": "heavy-wide downscale is avoided; heavy power-down is selective and bounded below v198's average downscale",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "v199_diagnostics": PATHS.relative(V199_DIAG),
            "v196_actions": PATHS.relative(V196_ACTIONS),
            "up_cap_log": UP_CAP,
            "down_cap_log": DOWN_CAP,
            "up_scale": UP_SCALE,
            "down_scale": DOWN_SCALE,
            "confidence_min_up": CONF_MIN_UP,
            "confidence_min_down": CONF_MIN_DOWN,
            "policy": "scale full complex H by a per-point scalar only; preserve normalized support and phase",
        },
        "action_counts": {
            "total_active": int(np.sum(active)),
            "label_counts": dict(Counter(labels.tolist())),
            "active_by_label_and_regime": label_action_summary(labels, regimes, active),
        },
        "diagnostics": {
            "overall": {
                "final_log_delta": summarize(log_delta),
                "power_factor": summarize(action_diag["power_factor"]),
                "amp_factor": summarize(action_diag["amp_factor"]),
                "actual_power_factor": summarize(apply_diag["actual_power_factor"]),
                "suggested_log_delta": summarize(action_diag["suggested_log_delta"]),
                "power_confidence_active": summarize(action_diag["power_confidence"][active]),
                "delta_knn_active": summarize(action_diag["delta_knn"][active]),
                "delta_model_active": summarize(action_diag["delta_model"][active]),
                "delta_consensus_active": summarize(action_diag["delta_consensus"][active]),
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "final_log_delta": action_diag["final_log_delta"],
                    "power_factor": action_diag["power_factor"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                    "suggested_log_delta": action_diag["suggested_log_delta"],
                    "active": active.astype(np.float32),
                    "power_confidence": action_diag["power_confidence"],
                },
            ),
        },
        "comparison_to_v196": comparison_to_v196,
        "comparison_to_v189": comparison_to_v189,
        "comparison_to_v198": comparison_to_v198,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
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

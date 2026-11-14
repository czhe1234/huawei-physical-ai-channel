#!/usr/bin/env python3
"""Generate v202: conflict-unlocked total-power continuation on top of v200.

v200 proved that the fixed total-power constraint limited v196.  v201 then
showed that reusing the v200 selector for support-only D/U-D/H-D movement did
not beat v200.  This candidate therefore goes back to the scalar total-power
field and makes the next aggressive step there.

The new action has two parts:

* continue v200's already validated up/down power actions when all evidence is
  still aligned;
* unlock a subset of v199 conflict points where train-kNN and geometry models
  strongly agree, while the opposing test-test consensus is weak relative to
  the local power variance.

No support projection, phase transport, or delay-mouth expansion is attempted.
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


VERSION = "v202_conflict_unlocked_power_field"

OUT_DIR = PATHS.cache_dir(202)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(202)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v202_Round1_Test_Channel.npy"
ACTION_OUT = OUT_DIR / "v202_conflict_unlocked_power_field_actions.npz"
REPORT = REPORT_DIR / "v202_conflict_unlocked_power_field_generate.json"
ZIP_OUT = PATHS.round1_versions / "v202_conflict_unlocked_power_field_candidate_pending.zip"

BASE = PATHS.cache_dir(196) / "v196_Round1_Test_Channel.npy"
V189_BASE = PATHS.cache_dir(189) / "v189_Round1_Test_Channel.npy"
V198_BASE = PATHS.cache_dir(198) / "v198_Round1_Test_Channel.npy"
V200_BASE = PATHS.cache_dir(200) / "v200_Round1_Test_Channel.npy"
V199_DIAG = PATHS.cache_dir(199) / "v199_test_power_diagnostics.npz"
V196_ACTIONS = PATHS.cache_dir(196) / "v196_train_guided_delay_actions.npz"
V200_ACTIONS = PATHS.cache_dir(200) / "v200_selective_twosided_power_gate_actions.npz"

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
FINAL_UP_CAP = {
    "clear": 0.115,
    "light": 0.055,
    "medium": 0.155,
    "heavy": 0.095,
}
FINAL_DOWN_CAP = {
    "clear": -0.065,
    "light": -0.045,
    "medium": -0.135,
    "heavy": -0.165,
}

CONT_UP_CAP = {
    "clear": 0.020,
    "light": 0.010,
    "medium": 0.024,
    "heavy": 0.012,
}
CONT_DOWN_CAP = {
    "clear": -0.012,
    "light": -0.008,
    "medium": -0.020,
    "heavy": -0.030,
}

CONFLICT_UP_CAP = {
    "clear": 0.070,
    "light": 0.025,
    "medium": 0.035,
    "heavy": 0.025,
}
CONFLICT_DOWN_CAP = {
    "clear": -0.025,
    "light": -0.018,
    "medium": -0.050,
    "heavy": -0.080,
}

CONFLICT_CONF_MIN = 0.70
CONFLICT_LOCAL_RATIO_MAX = 0.90


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def build_log_delta() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    require_file(V199_DIAG)
    require_file(V196_ACTIONS)
    require_file(V200_ACTIONS)
    diag = np.load(V199_DIAG)
    v196_actions = np.load(V196_ACTIONS)
    v200_actions = np.load(V200_ACTIONS)

    regimes = diag["regimes"].astype(str)
    labels = diag["power_label"].astype(str)
    suggested = diag["suggested_log_delta"].astype(np.float32)
    confidence = diag["power_confidence"].astype(np.float32)
    d_knn = diag["delta_knn"].astype(np.float32)
    d_model = diag["delta_model"].astype(np.float32)
    d_consensus = diag["delta_consensus"].astype(np.float32)
    model_spread = diag["test_model_spread"].astype(np.float32)
    local_std = diag["test_local_log_std"].astype(np.float32)
    test_agreement = diag["test_neighbor_agreement"].astype(np.float32)
    v196_agreement = v196_actions["test_neighbor_agreement"].astype(np.float32)
    v196_gain_max = v196_actions["gain_max"].astype(np.float32)
    v200_delta = v200_actions["final_log_delta"].astype(np.float32)

    raw_train_model = (0.5 * (d_knn + d_model)).astype(np.float32)
    local_ratio = (np.abs(d_consensus) / np.maximum(local_std, 1e-6)).astype(np.float32)

    continuation = (
        (np.abs(v200_delta) > 1e-7)
        & (np.sign(d_knn) == np.sign(d_model))
        & (np.sign(d_knn) == np.sign(d_consensus))
        & (np.abs(d_knn) > 0.25)
        & (np.abs(d_model) > 0.25)
        & (np.abs(d_consensus) > 0.16)
        & (confidence > 0.72)
    )
    conflict_unlock = (
        (labels == "conflict")
        & (np.sign(d_knn) == np.sign(d_model))
        & (np.abs(d_knn) > 0.45)
        & (np.abs(d_model) > 0.45)
        & (confidence >= CONFLICT_CONF_MIN)
        & (local_ratio < CONFLICT_LOCAL_RATIO_MAX)
    )

    extra = np.zeros(len(regimes), dtype=np.float32)

    for regime in ["clear", "light", "medium", "heavy"]:
        rmask = regimes == regime

        cont = rmask & continuation
        cont_mag = np.clip(
            0.007
            + 0.012 * (confidence - 0.72) / 0.25
            + 0.005 * np.minimum(np.abs(d_consensus), 1.5) / 1.5,
            0.0,
            0.026,
        )
        cont_value = (np.sign(v200_delta) * cont_mag).astype(np.float32)
        cont_up = cont & (v200_delta > 0)
        cont_down = cont & (v200_delta < 0)
        extra[cont_up] += np.minimum(cont_value[cont_up], CONT_UP_CAP[regime]).astype(np.float32)
        extra[cont_down] += np.maximum(cont_value[cont_down], CONT_DOWN_CAP[regime]).astype(np.float32)

        unlock = rmask & conflict_unlock
        unlock_quality = (
            np.clip((confidence - (CONFLICT_CONF_MIN - 0.02)) / 0.22, 0.0, 1.0)
            * np.clip(1.0 - local_ratio / CONFLICT_LOCAL_RATIO_MAX, 0.0, 1.0)
            * np.clip(1.0 - model_spread / 2.4, 0.35, 1.0)
        )
        unlock_mag = np.clip(
            0.020
            + 0.034 * unlock_quality
            + 0.016 * np.minimum(np.abs(raw_train_model), 2.5) / 2.5,
            0.0,
            0.085,
        )
        unlock_value = (np.sign(raw_train_model) * unlock_mag).astype(np.float32)
        unlock_up = unlock & (raw_train_model > 0)
        unlock_down = unlock & (raw_train_model < 0)
        extra[unlock_up] += np.minimum(unlock_value[unlock_up], CONFLICT_UP_CAP[regime]).astype(np.float32)
        extra[unlock_down] += np.maximum(unlock_value[unlock_down], CONFLICT_DOWN_CAP[regime]).astype(np.float32)

    final = (v200_delta + extra).astype(np.float32)
    for regime in ["clear", "light", "medium", "heavy"]:
        rmask = regimes == regime
        final[rmask] = np.clip(final[rmask], FINAL_DOWN_CAP[regime], FINAL_UP_CAP[regime])
    extra = (final - v200_delta).astype(np.float32)

    power_factor = np.exp(final).astype(np.float32)
    amp_factor = np.exp(0.5 * final).astype(np.float32)
    active = np.abs(final) > 1e-7
    extra_active = np.abs(extra) > 1e-7

    return final.astype(np.float32), {
        "regimes": regimes.astype("U8"),
        "power_label": labels.astype("U16"),
        "suggested_log_delta": suggested,
        "v200_log_delta": v200_delta,
        "extra_log_delta": extra,
        "final_log_delta": final.astype(np.float32),
        "power_factor": power_factor,
        "amp_factor": amp_factor,
        "active": active,
        "extra_active": extra_active,
        "continuation": continuation,
        "conflict_unlock": conflict_unlock,
        "raw_train_model_delta": raw_train_model,
        "local_consensus_ratio": local_ratio,
        "model_spread": model_spread,
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
        print(f"v202 power scale {end}/{len(source)}", flush=True)

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
            PATHS.code / "generate_v202_conflict_unlocked_power_field.py",
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
    comparison_to_v200 = track.compare_output_to(OUTPUT, V200_BASE, regimes)
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
            "v200 online improvement proves total power is a real bottleneck. v201 failed "
            "when the same selector was used for support-only delay movement. v202 returns "
            "to scalar total power: it continues v200's aligned actions and unlocks the "
            "subset of v199 conflict points where train/model power evidence is strong "
            "and the opposing test consensus is weak relative to local variance."
        ),
        "new_information_used": {
            "v200_online_feedback": "selective two-sided scalar total-power correction is validated online",
            "v201_negative_feedback": "using the same power selector for support-only D/U-D/H-D correction is not the next breakthrough",
            "v199_conflict_decomposition": "conflict points are split by train/model agreement and local-consensus strength, rather than all being frozen",
            "v198_negative_boundary": "heavy-wide downscale is still avoided; v202 heavy mean power remains far above v198",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "v200_base": PATHS.relative(V200_BASE),
            "v199_diagnostics": PATHS.relative(V199_DIAG),
            "v196_actions": PATHS.relative(V196_ACTIONS),
            "v200_actions": PATHS.relative(V200_ACTIONS),
            "final_up_cap_log": FINAL_UP_CAP,
            "final_down_cap_log": FINAL_DOWN_CAP,
            "continuation_up_cap_log": CONT_UP_CAP,
            "continuation_down_cap_log": CONT_DOWN_CAP,
            "conflict_up_cap_log": CONFLICT_UP_CAP,
            "conflict_down_cap_log": CONFLICT_DOWN_CAP,
            "conflict_conf_min": CONFLICT_CONF_MIN,
            "conflict_local_ratio_max": CONFLICT_LOCAL_RATIO_MAX,
            "policy": "scale full complex H by a per-point scalar only; preserve normalized support and phase",
        },
        "action_counts": {
            "total_active": int(np.sum(active)),
            "v200_active": int(np.sum(np.abs(action_diag["v200_log_delta"]) > 1e-7)),
            "extra_active": int(np.sum(action_diag["extra_active"])),
            "continuation_active": int(np.sum(action_diag["continuation"] & action_diag["extra_active"])),
            "conflict_unlock_active": int(np.sum(action_diag["conflict_unlock"] & action_diag["extra_active"])),
            "conflict_unlock_up": int(
                np.sum(action_diag["conflict_unlock"] & action_diag["extra_active"] & (action_diag["raw_train_model_delta"] > 0))
            ),
            "conflict_unlock_down": int(
                np.sum(action_diag["conflict_unlock"] & action_diag["extra_active"] & (action_diag["raw_train_model_delta"] < 0))
            ),
            "label_counts": dict(Counter(labels.tolist())),
            "active_by_label_and_regime": label_action_summary(labels, regimes, active),
        },
        "diagnostics": {
            "overall": {
                "final_log_delta": summarize(log_delta),
                "v200_log_delta": summarize(action_diag["v200_log_delta"]),
                "extra_log_delta": summarize(action_diag["extra_log_delta"]),
                "power_factor": summarize(action_diag["power_factor"]),
                "amp_factor": summarize(action_diag["amp_factor"]),
                "actual_power_factor": summarize(apply_diag["actual_power_factor"]),
                "suggested_log_delta": summarize(action_diag["suggested_log_delta"]),
                "extra_active": summarize(action_diag["extra_active"].astype(np.float32)),
                "continuation": summarize(action_diag["continuation"].astype(np.float32)),
                "conflict_unlock": summarize(action_diag["conflict_unlock"].astype(np.float32)),
                "power_confidence_active": summarize(action_diag["power_confidence"][active]),
                "delta_knn_active": summarize(action_diag["delta_knn"][active]),
                "delta_model_active": summarize(action_diag["delta_model"][active]),
                "delta_consensus_active": summarize(action_diag["delta_consensus"][active]),
            },
            "by_regime": by_regime_summary(
                regimes,
                {
                    "final_log_delta": action_diag["final_log_delta"],
                    "v200_log_delta": action_diag["v200_log_delta"],
                    "extra_log_delta": action_diag["extra_log_delta"],
                    "power_factor": action_diag["power_factor"],
                    "actual_power_factor": apply_diag["actual_power_factor"],
                    "suggested_log_delta": action_diag["suggested_log_delta"],
                    "active": active.astype(np.float32),
                    "extra_active": action_diag["extra_active"].astype(np.float32),
                    "continuation": action_diag["continuation"].astype(np.float32),
                    "conflict_unlock": action_diag["conflict_unlock"].astype(np.float32),
                    "power_confidence": action_diag["power_confidence"],
                },
            ),
        },
        "comparison_to_v196": comparison_to_v196,
        "comparison_to_v200": comparison_to_v200,
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

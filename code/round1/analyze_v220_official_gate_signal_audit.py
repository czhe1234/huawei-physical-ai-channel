#!/usr/bin/env python3
"""Audit whether official-data candidates contain enough local-gate signal.

This is an analysis-only experiment.  It does not generate a submission zip.

The current handoff rule requires any new recommended candidate to reach the
local v212 gate, ``score_traceback_leaked >= 0.645604``.  v219 showed that the
v212 jump mostly lives in the middle-weak support layer.  Before spending more
time on another package, this script checks whether the existing compliant
official-data branches (v213/v215/v216/v218) and simple mixtures among them
carry enough signal to plausibly pass that gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import analyze_v205_sionna_config_calibration as calibration
from analyze_v208_strict_historical_replay import TEST_TRUTH
from project_paths import PATHS


VERSION = "v220_official_gate_signal_audit"
VERSION_NUMBER = 220
BATCH = 8
LOCAL_GATE = 0.645604
TINY = 1e-30

REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v220_official_gate_signal_audit.json"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    paths: tuple[Path, ...]
    weights: tuple[float, ...]
    preserve_power_to_first: bool = False
    compliance_class: str = "official_data_or_ai_backbone"
    note: str = ""


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def candidate_specs() -> list[CandidateSpec]:
    cache = PATHS.artifacts / "version_cache"
    v209 = cache / "v209" / "v209_Round1_Test_Channel.npy"
    v212 = cache / "v212" / "v212_Round1_Test_Channel.npy"
    v213 = cache / "v213" / "v213_Round1_Test_Channel.npy"
    v214 = cache / "v214" / "v214_Round1_Test_Channel.npy"
    v215 = cache / "v215" / "v215_Round1_Test_Channel.npy"
    v216 = cache / "v216" / "v216_Round1_Test_Channel.npy"
    v218 = cache / "v218" / "v218_Round1_Test_Channel.npy"
    return [
        CandidateSpec("v209_baseline", (v209,), (1.0,), note="pre-Sionna AI backbone"),
        CandidateSpec(
            "v212_local_gate_reference",
            (v212,),
            (1.0,),
            compliance_class="sionna_diagnostic_reference",
            note="current online best and local gate reference; not an official-data restart",
        ),
        CandidateSpec("v213_physical_constraint", (v213,), (1.0,)),
        CandidateSpec("v215_pointcloud_neural_field", (v215,), (1.0,)),
        CandidateSpec("v216_oof_residual_net", (v216,), (1.0,)),
        CandidateSpec("v218_official_multispectrum", (v218,), (1.0,)),
        CandidateSpec(
            "v218_half_on_v209_power_preserved",
            (v209, v218),
            (0.5, 0.5),
            preserve_power_to_first=True,
            note="checks whether v218 direction is useful when v209 power is protected",
        ),
        CandidateSpec(
            "v213_v218_half_power_preserved",
            (v213, v218),
            (0.5, 0.5),
            preserve_power_to_first=True,
            note="best small official-data mixture observed in quick probe",
        ),
        CandidateSpec(
            "v209_v213_v218_probe_power_preserved",
            (v209, v213, v218),
            (0.15, 0.55, 0.30),
            preserve_power_to_first=True,
            note="official-data mixture probe; no v212/v217 teacher input",
        ),
        CandidateSpec(
            "v214_distilled_teacher_student",
            (v214,),
            (1.0,),
            compliance_class="teacher_distillation_diagnostic",
            note="uses v212 action trace as teacher; diagnostic only under current rules",
        ),
        CandidateSpec(
            "v214_125_on_v209_power_preserved",
            (v209, v214),
            (-0.25, 1.25),
            preserve_power_to_first=True,
            compliance_class="teacher_distillation_diagnostic",
            note="diagnostic overdrive of v214 teacher-student direction",
        ),
    ]


class Accumulator:
    def __init__(self, name: str) -> None:
        self.name = name
        self.pas: list[np.ndarray] = []
        self.pdp: list[np.ndarray] = []
        self.leaked_sh: list[np.ndarray] = []
        self.ad_support: list[np.ndarray] = []
        self.complex_coherence: list[np.ndarray] = []
        self.pred_power: list[np.ndarray] = []
        self.truth_power: list[np.ndarray] = []
        self.error = 0.0
        self.energy = 0.0
        self.finite = True

    def update(self, prediction: np.ndarray, truth: np.ndarray, truth_views: dict[str, np.ndarray]) -> None:
        self.finite = self.finite and bool(np.isfinite(prediction.real).all() and np.isfinite(prediction.imag).all())
        scores = calibration.spectral_scores(prediction, truth_views)
        pred_power = np.sum(np.abs(prediction) ** 2, axis=(1, 2, 3), dtype=np.float64)
        truth_power = truth_views["energy"]
        true_zero = truth_power <= 1e-25
        pred_zero = pred_power <= 1e-25
        for key in ("pas", "pdp", "leaked_sh", "ad_support", "complex_coherence"):
            value = scores[key].copy()
            if key != "complex_coherence":
                value[true_zero & pred_zero] = 1.0
                value[true_zero & ~pred_zero] = 0.0
            getattr(self, key).append(value)
        self.pred_power.append(pred_power)
        self.truth_power.append(truth_power)
        self.error += float(np.sum(np.abs(prediction - truth) ** 2, dtype=np.float64))
        self.energy += float(np.sum(np.abs(truth) ** 2, dtype=np.float64))

    def finish(self) -> dict[str, float | bool]:
        pas = float(np.mean(np.concatenate(self.pas)))
        pdp = float(np.mean(np.concatenate(self.pdp)))
        leaked = float(np.mean(np.concatenate(self.leaked_sh)))
        ad = float(np.mean(np.concatenate(self.ad_support)))
        coherence = float(np.mean(np.concatenate(self.complex_coherence)))
        nmse = self.error / max(self.energy, TINY)
        nmse_score = 1.0 / (1.0 + nmse)
        pred_power = np.concatenate(self.pred_power)
        truth_power = np.concatenate(self.truth_power)
        power_spearman = float(
            spearmanr(
                np.log(np.maximum(pred_power, TINY)),
                np.log(np.maximum(truth_power, TINY)),
            ).statistic
        )
        return {
            "finite": self.finite,
            "pas": pas,
            "pdp": pdp,
            "leaked_sh": leaked,
            "ad_support": ad,
            "complex_coherence": coherence,
            "global_nmse": float(nmse),
            "nmse_score": float(nmse_score),
            "power_spearman": power_spearman,
            "score_conventional": float(0.4 * pas + 0.4 * pdp + 0.2 * nmse_score),
            "score_traceback_leaked": float(0.4 * leaked + 0.4 * pdp + 0.2 * nmse_score),
        }


def score_candidates(specs: list[CandidateSpec]) -> dict[str, dict[str, float | bool]]:
    all_paths = sorted({path for spec in specs for path in spec.paths})
    for path in [TEST_TRUTH, *all_paths]:
        require_file(path)
    arrays = {path: np.load(path, mmap_mode="r") for path in all_paths}
    truth = np.load(TEST_TRUTH, mmap_mode="r")
    accum = {spec.name: Accumulator(spec.name) for spec in specs}
    n = len(truth)
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        truth_block = np.asarray(truth[start:end], dtype=np.complex64)
        truth_views = calibration.spectral_views(truth_block)
        loaded = {path: np.asarray(arrays[path][start:end], dtype=np.complex64) for path in all_paths}
        for spec in specs:
            pred = np.zeros_like(truth_block, dtype=np.complex64)
            for weight, path in zip(spec.weights, spec.paths):
                pred += np.float32(weight) * loaded[path]
            if spec.preserve_power_to_first:
                reference = loaded[spec.paths[0]]
                ref_power = np.maximum(np.sum(np.abs(reference) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY)
                pred_power = np.maximum(np.sum(np.abs(pred) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY)
                pred *= np.sqrt(ref_power / pred_power)[:, None, None, None].astype(np.float32)
            accum[spec.name].update(pred.astype(np.complex64), truth_block, truth_views)
        print(f"v220 local gate signal audit {end}/{n}", flush=True)
    return {name: item.finish() for name, item in accum.items()}


def main() -> None:
    specs = candidate_specs()
    scores = score_candidates(specs)
    ranked = sorted(scores, key=lambda name: float(scores[name]["score_traceback_leaked"]), reverse=True)
    best_official = max(
        (
            name
            for name in scores
            if next(spec for spec in specs if spec.name == name).compliance_class == "official_data_or_ai_backbone"
        ),
        key=lambda name: float(scores[name]["score_traceback_leaked"]),
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "If the v219 middle-weak support insight is already present in compliant official-data "
            "branches, their local traceback score or simple mixtures should approach the v212 gate."
        ),
        "local_gate": LOCAL_GATE,
        "candidate_specs": {
            spec.name: {
                "paths": [PATHS.relative(path) for path in spec.paths],
                "weights": list(spec.weights),
                "preserve_power_to_first": spec.preserve_power_to_first,
                "compliance_class": spec.compliance_class,
                "note": spec.note,
            }
            for spec in specs
        },
        "scores": scores,
        "ranked": ranked,
        "best_official_data_or_ai_backbone": best_official,
        "best_official_score_traceback_leaked": scores[best_official]["score_traceback_leaked"],
        "best_official_gap_to_gate": float(LOCAL_GATE - float(scores[best_official]["score_traceback_leaked"])),
        "decision": (
            "No v220 submission package is created.  The best official-data/AI-backbone probe remains far below "
            "the required v212 local gate, so delivering it would violate the current handoff rule."
        ),
        "outputs": {"report": PATHS.relative(REPORT)},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

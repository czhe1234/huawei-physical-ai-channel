#!/usr/bin/env python3
"""Apply the v337 H-conditional correction without changing v282 S/PDP mass."""

from __future__ import annotations

import json

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v337_componentwise_deformable_attention as v337
import analyze_v338_joint_component_pdp_carrier as v338
import generate_v187_leaked_pas_lifecycle_projection as leak
from generate_v87_easy_field_projection import summarize
from powerfield_common import TINY
from project_paths import PATHS


VERSION_NUMBER = 339
VERSION = "v339_mass_preserving_conditional_h"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = OUT_DIR / "v339_Round1_Test_Channel.npy"
TARGETS_OUT = OUT_DIR / "v339_conditional_h_targets.npz"
SOURCE = PATHS.cache_dir(282) / "v282_Round1_Test_Channel.npy"
V338_TARGETS = PATHS.cache_dir(338) / "v338_joint_component_pdp_targets.npz"

BATCH = 5
ITERATIONS = 4
FLOOR = 1e-11


def extract_sh(source: np.ndarray) -> np.ndarray:
    output = np.empty((len(source), v337.S_SIZE, v337.H_SIZE), dtype=np.float32)
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        spec = leak.leaked_forward(np.asarray(source[start:end], dtype=np.complex64))
        power = leak.leaked_cell_power(spec)
        output[start:end] = v337.normalize(
            power.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
        )
    return output


def project_conditional(target_sh: np.ndarray) -> dict[str, np.ndarray]:
    source = np.load(SOURCE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    diagnostics: dict[str, list[np.ndarray]] = {
        "sh_l1": [],
        "s_marginal_l1": [],
        "pdp_l1": [],
        "target_cos": [],
        "power_factor": [],
        "gain_min": [],
        "gain_max": [],
    }
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        p0 = np.maximum(
            np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY
        ).astype(np.float32)
        initial_sh, initial_d = v247.extract_current(h0)
        spec = leak.leaked_forward(h0)
        gain_min = np.ones(end - start, dtype=np.float32)
        gain_max = np.ones(end - start, dtype=np.float32)
        for _ in range(ITERATIONS):
            power = leak.leaked_cell_power(spec)
            current_sh = v337.normalize(
                power.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
            )
            ratio = np.maximum(target_sh[start:end], FLOOR) / np.maximum(
                current_sh, FLOOR
            )
            gain = np.clip(np.sqrt(ratio).astype(np.float32), 0.20, 5.0)
            spec *= gain[:, None, :, :, None, None]
            gain_min = np.minimum(gain_min, gain.reshape(end - start, -1).min(axis=1))
            gain_max = np.maximum(gain_max, gain.reshape(end - start, -1).max(axis=1))
        h1 = leak.leaked_inverse(spec).astype(np.complex64)
        p1 = np.maximum(
            np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY
        ).astype(np.float32)
        h1 *= np.sqrt(p0 / p1)[:, None, None, None].astype(np.float32)
        output[start:end] = h1
        output.flush()

        final_sh, final_d = v247.extract_current(h1)
        diagnostics["sh_l1"].append(
            np.sum(np.abs(final_sh - initial_sh), axis=(1, 2)).astype(np.float32)
        )
        diagnostics["s_marginal_l1"].append(
            np.sum(
                np.abs(final_sh.sum(axis=2) - initial_sh.sum(axis=2)), axis=1
            ).astype(np.float32)
        )
        diagnostics["pdp_l1"].append(
            np.sum(np.abs(final_d - initial_d), axis=1).astype(np.float32)
        )
        diagnostics["target_cos"].append(
            v337.row_cos(final_sh, target_sh[start:end])
        )
        diagnostics["power_factor"].append((p1 / p0).astype(np.float32))
        diagnostics["gain_min"].append(gain_min)
        diagnostics["gain_max"].append(gain_max)
        if end == len(source) or end % 100 == 0:
            print(f"v339 projection {end}/{len(source)}", flush=True)
    return {key: np.concatenate(value) for key, value in diagnostics.items()}


def main() -> None:
    for path in (SOURCE, V338_TARGETS):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = np.load(SOURCE, mmap_mode="r")
    current_sh = extract_sh(source)
    component_sh = np.load(V338_TARGETS)["component_sh"].astype(np.float32)
    target_s = current_sh.sum(axis=2)
    target_sh = v338.impose_s_marginal(component_sh, target_s)
    target_s_l1 = np.sum(
        np.abs(target_sh.sum(axis=2) - target_s), axis=1
    ).astype(np.float32)
    np.savez_compressed(
        TARGETS_OUT,
        current_sh=current_sh,
        component_sh=component_sh,
        target_sh=target_sh,
        target_s=target_s,
    )
    projection = project_conditional(target_sh)
    local_score = v208.score_prediction(OUTPUT)
    report = {
        "method": VERSION,
        "status": "strict_complete_local_diagnostic",
        "primary_hypothesis": (
            "v282 already carries the stronger official-only PDP; v337 should only redistribute "
            "H-angle conditionals inside each fixed leaked-S row so that PDP cannot be overwritten."
        ),
        "compliance_boundary": {
            "uses": [
                "v282 strict official-only complete H",
                "v337 official-only component conditional target",
                "official train H, coordinates and point-cloud features inherited by both",
            ],
            "does_not_use": [
                "v211/v212 outputs or labels",
                "synthetic training labels",
                "restored scene data or RT coefficients",
                "online-score optimization",
            ],
            "local_truth_usage": "one frozen score after target and projection were fixed",
        },
        "parameters": {
            "iterations": ITERATIONS,
            "gain_clip": [0.20, 5.0],
            "source": PATHS.relative(SOURCE),
        },
        "diagnostics": {
            "target_s_identity_l1": summarize(target_s_l1),
            **{key: summarize(value) for key, value in projection.items()},
        },
        "local_score": local_score,
        "references": {
            "v282_local": 0.5858171973760586,
            "v284_local": 0.5925112281386167,
            "v211_local": 0.6350959054536596,
            "v212_local": 0.6456036433098431,
        },
        "outputs": {
            "prediction": PATHS.relative(OUTPUT),
            "targets": PATHS.relative(TARGETS_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(local_score, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

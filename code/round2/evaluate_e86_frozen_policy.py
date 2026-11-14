#!/usr/bin/env python3
"""Evaluate one frozen E86 endpoint policy on the canonical 500 points."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from analyze_e86_surface_endpoint_v11_blend import Candidate, make_candidate
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics


ARTIFACT_DIR = PATHS.artifacts / "round2" / "surface_interaction_energy_model"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_surface_interaction_energy_model"
REPORT = ANALYSIS_DIR / "e86_frozen_policy.json"
OUTPUT = ARTIFACT_DIR / "e86_frozen_geometry_sequence_oof_channel.npy"
V11_CHANNEL = PATHS.artifacts / "round2" / "canonical_500_version_replay" / "V11_Canonical_500_OOF_Channel.npy"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
OOF_META = PATHS.artifacts / "round2" / "pre_sionna" / "round2_pre_sionna_oof_foundation_meta.npz"
GEOMETRY_PAS = ARTIFACT_DIR / "e86_geometry_sequence_rendered_pas_f16.npy"
GEOMETRY_PDP = ARTIFACT_DIR / "e86_geometry_sequence_rendered_pdp_f16.npy"
NO_SEQUENCE_PAS = ARTIFACT_DIR / "e86_no_sequence_rendered_pas_f16.npy"
NO_SEQUENCE_PDP = ARTIFACT_DIR / "e86_no_sequence_rendered_pdp_f16.npy"


def fold_summary(values, baseline, fold_id):
    rows = []
    for fold in range(5):
        mask = fold_id == fold
        selected = statistics(*values, mask)
        anchor = statistics(*baseline, mask)
        rows.append({
            "fold": fold,
            "selected": selected,
            "baseline": anchor,
            "delta": float(selected["score"] - anchor["score"]),
        })
    return rows


def main() -> None:
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    base_np = np.load(V11_CHANNEL, mmap_mode="r")
    truth_np = np.asarray(
        np.load(TRAIN_CHANNEL, mmap_mode="r")[query_ids], dtype=np.complex64
    )
    base = torch.as_tensor(np.array(base_np, copy=True), device=device)
    truth = torch.as_tensor(truth_np, device=device)
    truth_pas = pas_distribution(truth)
    truth_pdp = pdp_distribution(truth)
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    baseline = point_metrics(base, truth, truth_pas, truth_pdp)

    fixed = {
        "mode": "sqrt",
        "pas_beta": 0.20,
        "pdp_beta": 0.20,
        "order": "pas_pdp",
        "cycles": 4,
        "selection": "frozen after E86 grid; no per-fold choice",
    }
    endpoint_results = {}
    output = np.empty_like(np.asarray(base_np), dtype=np.complex64)
    for name, pas_path, pdp_path in (
        ("geometry_sequence", GEOMETRY_PAS, GEOMETRY_PDP),
        ("no_sequence", NO_SEQUENCE_PAS, NO_SEQUENCE_PDP),
    ):
        endpoint_pas = torch.as_tensor(
            np.asarray(np.load(pas_path, mmap_mode="r"), dtype=np.float32), device=device
        )
        endpoint_pdp = torch.as_tensor(
            np.asarray(np.load(pdp_path, mmap_mode="r"), dtype=np.float32), device=device
        )
        spec = Candidate(
            endpoint=name,
            mode=fixed["mode"],
            pas_beta=fixed["pas_beta"],
            pdp_beta=fixed["pdp_beta"],
            order=fixed["order"],
            cycles=fixed["cycles"],
        )
        with torch.inference_mode():
            prediction = make_candidate(
                base,
                base_pas,
                base_pdp,
                endpoint_pas,
                endpoint_pdp,
                spec,
            )
            values = point_metrics(prediction, truth, truth_pas, truth_pdp)
        summary = statistics(*values)
        endpoint_results[name] = {
            "summary": summary,
            "delta": float(summary["score"] - statistics(*baseline)["score"]),
            "folds": fold_summary(values, baseline, fold_id),
            "power_ratio_max_abs_error": float(
                np.max(
                    np.abs(
                        (
                            prediction.abs().square().sum(dim=(1, 2, 3))
                            / base.abs().square().sum(dim=(1, 2, 3)).clamp_min(1e-30)
                        ).cpu().numpy()
                        - 1.0
                    )
                )
            ),
        }
        if name == "geometry_sequence":
            output[:] = prediction.cpu().numpy().astype(np.complex64)

    np.save(OUTPUT, output)
    payload = {
        "experiment": "Round2-E86-frozen-policy",
        "status": "analysis_complete",
        "validation": "fixed 500-point complete official score",
        "baseline": statistics(*baseline),
        "fixed_policy": fixed,
        "endpoint_results": endpoint_results,
        "output": {
            "path": PATHS.relative(OUTPUT),
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": bool(np.isfinite(output.real).all() and np.isfinite(output.imag).all()),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

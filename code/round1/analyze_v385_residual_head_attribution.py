#!/usr/bin/env python3
"""Attribute v384 existence and conditional-mass residual heads."""

from __future__ import annotations

import json

import numpy as np
import torch

import build_v382_dense_lifecycle_examples as v382
import train_v375_synthetic_lifecycle_pretraining as v375
import train_v384_dense_boundary_residual as v384
from project_paths import PATHS


VERSION_NUMBER = 385
VERSION = "v385_residual_head_attribution"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"

MASS_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_AP_LIFT = 0.02
MIN_MASS_LIFT = 0.005
MIN_SCENE_WINS = 3


def load_data() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    examples = np.load(v384.EXAMPLES)
    data = {
        key: examples[key]
        for key in ("feature", "binary", "mass", "scene", "family", "group")
    }
    baseline_cache = np.load(v384.BASELINE_CACHE)
    baseline = {key: baseline_cache[key] for key in baseline_cache.files}
    checkpoint = torch.load(v384.CHECKPOINT, map_location="cpu", weights_only=False)
    return data, baseline, checkpoint


def full_predictions(
    data: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    checkpoint: dict[str, object],
    rows: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = v384.BoundaryResidualNet(data["feature"].shape[1])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    feature = np.clip((data["feature"][rows] - mean) / std, -8.0, 8.0).astype(
        np.float32
    )
    probability, full_log_mass = v384.predict(
        model,
        feature,
        baseline["probability"][rows],
        baseline["mass"][rows],
        device,
    )
    baseline_log_mass = v384.baseline_log_conditional(
        baseline["probability"][rows], baseline["mass"][rows]
    )
    return probability, full_log_mass, baseline_log_mass


def metrics_for(
    data: dict[str, np.ndarray],
    rows: np.ndarray,
    probability: np.ndarray,
    log_mass: np.ndarray,
) -> dict[str, float]:
    return v375.metrics(
        data["binary"][rows],
        data["mass"][rows],
        data["group"][rows],
        probability,
        log_mass,
    )


def blend_rows(
    data: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    checkpoint: dict[str, object],
    rows: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    full_probability, full_log_mass, baseline_log_mass = full_predictions(
        data, baseline, checkpoint, rows, device
    )
    baseline_probability = baseline["probability"][rows]
    attribution = {
        "baseline": metrics_for(
            data, rows, baseline_probability, baseline_log_mass
        ),
        "existence_only": metrics_for(
            data, rows, full_probability, baseline_log_mass
        ),
        "mass_only": metrics_for(
            data, rows, baseline_probability, full_log_mass
        ),
        "full": metrics_for(data, rows, full_probability, full_log_mass),
    }
    strengths = []
    for strength in MASS_STRENGTHS:
        log_mass = baseline_log_mass + strength * (
            full_log_mass - baseline_log_mass
        )
        strengths.append(
            {
                "strength": float(strength),
                **metrics_for(data, rows, full_probability, log_mass),
            }
        )
    attribution["mass_strengths"] = strengths
    return attribution, full_probability, full_log_mass, baseline_log_mass


def main() -> None:
    for path in (v384.EXAMPLES, v384.BASELINE_CACHE, v384.CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(path)
    data, baseline, checkpoint = load_data()
    scene = data["scene"].astype(np.int64)
    validation_rows = np.flatnonzero(np.isin(scene, v382.VALIDATION_SCENES))
    test_rows = np.flatnonzero(np.isin(scene, v382.TEST_SCENES))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation, _, _, _ = blend_rows(
        data, baseline, checkpoint, validation_rows, device
    )
    selected = max(
        validation["mass_strengths"],
        key=lambda row: (row["group_mass_cosine"], row["existence_ap"]),
    )
    strength = float(selected["strength"])
    test, full_probability, full_log_mass, baseline_log_mass = blend_rows(
        data, baseline, checkpoint, test_rows, device
    )
    selected_log_mass = baseline_log_mass + strength * (
        full_log_mass - baseline_log_mass
    )
    selected_test = metrics_for(
        data, test_rows, full_probability, selected_log_mass
    )
    scene_rows = []
    for scene_id in v382.TEST_SCENES:
        local = scene[test_rows] == scene_id
        rows = test_rows[local]
        base = metrics_for(
            data,
            rows,
            baseline["probability"][rows],
            v384.baseline_log_conditional(
                baseline["probability"][rows], baseline["mass"][rows]
            ),
        )
        current = metrics_for(
            data, rows, full_probability[local], selected_log_mass[local]
        )
        scene_rows.append(
            {
                "scene": int(scene_id),
                "baseline": base,
                "selected": current,
                "ap_delta": float(current["existence_ap"] - base["existence_ap"]),
                "mass_delta": float(
                    current["group_mass_cosine"] - base["group_mass_cosine"]
                ),
            }
        )
    baseline_test = test["baseline"]
    deltas = {
        "ap": float(selected_test["existence_ap"] - baseline_test["existence_ap"]),
        "mass": float(
            selected_test["group_mass_cosine"]
            - baseline_test["group_mass_cosine"]
        ),
        "conditional_log_mass_rmse_reduction": float(
            baseline_test["conditional_log_mass_rmse"]
            - selected_test["conditional_log_mass_rmse"]
        ),
    }
    scene_wins = int(
        sum(row["ap_delta"] > 0.0 and row["mass_delta"] > 0.0 for row in scene_rows)
    )
    passed = bool(
        deltas["ap"] >= MIN_AP_LIFT
        and deltas["mass"] >= MIN_MASS_LIFT
        and scene_wins >= MIN_SCENE_WINS
    )
    report = {
        "method": VERSION,
        "status": "head_attribution_passed" if passed else "analysis_only_no_transfer",
        "scope": "frozen v384 head attribution; no retraining/official target/ZIP",
        "primary_hypothesis": (
            "The v384 existence residual is valid, while a validation-selected weaker "
            "conditional-mass correction avoids the group-cosine regression."
        ),
        "parameters": {
            "mass_strengths": list(MASS_STRENGTHS),
            "selection": "maximum validation group_mass_cosine",
        },
        "validation": validation,
        "selected_mass_strength": strength,
        "test": {
            "attribution": test,
            "selected": selected_test,
            "baseline": baseline_test,
            "deltas": deltas,
            "scene_rows": scene_rows,
            "scene_wins": scene_wins,
        },
        "decision_rule": {
            "ap_lift": f">={MIN_AP_LIFT}",
            "mass_lift": f">={MIN_MASS_LIFT}",
            "scene_wins": f">={MIN_SCENE_WINS}/4",
            "passed": passed,
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "checkpoint": PATHS.relative(v384.CHECKPOINT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["validation"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(report["test"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(report["decision_rule"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

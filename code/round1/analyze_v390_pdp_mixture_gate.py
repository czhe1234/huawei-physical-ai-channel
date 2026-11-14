#!/usr/bin/env python3
"""Audit a held-scene mixture gate for the frozen v389 PDP residual."""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

import build_v382_dense_lifecycle_examples as v382
import train_v389_dense_geometry_pdp_residual as v389
from project_paths import PATHS


VERSION_NUMBER = 390
VERSION = "v390_pdp_mixture_gate"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
GATE_TREES = 500
MIN_COSINE_LIFT = 0.005
MIN_GATE_VS_CONSTANT = 0.002
MIN_EMD_REDUCTION = 0.0
MIN_SCENE_WINS = 3


def mix(base: np.ndarray, full: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    if np.ndim(alpha) == 0:
        output = base + float(alpha) * (full - base)
    else:
        output = base + np.asarray(alpha)[:, None] * (full - base)
    return v389.normalize_rows(np.maximum(output, 0.0).astype(np.float32))


def point_cosine(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.sum(truth * prediction, axis=1) / np.maximum(
        np.linalg.norm(truth, axis=1) * np.linalg.norm(prediction, axis=1), 1e-20
    )


def action_features(
    geometry: np.ndarray,
    baseline: np.ndarray,
    variation: np.ndarray,
    full: np.ndarray,
) -> np.ndarray:
    entropy_base = -np.sum(
        baseline * np.log(np.maximum(baseline, 1e-8)), axis=1
    ) / np.log(v389.N_DELAY)
    entropy_full = -np.sum(
        full * np.log(np.maximum(full, 1e-8)), axis=1
    ) / np.log(v389.N_DELAY)
    cosine = point_cosine(baseline, full)
    cdf_distance = np.mean(
        np.abs(np.cumsum(full, axis=1) - np.cumsum(baseline, axis=1)), axis=1
    )
    bins = np.arange(v389.N_DELAY, dtype=np.float32)
    center_base = np.sum(baseline * bins[None, :], axis=1)
    center_full = np.sum(full * bins[None, :], axis=1)
    action = np.column_stack(
        [
            entropy_base,
            np.max(baseline, axis=1),
            np.sum(variation, axis=1),
            np.max(variation, axis=1),
            np.sum(np.abs(full - baseline), axis=1),
            cosine,
            cdf_distance,
            entropy_full - entropy_base,
            np.max(full, axis=1) - np.max(baseline, axis=1),
            (center_full - center_base) / v389.N_DELAY,
            (np.argmax(full, axis=1) - np.argmax(baseline, axis=1)) / v389.N_DELAY,
        ]
    ).astype(np.float32)
    return np.concatenate([geometry, action], axis=1)


def oracle_alpha(
    truth: np.ndarray, baseline: np.ndarray, full: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.stack(
        [point_cosine(truth, mix(baseline, full, alpha)) for alpha in ALPHAS],
        axis=1,
    )
    selected = np.argmax(scores, axis=1)
    return np.asarray(ALPHAS, dtype=np.float32)[selected], np.max(scores, axis=1)


def fit_gate(feature: np.ndarray, target: np.ndarray, seed: int) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor(
        n_estimators=GATE_TREES,
        max_depth=12,
        min_samples_leaf=20,
        max_features=0.7,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(feature, target)
    return model


def main() -> None:
    for path in (v389.DATASET_CACHE, v389.CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(path)
    cached = np.load(v389.DATASET_CACHE)
    data = {key: cached[key] for key in cached.files}
    checkpoint = torch.load(v389.CHECKPOINT, map_location="cpu", weights_only=False)
    scene = data["scene"].astype(np.int64)
    validation_rows = np.flatnonzero(np.isin(scene, v382.VALIDATION_SCENES))
    test_rows = np.flatnonzero(np.isin(scene, v382.TEST_SCENES))
    geometry_raw = np.concatenate(
        [data["geometry"], data["context"]], axis=1
    ).astype(np.float32)
    mean = np.asarray(checkpoint["geometry_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["geometry_std"], dtype=np.float32)
    geometry = np.clip((geometry_raw - mean) / std, -8.0, 8.0).astype(np.float32)
    model = v389.SpectrumResidualNet(geometry.shape[1])
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    spectrum = v389.spectral_input(data["baseline"], data["variation"])
    validation_full = v389.predict(
        model,
        spectrum[validation_rows],
        geometry[validation_rows],
        data["baseline"][validation_rows],
        device,
    )
    test_full = v389.predict(
        model,
        spectrum[test_rows],
        geometry[test_rows],
        data["baseline"][test_rows],
        device,
    )
    validation_feature = action_features(
        geometry[validation_rows],
        data["baseline"][validation_rows],
        data["variation"][validation_rows],
        validation_full,
    )
    test_feature = action_features(
        geometry[test_rows],
        data["baseline"][test_rows],
        data["variation"][test_rows],
        test_full,
    )
    validation_alpha, validation_oracle_score = oracle_alpha(
        data["truth"][validation_rows],
        data["baseline"][validation_rows],
        validation_full,
    )
    gate = fit_gate(validation_feature, validation_alpha, 39001)
    predicted_alpha = np.clip(gate.predict(test_feature), 0.0, 1.0).astype(np.float32)

    rng = np.random.default_rng(39002)
    shuffled_alpha = validation_alpha.copy()
    for scene_id in v382.VALIDATION_SCENES:
        rows = np.flatnonzero(scene[validation_rows] == scene_id)
        shuffled_alpha[rows] = shuffled_alpha[rows][rng.permutation(len(rows))]
    shuffled_gate = fit_gate(validation_feature, shuffled_alpha, 39003)
    shuffled_predicted_alpha = np.clip(
        shuffled_gate.predict(test_feature), 0.0, 1.0
    ).astype(np.float32)

    constant_rows = []
    for alpha in ALPHAS:
        prediction = mix(
            data["baseline"][validation_rows], validation_full, alpha
        )
        current = v389.metrics(data["truth"][validation_rows], prediction)
        constant_rows.append({"alpha": float(alpha), **current})
    selected_constant = max(constant_rows, key=lambda row: row["cosine"])
    constant_alpha = float(selected_constant["alpha"])

    baseline_test = v389.metrics(
        data["truth"][test_rows], data["baseline"][test_rows]
    )
    full_test = v389.metrics(data["truth"][test_rows], test_full)
    constant_prediction = mix(
        data["baseline"][test_rows], test_full, constant_alpha
    )
    constant_test = v389.metrics(data["truth"][test_rows], constant_prediction)
    gate_prediction = mix(
        data["baseline"][test_rows], test_full, predicted_alpha
    )
    gate_test = v389.metrics(data["truth"][test_rows], gate_prediction)
    shuffled_gate_prediction = mix(
        data["baseline"][test_rows], test_full, shuffled_predicted_alpha
    )
    shuffled_gate_test = v389.metrics(
        data["truth"][test_rows], shuffled_gate_prediction
    )
    oracle_test_alpha, oracle_test_score = oracle_alpha(
        data["truth"][test_rows], data["baseline"][test_rows], test_full
    )
    oracle_test = v389.metrics(
        data["truth"][test_rows],
        mix(data["baseline"][test_rows], test_full, oracle_test_alpha),
    )
    scene_rows = []
    for scene_id in v382.TEST_SCENES:
        keep = scene[test_rows] == scene_id
        base = v389.metrics(
            data["truth"][test_rows][keep], data["baseline"][test_rows][keep]
        )
        current = v389.metrics(
            data["truth"][test_rows][keep], gate_prediction[keep]
        )
        constant = v389.metrics(
            data["truth"][test_rows][keep], constant_prediction[keep]
        )
        scene_rows.append(
            {
                "scene": int(scene_id),
                "baseline": base,
                "gate": current,
                "constant": constant,
                "cosine_lift": float(current["cosine"] - base["cosine"]),
                "gate_vs_constant": float(
                    current["cosine"] - constant["cosine"]
                ),
                "alpha_mean": float(np.mean(predicted_alpha[keep])),
            }
        )
    cosine_lift = gate_test["cosine"] - baseline_test["cosine"]
    gate_vs_constant = gate_test["cosine"] - constant_test["cosine"]
    emd_reduction = (baseline_test["emd"] - gate_test["emd"]) / max(
        baseline_test["emd"], 1e-20
    )
    scene_wins = int(
        sum(
            row["cosine_lift"] > 0.0 and row["gate_vs_constant"] > 0.0
            for row in scene_rows
        )
    )
    passed = bool(
        cosine_lift >= MIN_COSINE_LIFT
        and gate_vs_constant >= MIN_GATE_VS_CONSTANT
        and emd_reduction >= MIN_EMD_REDUCTION
        and scene_wins >= MIN_SCENE_WINS
    )
    report = {
        "method": VERSION,
        "status": "pdp_gate_passed" if passed else "analysis_only_no_transfer",
        "scope": "held-scene v389 mixture gate audit",
        "primary_hypothesis": (
            "Observable baseline uncertainty and residual-action statistics identify "
            "where the frozen v389 PDP correction should be applied."
        ),
        "parameters": {
            "alphas": list(ALPHAS),
            "gate_trees": GATE_TREES,
            "gate_train_scenes": list(v382.VALIDATION_SCENES),
            "gate_test_scenes": list(v382.TEST_SCENES),
        },
        "validation": {
            "constant_rows": constant_rows,
            "selected_constant_alpha": constant_alpha,
            "oracle_alpha_mean": float(np.mean(validation_alpha)),
            "oracle_cosine": float(np.mean(validation_oracle_score)),
        },
        "test": {
            "baseline": baseline_test,
            "full": full_test,
            "constant": constant_test,
            "gate": gate_test,
            "shuffled_label_gate": shuffled_gate_test,
            "oracle": oracle_test,
            "oracle_alpha_mean": float(np.mean(oracle_test_alpha)),
            "predicted_alpha_mean": float(np.mean(predicted_alpha)),
            "shuffled_predicted_alpha_mean": float(
                np.mean(shuffled_predicted_alpha)
            ),
            "cosine_lift": float(cosine_lift),
            "gate_vs_constant": float(gate_vs_constant),
            "gate_vs_shuffled_label_gate": float(
                gate_test["cosine"] - shuffled_gate_test["cosine"]
            ),
            "emd_reduction": float(emd_reduction),
            "scene_rows": scene_rows,
            "scene_wins": scene_wins,
        },
        "decision_rule": {
            "cosine_lift": f">={MIN_COSINE_LIFT}",
            "gate_vs_constant": f">={MIN_GATE_VS_CONSTANT}",
            "emd_reduction": f">={MIN_EMD_REDUCTION}",
            "scene_wins": f">={MIN_SCENE_WINS}/4",
            "passed": passed,
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["test"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(report["decision_rule"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

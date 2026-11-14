#!/usr/bin/env python3
"""Quantify the official PAS/PDP cosine denominator correction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from project_paths import PATHS


VERSION_NUMBER = 446
VERSION = "v446_metric_epsilon_change"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
TRAIN = PATHS.round1 / "Round1_Train_Channel.npy"
TRUTH = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
PREDICTIONS = {
    "v212": PATHS.cache_dir(212) / "v212_Round1_Test_Channel.npy",
    "v284": PATHS.cache_dir(284) / "v284_Round1_Test_Channel.npy",
    "v426": PATHS.cache_dir(426) / "v426_Round1_Test_Channel.npy",
    "v436": PATHS.cache_dir(436) / "v436_Round1_Test_Channel.npy",
}
EPSILONS = (1e-12, 1e-10, 1e-8, 1e-6)
BATCH = 8


def summarize(value: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(value, dtype=np.float64)
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "min": float(np.min(x)),
        "p01": float(np.quantile(x, 0.01)),
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
    }


def spectral_power(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value6 = h.reshape(len(h), 2, 16, 8, 4, 192)
    pas = np.abs(np.fft.fft2(value6, axes=(2, 3))).astype(np.float32) ** 2
    pas = pas.reshape(len(h), 256, 4, 192)
    pdp = np.abs(np.fft.ifft(h, axis=-1)).astype(np.float32) ** 2
    return pas, pdp


def cosine_terms(
    prediction: np.ndarray, truth: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numerator = np.sum(prediction * truth, axis=axis, dtype=np.float64)
    pred_norm = np.sqrt(np.sum(prediction * prediction, axis=axis, dtype=np.float64))
    truth_norm = np.sqrt(np.sum(truth * truth, axis=axis, dtype=np.float64))
    return numerator, pred_norm, truth_norm


def compare_channels(prediction_path: Path, truth_path: Path) -> dict[str, object]:
    prediction = np.load(prediction_path, mmap_mode="r")
    truth = np.load(truth_path, mmap_mode="r")
    if prediction.shape != truth.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {truth.shape}")
    accum = {
        name: {"numerator": [], "pred_norm": [], "truth_norm": []}
        for name in ("pas", "pdp")
    }
    nmse_numerator = 0.0
    nmse_denominator = 0.0
    for start in range(0, len(prediction), BATCH):
        end = min(start + BATCH, len(prediction))
        pred = np.asarray(prediction[start:end], dtype=np.complex64)
        target = np.asarray(truth[start:end], dtype=np.complex64)
        pred_pas, pred_pdp = spectral_power(pred)
        true_pas, true_pdp = spectral_power(target)
        for name, values, axis in (
            ("pas", (pred_pas, true_pas), 1),
            ("pdp", (pred_pdp, true_pdp), 3),
        ):
            numerator, pred_norm, truth_norm = cosine_terms(*values, axis)
            accum[name]["numerator"].append(numerator.reshape(-1))
            accum[name]["pred_norm"].append(pred_norm.reshape(-1))
            accum[name]["truth_norm"].append(truth_norm.reshape(-1))
        nmse_numerator += float(np.sum(np.abs(pred - target) ** 2, dtype=np.float64))
        nmse_denominator += float(np.sum(np.abs(target) ** 2, dtype=np.float64))
    merged = {
        name: {
            key: np.concatenate(parts)
            for key, parts in values.items()
        }
        for name, values in accum.items()
    }
    nmse = nmse_numerator / max(nmse_denominator, 1e-30)
    nmse_score = 1.0 / (1.0 + nmse)
    output: dict[str, object] = {
        "prediction": PATHS.relative(prediction_path),
        "truth": PATHS.relative(truth_path),
        "nmse": float(nmse),
        "nmse_score": float(nmse_score),
        "norms": {},
        "new_metric": {},
        "old_metric_by_epsilon": {},
    }
    new_scores = {}
    for name, values in merged.items():
        product = values["pred_norm"] * values["truth_norm"]
        denominator = np.where(product == 0.0, 1e-12, product)
        score = values["numerator"] / denominator
        new_scores[name] = float(np.mean(score))
        output["new_metric"][name] = new_scores[name]
        output["norms"][name] = {
            "prediction": summarize(values["pred_norm"]),
            "truth": summarize(values["truth_norm"]),
            "zero_product_count": int(np.sum(product == 0.0)),
        }
    output["new_metric"]["composite"] = float(
        0.4 * new_scores["pas"] + 0.4 * new_scores["pdp"] + 0.2 * nmse_score
    )
    for epsilon in EPSILONS:
        scores = {}
        for name, values in merged.items():
            denominator = (values["pred_norm"] + epsilon) * (
                values["truth_norm"] + epsilon
            )
            scores[name] = float(np.mean(values["numerator"] / denominator))
        composite = 0.4 * scores["pas"] + 0.4 * scores["pdp"] + 0.2 * nmse_score
        output["old_metric_by_epsilon"][f"{epsilon:.0e}"] = {
            **scores,
            "composite": float(composite),
            "new_minus_old": {
                "pas": float(new_scores["pas"] - scores["pas"]),
                "pdp": float(new_scores["pdp"] - scores["pdp"]),
                "composite": float(output["new_metric"]["composite"] - composite),
            },
        }
    return output


def self_penalty(path: Path) -> dict[str, object]:
    value = np.load(path, mmap_mode="r")
    norms = {"pas": [], "pdp": []}
    for start in range(0, len(value), BATCH):
        block = np.asarray(value[start : start + BATCH], dtype=np.complex64)
        pas, pdp = spectral_power(block)
        norms["pas"].append(
            np.sqrt(np.sum(pas * pas, axis=1, dtype=np.float64)).reshape(-1)
        )
        norms["pdp"].append(
            np.sqrt(np.sum(pdp * pdp, axis=3, dtype=np.float64)).reshape(-1)
        )
    norms = {name: np.concatenate(parts) for name, parts in norms.items()}
    output: dict[str, object] = {
        "source": PATHS.relative(path),
        "norms": {name: summarize(value) for name, value in norms.items()},
        "old_perfect_prediction_score": {},
    }
    for epsilon in EPSILONS:
        current = {}
        for name, value in norms.items():
            current[name] = float(np.mean(np.where(value == 0.0, 0.0, (value / (value + epsilon)) ** 2)))
            current[f"norm_le_epsilon_{name}"] = int(np.sum(value <= epsilon))
            current[f"norm_le_10epsilon_{name}"] = int(np.sum(value <= 10.0 * epsilon))
        current["weighted_pas_pdp"] = float(0.5 * (current["pas"] + current["pdp"]))
        output["old_perfect_prediction_score"][f"{epsilon:.0e}"] = current
    return output


def main() -> None:
    for path in (TRAIN, TRUTH, *PREDICTIONS.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "method": VERSION,
        "status": "metric_change_quantified_on_released_train_and_local_diagnostic_truth",
        "formula": {
            "old": "dot / ((norm_prediction + epsilon) * (norm_truth + epsilon))",
            "new": "dot / (norm_prediction * norm_truth), except zero product -> 1e-12",
            "tested_old_epsilons": list(EPSILONS),
        },
        "official_train_self_penalty": self_penalty(TRAIN),
        "local_diagnostic_versions": {
            name: compare_channels(path, TRUTH)
            for name, path in PREDICTIONS.items()
        },
        "evidence_boundary": (
            "v207 is a local Sionna diagnostic reference, not hidden online truth; "
            "official-train norm distributions are the direct released-data evidence."
        ),
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    concise = {
        "official_train_self_penalty": report["official_train_self_penalty"][
            "old_perfect_prediction_score"
        ],
        "version_new_minus_old": {
            name: {
                epsilon: values["new_minus_old"]
                for epsilon, values in result["old_metric_by_epsilon"].items()
            }
            for name, result in report["local_diagnostic_versions"].items()
        },
    }
    print(json.dumps(concise, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

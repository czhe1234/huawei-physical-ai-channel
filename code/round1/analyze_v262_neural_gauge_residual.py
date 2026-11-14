#!/usr/bin/env python3
"""v262: point-cloud-conditioned neural residual for propagation gauges."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v254_route_topology_spectral_field as v254
import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v261_crossfit_gauge_transport as v261
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 262
VERSION = "v262_neural_gauge_residual"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = PATHS.artifacts / "checkpoints" / VERSION
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v262_Round1_Test_Channel.npy"
TARGETS = OUT_DIR / "v262_neural_gauge_targets.npz"
ACTIONS = OUT_DIR / "v262_neural_gauge_actions.npz"
GAUGES = OUT_DIR / "v262_train_test_gauges.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"

LOCAL_GATE = 0.645604
FOLD_SEEDS = v254.FOLD_SEEDS
VAL_COUNT = 320
BASE_K = 4
BASE_POWER = 3.0
EPOCHS = 360
BATCH_SIZE = 512
RESIDUAL_STRENGTHS = (0.25, 0.5, 0.75, 1.0)
MODES = ("d", "sh", "joint")
SPECTRAL_BLENDS = (0.25, 0.5, 0.75)
MIN_OFFICIAL_DELTA = 5e-4


class GaugeResidualNet(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 320),
            nn.LayerNorm(320),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(320, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.04),
            nn.Linear(256, 192),
            nn.GELU(),
            nn.Linear(192, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    total = out.reshape(len(out), -1).sum(axis=1)
    valid = total > 1e-20
    out[valid] /= total[valid].reshape(
        (int(np.sum(valid)),) + (1,) * (out.ndim - 1)
    )
    out[~valid] = 0.0
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def metrics(
    pred_sh: np.ndarray,
    pred_d: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> dict[str, float]:
    sh = float(np.mean(row_cos(pred_sh, truth_sh)))
    d = float(np.mean(row_cos(pred_d, truth_d)))
    return {"sh": sh, "d": d, "weighted": 0.52 * sh + 0.48 * d}


def fourier_features(pos: np.ndarray) -> np.ndarray:
    center = np.asarray([0.0, 0.0], dtype=np.float64)
    scale = np.asarray([180.0, 180.0], dtype=np.float64)
    xy = (pos - center) / scale
    projections = np.stack(
        [xy[:, 0], xy[:, 1], xy[:, 0] + xy[:, 1], xy[:, 0] - xy[:, 1]],
        axis=1,
    )
    values = [xy.astype(np.float32)]
    for frequency in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        angle = np.pi * frequency * projections
        values.append(np.sin(angle).astype(np.float32))
        values.append(np.cos(angle).astype(np.float32))
    return np.concatenate(values, axis=1).astype(np.float32)


def build_features(train_pos: np.ndarray, test_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_node, test_node = v257.build_node_features(train_pos, test_pos)
    return (
        np.concatenate([train_node, fourier_features(train_pos)], axis=1).astype(np.float32),
        np.concatenate([test_node, fourier_features(test_pos)], axis=1).astype(np.float32),
    )


def interpolate_gauge(
    fit_pos: np.ndarray,
    fit_gauge: np.ndarray,
    query_pos: np.ndarray,
    exclude_self: bool = False,
) -> np.ndarray:
    search_k = BASE_K + 1 if exclude_self else BASE_K
    distance, local = cKDTree(fit_pos).query(query_pos, k=search_k)
    if exclude_self:
        distance = distance[:, 1:]
        local = local[:, 1:]
    weight = 1.0 / np.maximum(distance, 0.05) ** BASE_POWER
    weight /= weight.sum(axis=1, keepdims=True)
    return np.sum(weight[:, :, None] * fit_gauge[local], axis=1).astype(np.float32)


def train_model(
    features: np.ndarray,
    residual: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[GaugeResidualNet, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    feature_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std[feature_std < 1e-5] = 1.0
    target_mean = residual.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_std = residual.std(axis=0, dtype=np.float64).astype(np.float32)
    target_std[target_std < 1e-5] = 1.0
    x = ((features - feature_mean) / feature_std).astype(np.float32)
    y = ((residual - target_mean) / target_std).astype(np.float32)
    torch.manual_seed(seed)
    model = GaugeResidualNet(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=3e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.35)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(x))
        total = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            xb = torch.from_numpy(x[ids]).to(device)
            yb = torch.from_numpy(y[ids]).to(device)
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(ids)
        if epoch == 1 or epoch % 60 == 0 or epoch == EPOCHS:
            history.append({"epoch": epoch, "loss": total / len(x)})
    return model, feature_mean, feature_std, target_mean, target_std, {
        "history": history,
        "feature_dim": int(x.shape[1]),
        "residual": {
            "d": summarize(residual[:, 0]),
            "s": summarize(residual[:, 1]),
            "h": summarize(residual[:, 2]),
        },
    }


@torch.no_grad()
def predict_residual(
    model: GaugeResidualNet,
    features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    x = ((features - feature_mean) / feature_std).astype(np.float32)
    output: list[np.ndarray] = []
    for start in range(0, len(x), 1024):
        pred = model(torch.from_numpy(x[start : start + 1024]).to(device)).cpu().numpy()
        output.append(pred)
    return (np.concatenate(output, axis=0) * target_std + target_mean).astype(np.float32)


def main() -> None:
    for path in (TARGET_CACHE, BASE, v257.SCENE_CACHE, v257.GEODESIC_CACHE, v257.HEIGHT_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)[:, :2]
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)[:, :2]
    train_features, test_features = build_features(pos, test_pos)
    data = np.load(TARGET_CACHE)
    sh = data["sh"].astype(np.float32)
    d = data["d"].astype(np.float32)
    valid_ids = np.flatnonzero(data["valid"].astype(bool)).astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_names = [
        f"r{strength:g}_{mode}_b{blend:g}"
        for strength in RESIDUAL_STRENGTHS
        for mode in MODES
        for blend in SPECTRAL_BLENDS
    ]
    config_results: dict[str, list[dict[str, float]]] = {
        name: [] for name in config_names
    }
    raw_results: list[dict[str, float]] = []
    fold_reports: list[dict[str, object]] = []
    for fold_index, seed in enumerate(FOLD_SEEDS):
        matched = v254.choose_seeded_matched_indices(pos, test_pos, VAL_COUNT, seed)
        val_ids = np.intersect1d(matched, valid_ids)
        fit_ids = np.setdiff1d(valid_ids, val_ids)
        fit_gauge, gauge_report = v261.compute_subset_gauges(fit_ids, pos, sh, d)
        loo_gauge = interpolate_gauge(
            pos[fit_ids], fit_gauge, pos[fit_ids], exclude_self=True
        )
        residual = fit_gauge - loo_gauge
        model, feature_mean, feature_std, target_mean, target_std, training = train_model(
            train_features[fit_ids],
            residual,
            VERSION_NUMBER * 100 + fold_index,
            device,
        )
        base_val_gauge = interpolate_gauge(pos[fit_ids], fit_gauge, pos[val_ids])
        val_residual = predict_residual(
            model,
            train_features[val_ids],
            feature_mean,
            feature_std,
            target_mean,
            target_std,
            device,
        )
        (
            neighbour_local,
            weight,
            neighbour_sh,
            neighbour_d,
            raw_sh,
            raw_d,
        ) = v261.spectral_bank(fit_ids, pos[val_ids], pos, sh, d)
        raw_results.append(metrics(raw_sh, raw_d, sh[val_ids], d[val_ids]))
        for strength in RESIDUAL_STRENGTHS:
            query_gauge = base_val_gauge + strength * val_residual
            aligned_sh, aligned_d, _ = v261.aligned_targets(
                query_gauge,
                fit_gauge,
                neighbour_local,
                weight,
                neighbour_sh,
                neighbour_d,
            )
            for mode in MODES:
                for blend in SPECTRAL_BLENDS:
                    name = f"r{strength:g}_{mode}_b{blend:g}"
                    pred_sh, pred_d = v261.blend_targets(
                        raw_sh, raw_d, aligned_sh, aligned_d, mode, blend
                    )
                    config_results[name].append(
                        metrics(pred_sh, pred_d, sh[val_ids], d[val_ids])
                    )
        fold_reports.append(
            {
                "fold": fold_index,
                "seed": seed,
                "fit_count": int(len(fit_ids)),
                "validation_count": int(len(val_ids)),
                "gauge": gauge_report,
                "training": training,
                "predicted_residual": {
                    "d": summarize(val_residual[:, 0]),
                    "s": summarize(val_residual[:, 1]),
                    "h": summarize(val_residual[:, 2]),
                },
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"v262 official fold {fold_index + 1}/{len(FOLD_SEEDS)}", flush=True)

    raw_mean = {
        key: float(np.mean([row[key] for row in raw_results]))
        for key in ("sh", "d", "weighted")
    }
    summary: dict[str, dict[str, object]] = {}
    for name, rows in config_results.items():
        weighted = np.asarray([row["weighted"] for row in rows])
        summary[name] = {
            "mean": {
                key: float(np.mean([row[key] for row in rows]))
                for key in ("sh", "d", "weighted")
            },
            "std_weighted": float(np.std(weighted)),
            "folds": rows,
        }
    best_name = max(summary, key=lambda name: float(summary[name]["mean"]["weighted"]))
    raw_fold = np.asarray([row["weighted"] for row in raw_results])
    best_fold = np.asarray([row["weighted"] for row in config_results[best_name]])
    official_delta = float(best_fold.mean() - raw_fold.mean())
    official_wins = int(np.sum(best_fold > raw_fold))
    official_selected = bool(official_delta >= MIN_OFFICIAL_DELTA and official_wins >= 3)

    local_score = None
    actions = None
    final_training = None
    checkpoint = None
    if official_selected:
        strength_text, mode, blend_text = best_name.split("_")
        strength = float(strength_text[1:])
        blend = float(blend_text[1:])
        fit_gauge, final_gauge_report = v261.compute_subset_gauges(valid_ids, pos, sh, d)
        loo_gauge = interpolate_gauge(
            pos[valid_ids], fit_gauge, pos[valid_ids], exclude_self=True
        )
        residual = fit_gauge - loo_gauge
        model, feature_mean, feature_std, target_mean, target_std, final_training = train_model(
            train_features[valid_ids], residual, VERSION_NUMBER * 100 + 91, device
        )
        checkpoint = CHECKPOINT_DIR / "seed_26291.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "target_mean": target_mean,
                "target_std": target_std,
                "config": best_name,
            },
            checkpoint,
        )
        base_test_gauge = interpolate_gauge(pos[valid_ids], fit_gauge, test_pos)
        test_residual = predict_residual(
            model,
            test_features,
            feature_mean,
            feature_std,
            target_mean,
            target_std,
            device,
        )
        test_gauge = base_test_gauge + strength * test_residual
        (
            neighbour_local,
            weight,
            neighbour_sh,
            neighbour_d,
            raw_sh,
            raw_d,
        ) = v261.spectral_bank(valid_ids, test_pos, pos, sh, d)
        aligned_sh, aligned_d, shifts = v261.aligned_targets(
            test_gauge,
            fit_gauge,
            neighbour_local,
            weight,
            neighbour_sh,
            neighbour_d,
        )
        transported_sh, transported_d = v261.blend_targets(
            raw_sh, raw_d, aligned_sh, aligned_d, mode, blend
        )
        base = np.load(BASE, mmap_mode="r")
        current_sh, current_d = v247.extract_current(np.asarray(base, dtype=np.complex64))
        target_sh = normalize(current_sh + transported_sh - raw_sh)
        target_d = normalize(current_d + transported_d - raw_d)
        np.savez_compressed(
            TARGETS,
            sh=target_sh,
            d=target_d,
            raw_sh=raw_sh,
            raw_d=raw_d,
            aligned_sh=aligned_sh,
            aligned_d=aligned_d,
            method=np.asarray(best_name),
        )
        np.savez_compressed(
            GAUGES,
            valid_ids=valid_ids,
            train_gauge=fit_gauge,
            base_test_gauge=base_test_gauge,
            test_residual=test_residual,
            test_gauge=test_gauge,
            **shifts,
        )
        actions = v247.project(target_sh, target_d, OUTPUT)
        np.savez_compressed(ACTIONS, **actions)
        local_score = v208.score_prediction(OUTPUT)
    else:
        final_gauge_report = None

    ranking = sorted(
        summary,
        key=lambda name: float(summary[name]["mean"]["weighted"]),
        reverse=True,
    )
    report = {
        "method": VERSION,
        "status": (
            "gate_passed_needs_packaging"
            if local_score is not None
            and float(local_score["score_traceback_leaked"]) > LOCAL_GATE
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "A point-cloud-conditioned neural residual can correct the obstruction-boundary "
            "errors that prevent coordinate KNN from extrapolating synchronized gauges."
        ),
        "compliance_boundary": {
            "uses": [
                "official fit-only H pair shifts",
                "official train/test coordinates",
                "official point-cloud/geodesic/blocked features",
                "PyTorch gauge residual network",
                "strict v239 carrier after official selection",
            ],
            "does_not_use": [
                "validation H in fit gauge or neural training",
                "synthetic or Sionna labels",
                "teacher output",
                "local replay truth for selection",
                "hidden labels",
                "raw RT H",
            ],
        },
        "parameters": {
            "fold_seeds": list(FOLD_SEEDS),
            "base_gauge_k": BASE_K,
            "base_gauge_power": BASE_POWER,
            "epochs": EPOCHS,
            "residual_strengths": list(RESIDUAL_STRENGTHS),
            "modes": list(MODES),
            "spectral_blends": list(SPECTRAL_BLENDS),
            "device": str(device),
        },
        "folds": fold_reports,
        "selection": {
            "raw_mean": raw_mean,
            "best_name": best_name,
            "best": summary[best_name],
            "official_delta": official_delta,
            "official_wins": official_wins,
            "official_selected": official_selected,
            "top_20": [{"name": name, **summary[name]} for name in ranking[:20]],
        },
        "final_training": final_training,
        "final_gauge": final_gauge_report,
        "projection_diagnostics": (
            {key: summarize(value) for key, value in actions.items()}
            if actions is not None
            else None
        ),
        "local_gate": (
            {
                "required": LOCAL_GATE,
                "score": local_score,
                "passed": bool(float(local_score["score_traceback_leaked"]) > LOCAL_GATE),
                "gap": float(local_score["score_traceback_leaked"]) - LOCAL_GATE,
            }
            if local_score is not None
            else None
        ),
        "outputs": {
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "targets": PATHS.relative(TARGETS) if TARGETS.is_file() else None,
            "actions": PATHS.relative(ACTIONS) if ACTIONS.is_file() else None,
            "gauges": PATHS.relative(GAUGES) if GAUGES.is_file() else None,
            "checkpoint": PATHS.relative(checkpoint) if checkpoint is not None else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

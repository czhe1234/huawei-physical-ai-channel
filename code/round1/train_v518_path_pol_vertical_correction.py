#!/usr/bin/env python3
"""Learn transferable per-path polarization-by-vertical complex corrections."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v507_complex_path_frequency_oracle as v507
import analyze_v510_joint_delay_h_complex_oracle as v510
import analyze_v516_shared_complex_path_correction as v516
from project_paths import PATHS


VERSION_NUMBER = 518
VERSION = "v518_path_pol_vertical_correction"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT = REPORT_DIR / f"{VERSION}.json"
POINT_METRICS = CACHE_DIR / "v518_point_metrics.npz"
COEFFICIENT_CACHE = PATHS.cache_dir(516) / "v516_path_coefficients_points160.npz"
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"

MODE_NAMES = (
    "source_atom_fit",
    "global_pol_vertical",
    "predicted_magnitude_pol_vertical",
    "predicted_complex_pol_vertical",
    "shuffled_label_control",
    "oracle_pol_vertical",
    "oracle_full_coefficients",
)
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--seed", type=int, default=518_001)
    return parser.parse_args()


class CorrectionMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 16 * 3),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).reshape(-1, 16, 3)


def target_factors(
    source: np.ndarray, official: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source4 = source.reshape(*source.shape[:2], 2, 8, 4).astype(np.complex128)
    official4 = official.reshape(*official.shape[:2], 2, 8, 4).astype(np.complex128)
    numerator = np.sum(np.conj(source4) * official4, axis=4)
    denominator = np.maximum(np.sum(np.abs(source4) ** 2, axis=4), TINY)
    factor = numerator / denominator
    labels = np.stack(
        [
            np.log(np.clip(np.abs(factor), 1e-3, 1e3)),
            np.cos(np.angle(factor)),
            np.sin(np.angle(factor)),
        ],
        axis=4,
    ).reshape(*source.shape[:2], 16, 3)
    official_energy = np.sum(np.abs(official4) ** 2, axis=4).reshape(
        *source.shape[:2], 16
    )
    point_total = np.maximum(np.sum(official_energy, axis=(1, 2), keepdims=True), TINY)
    weights = official_energy / point_total
    return factor, labels.astype(np.float32), weights.astype(np.float32)


def input_features(
    base: np.ndarray,
    source: np.ndarray,
) -> np.ndarray:
    source = source.astype(np.complex128)
    energy = np.sum(np.abs(source) ** 2, axis=2, keepdims=True)
    scale = np.sqrt(np.maximum(energy, TINY))
    normalized = source / scale
    strongest = np.argmax(np.abs(source), axis=2)
    reference = np.take_along_axis(source, strongest[..., None], axis=2)
    reference /= np.maximum(np.abs(reference), TINY)
    relative = normalized * np.conj(reference)
    log_magnitude = np.log(np.maximum(np.abs(source), TINY))
    log_magnitude -= np.mean(log_magnitude, axis=2, keepdims=True)
    output = np.concatenate(
        [
            base,
            log_magnitude.astype(np.float32),
            relative.real.astype(np.float32),
            relative.imag.astype(np.float32),
        ],
        axis=2,
    )
    output[~np.isfinite(output)] = 0.0
    return output.astype(np.float32)


def label_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    magnitude = (prediction[..., 0] - target[..., 0]) ** 2
    direction = F.normalize(prediction[..., 1:3], dim=-1, eps=1e-8)
    direction_loss = torch.sum((direction - target[..., 1:3]) ** 2, dim=-1)
    local = magnitude + 0.5 * direction_loss
    return torch.sum(local * weight) / torch.clamp(torch.sum(weight), min=1e-8)


@torch.no_grad()
def validation_loss(
    model: CorrectionMLP,
    features: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    model.eval()
    return float(label_loss(model(features), labels, weights).cpu())


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    w_validation: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[CorrectionMLP, dict[str, object], np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    mean = x_train.mean(axis=0, keepdims=True)
    std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-4)
    train_x = torch.from_numpy((x_train - mean) / std).to(device)
    train_y = torch.from_numpy(y_train).to(device)
    train_w = torch.from_numpy(w_train).to(device)
    validation_x = torch.from_numpy((x_validation - mean) / std).to(device)
    validation_y = torch.from_numpy(y_validation).to(device)
    validation_w = torch.from_numpy(w_validation).to(device)
    model = CorrectionMLP(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_x))
        running = 0.0
        for start in range(0, len(order), args.batch_size):
            ids = torch.from_numpy(order[start : start + args.batch_size]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = label_loss(
                model(train_x[ids]), train_y[ids], train_w[ids]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(ids)
        if epoch == 1 or epoch % 5 == 0:
            local_validation = validation_loss(
                model, validation_x, validation_y, validation_w
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": running / len(order),
                    "validation_loss": local_validation,
                }
            )
            if local_validation < best_loss - 1e-5:
                best_loss = local_validation
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 5
                if stale >= args.patience:
                    break
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    return (
        model,
        {
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "history": history,
        },
        mean.astype(np.float32),
        std.astype(np.float32),
    )


@torch.no_grad()
def predict(
    model: CorrectionMLP,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    value = torch.from_numpy((features - mean) / std).to(device)
    return model(value).cpu().numpy().astype(np.float32)


def factors_from_labels(labels: np.ndarray) -> np.ndarray:
    magnitude = np.exp(np.clip(labels[..., 0], -4.0, 4.0))
    direction = labels[..., 1] + 1j * labels[..., 2]
    direction /= np.maximum(np.abs(direction), 1e-8)
    return magnitude * direction


def weighted_global_label(labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denominator = np.maximum(np.sum(weights, axis=0), TINY)
    return np.sum(labels * weights[..., None], axis=0) / denominator[..., None]


def apply_factors(source: np.ndarray, factors: np.ndarray) -> np.ndarray:
    source4 = source.reshape(len(source), 2, 8, 4)
    return (source4 * factors.reshape(len(source), 2, 8, 1)).reshape(len(source), 64)


def summarize(value: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(value, dtype=np.float64)[mask]
    return {
        "count": int(len(selected)),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "p10": float(np.quantile(selected, 0.10)),
        "p90": float(np.quantile(selected, 0.90)),
    }


def main() -> None:
    args = parse_args()
    for path in (COEFFICIENT_CACHE, TRAIN_POS, TRAIN_H):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(COEFFICIENT_CACHE) as loaded:
        data = {name: np.asarray(loaded[name]) for name in loaded.files}
    point_ids = data["indices"].astype(np.int64)
    valid = data["valid"].astype(bool)
    source = data["source_coefficients"].astype(np.complex128)
    official = data["official_coefficients"].astype(np.complex128)
    all_positions = np.load(TRAIN_POS).astype(np.float32)
    all_regimes = v501.regime_codes()
    regimes = all_regimes[point_ids]
    fold_codes = v501.spatial_fold_codes(all_positions, all_regimes)[point_ids]
    base_features, _, _, _ = v516.path_features(
        data, all_positions[point_ids], regimes
    )
    features = input_features(base_features, source)
    oracle_factors, labels, weights = target_factors(source, official)
    weights *= valid[..., None]

    path_points = np.repeat(np.arange(len(point_ids)), v516.PATH_LIMIT)[valid.reshape(-1)]
    flat_features = features[valid]
    flat_labels = labels[valid]
    flat_weights = weights[valid]
    predicted_labels = np.zeros_like(labels)
    control_labels = np.zeros_like(labels)
    global_labels = np.zeros_like(labels)
    fold_details = []
    device = torch.device(args.device)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for fold in sorted(np.unique(fold_codes).tolist()):
        train_mask = fold_codes[path_points] != fold
        validation_mask = ~train_mask
        train_x = flat_features[train_mask]
        validation_x = flat_features[validation_mask]
        train_y = flat_labels[train_mask]
        validation_y = flat_labels[validation_mask]
        train_w = flat_weights[train_mask]
        validation_w = flat_weights[validation_mask]

        model, training, mean, std = train_model(
            train_x,
            train_y,
            train_w,
            validation_x,
            validation_y,
            validation_w,
            args,
            args.seed + int(fold),
        )
        predicted = predict(model, validation_x, mean, std, device)

        rng = np.random.default_rng(args.seed + 10_000 + int(fold))
        shuffled_y = train_y[rng.permutation(len(train_y))]
        control, control_training, control_mean, control_std = train_model(
            train_x,
            shuffled_y,
            train_w,
            validation_x,
            validation_y,
            validation_w,
            args,
            args.seed + 20_000 + int(fold),
        )
        controlled = predict(control, validation_x, control_mean, control_std, device)
        global_value = weighted_global_label(train_y, train_w)

        validation_rows = np.flatnonzero(validation_mask)
        validation_points = path_points[validation_rows]
        local_counts: dict[int, int] = {}
        for local, point_row in enumerate(validation_points):
            path_row = local_counts.get(int(point_row), 0)
            predicted_labels[point_row, path_row] = predicted[local]
            control_labels[point_row, path_row] = controlled[local]
            global_labels[point_row, path_row] = global_value
            local_counts[int(point_row)] = path_row + 1
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "feature_mean": mean,
                "feature_std": std,
                "input_dim": features.shape[2],
                "fold": int(fold),
            },
            CHECKPOINT_DIR / f"fold_{int(fold)}.pt",
        )
        fold_details.append(
            {
                "fold": int(fold),
                "train_paths": int(np.sum(train_mask)),
                "validation_paths": int(np.sum(validation_mask)),
                "training": training,
                "control_training": control_training,
            }
        )
        print(
            f"v518 fold {fold}: epoch={training['best_epoch']} "
            f"loss={training['best_validation_loss']:.6f}",
            flush=True,
        )

    predicted_factors = factors_from_labels(predicted_labels)
    predicted_magnitude = np.abs(predicted_factors)
    global_factors = factors_from_labels(global_labels)
    control_factors = factors_from_labels(control_labels)

    truth = np.load(TRAIN_H, mmap_mode="r")
    metric_names = ("pdp", "leaked_full", "complex_coherence", "nmse", "traceback_like")
    metrics = {
        name: np.zeros((len(MODE_NAMES), len(point_ids)), dtype=np.float32)
        for name in metric_names
    }
    for row, point_id in enumerate(point_ids):
        local_valid = valid[row]
        tau = data["tau"][row, local_valid]
        h_center = data["h_center"][row, local_valid]
        source_local = source[row, local_valid]
        official_local = official[row, local_valid]
        coefficient_sets = {
            "source_atom_fit": source_local,
            "global_pol_vertical": apply_factors(
                source_local, global_factors[row, local_valid]
            ),
            "predicted_magnitude_pol_vertical": apply_factors(
                source_local, predicted_magnitude[row, local_valid]
            ),
            "predicted_complex_pol_vertical": apply_factors(
                source_local, predicted_factors[row, local_valid]
            ),
            "shuffled_label_control": apply_factors(
                source_local, control_factors[row, local_valid]
            ),
            "oracle_pol_vertical": apply_factors(
                source_local, oracle_factors[row, local_valid]
            ),
            "oracle_full_coefficients": official_local,
        }
        matrix = v510.joint_basis(tau, h_center)
        target_channel = np.asarray(truth[int(point_id)], dtype=np.complex64)
        for mode_index, mode in enumerate(MODE_NAMES):
            prediction = v510.channel_from_matrix(matrix @ coefficient_sets[mode])
            result = v507.complex_metrics(prediction, target_channel)
            for name in metric_names:
                metrics[name][mode_index, row] = result[name]
        if (row + 1) % 10 == 0 or row + 1 == len(point_ids):
            print(f"v518 render {row + 1}/{len(point_ids)}", flush=True)

    source_index = MODE_NAMES.index("source_atom_fit")
    global_index = MODE_NAMES.index("global_pol_vertical")
    summary = []
    for mode_index, mode in enumerate(MODE_NAMES):
        row = {"mode": mode, "overall": {}, "heavy": {}, "fold_gain_vs_global": []}
        for name, values in metrics.items():
            row["overall"][name] = summarize(
                values[mode_index], np.ones(len(point_ids), dtype=bool)
            )
            row["heavy"][name] = summarize(values[mode_index], regimes == 3)
        for fold in sorted(np.unique(fold_codes).tolist()):
            mask = fold_codes == fold
            row["fold_gain_vs_global"].append(
                float(
                    np.mean(metrics["traceback_like"][mode_index, mask])
                    - np.mean(metrics["traceback_like"][global_index, mask])
                )
            )
        row["positive_folds_vs_global"] = int(
            np.sum(np.asarray(row["fold_gain_vs_global"]) > 0.0)
        )
        summary.append(row)

    source_row = summary[source_index]
    global_row = summary[global_index]
    predicted_row = next(
        row for row in summary if row["mode"] == "predicted_complex_pol_vertical"
    )
    magnitude_row = next(
        row for row in summary if row["mode"] == "predicted_magnitude_pol_vertical"
    )
    control_row = next(
        row for row in summary if row["mode"] == "shuffled_label_control"
    )

    def mean_gain(row: dict[str, object]) -> float:
        return float(np.mean(row["fold_gain_vs_global"]))

    def spectral_gain(row: dict[str, object], baseline: dict[str, object]) -> float:
        return float(
            0.5
            * (
                row["overall"]["pdp"]["mean"]
                + row["overall"]["leaked_full"]["mean"]
                - baseline["overall"]["pdp"]["mean"]
                - baseline["overall"]["leaked_full"]["mean"]
            )
        )

    predicted_gain = mean_gain(predicted_row)
    magnitude_gain = mean_gain(magnitude_row)
    control_gain = mean_gain(control_row)
    predicted_spectral = spectral_gain(predicted_row, global_row)
    magnitude_spectral = spectral_gain(magnitude_row, global_row)
    supported = bool(
        max(predicted_gain, magnitude_gain) >= 0.005
        and max(
            predicted_row["positive_folds_vs_global"],
            magnitude_row["positive_folds_vs_global"],
        )
        >= 4
        and max(predicted_spectral, magnitude_spectral) >= 0.005
        and max(predicted_gain, magnitude_gain) - control_gain >= 0.003
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        POINT_METRICS,
        indices=point_ids,
        regimes=regimes,
        fold_codes=fold_codes,
        mode_names=np.asarray(MODE_NAMES),
        predicted_factors=predicted_factors.astype(np.complex64),
        global_factors=global_factors.astype(np.complex64),
        **metrics,
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The v517 per-path polarization-by-vertical oracle structure can be predicted "
            "from path geometry and Sionna complex port signatures at spatially unseen points."
        ),
        "point_count": int(len(point_ids)),
        "split": "v501 five spatial folds",
        "model": {
            "type": "four-layer MLP",
            "input_dim": int(features.shape[2]),
            "output": "16 per-path factors as log magnitude, phase cosine, phase sine",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
        },
        "fold_details": fold_details,
        "summary": summary,
        "decision": {
            "global_gain_vs_source": float(
                global_row["overall"]["traceback_like"]["mean"]
                - source_row["overall"]["traceback_like"]["mean"]
            ),
            "predicted_complex_gain_vs_global": predicted_gain,
            "predicted_complex_positive_folds_vs_global": predicted_row[
                "positive_folds_vs_global"
            ],
            "predicted_complex_spectral_gain_vs_global": predicted_spectral,
            "predicted_magnitude_gain_vs_global": magnitude_gain,
            "predicted_magnitude_positive_folds_vs_global": magnitude_row[
                "positive_folds_vs_global"
            ],
            "predicted_magnitude_spectral_gain_vs_global": magnitude_spectral,
            "shuffled_gain_vs_global": control_gain,
            "independent_margin_vs_shuffled": max(predicted_gain, magnitude_gain)
            - control_gain,
            "path_pol_vertical_correction_supported": supported,
            "gate": (
                "gain over global >= 0.005, at least 4/5 folds positive, spectral gain "
                ">= 0.005, and margin over shuffled >= 0.003"
            ),
        },
        "outputs": {
            "checkpoints": PATHS.relative(CHECKPOINT_DIR),
            "point_metrics": PATHS.relative(POINT_METRICS),
            "report": PATHS.relative(REPORT),
            "submission_zip": None,
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "global_gain_vs_source": report["decision"]["global_gain_vs_source"],
                "predicted_complex_gain_vs_global": predicted_gain,
                "predicted_complex_spectral_gain": predicted_spectral,
                "predicted_magnitude_gain_vs_global": magnitude_gain,
                "predicted_magnitude_spectral_gain": magnitude_spectral,
                "shuffled_gain_vs_global": control_gain,
                "supported": supported,
                "report": PATHS.relative(REPORT),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

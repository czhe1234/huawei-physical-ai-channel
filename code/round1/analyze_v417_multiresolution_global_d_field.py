#!/usr/bin/env python3
"""Learn a bounded multi-resolution planar residual for normalized global D."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v402_visible_path_residual_factorization as v402
import analyze_v412_path_sequence_identity_student as v412
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 417
VERSION = "v417_multiresolution_global_d_field"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v417_multiresolution_global_d_field_oof.npz"

RESOLUTIONS = (8, 16, 32, 64)
GRID_DIM = 24
FOURIER_FREQUENCIES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
NEIGHBORS = 16
DISTANCE_FLOOR = 2.0
EPOCHS = 180
BATCH_SIZE = 192
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 3e-4
RESIDUAL_BOUND = 1.25
TV_WEIGHT = 2e-5
MIN_D_DELTA = 0.01
MIN_FOLD_WINS = 4


def normalize(value: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = out.sum(axis=1, keepdims=True, dtype=np.float64)
    valid = total[:, 0] > 1e-20
    out[valid] /= total[valid].astype(np.float32)
    out[~valid] = 0.0
    return out.astype(np.float32)


def row_cos(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def circular_emd(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.roll(np.asarray(left, dtype=np.float64), -96, axis=1)
    y = np.roll(np.asarray(right, dtype=np.float64), -96, axis=1)
    return np.mean(np.abs(np.cumsum(x - y, axis=1)), axis=1).astype(np.float32)


def knn_anchor(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    xy: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    tree = cKDTree(xy[bank_ids])
    query_in_bank = np.isin(query_ids, bank_ids)
    search_k = NEIGHBORS + 1 if np.any(query_in_bank) else NEIGHBORS
    distance, local = tree.query(xy[query_ids], k=search_k)
    if distance.ndim == 1:
        distance = distance[:, None]
        local = local[:, None]
    mapped = bank_ids[local]
    donor = np.empty((len(query_ids), NEIGHBORS), dtype=np.int64)
    kept_distance = np.empty((len(query_ids), NEIGHBORS), dtype=np.float64)
    membership = np.zeros(len(xy), dtype=bool)
    membership[bank_ids] = True
    for row, query_id in enumerate(query_ids):
        keep = mapped[row] != query_id if membership[query_id] else np.ones(search_k, bool)
        donor[row] = mapped[row, keep][:NEIGHBORS]
        kept_distance[row] = np.asarray(distance[row])[keep][:NEIGHBORS]
    weight = 1.0 / np.maximum(kept_distance, DISTANCE_FLOOR) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    return normalize(
        np.einsum("qk,qkd->qd", weight, target[donor], optimize=True)
    )


class MultiResolutionGrid(nn.Module):
    def __init__(self, lower: np.ndarray, upper: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("lower", torch.tensor(lower, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(upper - lower, dtype=torch.float32))
        self.tables = nn.ParameterList()
        self.level_gate = nn.Parameter(torch.full((len(RESOLUTIONS),), -1.5))
        for resolution in RESOLUTIONS:
            table = nn.Parameter(torch.empty((resolution + 1) ** 2, GRID_DIM))
            nn.init.normal_(table, mean=0.0, std=0.01)
            self.tables.append(table)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp((xy - self.lower) / self.scale, 0.0, 1.0)
        output: list[torch.Tensor] = []
        gates = torch.sigmoid(self.level_gate)
        for level, (resolution, table) in enumerate(zip(RESOLUTIONS, self.tables)):
            coordinate = normalized * resolution
            base = torch.clamp(torch.floor(coordinate).long(), 0, resolution - 1)
            fraction = coordinate - base.float()
            x0, y0 = base[:, 0], base[:, 1]
            x1, y1 = x0 + 1, y0 + 1
            stride = resolution + 1
            v00 = table[x0 * stride + y0]
            v10 = table[x1 * stride + y0]
            v01 = table[x0 * stride + y1]
            v11 = table[x1 * stride + y1]
            fx, fy = fraction[:, :1], fraction[:, 1:2]
            value = (
                (1.0 - fx) * (1.0 - fy) * v00
                + fx * (1.0 - fy) * v10
                + (1.0 - fx) * fy * v01
                + fx * fy * v11
            )
            output.append(gates[level] * value)
        return torch.cat(output, dim=1)

    def total_variation(self) -> torch.Tensor:
        value = torch.zeros((), device=self.level_gate.device)
        for resolution, table in zip(RESOLUTIONS, self.tables):
            image = table.reshape(resolution + 1, resolution + 1, GRID_DIM)
            value = value + (image[1:] - image[:-1]).square().mean()
            value = value + (image[:, 1:] - image[:, :-1]).square().mean()
        return value


class GlobalDField(nn.Module):
    def __init__(self, node_dim: int, lower: np.ndarray, upper: np.ndarray) -> None:
        super().__init__()
        self.grid = MultiResolutionGrid(lower, upper)
        self.anchor_encoder = nn.Sequential(
            nn.Conv1d(1, 24, 7, padding=3, padding_mode="circular"),
            nn.GELU(),
            nn.Conv1d(24, 40, 5, stride=2, padding=2, padding_mode="circular"),
            nn.GELU(),
            nn.Conv1d(40, 48, 5, stride=2, padding=2, padding_mode="circular"),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(24),
            nn.Flatten(),
            nn.Linear(48 * 24, 192),
            nn.GELU(),
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, 192), nn.LayerNorm(192), nn.GELU(),
            nn.Linear(192, 128), nn.GELU(),
        )
        positional_dim = 4 * len(FOURIER_FREQUENCIES)
        field_dim = len(RESOLUTIONS) * GRID_DIM + positional_dim
        self.residual = nn.Sequential(
            nn.Linear(192 + 128 + field_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(512, 384),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(384, 192),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def positional(self, xy: torch.Tensor) -> torch.Tensor:
        normalized = (xy - self.grid.lower) / self.grid.scale
        values: list[torch.Tensor] = []
        for frequency in FOURIER_FREQUENCIES:
            angle = 2.0 * math.pi * frequency * normalized
            values.extend((torch.sin(angle), torch.cos(angle)))
        return torch.cat(values, dim=1)

    def forward(
        self, xy: torch.Tensor, node: torch.Tensor, anchor: torch.Tensor
    ) -> torch.Tensor:
        anchor_hidden = self.anchor_encoder(
            torch.sqrt(torch.clamp(anchor, min=0.0))[:, None]
        )
        feature = torch.cat(
            [
                anchor_hidden,
                self.node_encoder(node),
                self.grid(xy),
                self.positional(xy),
            ],
            dim=1,
        )
        residual = RESIDUAL_BOUND * torch.tanh(self.residual(feature))
        return F.softmax(torch.log(torch.clamp(anchor, min=1e-10)) + residual, dim=1)


def loss_function(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    cosine = 1.0 - F.cosine_similarity(prediction, truth, dim=1).mean()
    kl = torch.sum(
        truth * (torch.log(torch.clamp(truth, min=1e-10)) - torch.log(torch.clamp(prediction, min=1e-10))),
        dim=1,
    ).mean()
    cdf = torch.mean(torch.abs(torch.cumsum(prediction - truth, dim=1)))
    return cosine + 0.04 * kl + 0.15 * cdf


@torch.no_grad()
def predict(
    model: GlobalDField,
    xy: np.ndarray,
    node: np.ndarray,
    anchor: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(xy), 512):
        end = min(start + 512, len(xy))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            value = model(
                torch.from_numpy(xy[start:end].astype(np.float32)).to(device),
                torch.from_numpy(node[start:end]).to(device),
                torch.from_numpy(anchor[start:end]).to(device),
            )
        output.append(value.float().cpu().numpy())
    return normalize(np.concatenate(output))


def train_model(
    xy: np.ndarray,
    node: np.ndarray,
    anchor: np.ndarray,
    truth: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    seed: int,
    device: torch.device,
) -> tuple[GlobalDField, list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GlobalDField(node.shape[1], bounds[0], bounds[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    xy_tensor = torch.from_numpy(xy.astype(np.float32)).to(device)
    node_tensor = torch.from_numpy(node).to(device)
    anchor_tensor = torch.from_numpy(anchor).to(device)
    truth_tensor = torch.from_numpy(truth).to(device)
    rows = np.arange(len(xy), dtype=np.int64)
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        order = rng.permutation(rows)
        for start in range(0, len(order), BATCH_SIZE):
            batch_np = order[start : start + BATCH_SIZE]
            batch = torch.from_numpy(batch_np).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model(
                    xy_tensor[batch], node_tensor[batch], anchor_tensor[batch]
                )
                data_loss = loss_function(prediction, truth_tensor[batch])
                loss = data_loss + TV_WEIGHT * model.grid.total_variation()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(data_loss.detach().cpu())
        if epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS:
            training_prediction = predict(model, xy, node, anchor, device)
            history.append(
                {
                    "epoch": epoch,
                    "loss": total / math.ceil(len(rows) / BATCH_SIZE),
                    "train_cosine": float(np.mean(row_cos(training_prediction, truth))),
                }
            )
            print(
                f"v417 seed={seed} epoch={epoch} train_cos={history[-1]['train_cosine']:.6f}",
                flush=True,
            )
    return model, history


def run_fold(fold: int, device: torch.device) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    targets = np.load(v412.TARGET_CACHE)
    target = normalize(targets["d"])
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    xy = train_pos[:, :2]
    node, _ = v257.build_node_features(xy, test_pos[:, :2])
    folds = v315.disjoint_folds(valid_ids, xy)
    query_ids = folds[fold]
    fit_ids = np.setdiff1d(valid_ids, query_ids)
    fit_anchor = knn_anchor(fit_ids, fit_ids, xy, target)
    outer_anchor = knn_anchor(query_ids, fit_ids, xy, target)
    mean = node[fit_ids].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = node[fit_ids].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    fit_node = ((node[fit_ids] - mean) / std).astype(np.float32)
    outer_node = ((node[query_ids] - mean) / std).astype(np.float32)
    all_xy = np.concatenate([xy, test_pos[:, :2]], axis=0)
    bounds = (
        all_xy.min(axis=0).astype(np.float32) - 1.0,
        all_xy.max(axis=0).astype(np.float32) + 1.0,
    )
    model, history = train_model(
        xy[fit_ids],
        fit_node,
        fit_anchor,
        target[fit_ids],
        bounds,
        41700 + fold,
        device,
    )
    prediction = predict(
        model, xy[query_ids], outer_node, outer_anchor, device
    )
    truth = target[query_ids]
    base_cos = row_cos(outer_anchor, truth)
    pred_cos = row_cos(prediction, truth)
    base_emd = circular_emd(outer_anchor, truth)
    pred_emd = circular_emd(prediction, truth)
    report = {
        "fold": fold,
        "fit_count": int(len(fit_ids)),
        "query_count": int(len(query_ids)),
        "base_cosine": float(np.mean(base_cos)),
        "prediction_cosine": float(np.mean(pred_cos)),
        "cosine_delta": float(np.mean(pred_cos - base_cos)),
        "base_circular_emd": float(np.mean(base_emd)),
        "prediction_circular_emd": float(np.mean(pred_emd)),
        "emd_reduction": float(1.0 - np.mean(pred_emd) / np.mean(base_emd)),
        "point_win_fraction": float(np.mean(pred_cos > base_cos)),
        "history": history,
        "leakage_assertions": {
            "query_in_fit": int(np.intersect1d(query_ids, fit_ids).size),
            "query_truth_used_for_training_or_selection": False,
            "fixed_epoch_and_architecture": True,
        },
    }
    (REPORT_DIR / f"{VERSION}_fold{fold}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        OUT_DIR / f"v417_fold{fold}.npz",
        ids=query_ids,
        baseline=outer_anchor,
        truth=truth,
        prediction=prediction,
        point_delta=pred_cos - base_cos,
    )
    print(
        f"v417 fold={fold} delta={report['cosine_delta']:+.6f} "
        f"emd={report['emd_reduction']:+.2%}",
        flush=True,
    )


def aggregate() -> None:
    rows = [
        json.loads((REPORT_DIR / f"{VERSION}_fold{fold}.json").read_text(encoding="utf-8"))
        for fold in range(5)
    ]
    parts = [np.load(OUT_DIR / f"v417_fold{fold}.npz") for fold in range(5)]
    ids = np.concatenate([part["ids"] for part in parts])
    if len(ids) != 1984 or len(np.unique(ids)) != 1984:
        raise AssertionError("v417 OOF IDs are not a unique 1984-point cover")
    delta = np.asarray([row["cosine_delta"] for row in rows], dtype=np.float64)
    emd = np.asarray([row["emd_reduction"] for row in rows], dtype=np.float64)
    passed = bool(
        np.mean(delta) >= MIN_D_DELTA
        and np.sum(delta > 0.0) >= MIN_FOLD_WINS
        and np.mean(emd) >= 0.0
    )
    np.savez_compressed(
        OOF_CACHE,
        ids=ids,
        fold=np.concatenate([
            np.full(len(parts[fold]["ids"]), fold, dtype=np.int16) for fold in range(5)
        ]),
        baseline=np.concatenate([part["baseline"] for part in parts]),
        truth=np.concatenate([part["truth"] for part in parts]),
        prediction=np.concatenate([part["prediction"] for part in parts]),
        point_delta=np.concatenate([part["point_delta"] for part in parts]),
    )
    report = {
        "method": VERSION,
        "status": "official_oof_passed" if passed else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Global normalized D is a smoother planar physical field than fine whole-template "
            "identity and can be learned as a bounded residual over a strict KNN anchor."
        ),
        "compliance": {
            "uses": [
                "official train H normalized D",
                "official train/test coordinates",
                "official pointcloud-derived node features",
            ],
            "does_not_use": [
                "synthetic targets or test truth",
                "v212 output or teacher",
                "online labels or Sionna path coefficients",
            ],
        },
        "parameters": {
            "resolutions": list(RESOLUTIONS),
            "grid_dim": GRID_DIM,
            "epochs": EPOCHS,
            "residual_bound": RESIDUAL_BOUND,
            "tv_weight": TV_WEIGHT,
            "five_gpu_execution": True,
        },
        "fold_rows": rows,
        "aggregate": {
            "cosine_delta": summarize(delta),
            "positive_folds": int(np.sum(delta > 0.0)),
            "mean_emd_reduction": float(np.mean(emd)),
        },
        "selection": {
            "minimum_cosine_delta": MIN_D_DELTA,
            "minimum_positive_folds": MIN_FOLD_WINS,
            "passed": passed,
            "next_action": (
                "train full field, freeze test D, then evaluate local replay"
                if passed
                else "do not deploy this field"
            ),
        },
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if world_size != 5:
            raise ValueError("v417 distributed mode requires five workers")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        run_fold(local_rank, torch.device("cuda", local_rank))
        dist.barrier()
        if local_rank == 0:
            aggregate()
        dist.barrier()
        dist.destroy_process_group()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for fold in range(5):
            run_fold(fold, device)
        aggregate()


if __name__ == "__main__":
    main()

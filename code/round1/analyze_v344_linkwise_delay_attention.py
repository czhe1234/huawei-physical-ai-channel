#!/usr/bin/env python3
"""Strict official-only deformable attention for scorer-aligned global PDP."""

from __future__ import annotations

import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v208_strict_historical_replay as v208
import analyze_v254_route_topology_spectral_field as v254
import analyze_v274_planar_barrier_attention as v274
import analyze_v337_componentwise_deformable_attention as v337
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 344
VERSION = "v344_linkwise_delay_attention"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS_OUT = OUT_DIR / "v344_linkwise_delay_targets.npz"
OUTPUT = OUT_DIR / "v344_Round1_Test_Channel.npy"
SOURCE = PATHS.cache_dir(282) / "v282_Round1_Test_Channel.npy"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"

OUTER_SEEDS = (344, 1344, 2344, 3344, 4344)
OUTER_COUNT = 320
INNER_COUNT = 240
EPOCHS = 25
BATCH_SIZE = 24
FINAL_SEEDS = (34401, 34431, 34461)
MIN_OFFICIAL_DELTA = 0.01
ALPHA_GRID = np.linspace(0.0, 1.5, 13, dtype=np.float32)
N_DELAY = 192
OFFSETS = tuple(range(-4, 5))
FLOOR = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(value: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-30)
    return out.astype(np.float32)


def link_global_d(path) -> np.ndarray:
    h = np.load(path, mmap_mode="r")
    output = np.empty((len(h), N_DELAY), dtype=np.float32)
    for start in range(0, len(h), 8):
        end = min(start + 8, len(h))
        block = np.asarray(h[start:end], dtype=np.complex64)
        power = np.abs(np.fft.ifft(block, axis=-1)).astype(np.float32) ** 2
        power /= np.maximum(power.sum(axis=-1, keepdims=True), 1e-30)
        output[start:end] = power.sum(axis=(1, 2), dtype=np.float64).astype(np.float32)
    return normalize(output)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1, dtype=np.float64)
    norm = np.sqrt(
        np.sum(a * a, axis=1, dtype=np.float64)
        * np.sum(b * b, axis=1, dtype=np.float64)
    )
    return (numerator / np.maximum(norm, 1e-30)).astype(np.float32)


def basis() -> np.ndarray:
    x = 2.0 * np.pi * np.arange(N_DELAY, dtype=np.float32) / N_DELAY
    channels = [np.ones(N_DELAY, dtype=np.float32)]
    for order in range(1, 7):
        channels.extend([np.sin(order * x), np.cos(order * x)])
    return np.stack(channels).astype(np.float32)


def base_prediction(data: v274.ModelSet, target: np.ndarray) -> np.ndarray:
    return normalize(
        np.sum(
            data.base_weights[:, :, None] * target[data.neighbor_ids],
            axis=1,
            dtype=np.float64,
        )
    )


class DelayAttention(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        latent = 96
        spectral_basis = basis()
        self.query = nn.Sequential(
            nn.Linear(node_dim, 192), nn.SiLU(), nn.LayerNorm(192), nn.Linear(192, 64)
        )
        self.neighbor = nn.Sequential(
            nn.Linear(node_dim, 192), nn.SiLU(), nn.LayerNorm(192), nn.Linear(192, 64)
        )
        self.edge = nn.Sequential(
            nn.Linear(edge_dim, 96), nn.SiLU(), nn.LayerNorm(96), nn.Linear(96, 64)
        )
        self.trunk = nn.Sequential(
            nn.Linear(256, 192), nn.SiLU(), nn.Dropout(0.04), nn.Linear(192, latent), nn.SiLU()
        )
        self.coeff = nn.Linear(latent, len(spectral_basis))
        self.offset = nn.Linear(latent, len(OFFSETS))
        self.mass = nn.Linear(latent, len(OFFSETS))
        self.gate = nn.Sequential(nn.Linear(64 + latent, 96), nn.SiLU(), nn.Linear(96, 1))
        for layer in (self.coeff, self.offset, self.mass, self.gate[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.gate[-1].bias, -0.55)
        self.register_buffer("spectral_basis", torch.from_numpy(spectral_basis))
        self.register_buffer(
            "offset_prior",
            torch.tensor([-6.0 * abs(value) for value in OFFSETS], dtype=torch.float32),
        )

    def forward(
        self,
        query_node: torch.Tensor,
        neighbor_node: torch.Tensor,
        edge: torch.Tensor,
        base_weights: torch.Tensor,
        neighbor_d: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q = self.query(query_node)
        n = self.neighbor(neighbor_node)
        e = self.edge(edge)
        qn = q[:, None, :].expand_as(n)
        hidden = self.trunk(torch.cat([qn, n, qn * n, e], dim=2))
        component = torch.einsum(
            "bkc,cd->bkd", 2.5 * torch.tanh(self.coeff(hidden)), self.spectral_basis
        )
        offset_logit = self.offset_prior[None, None, :] + 3.0 * torch.tanh(self.offset(hidden))
        mass_gate = 0.9 * torch.tanh(self.mass(hidden))
        candidates = torch.stack(
            [torch.roll(neighbor_d, shifts=value, dims=2) for value in OFFSETS], dim=2
        )
        log_mass = torch.tanh((torch.log(candidates.clamp_min(1e-9)) + 7.0) / 4.0)
        logits = (
            torch.log(base_weights.clamp_min(1e-12))[:, :, None, None]
            + offset_logit[:, :, :, None]
            + component[:, :, None, :]
            + mass_gate[:, :, :, None] * log_mass
        )
        batch = len(query_node)
        weights = torch.softmax(logits.reshape(batch, -1, N_DELAY), dim=1)
        proposal = torch.sum(weights * candidates.reshape(batch, -1, N_DELAY), dim=1)
        anchor = torch.sum(base_weights[:, :, None] * neighbor_d, dim=1)
        pooled = torch.sum(base_weights[:, :, None] * hidden, dim=1)
        gate = 0.92 * torch.sigmoid(self.gate(torch.cat([q, pooled], dim=1)))
        prediction = anchor + gate * (proposal - anchor)
        prediction = prediction.clamp_min(0.0)
        prediction /= prediction.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return prediction, {"gate": gate[:, 0]}


@torch.no_grad()
def predict(
    model: DelayAttention,
    data: v274.ModelSet,
    node_raw: np.ndarray,
    stats: v274.Standardizer,
    target: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    node = np.clip((node_raw - stats.node_mean) / stats.node_std, -8.0, 8.0).astype(np.float32)
    output = np.empty((len(data.query_ids), N_DELAY), dtype=np.float32)
    gates = np.empty(len(data.query_ids), dtype=np.float32)
    for start in range(0, len(data.query_ids), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(data.query_ids))
        ids = data.neighbor_ids[start:end]
        block, diagnostics = model(
            torch.from_numpy(data.query_node[start:end]).to(device),
            torch.from_numpy(node[ids]).to(device),
            torch.from_numpy(data.edge[start:end]).to(device),
            torch.from_numpy(data.base_weights[start:end]).to(device),
            torch.from_numpy(target[ids]).to(device),
        )
        output[start:end] = block.cpu().numpy()
        gates[start:end] = diagnostics["gate"].cpu().numpy()
    return output, gates


def train(
    train_set: v274.ModelSet,
    validation: v274.ModelSet | None,
    node_raw: np.ndarray,
    stats: v274.Standardizer,
    target: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[DelayAttention, dict[str, object]]:
    set_seed(seed)
    model = DelayAttention(node_raw.shape[1], train_set.edge.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    node = np.clip((node_raw - stats.node_mean) / stats.node_std, -8.0, 8.0).astype(np.float32)
    rng = np.random.default_rng(seed)
    best_epoch = epochs
    best_score = -np.inf
    best_state = None
    if validation is not None:
        initial, _ = predict(model, validation, node_raw, stats, target, device)
        best_score = float(np.mean(row_cos(initial, target[validation.query_ids])))
        best_epoch = 0
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for start in range(0, len(train_set.query_ids), BATCH_SIZE):
            rows = rng.choice(len(train_set.query_ids), size=min(BATCH_SIZE, len(train_set.query_ids)), replace=False)
            ids = train_set.neighbor_ids[rows]
            prediction, _ = model(
                torch.from_numpy(train_set.query_node[rows]).to(device),
                torch.from_numpy(node[ids]).to(device),
                torch.from_numpy(train_set.edge[rows]).to(device),
                torch.from_numpy(train_set.base_weights[rows]).to(device),
                torch.from_numpy(target[ids]).to(device),
            )
            truth = torch.from_numpy(target[train_set.query_ids[rows]]).to(device)
            loss = 0.94 * (1.0 - F.cosine_similarity(prediction, truth, dim=1).mean())
            loss += 0.06 * (1.0 - F.cosine_similarity(torch.sqrt(prediction.clamp_min(1e-12)), torch.sqrt(truth.clamp_min(1e-12)), dim=1).mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            item: dict[str, object] = {"epoch": epoch, "loss": float(np.mean(losses))}
            if validation is not None:
                value, _ = predict(model, validation, node_raw, stats, target, device)
                score = float(np.mean(row_cos(value, target[validation.query_ids])))
                item["validation"] = score
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append(item)
            print(f"v344 seed={seed} epoch={epoch}/{epochs} loss={item['loss']:.6f} val={item.get('validation')}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"seed": seed, "best_epoch": best_epoch, "best_score": best_score, "history": history}


def build_sets(query_ids, bank_ids, geometry, node_raw):
    all_pos, signatures, blocker_count, _, footprint, z_max, x_min, y_min, resolution = geometry
    raw_train = v274.build_raw_set(bank_ids, bank_ids, all_pos, signatures, blocker_count, footprint, z_max, x_min, y_min, resolution)
    stats = v274.fit_standardizer(node_raw, bank_ids, raw_train)
    train_set = v274.apply_standardizer(raw_train, node_raw, stats)
    raw_query = v274.build_raw_set(query_ids, bank_ids, all_pos, signatures, blocker_count, footprint, z_max, x_min, y_min, resolution)
    query_set = v274.apply_standardizer(raw_query, node_raw, stats)
    return train_set, query_set, stats


def project_linkwise(target_d: np.ndarray) -> dict[str, np.ndarray]:
    source = np.load(SOURCE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    before, after = [], []
    for start in range(0, len(source), 8):
        end = min(start + 8, len(source))
        h = np.asarray(source[start:end], dtype=np.complex64)
        cir = np.fft.ifft(h, axis=-1)
        link_power = np.sum(np.abs(cir) ** 2, axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
        for _ in range(8):
            power = np.abs(cir).astype(np.float32) ** 2
            normalized = power / np.maximum(power.sum(axis=-1, keepdims=True), 1e-30)
            current = normalize(normalized.sum(axis=(1, 2), dtype=np.float64).astype(np.float32))
            ratio = np.maximum(target_d[start:end], FLOOR) / np.maximum(current, FLOOR)
            cir *= np.sqrt(ratio)[:, None, None, :]
            current_link = np.sum(np.abs(cir) ** 2, axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
            cir *= np.sqrt(link_power / np.maximum(current_link, 1e-30))
        h1 = np.fft.fft(cir, axis=-1).astype(np.complex64)
        output[start:end] = h1
        output.flush()
        before.append(current)
        power = np.abs(cir).astype(np.float32) ** 2
        normalized = power / np.maximum(power.sum(axis=-1, keepdims=True), 1e-30)
        after.append(normalize(normalized.sum(axis=(1, 2), dtype=np.float64).astype(np.float32)))
    before_d = np.concatenate(before)
    after_d = np.concatenate(after)
    return {"actual_l1": np.sum(np.abs(after_d - before_d), axis=1), "target_cos": row_cos(after_d, target_d)}


def main() -> None:
    target = link_global_d(TRAIN_H)
    valid_ids = np.flatnonzero(target.sum(axis=1) > 1e-20).astype(np.int64)
    geometry = v274.load_geometry()
    all_pos, signatures, blocker_count, scene_features = geometry[:4]
    node_raw = v274.build_node_features(all_pos, signatures, blocker_count, scene_features)
    train_pos = all_pos[: len(target)]
    test_pos = all_pos[len(target) :]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    ids_parts = []
    base_parts = []
    model_parts = []
    selected_epochs = []
    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = v254.choose_seeded_matched_indices(train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed)
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = np.intersect1d(v254.choose_seeded_matched_indices(train_pos[:, :2], test_pos[:, :2], INNER_COUNT, seed + 50_000), fit_ids)
        inner_fit = np.setdiff1d(fit_ids, inner_ids)
        inner_train, inner_val, inner_stats = build_sets(inner_ids, inner_fit, geometry, node_raw)
        _, inner_report = train(inner_train, inner_val, node_raw, inner_stats, target, seed, EPOCHS, device)
        epoch = int(inner_report["best_epoch"])
        selected_epochs.append(epoch)
        outer_train, outer_val, outer_stats = build_sets(outer_ids, fit_ids, geometry, node_raw)
        base = base_prediction(outer_val, target)
        if epoch == 0:
            model_value = base.copy()
            gates = np.zeros(len(outer_ids), dtype=np.float32)
            outer_report = {"best_epoch": 0, "history": []}
        else:
            model, outer_report = train(outer_train, None, node_raw, outer_stats, target, seed + 100_000, epoch, device)
            model_value, gates = predict(model, outer_val, node_raw, outer_stats, target, device)
        base_score = float(np.mean(row_cos(base, target[outer_ids])))
        model_score = float(np.mean(row_cos(model_value, target[outer_ids])))
        rows.append({"fold": fold, "seed": seed, "epoch": epoch, "base": base_score, "model": model_score, "raw_delta": model_score - base_score, "gate": summarize(gates), "inner": inner_report, "outer": outer_report})
        ids_parts.append(outer_ids); base_parts.append(base); model_parts.append(model_value)
        print(f"v344 fold={fold+1}/5 epoch={epoch} delta={model_score-base_score:+.6f}", flush=True)
    ids_all = np.concatenate(ids_parts); base_all = np.concatenate(base_parts); model_all = np.concatenate(model_parts)
    alpha_rows = []
    for alpha in ALPHA_GRID:
        value = normalize(base_all + float(alpha) * (model_all - base_all))
        alpha_rows.append({"alpha": float(alpha), "score": float(np.mean(row_cos(value, target[ids_all])))})
    best = max(alpha_rows, key=lambda item: item["score"])
    alpha = float(best["alpha"])
    fold_delta = []
    for row, ids, base, model_value in zip(rows, ids_parts, base_parts, model_parts):
        value = normalize(base + alpha * (model_value - base))
        delta = float(np.mean(row_cos(value, target[ids]))) - float(row["base"])
        row["blended_delta"] = delta; fold_delta.append(delta)
    mean_delta = float(np.mean(fold_delta)); wins = int(np.sum(np.asarray(fold_delta) > 0))
    selected = bool(alpha > 0 and mean_delta >= MIN_OFFICIAL_DELTA and wins >= 4)
    local_score = None; final_rows = []; projection = None
    if selected:
        test_ids = np.arange(len(target), len(all_pos), dtype=np.int64)
        full_train, test_set, stats = build_sets(test_ids, valid_ids, geometry, node_raw)
        test_base = base_prediction(test_set, target)
        epoch = max(int(np.median([x for x in selected_epochs if x > 0])), 1)
        predictions = []
        for seed in FINAL_SEEDS:
            model, training = train(full_train, None, node_raw, stats, target, seed, epoch, device)
            value, gate = predict(model, test_set, node_raw, stats, target, device)
            predictions.append(value); final_rows.append({"seed": seed, "training": training, "gate": summarize(gate)})
        test_model = normalize(np.mean(predictions, axis=0))
        current = link_global_d(SOURCE)
        target_test = normalize(current + alpha * (test_model - test_base))
        np.savez_compressed(TARGETS_OUT, base=test_base, model=test_model, current=current, target=target_test, alpha=np.asarray(alpha, dtype=np.float32))
        projection = project_linkwise(target_test)
        local_score = v208.score_prediction(OUTPUT)
    report = {"method": VERSION, "status": "strict_complete_local_diagnostic" if selected else "analysis_only_no_candidate", "compliance_boundary": {"uses": ["official train H scorer-aligned link-normalized D", "official train/test coordinates", "official point-cloud scene/barrier features", "strict v282 carrier after official selection"], "does_not_use": ["v211/v212 outputs or labels", "synthetic training labels", "restored scene or RT coefficients", "online score"]}, "parameters": {"outer_seeds": list(OUTER_SEEDS), "epochs": EPOCHS, "offsets": list(OFFSETS), "minimum_delta": MIN_OFFICIAL_DELTA}, "folds": rows, "official_selection": {"alpha": alpha, "alpha_rows": alpha_rows, "fold_delta": fold_delta, "mean_delta": mean_delta, "wins": wins, "selected": selected, "selected_epochs": selected_epochs}, "final_training": final_rows, "projection": {key: summarize(value) for key, value in projection.items()} if projection else None, "local_score": local_score, "outputs": {"prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None, "targets": PATHS.relative(TARGETS_OUT) if TARGETS_OUT.is_file() else None, "report": PATHS.relative(REPORT)}}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["official_selection"], ensure_ascii=False, indent=2), flush=True)
    if local_score is not None: print(json.dumps(local_score, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""v242: hierarchical Sionna-like physical path-energy model.

Path energy is decomposed into learned path-loss, incidence, excess-length,
reflection-plane, ordered-pair, and bounded neural residual terms.  All terms
are supervised only by official train-H NNLS path labels.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
import analyze_v233_official_joint_ad_manifold as v233
import analyze_v236_official_path_pdp_decoder as v236
import analyze_v238_raw_path_conditional_angle as v238
import analyze_v239_nnls_path_energy_decoder as v239
import analyze_v240_plane_family_path_energy as v240
import generate_v215_pointcloud_physics_neural_field as v215
from generate_v87_easy_field_projection import S_SC, summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 242
VERSION = "v242_hierarchical_physical_path_energy"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / "v242_hierarchical_physical_path_energy.json"
TARGET_OUT = OUT_DIR / "v242_test_physical_path_pdp.npy"
ACTION_OUT = OUT_DIR / "v242_hierarchical_physical_path_actions.npz"
BASE_V239 = v239.OUT_DIR / "v239_path_on_v237_Round1_Test_Channel.npy"

TRAIN_EPOCHS = 300
BATCH_SIZE = 128
SEEDS = (24201, 24237, 24271)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def physical_features(prefix: str, indices: np.ndarray | None) -> np.ndarray:
    cache = np.load(v215.PATH_CACHE)

    def take(name: str) -> np.ndarray:
        value = cache[f"{prefix}_{name}"].astype(np.float32)
        return value if indices is None else value[indices]

    return np.stack(
        [
            np.nan_to_num(take("path_length"), nan=400.0, posinf=400.0),
            np.nan_to_num(take("score"), nan=-4.0, posinf=4.0, neginf=-4.0),
            np.clip(np.nan_to_num(take("incidence_cos"), nan=1e-3), 1e-3, 1.0),
            np.maximum(np.nan_to_num(take("excess_length"), nan=300.0), 0.0),
            np.clip(np.nan_to_num(take("edge_margin"), nan=0.0), -1.0, 1.0),
        ],
        axis=2,
    ).astype(np.float32)


class HierarchicalPhysicalPathEnergy(nn.Module):
    def __init__(self, global_dim: int, path_dim: int):
        super().__init__()
        self.plane_coeff = nn.Embedding(v240.PLANE_COUNT, 1, padding_idx=0)
        self.pair_coeff = nn.Embedding(v240.PAIR_COUNT, 1, padding_idx=0)
        self.plane_emb = nn.Embedding(v240.PLANE_COUNT, 10, padding_idx=0)
        self.type_bias = nn.Parameter(torch.zeros(3))
        self.path_loss_raw = nn.Parameter(torch.full((3,), 0.55))
        self.score_coef = nn.Parameter(torch.full((3,), 0.45))
        self.incidence_raw = nn.Parameter(torch.full((3,), -0.35))
        self.excess_raw = nn.Parameter(torch.full((3,), -2.0))
        self.edge_coef = nn.Parameter(torch.full((3,), 0.12))
        self.global_net = nn.Sequential(
            nn.Linear(global_dim, 256), nn.SiLU(), nn.LayerNorm(256), nn.Dropout(0.05), nn.Linear(256, 96), nn.SiLU()
        )
        residual_dim = path_dim + 96 + 20
        self.path_residual = nn.Sequential(
            nn.Linear(residual_dim, 192), nn.SiLU(), nn.LayerNorm(192), nn.Dropout(0.05), nn.Linear(192, 96), nn.SiLU()
        )
        self.energy_residual = nn.Linear(96, 1)
        self.shift_head = nn.Linear(96, len(v239.SHIFTS))
        self.tail_head = nn.Sequential(nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, v239.TAIL_COUNT))
        nn.init.zeros_(self.plane_coeff.weight)
        nn.init.zeros_(self.pair_coeff.weight)
        nn.init.zeros_(self.energy_residual.weight)
        nn.init.zeros_(self.energy_residual.bias)

    def forward(
        self,
        global_features: torch.Tensor,
        path_numeric: torch.Tensor,
        path_type: torch.Tensor,
        path_valid: torch.Tensor,
        raw_delay: torch.Tensor,
        physical: torch.Tensor,
        plane1: torch.Tensor,
        plane2: torch.Tensor,
        pair_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        type_idx = torch.clamp(path_type - 1, 0, 2)
        length = physical[..., 0].clamp_min(1.0)
        score = physical[..., 1]
        incidence = physical[..., 2].clamp_min(1e-3)
        excess = physical[..., 3]
        edge = physical[..., 4]
        path_loss = 1.0 + F.softplus(self.path_loss_raw[type_idx])
        incidence_coef = F.softplus(self.incidence_raw[type_idx])
        excess_coef = F.softplus(self.excess_raw[type_idx])
        physical_logit = (
            self.type_bias[type_idx]
            - path_loss * torch.log(length)
            + self.score_coef[type_idx] * score
            + incidence_coef * torch.log(incidence)
            - excess_coef * (excess / 100.0)
            + self.edge_coef[type_idx] * edge
            + self.plane_coeff(plane1).squeeze(2)
            + self.plane_coeff(plane2).squeeze(2)
            + self.pair_coeff(pair_id).squeeze(2)
        )

        global_latent = self.global_net(global_features)
        global_path = global_latent[:, None, :].expand(-1, path_numeric.shape[1], -1)
        residual_input = torch.cat(
            [path_numeric, global_path, self.plane_emb(plane1), self.plane_emb(plane2)], dim=2
        )
        residual_latent = self.path_residual(residual_input)
        residual = 0.65 * torch.tanh(self.energy_residual(residual_latent).squeeze(2))
        path_logits = physical_logit[..., None] + residual[..., None] + self.shift_head(residual_latent)
        path_logits = path_logits.masked_fill(~path_valid[..., None], -1e9).flatten(1)
        logits = torch.cat([path_logits, self.tail_head(global_latent)], dim=1)
        weights = torch.softmax(logits, dim=1)
        atoms = v239.torch_atoms(raw_delay, path_type, path_valid)
        pred = torch.sum(weights[..., None] * atoms, dim=1)
        pred /= pred.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return pred, weights, residual


@dataclass
class DecoderOutput:
    d: np.ndarray
    weights: np.ndarray
    residual: np.ndarray


@torch.no_grad()
def predict_model(
    model: HierarchicalPhysicalPathEnergy,
    features: v233.FeatureSet,
    raw_delay: np.ndarray,
    physical: np.ndarray,
    family: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
) -> DecoderOutput:
    model.eval()
    p1, p2, pair = family
    atom_count = raw_delay.shape[1] * len(v239.SHIFTS) + v239.TAIL_COUNT
    d = np.empty((len(raw_delay), S_SC), dtype=np.float32)
    weights = np.empty((len(raw_delay), atom_count), dtype=np.float32)
    residual = np.empty(raw_delay.shape, dtype=np.float32)
    for start in range(0, len(raw_delay), 256):
        end = min(start + 256, len(raw_delay))
        pred, weight, resid = model(
            torch.from_numpy(features.global_features[start:end]).to(device),
            torch.from_numpy(features.path_numeric[start:end]).to(device),
            torch.from_numpy(features.path_type[start:end]).to(device),
            torch.from_numpy(features.path_valid[start:end]).to(device),
            torch.from_numpy(raw_delay[start:end]).to(device),
            torch.from_numpy(physical[start:end]).to(device),
            torch.from_numpy(p1[start:end]).to(device),
            torch.from_numpy(p2[start:end]).to(device),
            torch.from_numpy(pair[start:end]).to(device),
        )
        d[start:end] = pred.cpu().numpy()
        weights[start:end] = weight.cpu().numpy()
        residual[start:end] = resid.cpu().numpy()
    return DecoderOutput(d, weights, residual)


def train_model(
    features: v233.FeatureSet,
    raw_delay: np.ndarray,
    physical: np.ndarray,
    family: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_d: np.ndarray,
    target_weights: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    validation: tuple[
        v233.FeatureSet,
        np.ndarray,
        np.ndarray,
        tuple[np.ndarray, np.ndarray, np.ndarray],
        np.ndarray,
        np.ndarray,
    ] | None,
) -> tuple[HierarchicalPhysicalPathEnergy, dict[str, object]]:
    set_seed(seed)
    model = HierarchicalPhysicalPathEnergy(features.global_features.shape[1], features.path_numeric.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.0e-4, weight_decay=3.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    rng = np.random.default_rng(seed)
    p1, p2, pair = family
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(target_d))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            pred, weights, residual = model(
                torch.from_numpy(features.global_features[ids]).to(device),
                torch.from_numpy(features.path_numeric[ids]).to(device),
                torch.from_numpy(features.path_type[ids]).to(device),
                torch.from_numpy(features.path_valid[ids]).to(device),
                torch.from_numpy(raw_delay[ids]).to(device),
                torch.from_numpy(physical[ids]).to(device),
                torch.from_numpy(p1[ids]).to(device),
                torch.from_numpy(p2[ids]).to(device),
                torch.from_numpy(pair[ids]).to(device),
            )
            loss = v239.decoder_loss(
                pred,
                weights,
                torch.from_numpy(target_d[ids]).to(device),
                torch.from_numpy(target_weights[ids]).to(device),
            )
            loss = loss + 0.003 * torch.mean(residual**2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            item: dict[str, object] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            if validation is not None:
                vf, vd, vp, vfamily, vt, vw = validation
                val_out = predict_model(model, vf, vd, vp, vfamily, device)
                model_cos = float(np.mean(v239.row_cos(val_out.d, vt)))
                label_cos = float(np.mean(v239.row_cos(val_out.weights, vw)))
                item.update(
                    {
                        "val_model_cos": model_cos,
                        "val_weight_cos": label_cos,
                        "val_residual": summarize(val_out.residual),
                    }
                )
                if model_cos > best_metric:
                    best_metric = model_cos
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append(item)
            print(f"v242 seed={seed} epoch={epoch}/{epochs} {json.dumps(item)}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_cos": None if not np.isfinite(best_metric) else float(best_metric),
        "history": history,
    }


def run_variant(name: str, source, predicted_d: np.ndarray, alpha: np.ndarray, regimes: np.ndarray, clip: float) -> dict[str, object]:
    current = v225.current_delay_marginals(source)
    target_d = v239.normalize_d((1.0 - alpha[:, None]) * current["d"] + alpha[:, None] * predicted_d)
    targets = {"d": target_d, "du": current["du"], "dh": current["dh"]}
    output = OUT_DIR / f"v242_{name}_Round1_Test_Channel.npy"
    diag = v225.apply_delay_targets(source, output, targets, regimes, clip)
    return {
        "name": name,
        "source": PATHS.relative(source),
        "output": PATHS.relative(output),
        "score": v208.score_prediction(output),
        "alpha": summarize(alpha),
        "target_l1": summarize(v225.l1_rows(target_d, current["d"])),
        "diagnostics": {key: summarize(value) for key, value in diag.items()},
    }


def main() -> None:
    for path in [v239.BASE_V237, BASE_V239, v215.PATH_CACHE, v239.NNLS_CACHE, v238.GEOM_CACHE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_targets, _raw_targets, _log_power, valid_delay = v215.load_targets()
    target_all = v239.normalize_d(clean_targets["d"])
    _support_raw, valid_support, log_power = v238.build_raw_support()
    valid_ids = np.flatnonzero(valid_delay & valid_support).astype(np.int64)
    label_data = np.load(v239.NNLS_CACHE)
    nnls_weights = label_data["weights"].astype(np.float32)
    nnls_recon = label_data["reconstruction"].astype(np.float32)
    support = np.load(v238.OUT_DIR / "v238_train_raw_angle_delay_normalized.npy", mmap_mode="r")
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(v238.GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    bundle = v238.build_bundle(support, valid_ids, log_power, train_pos, test_pos, train_blocked, test_blocked)
    train_ids = bundle["train_ids"]
    val_ids = bundle["val_ids"]

    train_anchor_d = v239.normalize_d(np.asarray(bundle["train_anchor"]).sum(axis=(1, 2)))
    val_anchor_d = v239.normalize_d(np.asarray(bundle["val_anchor"]).sum(axis=(1, 2)))
    train_raw = v236.augment_raw_features(
        v233.raw_features(
            "train", train_ids, train_pos[train_ids], train_blocked[train_ids], bundle["train_anchor"],
            bundle["train_anchor_power"], bundle["w_train"],
        ),
        train_anchor_d,
    )
    val_raw = v236.augment_raw_features(
        v233.raw_features(
            "train", val_ids, train_pos[val_ids], train_blocked[val_ids], bundle["val_anchor"],
            bundle["val_anchor_power"], bundle["w_val"],
        ),
        val_anchor_d,
    )
    train_features, [val_features], _ = v233.standardize_feature_sets(train_raw, [val_raw])
    train_delay = v236.raw_path_delay("train", train_ids)
    val_delay = v236.raw_path_delay("train", val_ids)
    train_physical = physical_features("train", train_ids)
    val_physical = physical_features("train", val_ids)
    all_family = v240.raw_family_ids("train", None)
    train_family = tuple(value[train_ids] for value in all_family)
    val_family_raw = tuple(value[val_ids] for value in all_family)
    val_family = (
        val_family_raw[0], val_family_raw[1], v240.mask_unseen_pairs(val_family_raw[2], train_family[2])
    )
    model, validation_report = train_model(
        train_features,
        train_delay,
        train_physical,
        train_family,
        target_all[train_ids],
        nnls_weights[train_ids],
        SEEDS[0],
        TRAIN_EPOCHS,
        device,
        (val_features, val_delay, val_physical, val_family, target_all[val_ids], nnls_weights[val_ids]),
    )
    val_out = predict_model(model, val_features, val_delay, val_physical, val_family, device)
    val_alpha, calibration = v239.calibrate_alpha(
        val_anchor_d, val_out.d, target_all[val_ids], regime_of(train_blocked[val_ids])
    )
    validation_metrics = {
        "anchor_cos": float(np.mean(v239.row_cos(val_anchor_d, target_all[val_ids]))),
        "nnls_oracle_cos": float(np.mean(v239.row_cos(nnls_recon[val_ids], target_all[val_ids]))),
        "model_cos": float(np.mean(v239.row_cos(val_out.d, target_all[val_ids]))),
        "model_weight_cos": float(np.mean(v239.row_cos(val_out.weights, nnls_weights[val_ids]))),
        "residual": summarize(val_out.residual),
        "calibration": calibration,
    }
    best_epoch = max(10, int(validation_report["best_epoch"]))
    del model, val_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_anchor_d = v239.normalize_d(np.asarray(bundle["final_anchor"]).sum(axis=(1, 2)))
    test_anchor_d = v239.normalize_d(np.asarray(bundle["test_anchor"]).sum(axis=(1, 2)))
    final_raw = v236.augment_raw_features(
        v233.raw_features(
            "train", valid_ids, train_pos[valid_ids], train_blocked[valid_ids], bundle["final_anchor"],
            bundle["final_anchor_power"], bundle["w_final"],
        ),
        final_anchor_d,
    )
    test_raw = v236.augment_raw_features(
        v233.raw_features(
            "test", None, test_pos, test_blocked, bundle["test_anchor"], bundle["test_anchor_power"], bundle["w_test"],
        ),
        test_anchor_d,
    )
    final_features, [test_features], final_stats = v233.standardize_feature_sets(final_raw, [test_raw])
    final_delay = v236.raw_path_delay("train", valid_ids)
    test_delay = v236.raw_path_delay("test", None)
    final_physical = physical_features("train", valid_ids)
    test_physical = physical_features("test", None)
    final_family = tuple(value[valid_ids] for value in all_family)
    test_family_raw = v240.raw_family_ids("test", None)
    test_family = (
        test_family_raw[0], test_family_raw[1], v240.mask_unseen_pairs(test_family_raw[2], final_family[2])
    )
    seed_outputs = []
    final_training = []
    parameter_snapshots = []
    for seed in SEEDS:
        final_model, info = train_model(
            final_features,
            final_delay,
            final_physical,
            final_family,
            target_all[valid_ids],
            nnls_weights[valid_ids],
            seed,
            best_epoch,
            device,
            None,
        )
        torch.save(
            {"state_dict": final_model.state_dict(), "feature_stats": final_stats, "seed": seed, "epochs": best_epoch},
            CHECKPOINT_DIR / f"seed_{seed}.pt",
        )
        seed_outputs.append(predict_model(final_model, test_features, test_delay, test_physical, test_family, device))
        parameter_snapshots.append(
            {
                "path_loss": (1.0 + F.softplus(final_model.path_loss_raw)).detach().cpu().numpy().tolist(),
                "score_coef": final_model.score_coef.detach().cpu().numpy().tolist(),
                "incidence_coef": F.softplus(final_model.incidence_raw).detach().cpu().numpy().tolist(),
                "excess_coef": F.softplus(final_model.excess_raw).detach().cpu().numpy().tolist(),
                "edge_coef": final_model.edge_coef.detach().cpu().numpy().tolist(),
            }
        )
        final_training.append(info)
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_d = v239.normalize_d(np.mean(np.stack([item.d for item in seed_outputs]), axis=0))
    test_weights = np.mean(np.stack([item.weights for item in seed_outputs]), axis=0).astype(np.float32)
    residual = np.mean(np.stack([item.residual for item in seed_outputs]), axis=0).astype(np.float32)
    np.save(TARGET_OUT, test_d)
    agreement = np.mean(
        np.stack([
            v239.row_cos(seed_outputs[i].d, seed_outputs[j].d)
            for i in range(len(seed_outputs)) for j in range(i + 1, len(seed_outputs))
        ]), axis=0,
    ).astype(np.float32)
    confidence = np.clip((agreement - 0.88) / 0.115, 0.0, 1.0).astype(np.float32)
    regimes = regime_of(test_blocked)
    alpha_base = np.asarray([val_alpha[str(regime)] for regime in regimes], dtype=np.float32)
    variants = [
        (
            "physical_on_v237", v239.BASE_V237,
            np.clip(np.asarray([{"clear": 0.0, "light": 0.04, "medium": 0.22, "heavy": 0.38}[str(r)] for r in regimes], dtype=np.float32) * confidence, 0.0, 0.42),
            1.04,
        ),
        (
            "physical_on_v239", BASE_V239,
            np.clip(np.asarray([{"clear": 0.0, "light": 0.025, "medium": 0.14, "heavy": 0.24}[str(r)] for r in regimes], dtype=np.float32) * confidence, 0.0, 0.28),
            1.03,
        ),
        (
            "calibrated_on_v237", v239.BASE_V237,
            np.clip(alpha_base * (0.36 + 0.64 * confidence), 0.0, 0.92),
            1.04,
        ),
        (
            "physical_heavy_overdrive", v239.BASE_V237,
            np.clip(np.asarray([{"clear": 0.0, "light": 0.02, "medium": 0.32, "heavy": 0.56}[str(r)] for r in regimes], dtype=np.float32) * (0.44 + 0.56 * confidence), 0.0, 0.62),
            1.10,
        ),
    ]
    results = [run_variant(name, source, test_d, alpha.astype(np.float32), regimes, clip) for name, source, alpha, clip in variants]
    best = max(results, key=lambda item: item["score"]["score_traceback_leaked"])
    np.savez_compressed(
        ACTION_OUT,
        weights=test_weights,
        residual=residual,
        agreement=agreement,
        confidence=confidence,
        predicted_vs_anchor=v239.row_cos(test_d, test_anchor_d),
        predicted_vs_v239=v239.row_cos(test_d, v225.current_delay_marginals(BASE_V239)["d"]),
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": "A hierarchical Sionna-like path-loss law generalizes path energy better than global black-box prediction or same-family label averaging.",
        "compliance_boundary": {
            "generation_uses": [
                "official train H NNLS path-energy labels",
                "official train/test coordinates",
                "official point-cloud path length, incidence, excess length, plane IDs and path candidates",
                "v237/v239 strict official-data carriers",
            ],
            "generation_does_not_use": [
                "Sionna generated channel labels", "v212/v217/v221/v222 teacher outputs",
                "synthetic replay truth", "hidden online labels", "raw RT H",
            ],
        },
        "training": {
            "device": str(device),
            "valid_train_count": int(len(valid_ids)),
            "split_train_count": int(len(train_ids)),
            "validation_count": int(len(val_ids)),
            "best_epoch": best_epoch,
            "validation": validation_report,
            "validation_metrics": validation_metrics,
            "alpha_by_regime": val_alpha,
            "final_seeds": list(SEEDS),
            "final_training": final_training,
            "learned_physical_parameters": parameter_snapshots,
        },
        "test_diagnostics": {
            "agreement": summarize(agreement),
            "confidence": summarize(confidence),
            "residual": summarize(residual),
            "predicted_vs_anchor": summarize(v239.row_cos(test_d, test_anchor_d)),
            "predicted_vs_v239": summarize(v239.row_cos(test_d, v225.current_delay_marginals(BASE_V239)["d"])),
        },
        "variants": results,
        "local_gate": {
            "required_score_traceback_leaked": v239.GATE_SCORE,
            "best_variant": best["name"],
            "best_score_traceback_leaked": best["score"]["score_traceback_leaked"],
            "gap_to_gate": best["score"]["score_traceback_leaked"] - v239.GATE_SCORE,
        },
        "outputs": {
            "target": PATHS.relative(TARGET_OUT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

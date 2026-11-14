#!/usr/bin/env python3
"""v240: plane-family embeddings for directly supervised path energy.

The point-cloud path cache contains stable reflection-plane identities shared
by train and test.  v239 ignored those categorical path-family labels.  This
experiment adds first-plane, second-plane, and seen ordered-pair embeddings to
the v239 NNLS-supervised path-energy decoder.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
import analyze_v233_official_joint_ad_manifold as v233
import analyze_v236_official_path_pdp_decoder as v236
import analyze_v238_raw_path_conditional_angle as v238
import analyze_v239_nnls_path_energy_decoder as v239
import generate_v215_pointcloud_physics_neural_field as v215
from generate_v87_easy_field_projection import S_SC, summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 240
VERSION = "v240_plane_family_path_energy"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / "v240_plane_family_path_energy.json"
TARGET_OUT = OUT_DIR / "v240_test_plane_family_pdp.npy"
ACTION_OUT = OUT_DIR / "v240_plane_family_path_energy_actions.npz"
BASE_V239 = v239.OUT_DIR / "v239_path_on_v237_Round1_Test_Channel.npy"

TRAIN_EPOCHS = 240
BATCH_SIZE = 96
SEEDS = (24001, 24029, 24061)
PLANE_COUNT = 81
PAIR_COUNT = PLANE_COUNT * PLANE_COUNT + 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def raw_family_ids(prefix: str, indices: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = np.load(v215.PATH_CACHE)
    p1 = cache[f"{prefix}_plane_id"].astype(np.int64)
    p2 = cache[f"{prefix}_plane_id_2"].astype(np.int64)
    if indices is not None:
        p1 = p1[indices]
        p2 = p2[indices]
    p1 = np.clip(p1 + 1, 0, PLANE_COUNT - 1)
    p2 = np.clip(p2 + 1, 0, PLANE_COUNT - 1)
    pair = 1 + p1 * PLANE_COUNT + p2
    pair[(p1 == 0) & (p2 == 0)] = 0
    return p1.astype(np.int64), p2.astype(np.int64), pair.astype(np.int64)


def mask_unseen_pairs(pair: np.ndarray, seen_pair: np.ndarray) -> np.ndarray:
    seen = np.zeros(PAIR_COUNT, dtype=bool)
    seen[np.unique(seen_pair)] = True
    out = pair.copy()
    out[~seen[out]] = 0
    return out.astype(np.int64)


class PlaneFamilyPathDecoder(nn.Module):
    def __init__(self, global_dim: int, path_dim: int):
        super().__init__()
        d_model = 192
        self.type_emb = nn.Embedding(4, 24)
        self.plane_emb = nn.Embedding(PLANE_COUNT, 24, padding_idx=0)
        self.pair_emb = nn.Embedding(PAIR_COUNT, 16, padding_idx=0)
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, 448), nn.SiLU(), nn.LayerNorm(448), nn.Dropout(0.05), nn.Linear(448, d_model)
        )
        self.path_proj = nn.Sequential(
            nn.Linear(path_dim + 24 + 24 + 24 + 16, 336),
            nn.SiLU(),
            nn.LayerNorm(336),
            nn.Linear(336, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=6,
            dim_feedforward=768,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=5)
        self.path_trunk = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 256), nn.SiLU())
        self.global_trunk = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 256), nn.SiLU())
        self.path_energy = nn.Linear(256, 1)
        self.shift_weight = nn.Linear(256, len(v239.SHIFTS))
        self.tail_weight = nn.Linear(256, v239.TAIL_COUNT)

    def forward(
        self,
        global_features: torch.Tensor,
        path_numeric: torch.Tensor,
        path_type: torch.Tensor,
        path_valid: torch.Tensor,
        raw_delay: torch.Tensor,
        plane1: torch.Tensor,
        plane2: torch.Tensor,
        pair_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.global_proj(global_features).unsqueeze(1)
        path_input = torch.cat(
            [
                path_numeric,
                self.type_emb(path_type),
                self.plane_emb(plane1),
                self.plane_emb(plane2),
                self.pair_emb(pair_id),
            ],
            dim=2,
        )
        path = self.path_proj(path_input)
        tokens = torch.cat([query, path], dim=1)
        mask = torch.cat(
            [torch.zeros((len(path_valid), 1), dtype=torch.bool, device=path_valid.device), ~path_valid], dim=1
        )
        encoded = self.encoder(tokens, src_key_padding_mask=mask)
        global_latent = self.global_trunk(encoded[:, 0])
        path_latent = self.path_trunk(encoded[:, 1:])
        path_logits = self.path_energy(path_latent) + self.shift_weight(path_latent)
        path_logits = path_logits.masked_fill(~path_valid[..., None], -1e9).flatten(1)
        logits = torch.cat([path_logits, self.tail_weight(global_latent)], dim=1)
        weights = torch.softmax(logits, dim=1)
        atoms = v239.torch_atoms(raw_delay, path_type, path_valid)
        pred = torch.sum(weights[..., None] * atoms, dim=1)
        pred /= pred.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return pred, weights


@dataclass
class DecoderOutput:
    d: np.ndarray
    weights: np.ndarray


@torch.no_grad()
def predict_model(
    model: PlaneFamilyPathDecoder,
    features: v233.FeatureSet,
    raw_delay: np.ndarray,
    family: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
) -> DecoderOutput:
    model.eval()
    p1, p2, pair = family
    atom_count = raw_delay.shape[1] * len(v239.SHIFTS) + v239.TAIL_COUNT
    d = np.empty((len(raw_delay), S_SC), dtype=np.float32)
    weights = np.empty((len(raw_delay), atom_count), dtype=np.float32)
    for start in range(0, len(raw_delay), 256):
        end = min(start + 256, len(raw_delay))
        pred, weight = model(
            torch.from_numpy(features.global_features[start:end]).to(device),
            torch.from_numpy(features.path_numeric[start:end]).to(device),
            torch.from_numpy(features.path_type[start:end]).to(device),
            torch.from_numpy(features.path_valid[start:end]).to(device),
            torch.from_numpy(raw_delay[start:end]).to(device),
            torch.from_numpy(p1[start:end]).to(device),
            torch.from_numpy(p2[start:end]).to(device),
            torch.from_numpy(pair[start:end]).to(device),
        )
        d[start:end] = pred.cpu().numpy()
        weights[start:end] = weight.cpu().numpy()
    return DecoderOutput(d, weights)


def train_model(
    features: v233.FeatureSet,
    raw_delay: np.ndarray,
    family: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_d: np.ndarray,
    target_weights: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    validation: tuple[
        v233.FeatureSet,
        np.ndarray,
        tuple[np.ndarray, np.ndarray, np.ndarray],
        np.ndarray,
        np.ndarray,
    ] | None,
) -> tuple[PlaneFamilyPathDecoder, dict[str, object]]:
    set_seed(seed)
    model = PlaneFamilyPathDecoder(features.global_features.shape[1], features.path_numeric.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6.0e-4, weight_decay=2.6e-4)
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
            pred, weights = model(
                torch.from_numpy(features.global_features[ids]).to(device),
                torch.from_numpy(features.path_numeric[ids]).to(device),
                torch.from_numpy(features.path_type[ids]).to(device),
                torch.from_numpy(features.path_valid[ids]).to(device),
                torch.from_numpy(raw_delay[ids]).to(device),
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
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            item: dict[str, object] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            if validation is not None:
                val_features, val_delay, val_family, val_target, val_weights = validation
                val_out = predict_model(model, val_features, val_delay, val_family, device)
                model_cos = float(np.mean(v239.row_cos(val_out.d, val_target)))
                label_cos = float(np.mean(v239.row_cos(val_out.weights, val_weights)))
                item.update({"val_model_cos": model_cos, "val_weight_cos": label_cos})
                if model_cos > best_metric:
                    best_metric = model_cos
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append(item)
            print(f"v240 seed={seed} epoch={epoch}/{epochs} {json.dumps(item)}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_cos": None if not np.isfinite(best_metric) else float(best_metric),
        "history": history,
    }


def run_variant(
    name: str,
    source,
    predicted_d: np.ndarray,
    alpha: np.ndarray,
    regimes: np.ndarray,
    clip: float,
) -> dict[str, object]:
    current = v225.current_delay_marginals(source)
    target_d = v239.normalize_d((1.0 - alpha[:, None]) * current["d"] + alpha[:, None] * predicted_d)
    targets = {"d": target_d, "du": current["du"], "dh": current["dh"]}
    output = OUT_DIR / f"v240_{name}_Round1_Test_Channel.npy"
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
    for path in [v239.BASE_V226, v239.BASE_V237, BASE_V239, v215.PATH_CACHE, v238.GEOM_CACHE, v239.NNLS_CACHE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_targets, _raw_targets, _log_power, valid_delay = v215.load_targets()
    target_all = v239.normalize_d(clean_targets["d"])
    _support_raw, valid_support, log_power = v238.build_raw_support()
    valid = valid_delay & valid_support
    valid_ids = np.flatnonzero(valid).astype(np.int64)
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
    all_train_family = raw_family_ids("train", None)
    train_family = tuple(value[train_ids] for value in all_train_family)
    val_family_raw = tuple(value[val_ids] for value in all_train_family)
    val_family = (
        val_family_raw[0],
        val_family_raw[1],
        mask_unseen_pairs(val_family_raw[2], train_family[2]),
    )
    model, validation_report = train_model(
        train_features,
        train_delay,
        train_family,
        target_all[train_ids],
        nnls_weights[train_ids],
        SEEDS[0],
        TRAIN_EPOCHS,
        device,
        (val_features, val_delay, val_family, target_all[val_ids], nnls_weights[val_ids]),
    )
    val_out = predict_model(model, val_features, val_delay, val_family, device)
    val_alpha, calibration_report = v239.calibrate_alpha(
        val_anchor_d, val_out.d, target_all[val_ids], regime_of(train_blocked[val_ids])
    )
    validation_metrics = {
        "anchor_cos": float(np.mean(v239.row_cos(val_anchor_d, target_all[val_ids]))),
        "nnls_oracle_cos": float(np.mean(v239.row_cos(nnls_recon[val_ids], target_all[val_ids]))),
        "model_cos": float(np.mean(v239.row_cos(val_out.d, target_all[val_ids]))),
        "model_weight_cos": float(np.mean(v239.row_cos(val_out.weights, nnls_weights[val_ids]))),
        "calibration": calibration_report,
        "unseen_val_pair_fraction": float(np.mean((val_family_raw[2] != 0) & (val_family[2] == 0))),
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
    final_family = tuple(value[valid_ids] for value in all_train_family)
    test_family_raw = raw_family_ids("test", None)
    test_family = (
        test_family_raw[0],
        test_family_raw[1],
        mask_unseen_pairs(test_family_raw[2], final_family[2]),
    )
    seed_outputs = []
    final_training = []
    for seed in SEEDS:
        final_model, info = train_model(
            final_features,
            final_delay,
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
        seed_outputs.append(predict_model(final_model, test_features, test_delay, test_family, device))
        final_training.append(info)
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_d = v239.normalize_d(np.mean(np.stack([item.d for item in seed_outputs]), axis=0))
    test_weights = np.mean(np.stack([item.weights for item in seed_outputs]), axis=0).astype(np.float32)
    np.save(TARGET_OUT, test_d)
    agreement = np.mean(
        np.stack([
            v239.row_cos(seed_outputs[i].d, seed_outputs[j].d)
            for i in range(len(seed_outputs)) for j in range(i + 1, len(seed_outputs))
        ]),
        axis=0,
    ).astype(np.float32)
    confidence = np.clip((agreement - 0.88) / 0.115, 0.0, 1.0).astype(np.float32)
    regimes = regime_of(test_blocked)
    alpha_base = np.asarray([val_alpha[str(regime)] for regime in regimes], dtype=np.float32)
    variants = [
        (
            "plane_on_v237",
            v239.BASE_V237,
            np.clip(
                np.asarray([{"clear": 0.0, "light": 0.04, "medium": 0.20, "heavy": 0.34}[str(r)] for r in regimes], dtype=np.float32)
                * confidence,
                0.0,
                0.38,
            ),
            1.03,
        ),
        (
            "plane_on_v239",
            BASE_V239,
            np.clip(
                np.asarray([{"clear": 0.0, "light": 0.025, "medium": 0.12, "heavy": 0.21}[str(r)] for r in regimes], dtype=np.float32)
                * confidence,
                0.0,
                0.24,
            ),
            1.02,
        ),
        (
            "calibrated_on_v237",
            v239.BASE_V237,
            np.clip(alpha_base * (0.38 + 0.62 * confidence), 0.0, 0.90),
            1.04,
        ),
        (
            "heavy_overdrive_on_v237",
            v239.BASE_V237,
            np.clip(
                np.asarray([{"clear": 0.0, "light": 0.02, "medium": 0.28, "heavy": 0.50}[str(r)] for r in regimes], dtype=np.float32)
                * (0.45 + 0.55 * confidence),
                0.0,
                0.55,
            ),
            1.08,
        ),
    ]
    results = [
        run_variant(name, source, test_d, alpha.astype(np.float32), regimes, clip)
        for name, source, alpha, clip in variants
    ]
    best = max(results, key=lambda item: item["score"]["score_traceback_leaked"])
    np.savez_compressed(
        ACTION_OUT,
        agreement=agreement,
        confidence=confidence,
        weights=test_weights,
        predicted_vs_anchor=v239.row_cos(test_d, test_anchor_d),
        predicted_vs_v239=v239.row_cos(test_d, v225.current_delay_marginals(BASE_V239)["d"]),
        test_pair_id=test_family[2],
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": "Stable point-cloud plane identities define transferable reflection families and improve official geometry-to-path-energy mapping.",
        "compliance_boundary": {
            "generation_uses": [
                "official train H PDP",
                "official train/test coordinates",
                "official point-cloud plane IDs and multibounce path candidates",
                "v237/v239 strict official-data carriers",
            ],
            "generation_does_not_use": [
                "Sionna generated channel labels",
                "v212/v217/v221/v222 teacher outputs",
                "synthetic replay truth",
                "hidden online labels",
                "raw RT H",
            ],
        },
        "plane_family": {
            "plane_count_with_padding": PLANE_COUNT,
            "ordered_pair_embedding_count": PAIR_COUNT,
            "test_unseen_pair_fraction": float(np.mean((test_family_raw[2] != 0) & (test_family[2] == 0))),
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
        },
        "test_diagnostics": {
            "agreement": summarize(agreement),
            "confidence": summarize(confidence),
            "predicted_vs_anchor": summarize(v239.row_cos(test_d, test_anchor_d)),
            "predicted_vs_v239": summarize(v239.row_cos(test_d, v225.current_delay_marginals(BASE_V239)["d"])),
            "predicted_path_mass": summarize(test_weights[:, :-v239.TAIL_COUNT].sum(axis=1)),
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

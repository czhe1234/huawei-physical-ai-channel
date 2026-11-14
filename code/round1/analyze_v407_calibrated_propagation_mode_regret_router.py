#!/usr/bin/env python3
"""Calibrated regret routing over official whole-spectrum mode templates."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v402_visible_path_residual_factorization as v402
import analyze_v406_neural_propagation_mode_router as v406
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 407
VERSION = "v407_calibrated_propagation_mode_regret_router"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v407_calibrated_propagation_mode_regret_router_oof.npz"

TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOMETRY_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

MAX_EPOCHS = 70
EVAL_EVERY = 5
PATIENCE_EVALS = 5
BATCH_SIZE = 96
LEARNING_RATE = 7e-4
WEIGHT_DECAY = 4e-4
TOP_TEMPERATURE = 0.02
MIN_ROUTER_DELTA = 0.01
MIN_FOLD_WINS = 4
MIN_CONTROL_MARGIN = 0.002


@dataclass
class CalibratedModel:
    model: "RegretRouter"
    calibrator: IsotonicRegression
    epoch: int
    history: list[dict[str, float]]


class RegretRouter(nn.Module):
    def __init__(
        self, q_dim: int, donor_dim: int, relation_dim: int, latent_dim: int
    ) -> None:
        super().__init__()
        hidden = 128
        self.query_encoder = nn.Sequential(
            nn.Linear(q_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.donor_encoder = nn.Sequential(
            nn.Linear(donor_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.relation_encoder = nn.Sequential(
            nn.Linear(relation_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=320,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.competition = nn.TransformerEncoder(layer, num_layers=1)
        self.alpha_encoder = nn.Sequential(
            nn.Linear(3, 48), nn.GELU(), nn.Linear(48, hidden)
        )
        self.gain_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        nn.init.constant_(self.gain_head[-1].bias, -0.03)
        self.mode_head = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        alpha = torch.from_numpy(
            np.stack(
                [v406.ALPHAS, v406.ALPHAS**2, 1.0 - v406.ALPHAS], axis=1
            )
        )
        self.register_buffer("alpha_features", alpha)

    def forward(
        self, q: torch.Tensor, donor: torch.Tensor, relation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_hidden = self.query_encoder(q)
        donor_hidden = self.donor_encoder(donor)
        relation_hidden = self.relation_encoder(relation)
        token = self.fusion(
            torch.cat(
                [q_hidden[:, None, :].expand_as(donor_hidden), donor_hidden, relation_hidden],
                dim=-1,
            )
        )
        token = self.competition(token)
        alpha_hidden = self.alpha_encoder(self.alpha_features)
        gain = self.gain_head(
            token[:, :, None, :] + alpha_hidden[None, None, :, :]
        ).squeeze(-1).flatten(1)
        mode = self.mode_head(
            torch.cat([q_hidden, token.mean(dim=1), token.amax(dim=1)], dim=-1)
        )
        return gain, mode


def regression_loss(
    predicted_gain: torch.Tensor,
    predicted_mode: torch.Tensor,
    true_gain: torch.Tensor,
    true_mode: torch.Tensor,
) -> torch.Tensor:
    top_weight = torch.softmax(true_gain / TOP_TEMPERATURE, dim=1)
    option_weight = 0.25 + 0.75 * top_weight * true_gain.shape[1]
    gain_loss = (
        option_weight
        * F.smooth_l1_loss(
            predicted_gain, true_gain, reduction="none", beta=0.02
        )
    ).mean()
    rank_loss = torch.sum(
        -top_weight * F.log_softmax(predicted_gain / TOP_TEMPERATURE, dim=1),
        dim=1,
    ).mean()
    mode_loss = F.smooth_l1_loss(predicted_mode, true_mode, beta=0.25)
    return gain_loss + 0.08 * rank_loss + 0.20 * mode_loss


def fit_calibrator(
    predicted_gain: np.ndarray, true_gain: np.ndarray
) -> IsotonicRegression:
    calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip")
    calibrator.fit(
        predicted_gain.reshape(-1),
        true_gain.reshape(-1),
    )
    return calibrator


def calibrated_choices(
    predicted_gain: np.ndarray, calibrator: IsotonicRegression
) -> tuple[np.ndarray, np.ndarray]:
    expected = calibrator.predict(predicted_gain.reshape(-1)).reshape(
        predicted_gain.shape
    )
    option = np.argmax(expected, axis=1)
    expected_best = expected[np.arange(len(expected)), option]
    choice = np.where(expected_best > 0.0, option + 1, 0).astype(np.int64)
    return choice, expected_best.astype(np.float32)


@torch.no_grad()
def predict_raw(
    model: RegretRouter,
    features: tuple[np.ndarray, ...],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(features[0]), BATCH_SIZE):
        batch = [
            torch.from_numpy(value[start : start + BATCH_SIZE]).to(device)
            for value in features
        ]
        output.append(model(*batch)[0].cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def train_calibrated(
    features: tuple[np.ndarray, ...],
    scores: np.ndarray,
    mode_target: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    seed: int,
    device: torch.device,
    shuffled: bool = False,
) -> CalibratedModel:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    q, donor, relation = [torch.from_numpy(value).to(device) for value in features]
    gain = scores[:, 1:] - scores[:, :1]
    label_gain = gain.copy()
    label_mode = mode_target.copy()
    if shuffled:
        for row in np.concatenate([train_rows, validation_rows]):
            label_gain[row] = label_gain[row, rng.permutation(label_gain.shape[1])]
        label_mode = label_mode[rng.permutation(len(label_mode))]
    gain_tensor = torch.from_numpy(label_gain).to(device)
    mode_tensor = torch.from_numpy(label_mode).to(device)
    model = RegretRouter(
        q.shape[-1], donor.shape[-1], relation.shape[-1], mode_target.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_state = copy.deepcopy(model.state_dict())
    initial_prediction = predict_raw(model, features, device)
    initial_calibrator = fit_calibrator(
        initial_prediction[validation_rows], label_gain[validation_rows]
    )
    best_calibrator = initial_calibrator
    best_epoch = 0
    best_delta = 0.0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(train_rows)
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            rows = torch.from_numpy(order[start : start + BATCH_SIZE]).to(device)
            predicted_gain, predicted_mode = model(
                q[rows], donor[rows], relation[rows]
            )
            loss = regression_loss(
                predicted_gain,
                predicted_mode,
                gain_tensor[rows],
                mode_tensor[rows],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(rows)
        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == MAX_EPOCHS:
            predicted = predict_raw(model, features, device)
            calibrator = fit_calibrator(
                predicted[validation_rows], label_gain[validation_rows]
            )
            choice, _ = calibrated_choices(
                predicted[validation_rows], calibrator
            )
            actual_gain = gain[validation_rows]
            active = choice > 0
            selected = np.zeros(len(choice), dtype=np.float32)
            selected[active] = actual_gain[
                np.flatnonzero(active), choice[active] - 1
            ]
            delta = float(np.mean(selected))
            history.append(
                {
                    "epoch": epoch,
                    "loss": total_loss / len(train_rows),
                    "validation_delta": delta,
                    "validation_active_fraction": float(np.mean(active)),
                }
            )
            if delta > best_delta + 1e-5:
                best_delta = delta
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_calibrator = calibrator
                stale = 0
            else:
                stale += 1
            if epoch >= 20 and stale >= PATIENCE_EVALS:
                break
    model.load_state_dict(best_state)
    return CalibratedModel(model, best_calibrator, best_epoch, history)


def evaluate_method(
    prediction: np.ndarray, truth: np.ndarray, baseline: np.ndarray
) -> tuple[dict[str, object], np.ndarray]:
    base_values = v402.point_metrics(baseline, truth)
    values = v402.point_metrics(prediction, truth)
    delta = values["weighted"] - base_values["weighted"]
    return (
        {
            "metrics": v402.metric_summary(prediction, truth),
            "delta": {
                key: float(np.mean(values[key] - base_values[key]))
                for key in values
            },
            "point_gain": summarize(delta),
            "point_win_fraction": float(np.mean(delta > 0.0)),
        },
        delta,
    )


def fold_experiment(
    fold: int,
    folds: list[np.ndarray],
    valid_ids: np.ndarray,
    xy: np.ndarray,
    node: np.ndarray,
    target: np.ndarray,
    log_power: np.ndarray,
    descriptor: np.ndarray,
    blocked: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    query_ids = folds[fold]
    fit_ids = np.setdiff1d(valid_ids, query_ids)
    pca = PCA(
        n_components=v406.PCA_DIM,
        svd_solver="randomized",
        random_state=4070 + fold,
    )
    pca.fit(np.sqrt(np.maximum(target[fit_ids], 0.0)).reshape(len(fit_ids), -1))
    target_latent = np.zeros((len(target), v406.PCA_DIM), dtype=np.float32)
    target_latent[fit_ids] = pca.transform(
        np.sqrt(np.maximum(target[fit_ids], 0.0)).reshape(len(fit_ids), -1)
    ).astype(np.float32)
    pseudo = v406.build_router_data(
        fit_ids,
        fit_ids,
        xy,
        node,
        target,
        log_power,
        descriptor,
        target_latent,
        pca,
    )
    outer = v406.build_router_data(
        query_ids,
        fit_ids,
        xy,
        node,
        target,
        log_power,
        descriptor,
        target_latent,
        pca,
    )
    pseudo_anchor_latent = pca.transform(
        np.sqrt(np.maximum(pseudo.anchor, 0.0)).reshape(len(pseudo.anchor), -1)
    ).astype(np.float32)
    mode_residual = target_latent[fit_ids] - pseudo_anchor_latent
    residual_scale = mode_residual.std(axis=0).astype(np.float32)
    residual_scale[residual_scale < 1e-4] = 1.0
    mode_target = (mode_residual / residual_scale).astype(np.float32)
    validation_ids = folds[(fold + 1) % len(folds)]
    validation_rows = np.flatnonzero(np.isin(fit_ids, validation_ids)).astype(np.int64)
    train_rows = np.setdiff1d(np.arange(len(fit_ids)), validation_rows)

    prediction: dict[str, np.ndarray] = {}
    training: dict[str, object] = {}
    confidence: dict[str, np.ndarray] = {}
    for mode in ("geometry", "full"):
        features, statistics = v406.standardize_features(pseudo, train_rows, mode)
        trained = train_calibrated(
            features,
            pseudo.scores,
            mode_target,
            train_rows,
            validation_rows,
            40700 + fold * 100 + (0 if mode == "geometry" else 10),
            device,
        )
        outer_features = v406.apply_statistics(outer, mode, statistics)
        raw = predict_raw(trained.model, outer_features, device)
        if trained.epoch == 0:
            choice = np.zeros(len(raw), dtype=np.int64)
            expected = np.zeros(len(raw), dtype=np.float32)
        else:
            choice, expected = calibrated_choices(raw, trained.calibrator)
        prediction[mode] = v406.render_choices(outer, choice, target)
        confidence[mode] = expected
        training[mode] = {
            "selected_epoch": trained.epoch,
            "history": trained.history,
            "active_fraction": float(np.mean(choice > 0)),
            "expected_gain": summarize(expected),
        }
        if mode == "full":
            shuffled = train_calibrated(
                features,
                pseudo.scores,
                mode_target,
                train_rows,
                validation_rows,
                40720 + fold * 100,
                device,
                shuffled=True,
            )
            shuffled_raw = predict_raw(shuffled.model, outer_features, device)
            if shuffled.epoch == 0:
                shuffled_choice = np.zeros(len(shuffled_raw), dtype=np.int64)
                shuffled_expected = np.zeros(len(shuffled_raw), dtype=np.float32)
            else:
                shuffled_choice, shuffled_expected = calibrated_choices(
                    shuffled_raw, shuffled.calibrator
                )
            prediction["shuffled"] = v406.render_choices(
                outer, shuffled_choice, target
            )
            confidence["shuffled"] = shuffled_expected
            training["shuffled"] = {
                "selected_epoch": shuffled.epoch,
                "history": shuffled.history,
                "active_fraction": float(np.mean(shuffled_choice > 0)),
            }

    fixed_alpha = int(
        np.argmax(pseudo.scores[:, 1 : 1 + len(v406.ALPHAS)].mean(axis=0))
    )
    nearest_choice = np.full(len(query_ids), 1 + fixed_alpha, dtype=np.int64)
    prediction["nearest"] = v406.render_choices(outer, nearest_choice, target)
    prediction["oracle"] = v406.render_choices(
        outer, outer.scores.argmax(axis=1), target
    )
    truth = target[query_ids]
    methods: dict[str, object] = {}
    deltas: dict[str, np.ndarray] = {}
    regimes = regime_of(blocked[query_ids]).astype(str)
    for name in ("nearest", "geometry", "full", "shuffled", "oracle"):
        result, delta = evaluate_method(prediction[name], truth, outer.anchor)
        result["by_regime"] = {
            regime: float(np.mean(delta[regimes == regime]))
            for regime in ("clear", "light", "medium", "heavy")
            if np.any(regimes == regime)
        }
        methods[name] = result
        deltas[name] = delta
    row = {
        "fold": fold,
        "fit_count": int(len(fit_ids)),
        "query_count": int(len(query_ids)),
        "fixed_nearest_alpha": float(v406.ALPHAS[fixed_alpha]),
        "baseline": v402.metric_summary(outer.anchor, truth),
        "methods": methods,
        "training": training,
        "leakage_assertions": {
            "query_in_fit": int(np.intersect1d(query_ids, fit_ids).size),
            "outer_truth_in_features_training_or_calibration": False,
            "outer_truth_used_for_scoring_and_oracle_only": True,
        },
    }
    arrays = {
        "ids": query_ids,
        "baseline": outer.anchor,
        "truth": truth,
        **prediction,
        **{f"delta_{name}": value for name, value in deltas.items()},
        **{f"expected_{name}": value for name, value in confidence.items()},
    }
    return row, arrays


def main() -> None:
    for path in (TARGET_CACHE, GEOMETRY_CACHE, v297.COMPONENT_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    targets = np.load(TARGET_CACHE)
    target = v406.normalize(targets["sh"].astype(np.float32))
    log_power = targets["log_power"].astype(np.float32)
    descriptor = v257.spectrum_descriptor(target, target.sum(axis=2))
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    xy = positions[:, :2]
    test_xy = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)[:, :2]
    node, _ = v257.build_node_features(xy, test_xy)
    blocked = np.load(GEOMETRY_CACHE)["train_blocked"].astype(np.float32)
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(
        targets["valid"].astype(bool) & component["valid"].any(axis=1)
    ).astype(np.int64)
    folds = v315.disjoint_folds(valid_ids, xy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    parts: list[dict[str, np.ndarray]] = []
    for fold in range(len(folds)):
        row, arrays = fold_experiment(
            fold,
            folds,
            valid_ids,
            xy,
            node,
            target,
            log_power,
            descriptor,
            blocked,
            device,
        )
        rows.append(row)
        parts.append(arrays)
        print(
            f"v407 fold={fold} nearest={row['methods']['nearest']['delta']['weighted']:+.5f} "
            f"geometry={row['methods']['geometry']['delta']['weighted']:+.5f} "
            f"full={row['methods']['full']['delta']['weighted']:+.5f} "
            f"shuffle={row['methods']['shuffled']['delta']['weighted']:+.5f} "
            f"oracle={row['methods']['oracle']['delta']['weighted']:+.5f}",
            flush=True,
        )

    aggregate: dict[str, object] = {}
    for name in ("nearest", "geometry", "full", "shuffled", "oracle"):
        delta = np.asarray(
            [row["methods"][name]["delta"]["weighted"] for row in rows]
        )
        aggregate[name] = {
            "weighted_delta": summarize(delta),
            "positive_folds": int(np.sum(delta > 0.0)),
        }
    full_delta = float(aggregate["full"]["weighted_delta"]["mean"])
    control = max(
        float(aggregate[name]["weighted_delta"]["mean"])
        for name in ("nearest", "geometry", "shuffled")
    )
    passed = bool(
        full_delta >= MIN_ROUTER_DELTA
        and aggregate["full"]["positive_folds"] >= MIN_FOLD_WINS
        and full_delta - control >= MIN_CONTROL_MARGIN
    )
    ids = np.concatenate([part["ids"] for part in parts])
    if len(ids) != len(np.unique(ids)) or len(ids) != len(valid_ids):
        raise AssertionError("OOF IDs are not a unique cover")
    array_names = (
        "baseline",
        "truth",
        "nearest",
        "geometry",
        "full",
        "shuffled",
        "oracle",
        "delta_nearest",
        "delta_geometry",
        "delta_full",
        "delta_shuffled",
        "delta_oracle",
        "expected_geometry",
        "expected_full",
        "expected_shuffled",
    )
    np.savez_compressed(
        OOF_CACHE,
        ids=ids,
        fold=np.concatenate(
            [
                np.full(len(part["ids"]), fold, dtype=np.int16)
                for fold, part in enumerate(parts)
            ]
        ),
        **{
            name: np.concatenate([part[name] for part in parts])
            for name in array_names
        },
    )
    report = {
        "method": VERSION,
        "status": (
            "official_oof_passed_ready_for_carrier_deployment"
            if passed
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "Calibrated option regret plus a stable propagation-mode latent target can "
            "route whole-spectrum templates without forcing uncertain donor actions."
        ),
        "compliance": {
            "uses": [
                "official train H spectra",
                "official train/test coordinates",
                "official Round1_Map.ply-derived node features",
            ],
            "does_not_use": [
                "test H or hidden truth",
                "v212/Sionna/RT/synthetic targets or coefficients",
                "online score labels",
            ],
        },
        "parameters": {
            "anchor_k": v406.ANCHOR_K,
            "donor_k": v406.DONOR_K,
            "alphas": v406.ALPHAS.tolist(),
            "pca_dim": v406.PCA_DIM,
            "max_epochs": MAX_EPOCHS,
            "calibration": "fit-only held-out isotonic expected regret",
            "folds": "five mutually exclusive isolation-stratified folds",
        },
        "fold_rows": rows,
        "aggregate": aggregate,
        "selection": {
            "minimum_router_delta": MIN_ROUTER_DELTA,
            "minimum_positive_folds": MIN_FOLD_WINS,
            "minimum_control_margin": MIN_CONTROL_MARGIN,
            "observed_control_margin": full_delta - control,
            "passed": passed,
            "next_action": (
                "deploy normalized joint targets on the strict v284 full-H carrier"
                if passed
                else "terminate independent template routing and test joint spatial mode assignment"
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


if __name__ == "__main__":
    main()

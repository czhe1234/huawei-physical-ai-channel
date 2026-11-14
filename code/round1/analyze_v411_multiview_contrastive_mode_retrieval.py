#!/usr/bin/env python3
"""Learn view-invariant spectrum identity and retrieve whole-mode templates."""

from __future__ import annotations

import copy
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v402_visible_path_residual_factorization as v402
import analyze_v406_neural_propagation_mode_router as v406
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 411
VERSION = "v411_multiview_contrastive_mode_retrieval"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v411_multiview_contrastive_mode_retrieval_oof.npz"

TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOMETRY_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

EMBED_DIM = 64
TEACHER_EPOCHS = 60
STUDENT_EPOCHS = 80
BATCH_SIZE = 128
TEACHER_LR = 8e-4
STUDENT_LR = 9e-4
TEMPERATURE = 0.07
EVAL_EVERY = 5
PATIENCE_EVALS = 5
ALPHAS = np.asarray([0.25, 0.50, 0.75, 1.00], dtype=np.float32)
MARGIN_THRESHOLDS = np.asarray([0.00, 0.02, 0.05, 0.10], dtype=np.float32)
MIN_WEIGHTED_DELTA = 0.01
MIN_FOLD_WINS = 4
MIN_TEACHER_ORACLE_DELTA = 0.02


class SpectrumEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=(2, 1), padding=(2, 2)),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 2), padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=(2, 2), padding=1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((12, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 12 * 4, 256),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(256, EMBED_DIM),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        x = torch.sqrt(torch.clamp(spectrum, min=0.0))[:, None, :, :]
        return F.normalize(self.head(self.conv(x)), dim=1)


class QueryStudent(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(192, EMBED_DIM),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(feature), dim=1)


def make_views(targets: np.lib.npyio.NpzFile) -> np.ndarray:
    views = [
        v406.normalize(targets["psh"][:, 0].astype(np.float32)),
        v406.normalize(targets["psh"][:, 1].astype(np.float32)),
        v406.normalize(targets["shue"][:, :, :, [0, 2]].sum(axis=3)),
        v406.normalize(targets["shue"][:, :, :, [1, 3]].sum(axis=3)),
        v406.normalize(targets["shv"][:, :, :, 0::2].sum(axis=3)),
        v406.normalize(targets["shv"][:, :, :, 1::2].sum(axis=3)),
    ]
    return np.stack(views, axis=0).astype(np.float32)


def train_teacher(
    fit_ids: np.ndarray,
    views: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[SpectrumEncoder, list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpectrumEncoder().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TEACHER_LR, weight_decay=3e-4
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, TEACHER_EPOCHS + 1):
        model.train()
        order = rng.permutation(fit_ids)
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            left_view = rng.integers(0, len(views), size=len(ids))
            offset = rng.integers(1, len(views), size=len(ids))
            right_view = (left_view + offset) % len(views)
            left = torch.from_numpy(views[left_view, ids]).to(device)
            right = torch.from_numpy(views[right_view, ids]).to(device)
            left_embedding = model(left)
            right_embedding = model(right)
            logits = left_embedding @ right_embedding.T / TEMPERATURE
            label = torch.arange(len(ids), device=device)
            loss = 0.5 * (
                F.cross_entropy(logits, label)
                + F.cross_entropy(logits.T, label)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(ids)
        if epoch == 1 or epoch % 10 == 0 or epoch == TEACHER_EPOCHS:
            history.append({"epoch": epoch, "loss": total_loss / len(fit_ids)})
    return model, history


@torch.no_grad()
def encode_spectra(
    model: SpectrumEncoder,
    spectrum: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(spectrum), BATCH_SIZE):
        batch = torch.from_numpy(spectrum[start : start + BATCH_SIZE]).to(device)
        output.append(model(batch).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def query_data(
    query_ids: np.ndarray,
    fit_ids: np.ndarray,
    xy: np.ndarray,
    node: np.ndarray,
    target: np.ndarray,
    teacher: SpectrumEncoder,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    donor_ids, distance = v406.find_neighbors(
        query_ids, fit_ids, xy, v406.DONOR_K
    )
    anchor, _ = v406.weighted_anchor(target, donor_ids, distance)
    anchor_embedding = encode_spectra(teacher, anchor, device)
    descriptor = v257.spectrum_descriptor(anchor, anchor.sum(axis=2))
    feature = np.concatenate(
        [node[query_ids], anchor_embedding, descriptor], axis=1
    ).astype(np.float32)
    return anchor, donor_ids, feature


def standardize(
    train_feature: np.ndarray,
    *other: np.ndarray,
) -> tuple[np.ndarray, ...]:
    mean = train_feature.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_feature.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    return tuple(((value - mean) / std).astype(np.float32) for value in (train_feature,) + other)


def retrieve(
    query_embedding: np.ndarray,
    donor_ids: np.ndarray,
    teacher_embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    similarity = np.einsum(
        "qe,qke->qk", query_embedding, teacher_embedding[donor_ids], optimize=True
    )
    order = np.argsort(similarity, axis=1)
    rank = order[:, -1]
    best = similarity[np.arange(len(rank)), rank]
    second = similarity[np.arange(len(rank)), order[:, -2]]
    return rank.astype(np.int64), (best - second).astype(np.float32)


def render(
    anchor: np.ndarray,
    donor_ids: np.ndarray,
    rank: np.ndarray,
    margin: np.ndarray,
    target: np.ndarray,
    alpha: float,
    threshold: float,
) -> np.ndarray:
    prediction = anchor.copy()
    active = margin >= threshold
    rows = np.flatnonzero(active)
    donor = target[donor_ids[rows, rank[rows]]]
    prediction[active] = (1.0 - alpha) * anchor[active] + alpha * donor
    return v406.normalize(prediction)


def weighted_delta(
    prediction: np.ndarray, anchor: np.ndarray, truth: np.ndarray
) -> float:
    baseline = v402.point_metrics(anchor, truth)["weighted"]
    value = v402.point_metrics(prediction, truth)["weighted"]
    return float(np.mean(value - baseline))


def select_action(
    anchor: np.ndarray,
    donor_ids: np.ndarray,
    rank: np.ndarray,
    margin: np.ndarray,
    target: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, float, float, list[dict[str, float]]]:
    best_alpha = 0.0
    best_threshold = float("inf")
    best_delta = 0.0
    rows: list[dict[str, float]] = []
    for alpha in ALPHAS:
        for threshold in MARGIN_THRESHOLDS:
            prediction = render(
                anchor,
                donor_ids,
                rank,
                margin,
                target,
                float(alpha),
                float(threshold),
            )
            delta = weighted_delta(prediction, anchor, truth)
            rows.append(
                {
                    "alpha": float(alpha),
                    "threshold": float(threshold),
                    "delta": delta,
                    "active_fraction": float(np.mean(margin >= threshold)),
                }
            )
            if delta > best_delta:
                best_delta = delta
                best_alpha = float(alpha)
                best_threshold = float(threshold)
    return best_alpha, best_threshold, best_delta, rows


@torch.no_grad()
def predict_student(
    model: QueryStudent,
    feature: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(feature), BATCH_SIZE):
        batch = torch.from_numpy(feature[start : start + BATCH_SIZE]).to(device)
        output.append(model(batch).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def train_student(
    feature: np.ndarray,
    row_target_embedding: np.ndarray,
    bank_embedding: np.ndarray,
    donor_ids: np.ndarray,
    anchor: np.ndarray,
    target: np.ndarray,
    row_truth: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[QueryStudent, tuple[float, float], list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(feature).to(device)
    y = torch.from_numpy(row_target_embedding).to(device)
    model = QueryStudent(feature.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=STUDENT_LR, weight_decay=4e-4
    )
    best_state = copy.deepcopy(model.state_dict())
    best_action = (0.0, float("inf"))
    best_delta = 0.0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, STUDENT_EPOCHS + 1):
        model.train()
        order = rng.permutation(train_rows)
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            rows = torch.from_numpy(order[start : start + BATCH_SIZE]).to(device)
            prediction = model(x[rows])
            truth = y[rows]
            logits = prediction @ truth.T / TEMPERATURE
            label = torch.arange(len(rows), device=device)
            loss = (1.0 - torch.sum(prediction * truth, dim=1)).mean()
            loss = loss + 0.35 * F.cross_entropy(logits, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(rows)
        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == STUDENT_EPOCHS:
            embedding = predict_student(model, feature[validation_rows], device)
            rank, margin = retrieve(
                embedding,
                donor_ids[validation_rows],
                bank_embedding,
            )
            alpha, threshold, delta, _ = select_action(
                anchor[validation_rows],
                donor_ids[validation_rows],
                rank,
                margin,
                target,
                row_truth[validation_rows],
            )
            history.append(
                {
                    "epoch": epoch,
                    "loss": total_loss / len(train_rows),
                    "validation_delta": delta,
                    "validation_alpha": alpha,
                    "validation_threshold": threshold,
                }
            )
            if delta > best_delta + 1e-5:
                best_delta = delta
                best_action = (alpha, threshold)
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if epoch >= 20 and stale >= PATIENCE_EVALS:
                break
    model.load_state_dict(best_state)
    return model, best_action, history


def evaluate(
    prediction: np.ndarray, anchor: np.ndarray, truth: np.ndarray
) -> tuple[dict[str, object], np.ndarray]:
    base = v402.point_metrics(anchor, truth)
    values = v402.point_metrics(prediction, truth)
    delta = values["weighted"] - base["weighted"]
    return (
        {
            "metrics": v402.metric_summary(prediction, truth),
            "delta": {
                key: float(np.mean(values[key] - base[key])) for key in values
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
    views: np.ndarray,
    blocked: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    query_ids = folds[fold]
    fit_ids = np.setdiff1d(valid_ids, query_ids)
    validation_ids = folds[(fold + 1) % len(folds)]
    validation_rows = np.flatnonzero(np.isin(fit_ids, validation_ids)).astype(np.int64)
    train_rows = np.setdiff1d(np.arange(len(fit_ids)), validation_rows)
    teacher, teacher_history = train_teacher(
        fit_ids, views, 41100 + fold, device
    )
    teacher_embedding = np.zeros((len(target), EMBED_DIM), dtype=np.float32)
    teacher_embedding[fit_ids] = encode_spectra(
        teacher, target[fit_ids], device
    )
    fit_anchor, fit_donor, fit_feature = query_data(
        fit_ids, fit_ids, xy, node, target, teacher, device
    )
    outer_anchor, outer_donor, outer_feature = query_data(
        query_ids, fit_ids, xy, node, target, teacher, device
    )
    fit_feature, outer_feature = standardize(fit_feature, outer_feature)
    student, action, student_history = train_student(
        fit_feature,
        teacher_embedding[fit_ids],
        teacher_embedding,
        fit_donor,
        fit_anchor,
        target,
        target[fit_ids],
        train_rows,
        validation_rows,
        41150 + fold,
        device,
    )
    query_embedding = predict_student(student, outer_feature, device)
    rank, margin = retrieve(query_embedding, outer_donor, teacher_embedding)
    prediction = render(
        outer_anchor,
        outer_donor,
        rank,
        margin,
        target,
        action[0],
        action[1],
    )

    query_truth_embedding = encode_spectra(teacher, target[query_ids], device)
    teacher_rank, teacher_margin = retrieve(
        query_truth_embedding, outer_donor, teacher_embedding
    )
    teacher_alpha, teacher_threshold, _, _ = select_action(
        fit_anchor[validation_rows],
        fit_donor[validation_rows],
        retrieve(
            teacher_embedding[fit_ids][validation_rows],
            fit_donor[validation_rows],
            teacher_embedding,
        )[0],
        retrieve(
            teacher_embedding[fit_ids][validation_rows],
            fit_donor[validation_rows],
            teacher_embedding,
        )[1],
        target,
        target[fit_ids[validation_rows]],
    )
    teacher_prediction = render(
        outer_anchor,
        outer_donor,
        teacher_rank,
        teacher_margin,
        target,
        teacher_alpha,
        teacher_threshold,
    )
    option_scores = v406.option_scores(
        outer_anchor, target[outer_donor], target[query_ids]
    )
    full_oracle_choice = option_scores.argmax(axis=1)
    data = v406.RouterData(
        query_ids=query_ids,
        donor_ids=outer_donor,
        anchor=outer_anchor,
        scores=option_scores,
        q_geometry=np.empty((0,)),
        donor_geometry=np.empty((0,)),
        relation_geometry=np.empty((0,)),
        q_full=np.empty((0,)),
        donor_full=np.empty((0,)),
        relation_full=np.empty((0,)),
    )
    full_oracle = v406.render_choices(data, full_oracle_choice, target)

    truth = target[query_ids]
    methods: dict[str, object] = {}
    deltas: dict[str, np.ndarray] = {}
    regimes = regime_of(blocked[query_ids]).astype(str)
    for name, value in (
        ("student", prediction),
        ("teacher_oracle", teacher_prediction),
        ("full_oracle", full_oracle),
    ):
        result, delta = evaluate(value, outer_anchor, truth)
        result["by_regime"] = {
            regime: float(np.mean(delta[regimes == regime]))
            for regime in ("clear", "light", "medium", "heavy")
            if np.any(regimes == regime)
        }
        methods[name] = result
        deltas[name] = delta
    left_embedding = encode_spectra(teacher, views[4, fit_ids], device)
    right_embedding = encode_spectra(teacher, views[5, fit_ids], device)
    rng = np.random.default_rng(41190 + fold)
    row = {
        "fold": fold,
        "fit_count": int(len(fit_ids)),
        "query_count": int(len(query_ids)),
        "methods": methods,
        "student_action": {
            "alpha": action[0],
            "threshold": action[1],
            "active_fraction": float(np.mean(margin >= action[1])),
            "margin": summarize(margin),
        },
        "teacher_action": {
            "alpha": teacher_alpha,
            "threshold": teacher_threshold,
            "active_fraction": float(np.mean(teacher_margin >= teacher_threshold)),
        },
        "teacher_identity": {
            "same_point_cosine": float(np.mean(np.sum(left_embedding * right_embedding, axis=1))),
            "shuffled_cosine": float(
                np.mean(np.sum(left_embedding * right_embedding[rng.permutation(len(right_embedding))], axis=1))
            ),
        },
        "teacher_history": teacher_history,
        "student_history": student_history,
        "leakage_assertions": {
            "query_in_fit": int(np.intersect1d(query_ids, fit_ids).size),
            "outer_truth_in_student_or_action_selection": False,
            "outer_truth_used_for_scoring_and_teacher_oracles_only": True,
        },
    }
    arrays = {
        "ids": query_ids,
        "baseline": outer_anchor,
        "truth": truth,
        "student": prediction,
        "teacher_oracle": teacher_prediction,
        "full_oracle": full_oracle,
        **{f"delta_{name}": value for name, value in deltas.items()},
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
    views = make_views(targets)
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
            views,
            blocked,
            device,
        )
        rows.append(row)
        parts.append(arrays)
        print(
            f"v411 fold={fold} student={row['methods']['student']['delta']['weighted']:+.5f} "
            f"teacher={row['methods']['teacher_oracle']['delta']['weighted']:+.5f} "
            f"full={row['methods']['full_oracle']['delta']['weighted']:+.5f}",
            flush=True,
        )
    aggregate: dict[str, object] = {}
    for name in ("student", "teacher_oracle", "full_oracle"):
        delta = np.asarray(
            [row["methods"][name]["delta"]["weighted"] for row in rows]
        )
        aggregate[name] = {
            "weighted_delta": summarize(delta),
            "positive_folds": int(np.sum(delta > 0.0)),
        }
    passed = bool(
        aggregate["student"]["weighted_delta"]["mean"] >= MIN_WEIGHTED_DELTA
        and aggregate["student"]["positive_folds"] >= MIN_FOLD_WINS
        and aggregate["teacher_oracle"]["weighted_delta"]["mean"]
        >= MIN_TEACHER_ORACLE_DELTA
    )
    ids = np.concatenate([part["ids"] for part in parts])
    if len(ids) != len(np.unique(ids)) or len(ids) != len(valid_ids):
        raise AssertionError("OOF IDs are not a unique cover")
    names = (
        "baseline",
        "truth",
        "student",
        "teacher_oracle",
        "full_oracle",
        "delta_student",
        "delta_teacher_oracle",
        "delta_full_oracle",
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
            for name in names
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
            "A contrastive convolutional encoder can isolate the array-view-invariant "
            "propagation identity proven by v410, and geometry plus a KNN anchor can predict it."
        ),
        "compliance": {
            "uses": [
                "official train H multi-array views",
                "official train/test coordinates",
                "official Round1_Map.ply-derived node features",
            ],
            "does_not_use": [
                "test H or hidden truth",
                "v212/Sionna/RT/synthetic targets or online labels",
            ],
        },
        "parameters": {
            "embedding_dim": EMBED_DIM,
            "teacher_epochs": TEACHER_EPOCHS,
            "student_epochs": STUDENT_EPOCHS,
            "contrastive_temperature": TEMPERATURE,
            "alphas": ALPHAS.tolist(),
            "margin_thresholds": MARGIN_THRESHOLDS.tolist(),
        },
        "fold_rows": rows,
        "aggregate": aggregate,
        "selection": {
            "minimum_student_delta": MIN_WEIGHTED_DELTA,
            "minimum_positive_folds": MIN_FOLD_WINS,
            "minimum_teacher_oracle_delta": MIN_TEACHER_ORACLE_DELTA,
            "passed": passed,
            "next_action": (
                "train final strict router and deploy its joint spectrum target on the v284 carrier"
                if passed
                else "stop after five rounds and discuss the remaining geometry-to-identity gap"
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

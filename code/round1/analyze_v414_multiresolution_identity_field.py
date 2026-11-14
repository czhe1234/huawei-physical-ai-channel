#!/usr/bin/env python3
"""Learn a planar multi-resolution field of propagation identity."""

from __future__ import annotations

import copy
import json
import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v402_visible_path_residual_factorization as v402
import analyze_v406_neural_propagation_mode_router as v406
import analyze_v411_multiview_contrastive_mode_retrieval as v411
import analyze_v412_path_sequence_identity_student as v412
import analyze_v413_crossmodal_path_identity_clip as v413
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 414
VERSION = "v414_multiresolution_identity_field"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v414_multiresolution_identity_field_oof.npz"

RESOLUTIONS = (8, 16, 32, 64, 128)
GRID_DIM = 16
FOURIER_FREQUENCIES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
EPOCHS = 250
EVAL_EVERY = 10
PATIENCE_EVALS = 7
BATCH_SIZE = 128
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 2.0e-4
MIN_STUDENT_DELTA = 0.01
MIN_FOLD_WINS = 4
MIN_TEACHER_DELTA = 0.02


class MultiResolutionGrid(nn.Module):
    def __init__(self, lower: np.ndarray, upper: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("lower", torch.tensor(lower, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(upper - lower, dtype=torch.float32))
        self.tables = nn.ParameterList()
        for resolution in RESOLUTIONS:
            table = nn.Parameter(torch.empty((resolution + 1) ** 2, GRID_DIM))
            nn.init.normal_(table, mean=0.0, std=0.015)
            self.tables.append(table)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp((xy - self.lower) / self.scale, 0.0, 1.0)
        outputs: list[torch.Tensor] = []
        for resolution, table in zip(RESOLUTIONS, self.tables):
            coordinate = normalized * resolution
            base = torch.floor(coordinate).long()
            base = torch.clamp(base, 0, resolution - 1)
            fraction = coordinate - base.float()
            x0, y0 = base[:, 0], base[:, 1]
            x1, y1 = x0 + 1, y0 + 1
            stride = resolution + 1
            v00 = table[x0 * stride + y0]
            v10 = table[x1 * stride + y0]
            v01 = table[x0 * stride + y1]
            v11 = table[x1 * stride + y1]
            fx, fy = fraction[:, 0:1], fraction[:, 1:2]
            value = (
                (1.0 - fx) * (1.0 - fy) * v00
                + fx * (1.0 - fy) * v10
                + (1.0 - fx) * fy * v01
                + fx * fy * v11
            )
            outputs.append(value)
        return torch.cat(outputs, dim=1)


class IdentityField(nn.Module):
    def __init__(
        self, global_dim: int, lower: np.ndarray, upper: np.ndarray
    ) -> None:
        super().__init__()
        self.grid = MultiResolutionGrid(lower, upper)
        input_dim = global_dim + len(RESOLUTIONS) * GRID_DIM + 4 * len(FOURIER_FREQUENCIES)
        self.geometry = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.06),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, v411.EMBED_DIM),
        )
        self.spectrum_projection = nn.Sequential(
            nn.Linear(v411.EMBED_DIM, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, v411.EMBED_DIM),
        )

    def positional(self, xy: torch.Tensor) -> torch.Tensor:
        normalized = (xy - self.grid.lower) / self.grid.scale
        values: list[torch.Tensor] = []
        for frequency in FOURIER_FREQUENCIES:
            angle = 2.0 * math.pi * frequency * normalized
            values.extend([torch.sin(angle), torch.cos(angle)])
        return torch.cat(values, dim=1)

    def encode_geometry(
        self, xy: torch.Tensor, global_feature: torch.Tensor
    ) -> torch.Tensor:
        feature = torch.cat(
            [global_feature, self.grid(xy), self.positional(xy)], dim=1
        )
        return F.normalize(self.geometry(feature), dim=1)

    def encode_spectrum(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.spectrum_projection(embedding), dim=-1)


@torch.no_grad()
def project_spectrum(
    model: IdentityField, embedding: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(embedding), 512):
        tensor = torch.from_numpy(embedding[start : start + 512]).to(device)
        output.append(model.encode_spectrum(tensor).float().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


@torch.no_grad()
def predict_geometry(
    model: IdentityField,
    xy: np.ndarray,
    global_feature: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(xy), 512):
        position = torch.from_numpy(xy[start : start + 512].astype(np.float32)).to(device)
        feature = torch.from_numpy(global_feature[start : start + 512]).to(device)
        output.append(model.encode_geometry(position, feature).float().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def train_field(
    fit_xy: np.ndarray,
    fit_global: np.ndarray,
    teacher_embedding: np.ndarray,
    fit_ids: np.ndarray,
    fit_donor: np.ndarray,
    donor_utility: np.ndarray,
    fit_anchor: np.ndarray,
    target: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    seed: int,
    device: torch.device,
) -> tuple[IdentityField, tuple[float, float], list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xy_tensor = torch.from_numpy(fit_xy.astype(np.float32)).to(device)
    global_tensor = torch.from_numpy(fit_global).to(device)
    teacher_tensor = torch.from_numpy(teacher_embedding).to(device)
    fit_id_tensor = torch.from_numpy(fit_ids).to(device)
    donor_id_tensor = torch.from_numpy(fit_donor).to(device)
    utility_tensor = torch.from_numpy(donor_utility).to(device)
    model = IdentityField(fit_global.shape[1], bounds[0], bounds[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_state = copy.deepcopy(model.state_dict())
    best_action = (0.0, float("inf"))
    best_delta = 0.0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(train_rows)
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            rows_np = order[start : start + BATCH_SIZE]
            rows = torch.from_numpy(rows_np).to(device)
            geometry_embedding = model.encode_geometry(
                xy_tensor[rows], global_tensor[rows]
            )
            query_spectrum = model.encode_spectrum(
                teacher_tensor[fit_id_tensor[rows]]
            )
            donor_spectrum = model.encode_spectrum(
                teacher_tensor[donor_id_tensor[rows]]
            )
            loss = v413.crossmodal_loss(
                geometry_embedding,
                query_spectrum,
                donor_spectrum,
                utility_tensor[rows],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(rows_np)
        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            projected_teacher = project_spectrum(model, teacher_embedding, device)
            validation_geometry = predict_geometry(
                model,
                fit_xy[validation_rows],
                fit_global[validation_rows],
                device,
            )
            rank, margin = v411.retrieve(
                validation_geometry,
                fit_donor[validation_rows],
                projected_teacher,
            )
            alpha, threshold, delta, _ = v411.select_action(
                fit_anchor[validation_rows],
                fit_donor[validation_rows],
                rank,
                margin,
                target,
                target[fit_ids[validation_rows]],
            )
            target_cosine = float(
                np.mean(
                    np.sum(
                        validation_geometry
                        * projected_teacher[fit_ids[validation_rows]],
                        axis=1,
                    )
                )
            )
            history.append(
                {
                    "epoch": epoch,
                    "loss": total_loss / len(train_rows),
                    "validation_delta": delta,
                    "validation_crossmodal_cosine": target_cosine,
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
            if epoch >= 40 and stale >= PATIENCE_EVALS:
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


def run_fold(fold: int, device: torch.device) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    targets = np.load(v412.TARGET_CACHE)
    target = v406.normalize(targets["sh"].astype(np.float32))
    views = v411.make_views(targets)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    xy = train_pos[:, :2]
    node, _ = v257.build_node_features(xy, test_pos[:, :2])
    blocked = np.load(v412.GEOMETRY_CACHE)["train_blocked"].astype(np.float32)
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(
        targets["valid"].astype(bool) & component["valid"].any(axis=1)
    ).astype(np.int64)
    folds = v315.disjoint_folds(valid_ids, xy)
    query_ids = folds[fold]
    fit_ids = np.setdiff1d(valid_ids, query_ids)
    validation_ids = folds[(fold + 1) % len(folds)]
    validation_rows = np.flatnonzero(np.isin(fit_ids, validation_ids)).astype(np.int64)
    train_rows = np.setdiff1d(np.arange(len(fit_ids)), validation_rows)
    all_xy = np.concatenate([xy, test_pos[:, :2]], axis=0)
    lower = all_xy.min(axis=0).astype(np.float32) - 1.0
    upper = all_xy.max(axis=0).astype(np.float32) + 1.0

    teacher, teacher_history = v411.train_teacher(
        fit_ids, views, 41400 + fold, device
    )
    teacher_embedding = np.zeros((len(target), v411.EMBED_DIM), dtype=np.float32)
    teacher_embedding[fit_ids] = v411.encode_spectra(
        teacher, target[fit_ids], device
    )
    fit_anchor, fit_donor, fit_global = v412.make_global_feature(
        fit_ids, fit_ids, xy, node, target, teacher, device
    )
    outer_anchor, outer_donor, outer_global = v412.make_global_feature(
        query_ids, fit_ids, xy, node, target, teacher, device
    )
    fit_global, outer_global = v412.standardize_global(
        fit_global, outer_global, train_rows
    )
    utility, utility_diagnostic = v413.stable_donor_utility(
        fit_ids, fit_ids, fit_donor, xy, views, target
    )
    model, action, history = train_field(
        xy[fit_ids],
        fit_global,
        teacher_embedding,
        fit_ids,
        fit_donor,
        utility,
        fit_anchor,
        target,
        train_rows,
        validation_rows,
        (lower, upper),
        41450 + fold,
        device,
    )
    projected_teacher = project_spectrum(model, teacher_embedding, device)
    outer_geometry = predict_geometry(
        model, xy[query_ids], outer_global, device
    )
    rank, margin = v411.retrieve(
        outer_geometry, outer_donor, projected_teacher
    )
    prediction = v411.render(
        outer_anchor,
        outer_donor,
        rank,
        margin,
        target,
        action[0],
        action[1],
    )

    query_truth_teacher = v411.encode_spectra(teacher, target[query_ids], device)
    projected_query_truth = project_spectrum(model, query_truth_teacher, device)
    teacher_rank, teacher_margin = v411.retrieve(
        projected_query_truth, outer_donor, projected_teacher
    )
    validation_teacher_rank, validation_teacher_margin = v411.retrieve(
        projected_teacher[fit_ids[validation_rows]],
        fit_donor[validation_rows],
        projected_teacher,
    )
    teacher_action = v411.select_action(
        fit_anchor[validation_rows],
        fit_donor[validation_rows],
        validation_teacher_rank,
        validation_teacher_margin,
        target,
        target[fit_ids[validation_rows]],
    )
    teacher_prediction = v411.render(
        outer_anchor,
        outer_donor,
        teacher_rank,
        teacher_margin,
        target,
        teacher_action[0],
        teacher_action[1],
    )
    option_scores = v406.option_scores(
        outer_anchor, target[outer_donor], target[query_ids]
    )
    oracle_data = v406.RouterData(
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
    full_oracle = v406.render_choices(
        oracle_data, option_scores.argmax(axis=1), target
    )
    truth = target[query_ids]
    methods: dict[str, object] = {}
    deltas: dict[str, np.ndarray] = {}
    regimes = regime_of(blocked[query_ids]).astype(str)
    for name, value in (
        ("field_student", prediction),
        ("projected_teacher_oracle", teacher_prediction),
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
    report = {
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
            "alpha": teacher_action[0],
            "threshold": teacher_action[1],
            "active_fraction": float(
                np.mean(teacher_margin >= teacher_action[1])
            ),
        },
        "stable_utility": utility_diagnostic,
        "teacher_history": teacher_history,
        "student_history": history,
        "leakage_assertions": {
            "query_in_fit": int(np.intersect1d(query_ids, fit_ids).size),
            "outer_truth_in_student_or_action_selection": False,
            "outer_truth_used_for_scoring_and_oracles_only": True,
        },
    }
    (REPORT_DIR / f"{VERSION}_fold{fold}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        OUT_DIR / f"v414_fold{fold}.npz",
        ids=query_ids,
        baseline=outer_anchor,
        truth=truth,
        field_student=prediction,
        projected_teacher_oracle=teacher_prediction,
        full_oracle=full_oracle,
        delta_field_student=deltas["field_student"],
        delta_projected_teacher_oracle=deltas["projected_teacher_oracle"],
        delta_full_oracle=deltas["full_oracle"],
    )
    print(
        f"v414 fold={fold} student={methods['field_student']['delta']['weighted']:+.6f} "
        f"teacher={methods['projected_teacher_oracle']['delta']['weighted']:+.6f} "
        f"full={methods['full_oracle']['delta']['weighted']:+.6f}",
        flush=True,
    )


def aggregate() -> None:
    rows = [
        json.loads(
            (REPORT_DIR / f"{VERSION}_fold{fold}.json").read_text(encoding="utf-8")
        )
        for fold in range(5)
    ]
    parts = [np.load(OUT_DIR / f"v414_fold{fold}.npz") for fold in range(5)]
    result: dict[str, object] = {}
    for name in ("field_student", "projected_teacher_oracle", "full_oracle"):
        delta = np.asarray(
            [row["methods"][name]["delta"]["weighted"] for row in rows]
        )
        result[name] = {
            "weighted_delta": summarize(delta),
            "positive_folds": int(np.sum(delta > 0.0)),
        }
    passed = bool(
        result["field_student"]["weighted_delta"]["mean"] >= MIN_STUDENT_DELTA
        and result["field_student"]["positive_folds"] >= MIN_FOLD_WINS
        and result["projected_teacher_oracle"]["weighted_delta"]["mean"]
        >= MIN_TEACHER_DELTA
    )
    ids = np.concatenate([part["ids"] for part in parts])
    if len(ids) != 1984 or len(np.unique(ids)) != 1984:
        raise AssertionError("v414 OOF IDs are not a unique 1984-point cover")
    names = (
        "baseline",
        "truth",
        "field_student",
        "projected_teacher_oracle",
        "full_oracle",
        "delta_field_student",
        "delta_projected_teacher_oracle",
        "delta_full_oracle",
    )
    np.savez_compressed(
        OOF_CACHE,
        ids=ids,
        fold=np.concatenate(
            [np.full(len(parts[fold]["ids"]), fold, dtype=np.int16) for fold in range(5)]
        ),
        **{name: np.concatenate([part[name] for part in parts]) for name in names},
    )
    report = {
        "method": VERSION,
        "status": (
            "official_oof_passed_ready_for_final_training"
            if passed
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "A multi-resolution neural field can represent planar propagation boundaries "
            "that KNN and coordinate MLPs blur."
        ),
        "compliance": {
            "uses": [
                "official train H multi-array views",
                "official train/test coordinates",
                "official Round1_Map.ply-derived node features",
            ],
            "does_not_use": [
                "test H or hidden truth",
                "synthetic targets or Sionna coefficients",
                "v212 teacher output or online labels",
            ],
        },
        "parameters": {
            "grid_resolutions": list(RESOLUTIONS),
            "grid_dim": GRID_DIM,
            "fourier_frequencies": list(FOURIER_FREQUENCIES),
            "epochs": EPOCHS,
            "five_gpu_execution": True,
        },
        "fold_rows": rows,
        "aggregate": result,
        "selection": {
            "minimum_student_delta": MIN_STUDENT_DELTA,
            "minimum_positive_folds": MIN_FOLD_WINS,
            "minimum_teacher_delta": MIN_TEACHER_DELTA,
            "passed": passed,
            "next_action": (
                "train final field and evaluate frozen local traceback"
                if passed
                else "the train/test geometry field does not predict fine propagation identity"
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
    for path in (
        v412.TARGET_CACHE,
        v412.GEOMETRY_CACHE,
        v297.COMPONENT_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if world_size != 5:
            raise ValueError("v414 distributed mode requires exactly five workers")
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

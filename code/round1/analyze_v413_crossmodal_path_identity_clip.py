#!/usr/bin/env python3
"""Learn the geometry-predictable subspace of multi-view propagation identity."""

from __future__ import annotations

import copy
import json
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v293_interaction_physical_path_law as v293
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v402_visible_path_residual_factorization as v402
import analyze_v406_neural_propagation_mode_router as v406
import analyze_v411_multiview_contrastive_mode_retrieval as v411
import analyze_v412_path_sequence_identity_student as v412
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 413
VERSION = "v413_crossmodal_path_identity_clip"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v413_crossmodal_path_identity_clip_oof.npz"

STUDENT_EPOCHS = 100
EVAL_EVERY = 5
PATIENCE_EVALS = 6
BATCH_SIZE = 20
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 4e-4
UTILITY_TEMPERATURE = 0.02
CLIP_TEMPERATURE = 0.07
MIN_STUDENT_DELTA = 0.01
MIN_FOLD_WINS = 4
MIN_TEACHER_DELTA = 0.02


class CrossModalIdentity(nn.Module):
    def __init__(self, global_dim: int, path_dim: int, step_dim: int) -> None:
        super().__init__()
        self.geometry = v412.PathSequenceStudent(global_dim, path_dim, step_dim)
        self.spectrum_projection = nn.Sequential(
            nn.Linear(v411.EMBED_DIM, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.06),
            nn.Linear(128, v411.EMBED_DIM),
        )

    def encode_geometry(
        self,
        global_feature: torch.Tensor,
        path_feature: torch.Tensor,
        step_feature: torch.Tensor,
        path_valid: torch.Tensor,
        step_valid: torch.Tensor,
    ) -> torch.Tensor:
        return self.geometry(
            global_feature, path_feature, step_feature, path_valid, step_valid
        )

    def encode_spectrum(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.spectrum_projection(embedding), dim=-1)


def stable_donor_utility(
    query_ids: np.ndarray,
    fit_ids: np.ndarray,
    donor_ids: np.ndarray,
    xy: np.ndarray,
    views: np.ndarray,
    full_target: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    checked_ids, distance = v406.find_neighbors(
        query_ids, fit_ids, xy, v406.DONOR_K
    )
    if not np.array_equal(checked_ids, donor_ids):
        raise AssertionError("stable utility donor bank mismatch")
    candidate_sum = np.zeros(
        (len(query_ids), v406.DONOR_K, len(v406.ALPHAS)), dtype=np.float64
    )
    baseline_sum = np.zeros(len(query_ids), dtype=np.float64)
    sources = [views[index] for index in range(len(views))] + [full_target]
    for source in sources:
        anchor, _ = v406.weighted_anchor(source, donor_ids, distance)
        score = v406.option_scores(anchor, source[donor_ids], source[query_ids])
        baseline_sum += score[:, 0]
        candidate_sum += score[:, 1:].reshape(candidate_sum.shape)
    candidate_mean = candidate_sum / len(sources)
    baseline_mean = baseline_sum / len(sources)
    donor_utility = candidate_mean.max(axis=2) - baseline_mean[:, None]
    return donor_utility.astype(np.float32), {
        "mean_best_gain": float(np.mean(np.max(donor_utility, axis=1))),
        "positive_best_fraction": float(np.mean(np.max(donor_utility, axis=1) > 0.0)),
        "mean_candidate_gain": float(np.mean(donor_utility)),
    }


def crossmodal_loss(
    geometry_embedding: torch.Tensor,
    query_spectrum_embedding: torch.Tensor,
    donor_spectrum_embedding: torch.Tensor,
    donor_utility: torch.Tensor,
) -> torch.Tensor:
    utility_target = torch.softmax(donor_utility / UTILITY_TEMPERATURE, dim=1)
    donor_logit = torch.sum(
        geometry_embedding[:, None, :] * donor_spectrum_embedding, dim=2
    )
    soft_rank = torch.sum(
        -utility_target
        * F.log_softmax(donor_logit / CLIP_TEMPERATURE, dim=1),
        dim=1,
    ).mean()
    hard_rank = F.cross_entropy(
        donor_logit / CLIP_TEMPERATURE, donor_utility.argmax(dim=1)
    )
    batch_logit = geometry_embedding @ query_spectrum_embedding.T / CLIP_TEMPERATURE
    batch_label = torch.arange(len(geometry_embedding), device=geometry_embedding.device)
    clip_loss = 0.5 * (
        F.cross_entropy(batch_logit, batch_label)
        + F.cross_entropy(batch_logit.T, batch_label)
    )
    alignment = (
        1.0 - torch.sum(geometry_embedding * query_spectrum_embedding, dim=1)
    ).mean()
    return 0.42 * soft_rank + 0.18 * hard_rank + 0.28 * clip_loss + 0.12 * alignment


@torch.no_grad()
def predict_geometry(
    model: CrossModalIdentity,
    global_feature: torch.Tensor,
    path_feature: torch.Tensor,
    step_feature: torch.Tensor,
    path_valid: torch.Tensor,
    step_valid: torch.Tensor,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(global_feature), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(global_feature))
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=global_feature.is_cuda
        ):
            embedding = model.encode_geometry(
                global_feature[start:end],
                path_feature[start:end],
                step_feature[start:end],
                path_valid[start:end],
                step_valid[start:end],
            )
        output.append(embedding.float().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


@torch.no_grad()
def project_spectrum(
    model: CrossModalIdentity,
    embedding: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(embedding), 512):
        tensor = torch.from_numpy(embedding[start : start + 512]).to(device)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            value = model.encode_spectrum(tensor)
        output.append(value.float().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def train_crossmodal(
    fit_global: np.ndarray,
    fit_path: np.ndarray,
    fit_step: np.ndarray,
    fit_path_valid: np.ndarray,
    fit_step_valid: np.ndarray,
    teacher_embedding: np.ndarray,
    fit_ids: np.ndarray,
    fit_donor: np.ndarray,
    donor_utility: np.ndarray,
    fit_anchor: np.ndarray,
    target: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[CrossModalIdentity, tuple[float, float], list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    global_tensor = torch.from_numpy(fit_global).to(device)
    path_tensor = torch.from_numpy(fit_path).to(device)
    step_tensor = torch.from_numpy(fit_step).to(device)
    path_valid_tensor = torch.from_numpy(fit_path_valid).to(device)
    step_valid_tensor = torch.from_numpy(fit_step_valid).to(device)
    teacher_tensor = torch.from_numpy(teacher_embedding).to(device)
    utility_tensor = torch.from_numpy(donor_utility).to(device)
    donor_id_tensor = torch.from_numpy(fit_donor).to(device)
    fit_id_tensor = torch.from_numpy(fit_ids).to(device)
    model = CrossModalIdentity(
        fit_global.shape[1], fit_path.shape[2], fit_step.shape[3]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
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
            rows_np = order[start : start + BATCH_SIZE]
            rows = torch.from_numpy(rows_np).to(device)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                geometry_embedding = model.encode_geometry(
                    global_tensor[rows],
                    path_tensor[rows],
                    step_tensor[rows],
                    path_valid_tensor[rows],
                    step_valid_tensor[rows],
                )
                query_spectrum_embedding = model.encode_spectrum(
                    teacher_tensor[fit_id_tensor[rows]]
                )
                donor_spectrum_embedding = model.encode_spectrum(
                    teacher_tensor[donor_id_tensor[rows]]
                )
                loss = crossmodal_loss(
                    geometry_embedding,
                    query_spectrum_embedding,
                    donor_spectrum_embedding,
                    utility_tensor[rows],
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(rows_np)
        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == STUDENT_EPOCHS:
            projected_teacher = project_spectrum(model, teacher_embedding, device)
            validation_geometry = predict_geometry(
                model,
                global_tensor[validation_rows],
                path_tensor[validation_rows],
                step_tensor[validation_rows],
                path_valid_tensor[validation_rows],
                step_valid_tensor[validation_rows],
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
            if epoch >= 25 and stale >= PATIENCE_EVALS:
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
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_xy = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)[:, :2]
    xy = positions[:, :2]
    node, _ = v257.build_node_features(xy, test_xy)
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

    teacher, teacher_history = v411.train_teacher(
        fit_ids, views, 41300 + fold, device
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
    utility, utility_diagnostic = stable_donor_utility(
        fit_ids, fit_ids, fit_donor, xy, views, target
    )
    path_cache = np.load(v412.PATH_CACHE)
    interaction_cache = np.load(v293.FEATURE_CACHE)
    train_path, train_step, train_path_valid, train_step_valid = v412.build_geometry_features(
        path_cache, interaction_cache, "train", positions
    )
    model, action, student_history = train_crossmodal(
        fit_global,
        train_path[fit_ids],
        train_step[fit_ids],
        train_path_valid[fit_ids],
        train_step_valid[fit_ids],
        teacher_embedding,
        fit_ids,
        fit_donor,
        utility,
        fit_anchor,
        target,
        train_rows,
        validation_rows,
        41350 + fold,
        device,
    )
    projected_teacher = project_spectrum(model, teacher_embedding, device)
    outer_geometry = predict_geometry(
        model,
        torch.from_numpy(outer_global).to(device),
        torch.from_numpy(train_path[query_ids]).to(device),
        torch.from_numpy(train_step[query_ids]).to(device),
        torch.from_numpy(train_path_valid[query_ids]).to(device),
        torch.from_numpy(train_step_valid[query_ids]).to(device),
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
        ("crossmodal_student", prediction),
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
        "student_history": student_history,
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
        OUT_DIR / f"v413_fold{fold}.npz",
        ids=query_ids,
        baseline=outer_anchor,
        truth=truth,
        crossmodal_student=prediction,
        projected_teacher_oracle=teacher_prediction,
        full_oracle=full_oracle,
        delta_crossmodal_student=deltas["crossmodal_student"],
        delta_projected_teacher_oracle=deltas["projected_teacher_oracle"],
        delta_full_oracle=deltas["full_oracle"],
    )
    print(
        f"v413 fold={fold} student={methods['crossmodal_student']['delta']['weighted']:+.6f} "
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
    parts = [np.load(OUT_DIR / f"v413_fold{fold}.npz") for fold in range(5)]
    result: dict[str, object] = {}
    for name in ("crossmodal_student", "projected_teacher_oracle", "full_oracle"):
        delta = np.asarray(
            [row["methods"][name]["delta"]["weighted"] for row in rows]
        )
        result[name] = {
            "weighted_delta": summarize(delta),
            "positive_folds": int(np.sum(delta > 0.0)),
        }
    passed = bool(
        result["crossmodal_student"]["weighted_delta"]["mean"]
        >= MIN_STUDENT_DELTA
        and result["crossmodal_student"]["positive_folds"] >= MIN_FOLD_WINS
        and result["projected_teacher_oracle"]["weighted_delta"]["mean"]
        >= MIN_TEACHER_DELTA
    )
    ids = np.concatenate([part["ids"] for part in parts])
    if len(ids) != 1984 or len(np.unique(ids)) != 1984:
        raise AssertionError("v413 OOF IDs are not a unique 1984-point cover")
    names = (
        "baseline",
        "truth",
        "crossmodal_student",
        "projected_teacher_oracle",
        "full_oracle",
        "delta_crossmodal_student",
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
            "A jointly learned cross-modal space can discard view-invariant spectrum "
            "details that official-pointcloud paths cannot predict."
        ),
        "compliance": {
            "uses": [
                "official train H multi-array views",
                "official train/test coordinates",
                "official Round1_Map.ply-derived depth-16 path sequences",
            ],
            "does_not_use": [
                "test H or hidden truth",
                "synthetic targets or Sionna path coefficients",
                "v212 teacher output or online score labels",
            ],
        },
        "parameters": {
            "paths": v412.PATH_COUNT,
            "max_depth": v412.MAX_DEPTH,
            "student_epochs": STUDENT_EPOCHS,
            "utility_views": 7,
            "crossmodal_embedding_dim": v411.EMBED_DIM,
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
                "train final official-only cross-modal model and evaluate frozen local traceback"
                if passed
                else "cross-modal projection still does not make fine identity geometrically observable"
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
        v412.PATH_CACHE,
        v412.TARGET_CACHE,
        v412.GEOMETRY_CACHE,
        v293.FEATURE_CACHE,
        v297.COMPONENT_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if world_size != 5:
            raise ValueError("v413 distributed mode requires exactly five workers")
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

#!/usr/bin/env python3
"""Bridge geometry-predicted vertical fingerprints to query KNN spectra.

Candidates that collide in the same delay-horizontal-angle cell compete for a
single probability budget.  The only new identity signal is the cosine between
the candidate's geometry-predicted BS vertical fingerprint and a query vertical
spectrum predicted exclusively from outer-fit official H.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v309_unmatched_effective_scatterers as v309
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v318_geometry_array_fingerprint_manifold as v318
import analyze_v319_unseen_patch_vertical_manifold as v319
import analyze_v328_patch_existence_support_field as v328
import analyze_v402_visible_path_residual_factorization as v402
import analyze_v403_latent_occupancy_detection_paths as v403
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 404
VERSION = "v404_vertical_fingerprint_identity_bridge"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
FIGURE_DIR = REPORT_DIR / "版本四零四垂直指纹身份桥接图"
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v404_vertical_fingerprint_identity_oof.npz"
FIGURE = FIGURE_DIR / "同支撑候选的垂直指纹竞争结果.png"

TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOMETRY_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

MAX_PATHS = 8
MIN_AP_LIFT = 0.01
MIN_COLLISION_LIFT = 0.05
MIN_IDENTITY_CONTROL_LIFT = 0.001
MIN_JOINT_DELTA = 0.01
MIN_JOINT_WINS = 4


def configure_chinese_font() -> None:
    selected = "DejaVu Sans"
    for family in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=family), fallback_to_default=False
            )
            font_manager.fontManager.addfont(path)
            selected = font_manager.FontProperties(fname=path).get_name()
            break
        except ValueError:
            continue
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )


def candidate_numeric(
    point_id: np.ndarray,
    candidate_id: np.ndarray,
    positions: np.ndarray,
    candidate: dict[str, np.ndarray],
    patches: np.ndarray,
) -> np.ndarray:
    reflection = candidate["reflection_point"][point_id, candidate_id].astype(
        np.float32
    )
    path_length = candidate["path_length"][point_id, candidate_id].astype(np.float32)
    excess = candidate["excess_length"][point_id, candidate_id].astype(np.float32)
    patch_id = candidate["patch_id"][point_id, candidate_id].astype(np.int64)
    patch = patches[np.clip(patch_id, 0, len(patches) - 1)].astype(np.float32)
    ue = positions[point_id].astype(np.float32)
    bs = v304.BS.astype(np.float32)
    bs_to_reflection = reflection - bs[None]
    reflection_to_ue = ue - reflection
    ue_to_bs = bs[None] - ue
    departure = bs_to_reflection / np.maximum(
        np.linalg.norm(bs_to_reflection, axis=1, keepdims=True), 1e-6
    )
    arrival = reflection_to_ue / np.maximum(
        np.linalg.norm(reflection_to_ue, axis=1, keepdims=True), 1e-6
    )
    ue_boresight = ue_to_bs / np.maximum(
        np.linalg.norm(ue_to_bs, axis=1, keepdims=True), 1e-6
    )
    normal = np.concatenate(
        [patch[:, :2], np.zeros((len(patch), 1), dtype=np.float32)], axis=1
    )
    incidence = np.abs(np.sum(departure * normal, axis=1))
    outgoing = np.abs(np.sum(arrival * normal, axis=1))
    direct = np.linalg.norm(ue - bs[None], axis=1)
    return np.concatenate(
        [
            departure,
            arrival,
            ue_boresight,
            (ue - bs[None]) / 500.0,
            (reflection - bs[None]) / 500.0,
            np.stack(
                [
                    path_length / 800.0,
                    excess / 400.0,
                    direct / 500.0,
                    incidence,
                    outgoing,
                    patch[:, 0],
                    patch[:, 1],
                    patch[:, 5] / 40.0,
                    patch[:, 6] / 40.0,
                    np.log1p(patch[:, 7]) / 10.0,
                ],
                axis=1,
            ),
        ],
        axis=1,
    ).astype(np.float32)


def train_vertical_model(
    fit_ids: np.ndarray,
    data: dict[str, np.ndarray],
    seed: int,
    device: torch.device,
) -> tuple[v319.VerticalMLP, np.ndarray, np.ndarray]:
    fit = np.isin(data["point_id"], fit_ids)
    if np.sum(fit) < 100:
        raise RuntimeError("not enough fit-only fingerprint matches")
    mean = data["numeric"][fit].mean(axis=0)
    std = data["numeric"][fit].std(axis=0)
    std[std < 1e-5] = 1.0
    x = ((data["numeric"][fit] - mean) / std).astype(np.float32)
    weight = (
        np.sqrt(np.maximum(data["mass"][fit], 1e-5)) * data["confidence"][fit]
    ).astype(np.float32)
    model = v319.train(x, data["vertical"][fit], weight, seed, device)
    return model, mean.astype(np.float32), std.astype(np.float32)


def knn_marginal(
    bank: np.ndarray, neighbor_ids: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    return np.sum(
        weight[:, :, None, None, None] * bank[neighbor_ids],
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)


def sample_vertical_profiles(
    query_ids: np.ndarray,
    base_shv: np.ndarray,
    candidate: dict[str, np.ndarray],
    valid: np.ndarray,
    slope: float,
    intercept: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s_center, h_center = v403.candidate_centers(candidate, slope, intercept)
    row, candidate_id = np.nonzero(valid[query_ids])
    point = query_ids[row]
    profile = np.zeros((len(row), 8), dtype=np.float32)
    local_mass = np.zeros(len(row), dtype=np.float32)
    for query_row in range(len(query_ids)):
        active_rows = np.flatnonzero(row == query_row)
        if len(active_rows) == 0:
            continue
        active_candidate = candidate_id[active_rows]
        maps = np.moveaxis(base_shv[query_row], -1, 0)
        sampled = v403._sample_local_map(
            maps,
            s_center[query_ids[query_row], active_candidate],
            h_center[query_ids[query_row], active_candidate],
        )
        local_mass[active_rows] = sampled.sum(axis=1)
        profile[active_rows] = sampled / np.maximum(
            sampled.sum(axis=1, keepdims=True), 1e-30
        )
    return row.astype(np.int64), candidate_id.astype(np.int64), profile, local_mass


def profile_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (
        np.sum(left * right, axis=1)
        / np.maximum(np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-30)
    ).astype(np.float32)


def collision_groups(
    point_id: int,
    candidate: dict[str, np.ndarray],
    valid: np.ndarray,
    slope: float,
    intercept: float,
) -> list[np.ndarray]:
    active = np.flatnonzero(valid[point_id])
    if len(active) == 0:
        return []
    s_center, h_center = v403.candidate_centers(candidate, slope, intercept)
    keys = np.stack(
        [
            np.remainder(np.rint(s_center[point_id, active]).astype(np.int64), 192),
            np.remainder(np.rint(h_center[point_id, active]).astype(np.int64), 16),
        ],
        axis=1,
    )
    groups: dict[tuple[int, int], list[int]] = {}
    for candidate_id, key in zip(active, keys):
        groups.setdefault((int(key[0]), int(key[1])), []).append(int(candidate_id))
    return [np.asarray(value, dtype=np.int64) for value in groups.values()]


def compete(
    query_ids: np.ndarray,
    support: np.ndarray,
    identity: np.ndarray,
    candidate: dict[str, np.ndarray],
    valid: np.ndarray,
    slope: float,
    intercept: float,
) -> tuple[np.ndarray, dict[str, object]]:
    output = support.copy().astype(np.float32)
    collision_count = 0
    collision_candidate_count = 0
    for row, point_id in enumerate(query_ids):
        for group in collision_groups(
            int(point_id), candidate, valid, slope, intercept
        ):
            if len(group) <= 1:
                continue
            collision_count += 1
            collision_candidate_count += len(group)
            original = np.maximum(support[row, group], 0.0).astype(np.float64)
            budget = float(np.max(original))
            if budget <= 0.0:
                output[row, group] = 0.0
                continue
            score = original * np.maximum(identity[row, group], 1e-4)
            if float(np.sum(score)) <= 0.0:
                score = original
            output[row, group] = (budget * score / np.sum(score)).astype(np.float32)
    return output, {
        "collision_group_count": collision_count,
        "collision_candidate_count": collision_candidate_count,
        "collision_candidates_per_group": collision_candidate_count
        / max(collision_count, 1),
    }


def collision_top1(
    query_ids: np.ndarray,
    score: np.ndarray,
    truth_mass: np.ndarray,
    candidate: dict[str, np.ndarray],
    valid: np.ndarray,
    slope: float,
    intercept: float,
) -> dict[str, float]:
    correct = 0
    count = 0
    correct_mass = 0.0
    total_mass = 0.0
    for row, point_id in enumerate(query_ids):
        for group in collision_groups(
            int(point_id), candidate, valid, slope, intercept
        ):
            positive = group[truth_mass[row, group] > 0.0]
            if len(group) <= 1 or len(positive) != 1:
                continue
            selected = int(group[np.argmax(score[row, group])])
            target = int(positive[0])
            mass = float(truth_mass[row, target])
            count += 1
            total_mass += mass
            if selected == target:
                correct += 1
                correct_mass += mass
    return {
        "group_count": count,
        "accuracy": correct / max(count, 1),
        "mass_weighted_accuracy": correct_mass / max(total_mass, 1e-30),
    }


def existence_metrics(score: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    value = score[valid]
    mass = truth[valid]
    binary = mass > 0.0
    return {
        "average_precision": float(average_precision_score(binary, value)),
        **v328.precision_recall_at_target(value, mass, v328.PRECISION_TARGET),
    }


def spectrum_summary(
    prediction: np.ndarray, baseline: np.ndarray, truth: np.ndarray
) -> dict[str, object]:
    base = v402.point_metrics(baseline, truth)
    current = v402.point_metrics(prediction, truth)
    return {
        "metrics": v402.metric_summary(prediction, truth),
        "delta": {
            key: float(np.mean(current[key] - base[key])) for key in current
        },
        "point_win_fraction": float(np.mean(current["weighted"] > base["weighted"])),
    }


def fold_experiment(
    fold: int,
    query_ids: np.ndarray,
    valid_ids: np.ndarray,
    positions: np.ndarray,
    blocked: np.ndarray,
    targets: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    valid: np.ndarray,
    hard_mass: np.ndarray,
    fingerprint_data: dict[str, np.ndarray],
    patches: np.ndarray,
    slope: float,
    intercept: float,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    fit_ids = np.setdiff1d(valid_ids, query_ids)
    neighbor_ids, neighbor_weight = v402.neighbors(
        query_ids, fit_ids, positions[:, :2]
    )
    baseline = v402.knn_average(targets["sh"], neighbor_ids, neighbor_weight)
    base_shv = knn_marginal(targets["shv"], neighbor_ids, neighbor_weight)

    model, mean, std = train_vertical_model(
        fit_ids, fingerprint_data, 40_400 + fold, device
    )
    query_valid = valid[query_ids]
    pair_row, pair_candidate, observed_profile, local_mass = sample_vertical_profiles(
        query_ids, base_shv, candidate, valid, slope, intercept
    )
    pair_point = query_ids[pair_row]
    numeric = candidate_numeric(
        pair_point, pair_candidate, positions, candidate, patches
    )
    predicted_profile = v319.predict(
        model, ((numeric - mean) / std).astype(np.float32), device
    )
    identity_value = profile_cosine(predicted_profile, observed_profile)
    identity = np.zeros_like(query_valid, dtype=np.float32)
    identity[pair_row, pair_candidate] = identity_value
    (
        oracle_pair_row,
        oracle_pair_candidate,
        oracle_observed_profile,
        _oracle_local_mass,
    ) = sample_vertical_profiles(
        query_ids,
        targets["shv"][query_ids],
        candidate,
        valid,
        slope,
        intercept,
    )
    if not (
        np.array_equal(pair_row, oracle_pair_row)
        and np.array_equal(pair_candidate, oracle_pair_candidate)
    ):
        raise AssertionError("deployable and oracle candidate rows differ")
    oracle_identity_value = profile_cosine(
        predicted_profile, oracle_observed_profile
    )
    oracle_identity = np.zeros_like(identity)
    oracle_identity[pair_row, pair_candidate] = oracle_identity_value

    rng = np.random.default_rng(404_000 + fold)
    shuffled_identity = np.zeros_like(identity)
    for row in range(len(query_ids)):
        active = np.flatnonzero(query_valid[row])
        shuffled_identity[row, active] = identity[row, rng.permutation(active)]

    hard_binary = (hard_mass > 0.0).astype(np.float32)
    support = v403.patch_predict(
        candidate,
        valid,
        hard_binary,
        fit_ids,
        query_ids,
        positions[:, :2],
    )
    ones = np.ones_like(support, dtype=np.float32)
    competition, collision_report = compete(
        query_ids, support, ones, candidate, valid, slope, intercept
    )
    vertical, _ = compete(
        query_ids, support, identity, candidate, valid, slope, intercept
    )
    shuffled, _ = compete(
        query_ids, support, shuffled_identity, candidate, valid, slope, intercept
    )
    oracle_vertical, _ = compete(
        query_ids, support, oracle_identity, candidate, valid, slope, intercept
    )

    hard_energy = v403.patch_predict(
        candidate,
        valid,
        hard_mass,
        fit_ids,
        query_ids,
        positions[:, :2],
        log_average=True,
        source_weight=hard_binary,
    )
    methods = {
        "independent": support,
        "competition": competition,
        "vertical": vertical,
        "shuffled": shuffled,
        "oracle_vertical": oracle_vertical,
    }
    prediction: dict[str, np.ndarray] = {}
    render: dict[str, object] = {}
    for name, probability in methods.items():
        prediction[name], render[name] = v403.factorized_prediction(
            fit_ids,
            query_ids,
            targets["sh"],
            candidate,
            hard_mass[fit_ids],
            probability * hard_energy,
            neighbor_ids,
            neighbor_weight,
            slope,
            intercept,
        )

    truth_mass = hard_mass[query_ids]
    existence = {
        name: existence_metrics(score, truth_mass, query_valid)
        for name, score in methods.items()
    }
    top1 = {
        name: collision_top1(
            query_ids,
            score,
            truth_mass,
            candidate,
            valid,
            slope,
            intercept,
        )
        for name, score in methods.items()
    }
    truth = targets["sh"][query_ids]
    spectrum = {
        name: spectrum_summary(current, baseline, truth)
        for name, current in prediction.items()
    }
    base_values = v402.point_metrics(baseline, truth)
    vertical_values = v402.point_metrics(prediction["vertical"], truth)
    regimes = regime_of(blocked[query_ids]).astype(str)
    row = {
        "fold": fold,
        "fit_count": int(len(fit_ids)),
        "query_count": int(len(query_ids)),
        "fingerprint_training_count": int(
            np.sum(np.isin(fingerprint_data["point_id"], fit_ids))
        ),
        "identity": {
            "candidate_count": int(len(identity_value)),
            "deployable_cosine": summarize(identity_value),
            "oracle_query_cosine": summarize(oracle_identity_value),
            "local_mass": summarize(local_mass),
        },
        "collisions": collision_report,
        "existence": existence,
        "collision_top1": top1,
        "spectrum": spectrum,
        "vertical_by_regime": {
            regime: float(
                np.mean(
                    vertical_values["weighted"][regimes == regime]
                    - base_values["weighted"][regimes == regime]
                )
            )
            for regime in ("clear", "light", "medium", "heavy")
            if np.any(regimes == regime)
        },
        "render": render,
        "leakage_assertions": {
            "query_point_overlap_with_fit_ids": int(
                np.intersect1d(query_ids, fit_ids).size
            ),
            "query_h_used_for_prediction": False,
            "query_vertical_profile_source": "outer-fit KNN official H marginals",
            "query_h_used_for_oracle_attribution_only": True,
        },
    }
    arrays = {
        "ids": query_ids,
        "baseline": baseline,
        **prediction,
        "truth": truth,
        "support": support,
        "vertical_score": vertical,
        "shuffled_score": shuffled,
        "oracle_vertical_score": oracle_vertical,
    }
    return row, arrays


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    ap_lift = np.asarray(
        [
            row["existence"]["vertical"]["average_precision"]
            - row["existence"]["competition"]["average_precision"]
            for row in rows
        ]
    )
    ap_control = np.asarray(
        [
            row["existence"]["vertical"]["average_precision"]
            - row["existence"]["shuffled"]["average_precision"]
            for row in rows
        ]
    )
    top1_lift = np.asarray(
        [
            row["collision_top1"]["vertical"]["mass_weighted_accuracy"]
            - row["collision_top1"]["competition"]["mass_weighted_accuracy"]
            for row in rows
        ]
    )
    joint = np.asarray(
        [row["spectrum"]["vertical"]["delta"]["weighted"] for row in rows]
    )
    joint_vs_independent = np.asarray(
        [
            row["spectrum"]["vertical"]["metrics"]["weighted"]
            - row["spectrum"]["independent"]["metrics"]["weighted"]
            for row in rows
        ]
    )
    joint_vs_shuffle = np.asarray(
        [
            row["spectrum"]["vertical"]["metrics"]["weighted"]
            - row["spectrum"]["shuffled"]["metrics"]["weighted"]
            for row in rows
        ]
    )
    oracle_ap_lift = np.asarray(
        [
            row["existence"]["oracle_vertical"]["average_precision"]
            - row["existence"]["competition"]["average_precision"]
            for row in rows
        ]
    )
    oracle_top1_lift = np.asarray(
        [
            row["collision_top1"]["oracle_vertical"]["mass_weighted_accuracy"]
            - row["collision_top1"]["competition"]["mass_weighted_accuracy"]
            for row in rows
        ]
    )
    oracle_joint_lift = np.asarray(
        [
            row["spectrum"]["oracle_vertical"]["metrics"]["weighted"]
            - row["spectrum"]["competition"]["metrics"]["weighted"]
            for row in rows
        ]
    )
    passed = bool(
        np.mean(ap_lift) >= MIN_AP_LIFT
        and np.mean(top1_lift) >= MIN_COLLISION_LIFT
        and np.mean(joint_vs_shuffle) >= MIN_IDENTITY_CONTROL_LIFT
        and np.mean(joint) >= MIN_JOINT_DELTA
        and np.sum(joint > 0.0) >= MIN_JOINT_WINS
    )
    return {
        "vertical_ap_lift_vs_competition": summarize(ap_lift),
        "vertical_ap_lift_vs_shuffled": summarize(ap_control),
        "collision_mass_top1_lift": summarize(top1_lift),
        "vertical_joint_delta_vs_full_spectrum_knn": summarize(joint),
        "vertical_joint_lift_vs_independent": summarize(joint_vs_independent),
        "vertical_joint_lift_vs_shuffled": summarize(joint_vs_shuffle),
        "oracle_vertical_ap_lift_vs_competition": summarize(oracle_ap_lift),
        "oracle_collision_mass_top1_lift": summarize(oracle_top1_lift),
        "oracle_vertical_joint_lift_vs_competition": summarize(oracle_joint_lift),
        "joint_fold_wins": int(np.sum(joint > 0.0)),
        "passed": passed,
    }


def plot_report(report: dict[str, object]) -> None:
    configure_chinese_font()
    rows = report["fold_rows"]
    fold = np.arange(len(rows))
    methods = (
        "independent",
        "competition",
        "vertical",
        "shuffled",
        "oracle_vertical",
    )
    labels = (
        "独立候选",
        "仅互斥竞争",
        "垂直指纹竞争",
        "打乱指纹竞争",
        "真实查询垂直谱归因",
    )
    colors = ("#777777", "#3976a8", "#cf7a27", "#9b6aae", "#338b57")
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    for method, label, color in zip(methods, labels, colors):
        axes[0].plot(
            fold,
            [row["existence"][method]["average_precision"] for row in rows],
            "o-",
            color=color,
            label=label,
        )
        axes[1].plot(
            fold,
            [
                row["collision_top1"][method]["mass_weighted_accuracy"]
                for row in rows
            ],
            "o-",
            color=color,
            label=label,
        )
        axes[2].plot(
            fold,
            [row["spectrum"][method]["delta"]["weighted"] for row in rows],
            "o-",
            color=color,
            label=label,
        )
    axes[0].set_title("路径存在平均精确率")
    axes[1].set_title("冲突组质量加权首选正确率")
    axes[2].set_title("相对普通全谱近邻的联合谱增益")
    axes[2].axhline(0.0, color="#333333", linewidth=0.8)
    for axis in axes:
        axis.set_xlabel("空间折")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("v404 同 delay-angle 候选的垂直阵列指纹竞争", fontsize=15)
    figure.savefig(FIGURE, dpi=190, facecolor="white")
    plt.close(figure)


def main() -> None:
    required = (
        TARGET_CACHE,
        GEOMETRY_CACHE,
        v297.COMPONENT_CACHE,
        v304.REPORT,
        v305.MATCH_CACHE,
        v306.TRAIN_CANDIDATES,
        v306.SCENE_CACHE,
        v318.DATA_CACHE,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs: " + ", ".join(missing))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    target_data = np.load(TARGET_CACHE)
    targets = {key: target_data[key].astype(np.float32) for key in ("sh", "shv")}
    targets["sh"] = v402.normalize(targets["sh"])
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    blocked = np.load(GEOMETRY_CACHE)["train_blocked"].astype(np.float32)
    candidate_data = np.load(v306.TRAIN_CANDIDATES)
    candidate = {key: candidate_data[key] for key in candidate_data.files}
    valid, hard_mass = v328.build_labels(candidate)
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(
        target_data["valid"].astype(bool) & component["valid"].any(axis=1)
    ).astype(np.int64)
    folds = v315.disjoint_folds(valid_ids, positions[:, :2])
    fingerprint_data = v318.build_dataset()
    patches = np.load(v306.SCENE_CACHE)["wall_patches"].astype(np.float32)
    slope, intercept = v403.calibration_parameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict[str, object]] = []
    arrays: list[dict[str, np.ndarray]] = []
    for fold, query_ids in enumerate(folds):
        row, part = fold_experiment(
            fold,
            query_ids,
            valid_ids,
            positions,
            blocked,
            targets,
            candidate,
            valid,
            hard_mass,
            fingerprint_data,
            patches,
            slope,
            intercept,
            device,
        )
        rows.append(row)
        arrays.append(part)
        print(
            f"v404 fold={fold} AP={row['existence']['vertical']['average_precision'] - row['existence']['competition']['average_precision']:+.5f} "
            f"top1={row['collision_top1']['vertical']['mass_weighted_accuracy'] - row['collision_top1']['competition']['mass_weighted_accuracy']:+.5f} "
            f"joint={row['spectrum']['vertical']['delta']['weighted']:+.6f}",
            flush=True,
        )

    summary = aggregate(rows)
    ids = np.concatenate([part["ids"] for part in arrays])
    if len(ids) != len(np.unique(ids)) or len(ids) != len(valid_ids):
        raise AssertionError("OOF IDs are not a unique cover of valid points")
    np.savez_compressed(
        OOF_CACHE,
        ids=ids,
        fold=np.concatenate(
            [
                np.full(len(part["ids"]), fold, dtype=np.int16)
                for fold, part in enumerate(arrays)
            ]
        ),
        **{
            key: np.concatenate([part[key] for part in arrays])
            for key in (
                "baseline",
                "independent",
                "competition",
                "vertical",
                "shuffled",
                "oracle_vertical",
                "truth",
                "support",
                "vertical_score",
                "shuffled_score",
                "oracle_vertical_score",
            )
        },
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A fit-only KNN prediction of the query vertical array spectrum provides the "
            "missing query-side observable needed to select among point-cloud candidates "
            "that collide in delay and horizontal angle."
        ),
        "compliance": {
            "uses": [
                "official train H joint and vertical spectra",
                "official train coordinates",
                "official Round1 point-cloud reflection geometry",
            ],
            "does_not_use": [
                "query/outer H as deployable prediction input",
                "test H or hidden truth",
                "v212/Sionna/RT/synthetic target or online labels",
            ],
        },
        "isolation": {
            "spatial_predictor": "same patch-specific hard-label KNN for every method",
            "collision_budget": "maximum original support in each rounded (S,H) group",
            "identity_signal": "geometry-predicted vertical fingerprint cosine with outer-fit KNN query marginal",
            "control": "same identity values shuffled within each query point",
            "oracle": "outer true S-H-V marginal used only for attribution after prediction",
        },
        "parameters": {
            "fingerprint_model": "frozen v319 VerticalMLP and 100-epoch recipe",
            "collision_key": "rounded periodic S and H bins",
            "maximum_rendered_paths": MAX_PATHS,
            "folds": "five mutually exclusive isolation-stratified folds",
        },
        "fold_rows": rows,
        "aggregate": summary,
        "decision_rule": {
            "vertical_ap_lift_vs_competition": f">={MIN_AP_LIFT}",
            "collision_mass_top1_lift": f">={MIN_COLLISION_LIFT}",
            "joint_lift_vs_shuffled": f">={MIN_IDENTITY_CONTROL_LIFT}",
            "joint_delta_vs_full_spectrum_knn": f">={MIN_JOINT_DELTA}",
            "joint_fold_wins": f">={MIN_JOINT_WINS}/5",
            "passed": bool(summary["passed"]),
        },
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "figure": PATHS.relative(FIGURE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_report(report)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

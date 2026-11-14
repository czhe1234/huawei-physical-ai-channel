#!/usr/bin/env python3
"""Gate complete v546 path-action sets by their predicted point-level net gain."""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

import analyze_v545_path_block_family_oracle as v545
import analyze_v546_local_global_path_event_model as v546
import generate_v543_disagreement_conditioned_cell_fusion as v543
from project_paths import PATHS


VERSION_NUMBER = 547
VERSION = "v547_pointwise_path_action_gate"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
FIRST_LAYER_CACHE = CACHE_DIR / "v547_replayed_path_event_predictions.npz"
FEATURE_CACHE = CACHE_DIR / "v547_point_gate_features.npz"
OOF_CACHE = CACHE_DIR / "v547_pointwise_path_action_gate_oof.npz"
OUTPUT_DIR = PATHS.analysis / "visualizations" / "整点路径动作净收益门控"

FIGURES = {
    "ladder": OUTPUT_DIR / "01_整点门控与路径动作上限.png",
    "folds": OUTPUT_DIR / "02_五折门控增量与负控制.png",
    "regimes": OUTPUT_DIR / "03_传播区域收益与激活率.png",
    "calibration": OUTPUT_DIR / "04_预测净收益与真实净收益.png",
    "heavy": OUTPUT_DIR / "05_重遮挡区域门控效果空间分布.png",
}

HEIGHT = v545.HEIGHT
WIDTH = v545.WIDTH
SOURCE_COUNT = v545.SOURCE_COUNT
REGIMES = v545.REGIMES
REGIME_CN = v545.REGIME_CN
SEED = 54701
GATE_TREES = 700
MIN_SAMPLES_LEAF = 12
MAX_FEATURES = 0.60
FLOOR = 1e-10


def configure_chinese_font() -> None:
    selected = "DejaVu Sans"
    for family in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=family),
                fallback_to_default=False,
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
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 260,
        }
    )


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return value / np.maximum(value.sum(axis=1, keepdims=True), 1e-30)


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    right = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
    return (numerator / np.maximum(denominator, 1e-30)).astype(np.float32)


def row_entropy(value: np.ndarray) -> np.ndarray:
    value = normalize_rows(value)
    result = -np.sum(
        value.astype(np.float64) * np.log(np.maximum(value, 1e-30)), axis=1
    )
    return (result / math.log(value.shape[1])).astype(np.float32)


def row_js(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_rows(left).astype(np.float64)
    right = normalize_rows(right).astype(np.float64)
    middle = 0.5 * (left + right)
    left_kl = np.sum(
        left * np.log(np.maximum(left, 1e-30) / np.maximum(middle, 1e-30)), axis=1
    )
    right_kl = np.sum(
        right * np.log(np.maximum(right, 1e-30) / np.maximum(middle, 1e-30)), axis=1
    )
    return (0.5 * (left_kl + right_kl)).astype(np.float32)


def top_mass(value: np.ndarray, count: int) -> np.ndarray:
    value = normalize_rows(value)
    count = min(count, value.shape[1])
    return np.partition(value, -count, axis=1)[:, -count:].sum(axis=1)


def validate_inputs() -> None:
    required = (
        v545.OOF_SOURCE,
        v545.BANK_CACHE,
        v545.PATH_META,
        v545.TRAIN_POWER,
        v546.EVENT_CACHE,
        v546.OOF_CACHE,
    )
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing v547 inputs: {missing}")


def load_common_event_features(
    records: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.load(v545.OOF_SOURCE)
    fold_id = oof["fold_id"].astype(np.int16)
    static_train, _, _ = v543.load_static_features()
    common = np.concatenate(
        [
            static_train[records["point"]],
            records["path_features"],
            records["block_features"],
            records["candidate_features"],
        ],
        axis=1,
    ).astype(np.float32)
    common = np.nan_to_num(common, nan=0.0, posinf=1e6, neginf=-1e6)
    return common, fold_id


def derive_event_actions(predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    utility = predicted[:, :SOURCE_COUNT]
    choices = np.argmax(utility, axis=1).astype(np.int16)
    margin = utility[np.arange(len(choices)), choices]
    active = margin > 0.0
    gains = np.exp(
        np.clip(predicted[np.arange(len(choices)), SOURCE_COUNT + choices], -1.0, 1.0)
        * v546.LOG_GAIN_SCALE
    ).astype(np.float32)
    gains = np.clip(gains, v545.GAIN_MIN, v545.GAIN_MAX)
    choices[~active] = -1
    gains[~active] = 1.0
    return choices, gains


def replay_first_layer(
    records: dict[str, np.ndarray],
    positions: np.ndarray,
    common: np.ndarray,
    fold_id: np.ndarray,
) -> np.ndarray:
    if FIRST_LAYER_CACHE.is_file():
        cached = np.load(FIRST_LAYER_CACHE)
        predicted = cached["predicted"].astype(np.float32)
        if predicted.shape != (len(records["point"]), 2 * SOURCE_COUNT):
            raise ValueError(f"invalid first-layer cache shape: {predicted.shape}")
        return predicted

    record_fold = fold_id[records["point"]]
    target = np.concatenate(
        [records["utility_model"], records["log_gain"]], axis=1
    ).astype(np.float32)
    predicted = np.zeros_like(target)
    for outer in range(5):
        fit_mask = record_fold != outer
        query_mask = record_fold == outer
        local = v546.local_target_features(
            records,
            positions,
            fit_mask,
            records["utility_model"],
            records["log_gain"],
        )
        features = np.concatenate([common, local], axis=1).astype(np.float32)
        model = v546.make_model(v546.SEED + 100 + outer, v546.MODEL_TREES)
        model.fit(
            features[fit_mask],
            target[fit_mask],
            sample_weight=records["event_weight"][fit_mask],
        )
        predicted[query_mask] = model.predict(features[query_mask]).astype(np.float32)
        print(f"v547 replay first layer fold {outer + 1}/5", flush=True)

    choices, gains = derive_event_actions(predicted)
    frozen = np.load(v546.OOF_CACHE)
    frozen_choices = frozen["v543_choices"].astype(np.int16)
    frozen_gains = frozen["v543_gains"].astype(np.float32)
    mismatch = int(np.sum(choices != frozen_choices))
    gain_error = float(np.max(np.abs(gains - frozen_gains)))
    if mismatch or gain_error > 1e-6:
        raise ValueError(
            f"first-layer replay mismatch: choices={mismatch}, gain_error={gain_error}"
        )
    np.savez_compressed(
        FIRST_LAYER_CACHE,
        predicted=predicted,
        choices=choices,
        gains=gains,
        choice_mismatch=np.asarray(mismatch, dtype=np.int32),
        maximum_gain_error=np.asarray(gain_error, dtype=np.float32),
    )
    return predicted


def aggregate_event_features(
    records: dict[str, np.ndarray],
    predicted: np.ndarray,
    choices: np.ndarray,
    gains: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    point_count = 2000
    utility = predicted[:, :SOURCE_COUNT].astype(np.float32)
    candidate = records["candidate_features"].reshape(-1, SOURCE_COUNT, 10)
    blocks: list[np.ndarray] = []
    names: list[str] = []

    basic = np.zeros((point_count, 18), dtype=np.float32)
    source_fraction = np.zeros((point_count, SOURCE_COUNT), dtype=np.float32)
    source_cell_fraction = np.zeros_like(source_fraction)
    utility_mean = np.zeros_like(source_fraction)
    utility_max = np.zeros_like(source_fraction)
    path_mean = np.zeros((point_count, v546.PATH_FEATURE_DIM), dtype=np.float32)
    path_std = np.zeros_like(path_mean)
    block_mean = np.zeros(
        (point_count, records["block_features"].shape[1]), dtype=np.float32
    )
    block_std = np.zeros_like(block_mean)
    selected_mean = np.zeros((point_count, 10), dtype=np.float32)
    selected_std = np.zeros_like(selected_mean)

    order = np.argsort(records["point"], kind="stable")
    sorted_point = records["point"][order]
    boundaries = np.flatnonzero(np.diff(sorted_point)) + 1
    for group in np.split(order, boundaries):
        if not len(group):
            continue
        point = int(records["point"][group[0]])
        local_choice = choices[group]
        active = local_choice >= 0
        cells = records["cells"][group].astype(np.float64)
        total_cells = max(float(cells.sum()), 1.0)
        top = np.max(utility[group], axis=1)
        second = np.partition(utility[group], -2, axis=1)[:, -2]
        selected_gain = gains[group]
        log_gain = np.log(np.maximum(selected_gain, 1e-30))
        basic[point] = np.asarray(
            [
                len(group),
                float(cells.sum()) / (HEIGHT * WIDTH),
                float(np.mean(cells)) / (HEIGHT * WIDTH),
                float(np.std(cells)) / (HEIGHT * WIDTH),
                float(np.max(cells)) / (HEIGHT * WIDTH),
                float(np.mean(active)),
                float(cells[active].sum()) / total_cells,
                float(np.mean(top)),
                float(np.std(top)),
                float(np.min(top)),
                float(np.max(top)),
                float(np.mean(top - second)),
                float(np.std(top - second)),
                float(np.mean(np.mean(utility[group] > 0.0, axis=1))),
                float(np.mean(log_gain)),
                float(np.std(log_gain)),
                float(np.min(log_gain)),
                float(np.max(log_gain)),
            ],
            dtype=np.float32,
        )
        for source in range(SOURCE_COUNT):
            selected = local_choice == source
            source_fraction[point, source] = float(np.mean(selected))
            source_cell_fraction[point, source] = (
                float(cells[selected].sum()) / total_cells
            )
        utility_mean[point] = np.mean(utility[group], axis=0)
        utility_max[point] = np.max(utility[group], axis=0)
        path_mean[point] = np.mean(records["path_features"][group], axis=0)
        path_std[point] = np.std(records["path_features"][group], axis=0)
        block_mean[point] = np.mean(records["block_features"][group], axis=0)
        block_std[point] = np.std(records["block_features"][group], axis=0)
        selected_rows = np.flatnonzero(active)
        if len(selected_rows):
            selected_features = candidate[
                group[selected_rows], local_choice[selected_rows]
            ]
            selected_mean[point] = np.mean(selected_features, axis=0)
            selected_std[point] = np.std(selected_features, axis=0)

    blocks.extend(
        [
            basic,
            source_fraction,
            source_cell_fraction,
            utility_mean,
            utility_max,
            path_mean,
            path_std,
            block_mean,
            block_std,
            selected_mean,
            selected_std,
        ]
    )
    names.extend(
        [
            "路径数",
            "路径覆盖格比例",
            "平均路径格比例",
            "路径格比例标准差",
            "最大路径格比例",
            "激活路径比例",
            "激活格比例",
            "预测效用均值",
            "预测效用标准差",
            "预测效用最小值",
            "预测效用最大值",
            "候选效用间隔均值",
            "候选效用间隔标准差",
            "正效用候选比例",
            "预测能量对数均值",
            "预测能量对数标准差",
            "预测能量对数最小值",
            "预测能量对数最大值",
        ]
    )
    names.extend([f"候选{source}_路径比例" for source in range(SOURCE_COUNT)])
    names.extend([f"候选{source}_格比例" for source in range(SOURCE_COUNT)])
    names.extend([f"候选{source}_效用均值" for source in range(SOURCE_COUNT)])
    names.extend([f"候选{source}_效用最大值" for source in range(SOURCE_COUNT)])
    names.extend([f"路径特征{index}_均值" for index in range(path_mean.shape[1])])
    names.extend([f"路径特征{index}_标准差" for index in range(path_std.shape[1])])
    names.extend([f"路径块特征{index}_均值" for index in range(block_mean.shape[1])])
    names.extend([f"路径块特征{index}_标准差" for index in range(block_std.shape[1])])
    names.extend(
        [f"所选候选特征{index}_均值" for index in range(selected_mean.shape[1])]
    )
    names.extend(
        [f"所选候选特征{index}_标准差" for index in range(selected_std.shape[1])]
    )
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def pair_features(
    left: np.ndarray,
    right: np.ndarray,
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    left = normalize_rows(left)
    right = normalize_rows(right)
    left_grid = left.reshape(-1, HEIGHT, WIDTH)
    right_grid = right.reshape(-1, HEIGHT, WIDTH)
    left_delay = normalize_rows(left_grid.sum(axis=2))
    right_delay = normalize_rows(right_grid.sum(axis=2))
    left_angle = normalize_rows(left_grid.sum(axis=1))
    right_angle = normalize_rows(right_grid.sum(axis=1))
    values = np.column_stack(
        [
            row_cosine(left, right),
            0.5 * np.sum(np.abs(left - right), axis=1),
            row_js(left, right),
            np.max(np.abs(left - right), axis=1),
            row_cosine(left_delay, right_delay),
            0.5 * np.sum(np.abs(left_delay - right_delay), axis=1),
            row_js(left_delay, right_delay),
            row_cosine(left_angle, right_angle),
            0.5 * np.sum(np.abs(left_angle - right_angle), axis=1),
            row_js(left_angle, right_angle),
            row_entropy(left) - row_entropy(right),
            row_entropy(left_delay) - row_entropy(right_delay),
            row_entropy(left_angle) - row_entropy(right_angle),
            top_mass(left, 16) - top_mass(right, 16),
            top_mass(left, 64) - top_mass(right, 64),
            top_mass(left, 256) - top_mass(right, 256),
        ]
    ).astype(np.float32)
    names = [
        "联合余弦",
        "联合总变差",
        "联合JS散度",
        "联合最大差",
        "时延余弦",
        "时延总变差",
        "时延JS散度",
        "角度余弦",
        "角度总变差",
        "角度JS散度",
        "联合熵差",
        "时延熵差",
        "角度熵差",
        "前16格质量差",
        "前64格质量差",
        "前256格质量差",
    ]
    return values, [f"{prefix}_{name}" for name in names]


def spectral_action_features(
    base: np.ndarray,
    action: np.ndarray,
    anchor: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for left, right, prefix in (
        (action, base, "动作对v543"),
        (base, anchor, "v543对物理锚点"),
        (action, anchor, "动作对物理锚点"),
    ):
        values, local_names = pair_features(left, right, prefix)
        blocks.append(values)
        names.extend(local_names)

    base_grid = normalize_rows(base).reshape(-1, HEIGHT, WIDTH)
    action_grid = normalize_rows(action).reshape(-1, HEIGHT, WIDTH)
    base_coarse = base_grid.reshape(-1, 12, 16, 8, 2).sum(axis=(2, 4))
    action_coarse = action_grid.reshape(-1, 12, 16, 8, 2).sum(axis=(2, 4))
    coarse_delta = (action_coarse - base_coarse).reshape(len(base), -1)
    coarse_abs = np.abs(coarse_delta)
    blocks.extend([coarse_delta.astype(np.float32), coarse_abs.astype(np.float32)])
    names.extend([f"粗谱有符号变化_{index}" for index in range(coarse_delta.shape[1])])
    names.extend([f"粗谱绝对变化_{index}" for index in range(coarse_abs.shape[1])])
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def build_point_features(
    records: dict[str, np.ndarray],
    predicted: np.ndarray,
    choices: np.ndarray,
    gains: np.ndarray,
    base: np.ndarray,
    action: np.ndarray,
    anchor: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    if FEATURE_CACHE.is_file():
        cached = np.load(FEATURE_CACHE)
        features = cached["features"].astype(np.float32)
        names = cached["feature_names"].astype(str).tolist()
        return features, names

    static, _, static_names = v543.load_static_features()
    event, event_names = aggregate_event_features(records, predicted, choices, gains)
    spectral, spectral_names = spectral_action_features(base, action, anchor)
    features = np.concatenate([static, event, spectral], axis=1).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    names = list(static_names) + event_names + spectral_names
    if features.shape[1] != len(names):
        raise ValueError("feature name count mismatch")
    np.savez_compressed(
        FEATURE_CACHE,
        features=features,
        feature_names=np.asarray(names, dtype="U96"),
    )
    return features, names


def make_regressor(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=GATE_TREES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=False,
        n_jobs=-1,
        random_state=seed,
    )


def make_classifier(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=GATE_TREES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=False,
        n_jobs=-1,
        random_state=seed,
    )


def fit_gate(
    features: np.ndarray,
    true_gain: np.ndarray,
    fold_id: np.ndarray,
    regimes: np.ndarray,
    valid: np.ndarray,
    *,
    by_regime: bool,
    shuffled: bool,
    seed_offset: int,
) -> dict[str, np.ndarray]:
    predicted_gain = np.zeros(len(features), dtype=np.float32)
    positive_probability = np.zeros(len(features), dtype=np.float32)
    rng = np.random.default_rng(SEED + seed_offset)
    groups = REGIMES if by_regime else ("all",)

    for outer in range(5):
        for group_index, group in enumerate(groups):
            group_mask = np.ones(len(features), dtype=bool)
            if group != "all":
                group_mask = regimes == group
            fit = valid & (fold_id != outer) & group_mask
            query = valid & (fold_id == outer) & group_mask
            fit_ids = np.flatnonzero(fit)
            query_ids = np.flatnonzero(query)
            if not len(fit_ids) or not len(query_ids):
                continue
            target = true_gain[fit_ids].copy()
            if shuffled:
                target = target[rng.permutation(len(target))]
            label = target > 0.0
            regressor = make_regressor(
                SEED + seed_offset + outer * 17 + group_index * 101
            )
            regressor.fit(features[fit_ids], target)
            predicted_gain[query_ids] = regressor.predict(features[query_ids]).astype(
                np.float32
            )
            if np.unique(label).size == 1:
                positive_probability[query_ids] = float(label[0])
            else:
                classifier = make_classifier(
                    SEED + seed_offset + 10000 + outer * 17 + group_index * 101
                )
                classifier.fit(features[fit_ids], label)
                positive_probability[query_ids] = classifier.predict_proba(
                    features[query_ids]
                )[:, 1].astype(np.float32)
        print(
            f"v547 gate {'regime' if by_regime else 'global'}"
            f"{' shuffled' if shuffled else ''} fold {outer + 1}/5",
            flush=True,
        )

    active = valid & (predicted_gain > 0.0) & (positive_probability > 0.5)
    return {
        "predicted_gain": predicted_gain,
        "positive_probability": positive_probability,
        "active": active,
    }


def apply_gate(base: np.ndarray, action: np.ndarray, active: np.ndarray) -> np.ndarray:
    output = np.asarray(base, dtype=np.float32).copy()
    output[active] = action[active]
    return normalize_rows(output)


def score_variant(
    truth: np.ndarray,
    base: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    regimes: np.ndarray,
    fold_id: np.ndarray,
    active: np.ndarray,
) -> dict[str, object]:
    base_summary, base_metrics = v545.score_prediction(truth, base, valid, regimes)
    summary, metrics = v545.score_prediction(truth, prediction, valid, regimes)
    point_gain = metrics["weighted"] - base_metrics["weighted"]
    return {
        "summary": summary,
        "gain": float(
            summary["overall"]["weighted"] - base_summary["overall"]["weighted"]
        ),
        "active_fraction": float(np.mean(active[valid])),
        "active_count": int(np.sum(active & valid)),
        "fold_gain": [
            float(np.mean(point_gain[valid & (fold_id == fold)])) for fold in range(5)
        ],
        "regime_gain": {
            regime: float(np.mean(point_gain[valid & (regimes == regime)]))
            for regime in REGIMES
        },
        "regime_active_fraction": {
            regime: float(np.mean(active[valid & (regimes == regime)]))
            for regime in REGIMES
        },
        "point_gain": point_gain,
    }


def load_background() -> dict[str, object]:
    scene = np.load(v545.SCENE)
    wall = np.log1p(scene["grid_vertical_count"].astype(np.float32))
    positive = wall[wall > 0]
    x_min = float(scene["x_min"])
    y_min = float(scene["y_min"])
    resolution = float(scene["resolution"])
    return {
        "wall": wall,
        "extent": (
            x_min,
            x_min + wall.shape[1] * resolution,
            y_min,
            y_min + wall.shape[0] * resolution,
        ),
        "vmax": max(float(np.quantile(positive, 0.98)), 1.0),
    }


def decorate_map(axis: plt.Axes, background: dict[str, object]) -> None:
    axis.imshow(
        background["wall"],
        origin="lower",
        extent=background["extent"],
        cmap="Greys",
        vmin=0.0,
        vmax=background["vmax"],
        alpha=0.20,
        interpolation="nearest",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("横坐标（米）")
    axis.set_ylabel("纵坐标（米）")


def plot_results(
    base_score: float,
    action_score: float,
    global_result: dict[str, object],
    regime_result: dict[str, object],
    control_result: dict[str, object],
    oracle_score: float,
    true_gain: np.ndarray,
    gate: dict[str, np.ndarray],
    positions: np.ndarray,
    valid: np.ndarray,
    regimes: np.ndarray,
) -> None:
    labels = [
        "v543",
        "全部路径动作",
        "全局点级门控",
        "分区点级门控",
        "打乱收益控制",
        "逐点选择上限",
    ]
    values = [
        base_score,
        action_score,
        float(global_result["summary"]["overall"]["weighted"]),
        float(regime_result["summary"]["overall"]["weighted"]),
        float(control_result["summary"]["overall"]["weighted"]),
        oracle_score,
    ]
    colors = ["#64748b", "#dc2626", "#0891b2", "#16a34a", "#f59e0b", "#7c3aed"]
    figure, axis = plt.subplots(figsize=(12.8, 6.2), constrained_layout=True)
    bars = axis.bar(np.arange(len(values)), values, color=colors)
    axis.bar_label(bars, fmt="%.5f", padding=3)
    axis.set_xticks(np.arange(len(values)), labels, rotation=12)
    axis.set_ylim(max(0.79, min(values) - 0.02), min(0.87, max(values) + 0.015))
    axis.set_ylabel("训练集严格OOF联合谱综合相似度")
    axis.grid(axis="y", alpha=0.24)
    axis.set_title("整点净收益门控能否保留有效路径动作", fontsize=15, fontweight="bold")
    figure.savefig(FIGURES["ladder"], bbox_inches="tight")
    plt.close(figure)

    folds = np.arange(1, 6)
    figure, axis = plt.subplots(figsize=(10.8, 5.8), constrained_layout=True)
    for result, label, color in (
        (global_result, "全局门控", "#0891b2"),
        (regime_result, "分区门控", "#16a34a"),
        (control_result, "传播区内打乱收益", "#f59e0b"),
    ):
        axis.plot(
            folds,
            result["fold_gain"],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(folds)
    axis.set_xlabel("空间折")
    axis.set_ylabel("相对v543的综合分增量")
    axis.grid(alpha=0.24)
    axis.legend()
    axis.set_title("五个互斥空间折中的门控稳定性", fontsize=15, fontweight="bold")
    figure.savefig(FIGURES["folds"], bbox_inches="tight")
    plt.close(figure)

    x = np.arange(len(REGIMES))
    width = 0.26
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), constrained_layout=True)
    for offset, result, label, color in (
        (-width, global_result, "全局门控", "#0891b2"),
        (0.0, regime_result, "分区门控", "#16a34a"),
        (width, control_result, "打乱控制", "#f59e0b"),
    ):
        axes[0].bar(
            x + offset,
            [result["regime_gain"][regime] for regime in REGIMES],
            width,
            label=label,
            color=color,
        )
        axes[1].bar(
            x + offset,
            [result["regime_active_fraction"][regime] for regime in REGIMES],
            width,
            label=label,
            color=color,
        )
    for axis in axes:
        axis.set_xticks(x, [REGIME_CN[value] for value in REGIMES])
        axis.grid(axis="y", alpha=0.24)
        axis.legend()
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_ylabel("相对v543的综合分增量")
    axes[0].set_title("不同传播区域的真实收益", fontweight="bold")
    axes[1].set_ylabel("执行完整路径动作的点比例")
    axes[1].set_title("门控实际激活率", fontweight="bold")
    figure.suptitle(
        "门控是否学会区分传播区域与点内风险", fontsize=15, fontweight="bold"
    )
    figure.savefig(FIGURES["regimes"], bbox_inches="tight")
    plt.close(figure)

    predicted_gain = gate["predicted_gain"]
    probability = gate["positive_probability"]
    active = gate["active"]
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), constrained_layout=True)
    sample = valid
    axes[0].scatter(
        predicted_gain[sample],
        true_gain[sample],
        c=np.where(active[sample], "#16a34a", "#94a3b8"),
        s=12,
        alpha=0.55,
        edgecolors="none",
    )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].axvline(0.0, color="black", linewidth=1)
    axes[0].set_xlabel("模型预测整点净收益")
    axes[0].set_ylabel("真实OOF整点净收益")
    axes[0].set_title("回归门控校准", fontweight="bold")
    axes[0].grid(alpha=0.22)
    bins = np.linspace(0.0, 1.0, 11)
    center: list[float] = []
    observed: list[float] = []
    count: list[int] = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        local = sample & (probability >= lower) & (probability < upper)
        if np.sum(local) < 5:
            continue
        center.append(0.5 * (lower + upper))
        observed.append(float(np.mean(true_gain[local] > 0.0)))
        count.append(int(np.sum(local)))
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#64748b")
    scatter = axes[1].scatter(center, observed, c=count, cmap="viridis", s=65)
    axes[1].set_xlabel("预测正收益概率")
    axes[1].set_ylabel("实际正收益比例")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("分类门控校准", fontweight="bold")
    axes[1].grid(alpha=0.22)
    figure.colorbar(scatter, ax=axes[1], label="区间样本数")
    figure.suptitle(
        "第二层是否真正预测了完整动作净收益", fontsize=15, fontweight="bold"
    )
    figure.savefig(FIGURES["calibration"], bbox_inches="tight")
    plt.close(figure)

    heavy = valid & (regimes == "heavy")
    realized = np.where(active, true_gain, 0.0)
    limit = max(float(np.quantile(np.abs(true_gain[heavy]), 0.98)), 1e-4)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    background = load_background()
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), constrained_layout=True)
    for axis, values_local, title in (
        (axes[0], true_gain, "全部执行时的真实变化"),
        (axes[1], realized, "分区门控后的实际变化"),
    ):
        decorate_map(axis, background)
        scatter = axis.scatter(
            positions[heavy, 0],
            positions[heavy, 1],
            c=values_local[heavy],
            cmap="RdBu_r",
            norm=norm,
            s=20,
            edgecolors="none",
        )
        axis.set_title(title, fontweight="bold")
        figure.colorbar(scatter, ax=axis, label="逐点综合分增量")
    figure.suptitle("重遮挡点中门控保留了哪些路径动作", fontsize=15, fontweight="bold")
    figure.savefig(FIGURES["heavy"], bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    configure_chinese_font()
    validate_inputs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    oof = np.load(v545.OOF_SOURCE)
    truth = oof["truth"].astype(np.float32)
    anchor = oof["anchor"].astype(np.float32)
    base = oof["calibrated"].astype(np.float32)
    fold_id = oof["fold_id"].astype(np.int16)
    regimes = oof["regimes"].astype("U8")
    positions = np.load(v545.TRAIN_POS).astype(np.float32)
    power = np.load(v545.TRAIN_POWER).astype(np.float64)
    valid = power != 0.0
    v546_oof = np.load(v546.OOF_CACHE)
    action = v546_oof["v543_prediction"].astype(np.float32)
    records = v546.build_event_cache()
    common, replay_fold = load_common_event_features(records)
    if not np.array_equal(fold_id, replay_fold):
        raise ValueError("fold mismatch")
    predicted = replay_first_layer(records, positions, common, fold_id)
    choices, gains = derive_event_actions(predicted)
    features, feature_names = build_point_features(
        records, predicted, choices, gains, base, action, anchor
    )
    print(f"v547 point features={features.shape}", flush=True)

    base_summary, base_metrics = v545.score_prediction(truth, base, valid, regimes)
    action_summary, action_metrics = v545.score_prediction(
        truth, action, valid, regimes
    )
    true_gain = action_metrics["weighted"] - base_metrics["weighted"]
    oracle_active = valid & (true_gain > 0.0)
    oracle_prediction = apply_gate(base, action, oracle_active)
    oracle_summary, _ = v545.score_prediction(truth, oracle_prediction, valid, regimes)

    global_gate = fit_gate(
        features,
        true_gain,
        fold_id,
        regimes,
        valid,
        by_regime=False,
        shuffled=False,
        seed_offset=0,
    )
    regime_gate = fit_gate(
        features,
        true_gain,
        fold_id,
        regimes,
        valid,
        by_regime=True,
        shuffled=False,
        seed_offset=20000,
    )
    control_gate = fit_gate(
        features,
        true_gain,
        fold_id,
        regimes,
        valid,
        by_regime=True,
        shuffled=True,
        seed_offset=40000,
    )
    global_prediction = apply_gate(base, action, global_gate["active"])
    regime_prediction = apply_gate(base, action, regime_gate["active"])
    control_prediction = apply_gate(base, action, control_gate["active"])
    global_result = score_variant(
        truth,
        base,
        global_prediction,
        valid,
        regimes,
        fold_id,
        global_gate["active"],
    )
    regime_result = score_variant(
        truth,
        base,
        regime_prediction,
        valid,
        regimes,
        fold_id,
        regime_gate["active"],
    )
    control_result = score_variant(
        truth,
        base,
        control_prediction,
        valid,
        regimes,
        fold_id,
        control_gate["active"],
    )

    base_score = float(base_summary["overall"]["weighted"])
    action_score = float(action_summary["overall"]["weighted"])
    oracle_score = float(oracle_summary["overall"]["weighted"])
    regime_score = float(regime_result["summary"]["overall"]["weighted"])
    control_score = float(control_result["summary"]["overall"]["weighted"])
    gate_correlation = float(
        np.corrcoef(regime_gate["predicted_gain"][valid], true_gain[valid])[0, 1]
    )
    decision = {
        "v543_baseline_score": base_score,
        "ungated_path_action_score": action_score,
        "point_gate_oracle_score": oracle_score,
        "global_gate_score": float(global_result["summary"]["overall"]["weighted"]),
        "regime_gate_score": regime_score,
        "shuffled_control_score": control_score,
        "regime_gate_gain": regime_score - base_score,
        "gain_over_shuffled": regime_score - control_score,
        "predicted_vs_true_gain_pearson": gate_correlation,
        "heavy_gain": float(regime_result["regime_gain"]["heavy"]),
        "exceeds_v543": bool(regime_score > base_score),
        "heavy_not_harmed": bool(regime_result["regime_gain"]["heavy"] >= 0.0),
        "generate_test_candidate": bool(
            regime_score > base_score
            and regime_result["regime_gain"]["heavy"] >= 0.0
            and regime_score > control_score
        ),
    }

    plot_results(
        base_score,
        action_score,
        global_result,
        regime_result,
        control_result,
        oracle_score,
        true_gain,
        regime_gate,
        positions,
        valid,
        regimes,
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The final net gain of a complete set of OOF path actions is predictable from "
            "geometry, event composition and action-induced spectrum changes, allowing harmful "
            "heavy-region actions to fall back to v543 without hard-coding a light-only rule."
        ),
        "scope": {
            "train_points": 2000,
            "scored_nonzero_points": int(np.sum(valid)),
            "outer_spatial_folds": 5,
            "path_event_records": int(len(records["point"])),
            "point_feature_count": int(features.shape[1]),
            "test_data_used": False,
        },
        "model": {
            "type": "regime-conditioned ExtraTrees point-level hurdle gate",
            "trees_per_regressor_classifier": GATE_TREES,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "max_features": MAX_FEATURES,
            "activation_rule": "predicted net gain > 0 and positive-gain probability > 0.5",
            "action": "execute the complete frozen v546 path action or fall back to v543",
            "negative_control": "shuffle point-level net gains within each propagation regime",
        },
        "reference": {
            "v543": base_summary,
            "ungated_v546": action_summary,
            "point_gate_oracle": oracle_summary,
        },
        "global_gate": {
            key: value for key, value in global_result.items() if key != "point_gain"
        },
        "regime_gate": {
            key: value for key, value in regime_result.items() if key != "point_gain"
        },
        "shuffled_control": {
            key: value for key, value in control_result.items() if key != "point_gain"
        },
        "decision": decision,
        "outputs": {
            "first_layer": PATHS.relative(FIRST_LAYER_CACHE),
            "features": PATHS.relative(FEATURE_CACHE),
            "oof": PATHS.relative(OOF_CACHE),
            "figures": {key: PATHS.relative(path) for key, path in FIGURES.items()},
            "report": PATHS.relative(REPORT),
        },
    }
    np.savez_compressed(
        OOF_CACHE,
        fold_id=fold_id,
        valid=valid,
        regimes=regimes,
        true_gain=true_gain.astype(np.float32),
        global_predicted_gain=global_gate["predicted_gain"],
        global_probability=global_gate["positive_probability"],
        global_active=global_gate["active"],
        regime_predicted_gain=regime_gate["predicted_gain"],
        regime_probability=regime_gate["positive_probability"],
        regime_active=regime_gate["active"],
        control_predicted_gain=control_gate["predicted_gain"],
        control_probability=control_gate["positive_probability"],
        control_active=control_gate["active"],
        base_prediction=base,
        ungated_prediction=action,
        gated_prediction=regime_prediction,
        control_prediction=control_prediction,
    )
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Factor official spectra into visible physical paths and a learned residual."""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v300_path_gain_fingerprint_oracle as v300
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v309_unmatched_effective_scatterers as v309
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v325_path_phase_transport_identifiability as v325
import analyze_v328_patch_existence_support_field as v328
import analyze_v329_highprecision_complex_path_transport as v329
import analyze_v354_nested_virtual_source_transport as v354
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 402
VERSION = "v402_visible_path_residual_factorization"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
FIGURE_DIR = REPORT_DIR / "版本四零二可见路径分解图"
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v402_visible_path_residual_oof.npz"
FIGURE = FIGURE_DIR / "可见路径与剩余传播分解结果.png"

TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"

K_ANCHOR = 16
K_ENERGY = 8
DISTANCE_FLOOR = 0.25
MAX_PATCH_DISTANCE = 25.0
MAX_PATHS = 8
MAX_VISIBLE_MASS = 0.95
MIN_BREAKTHROUGH_DELTA = 0.01
MIN_FOLD_WINS = 4
METHODS = (
    "baseline",
    "residual_only",
    "oracle_visible_shape",
    "true_identity_true_energy",
    "true_identity_predicted_energy",
    "selected_true_oracle_energy",
    "deployable",
)
METHOD_LABELS = {
    "baseline": "普通全谱KNN",
    "residual_only": "仅剩余传播",
    "oracle_visible_shape": "真实可见路径形状+剩余传播",
    "true_identity_true_energy": "真实路径身份/能量+几何位置",
    "true_identity_predicted_energy": "真实路径身份+预测能量",
    "selected_true_oracle_energy": "预测存在性+真实命中能量",
    "deployable": "预测存在性/能量+剩余传播",
}
REGIMES = ("clear", "light", "medium", "heavy")


@dataclass(frozen=True)
class PathRow:
    kind: str
    patch: int
    candidate: int
    component: int
    mass: float
    observed_delay: float
    observed_h: float
    geometry_delay: float
    geometry_h: float


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


def normalize(value: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = out.sum(axis=(1, 2), keepdims=True, dtype=np.float64)
    return (out / np.maximum(total, 1e-30)).astype(np.float32)


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    y = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def point_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    pred = normalize(prediction)
    target = normalize(truth)
    sh = row_cosine(pred, target)
    pdp = row_cosine(pred.sum(axis=2), target.sum(axis=2))
    return {"sh": sh, "pdp": pdp, "weighted": 0.5 * (sh + pdp)}


def metric_summary(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    values = point_metrics(prediction, truth)
    return {key: float(np.mean(value)) for key, value in values.items()}


def neighbors(
    query_ids: np.ndarray, fit_ids: np.ndarray, xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    count = min(K_ANCHOR, len(fit_ids))
    distance, local = cKDTree(xy[fit_ids]).query(xy[query_ids], k=count)
    if count == 1:
        distance = distance[:, None]
        local = local[:, None]
    ids = fit_ids[local]
    weight = 1.0 / np.maximum(distance, DISTANCE_FLOOR) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    return ids.astype(np.int64), weight.astype(np.float32)


def knn_average(bank: np.ndarray, ids: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.sum(
        weight[:, :, None, None] * bank[ids], axis=1, dtype=np.float64
    ).astype(np.float32)


def exact_h_atom(center: float) -> np.ndarray:
    return v354.exact_atoms(np.asarray([center], dtype=np.float32), 16)[0]


def render_path(delay: float, h_profile: np.ndarray, mass: float) -> np.ndarray:
    d_atom = v354.exact_atoms(
        np.asarray([np.remainder(-float(delay), 192.0)], dtype=np.float32), 192
    )[0]
    h = np.maximum(np.asarray(h_profile, dtype=np.float32), 0.0)
    h /= np.maximum(np.sum(h, dtype=np.float64), 1e-30)
    return (float(mass) * d_atom[:, None] * h[None, :]).astype(np.float32)


def build_path_rows(
    train_pos: np.ndarray,
    blocked: np.ndarray,
    component: dict[str, np.ndarray],
    fingerprint: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> tuple[list[list[PathRow]], dict[str, object]]:
    inverse = np.load(v304.SCATTERER_CACHE)
    los_component = inverse["los_component"].astype(np.int64)
    h_center = v304.horizontal_center(fingerprint["horizontal"])
    calibration = json.loads(v304.REPORT.read_text(encoding="utf-8"))["los_calibration"]
    slope = float(calibration["slope_bins_per_meter"])
    intercept = float(calibration["intercept_bins"])
    paths: list[list[PathRow]] = [[] for _ in range(len(train_pos))]

    clear_ids = np.flatnonzero((blocked == 0) & component["valid"].any(axis=1))
    direct_length = np.linalg.norm(train_pos - v304.BS[None, :], axis=1)
    direct_azimuth = np.arctan2(
        train_pos[:, 1] - v304.BS[1], train_pos[:, 0] - v304.BS[0]
    )
    geometry_h = v304.expected_h_bin(direct_azimuth)
    for point_id in clear_ids:
        component_id = int(los_component[point_id])
        if not component["valid"][point_id, component_id]:
            continue
        paths[point_id].append(
            PathRow(
                kind="los",
                patch=-1,
                candidate=-1,
                component=component_id,
                mass=float(component["weight"][point_id, component_id]),
                observed_delay=float(component["delay"][point_id, component_id]),
                observed_h=float(h_center[point_id, component_id]),
                geometry_delay=intercept + slope * float(direct_length[point_id]),
                geometry_h=float(geometry_h[point_id]),
            )
        )

    matched = v325.matched_rows(v325.recover_center_gains())
    for row in range(len(matched["point"])):
        point_id = int(matched["point"][row])
        component_id = int(matched["component"][row])
        candidate_id = int(matched["candidate"][row])
        paths[point_id].append(
            PathRow(
                kind="reflection",
                patch=int(matched["patch"][row]),
                candidate=candidate_id,
                component=component_id,
                mass=float(matched["mass"][row]),
                observed_delay=float(component["delay"][point_id, component_id]),
                observed_h=float(h_center[point_id, component_id]),
                geometry_delay=intercept
                + slope * float(candidate["path_length"][point_id, candidate_id]),
                geometry_h=float(
                    v305.expected_h_bin(
                        candidate["reflection_point"][point_id, candidate_id][None]
                    )[0]
                ),
            )
        )
    return paths, {
        "los_path_count": int(sum(path.kind == "los" for rows in paths for path in rows)),
        "reflection_path_count": int(
            sum(path.kind == "reflection" for rows in paths for path in rows)
        ),
        "points_with_visible_path": int(sum(bool(rows) for rows in paths)),
    }


def decompose_visible(
    target: np.ndarray,
    paths: list[list[PathRow]],
    horizontal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    raw = np.zeros_like(target, dtype=np.float32)
    requested_mass = np.zeros(len(target), dtype=np.float32)
    for point_id, rows in enumerate(paths):
        for path in rows:
            raw[point_id] += render_path(
                path.observed_delay,
                horizontal[point_id, path.component],
                path.mass,
            )
            requested_mass[point_id] += path.mass
    visible = np.minimum(raw, target).astype(np.float32)
    residual = np.maximum(target - visible, 0.0).astype(np.float32)
    extracted = visible.sum(axis=(1, 2), dtype=np.float64)
    return visible, residual, {
        "requested_visible_mass_mean": float(np.mean(requested_mass)),
        "extracted_visible_mass_mean": float(np.mean(extracted)),
        "extracted_over_requested": float(
            np.sum(extracted) / np.maximum(np.sum(requested_mass), 1e-30)
        ),
        "points_with_extracted_visible_mass": int(np.sum(extracted > 1e-8)),
    }


class EnergyBank:
    def __init__(
        self,
        fit_ids: np.ndarray,
        xy: np.ndarray,
        paths: list[list[PathRow]],
    ) -> None:
        fit = set(int(value) for value in fit_ids)
        self.xy = xy
        self.los_point = np.asarray(
            [point for point in fit_ids if any(row.kind == "los" for row in paths[int(point)])],
            dtype=np.int64,
        )
        self.los_mass = np.asarray(
            [next(row.mass for row in paths[int(point)] if row.kind == "los") for point in self.los_point],
            dtype=np.float32,
        )
        self.los_tree = cKDTree(xy[self.los_point]) if len(self.los_point) else None
        patch_rows: dict[int, list[tuple[int, float]]] = {}
        for point_id, rows in enumerate(paths):
            if point_id not in fit:
                continue
            for row in rows:
                if row.kind == "reflection":
                    patch_rows.setdefault(row.patch, []).append((point_id, row.mass))
        self.patch: dict[int, tuple[np.ndarray, np.ndarray, cKDTree]] = {}
        for patch, values in patch_rows.items():
            point = np.asarray([value[0] for value in values], dtype=np.int64)
            mass = np.asarray([value[1] for value in values], dtype=np.float32)
            self.patch[patch] = (point, mass, cKDTree(xy[point]))

    @staticmethod
    def weighted_log_mass(distance: np.ndarray, mass: np.ndarray) -> float:
        weight = 1.0 / np.maximum(distance, DISTANCE_FLOOR) ** 2
        weight /= np.sum(weight)
        return float(np.exp(np.sum(weight * np.log(np.maximum(mass, 1e-5)))))

    def los(self, query_xy: np.ndarray) -> float:
        if self.los_tree is None:
            return 0.0
        count = min(K_ENERGY, len(self.los_point))
        distance, local = self.los_tree.query(query_xy, k=count)
        return self.weighted_log_mass(
            np.atleast_1d(distance), self.los_mass[np.atleast_1d(local)]
        )

    def reflection(self, patch: int, query_xy: np.ndarray) -> float:
        if patch not in self.patch:
            return 0.0
        point, mass, tree = self.patch[patch]
        count = min(K_ENERGY, len(point))
        distance, local = tree.query(query_xy, k=count)
        distance = np.atleast_1d(distance)
        local = np.atleast_1d(local)
        keep = distance <= MAX_PATCH_DISTANCE
        if not np.any(keep):
            return 0.0
        return self.weighted_log_mass(distance[keep], mass[local[keep]])


def cap_paths(rows: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    total = float(sum(row[2] for row in rows))
    if total <= MAX_VISIBLE_MASS or total <= 0.0:
        return rows
    scale = MAX_VISIBLE_MASS / total
    return [(delay, h, mass * scale) for delay, h, mass in rows]


def render_rows(rows: list[tuple[float, float, float]]) -> np.ndarray:
    output = np.zeros((192, 16), dtype=np.float32)
    for delay, h, mass in cap_paths(rows):
        output += render_path(delay, exact_h_atom(h), mass)
    return output


def true_geometry_rows(
    point_id: int,
    paths: list[list[PathRow]],
    energy: EnergyBank | None,
) -> list[tuple[float, float, float]]:
    rows = []
    for path in paths[point_id]:
        mass = path.mass
        if energy is not None:
            mass = (
                energy.los(energy.xy[point_id])
                if path.kind == "los"
                else energy.reflection(path.patch, energy.xy[point_id])
            )
        if mass > 0.0:
            rows.append((path.geometry_delay, path.geometry_h, mass))
    return rows


def selected_candidate_ids(
    point_id: int,
    selector_row: np.ndarray,
    candidate: dict[str, np.ndarray],
    threshold: float,
) -> list[int]:
    active = np.flatnonzero(
        (candidate["score"][point_id] > 0.0) & (selector_row >= threshold)
    )
    active = active[np.argsort(selector_row[active])[::-1]]
    selected = []
    used_patch: set[int] = set()
    for candidate_id in active:
        patch = int(candidate["patch_id"][point_id, candidate_id])
        if patch in used_patch:
            continue
        selected.append(int(candidate_id))
        used_patch.add(patch)
        if len(selected) == MAX_PATHS:
            break
    return selected


def predicted_rows(
    point_id: int,
    selector_row: np.ndarray,
    threshold: float,
    candidate: dict[str, np.ndarray],
    paths: list[list[PathRow]],
    energy: EnergyBank,
    blocked: np.ndarray,
    train_pos: np.ndarray,
    oracle_true_energy_only: bool,
) -> list[tuple[float, float, float]]:
    calibration = json.loads(v304.REPORT.read_text(encoding="utf-8"))["los_calibration"]
    slope = float(calibration["slope_bins_per_meter"])
    intercept = float(calibration["intercept_bins"])
    rows: list[tuple[float, float, float]] = []
    if blocked[point_id] == 0:
        distance = float(np.linalg.norm(train_pos[point_id] - v304.BS))
        azimuth = np.arctan2(
            train_pos[point_id, 1] - v304.BS[1],
            train_pos[point_id, 0] - v304.BS[0],
        )
        true_los = next(
            (path for path in paths[point_id] if path.kind == "los"), None
        )
        mass = (
            true_los.mass
            if oracle_true_energy_only and true_los is not None
            else energy.los(train_pos[point_id, :2])
        )
        if mass > 0.0:
            rows.append(
                (
                    intercept + slope * distance,
                    float(v304.expected_h_bin(np.asarray([azimuth]))[0]),
                    mass,
                )
            )

    truth_patch = {
        path.patch: path for path in paths[point_id] if path.kind == "reflection"
    }
    for candidate_id in selected_candidate_ids(
        point_id, selector_row, candidate, threshold
    ):
        patch = int(candidate["patch_id"][point_id, candidate_id])
        if oracle_true_energy_only:
            if patch not in truth_patch:
                continue
            mass = truth_patch[patch].mass
        else:
            mass = energy.reflection(patch, train_pos[point_id, :2])
        if mass <= 0.0:
            continue
        delay = intercept + slope * float(candidate["path_length"][point_id, candidate_id])
        h = float(
            v305.expected_h_bin(
                candidate["reflection_point"][point_id, candidate_id][None]
            )[0]
        )
        rows.append((delay, h, mass))
    return rows


def summarize_fold(
    predictions: dict[str, np.ndarray], truth: np.ndarray, regimes: np.ndarray
) -> dict[str, object]:
    base = metric_summary(predictions["baseline"], truth)
    output: dict[str, object] = {"base": base, "methods": {}}
    for method, prediction in predictions.items():
        metrics = metric_summary(prediction, truth)
        values = point_metrics(prediction, truth)
        base_values = point_metrics(predictions["baseline"], truth)
        output["methods"][method] = {
            **metrics,
            "delta": {key: metrics[key] - base[key] for key in metrics},
            "point_win_fraction": float(
                np.mean(values["weighted"] > base_values["weighted"])
            ),
            "by_regime": {
                regime: {
                    "count": int(np.sum(regimes == regime)),
                    "weighted_delta": float(
                        np.mean(
                            values["weighted"][regimes == regime]
                            - base_values["weighted"][regimes == regime]
                        )
                    )
                    if np.any(regimes == regime)
                    else 0.0,
                }
                for regime in REGIMES
            },
        }
    return output


def plot_summary(report: dict[str, object]) -> None:
    configure_chinese_font()
    methods = [method for method in METHODS if method != "baseline"]
    overall = [report["aggregate"][method]["weighted_delta"] for method in methods]
    medium = [
        report["aggregate"][method]["by_regime"]["medium"] for method in methods
    ]
    heavy = [
        report["aggregate"][method]["by_regime"]["heavy"] for method in methods
    ]
    labels = [METHOD_LABELS[method] for method in methods]
    x = np.arange(len(methods))
    figure, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    axis = axes[0]
    width = 0.25
    axis.bar(x - width, overall, width, color="#4978a8", label="全部点")
    axis.bar(x, medium, width, color="#ed8b2f", label="中遮挡")
    axis.bar(x + width, heavy, width, color="#c73945", label="重遮挡")
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.axhline(MIN_BREAKTHROUGH_DELTA, color="#2d7f4f", linestyle="--", linewidth=1.0)
    axis.set_xticks(x, labels, rotation=24, ha="right")
    axis.set_ylabel("相对普通全谱 KNN 的联合增益")
    axis.set_title("不同可见路径信息层级的收益")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)

    axis = axes[1]
    sh = [report["aggregate"][method]["sh_delta"] for method in methods]
    pdp = [report["aggregate"][method]["pdp_delta"] for method in methods]
    axis.bar(x - 0.18, sh, 0.36, color="#5f9c76", label="角度-时延谱")
    axis.bar(x + 0.18, pdp, 0.36, color="#8b62a8", label="PDP")
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_xticks(x, labels, rotation=24, ha="right")
    axis.set_ylabel("指标增益")
    axis.set_title("角度谱与时延谱贡献分解")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.suptitle("v402 可见物理路径 + 剩余传播分解", fontsize=16, fontweight="bold")
    figure.savefig(FIGURE, dpi=190, facecolor="white")
    plt.close(figure)


def main() -> None:
    required = (
        TARGET_CACHE,
        GEOMETRY,
        v297.COMPONENT_CACHE,
        v300.FINGERPRINT_CACHE,
        v304.SCATTERER_CACHE,
        v305.MATCH_CACHE,
        v306.TRAIN_CANDIDATES,
        v329.REPORT,
    )
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing v402 inputs: {missing}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    target_data = np.load(TARGET_CACHE)
    target = normalize(target_data["sh"].astype(np.float32))
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    component_data = np.load(v297.COMPONENT_CACHE)
    component = {key: component_data[key] for key in component_data.files}
    fingerprint_data = np.load(v300.FINGERPRINT_CACHE)
    fingerprint = {key: fingerprint_data[key] for key in fingerprint_data.files}
    candidate_data = np.load(v306.TRAIN_CANDIDATES)
    candidate = {key: candidate_data[key] for key in candidate_data.files}
    candidate_valid, candidate_label = v328.build_labels(candidate)
    geometry = np.load(GEOMETRY)
    blocked = geometry["train_blocked"].astype(np.int32)
    regimes_all = regime_of(blocked).astype(str)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    paths, path_summary = build_path_rows(
        train_pos, blocked, component, fingerprint, candidate
    )
    visible, residual, decomposition = decompose_visible(
        target, paths, fingerprint["horizontal"]
    )
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    v329_report = json.loads(v329.REPORT.read_text(encoding="utf-8"))

    oof_ids: list[np.ndarray] = []
    oof_predictions: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    fold_rows = []
    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, query_ids)
        neighbor_ids, weight = neighbors(query_ids, fit_ids, train_pos[:, :2])
        baseline = normalize(knn_average(target, neighbor_ids, weight))
        residual_anchor = knn_average(residual, neighbor_ids, weight)
        energy = EnergyBank(fit_ids, train_pos[:, :2], paths)
        selector, _truth_selector = v329.selector_matrix(
            candidate,
            candidate_valid,
            candidate_label,
            fit_ids,
            query_ids,
            train_pos[:, :2],
        )
        threshold = float(v329_report["fold_rows"][fold]["threshold"])
        predictions = {
            "baseline": baseline,
            "residual_only": normalize(residual_anchor),
            "oracle_visible_shape": normalize(residual_anchor + visible[query_ids]),
        }
        true_true = np.zeros_like(residual_anchor)
        true_pred = np.zeros_like(residual_anchor)
        selected_true = np.zeros_like(residual_anchor)
        deployable = np.zeros_like(residual_anchor)
        deployable_path_count = np.zeros(len(query_ids), dtype=np.int16)
        for row, point_id in enumerate(query_ids):
            true_true[row] = render_rows(
                true_geometry_rows(int(point_id), paths, energy=None)
            )
            true_pred[row] = render_rows(
                true_geometry_rows(int(point_id), paths, energy=energy)
            )
            selected_oracle_rows = predicted_rows(
                int(point_id),
                selector[row],
                threshold,
                candidate,
                paths,
                energy,
                blocked,
                train_pos,
                oracle_true_energy_only=True,
            )
            selected_true[row] = render_rows(selected_oracle_rows)
            predicted = predicted_rows(
                int(point_id),
                selector[row],
                threshold,
                candidate,
                paths,
                energy,
                blocked,
                train_pos,
                oracle_true_energy_only=False,
            )
            deployable[row] = render_rows(predicted)
            deployable_path_count[row] = len(predicted)
        predictions.update(
            {
                "true_identity_true_energy": normalize(residual_anchor + true_true),
                "true_identity_predicted_energy": normalize(residual_anchor + true_pred),
                "selected_true_oracle_energy": normalize(residual_anchor + selected_true),
                "deployable": normalize(residual_anchor + deployable),
            }
        )
        fold_report = summarize_fold(
            predictions, target[query_ids], regimes_all[query_ids]
        )
        fold_report.update(
            {
                "fold": fold,
                "fit_count": int(len(fit_ids)),
                "query_count": int(len(query_ids)),
                "selector_threshold": threshold,
                "deployable_active_fraction": float(
                    np.mean(deployable_path_count > 0)
                ),
                "deployable_path_count_mean": float(
                    np.mean(deployable_path_count)
                ),
                "leakage_assertions": {
                    "query_in_fit": int(len(np.intersect1d(query_ids, fit_ids))),
                    "query_in_neighbor_bank": int(
                        np.sum(np.isin(neighbor_ids, query_ids))
                    ),
                },
            }
        )
        fold_rows.append(fold_report)
        oof_ids.append(query_ids)
        for method in METHODS:
            oof_predictions[method].append(predictions[method])
        print(
            f"v402 fold={fold} deployable="
            f"{fold_report['methods']['deployable']['delta']['weighted']:+.6f} "
            f"oracle={fold_report['methods']['oracle_visible_shape']['delta']['weighted']:+.6f}",
            flush=True,
        )

    ids = np.concatenate(oof_ids)
    order = np.argsort(ids)
    ids = ids[order]
    merged = {
        method: np.concatenate(values, axis=0)[order]
        for method, values in oof_predictions.items()
    }
    aggregate: dict[str, object] = {}
    truth = target[ids]
    base_values = point_metrics(merged["baseline"], truth)
    for method in METHODS:
        values = point_metrics(merged[method], truth)
        aggregate[method] = {
            "sh": float(np.mean(values["sh"])),
            "pdp": float(np.mean(values["pdp"])),
            "weighted": float(np.mean(values["weighted"])),
            "sh_delta": float(np.mean(values["sh"] - base_values["sh"])),
            "pdp_delta": float(np.mean(values["pdp"] - base_values["pdp"])),
            "weighted_delta": float(
                np.mean(values["weighted"] - base_values["weighted"])
            ),
            "point_win_fraction": float(
                np.mean(values["weighted"] > base_values["weighted"])
            ),
            "fold_wins": int(
                sum(
                    row["methods"][method]["delta"]["weighted"] > 0.0
                    for row in fold_rows
                )
            ),
            "by_regime": {
                regime: float(
                    np.mean(
                        values["weighted"][regimes_all[ids] == regime]
                        - base_values["weighted"][regimes_all[ids] == regime]
                    )
                )
                for regime in REGIMES
            },
        }

    deployable_passed = (
        aggregate["deployable"]["weighted_delta"] >= MIN_BREAKTHROUGH_DELTA
        and aggregate["deployable"]["fold_wins"] >= MIN_FOLD_WINS
    )
    oracle_passed = (
        aggregate["oracle_visible_shape"]["weighted_delta"]
        >= MIN_BREAKTHROUGH_DELTA
        and aggregate["oracle_visible_shape"]["fold_wins"] >= MIN_FOLD_WINS
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "High-confidence LOS and one-bounce components can be removed from official "
            "training spectra, transported with geometry, and recombined with a simpler "
            "KNN residual to improve unseen medium/heavy predictions."
        ),
        "compliance": {
            "uses": [
                "official train H-derived continuous components and spectra",
                "official train coordinates",
                "official Round1 point-cloud blocker and wall candidates",
            ],
            "does_not_use": [
                "test channel truth",
                "v212/Sionna/RT/synthetic target",
                "online scores as labels",
            ],
            "oracle_methods": (
                "query path identities/energies are used only for attribution, never by deployable"
            ),
        },
        "parameters": {
            "folds": "five mutually exclusive isolation-stratified folds",
            "anchor_k": K_ANCHOR,
            "energy_k": K_ENERGY,
            "maximum_patch_distance_m": MAX_PATCH_DISTANCE,
            "maximum_predicted_paths": MAX_PATHS,
            "selector_thresholds": "frozen from v329 nested 70%-precision folds",
            "breakthrough_delta": MIN_BREAKTHROUGH_DELTA,
        },
        "path_summary": path_summary,
        "decomposition": decomposition,
        "fold_rows": fold_rows,
        "aggregate": aggregate,
        "decision": {
            "deployable_passed": bool(deployable_passed),
            "oracle_visible_shape_passed": bool(oracle_passed),
            "packaged": False,
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "oof": PATHS.relative(OOF_CACHE),
            "figure": PATHS.relative(FIGURE),
            "zip": None,
        },
    }
    np.savez_compressed(
        OOF_CACHE,
        ids=ids,
        regimes=regimes_all[ids],
        truth=truth,
        **merged,
    )
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_summary(report)
    print(
        json.dumps(
            {
                "deployable": aggregate["deployable"],
                "oracle_visible_shape": aggregate["oracle_visible_shape"],
                "decision": report["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

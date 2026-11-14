#!/usr/bin/env python3
"""v229 analysis: official path-delay gain field.

v226 showed that public point-cloud path lengths provide a real PDP/delay
signal, but its alpha was hand-set by regime.  v229 keeps the same strict data
boundary and learns the trust/gain of that path-delay prior from released train
H:

* targets come only from official train H delay marginals;
* features come from official train/test positions, point-cloud geometry,
  geodesic features, and public multibounce path candidates;
* generation does not use Sionna generated channels, v212/v217 teachers,
  hidden labels, or synthetic replay truth.

The v208 local scorer is used only after generating analysis outputs as the
hard gate diagnostic.  No submission ZIP is produced unless the local v212 gate
is exceeded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
import analyze_v226_official_path_kernel_prior as v226
import generate_v215_pointcloud_physics_neural_field as v215
from generate_v87_easy_field_projection import summarize
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 229
VERSION = "v229_official_path_gain_field"

OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v229_official_path_gain_field.json"
ACTION_OUT = OUT_DIR / "v229_official_path_gain_actions.npz"

GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"
BASE_V224 = PATHS.cache_dir(224) / "v224_mix0.58_it8_moderate_Round1_Test_Channel.npy"
BASE_V226 = PATHS.cache_dir(226) / "v226_early_shift_v224_path_heavy_Round1_Test_Channel.npy"

LOCAL_GATE = 0.645604
FLOOR = 1e-12
PCA_DIMS = 48


def normalize(x: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=axis, keepdims=True), TINY)
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = a.reshape(len(a), -1).astype(np.float64)
    y = b.reshape(len(b), -1).astype(np.float64)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def l1_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(
        np.abs(a.reshape(len(a), -1) - b.reshape(len(b), -1)),
        axis=1,
    ).astype(np.float32)


def path_kernel(prefix: str, mode: str) -> np.ndarray:
    specs = {
        "v226_early": (1.00, 0.58, 1.20, (0.38, 0.10, 0.00), 0.014, -0.25),
        "sharp": (1.10, 0.50, 1.00, (0.45, 0.04, -0.04), 0.006, -0.10),
        "reflect": (0.82, 0.78, 1.65, (0.18, 0.22, 0.10), 0.026, 0.20),
    }
    temp, sig_los, sig_ref, bias, tail, shift = specs[mode]
    d, _diag = v226.path_kernel_d(prefix, temp, sig_los, sig_ref, bias, tail, shift)
    return d.astype(np.float32)


def build_context() -> dict[str, object]:
    clean, _raw, _log_power, valid = v215.load_targets()
    targets = {key: clean[key].astype(np.float32) for key in ("d", "du", "dh")}
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked)
    test_regimes = regime_of(test_blocked)
    valid_ids = np.flatnonzero(valid.astype(bool))

    train_knn = v225.build_train_anchors(targets, valid_ids, train_pos, train_blocked)
    test_graph = v226.v224_anchor_delay()
    current_v224 = v225.current_delay_marginals(BASE_V224)
    current_v226 = v225.current_delay_marginals(BASE_V226)

    path_train = {name: path_kernel("train", name) for name in ("v226_early", "sharp", "reflect")}
    path_test = {name: path_kernel("test", name) for name in ("v226_early", "sharp", "reflect")}

    # The gain target says whether the public path-delay prior is closer to
    # official train H than the official-H KNN anchor for this row.
    train_gain = {
        name: row_cos(path_train[name][valid_ids], targets["d"][valid_ids])
        - row_cos(train_knn["d"], targets["d"][valid_ids])
        for name in path_train
    }
    best_train_mode = np.argmax(np.stack([train_gain[name] for name in path_train], axis=1), axis=1)

    current_train = {
        "d": normalize(
            0.60 * train_knn["d"] + 0.40 * path_train["v226_early"][valid_ids],
            axis=1,
        ),
        "du": train_knn["du"],
        "dh": train_knn["dh"],
    }
    train_x_raw = v225.make_features(
        "train",
        train_pos[valid_ids],
        train_blocked[valid_ids],
        train_regimes[valid_ids],
        train_knn,
        current=current_train,
        row_ids=valid_ids,
    )
    # Add explicit path-vs-anchor diagnostics that are not present in the
    # generic feature builder.
    extra_train = []
    for name in ("v226_early", "sharp", "reflect"):
        p = path_train[name][valid_ids]
        extra_train.extend(
            [
                row_cos(p, train_knn["d"]),
                l1_rows(p, train_knn["d"]),
                np.argmax(p, axis=1).astype(np.float32) / 191.0,
                np.max(p, axis=1).astype(np.float32),
            ]
        )
    train_x_raw = np.concatenate([train_x_raw, np.stack(extra_train, axis=1)], axis=1).astype(np.float32)

    test_current_for_features = {
        "d": normalize(
            0.45 * current_v226["d"] + 0.35 * test_graph["d"] + 0.20 * path_test["v226_early"],
            axis=1,
        ),
        "du": current_v226["du"],
        "dh": current_v226["dh"],
    }
    test_x_raw = v225.make_features(
        "test",
        test_pos,
        test_blocked,
        test_regimes,
        test_graph,
        current=test_current_for_features,
    )
    extra_test = []
    for name in ("v226_early", "sharp", "reflect"):
        p = path_test[name]
        extra_test.extend(
            [
                row_cos(p, test_graph["d"]),
                l1_rows(p, test_graph["d"]),
                np.argmax(p, axis=1).astype(np.float32) / 191.0,
                np.max(p, axis=1).astype(np.float32),
            ]
        )
    test_x_raw = np.concatenate([test_x_raw, np.stack(extra_test, axis=1)], axis=1).astype(np.float32)
    train_x, test_x, feature_stats = v225.standardize(train_x_raw, test_x_raw)

    basis, train_coeff = v215.fit_basis(
        normalize(0.55 * train_knn["d"] + 0.45 * path_train["v226_early"][valid_ids], axis=1),
        targets["d"][valid_ids],
        PCA_DIMS,
    )

    return {
        "targets": targets,
        "valid_ids": valid_ids,
        "train_x": train_x,
        "test_x": test_x,
        "feature_stats": feature_stats,
        "train_regimes": train_regimes[valid_ids],
        "test_regimes": test_regimes,
        "train_knn": train_knn,
        "test_graph": test_graph,
        "current_v224": current_v224,
        "current_v226": current_v226,
        "path_train": path_train,
        "path_test": path_test,
        "train_gain": train_gain,
        "best_train_mode": best_train_mode,
        "basis": basis,
        "train_coeff": train_coeff,
    }


def fit_gain_models(ctx: dict[str, object]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    train_x = ctx["train_x"]  # type: ignore[assignment]
    test_x = ctx["test_x"]  # type: ignore[assignment]
    train_gain: dict[str, np.ndarray] = ctx["train_gain"]  # type: ignore[assignment]
    preds: dict[str, np.ndarray] = {}
    diagnostics: dict[str, object] = {}
    for name, y in train_gain.items():
        model = ExtraTreesRegressor(
            n_estimators=900,
            min_samples_leaf=6,
            max_features=0.70,
            bootstrap=True,
            random_state=22900 + len(name),
            n_jobs=-1,
        )
        model.fit(train_x, y)
        preds[name] = model.predict(test_x).astype(np.float32)
        fitted = model.predict(train_x).astype(np.float32)
        diagnostics[f"{name}_train_gain"] = summarize(y)
        diagnostics[f"{name}_fit_corr"] = float(np.corrcoef(y, fitted)[0, 1])
        print(f"v229 fitted gain {name}", flush=True)
    return preds, diagnostics


def fit_d_residual_models(ctx: dict[str, object]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    train_x = ctx["train_x"]  # type: ignore[assignment]
    test_x = ctx["test_x"]  # type: ignore[assignment]
    train_coeff = ctx["train_coeff"]  # type: ignore[assignment]
    basis = ctx["basis"]  # type: ignore[assignment]
    path_test: dict[str, np.ndarray] = ctx["path_test"]  # type: ignore[assignment]
    test_graph: dict[str, np.ndarray] = ctx["test_graph"]  # type: ignore[assignment]
    anchors = {
        "graph_path": normalize(0.55 * test_graph["d"] + 0.45 * path_test["v226_early"], axis=1),
        "path_heavy": normalize(0.28 * test_graph["d"] + 0.72 * path_test["v226_early"], axis=1),
    }
    models = {
        "d_extratrees": ExtraTreesRegressor(
            n_estimators=850,
            min_samples_leaf=5,
            max_features=0.66,
            bootstrap=True,
            random_state=22951,
            n_jobs=-1,
        ),
        "d_histgb": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=190,
                learning_rate=0.040,
                l2_regularization=0.10,
                max_leaf_nodes=28,
                min_samples_leaf=28,
                random_state=22963,
            ),
            n_jobs=1,
        ),
    }
    pred: dict[str, np.ndarray] = {}
    diag: dict[str, object] = {}
    for model_name, model in models.items():
        model.fit(train_x, train_coeff)
        coeff = model.predict(test_x).astype(np.float32)
        for anchor_name, anchor in anchors.items():
            key = f"{model_name}_{anchor_name}"
            pred[key] = v225.reconstruct(anchor, coeff, basis)
        train_fit = model.predict(train_x).astype(np.float32)
        diag[f"{model_name}_coeff_fit_mae"] = float(np.mean(np.abs(train_fit - train_coeff)))
        print(f"v229 fitted residual {model_name}", flush=True)
    return pred, diag


def alpha_from_gain(
    gain_pred: np.ndarray,
    regimes: np.ndarray,
    max_alpha: dict[str, float],
    threshold: float,
    scale: float,
) -> np.ndarray:
    cap = np.asarray([max_alpha[str(r)] for r in regimes], dtype=np.float32)
    score = 1.0 / (1.0 + np.exp(-(gain_pred - threshold) / max(scale, 1e-4)))
    score = np.clip(score, 0.0, 1.0).astype(np.float32)
    return (cap * score).astype(np.float32)


def build_targets_from_gain(
    ctx: dict[str, object],
    gain_pred: dict[str, np.ndarray],
    variant: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    regimes: np.ndarray = ctx["test_regimes"]  # type: ignore[assignment]
    current: dict[str, np.ndarray] = ctx["current_v226"]  # type: ignore[assignment]
    graph: dict[str, np.ndarray] = ctx["test_graph"]  # type: ignore[assignment]
    path_test: dict[str, np.ndarray] = ctx["path_test"]  # type: ignore[assignment]
    if variant == "selective_early":
        gain = gain_pred["v226_early"]
        alpha = alpha_from_gain(
            gain,
            regimes,
            {"clear": 0.0, "light": 0.04, "medium": 0.34, "heavy": 0.62},
            threshold=-0.003,
            scale=0.018,
        )
        target_raw = normalize(0.40 * graph["d"] + 0.60 * path_test["v226_early"], axis=1)
    elif variant == "mode_switch":
        stacked = np.stack([gain_pred[name] for name in ("v226_early", "sharp", "reflect")], axis=1)
        mode = np.argmax(stacked, axis=1)
        best_gain = stacked[np.arange(len(stacked)), mode]
        alpha = alpha_from_gain(
            best_gain,
            regimes,
            {"clear": 0.0, "light": 0.05, "medium": 0.40, "heavy": 0.72},
            threshold=-0.006,
            scale=0.020,
        )
        candidates = np.stack(
            [path_test["v226_early"], path_test["sharp"], path_test["reflect"]],
            axis=1,
        )
        path = candidates[np.arange(len(candidates)), mode]
        target_raw = normalize(0.30 * graph["d"] + 0.70 * path, axis=1)
    else:
        raise ValueError(variant)
    shape = (len(regimes), 1)
    target = {
        "d": normalize((1.0 - alpha.reshape(shape)) * current["d"] + alpha.reshape(shape) * target_raw, axis=1),
        "du": current["du"],
        "dh": current["dh"],
    }
    return target, {"alpha": alpha.astype(np.float32)}


def build_targets_from_residual(
    ctx: dict[str, object],
    residual_pred: dict[str, np.ndarray],
    key: str,
    alpha_by_regime: dict[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    regimes: np.ndarray = ctx["test_regimes"]  # type: ignore[assignment]
    current: dict[str, np.ndarray] = ctx["current_v226"]  # type: ignore[assignment]
    alpha = np.asarray([alpha_by_regime[str(r)] for r in regimes], dtype=np.float32)
    shape = (len(regimes), 1)
    target = {
        "d": normalize(
            (1.0 - alpha.reshape(shape)) * current["d"]
            + alpha.reshape(shape) * residual_pred[key],
            axis=1,
        ),
        "du": current["du"],
        "dh": current["dh"],
    }
    return target, {"alpha": alpha.astype(np.float32)}


def run_variant(
    name: str,
    source: Path,
    target: dict[str, np.ndarray],
    regimes: np.ndarray,
    gain_clip_scale: float,
    diagnostics: dict[str, np.ndarray],
) -> dict[str, object]:
    out = OUT_DIR / f"v229_{name}_Round1_Test_Channel.npy"
    diag = v225.apply_delay_targets(source, out, target, regimes, gain_clip_scale)
    score = v208.score_prediction(out)
    return {
        "name": name,
        "source": PATHS.relative(source),
        "output": PATHS.relative(out),
        "gain_clip_scale": gain_clip_scale,
        "score": score,
        "diagnostics": {key: summarize(value) for key, value in diag.items()},
        "target_alpha": summarize(diagnostics["alpha"]),
    }


def main() -> None:
    for path in [GEOM_CACHE, BASE_V224, BASE_V226, v226.PATH_CACHE, v225.V224_DELAY_TARGET]:
        if not path.is_file():
            raise FileNotFoundError(path)
    v225.PROJ_BATCH = 10
    ctx = build_context()
    gain_pred, gain_diag = fit_gain_models(ctx)
    residual_pred, residual_diag = fit_d_residual_models(ctx)

    regimes: np.ndarray = ctx["test_regimes"]  # type: ignore[assignment]
    variants: list[dict[str, object]] = []
    for name, source, clip in [
        ("selective_early_from_v226", BASE_V226, 1.04),
        ("selective_early_from_v224", BASE_V224, 1.10),
        ("mode_switch_from_v226", BASE_V226, 1.08),
    ]:
        mode = "selective_early" if name.startswith("selective") else "mode_switch"
        target, diag = build_targets_from_gain(ctx, gain_pred, mode)
        result = run_variant(name, source, target, regimes, clip, diag)
        variants.append(result)
        print(json.dumps({"name": name, "score": result["score"]}, ensure_ascii=False), flush=True)

    for key, alpha, clip in [
        ("d_extratrees_graph_path", {"clear": 0.0, "light": 0.035, "medium": 0.28, "heavy": 0.46}, 1.02),
        ("d_extratrees_path_heavy", {"clear": 0.0, "light": 0.045, "medium": 0.36, "heavy": 0.58}, 1.08),
        ("d_histgb_graph_path", {"clear": 0.0, "light": 0.030, "medium": 0.24, "heavy": 0.40}, 1.00),
    ]:
        target, diag = build_targets_from_residual(ctx, residual_pred, key, alpha)
        name = f"{key}_from_v226"
        result = run_variant(name, BASE_V226, target, regimes, clip, diag)
        variants.append(result)
        print(json.dumps({"name": name, "score": result["score"]}, ensure_ascii=False), flush=True)

    best = max(variants, key=lambda item: item["score"]["score_traceback_leaked"])
    np.savez_compressed(
        ACTION_OUT,
        test_regimes=regimes.astype("U8"),
        **{f"gain_{key}": value for key, value in gain_pred.items()},
        **{f"residual_{key}": value for key, value in residual_pred.items()},
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The official path-delay prior from v226 is useful, but its trust "
            "should be learned from official train-H gain rather than fixed by regime."
        ),
        "compliance_boundary": {
            "generation_uses": [
                "official train H delay marginals",
                "official train/test coordinates",
                "official point-cloud path candidates and geometry caches",
                "v224/v226 official-data carriers",
            ],
            "generation_does_not_use": [
                "Sionna generated channel labels",
                "v212/v217 teacher outputs",
                "hidden online labels",
                "synthetic replay truth",
            ],
            "note": "Analysis-only official-data path-gain probe; no ZIP unless local v212 gate is exceeded.",
        },
        "local_gate_required": LOCAL_GATE,
        "train_diagnostics": {
            "valid_train_count": int(len(ctx["valid_ids"])),  # type: ignore[arg-type]
            "gain": gain_diag,
            "residual": residual_diag,
            "train_gain_summary": {
                key: summarize(value) for key, value in (ctx["train_gain"]).items()  # type: ignore[union-attr]
            },
        },
        "variants": variants,
        "best_variant": best["name"],
        "best_score_traceback_leaked": best["score"]["score_traceback_leaked"],
        "gate_passed": bool(best["score"]["score_traceback_leaked"] >= LOCAL_GATE),
        "outputs": {
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

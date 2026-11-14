#!/usr/bin/env python3
"""v231 analysis: official learned leaked S-H support field.

v226 improved PDP but still leaves a large leaked S-H gap to the v212 local
gate.  v227/v228 showed that hand-written path-angle formulas are weak.  v231
keeps the strict official-data boundary and learns the leaked S-H marginal from
official train H using geometry/path features plus graph/current anchors.

No Sionna generated labels, v212/v217 teachers, hidden labels, or synthetic
truth are used for generation.  The v208 scorer is only the post-generation
hard diagnostic gate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
import generate_v187_leaked_pas_lifecycle_projection as leak
import generate_v209_adaptive_metric_projection_controller as v209
import generate_v218_official_multispectrum_physics_ai as v218
from generate_v171_full_complex_ad_tail_denoising import top_mask
from generate_v87_easy_field_projection import N_ANT, summarize
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 231
VERSION = "v231_official_leaked_sh_field"

OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v231_official_leaked_sh_field.json"
ACTION_OUT = OUT_DIR / "v231_official_leaked_sh_actions.npz"

GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"
BASE_V226 = PATHS.cache_dir(226) / "v226_early_shift_v224_path_heavy_Round1_Test_Channel.npy"
V224_LEAKED_TARGET = PATHS.cache_dir(224) / "v224_leaked_mix0.58_it8.npy"

FLOOR = 1e-12
LOCAL_GATE = 0.645604
PCA_DIMS = 64
PROJ_BATCH = 10


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
    return np.sum(np.abs(a.reshape(len(a), -1) - b.reshape(len(b), -1)), axis=1).astype(np.float32)


def denoise_sh(x: np.ndarray, strength: float) -> np.ndarray:
    arr = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    flat = arr.reshape(len(arr), -1)
    floor = np.quantile(flat, 0.18, axis=1).astype(np.float32)
    clean = np.maximum(flat - strength * floor[:, None], 0.0)
    empty = clean.sum(axis=1) <= 1e-20
    clean[empty] = flat[empty]
    return normalize(clean.reshape(arr.shape), axis=(1, 2))


def sh_stats(sh: np.ndarray) -> np.ndarray:
    arr = normalize(sh, axis=(1, 2))
    n, s_count, h_count = arr.shape
    s_idx = np.arange(s_count, dtype=np.float32) / max(s_count - 1, 1)
    h_idx = np.arange(h_count, dtype=np.float32) / max(h_count - 1, 1)
    s_m = arr.sum(axis=2)
    h_m = arr.sum(axis=1)
    entropy = -np.sum(arr.reshape(n, -1) * np.log(np.maximum(arr.reshape(n, -1), FLOOR)), axis=1) / math.log(
        s_count * h_count
    )
    return np.concatenate(
        [
            np.sum(s_m * s_idx[None, :], axis=1)[:, None],
            np.sum(h_m * h_idx[None, :], axis=1)[:, None],
            np.max(arr.reshape(n, -1), axis=1)[:, None],
            entropy[:, None].astype(np.float32),
            s_m[:, :: max(s_count // 16, 1)][:, :16],
            h_m,
        ],
        axis=1,
    ).astype(np.float32)


def weighted_average(target: np.ndarray, idx: np.ndarray, weights: np.ndarray) -> np.ndarray:
    selected = target[idx]
    shape = (len(idx), idx.shape[1]) + (1,) * (target.ndim - 1)
    return normalize(np.sum(selected * weights.reshape(shape), axis=1), axis=tuple(range(1, target.ndim)))


def current_sh(path: Path) -> np.ndarray:
    source = np.load(path, mmap_mode="r")
    parts = []
    for start in range(0, len(source), PROJ_BATCH):
        end = min(start + PROJ_BATCH, len(source))
        h = np.asarray(source[start:end], dtype=np.complex64)
        power = leak.leaked_cell_power(leak.leaked_forward(h))
        support = power / np.maximum(power.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        parts.append(support.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32))
    return normalize(np.concatenate(parts, axis=0), axis=(1, 2))


def graph_sh() -> np.ndarray:
    support = np.load(V224_LEAKED_TARGET, mmap_mode="r")
    return normalize(np.asarray(support).sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32), axis=(1, 2))


def fit_basis(anchor: np.ndarray, target: np.ndarray) -> tuple[v218.v215.Basis, np.ndarray]:
    return v218.v215.fit_basis(anchor, target, PCA_DIMS)


def reconstruct(anchor: np.ndarray, coeff_norm: np.ndarray, basis: v218.v215.Basis) -> np.ndarray:
    delta = (coeff_norm.astype(np.float32) * basis.coeff_scale[None, :]) @ basis.components
    logits = np.log(np.maximum(anchor.reshape(len(anchor), -1), FLOOR)) + basis.mean[None, :] + delta
    logits -= logits.max(axis=1, keepdims=True)
    flat = np.exp(logits).astype(np.float32)
    flat /= np.maximum(flat.sum(axis=1, keepdims=True), TINY)
    return flat.reshape((len(anchor),) + basis.shape).astype(np.float32)


def build_features(
    prefix: str,
    pos: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
    anchor: np.ndarray,
    current: np.ndarray,
    row_ids: np.ndarray | None = None,
) -> np.ndarray:
    scene = v225.scene_features(prefix)
    paths = v225.path_features(prefix)
    if row_ids is not None:
        scene = scene[row_ids]
        paths = paths[row_ids]
    x = np.concatenate(
        [
            v225.position_features(pos, blocked, regimes),
            scene,
            paths,
            sh_stats(anchor),
            sh_stats(current),
            l1_rows(anchor, current)[:, None],
            row_cos(anchor, current)[:, None],
        ],
        axis=1,
    )
    return np.nan_to_num(x, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    mean = train_x.mean(axis=0).astype(np.float32)
    std = np.maximum(train_x.std(axis=0), 1e-5).astype(np.float32)
    return (
        np.clip((train_x - mean) / std, -8.0, 8.0).astype(np.float32),
        np.clip((test_x - mean) / std, -8.0, 8.0).astype(np.float32),
        {"mean": mean, "std": std},
    )


def build_context() -> dict[str, object]:
    leaked = v209.build_train_leaked_marginals()
    target_all = denoise_sh(leaked["sh"], 0.72)
    valid = leaked["valid"].astype(bool)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(GEOM_CACHE)
    train_blocked = geom["train_blocked"].astype(np.float32)
    test_blocked = geom["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked)
    test_regimes = regime_of(test_blocked)
    valid_ids = np.flatnonzero(valid)

    idx_train, w_train = v218.neighbor_index_weights(
        train_pos[valid_ids],
        train_blocked[valid_ids],
        train_pos[valid_ids],
        train_blocked[valid_ids],
        query_ids=valid_ids,
        bank_ids=valid_ids,
        k=84,
    )
    train_anchor = weighted_average(target_all[valid_ids], idx_train, w_train)
    train_current = train_anchor

    idx_test, w_test = v218.neighbor_index_weights(
        test_pos,
        test_blocked,
        train_pos[valid_ids],
        train_blocked[valid_ids],
        k=84,
    )
    test_knn = weighted_average(target_all[valid_ids], idx_test, w_test)
    test_graph = graph_sh()
    test_current = current_sh(BASE_V226)
    test_anchor = normalize(0.58 * test_graph + 0.42 * test_knn, axis=(1, 2))

    basis, coeff = fit_basis(train_anchor, target_all[valid_ids])
    train_x_raw = build_features(
        "train",
        train_pos[valid_ids],
        train_blocked[valid_ids],
        train_regimes[valid_ids],
        train_anchor,
        train_current,
        row_ids=valid_ids,
    )
    test_x_raw = build_features(
        "test",
        test_pos,
        test_blocked,
        test_regimes,
        test_anchor,
        test_current,
    )
    train_x, test_x, feature_stats = standardize(train_x_raw, test_x_raw)
    return {
        "target": target_all[valid_ids],
        "valid_ids": valid_ids,
        "train_x": train_x,
        "test_x": test_x,
        "feature_stats": feature_stats,
        "basis": basis,
        "coeff": coeff,
        "train_anchor": train_anchor,
        "test_anchor": test_anchor,
        "test_current": test_current,
        "test_graph": test_graph,
        "test_knn": test_knn,
        "test_regimes": test_regimes,
    }


def fit_predict(ctx: dict[str, object]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    train_x = ctx["train_x"]  # type: ignore[assignment]
    test_x = ctx["test_x"]  # type: ignore[assignment]
    coeff = ctx["coeff"]  # type: ignore[assignment]
    basis = ctx["basis"]  # type: ignore[assignment]
    anchor = ctx["test_anchor"]  # type: ignore[assignment]
    specs = {
        "extratrees_leaf4": ExtraTreesRegressor(
            n_estimators=900,
            min_samples_leaf=4,
            max_features=0.68,
            bootstrap=True,
            random_state=23104,
            n_jobs=-1,
        ),
        "histgb": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.04,
                l2_regularization=0.10,
                max_leaf_nodes=24,
                min_samples_leaf=26,
                random_state=23117,
            ),
            n_jobs=1,
        ),
    }
    preds: dict[str, np.ndarray] = {}
    diag: dict[str, object] = {}
    for name, model in specs.items():
        model.fit(train_x, coeff)
        pred_coeff = model.predict(test_x).astype(np.float32)
        preds[name] = reconstruct(anchor, pred_coeff, basis)
        fit_coeff = model.predict(train_x).astype(np.float32)
        diag[f"{name}_coeff_fit_mae"] = float(np.mean(np.abs(fit_coeff - coeff)))
        print(f"v231 fitted {name}", flush=True)
    return preds, diag


def apply_sh_target(
    source_path: Path,
    output_path: Path,
    target_sh: np.ndarray,
    regimes: np.ndarray,
    gain_scale: float,
) -> dict[str, np.ndarray]:
    source = np.load(source_path, mmap_mode="r")
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.complex64, shape=source.shape)
    diag: dict[str, list[np.ndarray]] = {
        "sh_l1_actual": [],
        "full_l1_actual": [],
        "gain_min": [],
        "gain_p90": [],
        "power_factor": [],
    }
    for start in range(0, len(source), PROJ_BATCH):
        end = min(start + PROJ_BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        spec0 = leak.leaked_forward(h0)
        power0 = leak.leaked_cell_power(spec0)
        total0 = np.maximum(power0.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        support0 = power0 / total0[:, None, None, None, None, None]
        base_sh = support0.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
        work = support0.copy()
        for _ in range(6):
            ratio = np.maximum(target_sh[start:end], FLOOR) / np.maximum(
                work.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32),
                FLOOR,
            )
            v209.apply_leaked_ratio(work, ratio, "sh", 0.82)
            work = normalize(work.reshape(end - start, -1), axis=1).reshape(support0.shape)
        gain = np.sqrt(work / np.maximum(support0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, 0.20, 5.40 * gain_scale)
        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.50, 2.25 * gain_scale)
        gain[strong1024] = np.clip(gain[strong1024], 0.66, 1.62 * gain_scale)
        gain[strong128] = np.clip(gain[strong128], 0.82, 1.26 * gain_scale)
        for local, regime in enumerate(regimes[start:end].astype(str)):
            if regime == "clear":
                gain[local] = 1.0
            elif regime == "light":
                gain[local] = 1.0 + 0.35 * (gain[local] - 1.0)
        h1 = leak.leaked_inverse(spec0 * gain)
        p0 = np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY)
        p1 = np.maximum(np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY)
        h1 *= np.sqrt(p0 / p1)[:, None, None, None].astype(np.float32)
        out[start:end] = h1.astype(np.complex64)
        out.flush()
        power1 = leak.leaked_cell_power(leak.leaked_forward(h1.astype(np.complex64)))
        support1 = power1 / np.maximum(power1.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        final_sh = support1.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32)
        diag["sh_l1_actual"].append(l1_rows(final_sh, base_sh))
        diag["full_l1_actual"].append(np.sum(np.abs(support1 - support0), axis=(1, 2, 3, 4, 5)).astype(np.float32))
        gflat = gain.reshape(end - start, -1)
        diag["gain_min"].append(gflat.min(axis=1).astype(np.float32))
        diag["gain_p90"].append(np.quantile(gflat, 0.90, axis=1).astype(np.float32))
        diag["power_factor"].append((np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64) / p0).astype(np.float32))
        if end == len(source) or end % 100 == 0:
            print(f"v231 sh projection {output_path.name} {end}/{len(source)}", flush=True)
    return {key: np.concatenate(value).astype(np.float32) for key, value in diag.items()}


def build_target(current: np.ndarray, pred: np.ndarray, regimes: np.ndarray, alpha_by_regime: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.asarray([alpha_by_regime[str(r)] for r in regimes], dtype=np.float32)
    shape = (len(alpha), 1, 1)
    target = normalize((1.0 - alpha.reshape(shape)) * current + alpha.reshape(shape) * pred, axis=(1, 2))
    return target, alpha


def run_variant(
    name: str,
    pred: np.ndarray,
    current: np.ndarray,
    regimes: np.ndarray,
    alpha_by_regime: dict[str, float],
    gain_scale: float,
) -> dict[str, object]:
    target, alpha = build_target(current, pred, regimes, alpha_by_regime)
    out = OUT_DIR / f"v231_{name}_Round1_Test_Channel.npy"
    diag = apply_sh_target(BASE_V226, out, target, regimes, gain_scale)
    score = v208.score_prediction(out)
    return {
        "name": name,
        "output": PATHS.relative(out),
        "alpha": alpha_by_regime,
        "gain_scale": gain_scale,
        "score": score,
        "diagnostics": {key: summarize(value) for key, value in diag.items()},
        "target_alpha": summarize(alpha),
        "target_l1_to_current": summarize(l1_rows(target, current)),
    }


def main() -> None:
    for path in [BASE_V226, V224_LEAKED_TARGET, GEOM_CACHE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    ctx = build_context()
    preds, model_diag = fit_predict(ctx)
    regimes: np.ndarray = ctx["test_regimes"]  # type: ignore[assignment]
    current: np.ndarray = ctx["test_current"]  # type: ignore[assignment]
    graph: np.ndarray = ctx["test_graph"]  # type: ignore[assignment]
    variants = []
    for name, pred, alpha, gain_scale in [
        ("graph_moderate", graph, {"clear": 0.0, "light": 0.035, "medium": 0.20, "heavy": 0.32}, 1.00),
        ("tree_moderate", preds["extratrees_leaf4"], {"clear": 0.0, "light": 0.045, "medium": 0.28, "heavy": 0.44}, 1.04),
        ("tree_strong", preds["extratrees_leaf4"], {"clear": 0.0, "light": 0.060, "medium": 0.40, "heavy": 0.62}, 1.08),
        ("histgb_moderate", preds["histgb"], {"clear": 0.0, "light": 0.040, "medium": 0.26, "heavy": 0.40}, 1.02),
    ]:
        result = run_variant(name, pred, current, regimes, alpha, gain_scale)
        variants.append(result)
        print(json.dumps({"name": name, "score": result["score"]}, ensure_ascii=False), flush=True)
    best = max(variants, key=lambda item: item["score"]["score_traceback_leaked"])
    np.savez_compressed(
        ACTION_OUT,
        test_regimes=regimes.astype("U8"),
        current=current,
        graph=graph,
        **{f"pred_{key}": value for key, value in preds.items()},
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": "The v226-to-v212 local gap is largely leaked S-H / AD support, so learn an official train-H supervised leaked S-H field instead of hand-written path-angle projection.",
        "compliance_boundary": {
            "generation_uses": [
                "official train H leaked S-H marginals",
                "official train/test coordinates",
                "official point-cloud and path feature caches",
                "v226 official-data carrier",
                "v224 official graph leaked support anchor",
            ],
            "generation_does_not_use": [
                "Sionna generated channel labels",
                "v212/v217 teacher outputs",
                "hidden online labels",
                "synthetic replay truth",
            ],
        },
        "local_gate_required": LOCAL_GATE,
        "diagnostics": {
            "valid_train_count": int(len(ctx["valid_ids"])),  # type: ignore[arg-type]
            "model": model_diag,
            "train_anchor_cos": summarize(row_cos(ctx["train_anchor"], ctx["target"])),  # type: ignore[arg-type]
            "test_graph_l1_to_current": summarize(l1_rows(graph, current)),
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

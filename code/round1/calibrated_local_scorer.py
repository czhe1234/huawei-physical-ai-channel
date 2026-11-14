#!/usr/bin/env python3
"""Calibrated local scorer for Round1 regime candidates.

This is a decision-support tool, not a replacement for the online judge.  The
raw local C values have repeatedly disagreed with online ordering, so this
script keeps the raw metrics visible but also fits a tiny calibrated surrogate
from known online regime submissions.

Design constraints:
- keep the model simple enough to audit with 7-10 online points;
- separate standard regime evaluations from incompatible geometry variants;
- report validation error and uncertainty instead of pretending the prediction
  is exact.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiment_record_index import load_online_rows as load_record_online_rows


KEY_SETS = ("feature_matched", "stress_feature_boundary", "clear_los", "heavy_blocked")
METRICS = ("score", "pas", "pdp", "nmse", "inv_nmse", "rank_score")
TEST_BRANCH_COUNTS = {"clear": 103.0, "light": 35.0, "medium": 83.0, "heavy": 279.0}
TEST_TOTAL = sum(TEST_BRANCH_COUNTS.values())


def parse_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def param_key(clear, light, medium, heavy):
    vals = (clear, light, medium, heavy)
    if any(v is None for v in vals):
        return None
    return tuple(round(float(v), 6) for v in vals)


def parse_regime_params(text):
    if not text:
        return None
    normalized = text.replace(" ", "")
    full = re.search(
        r"clear=([0-9.]+),light=([0-9.]+),medium=([0-9.]+),heavy=([0-9.]+)",
        normalized,
    )
    if full:
        return param_key(*(float(full.group(i)) for i in range(1, 5)))

    else_match = re.search(r"clear=([0-9.]+),else=([0-9.]+)", normalized)
    if else_match:
        clear = float(else_match.group(1))
        other = float(else_match.group(2))
        return param_key(clear, other, other, other)

    global_rho = re.search(r"rho=([0-9.]+)", normalized)
    if global_rho:
        rho = float(global_rho.group(1))
        return param_key(rho, rho, rho, rho)

    return None


def load_online_rows(record_path):
    rows = []
    for source in load_record_online_rows(record_path):
        version = str(source["version"])
        method = str(source["method"])
        params = str(source["params"])
        online = parse_float(source["online"])
        key = parse_regime_params(params)
        rows.append(
            {
                "version": version,
                "method": method,
                "params": params,
                "online": online,
                "param_key": key,
                "conclusion": str(source.get("conclusion") or ""),
            }
        )
    return rows


def online_by_param(online_rows):
    by_key = {}
    versions = defaultdict(list)
    for row in online_rows:
        if row["method"] != "regime_s1style_geom_blend":
            continue
        if row["param_key"] is None or row["online"] is None:
            continue
        key = row["param_key"]
        versions[key].append(row["version"])
        by_key[key] = max(row["online"], by_key.get(key, -math.inf))
    return by_key, versions


def source_is_standard(path):
    name = os.path.basename(path)
    # geom_rtheta uses a different geometry branch but shares rho tuples with
    # standard candidates, so assigning standard online scores to it is invalid.
    return "geom_rtheta" not in name


def load_eval_json(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    combos = {}
    for c in payload.get("combos", []):
        key = param_key(c.get("clear"), c.get("light"), c.get("medium"), c.get("heavy"))
        combos[c["name"]] = {
            "name": c["name"],
            "clear": c.get("clear"),
            "light": c.get("light"),
            "medium": c.get("medium"),
            "heavy": c.get("heavy"),
            "param_key": key,
        }

    by_param = defaultdict(lambda: {"names": set(), "sources": set(), "sets": defaultdict(list)})
    for set_obj in payload.get("sets", []):
        set_name = set_obj.get("name")
        results = set_obj.get("results", [])
        denom = max(1, len(results) - 1)
        for rank, row in enumerate(results, start=1):
            name = row["name"]
            combo = combos.get(name)
            if not combo or combo["param_key"] is None:
                continue
            key = combo["param_key"]
            rec = by_param[key]
            rec["params"] = combo
            rec["names"].add(name)
            rec["sources"].add(os.path.basename(path))
            rec["sets"][set_name].append(
                {
                    "score": row.get("score"),
                    "pas": row.get("pas"),
                    "pdp": row.get("pdp"),
                    "nmse": row.get("global_nmse", row.get("global_nmse_direct")),
                    "inv_nmse": row.get("inv_nmse"),
                    "rank_score": 1.0 - ((rank - 1) / denom),
                    "count": row.get("count"),
                }
            )
    return by_param


def median_or_none(vals):
    clean = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    if not clean:
        return None
    return float(np.median(clean))


def aggregate_records(eval_glob, include_nonstandard=False):
    merged = defaultdict(lambda: {"names": set(), "sources": set(), "sets": defaultdict(list)})
    for path in sorted(glob.glob(eval_glob)):
        if not include_nonstandard and not source_is_standard(path):
            continue
        for key, rec in load_eval_json(path).items():
            out = merged[key]
            out["params"] = rec["params"]
            out["names"].update(rec["names"])
            out["sources"].update(rec["sources"])
            for set_name, rows in rec["sets"].items():
                out["sets"][set_name].extend(rows)

    records = []
    for key, rec in merged.items():
        params = rec["params"]
        out = {
            "param_key": key,
            "params": {
                "clear": params["clear"],
                "light": params["light"],
                "medium": params["medium"],
                "heavy": params["heavy"],
            },
            "names": sorted(rec["names"]),
            "sources": sorted(rec["sources"]),
            "features": {},
        }
        for set_name in KEY_SETS:
            rows = rec["sets"].get(set_name, [])
            for metric in METRICS:
                out["features"][f"{set_name}_{metric}"] = median_or_none(
                    row.get(metric) for row in rows
                )

        score_vals = [
            out["features"].get(f"{set_name}_score")
            for set_name in KEY_SETS
            if out["features"].get(f"{set_name}_score") is not None
        ]
        rank_vals = [
            out["features"].get(f"{set_name}_rank_score")
            for set_name in KEY_SETS
            if out["features"].get(f"{set_name}_rank_score") is not None
        ]
        out["features"]["local_mean_score"] = (
            float(np.mean(score_vals)) if score_vals else None
        )
        out["features"]["local_mean_rank_score"] = (
            float(np.mean(rank_vals)) if rank_vals else None
        )
        out["features"]["coverage"] = float(len(score_vals))

        c, l, m, h = key
        out["features"].update(
            {
                "rho_clear": c,
                "rho_light": l,
                "rho_medium": m,
                "rho_heavy": h,
                "rho_mean_test": (
                    TEST_BRANCH_COUNTS["clear"] * c
                    + TEST_BRANCH_COUNTS["light"] * l
                    + TEST_BRANCH_COUNTS["medium"] * m
                    + TEST_BRANCH_COUNTS["heavy"] * h
                )
                / TEST_TOTAL,
                "nonclear_mean": (
                    TEST_BRANCH_COUNTS["light"] * l
                    + TEST_BRANCH_COUNTS["medium"] * m
                    + TEST_BRANCH_COUNTS["heavy"] * h
                )
                / (TEST_TOTAL - TEST_BRANCH_COUNTS["clear"]),
                "medium_heavy_mean": 0.5 * (m + h),
                "medium_heavy_min": min(m, h),
                "light_sq": l * l,
                "light_excess_030": max(0.0, l - 0.30),
                "clear_excess_010": max(0.0, c - 0.10),
            }
        )
        records.append(out)
    return records


BASE_FEATURES = [
    "rho_clear",
    "rho_light",
    "rho_medium",
    "rho_heavy",
    "rho_mean_test",
    "nonclear_mean",
    "medium_heavy_mean",
    "medium_heavy_min",
    "light_sq",
    "light_excess_030",
    "clear_excess_010",
    "feature_matched_score",
    "stress_feature_boundary_score",
    "clear_los_score",
    "heavy_blocked_score",
    "feature_matched_nmse",
    "stress_feature_boundary_nmse",
    "clear_los_nmse",
    "heavy_blocked_nmse",
    "local_mean_score",
    "local_mean_rank_score",
    "coverage",
]


PARAM_FEATURES = [
    "rho_clear",
    "rho_light",
    "rho_medium",
    "rho_heavy",
    "rho_mean_test",
    "nonclear_mean",
    "medium_heavy_mean",
    "medium_heavy_min",
    "light_sq",
    "light_excess_030",
    "clear_excess_010",
]


def build_matrix(records, feature_names, train_records=None):
    if train_records is None:
        train_records = records
    medians = {}
    for name in feature_names:
        vals = [r["features"].get(name) for r in train_records]
        clean = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
        medians[name] = float(np.median(clean)) if clean else 0.0

    X = np.zeros((len(records), len(feature_names)), dtype=np.float64)
    missing = np.zeros((len(records),), dtype=np.float64)
    for i, rec in enumerate(records):
        for j, name in enumerate(feature_names):
            val = rec["features"].get(name)
            if val is None or not np.isfinite(float(val)):
                X[i, j] = medians[name]
                missing[i] += 1.0
            else:
                X[i, j] = float(val)
    return X, missing, medians


def standardize_train(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean, std, (X - mean) / std


def ridge_fit(X, y, alpha):
    # y is centered before this function.
    eye = np.eye(X.shape[1], dtype=np.float64)
    return np.linalg.solve(X.T @ X + alpha * eye, X.T @ y)


def ridge_predict(train_X, train_y, pred_X, alpha):
    mean_x, std_x, Xs = standardize_train(train_X)
    y_mean = float(train_y.mean())
    ys = train_y - y_mean
    w = ridge_fit(Xs, ys, alpha)
    return y_mean + ((pred_X - mean_x) / std_x) @ w


def loocv_ridge(records, feature_names, online_by_key, alphas):
    train = [r for r in records if online_by_key.get(r["param_key"]) is not None]
    if len(train) < 4:
        raise RuntimeError("Need at least 4 online regime points for calibration")
    y = np.array([online_by_key[r["param_key"]] for r in train], dtype=np.float64)
    X, _missing, _medians = build_matrix(train, feature_names, train)

    best = None
    diagnostics = []
    for alpha in alphas:
        preds = []
        for i in range(len(train)):
            mask = np.ones(len(train), dtype=bool)
            mask[i] = False
            pred = ridge_predict(X[mask], y[mask], X[i : i + 1], alpha)[0]
            preds.append(pred)
        preds = np.array(preds)
        err = preds - y
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err * err)))
        pair_acc = pairwise_accuracy(preds, y)
        diag = {"alpha": alpha, "mae": mae, "rmse": rmse, "pairwise_accuracy": pair_acc}
        diagnostics.append(diag)
        if best is None or (mae, -pair_acc) < (best["mae"], -best["pairwise_accuracy"]):
            best = diag
    return train, diagnostics, best


def pairwise_accuracy(signal, target):
    ok = 0
    total = 0
    for i in range(len(signal)):
        for j in range(i + 1, len(signal)):
            ds = signal[i] - signal[j]
            dt = target[i] - target[j]
            if abs(ds) < 1e-12 or abs(dt) < 1e-12:
                continue
            total += 1
            if ds * dt > 0:
                ok += 1
    return float(ok / total) if total else None


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    sx = rx.std()
    sy = ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return None
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def signal_reliability(train_records, online_by_key):
    y = np.array([online_by_key[r["param_key"]] for r in train_records], dtype=np.float64)
    out = []
    for name in BASE_FEATURES:
        vals = [r["features"].get(name) for r in train_records]
        idx = [i for i, v in enumerate(vals) if v is not None and np.isfinite(float(v))]
        if len(idx) < 4:
            continue
        x = np.array([float(vals[i]) for i in idx], dtype=np.float64)
        yy = y[idx]
        # For nmse lower is better; invert sign before pairwise accuracy.
        oriented = -x if name.endswith("_nmse") else x
        out.append(
            {
                "signal": name,
                "n": len(idx),
                "pairwise_accuracy": pairwise_accuracy(oriented, yy),
                "spearman": spearman(oriented, yy),
            }
        )
    out.sort(
        key=lambda r: (
            -(r["pairwise_accuracy"] if r["pairwise_accuracy"] is not None else -1),
            -(r["spearman"] if r["spearman"] is not None else -1),
        )
    )
    return out


def add_predictions(records, online_by_key):
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    train_all, diag_all, best_all = loocv_ridge(records, BASE_FEATURES, online_by_key, alphas)
    train_param, diag_param, best_param = loocv_ridge(records, PARAM_FEATURES, online_by_key, alphas)

    y_all = np.array([online_by_key[r["param_key"]] for r in train_all], dtype=np.float64)
    X_train_all, _miss_train_all, _ = build_matrix(train_all, BASE_FEATURES, train_all)
    X_all, miss_all, _ = build_matrix(records, BASE_FEATURES, train_all)
    pred_all = ridge_predict(X_train_all, y_all, X_all, best_all["alpha"])

    y_param = np.array([online_by_key[r["param_key"]] for r in train_param], dtype=np.float64)
    X_train_param, _miss_train_param, _ = build_matrix(train_param, PARAM_FEATURES, train_param)
    X_param, miss_param, _ = build_matrix(records, PARAM_FEATURES, train_param)
    pred_param = ridge_predict(X_train_param, y_param, X_param, best_param["alpha"])

    train_param_mat = np.array([[r["features"][k] for k in PARAM_FEATURES] for r in train_param])
    mean_p = train_param_mat.mean(axis=0)
    std_p = train_param_mat.std(axis=0)
    std_p[std_p < 1e-12] = 1.0

    for i, rec in enumerate(records):
        param_vec = np.array([rec["features"][k] for k in PARAM_FEATURES])
        dist = float(np.min(np.sqrt(np.sum(((train_param_mat - param_vec) / std_p) ** 2, axis=1))))
        pred = 0.60 * float(pred_all[i]) + 0.40 * float(pred_param[i])
        known = online_by_key.get(rec["param_key"])
        uncertainty = (
            0.60 * best_all["mae"]
            + 0.40 * best_param["mae"]
            + 0.00003 * miss_all[i]
            + 0.00008 * dist
        )
        penalty, penalty_notes = risk_penalty(rec)
        rec["online"] = known
        rec["calibrated"] = {
            "pred_online": known if known is not None else pred,
            "model_pred_online": pred,
            "uncertainty": uncertainty,
            "lower": (known if known is not None else pred) - uncertainty,
            "upper": (known if known is not None else pred) + uncertainty,
            "risk_penalty": 0.0 if known is not None else penalty,
            "decision_score": (
                known if known is not None else pred - uncertainty - penalty
            ),
            "risk_penalty_notes": penalty_notes,
            "param_distance": dist,
            "missing_features": float(miss_all[i] + miss_param[i]),
        }
        rec["recommendation"] = recommendation(rec)

    model = {
        "all_feature_model": {"best": best_all, "diagnostics": diag_all, "features": BASE_FEATURES},
        "param_model": {"best": best_param, "diagnostics": diag_param, "features": PARAM_FEATURES},
        "train_versions": [
            {
                "param_key": r["param_key"],
                "params": r["params"],
                "online": online_by_key[r["param_key"]],
            }
            for r in train_all
        ],
        "signal_reliability": signal_reliability(train_all, online_by_key),
    }
    return records, model


def recommendation(rec):
    if rec["online"] is not None:
        return "known_online"
    p = rec["params"]
    c, l, m, h = p["clear"], p["light"], p["medium"], p["heavy"]
    notes = []
    if c > 0.10:
        notes.append("clear_high_risk")
    if l >= 1.0:
        notes.append("light_1_refuted")
    elif l > 0.30:
        notes.append("light_above_current_best")
    elif l < 0.30:
        notes.append("tests_lower_light")
    if m > 1.0 or h > 1.0:
        notes.append("beyond_confirmed_geometry_range")
    if m >= 1.0 and h >= 1.0:
        notes.append("keeps_medium_heavy_max")
    elif m < 1.0 or h < 1.0:
        notes.append("weakens_confirmed_medium_heavy")
    return ", ".join(notes) if notes else "neutral"


def risk_penalty(rec):
    """Risk penalty in online-score units.

    These penalties encode only facts already observed online:
    - light=1.00 lost to light=0.50 and light=0.30;
    - clear above 0.10 was historically worse;
    - medium/heavy=1.00 is confirmed, but >1.00 is extrapolation.
    """

    if rec.get("online") is not None:
        return 0.0, []
    p = rec["params"]
    c, l, m, h = p["clear"], p["light"], p["medium"], p["heavy"]
    penalty = 0.0
    notes = []

    if l >= 1.0:
        penalty += 0.00075
        notes.append("light>=1 refuted")
    elif l > 0.30:
        penalty += 0.00025 * ((l - 0.30) / 0.20)
        notes.append("light above current best")

    if c > 0.10:
        penalty += 0.00045 * ((c - 0.10) / 0.20)
        notes.append("clear above safe value")

    over = max(0.0, m - 1.0) + max(0.0, h - 1.0)
    if over > 0:
        penalty += 0.00055 * (over / 0.20)
        notes.append("medium/heavy > 1 extrapolation")

    under = max(0.0, 1.0 - m) + max(0.0, 1.0 - h)
    if under > 0 and h >= 0.8:
        penalty += 0.00018 * (under / 0.20)
        notes.append("weakens confirmed medium/heavy")

    return float(penalty), notes


def fmt(x, digits=6):
    if x is None:
        return "-"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


def params_text(rec):
    p = rec["params"]
    return f"{p['clear']:.2f}/{p['light']:.2f}/{p['medium']:.2f}/{p['heavy']:.2f}"


def sort_candidate_key(rec):
    online = rec.get("online")
    if online is not None:
        return (0, -online)
    cal = rec["calibrated"]
    return (1, -cal["decision_score"], -cal["lower"])


def build_markdown(records, model, online_rows, out_md):
    lines = []
    lines.append("# Calibrated Round1 Local Scorer")
    lines.append("")
    lines.append("本文件由 `code/calibrated_local_scorer.py` 生成。它在原始本地 C 的基础上，用已知线上提交结果做小样本校准。")
    lines.append("")
    lines.append("## How To Use")
    lines.append("")
    lines.append("- `pred_online` 是校准模型给出的线上分数预测；已提交版本直接显示真实 online。")
    lines.append("- `uncertainty` 是基于 leave-one-out 误差、缺失本地特征、参数外推距离得到的粗略不确定性。")
    lines.append("- `lower/upper` 用于提交决策：优先选择 `lower` 不太差、且能验证明确假设的候选。")
    lines.append("- 这个系统不能替代线上，但比单看 `local_mean_score` 更接近当前线上规律。")
    lines.append("")

    lines.append("## Calibration Diagnostics")
    lines.append("")
    best_all = model["all_feature_model"]["best"]
    best_param = model["param_model"]["best"]
    lines.append("| model | alpha | LOO MAE | LOO RMSE | LOO pairwise accuracy |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| all features | {best_all['alpha']:.5g} | {best_all['mae']:.6f} | {best_all['rmse']:.6f} | {best_all['pairwise_accuracy']:.3f} |"
    )
    lines.append(
        f"| params only | {best_param['alpha']:.5g} | {best_param['mae']:.6f} | {best_param['rmse']:.6f} | {best_param['pairwise_accuracy']:.3f} |"
    )
    lines.append("")

    lines.append("## Signal Reliability")
    lines.append("")
    lines.append("| signal | n | pairwise accuracy | spearman |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in model["signal_reliability"][:18]:
        lines.append(
            f"| `{row['signal']}` | {row['n']} | {fmt(row['pairwise_accuracy'], 3)} | {fmt(row['spearman'], 3)} |"
        )
    lines.append("")

    lines.append("## Known Online Regime Versions")
    lines.append("")
    lines.append("| version | params c/l/m/h | online | conclusion |")
    lines.append("| --- | --- | ---: | --- |")
    for row in online_rows:
        if row["method"] != "regime_s1style_geom_blend" or row["param_key"] is None:
            continue
        key = row["param_key"]
        lines.append(
            f"| {row['version']} | {key[0]:.2f}/{key[1]:.2f}/{key[2]:.2f}/{key[3]:.2f} | {fmt(row['online'])} | {row['conclusion']} |"
        )
    lines.append("")

    lines.append("## Candidate Ranking")
    lines.append("")
    lines.append("| candidate names | params c/l/m/h | online | pred_online | uncertainty | risk_penalty | decision_score | lower | upper | local_mean | feature | stress | clear | heavy | recommendation |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for rec in sorted(records, key=sort_candidate_key):
        cal = rec["calibrated"]
        f = rec["features"]
        names = ", ".join(rec["names"][:3])
        if len(rec["names"]) > 3:
            names += ", ..."
        lines.append(
            "| {names} | {params} | {online} | {pred} | {unc} | {risk_penalty} | {decision} | {lower} | {upper} | "
            "{local} | {feature} | {stress} | {clear} | {heavy} | {recmd} |".format(
                names=names,
                params=params_text(rec),
                online=fmt(rec.get("online")),
                pred=fmt(cal["pred_online"]),
                unc=fmt(cal["uncertainty"]),
                risk_penalty=fmt(cal["risk_penalty"]),
                decision=fmt(cal["decision_score"]),
                lower=fmt(cal["lower"]),
                upper=fmt(cal["upper"]),
                local=fmt(f.get("local_mean_score")),
                feature=fmt(f.get("feature_matched_score")),
                stress=fmt(f.get("stress_feature_boundary_score")),
                clear=fmt(f.get("clear_los_score")),
                heavy=fmt(f.get("heavy_blocked_score")),
                recmd=rec["recommendation"],
            )
        )
    lines.append("")

    lines.append("## Practical Decision Rule")
    lines.append("")
    lines.append("1. 已知 online 最高优先作为回退包。")
    lines.append("2. 新候选要么 `pred_online` 高于当前 best，要么能验证一个高价值单变量假设。")
    lines.append("3. 如果候选削弱了已被线上确认的 `medium/heavy=1.00`，除非预测优势明显，否则不优先提交。")
    lines.append("4. 对当前阶段，light 低权重区间是主要待验证方向；`light=1.00` 已被线上证伪。")

    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-glob", default=str(PATHS.analysis / "results/json/regime_rho_combo_eval*.json"))
    parser.add_argument("--experiment-record", default=str(PATHS.experiment_records / "EXPERIMENT_RECORD.md"))
    parser.add_argument("--out-json", default=str(PATHS.analysis / "results/json/calibrated_local_scoreboard.json"))
    parser.add_argument("--out-md", default=str(PATHS.analysis / "results/calibrated_local_scoreboard.md"))
    parser.add_argument("--include-nonstandard", action="store_true")
    args = parser.parse_args()

    online_rows = load_online_rows(args.experiment_record)
    online_by_key, versions_by_key = online_by_param(online_rows)
    records = aggregate_records(args.eval_glob, include_nonstandard=args.include_nonstandard)
    records, model = add_predictions(records, online_by_key)
    for rec in records:
        rec["versions"] = versions_by_key.get(rec["param_key"], [])

    payload = {
        "notes": {
            "standard_sources_only": not args.include_nonstandard,
            "excluded_reason": "geom_rtheta sources are excluded by default because they share rho tuples with standard candidates but use a different geometry branch.",
        },
        "model": model,
        "records": records,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    build_markdown(records, model, online_rows, args.out_md)
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "out_md": args.out_md,
                "records": len(records),
                "online_train_points": len(model["train_versions"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

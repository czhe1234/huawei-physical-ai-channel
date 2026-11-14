#!/usr/bin/env python3
"""Analyze relationships between local validation signals and online scores.

This report is intentionally separate from ``calibrated_local_scorer.py``:

- the calibrated scorer ranks standard regime-rho candidates;
- this analyzer explains which local signals historically agreed with online
  results, and which signals became misleading.

The strongest lesson from v45/v46 is that a new mechanism can look good on a
local proxy while harming online score.  This script makes that failure mode
explicit instead of hiding it behind one scalar prediction.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
import math
from pathlib import Path

import numpy as np

from calibrated_local_scorer import (
    BASE_FEATURES,
    PARAM_FEATURES,
    aggregate_records,
    build_matrix,
    fmt,
    load_online_rows,
    online_by_param,
    pairwise_accuracy,
    ridge_predict,
    spearman,
)


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return None
    sx = x.std()
    sy = y.std()
    if sx < 1e-12 or sy < 1e-12:
        return None
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def online_map_by_version(rows):
    out = {}
    for row in rows:
        if row.get("online") is not None:
            out[row["version"]] = row["online"]
    return out


def standard_training_records(records, online_by_key):
    train = []
    for rec in records:
        online = online_by_key.get(rec["param_key"])
        if online is None:
            continue
        item = {
            "param_key": rec["param_key"],
            "params": rec["params"],
            "names": rec["names"],
            "features": rec["features"],
            "online": float(online),
        }
        train.append(item)
    train.sort(key=lambda r: r["online"])
    return train


def feature_relationships(train):
    rows = []
    y_all = np.array([r["online"] for r in train], dtype=np.float64)
    for name in BASE_FEATURES:
        xs = []
        ys = []
        for r in train:
            value = r["features"].get(name)
            if value is None or not np.isfinite(float(value)):
                continue
            xs.append(float(value))
            ys.append(r["online"])
        if len(xs) < 4:
            continue
        x = np.array(xs, dtype=np.float64)
        y = np.array(ys, dtype=np.float64)
        oriented = -x if name.endswith("_nmse") else x
        rows.append(
            {
                "signal": name,
                "n": len(x),
                "pearson": pearson(oriented, y),
                "spearman": spearman(oriented, y),
                "pairwise_accuracy": pairwise_accuracy(oriented, y),
                "x_min": float(np.min(x)),
                "x_max": float(np.max(x)),
            }
        )
    rows.sort(
        key=lambda r: (
            -(r["pairwise_accuracy"] if r["pairwise_accuracy"] is not None else -1),
            -(r["spearman"] if r["spearman"] is not None else -1),
            -(abs(r["pearson"]) if r["pearson"] is not None else -1),
        )
    )
    return rows


def loocv_model(train, feature_names, alphas):
    y = np.array([r["online"] for r in train], dtype=np.float64)
    X, _missing, _ = build_matrix(train, feature_names, train)
    best = None
    all_rows = []
    for alpha in alphas:
        preds = []
        for i in range(len(train)):
            mask = np.ones(len(train), dtype=bool)
            mask[i] = False
            preds.append(ridge_predict(X[mask], y[mask], X[i : i + 1], alpha)[0])
        preds = np.asarray(preds, dtype=np.float64)
        err = preds - y
        row = {
            "alpha": float(alpha),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "pairwise_accuracy": pairwise_accuracy(preds, y),
        }
        all_rows.append(row)
        if best is None or (row["mae"], -row["pairwise_accuracy"]) < (
            best["mae"],
            -best["pairwise_accuracy"],
        ):
            best = row
    return {"best": best, "all": all_rows}


def heavy_curve(rows):
    points = []
    for row in rows:
        if row["method"] != "regime_s1style_geom_blend":
            continue
        key = row.get("param_key")
        if key is None or row.get("online") is None:
            continue
        c, l, m, h = key
        if abs(c) < 1e-9 and abs(l) < 1e-9 and abs(m - 1.0) < 1e-9:
            points.append({"version": row["version"], "heavy": h, "online": row["online"]})
    points.sort(key=lambda r: r["heavy"])
    fit = None
    if len(points) >= 4:
        h = np.array([p["heavy"] for p in points], dtype=np.float64)
        y = np.array([p["online"] for p in points], dtype=np.float64)
        coef = np.polyfit(h, y, deg=2)
        a, b, c0 = coef
        peak_h = None
        peak_y = None
        if abs(a) > 1e-12:
            peak_h = float(-b / (2.0 * a))
            peak_y = float(np.polyval(coef, peak_h))
        preds = np.polyval(coef, h)
        fit = {
            "degree": 2,
            "coef": [float(v) for v in coef],
            "rmse": float(np.sqrt(np.mean((preds - y) ** 2))),
            "peak_heavy": peak_h,
            "peak_online": peak_y,
        }
    return {"points": points, "quadratic_fit": fit}


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def find_result(results, name):
    for row in results:
        if row.get("name") == name:
            return row
    return None


def mean_named_from_sets(payload, names):
    values = {name: [] for name in names}
    for set_obj in payload.get("sets", []):
        for row in set_obj.get("results", []):
            name = row.get("name")
            if name in values and row.get("score") is not None:
                values[name].append(float(row["score"]))
    return {name: (float(np.mean(vals)) if vals else None) for name, vals in values.items()}


def special_case_report(paths, online_by_version):
    out = []

    v44 = load_json(paths["v44"])
    if v44 is not None:
        means = mean_named_from_sets(v44, ["adaptive_v44", "fixed_h4", "fixed_h3"])
        out.append(
            {
                "version": "v44",
                "method": "adaptive_heavy_rho",
                "online": online_by_version.get("v44"),
                "baseline_version": "v36",
                "baseline_online": online_by_version.get("v36"),
                "local_signal": "adaptive_v44 local mean score minus fixed_h4",
                "local_delta": (
                    means["adaptive_v44"] - means["fixed_h4"]
                    if means["adaptive_v44"] is not None and means["fixed_h4"] is not None
                    else None
                ),
                "online_delta": (
                    online_by_version.get("v44") - online_by_version.get("v36")
                    if online_by_version.get("v44") is not None
                    and online_by_version.get("v36") is not None
                    else None
                ),
                "interpretation": "本地分组优于 fixed_h4，但线上低于统一 v36，说明当前 heavy 子分组规则不准。",
            }
        )

    v45 = load_json(paths["v45"])
    if v45 is not None:
        pathmix = find_result(v45.get("results", []), "pathmix_1")
        old = find_result(v45.get("results", []), "old_v36_rule")
        out.append(
            {
                "version": "v45",
                "method": "path_candidate_geom_blend",
                "online": online_by_version.get("v45"),
                "baseline_version": "v36",
                "baseline_online": online_by_version.get("v36"),
                "local_signal": "pathmix_1 score minus old_v36_rule score",
                "local_delta": (
                    float(pathmix["score"]) - float(old["score"])
                    if pathmix is not None and old is not None
                    else None
                ),
                "online_delta": (
                    online_by_version.get("v45") - online_by_version.get("v36")
                    if online_by_version.get("v45") is not None
                    and online_by_version.get("v36") is not None
                    else None
                ),
                "interpretation": "本地 PAS/PDP 小幅提升但线上下降，说明当前 specular path candidate 约束不是可靠线上信号。",
            }
        )

    v46 = load_json(paths["v46"])
    if v46 is not None:
        best = find_result(v46.get("results", []), "att_th1e-06_k64_f0_l8")
        out.append(
            {
                "version": "v46",
                "method": "deadzone_attenuated_v36",
                "online": online_by_version.get("v46"),
                "baseline_version": "v36",
                "baseline_online": online_by_version.get("v36"),
                "local_signal": "local NMSE delta vs v36",
                "local_delta": best.get("delta_nmse_vs_v36") if best is not None else None,
                "online_delta": (
                    online_by_version.get("v46") - online_by_version.get("v36")
                    if online_by_version.get("v46") is not None
                    and online_by_version.get("v36") is not None
                    else None
                ),
                "interpretation": "本地 NMSE 大幅改善但线上崩溃，是最强反例：低功率/死区衰减不能用当前本地指标背书。",
            }
        )
    return out


def md_table(lines, headers, rows):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")


def write_report(payload, out_md):
    lines = []
    lines.append("# Online-Local Relation Report")
    lines.append("")
    lines.append("本报告由 `code/online_local_relation_analyzer.py` 生成，用来分析历史线上结果和本地指标的关系。")
    lines.append("")
    lines.append("## 结论摘要")
    lines.append("")
    lines.append("1. 对 `regime_s1style_geom_blend` 同族参数扫描，线上结果主要跟 `rho_heavy/rho_medium`、`medium_heavy_mean` 相关。")
    lines.append("2. 普通本地 `score/NMSE` 不是最可靠信号；特别是 `heavy_blocked_score` 和部分 NMSE 信号会反向误导。")
    lines.append("3. v44/v45/v46 证明：一旦换机制，本地小幅提升甚至大幅 NMSE 改善都不能直接预测线上。")
    lines.append("4. 当前评价器应分两类使用：同族 rho 扫描可以做数值预测；新机制只能做风险分析和线上试探优先级。")
    lines.append("")

    lines.append("## 标准 Regime-Rho 校准")
    lines.append("")
    model_rows = []
    for name, diag in payload["models"].items():
        best = diag["best"]
        model_rows.append(
            [
                name,
                fmt(best["alpha"], 5),
                fmt(best["mae"]),
                fmt(best["rmse"]),
                fmt(best["pairwise_accuracy"], 3),
            ]
        )
    md_table(lines, ["model", "alpha", "LOO MAE", "LOO RMSE", "pairwise acc"], model_rows)
    lines.append("")

    lines.append("## 最相关本地信号")
    lines.append("")
    rel_rows = []
    for row in payload["feature_relationships"][:20]:
        rel_rows.append(
            [
                f"`{row['signal']}`",
                str(row["n"]),
                fmt(row["pairwise_accuracy"], 3),
                fmt(row["spearman"], 3),
                fmt(row["pearson"], 3),
            ]
        )
    md_table(lines, ["signal", "n", "pairwise acc", "spearman", "pearson"], rel_rows)
    lines.append("")

    lines.append("## Heavy 曲线")
    lines.append("")
    curve_rows = []
    for row in payload["heavy_curve"]["points"]:
        curve_rows.append([row["version"], fmt(row["heavy"], 2), fmt(row["online"])])
    md_table(lines, ["version", "heavy rho", "online"], curve_rows)
    fit = payload["heavy_curve"].get("quadratic_fit")
    if fit:
        lines.append("")
        lines.append(
            f"二次拟合峰值约在 `heavy={fit['peak_heavy']:.3f}`，预测线上约 `{fit['peak_online']:.6f}`，拟合 RMSE `{fit['rmse']:.6f}`。"
        )
    lines.append("")

    lines.append("## 新机制反例")
    lines.append("")
    special_rows = []
    for row in payload["special_cases"]:
        special_rows.append(
            [
                row["version"],
                row["method"],
                fmt(row["online"]),
                fmt(row["online_delta"]),
                fmt(row["local_delta"]),
                row["interpretation"],
            ]
        )
    md_table(lines, ["version", "method", "online", "online delta vs v36", "local delta", "interpretation"], special_rows)
    lines.append("")

    lines.append("## 使用规则")
    lines.append("")
    lines.append("- 如果候选仍是 `clear/light/medium/heavy` 同族参数变化，可以参考数值预测和 heavy 曲线。")
    lines.append("- 如果候选改了机制，例如路径候选、死区衰减、神经 residual，不能只看本地 score/NMSE；必须进入“新机制风险”判断。")
    lines.append("- 当前已知强规则：`clear=0, light=0, medium=1` 保持不动；`heavy` 的有效区间大约在 `3-4.5`，`5` 已过强；不要整体压低 heavy 功率。")
    lines.append("")

    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-glob", default=str(PATHS.analysis / "results/json/regime_rho_combo_eval*.json"))
    parser.add_argument("--experiment-record", default=str(PATHS.experiment_records / "EXPERIMENT_RECORD.md"))
    parser.add_argument("--out-json", default=str(PATHS.analysis / "results/json/online_local_relation_report.json"))
    parser.add_argument("--out-md", default=str(PATHS.analysis / "results/online_local_relation_report.md"))
    parser.add_argument("--v44-json", default=str(PATHS.analysis / "results/json/adaptive_heavy_rho_eval_v44.json"))
    parser.add_argument("--v45-json", default=str(PATHS.analysis / "results/json/path_candidate_geom_blend_eval_v45_strong_500.json"))
    parser.add_argument("--v46-json", default=str(PATHS.analysis / "results/json/deadzone_attenuation_eval_500.json"))
    args = parser.parse_args()

    online_rows = load_online_rows(args.experiment_record)
    online_by_key, _versions = online_by_param(online_rows)
    online_by_version = online_map_by_version(online_rows)
    records = aggregate_records(args.eval_glob)
    train = standard_training_records(records, online_by_key)

    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    payload = {
        "notes": {
            "standard_scope": "Only standard regime_s1style_geom_blend parameter rows are used for numeric calibration.",
            "special_scope": "v44/v45/v46 are reported as out-of-family evidence, not used to fit the standard numeric predictor.",
        },
        "online_train_count": len(train),
        "models": {
            "params_only": loocv_model(train, PARAM_FEATURES, alphas),
            "all_local_features": loocv_model(train, BASE_FEATURES, alphas),
        },
        "feature_relationships": feature_relationships(train),
        "heavy_curve": heavy_curve(online_rows),
        "special_cases": special_case_report(
            {"v44": args.v44_json, "v45": args.v45_json, "v46": args.v46_json},
            online_by_version,
        ),
    }

    Path(args.out_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(payload, args.out_md)
    print(json.dumps({"out_json": args.out_json, "out_md": args.out_md}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

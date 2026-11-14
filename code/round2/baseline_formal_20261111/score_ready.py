"""Score each completed formal five-fold method and update the comparison."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Scoring uses a CPU-only CUDA mask")

import numpy as np
from common.official_metrics import (
    PointMetricStatistics, aggregate_statistics, point_metric_statistics,
)
from project_paths import PATHS
from round2.baseline_formal_20261111 import score as S
from round2.paper_baseline_training_v1 import common as C

R = S.R
A = S.A
METRICS = ("pas", "pdp", "nmse", "score")


def ready_methods():
    return [
        method for method in S.METHODS
        if all((A / method / f"outer{k}/prediction/manifest.json").is_file() for k in range(5))
    ]


def seal_method(method, inputs):
    S.F.SCORE_DIR = R / "evaluation"
    S.F.method_root = lambda name: A / name
    groups = S.F.manifests_for(method, inputs, verify_hashes=True)
    if groups is None:
        raise RuntimeError(f"Five-fold predictions are incomplete for {method}")
    training = {}
    for _, manifest, prediction in groups:
        training[str(manifest["outer"])] = S.training_evidence(method, manifest)
        values = np.load(prediction, mmap_mode="r", allow_pickle=False)
        for start in range(0, len(values), 8):
            if not np.isfinite(values[start:start + 8]).all():
                raise RuntimeError(f"Nonfinite predictions for {method}")
    manifests = {PATHS.relative(path): C.sha256(path) for path, _, _ in groups}
    path = R / "evaluation" / method / "predictions_sealed.json"
    payload = {
        "method": method,
        "query_count": 500,
        "input_identity": inputs.identity,
        "manifests": manifests,
        "training_evidence": training,
        "sealed_before_method_truth_scoring": True,
        "other_methods_training_frozen": True,
    }
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(f"Sealed method evidence changed for {method}")
    else:
        C.atomic_json(path, payload)
    return groups


def verify_method(method, groups, inputs):
    result_path = R / "evaluation" / method / "results.json"
    audit_path = result_path.parent / "independent_verification.json"
    recorded = json.loads(result_path.read_text())
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if (audit["result_sha256"] == C.sha256(result_path)
                and audit["five_prediction_manifest_hashes"] == recorded["prediction_manifests"]
                and audit["official_metric_sha256"] == C.sha256(PATHS.code_root / "common/official_metrics.py")):
            return audit
        raise RuntimeError(f"Existing verification identity changed for {method}")

    truth = np.load(C.RAW, mmap_mode="r", allow_pickle=False)
    query = np.asarray(inputs.query_ids, dtype=np.int64)
    lookup = {int(row): index for index, row in enumerate(query)}
    arrays = {key: np.zeros(500, dtype=np.float64)
              for key in ("pas", "pdp", "squared_error", "truth_energy")}
    for _, manifest, prediction in groups:
        values = np.load(prediction, mmap_mode="r", allow_pickle=False)
        rows = np.asarray(manifest["row_ids"], dtype=np.int64)
        for start in range(0, len(rows), 4):
            ids = rows[start:start + 4]
            statistics = point_metric_statistics(
                np.asarray(values[start:start + len(ids)]), np.asarray(truth[ids]))
            indices = [lookup[int(row)] for row in ids]
            for key in arrays:
                arrays[key][indices] = getattr(statistics, key)
    metrics = aggregate_statistics(PointMetricStatistics(**arrays))
    differences = {key: abs(float(metrics[key]) - float(recorded["metrics"][key]))
                   for key in metrics}
    if max(differences.values()) >= 1e-10:
        raise RuntimeError((method, differences))
    zero_count = int(np.sum(arrays["truth_energy"] == 0))
    if zero_count != 31 or recorded["zero_truth_row_count"] != 31:
        raise RuntimeError("Zero-channel rows differ from the fixed500 protocol")
    statistics_path = result_path.parent / "point_statistics.npz"
    if C.sha256(statistics_path) != recorded["point_statistics_sha256"]:
        raise RuntimeError("Stored point statistics changed")
    with np.load(statistics_path, allow_pickle=False) as stored:
        if not np.array_equal(stored["row_ids"], query):
            raise RuntimeError("Scored row order differs")
        point_differences = {
            key: float(np.max(np.abs(stored[key] - arrays[key]))) for key in arrays}
    if max(point_differences.values()) >= 1e-10:
        raise RuntimeError((method, point_differences))
    audit = {
        "status": "passed",
        "method": method,
        "metrics": metrics,
        "max_aggregate_delta": max(differences.values()),
        "max_point_statistic_deltas": point_differences,
        "zero_truth_count": zero_count,
        "result_sha256": C.sha256(result_path),
        "five_prediction_manifest_hashes": recorded["prediction_manifests"],
        "official_metric_sha256": C.sha256(PATHS.code_root / "common/official_metrics.py"),
        "verifier_sha256": C.sha256(PATHS.code_root / "round2/baseline_formal_20261111/score_ready.py"),
    }
    C.atomic_json(audit_path, audit)
    return audit


def write_table(inputs):
    old_path, old = S.old_rows()
    if old["query_ids_sha256"] != C.digest_ids(inputs.query_ids):
        raise RuntimeError("Historical score query set differs")
    retained = {row["key"]: row for row in old["rows"]}
    rows = [{"key": "v16", "method": "V16（历史版本）",
             "metrics": retained["v16"]["metrics"], "status": "retained"}]
    differences = {}
    completed = []
    for method in S.METHODS:
        folder = R / "evaluation" / method
        result_path = folder / "results.json"
        audit_path = folder / "independent_verification.json"
        count = sum(len(json.loads((A / method / f"outer{k}/prediction/manifest.json").read_text())["row_ids"])
                    for k in range(5)
                    if (A / method / f"outer{k}/prediction/manifest.json").is_file())
        metrics = None
        if result_path.exists() and audit_path.exists():
            result = json.loads(result_path.read_text())
            audit = json.loads(audit_path.read_text())
            if audit["status"] != "passed" or audit["result_sha256"] != C.sha256(result_path):
                raise RuntimeError("Score verification changed")
            metrics = result["metrics"]
            completed.append(method)
            differences[method] = {
                key: metrics[key] - retained[method]["metrics"][key] for key in METRICS}
        rows.append({"key": method, "method": S.LABEL[method], "metrics": metrics,
                     "status": "verified500" if metrics else "training_or_prediction_in_progress",
                     "prediction_count": count})
    rows.append({"key": "wrfgs_full_h_v1", "method": "WRF-GS+（完整H适配）",
                 "metrics": retained["wrfgs_full_h_v1"]["metrics"], "status": "retained"})
    payload = {
        "rows": rows, "old_baseline_deltas": differences,
        "query_ids_sha256": C.digest_ids(inputs.query_ids),
        "completed_new_methods": completed,
        "all_new_methods_formal500": len(completed) == len(S.METHODS),
        "retained_source": PATHS.relative(old_path), "retained_source_sha256": C.sha256(old_path),
        "scoring_rule": "score each method after its own five predictions are sealed",
        "ongoing_training_uses_scored_outer_results": False,
        "V16_historical_training_coverage_difference_retained": True,
    }
    C.atomic_json(R / "evaluation/comparison.json", payload)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    header = ["方法", "PAS ↑", "PDP ↑", "NMSE（线性）↓", "综合分 ↑", "状态"]
    writer.writerow(header)
    lines = ["# 华为仿真500点最新成绩", "", "| " + " | ".join(header) + " |",
             "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        values = [f'{row["metrics"][key]:.6f}' if row["metrics"] else "待完成"
                  for key in METRICS]
        status = "已有成绩" if row["status"] == "retained" else (
            "500点已核验" if row["metrics"] else f'已生成{row["prediction_count"]}/500点')
        entry = [row["method"], *values, status]
        writer.writerow(entry)
        lines.append("| " + " | ".join(entry) + " |")
    lines += [
        "",
        "每种修复版方法完成自身五组预测后，使用原500点指标评分并独立复算。",
        "V16与WRF-GS+使用已保存成绩。V16历史训练覆盖差异继续披露。",
        "PAS与PDP为余弦相似度。NMSE为线性误差比。综合分为0.4 PAS+0.4 PDP+0.2/(1+global NMSE)。",
        "500点包含31个零信道位置。所有运行方法继续执行已冻结的训练配置。",
    ]
    text = "\n".join(lines) + "\n"
    (R / "evaluation/comparison.csv").write_text("\ufeff" + buffer.getvalue())
    (R / "evaluation/RESULTS.md").write_text(text)
    (R / "RESULTS.md").write_text(text)
    return payload


def main(methods=None):
    if methods is None:
        methods = ready_methods()
    inputs = C.SimulationInputs()
    completed = {}
    started = time.monotonic()
    for method in methods:
        groups = seal_method(method, inputs)
        result = S.F.score_method(method, inputs)
        if result is None:
            raise RuntimeError(f"Missing completed result for {method}")
        verify_method(method, groups, inputs)
        completed[method] = result["metrics"]
    table = write_table(inputs)
    print(json.dumps({"status": "completed_methods_scored", "metrics": completed,
                      "all_methods_complete": table["all_new_methods_formal500"],
                      "elapsed_seconds": time.monotonic() - started},
                     ensure_ascii=False), flush=True)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=None)
    arguments = parser.parse_args()
    main(arguments.methods.split(",") if arguments.methods else None)


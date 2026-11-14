"""Make the requested table from accepted, archived scores; never run a model."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from project_paths import PATHS

OUT = PATHS.analysis / "results/round2_existing_comparison_20261101"
SUMMARY = PATHS.analysis / "results/round2_official_sionna_path_calibrator/fixed500_report.json"
DEPLOY = PATHS.root / "final_code/V16_fast_repro_submission/artifacts/round2/physical_ai_submission/deployment_manifest.json"
PROTOCOL = PATHS.analysis / "protocols/huawei_four_baseline_adapters_recovery_v2.json"
ELIGIBILITY = PATHS.analysis / "audits/simulation_recovery_v1/method_eligibility_recovery.json"
SETUP = PATHS.round2 / "Round2_Setup.json"
METRIC_CODE = PATHS.code_root / "common/official_metrics.py"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def identity(path):
    st = path.stat()
    return dict(path=PATHS.relative(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                inode=st.st_ino, sha256=sha(path))


def write_new(name, value):
    path = OUT / name
    if path.exists():
        raise FileExistsError("Preserve existing table output: " + str(path))
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def main():
    if not (OUT / "PLAN.md").exists():
        raise RuntimeError("Route contract missing")
    sources = [SUMMARY, DEPLOY, PROTOCOL, ELIGIBILITY, SETUP, METRIC_CODE]
    before = [identity(p) for p in sources]
    selected = json.loads(SUMMARY.read_text())["selected"]
    baseline = json.loads(DEPLOY.read_text())["baseline"]
    protocol = json.loads(PROTOCOL.read_text())
    if selected["count"] != 500:
        raise ValueError("Accepted fixed500 source count changed")
    if baseline["source_archive"] != "V16_source_online_0.686754.zip":
        raise ValueError("Wrong user-validated V16 archive")
    if baseline["source_archive_sha256"] != "18a7e257ac42a9a88c2c4941b9ed547537084b25b425ad8584b32d79689939a6":
        raise ValueError("V16 deployment source identity changed")
    if not all(m.get("trained") is False for m in protocol["methods"]):
        raise ValueError("Prediction inventory changed; recheck availability before filling table")
    for key in ("pas", "pdp", "nmse", "nmse_score", "score"):
        if not math.isfinite(selected[key]):
            raise ValueError("Nonfinite archived metric")
    if selected["nmse"] < 0:
        raise ValueError("NMSE must be a nonnegative global linear ratio")
    weights = json.loads(SETUP.read_text())["w"]
    if weights != [0.4, 0.4, 0.2]:
        raise ValueError("Official weighting changed")
    nmse_score = 1.0 / (1.0 + selected["nmse"])
    recalculated = weights[0]*selected["pas"] + weights[1]*selected["pdp"] + weights[2]*nmse_score
    discrepancy = abs(recalculated-selected["score"])
    if discrepancy > 1e-6 or abs(nmse_score-selected["nmse_score"]) > 1e-6:
        raise AssertionError("Archived components and competition formula disagree")

    rows = [dict(method="我们的方法（V16）", pas=selected["pas"], pdp=selected["pdp"],
                 nmse=selected["nmse"], score=recalculated, status="accepted_archived_fixed500_summary",
                 raw_H_rescored=False)]
    for method in ("NeRF²", "NeWRF", "nGRF", "WRF-GS+（功率版）"):
        rows.append(dict(method=method, pas=None, pdp=None, nmse=None, score=None,
                         status="missing_huawei_simulation_prediction", raw_H_rescored=False))
    header = ["方法", "PAS ↑", "PDP ↑", "NMSE（线性）↓", "比赛总分 ↑"]
    def cell(row, key):
        if row[key] is not None:
            return format(row[key], ".6f")
        if row["method"].startswith("WRF-GS+") and key != "pas":
            return "不适用"
        return "缺少预测"
    formatted = [[r["method"]]+[cell(r,k) for k in ("pas","pdp","nmse","score")] for r in rows]
    csv_path = OUT / "comparison.csv"
    if csv_path.exists():
        raise FileExistsError(csv_path)
    with csv_path.open("x", newline="", encoding="utf-8-sig") as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(formatted)
    lines = ["# 华为仿真现有结果表", "",
             "V16采用用户已验证版本的历史500点验证汇总。本轮复核总分公式，未重新从H计算PAS、PDP或NMSE。", "",
             "| " + " | ".join(header) + " |", "|---|---:|---:|---:|---:|"]
    lines += ["| " + " | ".join(r) + " |" for r in formatted]
    lines += ["",
              f"V16旧线上总分：{baseline['online_score']:.6f}，单独保留。线上Test与本地500点验证是两组数据；线上逐项指标和Test真值未留存，未混填。",
              "原fixed500预测及逐点充分统计已缺失，以上前三项沿用冻结机器报告selected；当前新物理输入恢复版未用于本表。",
              "四个文献方法均缺可用于华为192频点仿真的预测。NeWRF随机小网络检查、实测单频NeRF²结果均不作为仿真成绩。当前WRF-GS+功率版只适用PAS。",
              "未训练、未拟合、未推理、未执行新射线计算。原数据、模型、预测和旧成绩保持不变。",
              "总分 = 0.4 PAS + 0.4 PDP + 0.2/(1+global NMSE)。仅一行有成绩，不据此判定横向最优。", ""]
    after = [identity(p) for p in sources]
    if before != after:
        raise RuntimeError("A protected source changed")
    result = dict(schema="huawei-existing-result-comparison-v1", generated_utc=datetime.now(timezone.utc).isoformat(),
                  scope="user_accepted_existing_V16_and_available_literature_predictions",
                  rows=rows, archived_V16_selected=selected, archived_online_total=baseline["online_score"],
                  online_and_local_datasets_separate=True, nmse_scale="linear_global_error_energy_ratio",
                  formula_weights=weights, recalculated_total=recalculated, total_absolute_difference=discrepancy,
                  original_prediction_rescored=False, trained=False, fitted=False, model_forward=False,
                  ray_tracing=False, source_files_preserved=(before==after), source_identities=before,
                  missing_methods=[r["method"] for r in rows[1:]],
                  excluded_new_recovery="legacy_fixed_weights_regenerated_query_v3",
                  online_baseline_identity=baseline,
                  environment=dict(python=platform.python_version(), device="CPU", dependencies_changed=False),
                  code=PATHS.relative(Path(__file__).resolve()), code_sha256=sha(Path(__file__).resolve()))
    write_new("comparison.md", "\n".join(lines))
    write_new("results.json", json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print("\n".join(lines[4:11]))
    print(json.dumps(dict(formula_difference=discrepancy,source_files_preserved=True,trained=False,raw_H_rescored=False)))

if __name__ == "__main__":
    main()

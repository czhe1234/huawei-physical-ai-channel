"""Score the retained and frequency-minibatch fold selection. CPU only."""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
import time
import traceback
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Use a CPU-only device mask for scoring")

import numpy as np
from project_paths import PATHS
from round2.paper_baseline_training_v1 import common as C
from round2.paper_baseline_training_v1 import final_score as F
from round2.baseline_formal_20261111 import score as Parent
from round2.baseline_formal_20261111 import score_ready as Check
from round2.baseline_formal_20261111 import verify as Protect

VERSION = "freqminibatch_20261112"
R = C.REPORT / VERSION
A = C.ART / VERSION
OLD_R = C.REPORT / "formal_20261111"
OLD_A = C.ART / "formal_20261111"
METHODS = ("nerf2", "newrf")
NEW_FOLDS = {"nerf2": (2, 3, 4), "newrf": (4,)}
LABELS = {"nerf2": "NeRF²（频点小批量续训）", "newrf": "NeWRF（频点小批量续训）"}
METRICS = ("pas", "pdp", "nmse", "score")


def selected_report(method, fold):
    return (R if fold in NEW_FOLDS[method] else OLD_R) / method


def selected_artifact(method, fold):
    return (A if fold in NEW_FOLDS[method] else OLD_A) / method / f"outer{fold}"


class FoldRoot:
    """Resolve only the five registered fold paths for the original validator."""

    def __init__(self, method):
        if method not in METHODS:
            raise ValueError(method)
        self.method = method

    def __truediv__(self, part):
        if part not in [f"outer{k}" for k in range(5)]:
            raise ValueError(("Unregistered fold path", part))
        return selected_artifact(self.method, int(part[5:]))


def preservation():
    evidence = json.loads((R / "preservation_before.json").read_text())
    for relative, digest in evidence["hashes"].items():
        if C.sha256(PATHS.root / relative) != digest:
            raise RuntimeError(("Retained file changed", relative))
    for item in evidence["stats"]:
        st = (PATHS.root / item["path"]).stat()
        if [st.st_size, st.st_mtime_ns, st.st_ino] != item["stat"]:
            raise RuntimeError(("Retained model or array changed", item["path"]))
    return {"parent": Protect.preserve(), "hashes": len(evidence["hashes"]),
            "model_array_stats": len(evidence["stats"]), "stats_are_not_content_hashes": True}


def validate_selection(inputs):
    selection = json.loads((R / "selected_inputs.json").read_text())
    if selection["input_identity"] != inputs.identity:
        raise RuntimeError("Input identity changed")
    if selection["query_ids_sha256"] != C.digest_ids(inputs.query_ids):
        raise RuntimeError("Query row identity changed")
    for relative, digest in selection["retained_manifest_hashes"].items():
        if C.sha256(PATHS.root / relative) != digest:
            raise RuntimeError(("Retained prediction manifest changed", relative))
    for relative, digest in selection["retained_result_hashes"].items():
        if C.sha256(PATHS.root / relative) != digest:
            raise RuntimeError(("Retained score changed", relative))
    return selection


def training_evidence(method, manifest):
    fold = int(manifest["outer"])
    previous = Parent.R
    try:
        Parent.R = selected_report(method, fold).parent
        result = Parent.training_evidence(method, manifest)
    finally:
        Parent.R = previous
    if fold in NEW_FOLDS[method]:
        role = selected_report(method, fold) / f"outer{fold}/refit/result.json"
        payload = json.loads(role.read_text())
        if method == "nerf2":
            frequency = payload["frequency_sampling"]
            k = int(frequency["training_K"])
            fields = frequency["per_field"]
            if len(fields) != 8:
                raise RuntimeError("NeRF2 frequency field coverage")
            for name, field in fields.items():
                counts = np.asarray(field["tone_step_counts"], dtype=np.int64)
                scalars = np.asarray(field["tone_complex_supervision_elements"], dtype=np.int64)
                inherited = int(field["full192_pre_migration_updates"])
                sampled = int(field["minibatch_updates"])
                updates = int(payload["actual_coverage"][name]["actual_optimizer_updates"])
                if counts.shape != (192,) or scalars.shape != (192,) or min(counts) <= 0 or min(scalars) <= 0:
                    raise RuntimeError("NeRF2 incomplete frequency coverage")
                if int(counts.sum()) != 192 * inherited + k * sampled or inherited + sampled != updates:
                    raise RuntimeError("NeRF2 frequency/update budget mismatch")
                if int(scalars.sum()) != field["actual_complex_supervision_elements"]:
                    raise RuntimeError("NeRF2 actual supervision count mismatch")
        else:
            k = int(payload["frequency_minibatch"])
            if payload["full_inference_tones"] != 192 or len(payload["fields"]) != 8:
                raise RuntimeError("NeWRF complete-H frequency identity")
            for field in payload["fields"].values():
                frequency = field["frequency_exposure"]
                counts = np.asarray(frequency["tone_selected_updates"], dtype=np.int64)
                scalars = np.asarray(frequency["tone_scalar_gradient_labels"], dtype=np.int64)
                inherited = int(frequency["legacy_fullband_updates"])
                sampled = int(frequency["minibatch_updates"])
                updates = sum(int(value["updates"]) for value in field["stages"].values())
                if counts.shape != (192,) or scalars.shape != (192,) or min(counts) <= 0 or min(scalars) <= 0:
                    raise RuntimeError("NeWRF incomplete frequency coverage")
                if int(counts.sum()) != 192 * inherited + k * sampled or inherited + sampled != updates:
                    raise RuntimeError("NeWRF frequency/update budget mismatch")
    return result


def configure_original_scorer():
    F.SCORE_DIR = R / "evaluation"
    F.method_root = lambda method: FoldRoot(method)
    Check.R = R


def seal(method, inputs):
    configure_original_scorer()
    groups = F.manifests_for(method, inputs, verify_hashes=True)
    if groups is None:
        raise RuntimeError(("Incomplete five-fold predictions", method))
    evidence = {}
    for _, manifest, path in groups:
        evidence[str(manifest["outer"])] = training_evidence(method, manifest)
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        for lo in range(0, len(values), 8):
            if not np.isfinite(values[lo:lo + 8]).all():
                raise RuntimeError("Nonfinite full-H predictions")
    payload = {
        "method": method, "query_count": 500, "input_identity": inputs.identity,
        "manifests": {PATHS.relative(p): C.sha256(p) for p, _, _ in groups},
        "training_evidence": evidence,
        "sealed_before_method_truth_scoring": True,
        "selection_depends_on_completion_not_test_score": True,
        "new_frequency_minibatch_folds": list(NEW_FOLDS[method]),
        "retained_full_frequency_folds": [k for k in range(5) if k not in NEW_FOLDS[method]],
        "other_training_not_tuned_on_scored_outer_results": True,
    }
    target = R / "evaluation" / method / "predictions_sealed.json"
    if target.exists():
        if json.loads(target.read_text()) != payload:
            raise RuntimeError("Sealed prediction identities changed")
    else:
        C.atomic_json(target, payload)
    return groups


def prediction_count(method):
    count = 0
    for fold in range(5):
        p = selected_artifact(method, fold) / "prediction/manifest.json"
        if p.is_file():
            count += len(json.loads(p.read_text())["row_ids"])
    return count


def table(inputs):
    validate_selection(inputs)
    old_path, old = Parent.old_rows()
    retained = {row["key"]: row for row in old["rows"]}
    ng_path = OLD_R / "evaluation/ngrf/results.json"
    ng = json.loads(ng_path.read_text())
    ng_audit = json.loads((ng_path.parent / "independent_verification.json").read_text())
    if ng_audit["status"] != "passed" or ng_audit["result_sha256"] != C.sha256(ng_path):
        raise RuntimeError("Retained nGRF verification differs")
    rows = [{"key": "v16", "method": "V16（历史版本）", "metrics": retained["v16"]["metrics"],
             "status": "retained", "origin": PATHS.relative(old_path)}]
    for method in METHODS:
        p = R / "evaluation" / method / "results.json"
        a = p.parent / "independent_verification.json"
        metrics = None
        if p.is_file() and a.is_file():
            result = json.loads(p.read_text())
            audit = json.loads(a.read_text())
            if audit["status"] != "passed" or audit["result_sha256"] != C.sha256(p):
                raise RuntimeError("New score verification changed")
            metrics = result["metrics"]
        rows.append({"key": method, "method": LABELS[method], "metrics": metrics,
                     "status": "verified500" if metrics else "in_progress",
                     "prediction_count": prediction_count(method),
                     "new_frequency_minibatch_folds": list(NEW_FOLDS[method]),
                     "retained_full_frequency_folds": [k for k in range(5) if k not in NEW_FOLDS[method]]})
    rows += [{"key": "ngrf", "method": "nGRF（已核验修复版）", "metrics": ng["metrics"],
              "status": "retained_verified500", "origin": PATHS.relative(ng_path)},
             {"key": "wrfgs_full_h_v1", "method": "WRF-GS+（完整H适配）",
              "metrics": retained["wrfgs_full_h_v1"]["metrics"], "status": "retained",
              "origin": PATHS.relative(old_path)}]
    payload = {"rows": rows, "query_ids_sha256": C.digest_ids(inputs.query_ids),
               "all_methods_complete": all(row["metrics"] is not None for row in rows),
               "selection_contract_sha256": C.sha256(R / "contract.json"),
               "retained_fold_index_sha256": C.sha256(R / "selected_inputs.json"),
               "no_test_phase_delay_amplitude_alignment": True,
               "V16_historical_training_coverage_difference_retained": True,
               "stochastic_training_changed_not_stepwise_numeric_equivalence": True}
    C.atomic_json(R / "evaluation/comparison.json", payload)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    header = ["方法", "PAS ↑", "PDP ↑", "NMSE（线性）↓", "综合分 ↑", "状态"]
    writer.writerow(header)
    lines = ["# 华为仿真500点最新成绩", "", "| " + " | ".join(header) + " |",
             "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        cells = [f'{row["metrics"][k]:.6f}' if row["metrics"] else "待完成" for k in METRICS]
        status = "500点已核验" if row["status"] in ("verified500", "retained_verified500") else (
            "已有成绩" if row["status"] == "retained" else f'已生成{row["prediction_count"]}/500点')
        item = [row["method"], *cells, status]
        writer.writerow(item)
        lines.append("| " + " | ".join(item) + " |")
    lines += ["",
              "NeRF²复用已封存的两组，后三组采用频点小批量；NeWRF复用已封存的四组，最后一组采用频点小批量。当前角色从完整检查点续训；尚未开始的refit仍按原规则fresh训练。",
              "频点抽样改变训练轨迹。原优化器步数、位置覆盖、验证选择和refit规则继续保留。各折全频及抽样训练量分别披露。",
              "输出仍为完整192频、256×4端口的原始复数H。同500点、同划分、同频序和同指标。31个零信道位置保留。没有使用测试CSI修正幅度、相位或时延。",
              "PAS/PDP为余弦相似度；NMSE为总平方误差/总真值能量。总分=0.4 PAS+0.4 PDP+0.2/(1+global NMSE)。",
              "V16、WRF-GS+及nGRF沿用已保存结果。V16历史训练覆盖差异继续披露；各方法不是原论文原样复现，也不声明计算预算相等。"]
    text = "\n".join(lines) + "\n"
    (R / "evaluation/comparison.csv").write_text("\ufeff" + buffer.getvalue())
    (R / "evaluation/RESULTS.md").write_text(text)
    (R / "RESULTS.md").write_text(text)
    return payload


def score_method(method, inputs):
    groups = seal(method, inputs)
    result = F.score_method(method, inputs)
    if result is None:
        raise RuntimeError("Expected a completed method")
    Check.verify_method(method, groups, inputs)
    print(json.dumps({"event": "method_scored_and_verified", "method": method,
                      "metrics": result["metrics"]}), flush=True)


def finalize(inputs):
    preserve = preservation()
    indices = {}
    for method in METHODS:
        indices[method] = {}
        for fold in range(5):
            r = selected_report(method, fold) / f"outer{fold}"
            indices[method][str(fold)] = {
                role: {"path": PATHS.relative(r / role / "result.json"),
                       "sha256": C.sha256(r / role / "result.json")}
                for role in ("screen", "refit")}
    C.atomic_json(R / "evaluation/training_report_index.json", indices)
    payload = table(inputs)
    C.atomic_json(R / "evaluation/completion.json", {
        "status": "scored_and_verified", "rows": payload["rows"],
        "old_file_protection": preserve, "completed_utc": F.now(),
        "all_query_counts": 500, "all_zero_truth_rows_retained": 31})
    marker = "- 834频点小批量续训的500点评分与独立核验完成"
    record = PATHS.root / "experiment_records/EXPERIMENT_RECORD.md"
    if marker not in record.read_text():
        values = "; ".join(
            row["method"] + " " + ", ".join(f'{k}={row["metrics"][k]:.9f}' for k in METRICS)
            for row in payload["rows"])
        with record.open("a") as output:
            output.write("\n" + marker + "：" + values +
                         "。原文件保护通过；训练与频点预算见freqminibatch_20261112的逐折来源及QA记录。\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check", "table", "watch"), default="watch")
    parser.add_argument("--wall-hours", type=float, default=72)
    args = parser.parse_args()
    (R / "evaluation").mkdir(parents=True, exist_ok=True)
    inputs = C.SimulationInputs()
    validate_selection(inputs)
    if args.mode == "check":
        print(json.dumps({"status": "passed", "protection": preservation()}), flush=True)
        return
    if args.mode == "table":
        table(inputs)
        return
    deadline = time.monotonic() + args.wall_hours * 3600
    with (R / "evaluation/coordinator.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        while True:
            counts = {method: prediction_count(method) for method in METHODS}
            if counts != previous:
                print(json.dumps({"event": "waiting_for_complete_methods", "utc": F.now(),
                                  "prediction_counts": counts}), flush=True)
                previous = counts
            for method, count in counts.items():
                if count == 500 and not (R / "evaluation" / method / "independent_verification.json").is_file():
                    score_method(method, inputs)
            current = table(inputs)
            if current["all_methods_complete"]:
                finalize(inputs)
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Registered completion window expired")
            time.sleep(30)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        (R / "evaluation").mkdir(parents=True, exist_ok=True)
        with (R / "evaluation/failure.log").open("a") as output:
            output.write(traceback.format_exc() + "\n")
        raise

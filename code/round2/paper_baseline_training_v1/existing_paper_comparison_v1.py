"""Build a qualified comparison from existing frozen results; no model or scorer execution."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project_paths import find_project_root

ROOT = find_project_root()
OLD = ROOT / "analysis/results/round2_paper_four_baselines_v1"
OUT = ROOT / "analysis/results/round2_paper_existing_results_audit_20261101"

def load(p):
    return json.loads(p.read_text())

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def new_text(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8", newline="") as f:
        f.write(value)

def main():
    audits = ["field_baselines_audit.json", "spectrum_baselines_audit.json",
              "metric_provenance_audit.json"]
    for name in audits:
        if not (OUT / name).is_file():
            raise RuntimeError("Required audit missing: " + name)
    old = load(OLD / "comparison.json")
    metric_audit = load(OUT / "metric_provenance_audit.json")
    if metric_audit["arithmetic_verification_status"] != "passed":
        raise RuntimeError("Saved-score arithmetic has not passed")
    labels = {
        "v16": "V16（已验证版本）",
        "nerf2": "NeRF²（任务适配版）",
        "newrf": "NeWRF（任务适配版）",
        "ngrf": "nGRF（任务适配版）",
        "wrfgs": "WRF-GS+（功率适配版）",
    }
    status = {
        "v16": "沿用历史汇总",
        "nerf2": "可沿用，有适配限制",
        "newrf": "可沿用，有适配限制",
        "ngrf": "可沿用，有适配限制",
        "wrfgs": "仅PAS；有限预算适配",
    }
    notes = {
        "v16": "用户验证的历史500点汇总；不重训。原逐点H/统计缺失，未重新恢复训练链；线上0.686754另列。",
        "nerf2": "位置输入、共享频率条件标量头，区别于原MIMO多频率输出头。最终checkpoint未被原预测manifest直接SHA绑定，其他来源/配置/状态链一致。",
        "newrf": "训练用FIT缓存射线；查询用64条FIT派生图像源/球面候选。是已声明的先验差异，未发现具体索引或尺度错误。",
        "ngrf": "保留高斯相干求和核心，加入频率条件及固定256×4复数输出头，使用适配损失；只反映保留阵列、频点与配置的结果。",
        "wrfgs": "本任务输出为DFT码本功率，不含可识别的复相位；仅PAS。每站1024高斯，原作者初始配置200000高斯且200000次训练；当前高斯规模与训练预算显著缩减。不得补造H、PDP、NMSE或综合分。",
    }
    rows = []
    for r in old["rows"]:
        key = r["key"]
        m = r["metrics"]
        assert m["count"] == 500
        if key != "wrfgs":
            actual = .4*m["pas"] + .4*m["pdp"] + .2/(1+m["nmse"])
            if not math.isclose(actual, m["score"], abs_tol=1e-12, rel_tol=0):
                raise RuntimeError("Total-score arithmetic: " + key)
        else:
            assert all(m[k] is None for k in ["pdp", "nmse", "nmse_score", "score"])
        rows.append({
            "method_key": key, "method": labels[key], "metrics": m,
            "reuse_status": status[key], "caveat": notes[key],
            "metrics_origin": "analysis/results/round2_paper_four_baselines_v1/comparison.json",
            "selected_artifact_root": r.get("selected_artifact_root"),
            "accepted_for_supported_metrics": True,
            "unsupported_metrics": ["pdp", "nmse", "score"] if key == "wrfgs" else [],
            "original_paper_exact_reproduction": False if key != "v16" else None,
            "new_model_execution": False, "new_formal_rescore": False,
        })
    before = load(OUT / "protected_before.json")
    after = {rel: digest(ROOT/rel) for rel in before}
    if before != after:
        raise RuntimeError("An old protected score or manifest changed")
    contract = {
        "schema": "retained-baseline-metric-contract-v1",
        "dataset": "Huawei Round2 simulation",
        "raw_channel_axes": ["position", "BS:2pol*16horizontal*8vertical", "UE:2pol*2spatial", "192ascendingF"],
        "channel_shape_per_position": [256, 4, 192],
        "frequency_Hz": "6690000000 + (k-96)*30000, k=0..191",
        "evaluation_positions": 500,
        "canonical_row_ids_sha256": metric_audit["query500_ids_sha256"],
        "fold_counts": metric_audit["fold_counts"],
        "zero_truth_rows": 31,
        "zero_rows_excluded": False,
        "metrics": {
            "pas": {"direction": "higher", "description": "BS-array power-spectrum cosine; mean over UE ports, frequencies and positions",
                    "derivation": "2D BS FFT separately in 16x8 H/V coordinates per polarization; concatenate BS polarization; cosine along256",
                    "origin_path": "code/common/official_metrics.py", "source_ref": "metric_provenance_audit.json"},
            "pdp": {"direction": "higher", "description": "Delay power-spectrum cosine; mean over BS/UE links and positions",
                    "derivation": "192-point IFFT over frequency then magnitude squared; cosine along192 delays",
                    "origin_path": "code/common/official_metrics.py", "source_ref": "metric_provenance_audit.json"},
            "nmse": {"direction": "lower", "description": "Global linear NMSE",
                     "derivation": "sum(abs(Hpred-Htrue)^2)/sum(abs(Htrue)^2), all500; not mean per-position or per-fold NMSE",
                     "origin_path": "code/common/official_metrics.py", "source_ref": "metric_provenance_audit.json"},
            "score": {"direction": "higher", "description": "Competition composite",
                      "derivation": "0.4*PAS+0.4*PDP+0.2/(1+global_NMSE)",
                      "origin_path": "code/round2/paper_baseline_training_v1/final_score.py", "source_ref": "metric_provenance_audit.json"},
        },
        "zero_norm_cosine": "zero denominator replaced by1e-12; numerator0 gives cosine0, including both-zero vectors",
        "test_target_phase_delay_amplitude_alignment": False,
        "scale_source_for_model_outputs": "FIT-derived scales only; output scoring uses archived raw H or power",
        "seed_scope": "one retained seed/configuration per method, with model/fold-specific derived RNG",
        "original_scores_preserved": True,
        "V16": {"basis": "user-validated archived fixed500 aggregate",
                "online_total_separate": .686754,
                "raw_H_and_point_stats_available": False,
                "retrained_or_same_chain_reconstructed": False},
        "numeric_note": "Same mathematical definitions, differing V16 historical vs current numerical paths. FP32 products have an extreme underflow counterexample. Two actual NeWRF weak samples showed no norm collapse and only small subnormal differences; no full rescore.",
        "no_fit_NMSE_threshold_required_for_reuse": True,
        "claim_scope": "These retained task adaptations/configurations; not original-paper exact reproduction or globally optimal literature results",
    }
    payload = {
        "schema": "paper-existing-results-comparison-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "route": "reuse_and_verify_existing_results",
        "rows": rows, "metric_contract": "json/metric_contract.json",
        "audit_reports_sha256": {name: digest(OUT/name) for name in audits},
        "input_identity": old["input_identity"],
        "historical_V16_online_total": .686754,
        "V16_retrained": False, "old_scores_overwritten": False,
        "new_training": False, "new_predictions": False, "new_formal_full_rescore": False,
        "small_numerical_diagnostic_rows": [1088, 3784],
        "formal_score_use": "Retained local500 values with explicit task-adaptation and provenance notes",
        "limitations": [
            "one retained seed/configuration, no seed variance or optimal-budget claim",
            "V16 original point predictions unavailable; no paired confidence interval or regenerated training chain",
            "WRF-GS+ missing H-derived metrics because its retained output is power-only",
            "NeRF2 original manifest lacks final checkpoint SHA binding",
            "Adaptation and optimization limitations remain disclosed; low score alone is not a correctness failure"],
    }
    cols = ["method", "pas", "pdp", "nmse", "score", "reuse_status", "caveat"]
    b = io.StringIO()
    w = csv.DictWriter(b, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(dict(method=r["method"], **{k:r["metrics"][k] for k in ["pas","pdp","nmse","score"]},
                        reuse_status=r["reuse_status"], caveat=r["caveat"]))
    def fmt(x):
        return "—" if x is None else f"{x:.6f}"
    md = ["# 华为仿真数据：已有500点结果对比", "",
          "四个基线的必要正确性检查已完成。本表沿用已保存的模型输出与原成绩，不训练、不调参、不重新生成预测。", "",
          "| 方法 | PAS ↑ | PDP ↑ | global NMSE ↓ | 综合分 ↑ | 核验状态 |",
          "|---|---:|---:|---:|---:|---|"]
    for r in rows:
        values = [fmt(r["metrics"][k]) for k in ["pas","pdp","nmse","score"]]
        if r["method_key"] == "v16":
            values = ["**"+v+"**" for v in values]
        md.append("| "+r["method"]+" | "+" | ".join(values)+" | "+r["reuse_status"]+" |")
    md += ["", "综合分 = 0.4 PAS + 0.4 PDP + 0.2/(1+global NMSE)。NMSE 为线性全局比值。500点全部保留，其中31点为零信道。五个空间留出折样本数为103、71、63、110、153。",
           "", "“—”表示该输出不支持此指标，不能当作0分，也不能用功率拼出复H补齐。",
           "", "V16沿用用户已验证的历史500点汇总。线上综合分0.686754单独保留；表中0.686375是本地汇总。其原逐点H/统计文件缺失，因此没有重新恢复训练链或报告配对显著性。历史与当前评分使用相同数学定义，不声称数值执行路径逐项相同。",
           "", "四个文献方法均为本任务适配实现。表中的低分作为保留配置的实际结果报告；不以低分或任意拟合阈值排除，也不据此断言原文献方法无效或已经得到其最佳性能。",
           "", "适配及仍需说明的限制：", ""]
    md += ["- "+r["method"]+"："+r["caveat"] for r in rows if r["method_key"] != "v16"]
    md += ["", "已核对预测覆盖、文件哈希、模型/源码/配置关系、频点和端口、FIT尺度、划分隔离、真实优化器更新以及现有分数聚合。未发现足以判定这些适配结果算错的具体工程错误。NeRF²最终checkpoint的原始SHA绑定缺失，按溯源限制保留，不补造历史证明。",
           "", "评分浮点抽查：FP32功率乘积在合成极小值下确有下溢反例。对NeWRF两个既有弱样本的检查未发现范数归零；更高精度计算的单点PDP变化约-2.42e-6与-3.97e-7。此有限检查不能量化全部样本的数值差异，也没有证据支持把总体低分归因于该问题。本轮沿用原成绩，不全表重评分。",
           "", "这组保留结果中，V16的PAS、PDP和综合分最高，global NMSE最低。该描述只针对表中实现和配置。训练耗时、硬件预算与调参投入未统一，不能据本表声称计算效率优势。",
           "", "复核入口：field_baselines_audit.json、spectrum_baselines_audit.json、metric_provenance_audit.json，以及json/metric_contract.json。旧成绩与旧表均保留。"]
    tex = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & PAS$\uparrow$ & PDP$\uparrow$ & NMSE$\downarrow$ & Score$\uparrow$ \\",
        r"\midrule"]
    tex_labels = {"v16": r"Ours (V16)$^{\dagger}$", "nerf2": r"NeRF$^2$ (adapted)",
                  "newrf": "NeWRF (adapted)", "ngrf": "nGRF (adapted)",
                  "wrfgs": r"WRF-GS+ (power)$^{\ddagger}$"}
    for r in rows:
        v = ["--" if r["metrics"][k] is None else f"{r['metrics'][k]:.6f}"
             for k in ["pas","pdp","nmse","score"]]
        if r["method_key"] == "v16":
            v = [r"\textbf{"+x+"}" for x in v]
        tex.append(tex_labels[r["method_key"]] + " & " + " & ".join(v) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}",
            "% Requires booktabs. All literature rows are task adaptations.",
            "% dagger: retained user-validated historical fixed500 aggregate; no retraining.",
            "% ddagger: power-only output supports PAS, not H/PDP/NMSE/composite.",
            "% All500 positions including31 zero-channel rows. Linear global NMSE.",
            "% Score=.4 PAS+.4 PDP+.2/(1+global NMSE). Single retained seed/config.",
            "% No claim of exact original-paper reproduction, best baseline tuning, or matched training cost."]
    new_text(OUT/"json/metric_contract.json", json.dumps(contract, ensure_ascii=False, indent=2)+"\n")
    new_text(OUT/"paper_comparison.json", json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    new_text(OUT/"paper_comparison.csv", b.getvalue())
    new_text(OUT/"paper_comparison.md", "\n".join(md)+"\n")
    new_text(OUT/"paper_comparison.tex", "\n".join(tex)+"\n")
    verification = {
        "status": "completed_reuse_with_explicit_caveats",
        "protected_originals_sha256_after": after, "protected_originals_unchanged": True,
        "tables_exactly_copy_original_metric_values": True,
        "only_null_metrics": {"wrfgs": ["pdp","nmse","score"]},
        "audit_sha256": payload["audit_reports_sha256"],
        "builder_sha256": digest(Path(__file__)),
        "no_training_or_inference": True, "no_formal_full_rescore": True,
    }
    new_text(OUT/"verification_summary.json",json.dumps(verification,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"rows":len(rows), "protected_originals_unchanged":True,
                      "status":verification["status"]}, ensure_ascii=False))

if __name__ == "__main__":
    main()

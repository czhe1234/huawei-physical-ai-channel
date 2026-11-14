"""Assemble verified development-fitting evidence; never imports a model or scores a test set."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project_paths import find_project_root

ROOT = find_project_root()
BASE = ROOT / "analysis/results/round2_paper_four_baselines_v1"
OUT = BASE / "fit_repair_paper_v2_20261101"

def read(rel):
    return json.loads((BASE / rel).read_text())

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def preservation():
    before = read("fit_repair_dev_20261101/frozen_sources_and_models_before.json")
    source = {rel: sha(ROOT / rel) == expected for rel, expected in before["source_sha256"].items()}
    old_stat = []
    for expected in before["old_model_stat"]:
        st = (ROOT / expected["path"]).stat()
        old_stat.append({"path": expected["path"], "unchanged": all(
            actual == expected[key] for key, actual in
            [("size", st.st_size), ("mtime_ns", st.st_mtime_ns), ("inode", st.st_ino)])})
    protected = read("fit_repair_dev_20261101/protected_before.json")
    hashes = {rel: sha(ROOT / rel) == expected for rel, expected in protected.items()}
    result = {"source_sha256": source, "old_model_metadata": old_stat,
              "old_report_and_manifest_sha256": hashes,
              "old_model_check_is_metadata_not_full_content_hash": True}
    result["pass"] = all(source.values()) and all(x["unchanged"] for x in old_stat) and all(hashes.values())
    if not result["pass"]:
        raise RuntimeError("An old protected artifact changed")
    return result

def main():
    stages = []
    rel = "fit_repair_paper_v7_low_lr_ue01_20261101/nerf2/result.json"
    d = read(rel)
    s = d["final_stats"]
    stages.append(dict(method="NeRF2", stage="single_position_two_UE", positions=1,
        BS=128, UE=2, F=192, nmse=s["pooled"]["nmse"],
        energy_ratio=s["pooled"]["energy_ratio"],
        gate_pass=d["complete_field_recovered"], report=rel,
        scope="site0 BS-pol0 UE-pol0; one of eight independent fields",
        per_UE=s["per_UE"], updates_in_stage=d["v7_new_updates"]))

    rel = "fit_repair_paper_v8_two_position_field_20261101/nerf2/result.json"
    if (BASE / rel).exists():
        d = read(rel)
        s = d["final_stats"]
        stages.append(dict(method="NeRF2", stage="two_position_two_UE", positions=2,
            BS=128, UE=2, F=192, nmse=s["pooled"]["nmse"],
            energy_ratio=s["pooled"]["energy_ratio"],
            gate_pass=bool(d.get("two_position_field_fit", d.get("two_position_field_recovered", d.get("complete_field_recovered", False)))),
            report=rel, scope="site0 BS-pol0 UE-pol0; one of eight independent fields",
            per_position=s.get("per_position", [])))
    rel = "fit_repair_paper_v10_coarse_init_signal_warm_20261101/newrf/result.json"
    d = read(rel)
    s = d["final_fine"]
    stages.append(dict(method="NeWRF", stage="two_position_one_UE", positions=2,
        BS=128, UE=1, F=192, nmse=s["nmse"], energy_ratio=s["energy_ratio"],
        gate_pass=d["status"] == "full_two_position_field_pass", report=rel,
        scope="site0 field0 UE0; not the complete 256x4x192 channel",
        per_position=s["per_position"], updates_in_stage=d["total_updates"]))

    independent = {}
    for name in [
        "independent_progress_verification.json",
        "independent_nerf2_v5_verification.json",
        "independent_nerf2_v7_verification.json",
        "independent_nerf2_v8_pretraining_verification.json",
        "independent_nerf2_v8_final_verification.json",
        "independent_newrf_v11_source_verification.json",
        "independent_newrf_v11_verification.json",
        "independent_newrf_v11b_verification.json",
        "independent_newrf_v11_cpu_verification.json",
        "independent_newrf_sampling_source_review.json",
        "independent_newrf_2x2_verification.json"]:
        p = OUT / name
        if p.exists():
            independent[name] = {"sha256": sha(p), "report": str(p.relative_to(BASE))}
    query_rel = "fit_repair_paper_v11_query_route_20261101/newrf/attempt_cpu/CPU_execution_addendum.json"
    query_check = read(query_rel)
    preserve = preservation()
    result = dict(
        purpose="development fitting and adapter correctness, not paper performance ranking",
        paper_main_table_ready=False,
        dataset="Huawei Round2_Train_Channel.npy; existing outer0 screen FIT only",
        raw_H_shape=[4000, 256, 4, 192],
        parent_fit_rms=2.5475131620037768e-5,
        unit_rows=[2666, 2042], raw_row_selection_precedes_unit_predictions=True,
        stages=stages, independent_reports=independent, preservation=preserve,
        query_interface_check={"report": query_rel, "CPU_FIT_reference_relative_L2": query_check["relative_L2"],
            "tolerance": 1e-5, "query_forward_executed": False,
            "CPU_reference_gate_pass": False, "GPU_resources_unavailable": True},
        not_completed=["repaired versions on the complete original 16FIT panel and all 8 fields",
                       "fresh training on the formally defined complete FIT splits after the health gate",
                       "independent spatial validation with complete H",
                       "formal unified baseline comparison"],
        formal_test_opened=False, north_confirmation_opened=False,
        V16_retrained=False, historical_scores_replaced=False,
        no_test_target_phase_or_gain_alignment=True,
        uncertainty="adaptations, not exact reproductions of the original papers; ports and frequencies follow the inherited competition contract",
        statistical_unit="training position; BS, UE and frequencies are correlated tensor coordinates",
        confidence_interval=None)
    (OUT / "fit_health_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    fields = ["method", "stage", "positions", "BS", "UE", "F", "nmse", "energy_ratio", "gate_pass", "report"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(stages)
    (OUT / "fit_health_summary.csv").write_text(buf.getvalue())
    lines = [
        "# NeRF² 与 NeWRF 适配修复：训练拟合检查",
        "",
        "本报告记录小规模训练拟合与正确性检查。当前结果不能作为论文主表中的基线成绩。",
        "所有数字来自已保存的训练位置预测。未打开正式测试；未重训 V16；旧成绩保留。",
        "",
        "| 方法与范围 | 位置数 | 阵列位置×UE端口×频点 | NMSE | 预测能量/真值能量 | 预定拟合门槛 |",
        "|---|---:|---|---:|---:|---|"]
    for r in stages:
        label = {"single_position_two_UE": "单位置、两个UE端口",
                 "two_position_two_UE": "两位置、两个UE端口",
                 "two_position_one_UE": "两位置、单UE端口"}[r["stage"]]
        lines.append(f"| {r['method']}：{label} | {r['positions']} | {r['BS']}×{r['UE']}×{r['F']} | {r['nmse']:.6f} | {r['energy_ratio']:.6f} | {'通过' if r['gate_pass'] else '未通过'} |")
    lines += [
        "",
        "NeRF²两位置训练未通过：原位置2666的NMSE为0.08448、能量比0.7924；新增位置2042的NMSE为1.24925、能量比0.2580。整体NMSE为0.2334，不能掩盖新位置未拟合。这里是训练拟合失败，不是独立位置的泛化成绩。",
        "",
        "表中方法的端口范围不同。这是各自的拟合验收，不能用它给两种方法排性能名次。",
        "这里的128个阵列位置属于基站阵列。完整任务仍是每个位置256×4×192的复信道。",
        "",
        "NeRF²：相同500步下，共享单频率输出与192频率输出头的NMSE分别约为0.894与0.640。192频率输出头继续训练到1000步后，单链路NMSE为0.00134；不能把这个额外训练终点当作同预算比较。随后扩展到128个基站阵列位置与两个UE端口。位置输入、192频率输出和实际频点物理相位属于本项目的适配，不是原论文的逐项复现。",
        "",
        "NeWRF：细网络最初通过压低密度而把输出压到接近零。固定密度对照只用于定位问题。最终可运行方案不保存固定密度或固定采样点：先拟合粗网络，将粗网络参数复制给细网络；预热时只让信号路径更新共享特征；再恢复完整梯度，以较低学习率训练。最后用原生构建器重新加载，输出逐值一致。",
        "",
        "失败对照、初始状态、每批标签、随机状态和优化器状态均保留。NeWRF四格对照的提前停止步数不同；额外的完整梯度训练结果单列，不能当成同预算胜负。独立核验记录见同目录JSON文件。",
        "",
        "NeWRF训练采样与完整拟合检查未发现随机采样或train/eval分支不一致。NeWRF训练两点分别用29/17条FIT几何方向，查询接口用64条FIT派生候选方向。该设计差异需要固定模型双接口检查。三张允许GPU当时均被其他作业占用；CPU替代执行与已有GPU FIT预测的相对L2差为0.000361234，大于预注册的0.00001。已按协议停止，query接口未调用、没有query成绩。CPU/GPU与batch形状的影响尚未分离。实际CPU身份与旧comparison文字标签的差异已另写更正，不改已执行源。不能通过目标信道修正射线、幅度、相位或时延。",
        "",
        "后续NeRF²的8个子网络共享log_output_gain，须采用联合均衡调度并保护统一优化器状态，不能直接复制单field充分训练后依次串行。当前192频率输出头renderer只验证了一个field，尚不是完整256×4×192输出的适配验收。",
        "",
        "原定16点中的14个非零点跨点能量跨度约82–89 dB；其余两点全零。NeWRF实际损失使用batch联合目标能量分母，fine/coarse权重为0.9/0.1，没有power floor。目标能量份额不等于真实梯度份额。后续需要分别检查强、中、弱、零信号位置，不能用均衡抽样代替逐位置验收。两位置单元都在同一site，不能外推到另一个site。2666不在原16点内，严格16点重建应从该面板内的fresh初始化开始。",
        "",
        "本轮能支持：部分关键子网络可在保留宽带复数标签、端口与相位口径的条件下拟合训练信道。不能支持：完整H的空间泛化、论文中的方法排名，或原论文方法在本数据上无效。",
        "",
        "论文主表仍缺：先完成原定16个位置与全部8个子网络的健康检查；再按正式完整FIT划分训练，完成独立空间验证；最后冻结方案并做正式比较。16点只是健康面板，不是正式完整训练集。原先完整16位置的失败运行保留，不能用局部修复结果替换。",
        "",
        f"保留核验：{len(preserve['source_sha256'])}份旧源码SHA一致；{len(preserve['old_model_metadata'])}个旧模型的大小、修改时间和inode一致；{len(preserve['old_report_and_manifest_sha256'])}份旧成绩/清单SHA一致。旧模型检查属于文件元数据检查，不冒充逐字节内容哈希。",
        "",
        "主要证据入口：",
        ""]
    lines += [f"- {r['report']}" for r in stages]
    lines += [f"- fit_repair_paper_v2_20261101/{name}" for name in independent]
    lines += [f"- {query_rel}",
        "- fit_repair_paper_v10_coarse_init_signal_warm_20261101/newrf/factorial_2x2_comparison.md",
        "- fit_repair_paper_v10_coarse_init_signal_warm_20261101/newrf/panel16_energy_loss_audit.md",
        "- fit_repair_paper_v10_coarse_init_signal_warm_20261101/newrf/panel16_fullH_schedule_proposal.md"]
    (OUT / "fit_health_report_zh.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"stages": [{k:r[k] for k in fields if k in r} for r in stages],
                      "preservation_pass": preserve["pass"]}, ensure_ascii=False))

if __name__ == "__main__":
    main()

"""Compile experiment 832 development outcomes; no training or channel scoring."""
from __future__ import annotations
import csv, hashlib, io, json
from pathlib import Path
from project_paths import PATHS
R=PATHS.analysis/"results/round2_paper_four_baselines_v1/repair_20261111"
def read(rel):
    return json.loads((R/rel).read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for a in iter(lambda:f.read(4*1024*1024),b""):h.update(a)
    return h.hexdigest()
def slim_field(s):
    return {"pooled":s["pooled"],"per_position":[
        {"row_id":v["row_id"],"pooled":v["pooled"],
         "per_UE":[{k:x for k,x in u.items() if not k.startswith("BS_")} for u in v.get("per_UE",[])]}
         for v in s["per_position"]]}
def slim_ngrf(v):
    s=v.get("FIT",v.get("final"))
    return {"branch":v.get("branch",v.get("weight")),"pooled":s["pooled"],
            "per_point":s["per_point"],"health":v.get("health"),
            "wall_seconds":v.get("wall_seconds"),"peak_cuda_bytes":v.get("peak_cuda_bytes"),
            "checkpoint":v.get("checkpoint"),"checkpoint_sha256":v.get("checkpoint_sha256"),
            "raw_prediction":v.get("raw_prediction"),"raw_prediction_sha256":v.get("raw_prediction_sha256"),
            "reuse_status":v.get("reuse_status")}
def main():
    n2=read("nerf2/stability_confirmation/result.json")
    n2pair=read("nerf2/position_loss_v2/result.json")
    nw=read("newrf/native_finish/result.json")
    ng=read("ngrf/adam_copy_fix_v2/position_loss/results.json")
    ngval=read("ngrf/adam_copy_fix_v2/position_loss/fixed_endpoint_validation/results.json")
    assert all(d["status"]=="complete" for d in (n2,n2pair,ng,ngval))
    pair=[slim_ngrf(v) for v in read("ngrf/paired/results.json")["variants"]]
    drop=[slim_ngrf(v) for v in read("ngrf/adam_copy_fix_v2/dropout/results.json")["variants"]]
    pos=[slim_ngrf(v) for v in ng["variants"]]
    candidate=next(v for v in pos if v["branch"]=="loss_position_relative")
    audits={}
    for rel in ["newrf/independent_verification.json","independent_nerf2_v2_check.json",
                "independent_nerf2_stability_check.json","independent_ngrf_final_check_v2.json",
                "final_preservation_and_source_check.json"]:
        p=R/rel
        assert p.is_file(),rel
        d=read(rel);audits[rel]={"sha256":sha(p),"status":d.get("status"),"strict_valid":d.get("strict_valid")}
        assert not str(d.get("status","")).startswith("failed"),rel
    owners={}
    for name,rel in [("NeRF2","nerf2/summary.json"),("nGRF","ngrf/adam_copy_fix_v2/summary.json")]:
        if (R/rel).is_file():owners[name]=read(rel)
    summary={
        "experiment":832,"role":"development repair and fitting verification",
        "formal500_scored":False,"V16_retrained":False,"old_results_preserved":True,
        "metric_definition":"FIT pooled NMSE=sum(abs(Hhat-H)^2)/sum(abs(H)^2), no truth-based alignment",
        "not_a_method_ranking":"Scopes differ. FIT pooled NMSE is not the canonical500 paper comparison.",
        "NeRF2":{"scope":"2 FIT rows 2666,2042; site0 BS-pol0 UE-pol0; 128 BS x 2 UE x 192 frequencies; 1 of 8 fields",
                 "fixed2000":{"control":slim_field(n2pair["control"]["full_FIT_stats"]),
                              "candidate":slim_field(n2pair["candidate"]["full_FIT_stats"]),
                              "candidate_stable_checks":n2pair["candidate"]["stable_checks"]},
                 "final":slim_field(n2["full_FIT_stats"]),"fit_passed":n2["fit_passed"],
                 "total_optimizer_updates":n2["total_optimizer_updates"],
                 "additional_stability_updates":n2["additional_optimizer_updates"],
                 "total_stable_checks":n2["total_consecutive_stable_checks"],
                 "result":"nerf2/stability_confirmation/result.json",
                 "metadata_errata":"nerf2/stability_confirmation/metadata_errata.json"},
        "NeWRF":{"scope":"2 FIT rows 2666,2042; site0 field0; 128 BS x 1 UE x 192 frequencies; 1 of 8 fields",
                 "final":nw["actual_query_metrics"],"fit_passed":True,
                 "query_train_relative_L2":nw["training_vs_actual_query"]["relative_L2"],
                 "clean_reload":nw["clean_reload"],"total_optimizer_updates":nw["combined_optimizer_updates"],
                 "budget_accounted_seconds":nw["combined_active_wall_budget_accounted_seconds"],
                 "exact_total_wall_seconds":nw["combined_active_wall_seconds"],
                 "recorded_peak_cuda_bytes":nw["peak_cuda_bytes"],
                 "recorded_peak_not_complete_timedout_stage":True,
                 "result":"newrf/native_finish/result.json"},
        "nGRF":{"scope":"16 FIT rows incl 1 zero; both sites; full 256 BS x 4 UE x 192 frequencies",
                "phase_weight_pair":pair,"corrected_dropout_pair":drop,"corrected_position_pair":pos,
                "candidate":candidate,"successful_path_updates":8000,
                "fixed16_development_validation":{"row_ids":ngval["row_ids"],
                    "variants":[{"branch":v["branch"],"metrics":v["metrics"]} for v in ngval["variants"]],
                    "endpoint_selection_from_validation":False},
                "result":"ngrf/adam_copy_fix_v2/position_loss/results.json"},
        "known_correction":"Old v1 paired continuation shared CPU Adam step state. Actual gradient-call count was unchanged; bias-correction starting state differed. Invalid affected arms were rerun with deepcopy and full immutable-state fingerprints.",
        "owner_summaries":owners,"independent_audits":audits}
    (R/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    n2s=n2["full_FIT_stats"]
    n2p=n2s["per_position"];nwp=nw["actual_query_metrics"]["per_position"]
    ngp=[p for p in candidate["per_point"] if not p["zero_truth"]]
    zeros=[p for p in candidate["per_point"] if p["zero_truth"]]
    rows=[
        ["NeRF2","2 FIT, 1/8 fields,128x2x192",2,n2s["pooled"]["nmse"],max(p["pooled"]["nmse"] for p in n2p),
         min(p["pooled"]["energy_ratio"] for p in n2p),max(p["pooled"]["energy_ratio"] for p in n2p),n2["fit_passed"],False],
        ["NeWRF","2 FIT,1/8 fields,128x1x192",2,nw["actual_query_metrics"]["nmse"],max(p["nmse"] for p in nwp),
         min(p["energy_ratio"] for p in nwp),max(p["energy_ratio"] for p in nwp),True,False],
        ["nGRF","16 FIT,full H,256x4x192",16,candidate["pooled"]["complex_nmse"],max(p["complex_nmse"] for p in ngp),
         min(p["prediction_to_truth_energy"] for p in ngp),max(p["prediction_to_truth_energy"] for p in ngp),
         candidate["health"]["all_point_health"],False]]
    with (R/"development_fit_summary.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["method","scope","FIT_positions","pooled_FIT_NMSE","worst_nonzero_position_NMSE",
            "min_position_energy_ratio","max_position_energy_ratio","small_fit_gate_passed","canonical500_scored"])
        w.writerows(rows)
    table="\n".join("| "+" | ".join([str(x) if not isinstance(x,float) else f"{x:.6f}" for x in r])+" |" for r in rows)
    valtable = "\n".join("| "+v["branch"]+" | "+" | ".join(f"{v['metrics'][k]:.6f}" for k in ("pas","pdp","nmse","score"))+" |" for v in ngval["variants"])
    # Schema-safe output: official metric dictionaries are copied exactly above.
    report=f"""# 三基线修复与小规模验收（实验832）

本轮完成代码修复、有限训练及独立核验。V16不重训。旧模型、旧成绩、原始数据保留。本轮没有运行正式500点测试。

## 验收结果

下表检查各方法是否能拟合其小规模训练数据。覆盖范围不同，不能用它作方法排名。NMSE为线性误差比，越低越好；这里的pooled FIT NMSE也不是论文500点的正式成绩。

| 方法 | 覆盖范围 | 训练位置数 | pooled FIT NMSE | 最差非零位置NMSE | 最小位置能量比 | 最大位置能量比 | 小规模通过 | 已跑500点 |
|---|---|---:|---:|---:|---:|---:|---|---|
{table}

能量比为预测总能量/真值总能量。nGRF的零信道另验泄漏，没有删除该样本。其最终零点泄漏为{zeros[0]["prediction_mean_power_FIT_RMS_squared"]:.9g} FIT-RMS²。

## 实际修复

这些是本项目面向宽带复数H任务的适配版本。输入输出一致不等于原论文逐项复现。论文需披露此次更改和训练预算。

**NeRF²**：恢复两位置从头联合更新，保留本项目适配版的192输出头、324条射线、32采样点及传播设置。补全低学习率阶段。固定2000步的匹配对照只改变损失归一化，同时改变整体尺度和逐位置权重。该对照检验归一化方案的整体作用，未单独隔离绝对损失尺度的影响。候选在两位置都达到原数值门槛；控制的弱位置仍未通过。再从固定候选端点继续500步，完成第3次连续通过。额外500步单独记录，未混入等预算对照。成功路径累计{n2["total_optimizer_updates"]}次联合更新。仅训练1/8 field，尚未验收全部端口与全场景。

**NeWRF**：主要问题是训练方向输入与实际查询方向输入不一致。同模型两点的训练路线NMSE为0.007744，换成实际查询路线后为1.686031。修复为预测时可得的统一查询方向，并进行分阶段训练。最终实际查询NMSE为{nw["actual_query_metrics"]["nmse"]:.9f}。训练提供器与实际查询输出完全一致；重新加载模型也一致。成功路径累计8500次更新。仅两位置、一个field，不能推广成全部场景已恢复。

**nGRF**：固定相同初始化、随机状态、采样与3000步预算，将复数误差权重0.1改为1，FIT NMSE由0.499507降至0.030206。之后核查梯度、dropout及空间权重，没有把失败简单归因为没有梯度或几何覆盖不足。有限对照采用dropout关闭与逐位置相对误差目标。最终15个非零训练位置逐点核验，零点另验泄漏。最终15/15个非零点通过原门槛，最差位置NMSE为{max(p["complex_nmse"] for p in ngp):.6f}，位置能量比为{min(p["prediction_to_truth_energy"] for p in ngp):.6f}–{max(p["prediction_to_truth_energy"] for p in ngp):.6f}。训练分母只来自允许FIT标签，不用验证或测试信道修正预测。最终成功路径8000次更新。固定2000步对照的global pooled NMSE为{pos[0]["pooled"]["complex_nmse"]:.9f}，逐位置候选为{candidate["pooled"]["complex_nmse"]:.9f}；候选的总体能量加权误差较高，但将控制仍失败的3个位置恢复。这是目标权衡，两个结果都保留。

## 保留的开发验证结论

nGRF在训练端点和方案固定后，使用原16个开发验证位置检查；没有用这些结果挑端点或回流调参。候选验证NMSE仍为{next(v["metrics"]["nmse"] for v in ngval["variants"] if v["branch"]=="loss_position_relative"):.9f}，没有空间泛化恢复证据。准确指标见summary.json和其原始验证JSON。其空间泛化结论需按下表判断，训练拟合通过不能代替它。

| 固定训练端点 | 开发PAS | 开发PDP | 开发NMSE | 开发综合分 |
|---|---:|---:|---:|---:|
{valtable}

## 独立核验与纠错

核验保存的原始复数预测、逐位置与接收通道误差、输出能量、模型重新加载、输入路线、父模型哈希及实际优化器状态。未做依赖测试真值的相位、时延或幅度校正。

审计发现，旧v1续训对照在加载Adam时共享了CPU step张量。第二支的内部偏差校正计数起点因此错误。实际梯度更新调用次数没有凭空增加。保留全部v1文件，并在新v2中深复制完整状态，核验step、一阶矩、二阶矩、随机状态及训练采样。仅重跑受影响支路和后续依赖。最初nGRF权重0.1/1对照和NeWRF单线恢复不受影响。nGRF复用的有效首支dropout-on缺当时内存中一、二阶矩的瞬时快照；其起点按磁盘父状态、执行顺序及保存终点核查。该限制保留在独立审核中。

NeRF²稳定性配置有两处沿用文本：其实际父端点为12717次，新增500次中的QA20也计入这500次。独立metadata_errata.json给出执行证据；冻结配置保留。

旧文件保护核验见final_preservation_and_source_check.json：77个文件用SHA256核验，256个旧模型及3份原始数组核验size/mtime/inode，主实验记录旧前缀SHA256保持。后两项为stat核验，不声称逐字节哈希。新代码另做AST与仓库路径规则检查。

## 目前可以说明什么

NeRF²和NeWRF通过各自两位置、单field的小范围验收。nGRF的完整H结论按逐点健康检查判定。这些训练拟合结果不能说明未见位置上的PAS、PDP、NMSE已经改善，也不能更新500点方法排名。NeRF²与NeWRF仍需扩展到全部field和既定训练覆盖；nGRF仍需解决开发位置的空间泛化。原500点表继续作为旧版本结果保留。

资源与所有分支（含作废尝试）的记录保留在各方法summary和原始日志中。NeWRF超时子段缺完整实测wall/峰值；采用保守墙钟记账1882.187秒，没有补造缺失值。修复分支预算按PLAN.md中的逐项有限登记执行，不宣称方法间算力相等。

## 文件

- 机器可读汇总：summary.json。
- 可导入表格：development_fit_summary.csv。
- 各阶段配置：config.json及各子目录独立config.json。
- 独立核验：independent_nerf2_v2_check.json、independent_nerf2_stability_check.json、independent_ngrf_final_check_v2.json、newrf/independent_verification.json。
- 源码：code/round2/baseline_repair_20261111（服务器项目相对路径）。
- 模型和原始预测：artifacts/round2/paper_four_baselines_v1/repair_20261111。
"""
    (R/"RESULTS.md").write_text(report)
    print(json.dumps({"written":["summary.json","development_fit_summary.csv","RESULTS.md"],
        "small_fit":[[r[0],r[-2]] for r in rows],"validation":summary["nGRF"]["fixed16_development_validation"]},ensure_ascii=False))
if __name__=="__main__":main()


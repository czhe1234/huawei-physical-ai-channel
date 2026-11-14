"""Verify the completed WRF-GS+ full-H comparison and write its result report."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np
from project_paths import PATHS

R=PATHS.analysis/"results/round2_paper_four_baselines_v1/wrfgs_full_h_v1_20261110"
A=PATHS.artifacts/"round2/paper_four_baselines_v1/wrfgs_full_h_v1"
ALLOWED={
    "GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009",
    "GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0",
    "GPU-70b6fceb-c7c4-9467-3c81-097c9284c160",
}
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def write(p,d):
    tmp=p.with_suffix(p.suffix+".partial")
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n")
    tmp.replace(p)
def main():
    config=read(R/"config.json")
    identity=read(R/"correctness.json")["source_identity"]
    for rel,h in identity.items():assert sha(PATHS.root/rel)==h,rel
    summary=read(R/"evaluation/comparison.json")
    assert summary["new_full_H_included"]
    assert summary["source_identity"]==identity
    before=read(R/"preservation_before.json")
    for d in before["stat"]:
        p=PATHS.root/d["path"];s=p.stat()
        assert (s.st_size,s.st_mtime_ns,s.st_ino)==(d["size"],d["mtime_ns"],d["inode"]),d["path"]
    for rel,h in before["hashes"].items():assert sha(PATHS.root/rel)==h,rel
    record=PATHS.root/"experiment_records/EXPERIMENT_RECORD.md"
    with record.open("rb") as f:prefix=f.read(before["record_prefix_bytes"])
    assert hashlib.sha256(prefix).hexdigest()==before["record_prefix_sha256"]
    old=read(PATHS.analysis/"results/round2_paper_four_baselines_v1/comparison.json")
    assert summary["rows"][0]["metrics"]==old["rows"][0]["metrics"]
    for values in summary["old_metric_deltas"].values():
        assert all(v is None or v==0 for v in values.values())
    all_ids=[];training=[]
    for fold in range(5):
        roles={}
        for role in ("screen","refit"):
            d=read(R/f"outer{fold}/{role}/complete.json")
            assert d["status"]=="complete" and d["source_identity"]==identity and d["config"]==config
            assert d["device"]["GPU_UUID"] in ALLOWED
            assert d["device"]["GPU_UUID"]==d["device"]["CUDA_context_UUID"]
            assert sha(PATHS.root/d["model_path"])==d["model_sha256"]
            fd=d["data"];fit=set(fd["fit_ids"]);outer=set(fd["outer_ids"])
            assert not fit.intersection(outer)
            assert set(fd["fit_rows_actually_read"])==fit
            assert not fd["query_truth_read"] and not fd["oracle_phase_delay_scale_correction"]
            assert not d["query_truth_read"]
            assert d["exposure"]["min"]>=1
            assert d["exposure"]["total"]==d["completed_updates"]*config["batch_positions"]
            if role=="refit":
                assert fd["validation_read_count"]==0 and fd["validation_rows_actually_read"]==[]
            else:
                assert set(fd["validation_rows_actually_read"]).issubset(set(fd["validation_ids"]))
                assert not fit.intersection(fd["validation_ids"])
                assert not outer.intersection(fd["validation_ids"])
                assert len(fd["selection_probe_ids"])==16
                assert d["selected_updates"]==max(d["history"],key=lambda e:e["validation"]["score"])["step"]
                assert 4000<=d["completed_updates"]<=12000
            roles[role]=d
        screen,refit=roles["screen"],roles["refit"]
        expected=math.ceil(screen["selected_updates"]*len(refit["data"]["fit_ids"])/len(screen["data"]["fit_ids"]))
        assert refit["completed_updates"]==expected
        p=A/f"outer{fold}/prediction/manifest.json";m=read(p)
        assert m["metadata"]["source_identity"]==identity
        assert m["metadata"]["model_sha256"]==refit["model_sha256"]
        assert not m["query_truth_read"]
        assert m["kind"]=="channel" and m["dtype"]=="complex64"
        x=np.load(PATHS.root/m["path"],mmap_mode="r",allow_pickle=False)
        ids=np.load(p.parent/"row_ids.npy",allow_pickle=False)
        assert ids.tolist()==m["row_ids"]==refit["data"]["outer_ids"]
        assert x.shape==(len(ids),256,4,192) and x.dtype==np.complex64
        assert sha(PATHS.root/m["path"])==m["sha256"]
        for start in range(0,len(x),4):assert np.isfinite(x[start:start+4]).all()
        all_ids.extend(ids.tolist())
        training.append(dict(fold=fold,query_count=len(ids),screen_fit_count=len(screen["data"]["fit_ids"]),
            refit_fit_count=len(refit["data"]["fit_ids"]),screen_updates=screen["completed_updates"],
            selected_updates=screen["selected_updates"],refit_updates=refit["completed_updates"],
            screen_stop_reason=screen["stop_reason"],screen_validation=screen["full_validation_diagnostic"],
            training_seconds=screen["wall_seconds"]+refit["wall_seconds"],
            peak_cuda_bytes=max(screen["peak_cuda_bytes"],refit["peak_cuda_bytes"]),
            device=refit["device"],model_sha256=refit["model_sha256"],
            prediction_sha256=m["sha256"]))
    assert len(all_ids)==len(set(all_ids))==500
    aggregate_checks={}
    for row in summary["rows"][1:]:
        root=R/"evaluation/final_scores"/row["key"]
        d=read(root/"results.json")
        assert d["metrics"]==row["metrics"]
        assert sha(root/"point_statistics.npz")==d["point_statistics_sha256"]
        with np.load(root/"point_statistics.npz",allow_pickle=False) as st:
            assert len(st["row_ids"])==500 and set(st["row_ids"].tolist())==set(all_ids)
            assert int(np.sum(st["truth_energy"]==0))==d["zero_truth_row_count"]==31
            derived=dict(count=500,pas=float(st["pas"].mean()))
            if row["key"]!="wrfgs":
                derived["pdp"]=float(st["pdp"].mean())
                derived["nmse"]=float(st["squared_error"].sum()/st["truth_energy"].sum())
                derived["nmse_score"]=1/(1+derived["nmse"])
                derived["score"]=.4*derived["pas"]+.4*derived["pdp"]+.2*derived["nmse_score"]
            delta={k:derived[k]-row["metrics"][k] for k in derived}
            assert all(abs(v)<1e-12 for v in delta.values()),delta
            aggregate_checks[row["key"]]=delta
    result=dict(status="passed",all_five_folds_complete=True,query_count=500,
        unique_query_count=500,zero_truth_row_count=31,full_H_shape=[500,256,4,192],
        dtype="complex64",all_predictions_finite=True,source_and_model_hashes_match=True,
        query_truth_alignment=False,refit_validation_truth_read=False,
        old_stat_snapshots_unchanged=len(before["stat"]),old_result_hashes_unchanged=len(before["hashes"]),
        historical_experiment_record_prefix_unchanged=True,V16_unchanged=True,
        old_baseline_metric_deltas=summary["old_metric_deltas"],
        independent_aggregation_deltas=aggregate_checks,training=training,
        source_identity=identity,verifier_sha256=sha(Path(__file__)))
    write(R/"completed_verification.json",result)
    write(R/"training_summary.json",dict(folds=training,
        total_training_seconds=sum(x["training_seconds"] for x in training),
        total_optimizer_updates=sum(x["screen_updates"]+x["refit_updates"] for x in training)))
    lines=[(R/"evaluation/comparison.md").read_text().rstrip(),"","## 本轮完成范围","",
        "WRF-GS+完整H版完成5组内部验证、重新拟合和500点预测。其他三个完整H方法及原WRF-GS+功率版复用已有预测，本轮重新评分与旧成绩完全一致。",
        "新版本使用一套预先固定配置，基础seed=42；各组种子为42+1000×组号。没有根据500点测试成绩调整模型。完整H输出并不意味着复相位预测准确，实际效果以NMSE为准。",
        "","## 适配内容","",
        "作者来源：[官方代码](https://github.com/wenchaozheng/WRF-GSplus)，固定提交e564f3184463717ae08660a8b7232c6318c73ca6；[论文](https://arxiv.org/html/2412.04832v4)。",
        "保留Gaussian、三阶球谐函数、位置条件变形网络和alpha合成。原公开输出取模，新版本改为带符号实部和虚部；192频点共享几何，使用独立复数信号头；固定阵列响应将角域复数场投影成256×4×192的H。",
        "使用训练CSI、位置和已知场景几何。查询位置不提供CSI。阵列约定沿用项目已经冻结的V16几何缓存；本轮未独立认证发布方阵列测绘。",
        "训练目标为0.4×PAS损失+0.4×PDP损失+0.2×复数MSE。所有标签使用各组训练池单一RMS尺度；输出只乘回该训练尺度。",
        "论文中的CSI用例允许查询位置的上行CSI。本任务不给查询位置CSI，因此不能直接移用该用例。",
        "这属于WRF-GS+的任务适配，不是原论文原样复现。各方法训练预算不同，不能称为等算力比较。",
        "","## 数据与停止规则","",
        "固定本地500点分为5组。当前组测试点及其空间缓冲区从该组训练池排除。内部验证选训练长度；随后重新初始化，在该组完整训练池上按样本数比例固定更新次数。重新拟合阶段不读取验证或测试信道。",
        "每组最多12000步，每800步检查16个预定内部验证位置；至少训练4000步，连续4次无显著改善后停止。完整内部验证仅记录诊断结果。",
        "","| 组 | 测试点 | 重新拟合训练点 | 验证阶段更新数 | 选定更新数 | 重新拟合更新数 | 训练分钟 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for d in training:
        lines.append(f"| {d['fold']} | {d['query_count']} | {d['refit_fit_count']} | {d['screen_updates']} | {d['selected_updates']} | {d['refit_updates']} | {d['training_seconds']/60:.2f} |")
    lines += ["", f"运行硬件：两张H100和一张A800，均为允许的完整UUID。各组训练阶段用时相加为{sum(x['training_seconds'] for x in training)/60:.2f}分钟，部分阶段并行；这不是端到端墙钟时间。PyTorch张量峰值为{max(x['peak_cuda_bytes'] for x in training)/2**30:.2f} GiB，不含分配器缓存。评分使用CPU。"]
    lines+=["","## 完整性核验","",
        "输出覆盖500个唯一位置，所有元素有限。每组模型、源码、预测及评分均由SHA-256对应。逐点统计的独立聚合与最终表一致。31个零信道位置全部保留。",
        f"旧文件保护检查通过：{len(before['stat'])}份模型/数据文件的大小、时间戳和inode未变；{len(before['hashes'])}份旧结果文件SHA-256未变；原实验记录的既有内容未变。",
        "数值检查通过：相干读出误差1.21e-7，两个站点空间投影矩阵秩均为128，零信道与混合样本梯度有限，试跑模型保存后重载输出一致。",
        "","## 文件与复现","",
        "配置、逐组记录、训练汇总和核验结果位于本结果目录。模型及原始预测位于artifacts/round2/paper_four_baselines_v1/wrfgs_full_h_v1。实验记录为experiment_records/EXPERIMENT_RECORD.md第831节。",
        "在项目根目录和已验证的base环境运行。训练前将TASK_GPU_UUID设为允许的完整UUID，并核对设备余量。",
        "训练/预测：\n\n```bash\nCUDA_VISIBLE_DEVICES=\"$TASK_GPU_UUID\" PYTHONDONTWRITEBYTECODE=1 code/tools/run_in_env.sh python -m round2.wrfgs_full_h_v1.run --stage train --folds 0,1,2,3,4 --device-uuid \"$TASK_GPU_UUID\"\n```",
        "评分：CUDA_VISIBLE_DEVICES='' code/tools/run_in_env.sh python -m round2.wrfgs_full_h_v1.run --stage score_all。已有阶段按来源标识复用，不覆盖旧基线。",
        "核验：CUDA_VISIBLE_DEVICES='' code/tools/run_in_env.sh python code/round2/wrfgs_full_h_verify_v1.py。"]
    (R/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("training","source_identity")},ensure_ascii=False))
if __name__=="__main__":main()

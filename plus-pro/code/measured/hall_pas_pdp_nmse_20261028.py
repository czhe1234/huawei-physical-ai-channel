"""Frozen south outputs: PAS, relative PDP and unaligned original complex NMSE."""
from __future__ import annotations
import csv,hashlib,json,re,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from project_paths import PATHS
from measured.hall_adapters.common import save_json,sha256
from measured.hall_structured_v2.data import SouthDevelopment
from measured.hall_metric_protocol_v1.metrics import compare
from common.official_metrics import point_metric_statistics,ArrayLayout
REPORT=PATHS.analysis/"results/hall_south_pas_pdp_nmse_20261028"
OLD=PATHS.analysis/"results/hall_south_metric_reevaluation_20261025"
TRIAL=PATHS.analysis/"results/hall_south_objective_trial_20261028"
WORK=PATHS.artifacts/"measured/hall_sw_objective_trial_20261028"

def snapshot():
    result={}
    for base in [PATHS.code_root,PATHS.analysis,PATHS.artifacts/"measured",PATHS.datasets,PATHS.root/"final_code"]:
      for p in base.rglob("*"):
        if not p.is_file() or p.is_symlink() or "__pycache__" in p.parts or REPORT in p.parents:continue
        s=p.stat();result[str(p.relative_to(PATHS.root))]=(s.st_size,s.st_mtime_ns)
    return result

def nmse(pred,truth):
    p=np.asarray(pred,np.complex128);t=np.asarray(truth,np.complex128)
    squared=np.abs(p-t)**2
    numerator=squared.sum(axis=(-3,-2,-1))
    denominator=(np.abs(t)**2).sum(axis=(-3,-2,-1))
    if not (denominator>0).all():raise ValueError("Zero reference panel; explicit new mask contract required")
    panel_ratios=numerator.sum(0)/denominator.sum(0)
    per_cell=numerator/denominator
    return dict(raw_nmse=float(numerator.sum()/denominator.sum()),raw_nmse_by_panel=panel_ratios.tolist(),
        raw_nmse_point_panel_macro=float(per_cell.mean()),error_energy=float(numerator.sum()),truth_energy=float(denominator.sum())),per_cell,numerator,denominator

def tests():
    t=np.ones((2,2,8,4,16),np.complex128);t[1]*=3
    assert nmse(t,t)[0]["raw_nmse"]==0
    assert nmse(np.zeros_like(t),t)[0]["raw_nmse"]==1
    assert nmse(-t,t)[0]["raw_nmse"]==4
    p=t.copy();p[0]=0
    d,*_=nmse(p,t)
    assert abs(d["raw_nmse"]-.1)<1e-14 and abs(d["raw_nmse_point_panel_macro"]-.5)<1e-14
    return dict(passed=True,identical=0,zero=1,opposite_phase=4,unequal_energy_global=.1,unequal_energy_macro=.5)

def run():
    start=time.monotonic()
    if REPORT.exists():raise FileExistsError("Preserve existing evaluation")
    REPORT.mkdir(parents=True)
    before=snapshot();checks=tests()
    historical=json.loads((OLD/"evaluation.json").read_text())
    trial_contract=json.loads((TRIAL/"contract.json").read_text())
    reader=SouthDevelopment()
    assert historical["data_identity"]==trial_contract["data_identity"]==reader.data.identity
    assert historical["rows"]==trial_contract["validation_rows"]==reader.validation_ids.tolist()
    cases=[]
    for record in historical["full_channel_records"]:
        cases.append(dict(name=record["name"],label=record["label"],path=record["prediction_file"],
            sha256=record["prediction_sha256"],expected=record["scores"],provenance=record["row_provenance"]))
    labels={"old":"沿用旧训练目标继续训练","pdp":"增加相对时延监督","joint":"兼顾相对时延、方向谱与联合分布"}
    for name,label in labels.items():
        r=json.loads((TRIAL/(name+".json")).read_text());assert r["complete"] and r["epoch"]==40 and r["steps"]==120
        p=WORK/name/"final_channel.npz"
        cases.append(dict(name="trial_"+name,label=label,path=str(p.relative_to(PATHS.root)),
            sha256=sha256(p),expected=r["final_validation"],provenance="Explicit validation_ids; fixed40-epoch endpoint"))
    protocol=PATHS.code_root/"measured/hall_metric_protocol_v1/metrics.py"
    assert sha256(protocol)==historical["metric_source_sha256"]
    ledger=PATHS.root/"experiment_records/EXPERIMENT_RECORD.md";prefix=ledger.read_bytes()
    item=max(map(int,re.findall(r"^## (\d+)\.",prefix.decode(),re.M)))+1
    contract=dict(question="What are PAS, relative PDP and raw complex NMSE for the fixed existing south outputs?",
        scope="Frozen-output rescoring only,16 complex outputs; no training,checkpoint load,reselection,RT,formal TEST or north",
        data_identity=reader.data.identity,validation_rows=reader.validation_ids.tolist(),
        PAS_PDP="Frozen per-frequency PAS and independent one-reference-per-panel relative PDP",
        raw_NMSE="sum over all positions/panels/elements/frequencies |prediction-truth|^2 / sum |truth|^2; FP64 accumulation",
        raw_NMSE_alignment="None: no phase,delay,amplitude or self-reference transform",
        raw_NMSE_auxiliary="Per-panel ratios and mean per-position/panel ratios retained separately",
        total_score=None,cases=cases,metric_source_sha256=sha256(protocol),source_sha256=sha256(Path(__file__)),
        limitations="Same48 reused development positions in one4m block; absolute phase/time references unverified; raw NMSE is supplementary",
        ledger_item=item)
    save_json(REPORT/"contract.json",contract)
    with ledger.open("a") as f:f.write(f"\n\n## {item}. 2026-10-28 South frozen PAS relative-PDP and raw-NMSE comparison\n\n### Contract\n- User authorizes viewing saved-model PAS/PDP/NMSE.16 complex outputs, same48 south validation positions; no retraining,reselection,formal TEST,north or RT.\n- Raw NMSE uses official global summed-error/summed-reference-energy aggregation on untouched channels. Relative reference applies only to PDP; no new total score. Contract: analysis/results/hall_south_pas_pdp_nmse_20261028/contract.json.\n")
    predictions={}
    for case in cases:
        p=PATHS.root/case["path"];assert sha256(p)==case["sha256"],case["name"]
        with np.load(p,allow_pickle=False) as z:
            if "validation_ids" in z:assert np.array_equal(z["validation_ids"],reader.validation_ids)
            elif case["provenance"]!="old fixed48-row contract plus reproduced original direction/beam scores":raise ValueError("Unverified row order")
            h=np.array(z["validation"])
        assert h.shape==(48,2,8,4,1024) and h.dtype==np.complex64 and np.isfinite(h).all()
        predictions[case["name"]]=h
    assert not reader.access
    reader.allow_validation=True;truth=reader.reference("validation")
    records=[];points=[];independent=[]
    for case in cases:
        name=case["name"];h=predictions[name];structure=compare(h,truth)
        parity={k:abs(structure[k]-case["expected"][k]) for k in ["pas_cosine","relative_pdp_cosine"]}
        assert max(parity.values())<1e-12,(name,parity)
        error,per,num,den=nmse(h,truth)
        # Independent real/imag arithmetic avoids complex absolute/square implementation.
        dr=h.real.astype(np.float64)-truth.real.astype(np.float64)
        di=h.imag.astype(np.float64)-truth.imag.astype(np.float64)
        independent_nmse=float((dr*dr+di*di).sum()/((truth.real.astype(np.float64)**2+truth.imag.astype(np.float64)**2).sum()))
        assert abs(independent_nmse-error["raw_nmse"])<1e-10*max(1,error["raw_nmse"])
        if name in ["ours_default","frequency_RBF","trial_joint"]:
            axis=lambda x:x.reshape(48,2,32,1024).transpose(0,2,1,3)
            official=point_metric_statistics(axis(h),axis(truth),ArrayLayout(1,8,4))
            official_nmse=float(official.squared_error.sum()/official.truth_energy.sum())
            assert abs(official_nmse-error["raw_nmse"])<1e-5*max(1,error["raw_nmse"])
            independent.append(dict(name=name,official_nmse=official_nmse,FP64_nmse=error["raw_nmse"],difference=official_nmse-error["raw_nmse"]))
        record=dict(name=name,label=case["label"],pas_cosine=structure["pas_cosine"],
            relative_pdp_cosine=structure["relative_pdp_cosine"],**error,by_panel=structure["by_panel"],
            source=case["path"],prediction_sha256=case["sha256"],PAS_PDP_parity=parity)
        records.append(record)
        for i,row in enumerate(reader.validation_ids):
            for panel in range(2):
                points.append(dict(method=name,row_id=int(row),panel=panel,raw_nmse=float(per[i,panel]),
                    squared_error=float(num[i,panel]),truth_energy=float(den[i,panel])))
        print("SCORED",name,json.dumps({k:record[k] for k in ["pas_cosine","relative_pdp_cosine","raw_nmse"]}),flush=True)
    fields=["name","label","pas_cosine","relative_pdp_cosine","raw_nmse","raw_nmse_point_panel_macro","error_energy","truth_energy"]
    with (REPORT/"comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(records)
    with (REPORT/"per_position_panel_nmse.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
    changed=[k for k,v in before.items() if not (PATHS.root/k).is_file() or ((PATHS.root/k).stat().st_size,(PATHS.root/k).stat().st_mtime_ns)!=tuple(v)]
    assert not changed,changed
    for case in cases:assert sha256(PATHS.root/case["path"])==case["sha256"]
    assert ledger.read_bytes().startswith(prefix)
    check=subprocess.run([sys.executable,str(PATHS.code_root/"tools/check_hardcoded_paths.py")],capture_output=True,text=True)
    diagnostics=(check.stdout+check.stderr).splitlines()
    own=[x for x in diagnostics if "hall_pas_pdp_nmse_20261028.py:" in x or "hall_south_pas_pdp_nmse_20261028/" in x]
    assert not own,own
    save_json(REPORT/"verification.json",dict(passed=True,preserved_files=len(before),changed_old_files=changed,
        ledger_prefix=True,operator_tests=checks,official_arithmetic=independent,
        global_path_returncode=check.returncode,global_path_diagnostic_lines=len(diagnostics),new_scope_path_failures=own,
        source_sha256=sha256(Path(__file__)),metric_source_sha256=sha256(protocol)))
    save_json(REPORT/"evaluation.json",dict(complete=True,records=records,unavailable=historical["broadband_only_records"],
        seconds=time.monotonic()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        absolute_phase_or_time_claim=False,new_training=False,checkpoint_loads=False,north=False,formal_test=False))
    selected=["ours_default","trial_old","trial_pdp","trial_joint","frequency_RBF","spectral_mlp_w1","ours_phase_safe","nearest_FIT"]
    lookup={r["name"]:r for r in records}
    text=["# 南侧PAS、相对PDP和原始复信道NMSE","","全部来自已经保存的预测。固定同一批48个南侧验证位置、两副接收阵列、32个真实接收通道和1024个频点。没有重训或重新选择检查点。","",
        "PAS和相对PDP越大越好。原始NMSE越小越好。原始NMSE采用比赛的全样本误差能量总和除以真值能量总和，不能与逐点NMSE平均混称。","",
        "相对PDP：每个位置每副阵列的32个接收通道共同取一个参考。预测与真值各自计算参考。原始NMSE：直接比较保存的预测与真值，不使用该参考变换，也不拟合两者的相位、时延或幅度。","",
        "| 方法 | PAS↑ | 相对PDP↑ | 原始NMSE↓ |","|---|---:|---:|---:|"]
    labels={"ours_default":"原模型","trial_old":"旧目标继续训练","trial_pdp":"增加相对时延监督","trial_joint":"兼顾方向谱、相对时延与联合分布","frequency_RBF":"核回归基线","spectral_mlp_w1":"位置MLP基线","ours_phase_safe":"早期方向谱版本","nearest_FIT":"最近训练点基线"}
    for name in selected:
        r=lookup[name];text.append(f"|{labels[name]}|{r['pas_cosine']:.4f}|{r['relative_pdp_cosine']:.4f}|{r['raw_nmse']:.4f}|")
    text+=["","当前缺少已核实的统一绝对相位、时延参考。因此原始NMSE同时可能受到参考差异和预测误差影响。高误差不能全部归因于参考，也不能只按该指标断言结构能力无效。全零预测的NMSE为1，作为数值参照；它没有方向和时延结构。","",
        "原始NMSE没有参与本轮模型训练或检查点选择。PAS和相对PDP仍是主要结构指标。没有把三项拼成比赛总分，因为相对PDP与原比赛PDP的参考不同。WRF-GS+等现有宽带功率输出不提供复信道，这三项记为不适用。","",
        "本表属于反复使用的南侧开发验证。北侧与正式测试没有使用。完整16组结果保存在comparison.csv，逐位置/阵列的原始NMSE、分子和分母保存在per_position_panel_nmse.csv。"]
    (REPORT/"scores_zh.md").write_text("\n".join(text)+"\n")
    with ledger.open("a") as f:
        f.write("\n### Completion\n- Recomputed frozen PAS/relative-PDP and added unaligned global raw complex NMSE for16 saved outputs on unchanged48 south validation positions. Old structure scores reproduce within1e-12. No checkpoint loads,optimizer,training,reselection,RT,formal TEST or north.\n")
        for name in selected:
            r=lookup[name];f.write("- "+name+": "+json.dumps({k:r[k] for k in ["pas_cosine","relative_pdp_cosine","raw_nmse"]})+"\n")
        f.write("- Global raw-NMSE arithmetic checked against independent real/imag sums and official implementation; unequal-energy synthetic test distinguishes global ratio from macro mean. Old inputs/results and ledger prefix preserved. Raw NMSE is supplementary under unverified absolute phase/time references; no combined score. Evidence: analysis/results/hall_south_pas_pdp_nmse_20261028/scores_zh.md,evaluation.json,comparison.csv,verification.json,per_position_panel_nmse.csv.\n")
    print("COMPLETE",json.dumps([lookup[x] for x in selected]),flush=True)

if __name__=="__main__":
    try:run()
    except BaseException:
        if REPORT.exists():save_json(REPORT/"failure.json",dict(traceback=traceback.format_exc()))
        raise

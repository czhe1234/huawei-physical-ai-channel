"""Replay one frozen south checkpoint and locate structural changes; CPU only."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, os, platform, re, resource, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from measured.hall_adapters.common import sha256, save_json
from measured.hall_structured_v2.data import SouthDevelopment, slice_prepared
from measured.hall_structured_v2.foundation import prepare
from measured.hall_structured_v2.observables import FitPower
from measured.hall_structured_v2.config import Config
from measured.hall_structured_v2.model import HallStructured
from measured.hall_structured_v2 import synthesis
from measured.hall_metric_protocol_v1.metrics import compare
REPORT=PATHS.analysis/"results/hall_south_joint_stage_diagnostic_20261028"
OLD_REPORT=PATHS.analysis/"results/hall_south_structured_development_v2"
OLD_WORK=PATHS.artifacts/"measured/hall_sw_structured_development_v2"
KEYS=("pas_cosine","relative_pdp_cosine","receive_energy_TV","receive_delay_joint_TV")

def snapshot():
    out={}
    for base in [PATHS.code_root,PATHS.analysis,PATHS.artifacts/"measured",PATHS.datasets,PATHS.root/"final_code"]:
        for p in base.rglob("*"):
            if not p.is_file() or p.is_symlink() or "__pycache__" in p.parts or REPORT in p.parents:continue
            st=p.stat()
            out[str(p.relative_to(PATHS.root))]=(st.st_size,st.st_mtime_ns)
    return out

def run():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("CPU-only execution requires hidden GPUs")
    torch.set_num_threads(2);torch.manual_seed(20261028)
    if (torch.__version__.split("+")[0],np.__version__)!=("2.5.1","1.26.4"):
        raise RuntimeError("Previously verified runtime changed")
    if REPORT.exists():raise FileExistsError("Preserve the previous attempt")
    REPORT.mkdir(parents=True)
    start=time.monotonic();before=snapshot()
    old=json.loads((OLD_REPORT/"ours_default.json").read_text());ct=old["contract"]
    fixed=PATHS.artifacts/"measured/hall_sw_objective_trial_20261028/joint/final.pt"
    saved=PATHS.artifacts/"measured/hall_sw_objective_trial_20261028/joint/final_channel.npz"
    source_files=list((PATHS.code_root/"measured/hall_structured_v2").glob("*.py"))
    model_sources={str(p.relative_to(PATHS.root)):sha256(p) for p in source_files}
    assert all(ct["source_sha256"][k]==v for k,v in model_sources.items())
    frozen={str(p.relative_to(PATHS.root)):sha256(p) for p in [fixed,saved,Path(__file__)]}
    ledger=PATHS.root/"experiment_records/EXPERIMENT_RECORD.md"
    prefix=ledger.read_bytes()
    item=max(map(int,re.findall(r"^## (\d+)\.",prefix.decode(),re.M)))+1
    contract=dict(primary_hypothesis="The learned foundation and/or alternating projections change relative-delay structure; localize where the observed PDP degradation appears.",
        evidence="Frozen fixed-endpoint joint-target checkpoint, same96 FIT/48 development validation positions, publisher preparation and frozen metric protocol.",
        success_criterion="Regenerate final output within relative RMS5e-4 and final scores within2e-5 of saved output, then report every projection stage with unchanged weights.",
        scope="Forward-trace diagnostic; not a trained ablation or causal attribution.",
        model_selection=False,training=False,formal_test=False,north=False,new_RT=False,
        max_cpu_seconds=300,CPU_threads=2,source_sha256=model_sources,artifact_sha256=frozen,
        data_identity=ct["data_identity"],validation_rows=ct["validation_rows"],ledger_item=item,
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,visible_GPUs=os.environ.get("CUDA_VISIBLE_DEVICES")))
    save_json(REPORT/"contract.json",contract)
    with ledger.open("a") as f:f.write(f"\n\n## {item}. 2026-10-28 South joint-target frozen-checkpoint projection-stage diagnostic\n\n### Contract\n- Resume south development using one fixed old-selected checkpoint. CPU-only forward tracing and same48 validation positions; no training, selection, formal TEST, north or RT.\n- Primary question: locate where relative-delay structure changes across the learned foundation and projection sequence. This is a forward-trace diagnostic, not a trained ablation.\n- Contract: analysis/results/hall_south_joint_stage_diagnostic_20261028/contract.json.\n")
    reader=SouthDevelopment();assert reader.data.identity==ct["data_identity"]
    bank=reader.fit();query=reader.query("validation");assert query.row_ids.tolist()==ct["validation_rows"]
    data=prepare(bank,query,16,"self",reader.geometry(query.row_ids))
    power=FitPower.from_fit(bank.channel,bank.row_ids)
    model=HallStructured(bank.spec,power,Config(**ct["config"]))
    checkpoint=torch.load(fixed,map_location="cpu",weights_only=False)
    model.load_state_dict(checkpoint["state_dict"]);model.eval()
    state_before={k:v.detach().clone() for k,v in model.state_dict().items()}
    stages={};active={}
    originals={name:getattr(synthesis,name) for name in ["angle_project","delay_project","element_project"]}
    def hook(_module,_args,output):
        active["foundation"]=output[0].detach().numpy().copy()
    handle=model.foundation.register_forward_hook(hook)
    def wrapper(name):
        original=originals[name]
        def wrapped(*args,**kwargs):
            value=original(*args,**kwargs)
            active[name].append(value.detach().numpy().copy())
            return value
        return wrapped
    try:
        for name in originals:setattr(synthesis,name,wrapper(name))
        with torch.inference_mode():
            for first in range(0,len(query.row_ids),16):
                if time.monotonic()-start>contract["max_cpu_seconds"]:raise TimeoutError("CPU replay bound")
                active={"angle_project":[],"delay_project":[],"element_project":[]}
                block=slice_prepared(data,np.arange(first,min(first+16,len(query.row_ids))))
                result=model(block)
                outputs={"foundation":active["foundation"]}
                for cycle in range(4):
                    for short,name in [("angle","angle_project"),("delay","delay_project"),("receive","element_project")]:
                        assert len(active[name])==4
                        outputs[f"cycle{cycle+1}_{short}"]=active[name][cycle]
                outputs["final"]=result["channel"].numpy().copy()
                for name,value in outputs.items():stages.setdefault(name,[]).append(value)
                print("REPLAY",min(first+16,len(query.row_ids)),len(query.row_ids),flush=True)
    finally:
        handle.remove()
        for name,value in originals.items():setattr(synthesis,name,value)
    assert all(torch.equal(state_before[k],v) for k,v in model.state_dict().items())
    stages={k:np.concatenate(v) for k,v in stages.items()}
    with np.load(saved,allow_pickle=False) as z:
        assert np.array_equal(z["validation_ids"],query.row_ids);expected=np.array(z["validation"])
    relative_rms=float(np.linalg.norm(stages["final"]-expected)/np.linalg.norm(expected))
    assert relative_rms<5e-4,relative_rms
    assert all(x["role"]=="fit" for x in reader.access)
    # Held-out development channels become available only after all outputs are fixed.
    reader.allow_validation=True;truth=reader.reference("validation")
    records=[];points=[]
    for name,prediction in stages.items():
        if time.monotonic()-start>contract["max_cpu_seconds"]:raise TimeoutError("CPU scoring bound")
        score=compare(prediction,truth)
        records.append(dict(stage=name,**{k:score[k] for k in KEYS},by_panel=score["by_panel"]))
        for i,rid in enumerate(query.row_ids):
            single=compare(prediction[i:i+1],truth[i:i+1])
            for panel in range(2):
                points.append(dict(stage=name,row_id=int(rid),panel=panel,**{k:single["by_panel"][k][panel] for k in KEYS}))
        print("STAGE",name,json.dumps({k:score[k] for k in KEYS}),flush=True)
    expected_scores=compare(expected,truth)
    score_delta={k:abs(records[-1][k]-expected_scores[k]) for k in KEYS}
    assert max(score_delta.values())<2e-5,score_delta
    for row in records:
        for k in KEYS:
            a=[x[k] for x in points if x["stage"]==row["stage"]]
            assert abs(float(np.mean(a))-row[k])<1e-11
    comparisons=[]
    for i,row in enumerate(records):
        if not i:continue
        previous=records[i-1]
        per=[]
        for j in range(48):
            current=[x["relative_pdp_cosine"] for x in points if x["stage"]==row["stage"] and x["row_id"]==int(query.row_ids[j])]
            prior=[x["relative_pdp_cosine"] for x in points if x["stage"]==previous["stage"] and x["row_id"]==int(query.row_ids[j])]
            per.append(float(np.mean(current)-np.mean(prior)))
        comparisons.append(dict(stage=row["stage"],previous=previous["stage"],PDP_delta=row["relative_pdp_cosine"]-previous["relative_pdp_cosine"],
            positions_better=int(np.count_nonzero(np.array(per)>1e-8)),positions_worse=int(np.count_nonzero(np.array(per)<-1e-8))))
    unchanged=[k for k,v in before.items() if not (PATHS.root/k).is_file() or ((PATHS.root/k).stat().st_size,(PATHS.root/k).stat().st_mtime_ns)!=v]
    assert not unchanged,unchanged
    assert ledger.read_bytes().startswith(prefix)
    for name,digest in frozen.items():assert sha256(PATHS.root/name)==digest
    checker=PATHS.code_root/"tools/check_hardcoded_paths.py"
    check=subprocess.run([sys.executable,str(checker)],capture_output=True,text=True)
    output=check.stdout+check.stderr
    own=[line for line in output.splitlines() if "hall_stage_diagnostic_v1.py:" in line or "hall_south_joint_stage_diagnostic_20261028/" in line]
    save_json(REPORT/"path_checks.json",dict(global_returncode=check.returncode,global_diagnostic_lines=output.splitlines(),new_scope_failures=own))
    assert not own,own
    elapsed=time.monotonic()-start
    result=dict(complete=True,stages=records,adjacent_changes=comparisons,relative_RMS_error=relative_rms,
        final_score_deltas=score_delta,epoch=checkpoint["epoch"],elapsed_seconds=elapsed,
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        preserved_files=len(before),all_model_buffers_and_parameters_unchanged=True,access=reader.access,
        limitations=["One repeatedly used4m validation block, one checkpoint, no new fit.",
            "Projection order is fixed; local changes do not estimate independently trained component contributions.",
            "The learned foundation differs from the old power-fusion system; their difference is not attributed to one head."])
    save_json(REPORT/"diagnostic.json",result)
    save_json(REPORT/"verification.json",dict(passed=True,relative_RMS_error=relative_rms,final_score_deltas=score_delta,
        preserved_files=len(before),changed_files=unchanged,ledger_prefix=True,unchanged_parameters=True,
        validation_labels_accessed_after_prediction=True,formal_test=False,north=False,training=False,new_source_sha256=sha256(Path(__file__))))
    for filename,rows,fields in [("stages.csv",records,["stage",*KEYS]),("per_position_panel.csv",points,["stage","row_id","panel",*KEYS])]:
        with (REPORT/filename).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    lines=["# 南侧已有模型的分阶段诊断","","固定原来选中的默认模型。沿用96个训练位置和48个验证位置。没有重训，没有重新选择检查点。没有使用正式测试或北侧数据。","",
        "本次只观察同一个预测在合成步骤中的变化。它不等于重新训练后的模块消融。","",
        "| 阶段 | 逐频点方向谱↑ | 相对PDP↑ | 通道能量占比误差↓ | 通道—时延联合误差↓ |",
        "|---|---:|---:|---:|---:|"]
    for row in records:lines.append("|"+row["stage"]+"|"+"|".join(f'{row[k]:.6f}' for k in KEYS)+"|")
    lines+=["",f"固定检查点轮次：{checkpoint['epoch']}。CPU重现与保存输出的相对均方根差：{relative_rms:.3g}。",
        f"用时{elapsed:.1f}秒。CPU线程2。峰值内存{result['peak_RSS_MiB']:.1f}MiB。",
        "参数与缓冲区未改变。验证真值只在全部预测固定后用于评价。原数据、旧模型和旧结果保持不变。","",
        "相邻阶段的变化同时受输入和固定执行顺序影响。它不能独立证明某个模块训练后的贡献。本轮模型已经使用直接PAS、相对PDP和联合分布监督。这里检查当前修正过程，不沿用旧模型的退步归因。"]
    (REPORT/"diagnostic_zh.md").write_text("\n".join(lines)+"\n")
    with ledger.open("a") as f:
        f.write("\n### Completion\n- Traced the unchanged default epoch"+str(checkpoint["epoch"])+" checkpoint on48 south validation positions through learned foundation, all12 alternating projection stages and final output. All outputs precede validation-label access. No training, checkpoint selection, formal TEST, north or new RT.\n")
        f.write("- Final saved-output relative RMS difference"+str(relative_rms)+"; final new-metric deltas"+json.dumps(score_delta)+". Parameters/buffers unchanged; "+str(len(before))+" preexisting file identities and ledger prefix preserved.\n")
        f.write("- Stage metrics and paired position counts are descriptive forward-trace diagnostics, not independently trained component contributions. Evidence: analysis/results/hall_south_joint_stage_diagnostic_20261028/diagnostic.json,diagnostic_zh.md,stages.csv,per_position_panel.csv,verification.json.\n")
    print("COMPLETE",json.dumps(dict(elapsed_seconds=elapsed,relative_RMS_error=relative_rms,stages=records)),flush=True)

if __name__=="__main__":
    try:run()
    except BaseException:
        if REPORT.exists():save_json(REPORT/"failure.json",dict(traceback=traceback.format_exc()))
        raise

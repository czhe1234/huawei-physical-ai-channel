"""Verify and report the matched objective-only trial without training."""
import csv,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from measured.hall_adapters.common import sha256,save_json
from measured.hall_objective_trial_v1.run import REPORT,WORK,SOURCE,snapshot,VARIANTS
from measured.hall_objective_trial_v1.objectives import KEYS
from measured.hall_structured_v2.data import SouthDevelopment,slice_prepared
from measured.hall_structured_v2.foundation import prepare
from measured.hall_structured_v2.observables import FitPower
from measured.hall_structured_v2.model import HallStructured
from measured.hall_structured_v2.config import Config
from measured.hall_metric_protocol_v1.metrics import compare

def main():
    torch.set_num_threads(2)
    contract=json.loads((REPORT/"contract.json").read_text())
    results={n:json.loads((REPORT/(n+".json")).read_text()) for n in VARIANTS}
    assert all(r["complete"] and r["steps"]==120 and r["epoch"]==40 for r in results.values())
    assert len({r["initial_state_sha256"] for r in results.values()})==1
    assert len({r["order_sha256"] for r in results.values()})==1
    before=json.loads((REPORT/"preservation_before.json").read_text());after=snapshot()
    changed=[k for k,v in before.items() if after.get(k)!=v];assert not changed,changed
    ledger=PATHS.root/"experiment_records/EXPERIMENT_RECORD.md";prefix=(WORK/"ledger_before.bin").read_bytes()
    assert ledger.read_bytes().startswith(prefix)
    for name,digest in contract["source_sha256"].items():assert sha256(PATHS.root/name)==digest
    for name,digest in contract["old_model_sources"].items():assert sha256(PATHS.root/name)==digest
    reader=SouthDevelopment();bank=reader.fit();query=reader.query("validation")
    data=prepare(bank,query,16,"self",reader.geometry(query.row_ids))
    power=FitPower.from_fit(bank.channel,bank.row_ids)
    common=json.loads((PATHS.analysis/"results/hall_south_structured_development_v2/ours_default.json").read_text())["contract"]
    predictions={};regeneration={}
    for name,r in results.items():
        with np.load(WORK/name/"final_channel.npz",allow_pickle=False) as z:
            assert np.array_equal(z["validation_ids"],query.row_ids)
            h=np.array(z["validation"])
        assert h.shape==(48,2,8,4,1024) and h.dtype==np.complex64 and np.isfinite(h).all()
        checkpoint=torch.load(WORK/name/"final.pt",map_location="cpu",weights_only=False)
        assert checkpoint["epoch"]==40 and checkpoint["steps"]==120 and checkpoint["variant"]==name
        m=HallStructured(bank.spec,power,Config(**common["config"]));m.load_state_dict(checkpoint["state_dict"]);m.eval()
        with torch.inference_mode():small=m(slice_prepared(data,np.arange(3)))["channel"].numpy()
        diff=float(np.linalg.norm(small-h[:3])/np.linalg.norm(h[:3]));assert diff<5e-4,diff
        regeneration[name]=diff;predictions[name]=h
    assert all(x["role"]=="fit" for x in reader.access)
    reader.allow_validation=True;truth=reader.reference("validation")
    points=[];metrics_delta={};pairwise={}
    for name,h in predictions.items():
        actual=compare(h,truth)
        metrics_delta[name]={k:abs(actual[k]-results[name]["final_validation"][k]) for k in KEYS}
        assert max(metrics_delta[name].values())<1e-12
        for i,row in enumerate(query.row_ids):
            score=compare(h[i:i+1],truth[i:i+1])
            for panel in range(2):points.append(dict(method=name,row_id=int(row),panel=panel,**{k:score["by_panel"][k][panel] for k in KEYS}))
    with np.load(PATHS.artifacts/"measured/hall_sw_structured_development_v2/models/ours_default/best_channel.npz",allow_pickle=False) as z:
        start_prediction=np.array(z["validation"])
    initial=compare(start_prediction,truth)
    for i,row in enumerate(query.row_ids):
        score=compare(start_prediction[i:i+1],truth[i:i+1])
        for panel in range(2):points.append(dict(method="initial",row_id=int(row),panel=panel,**{k:score["by_panel"][k][panel] for k in KEYS}))
    lookup={(name,int(rid),key):float(np.mean([p[key] for p in points if p["method"]==name and p["row_id"]==int(rid)]))
        for name in ["initial",*VARIANTS] for rid in query.row_ids for key in KEYS}
    for name,h in predictions.items():
        for other in ["initial","old","pdp"]:
            if other==name:continue
            prior=start_prediction if other=="initial" else predictions[other]
            contrasts={}
            for key in KEYS:
                deltas=[]
                for i in range(48):
                    a=lookup[(name,int(query.row_ids[i]),key)];b=lookup[(other,int(query.row_ids[i]),key)]
                    deltas.append(a-b)
                d=np.array(deltas);direction=1 if "cosine" in key else -1
                contrasts[key]=dict(mean_delta=float(d.mean()),positions_improved=int((direction*d>1e-9).sum()),positions_worsened=int((direction*d< -1e-9).sum()))
            pairwise[name+"_vs_"+other]=contrasts
    rows=[dict(method="initial",**{k:initial[k] for k in KEYS})]
    rows += [dict(method=n,**{k:results[n]["final_validation"][k] for k in KEYS}) for n in VARIANTS]
    base=results["old"]["final_validation"]
    gates={}
    for name in ["pdp","joint"]:
        m=results[name]["final_validation"]
        gates[name]=dict(PDP_gain=m["relative_pdp_cosine"]-base["relative_pdp_cosine"],
            PAS_change=m["pas_cosine"]-base["pas_cosine"],receive_TV_change=m["receive_energy_TV"]-base["receive_energy_TV"])
        gates[name]["passed"]=bool(gates[name]["PDP_gain"]>=.01 and gates[name]["PAS_change"]>=-.005 and gates[name]["receive_TV_change"]<=.005)
    for file,data_rows,fields in [("comparison.csv",rows,["method",*KEYS]),("per_position_panel.csv",points,["method","row_id","panel",*KEYS])]:
        with (REPORT/file).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data_rows)
    curves=[]
    for n,r in results.items():
        for row in r["curve"]:curves.append(dict(method=n,epoch=row["epoch"],steps=row["steps"],**row["validation"]))
    with (REPORT/"curves.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["method","epoch","steps",*KEYS]);w.writeheader();w.writerows(curves)
    save_json(REPORT/"pairwise.json",pairwise)
    checker=PATHS.code_root/"tools/check_hardcoded_paths.py"
    c=subprocess.run([sys.executable,str(checker)],capture_output=True,text=True)
    diag=(c.stdout+c.stderr).splitlines()
    own=[line for line in diag if "hall_objective_trial_v1/" in line or "hall_objective_trial_report_v1.py:" in line or "hall_south_objective_trial_20261028/" in line]
    save_json(REPORT/"path_checks.json",dict(global_returncode=c.returncode,global_diagnostics=diag,new_scope_failures=own));assert not own,own
    verified=dict(passed=True,same_initial_weights=True,same_batch_order=True,fixed_endpoint=True,
        preserved_files=len(before),changed_old_files=changed,ledger_prefix=True,
        checkpoint_relative_RMS=regeneration,numpy_metric_differences=metrics_delta,
        finalizer_sha256=sha256(Path(__file__)),access=reader.access,formal_test=False,north=False)
    save_json(REPORT/"verification.json",verified)
    total=sum(r["seconds"] for r in results.values())
    summary=dict(complete=True,results=rows,screening=gates,model_seconds_total=total,
        peak_GPU_GiB=max(r["peak_GPU_GiB"] for r in results.values()),GPU_UUID=results["old"]["environment"]["GPU_UUID"],
        limitations=["One seed and one repeatedly used4m validation block; no independent confirmation.",
            "Finetuning from an old-validation-selected checkpoint; does not establish from-scratch optimality.",
            "C adds PAS and joint losses together; not an attribution to a single head.",
            "Geometry cache and architecture unchanged; no new geometry contribution evidence."])
    save_json(REPORT/"summary.json",summary)
    figure(rows)
    labels={"initial":"共同起点","old":"沿用旧目标","pdp":"增加相对时延监督","joint":"再兼顾方向谱与联合分布"}
    text=["# 南侧训练目标对照","","本轮只修改训练目标。三组均从同一个旧模型第25轮检查点开始。均使用96个训练位置、48个验证位置、相同随机种子、相同批次顺序、相同优化器及120次更新。表内是预先固定的第40轮终点，没有重新挑选检查点。","",
        "| 条件 | 逐频点方向谱↑ | 相对PDP↑ | 通道能量占比误差↓ | 通道—时延联合误差↓ |",
        "|---|---:|---:|---:|---:|"]
    for row in rows:text.append("|"+labels[row["method"]]+"|"+"|".join(f'{row[k]:.6f}' for k in KEYS)+"|")
    text+=["","最后一组相对共同起点改善了方向谱、相对PDP和通道—时延联合误差。通道能量占比误差小幅增加。相对同预算旧目标组，最后一组四项指标都更好。","",
        "最后一组的相对PDP也超过原模型测量底座约0.5985。不过方向谱仍低于已有核回归约0.7607，通道能量占比误差也仍高于训练平均分配参照约0.2157。不能据此声称实测全面领先。","",
        "这支持训练目标不一致是当前退步的可改善因素之一。它没有证明唯一原因，也没有证明几何路径模块的独立贡献。","",
        "本轮是一个随机种子、一个反复使用的4米验证块上的开发比较。北侧和正式测试保持未用。两副阵列的完整结果保存在各组JSON中，逐位置差异保存在pairwise.json。","",
        f"三组模型过程合计{total:.1f}秒。设备为允许的A800。峰值显存{summary['peak_GPU_GiB']:.2f}GiB。完整曲线、配置、日志、模型和预测均已保存。","",
        "![固定终点比较](objective_comparison.png)","",
        "图注：同一批48个南侧验证位置。两副阵列分别计算指标后等权平均。Initial为共同起点；Old target为沿用旧目标；+PDP为增加直接相对PDP监督；+PAS/joint在此基础上增加直接PAS和0.25倍联合TV监督。每组一个随机种子。无误差线或独立重复推断。"]
    (REPORT/"trial_zh.md").write_text("\n".join(text)+"\n")
    if "### Completion" not in ledger.read_text().split("## "+str(contract["ledger_item"])+".",1)[-1]:
        with ledger.open("a") as f:
            f.write("\n### Completion\n- Three matched40-epoch/120-update finetunes complete on the permitted A800, fixed final endpoint and identical initial-state/batch-order hashes. No architecture, old-source, split, formal TEST, north or RT changes.\n")
            for row in rows:f.write("- "+row["method"]+": "+json.dumps({k:row[k] for k in KEYS})+"\n")
            f.write("- Joint-target variant improves PAS/PDP/joint TV versus common initial; receive TV slightly worsens. All four improve versus matched old-target continuation. Both new-target variants pass predeclared developmental screening. This supports objective mismatch as one improvable factor, not a unique cause, new geometry causality or independent confirmation.\n")
            f.write("- Preserved"+str(len(before))+" prior file identities and ledger prefix; same initial weights and batch order verified. CPU checkpoint spot regeneration and NumPy final-metric recomputation pass. Total model-process"+str(total)+"seconds. Evidence: analysis/results/hall_south_objective_trial_20261028/trial_zh.md,summary.json,comparison.csv,curves.csv,pairwise.json,verification.json; models in artifacts/measured/hall_sw_objective_trial_20261028.\n")
    print("VERIFIED",json.dumps(summary),flush=True)

def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.findfont("Arial",fallback_to_default=False)
    plt.rcParams.update({"font.family":"Arial","font.size":6,"axes.labelsize":7,"xtick.labelsize":5.5,"ytick.labelsize":5.5,
        "axes.linewidth":.5,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,112/25.4))
    colors=["#888888","#000000","#D55E00","#009E73"];markers=["o","s","^","D"]
    names=["Initial","Old target","+ PDP","+ PAS / joint"]
    ylabels=["Per-frequency PAS cosine","Relative PDP cosine","Receive-energy TV","Receive-delay joint TV"]
    for letter,ax,key,label in zip("abcd",axes.flat,KEYS,ylabels):
        vals=[r[key] for r in rows];spread=max(vals)-min(vals);padding=max(spread*.35,.004)
        ax.set_ylim(min(vals)-padding,max(vals)+padding*1.6)
        for i,(value,c,m) in enumerate(zip(vals,colors,markers)):
            ax.plot(i,value,m,color=c,markersize=4,linestyle="none")
            ax.annotate(f"{value:.4f}",(i,value),xytext=(0,7),textcoords="offset points",ha="center",fontsize=5.5)
        ax.set_xticks(range(4),names);ax.set_xlim(-.45,3.45);ax.set_ylabel(label)
        ax.spines[["top","right"]].set_visible(False);ax.tick_params(direction="out",width=.5,length=2)
        ax.text(-.17,1.06,letter,transform=ax.transAxes,fontweight="bold",fontsize=8,va="top")
    fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.94,wspace=.30,hspace=.45)
    for ext in ["png","pdf","svg","eps"]:
        fig.savefig(REPORT/("objective_comparison."+ext),dpi=400)
    plt.close(fig)

if __name__=="__main__":main()

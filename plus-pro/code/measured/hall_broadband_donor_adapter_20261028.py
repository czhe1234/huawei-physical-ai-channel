"""Frozen broadband baselines with a fixed FIT-donor complex-channel adapter."""
from __future__ import annotations
import csv,json,re,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from measured.hall_adapters.common import sha256,save_json
from measured.hall_adapters.metrics import directional_power,normalized_power,primary_metrics,aggregate_primary
from measured.hall_structured_v2.data import SouthDevelopment
from measured.hall_structured_v2.foundation import prepare
from measured.hall_structured_development.baselines import write_band
from measured.hall_metric_protocol_v1.metrics import compare
from measured.hall_pas_pdp_nmse_20261028 import nmse
REPORT=PATHS.analysis/"results/hall_south_broadband_donor_20261028"
WORK=PATHS.artifacts/"measured/hall_sw_broadband_donor_20261028"
OLD=PATHS.analysis/"results/hall_south_metric_reevaluation_20261025"
SCORES=PATHS.analysis/"results/hall_south_pas_pdp_nmse_20261028"

def snapshot():
    result={}
    for base in [PATHS.code_root,PATHS.analysis,PATHS.artifacts/"measured",PATHS.datasets,PATHS.root/"final_code"]:
      for p in base.rglob("*"):
        if not p.is_file() or p.is_symlink() or "__pycache__" in p.parts or any(d in p.parents for d in [REPORT,WORK]):continue
        s=p.stat();result[str(p.relative_to(PATHS.root))]=(s.st_size,s.st_mtime_ns)
    return result

def adapt(carrier,power):
    h=np.asarray(carrier,np.complex128);p=np.asarray(power,np.float64)
    if h.ndim!=5 or h.shape[1:4]!=(2,8,4) or p.shape!=h.shape[:2]+(32,):raise ValueError("Layout mismatch")
    if not np.isfinite(h).all():raise ValueError("Nonfinite carrier")
    q=normalized_power(p)
    z=np.fft.fft2(h,axes=(-3,-2),norm="ortho").reshape(*h.shape[:2],32,h.shape[-1])
    broad=(np.abs(z)**2).mean(-1);energy=broad.sum(-1,keepdims=True)
    target=q*energy
    missing=(broad==0)&(target>0)
    if missing.any():raise ValueError("A requested direction has zero donor support; no phase or spectrum fabricated")
    gain=np.sqrt(np.divide(target,broad,out=np.zeros_like(target),where=broad>0))
    output=np.fft.ifft2((z*gain[...,None]).reshape(h.shape),axes=(-3,-2),norm="ortho").astype(np.complex64)
    return output,dict(zero_donor_directions=int((broad==0).sum()),unsupported_requested_directions=int(missing.sum()),
        zero_prediction_panels=int((p.sum(-1)==0).sum()),donor_min_beam_power=float(broad.min()),
        direction_gain_max=float(gain.max()),direction_gain_median=float(np.median(gain)))

def tests():
    rng=np.random.default_rng(20261028)
    h=(rng.normal(size=(2,2,8,4,64))+1j*rng.normal(size=(2,2,8,4,64))).astype(np.complex64)
    p=rng.uniform(.1,1,size=(2,2,32))
    out,d=adapt(h,p)
    profile=float(np.max(np.abs(normalized_power(directional_power(out))-normalized_power(p))))
    energy=float(np.max(np.abs(directional_power(out).sum(-1)/directional_power(h).sum(-1)-1)))
    assert profile<2e-7 and energy<2e-7
    same,_=adapt(h,directional_power(h));assert np.linalg.norm(same-h)/np.linalg.norm(h)<2e-7
    zero,_=adapt(h,np.zeros_like(p));assert np.all(zero==0)
    bad=np.ones_like(h);rejected=False
    try:adapt(bad,p)
    except ValueError:rejected=True
    assert rejected
    return dict(passed=True,direction_profile_error=profile,total_energy_relative_error=energy,
        identity_and_zero_prediction=True,zero_support_rejected=True)

def run():
    start=time.monotonic();torch.set_num_threads(2)
    if REPORT.exists() or WORK.exists():raise FileExistsError("Do not overwrite an attempt")
    REPORT.mkdir(parents=True);WORK.mkdir(parents=True)
    before=snapshot();test=tests()
    previous=json.loads((OLD/"evaluation.json").read_text())
    existing=json.loads((SCORES/"evaluation.json").read_text())
    reader=SouthDevelopment()
    assert previous["data_identity"]==reader.data.identity and previous["rows"]==reader.validation_ids.tolist()
    cases=previous["broadband_only_records"]
    assert [x["name"] for x in cases]==["wrf_g64","wrf_g128","mlp_levels3","mlp_levels6"]
    ledger=PATHS.root/"experiment_records/EXPERIMENT_RECORD.md";prefix=ledger.read_bytes()
    item=max(map(int,re.findall(r"^## (\d+)\.",prefix.decode(),re.M)))+1
    metric=PATHS.code_root/"measured/hall_metric_protocol_v1/metrics.py"
    assert sha256(metric)==previous["metric_source_sha256"]
    contract=dict(question="How do fixed WRF/MLP broadband directional predictors perform when coupled to the same nearest FIT donor carrier?",
        primary_hypothesis="A deterministic donor adapter can preserve each saved broadband directional prediction and provide a valid derived complex output without query CSI.",
        acceptance="Broadband normalized profile error<2e-5; donor total-energy relative error<2e-5 for nonzero outputs; frozen write_band forward relativeRMS<5e-5; no unsupported donor directions; reproduce old direction/beam scores",
        name="WRF-GS+ broadband directional power plus FIT-donor total energy,spectrum and phase",
        source_sha256=sha256(Path(__file__)),data_identity=reader.data.identity,fit_rows=reader.fit_ids.tolist(),validation_rows=reader.validation_ids.tolist(),
        input_predictions=[dict(name=x["name"],path=x["prediction_file"],sha256=x["prediction_sha256"]) for x in cases],
        donor_rule="Same prepared.carrier as spectral RBF/MLP; nearest allowed96-FIT point by xy distance, deterministic row-ID tie break",
        energy_rule="Normalize model output across32 directions; supply absolute total energy from that donor independently per panel",
        frequency_phase_rule="Keep donor within-direction frequency shape and phase; one positive real direction gain shared by all frequencies",
        zero_rule="Zero model output stays zero and receives frozen failure penalties; positive requested power in an exactly zero donor direction aborts the adapter without dropping samples",
        metrics="Frozen per-frequency PAS,self-referenced relative PDP,unaligned original global NMSE; energy/joint and old beam metrics also retained",
        selection="Retain both64/128 WRF and both old power MLP checkpoints; no new model or epoch selection",
        scope="Fixed-output hybrid baseline development, not native full-channel WRF replication or matched-new-objective optimization",
        new_training=False,checkpoint_loads=False,new_RT=False,formal_test=False,north=False,ledger_item=item)
    save_json(REPORT/"contract.json",contract)
    with ledger.open("a") as f:f.write(f"\n\n## {item}. 2026-10-28 Frozen broadband WRF and MLP with FIT-donor carrier\n\n### Contract\n- Complete a concrete fair derived baseline from saved outputs. Same96 FIT/48 validation; WRF supplies normalized broadband directions, nearest permitted FIT supplies total energy,frequency shape and phase. No query-truth correction.\n- Both64/128 WRF and old power MLP variants retained; no new training/checkpoint selection/RT/TEST/north. Contract: analysis/results/hall_south_broadband_donor_20261028/contract.json.\n")
    bank=reader.fit();query=reader.query("validation")
    data=prepare(bank,query,16,"self",None);carrier=data.carrier.numpy()
    donor_ids=np.array([ids[0] for ids in data.source_ids],np.int64)
    assert set(donor_ids).issubset(set(reader.fit_ids)) and not set(donor_ids)&set(reader.validation_ids)
    with np.load(PATHS.artifacts/"measured/hall_sw_metric_reevaluation_v1/nearest_channel.npz",allow_pickle=False) as z:
        assert np.array_equal(z["validation_ids"],query.row_ids) and np.array_equal(z["validation"],carrier)
    donor_energy=directional_power(carrier).sum(-1)
    predictions={};diagnostics={};parity={};power_outputs={}
    for case in cases:
        p=PATHS.root/case["prediction_file"];assert sha256(p)==case["prediction_sha256"]
        with np.load(p,allow_pickle=False) as z:power=np.array(z["validation"])
        assert power.shape==(48,2,32) and np.isfinite(power).all() and (power>=0).all()
        out,diag=adapt(carrier,power);actual=directional_power(out);expected=normalized_power(power)
        error=float(np.max(np.abs(normalized_power(actual)-expected)))
        nonzero=power.sum(-1)>0
        energy_error=float(np.max(np.abs(actual.sum(-1)[nonzero]/donor_energy[nonzero]-1))) if nonzero.any() else 0
        assert error<2e-5 and energy_error<2e-5
        # Independent frozen RBF/MLP writer with one broadband subband.
        desired=expected*donor_energy[...,None]
        with torch.no_grad():reference=write_band(torch.tensor(carrier[:3]),torch.tensor(desired[:3,...,None],dtype=torch.float32)).numpy()
        relative=float(np.linalg.norm(out[:3]-reference)/np.linalg.norm(out[:3]));assert relative<5e-5
        diag.update(direction_profile_max_error=error,donor_energy_max_relative_error=energy_error,
            frozen_oneband_write_band_relative_RMS=relative)
        name=case["name"];diagnostics[name]=diag;predictions[name]=out;power_outputs[name]=power
        with (WORK/(name+"_channel.npz")).open("xb") as f:
            np.savez_compressed(f,validation=out,validation_ids=query.row_ids,donor_ids=donor_ids,fit_ids=reader.fit_ids)
        print("ADAPTED",name,json.dumps(diag),flush=True)
    assert all(x["role"]=="fit" for x in reader.access)
    reader.allow_validation=True;truth=reader.reference("validation");reference_power=directional_power(truth)
    records=[];points=[]
    for case in cases:
        name=case["name"];out=predictions[name];score=compare(out,truth)
        raw,*_=nmse(out,truth)
        auxiliary=aggregate_primary(primary_metrics(dict(channel=out),reference_power))
        original=case["auxiliary"]
        delta={k:abs(auxiliary[k]-original[k]) for k in ["mean_direction_cosine","mean_beam_loss_db"]}
        assert max(delta.values())<5e-6,delta
        r=dict(name=name+"_FIT_donor",original_name=name,label=case["label"]+" + 训练点供体适配",
            pas_cosine=score["pas_cosine"],relative_pdp_cosine=score["relative_pdp_cosine"],raw_nmse=raw["raw_nmse"],
            receive_energy_TV=score["receive_energy_TV"],receive_delay_joint_TV=score["receive_delay_joint_TV"],
            raw_nmse_by_panel=raw["raw_nmse_by_panel"],by_panel=score["by_panel"],
            auxiliary=auxiliary,original_auxiliary=original,auxiliary_differences=delta,adapter_diagnostics=diagnostics[name],
            prediction_file=str((WORK/(name+"_channel.npz")).relative_to(PATHS.root)),prediction_sha256=sha256(WORK/(name+"_channel.npz")))
        records.append(r)
        for i,row in enumerate(query.row_ids):
            ss=compare(out[i:i+1],truth[i:i+1]);nn,*_=nmse(out[i:i+1],truth[i:i+1])
            for panel in range(2):
                points.append(dict(method=r["name"],row_id=int(row),panel=panel,pas_cosine=ss["by_panel"]["pas_cosine"][panel],
                    relative_pdp_cosine=ss["by_panel"]["relative_pdp_cosine"][panel],raw_nmse=nn["raw_nmse_by_panel"][panel]))
        print("SCORED",r["name"],json.dumps({k:r[k] for k in ["pas_cosine","relative_pdp_cosine","raw_nmse"]}),flush=True)
    selected=["nearest_FIT","frequency_RBF","spectral_mlp_w1","trial_joint"]
    references=[x for x in existing["records"] if x["name"] in selected]
    fields=["name","label","pas_cosine","relative_pdp_cosine","raw_nmse"]
    with (REPORT/"comparison.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(records+references)
    with (REPORT/"per_position_panel.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
    changed=[k for k,v in before.items() if not (PATHS.root/k).is_file() or ((PATHS.root/k).stat().st_size,(PATHS.root/k).stat().st_mtime_ns)!=v]
    assert not changed,changed
    for case in cases:assert sha256(PATHS.root/case["prediction_file"])==case["prediction_sha256"]
    assert ledger.read_bytes().startswith(prefix)
    pathcheck=subprocess.run([sys.executable,str(PATHS.code_root/"tools/check_hardcoded_paths.py")],capture_output=True,text=True)
    lines=(pathcheck.stdout+pathcheck.stderr).splitlines()
    own=[x for x in lines if "hall_broadband_donor_adapter_20261028.py:" in x or "hall_south_broadband_donor_20261028/" in x or "hall_sw_broadband_donor_20261028/" in x]
    assert not own,own
    save_json(REPORT/"verification.json",dict(passed=True,operator_tests=test,adapter_checks=diagnostics,
        identical_RBF_MLP_donor_carriers=True,preserved_files=len(before),changed_old_files=changed,ledger_prefix=True,
        all_donor_ids_FIT_only=True,validation_truth_accessed_after_all_outputs_saved=True,
        frozen_metric_source_sha256=sha256(metric),source_sha256=sha256(Path(__file__)),
        global_path_returncode=pathcheck.returncode,new_scope_path_failures=own))
    result=dict(complete=True,records=records,existing_references=references,donor_ids=donor_ids.tolist(),access=reader.access,
        seconds=time.monotonic()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        training=False,checkpoint_loads=False,formal_test=False,north=False,
        limitations=["Hybrid systems inherit donor frequency and phase; not learned WRF frequency/delay recovery.",
            "Existing broadband checkpoints were trained/selected on different objectives from the latest proposed method.",
            "Single reused4m validation block; no generalization or native-paper replication claim."])
    save_json(REPORT/"evaluation.json",result)
    text=["# 宽带baseline的训练点供体适配","","本轮复用已有WRF-GS+与功率MLP输出，没有重训、读取检查点或重新选择版本。供体选择与已有核回归、频谱MLP一致。全部来自同一批96个训练位置和48个验证位置。","",
        "WRF-GS+负责32个方向的宽带能量占比。最近的允许训练点提供总能量、每方向频谱形状和相位。预测在接触验证真值之前固定并保存。这是派生混合baseline，不能标成WRF-GS+原论文的完整复信道复现。","",
        "| 方法 | PAS↑ | 相对PDP↑ | 原始NMSE↓ |","|---|---:|---:|---:|"]
    labels={"nearest_FIT":"最近训练点","frequency_RBF":"核回归","spectral_mlp_w1":"频谱MLP","trial_joint":"我们的最新联合目标版本"}
    for rr in records+references:text.append(f"|{labels.get(rr['name'],rr['label'])}|{rr['pas_cosine']:.4f}|{rr['relative_pdp_cosine']:.4f}|{rr['raw_nmse']:.4f}|")
    text+=["","核验内容：方向能量占比重建、供体总能量保持、与现有核回归/MLP的一频带写回算子对照，以及原有宽带方向/波束成绩复现。遇到供体某方向严格为零且预测要求非零时，不补造相位，不删除样本；本轮会停止并报告。本轮实际触发数记录在verification.json。","",
        "相对PDP各自取参考；原始NMSE没有相位、时延或幅度对齐。原始NMSE仍受尚未核实的绝对参考问题影响。没有把三指标合成比赛总分。","",
        "这次只补齐已有宽带输出的派生评价。它没有证明WRF-GS+自身学会频率/时延结构，也不等于各算法按相同新训练目标充分优化后的公平上限。北侧、正式测试和RT均未使用。"]
    (REPORT/"adapter_zh.md").write_text("\n".join(text)+"\n")
    with ledger.open("a") as f:
        f.write("\n### Completion\n- Four derived complex outputs completed from frozen WRF64/128 and old power-MLP3/6. Same nearest FIT donor as existing spectral RBF/MLP; donor total energy retained, model broadband power normalized. No query truth used for output construction.\n")
        for rr in records:f.write("- "+rr["name"]+": "+json.dumps({k:rr[k] for k in ["pas_cosine","relative_pdp_cosine","raw_nmse"]})+"\n")
        f.write("- Adapter profile/energy invariants, frozen oneband writer parity and original direction/beam scores verified; preserved"+str(len(before))+" old file identities and ledger prefix. This is a hybrid donor baseline, not native full-channel WRF reproduction or equal-new-objective algorithm optimality. No training/checkpoint loads/RT/TEST/north. Evidence: analysis/results/hall_south_broadband_donor_20261028/adapter_zh.md,evaluation.json,comparison.csv,verification.json.\n")
    print("COMPLETE",json.dumps(records),flush=True)

if __name__=="__main__":
    try:run()
    except BaseException:
        if REPORT.exists():save_json(REPORT/"failure.json",dict(traceback=traceback.format_exc()))
        raise

"""No-update nGRF gradient/energy diagnosis on its fixed weight1 endpoint."""
from __future__ import annotations
import argparse,json,math,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from round2.paper_baseline_training_v1 import common as C
from round2.baseline_repair_20261111 import ngrf_repair as R

OUT=R.REPORT/"gradient_diagnostic"
ART=R.ART/"gradient_diagnostic"
def cosine(a,b):
    den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/den) if den else None
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--gpu-uuid",required=True);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    gpu=args.gpu_uuid
    free=int(subprocess.check_output(["nvidia-smi","-i",gpu,"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
    if free<24*1024:raise RuntimeError("Requires at least24GiB free on the assigned allowed UUID")
    device_info=C.device_guard(gpu);device=torch.device("cuda:0")
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    trained=json.loads((R.REPORT/"paired/results.json").read_text())
    chosen=next(x for x in trained["variants"] if x["weight"]==1.)
    assert C.sha256(PATHS.root/chosen["checkpoint"])==chosen["checkpoint_sha256"]
    assert C.sha256(PATHS.root/chosen["raw_prediction"])==chosen["raw_prediction_sha256"]
    data=C.FoldData(0,"screen","ngrf");rows=R.PANEL
    assert not set(rows).intersection(data.outer_ids)
    actual=data.read_fit(rows,normalized=False);raw_pred=np.load(PATHS.root/chosen["raw_prediction"],allow_pickle=False)
    target=actual.astype(np.complex128);pred=raw_pred.astype(np.complex128)
    te=np.sum(np.abs(target)**2,(1,2,3));ce=np.sum(np.abs(pred-target)**2,(1,2,3))
    me=np.sum((np.abs(pred)-np.abs(target))**2,(1,2,3));sites=data.sites(rows)
    strength=np.full(len(rows),-np.inf)
    for s in (0,1):
        ix=(sites==s)&(te>0);strength[ix]=10*np.log10(te[ix]/te[ix].max())
    table=[]
    for i,row in enumerate(rows):
        group=("exact_zero" if te[i]==0 else "strong_ge_minus20dB" if strength[i]>=-20 else
            "medium_minus40_to_minus20dB" if strength[i]>=-40 else "weak_lt_minus40dB")
        table.append(dict(row_id=int(row),site=int(sites[i]),energy_group=group,
            target_raw_energy=float(te[i]),target_energy_fraction=float(te[i]/te.sum()),
            complex_SSE_raw=float(ce[i]),magnitude_SSE_raw=float(me[i]),
            complex_numerator_fraction=float(ce[i]/ce.sum()),
            combined_data_numerator_fraction=float((ce[i]+me[i])/(ce.sum()+me.sum())),
            full_complexNMSE_contribution=float(ce[i]/te.sum()),
            full_data_objective_contribution=float((ce[i]+me[i])/te.sum()),
            full_point_complexNMSE=float(ce[i]/te[i]) if te[i] else None))
    frequencies=np.linspace(0,191,12,dtype=np.int64)
    model=R.model_build(data,device)
    model.load_state_dict(torch.load(PATHS.root/chosen["checkpoint"],map_location=device,weights_only=True)["model"])
    model.eval()
    before_hash=R.model_hash(model);rng_before=dict(cpu=R.tensor_hash(torch.get_rng_state()),cuda=R.tensor_hash(torch.cuda.get_rng_state()))
    target_t=torch.tensor(data.read_fit(rows,frequencies),device=device)
    denominator=target_t.abs().square().sum().clamp_min(1e-10)
    p=torch.tensor(data.positions(rows),device=device,dtype=torch.float64)
    site=torch.tensor(sites,device=device,dtype=torch.long)
    freq=torch.tensor(data.frequencies(frequencies),device=device,dtype=torch.float64)
    parameters=[(n,x) for n,x in model.named_parameters() if x.requires_grad]
    layout=[dict(name=n,shape=list(x.shape),elements=x.numel()) for n,x in parameters]
    vectors=[];sample_energy=[];sample_errors=[];replay=[];started=time.monotonic()
    for i in range(len(rows)):
        value=model(p[i:i+1],site[i:i+1],freq)
        reference=np.take(raw_pred[i:i+1],frequencies,axis=-1)/data.scale
        current=value.detach().cpu().numpy()
        rel=float(np.linalg.norm((current-reference).reshape(-1))/max(np.linalg.norm(reference.reshape(-1)),1e-30))
        assert rel<=1e-5,("subset same-checkpoint forward differs",int(rows[i]),rel)
        replay.append(rel)
        error=(value-target_t[i:i+1]).abs().square().sum()
        gradients=torch.autograd.grad(error/denominator,[x for _,x in parameters],allow_unused=True)
        flat=torch.cat([g.reshape(-1) if g is not None else torch.zeros_like(param).reshape(-1)
            for g,(_,param) in zip(gradients,parameters)])
        assert torch.isfinite(flat).all()
        vector=flat.detach().cpu().numpy()
        vectors.append(vector);sample_energy.append(float(target_t[i].abs().square().sum()))
        sample_errors.append(float(error.detach()))
        print(json.dumps(dict(position=int(rows[i]),index=i+1,total=16,gradient_norm=float(np.linalg.norm(vector)),
            elapsed_seconds=time.monotonic()-started)),flush=True)
        if time.monotonic()-started>120:raise RuntimeError("Bounded gradient diagnosis exceeded120seconds")
    vectors=np.stack(vectors);v64=vectors.astype(np.float64);global_gradient=v64.sum(0)
    norms=np.linalg.norm(v64,axis=1);sample_energy=np.array(sample_energy)
    total_energy=float(denominator);fraction=sample_energy/total_energy
    for i,d in enumerate(table):
        own_site=sites==sites[i];strong=own_site&(strength>=-20)&(te>0)
        d["gradient_probe"]=dict(sampled_target_energy=float(sample_energy[i]),
            sampled_target_energy_fraction=float(fraction[i]),sampled_complex_error=float(sample_errors[i]),
            contribution_gradient_norm=float(norms[i]),
            per_point_relative_error_gradient_norm=float(norms[i]/fraction[i]) if fraction[i]>0 else None,
            cosine_with_global_complex_gradient=cosine(v64[i],global_gradient),
            cosine_with_same_site_complex_gradient=cosine(v64[i],v64[own_site].sum(0)),
            cosine_with_same_site_strong_complex_gradient=cosine(v64[i],v64[strong].sum(0)),
            nonzero_gradient_elements=int(np.count_nonzero(vectors[i])))
    after_hash=R.model_hash(model);assert before_hash==after_hash
    rng_after=dict(cpu=R.tensor_hash(torch.get_rng_state()),cuda=R.tensor_hash(torch.cuda.get_rng_state()))
    assert rng_before==rng_after
    C.atomic_npy(ART/"gradient_vectors.npy",vectors)
    gram=v64@v64.T
    result=dict(status="complete",kind="fixed-checkpoint FIT-only no-update diagnostic",
        source_sha256=C.sha256(Path(__file__)),training_source_identity=chosen["source_identity"],
        checkpoint=chosen["checkpoint"],checkpoint_sha256=chosen["checkpoint_sha256"],
        actual_state_hash_before=before_hash,actual_state_hash_after=after_hash,
        RNG_before=rng_before,RNG_after=rng_after,optimizer_created=False,parameter_updates=False,
        full_H_energy_scope=[16,256,4,192],full_data_numerator="sum(|pred-target|^2)+sum((abs(pred)-abs(target))^2), excluding shared activation regularizer",
        gradient_scope=dict(rows=rows.tolist(),BS_ports=list(range(256)),UE_ports=list(range(4)),
            frequency_indices=frequencies.tolist(),frequency_Hz=data.frequencies(frequencies).tolist(),
            mode="eval; dropout disabled; no stochastic training gradients claimed",
            gradient_definition="gradient of per-position complex SSE divided by sampled panel total target energy",
            per_point_gradient_rescaling="divide pooled contribution gradient by that position target-energy fraction; undefined for exactzero",
            regularizer_included=False,magnitude_term_included=False,gradient_clip_applied=False,
            optimizer_preconditioning_applied=False),
        subset_replay_relative_L2=replay,points=table,gradient_gram=gram.tolist(),parameter_layout=layout,
        raw_gradient_vectors=PATHS.relative(ART/"gradient_vectors.npy"),
        gradient_vectors_sha256=C.sha256(ART/"gradient_vectors.npy"),device=device_info,
        elapsed_seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        validation_labels_read=False,outer_truth_read=False,formal_test=False,
        interpretation_limit="Nonzero gradients show local sensitivity, not representational sufficiency or guaranteed learning. Direction diagnosis is raw complex-term eval gradient only.")
    C.atomic_json(OUT/"results.json",result)
    print(json.dumps(dict(status="complete",points=16,model_unchanged=True,RNG_unchanged=True)),flush=True)
if __name__=="__main__":
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/"failure.log").open("a") as f:f.write(traceback.format_exc()+"\n")
        raise

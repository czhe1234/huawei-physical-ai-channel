"""Bounded no-update full-panel nGRF dropout diagnostic."""
from __future__ import annotations
import argparse,hashlib,json,math,random,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from round2.baseline_repair_20261111 import ngrf_repair as R
from round2.paper_baseline_training_v1 import common as C
from project_paths import PATHS

OUT=R.REPORT/"dropout_diagnostic"
ART=R.ART/"dropout_diagnostic"
def rng_save():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all())
def rng_hash(s):
    n=s["numpy"]
    return dict(python=hashlib.sha256(repr(s["python"]).encode()).hexdigest(),
        numpy=hashlib.sha256(repr((n[0],n[2],n[3],n[4])).encode()+n[1].tobytes()).hexdigest(),
        cpu=R.tensor_hash(s["cpu"]),cuda=[R.tensor_hash(x) for x in s["cuda"]])
def rng_restore(s):
    random.setstate(s["python"]);np.random.set_state(s["numpy"]);torch.set_rng_state(s["cpu"])
    torch.cuda.set_rng_state_all(s["cuda"])
def metrics(p,t):
    p=np.asarray(p,np.complex128);t=np.asarray(t,np.complex128)
    energy=float(np.sum(np.abs(t)**2));pe=float(np.sum(np.abs(p)**2))
    error=float(np.sum(np.abs(p-t)**2))
    if energy==0:return dict(complex_nmse=None,signed_complex_correlation=None,weighted_phase_cosine=None,
        defined_phase_target_energy_fraction=None,prediction_mean_power_raw=pe/p.size)
    defined=(np.abs(p)>0)&(np.abs(t)>0);weight=np.abs(t[defined])**2
    phase=np.real(p[defined]*t[defined].conj())/(np.abs(p[defined])*np.abs(t[defined]))
    return dict(complex_nmse=error/energy,prediction_to_truth_energy=pe/energy,
        signed_complex_correlation=float(np.real(np.sum(p.conj()*t))/math.sqrt(pe*energy)) if pe else None,
        weighted_phase_cosine=float(np.sum(weight*phase)/weight.sum()) if weight.sum() else None,
        defined_phase_target_energy_fraction=float(weight.sum()/energy))
@torch.no_grad()
def forward(model,data,rows,device):
    value=model(torch.tensor(data.positions(rows),device=device,dtype=torch.float64),
        torch.tensor(data.sites(rows),device=device,dtype=torch.long),
        torch.tensor(data.frequencies(),device=device,dtype=torch.float64))
    assert value.shape==(16,256,4,192) and value.dtype==torch.complex64 and torch.isfinite(value).all()
    return (value.cpu().numpy()*data.scale).astype(np.complex64)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--gpu-uuid",required=True);ap.add_argument("--draws",type=int,default=8)
    args=ap.parse_args();assert 2<=args.draws<=8
    OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    free=int(subprocess.check_output(["nvidia-smi","-i",args.gpu_uuid,"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
    if free<24*1024:raise RuntimeError("Requires24GiB free on the assigned allowed UUID")
    gpu=C.device_guard(args.gpu_uuid);device=torch.device("cuda:0")
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    done=json.loads((R.REPORT/"paired/results.json").read_text())
    endpoint=next(x for x in done["variants"] if x["weight"]==1.)
    protected=[PATHS.root/endpoint["checkpoint"],R.ART/"paired/initial.pt",
        R.ART/"paired/weight_0p1/latest.pt",R.ART/"paired/weight_1p0/latest.pt",Path(R.__file__)]
    file_before={PATHS.relative(p):C.sha256(p) for p in protected}
    assert C.sha256(PATHS.root/endpoint["checkpoint"])==endpoint["checkpoint_sha256"]
    data=C.FoldData(0,"screen","ngrf");rows=R.PANEL;truth=data.read_fit(rows,normalized=False)
    model=R.model_build(data,device)
    model.load_state_dict(torch.load(PATHS.root/endpoint["checkpoint"],map_location=device,weights_only=True)["model"])
    model.eval();before_hash=R.model_hash(model)
    dropout=[dict(module=n,class_name=type(x).__name__,p=float(x.p),inplace=bool(x.inplace))
        for n,x in model.named_modules() if isinstance(x,torch.nn.modules.dropout._DropoutNd)]
    eval_pred=forward(model,data,rows,device)
    saved=np.load(PATHS.root/endpoint["raw_prediction"],allow_pickle=False)
    replay=float(np.linalg.norm((eval_pred-saved).reshape(-1))/max(np.linalg.norm(saved.reshape(-1)),1e-30))
    assert replay<=1e-5,("same checkpoint full-batch eval replay",replay)
    C.seed_all(217);saved_rng=rng_save();before_rng_hash=rng_hash(saved_rng)
    n=saved_rng["numpy"]
    C.atomic_torch_save(ART/"rng_before.pt",dict(python_rng=saved_rng["python"],
        numpy_rng=(n[0],torch.tensor(n[1].astype(np.int64)),int(n[2]),int(n[3]),float(n[4])),
        cpu_rng=saved_rng["cpu"],cuda_rng=saved_rng["cuda"]))
    started=time.monotonic();samples=[]
    try:
        model.train()
        for i in range(args.draws):
            value=forward(model,data,rows,device);samples.append(value)
            print(json.dumps(dict(draw=i+1,total=args.draws,elapsed_seconds=time.monotonic()-started,
                pooled=metrics(value,truth))),flush=True)
            if time.monotonic()-started>120:raise RuntimeError("Bounded dropout probe exceeded120seconds")
    finally:
        model.eval();rng_restore(saved_rng)
    after_rng_hash=rng_hash(rng_save());after_hash=R.model_hash(model)
    assert after_rng_hash==before_rng_hash and after_hash==before_hash
    file_after={PATHS.relative(p):C.sha256(p) for p in protected};assert file_before==file_after
    samples=np.stack(samples);z=samples.astype(np.complex128);mean=z.mean(0)
    # Population variance makes E||X-T||² = ||E X-T||² + Var(X) exact for these draws.
    variance=np.mean(np.abs(z-mean[None])**2,axis=0)
    deviation=np.mean(np.abs(z-eval_pred.astype(np.complex128)[None])**2,axis=0)
    bias=np.abs(mean-eval_pred)**2
    assert np.allclose(deviation,variance+bias,rtol=1e-10,atol=1e-30)
    te=np.sum(np.abs(truth.astype(np.complex128))**2,axis=(1,2,3));sites=data.sites(rows)
    per=[]
    for i,row in enumerate(rows):
        e=float(te[i]);ve=float(variance[i].sum());de=float(deviation[i].sum());be=float(bias[i].sum())
        draw_metrics=[metrics(x[i],truth[i]) for x in z]
        per.append(dict(row_id=int(row),site=int(sites[i]),target_raw_energy=e,
            eval_metrics=metrics(eval_pred[i],truth[i]),stochastic_mean_metrics=metrics(mean[i],truth[i]),
            training_draw_metrics=draw_metrics,
            stochastic_NMSE_mean=float(np.mean([x["complex_nmse"] for x in draw_metrics])) if e else None,
            variance_to_truth_energy=ve/e if e else None,
            unbiased_variance_to_truth_energy=ve*args.draws/(args.draws-1)/e if e else None,
            stochastic_deviation_from_eval_to_truth_energy=de/e if e else None,
            stochastic_mean_bias_from_eval_to_truth_energy=be/e if e else None,
            variance_mean_power_raw=ve/variance[i].size,
            stochastic_deviation_from_eval_mean_power_raw=de/variance[i].size,
            variance_mean_power_FIT_RMS_squared=ve/variance[i].size/data.scale**2))
    energy=float(te.sum());C.atomic_npy(ART/"train_raw_H_draws.npy",samples)
    C.atomic_npy(ART/"eval_raw_H.npy",eval_pred);C.atomic_npy(ART/"row_ids.npy",rows)
    result=dict(status="complete",kind="fixed-lambda1-checkpoint dropout-only no-update probe",
        checkpoint=endpoint["checkpoint"],checkpoint_sha256=endpoint["checkpoint_sha256"],
        source_sha256=C.sha256(Path(__file__)),training_source_identity=endpoint["source_identity"],
        full_H_shape=[16,256,4,192],draws=args.draws,diagnostic_seed=217,
        dropout_modules=dropout,exact_training_batch_semantics=False,
        scope="8 train-mode full192-tone forwards, same16FIT/full256x4ports; actual training drew8tones/step. Attribute masks shared across all positions at same site, decoder masks vary by Gaussian/frequency chunk.",
        eval_pooled=metrics(eval_pred,truth),stochastic_mean_pooled=metrics(mean,truth),
        pooled_mean_train_draw_NMSE=float(np.mean([metrics(x,truth)["complex_nmse"] for x in z])),
        pooled_variance_to_truth_energy=float(variance.sum()/energy),
        pooled_stochastic_deviation_from_eval_to_truth_energy=float(deviation.sum()/energy),
        points=per,model_state_hash_before=before_hash,model_state_hash_after=after_hash,
        RNG_hash_before=before_rng_hash,RNG_hash_after=after_rng_hash,
        protected_file_hashes_before=file_before,protected_file_hashes_after=file_after,
        restored_training_mode="eval",optimizer_created=False,backward_called=False,parameter_updates=False,
        full_batch_eval_replay_relative_L2=replay,validation_labels_read=False,outer_truth_read=False,
        elapsed_seconds=time.monotonic()-started,device=gpu,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        sample_predictions=PATHS.relative(ART/"train_raw_H_draws.npy"),
        sample_predictions_sha256=C.sha256(ART/"train_raw_H_draws.npy"),
        interpretation_limit="Eight Monte Carlo draws estimate stochastic variance; no convergence, generalization, or training intervention is claimed.")
    C.atomic_json(OUT/"results.json",result)
    print(json.dumps(dict(status="complete",eval=result["eval_pooled"],train_NMSE_mean=result["pooled_mean_train_draw_NMSE"],
        variance_to_truth=result["pooled_variance_to_truth_energy"],files_and_model_and_RNG_unchanged=True)),flush=True)
if __name__=="__main__":
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/"failure.log").open("a") as f:f.write(traceback.format_exc()+"\n")
        raise

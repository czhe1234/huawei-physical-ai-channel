"""Fixed nGRF Gaussian spatial-weight coverage; no learning or validation."""
from __future__ import annotations
import argparse,json,math,subprocess
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from round2.paper_baseline_training_v1 import common as C
from round2.baseline_repair_20261111 import ngrf_repair as R
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--gpu-uuid",required=True);args=ap.parse_args()
    out=R.REPORT/"spatial_diagnostic";art=R.ART/"spatial_diagnostic"
    out.mkdir(parents=True,exist_ok=True);art.mkdir(parents=True,exist_ok=True)
    free=int(subprocess.check_output(["nvidia-smi","-i",args.gpu_uuid,"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
    assert free>=24*1024
    gpu=C.device_guard(args.gpu_uuid);device=torch.device("cuda:0");torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    final=json.loads((R.REPORT/"paired/results.json").read_text())
    entry=next(x for x in final["variants"] if x["weight"]==1.)
    assert C.sha256(PATHS.root/entry["checkpoint"])==entry["checkpoint_sha256"]
    data=C.FoldData(0,"screen","ngrf");rows=R.PANEL;sites=data.sites(rows)
    model=R.model_build(data,device)
    model.load_state_dict(torch.load(PATHS.root/entry["checkpoint"],map_location=device,weights_only=True)["model"])
    model.eval();before=R.model_hash(model)
    before_rng=dict(cpu=R.tensor_hash(torch.get_rng_state()),cuda=R.tensor_hash(torch.cuda.get_rng_state()))
    summaries=[]
    with torch.no_grad():
        for site,field in model.fields.items():
            index=np.flatnonzero(sites==int(site));selected=rows[index]
            positions=torch.tensor(data.positions(selected),device=device,dtype=torch.float64)
            latent,activation=field.native.get_attributes_and_activation(field.fixed_tx.float())
            _,inverse=field.native.get_covariance(return_inverse=True,eps=1e-10)
            weights=field.weights(positions,activation,inverse).cpu().numpy().astype(np.float64)
            assert np.isfinite(weights).all() and (weights>=0).all()
            xyz=field.native.get_xyz.detach().cpu().numpy().astype(np.float64)+field.coordinate_center.cpu().numpy()
            distances=np.linalg.norm(data.positions(selected)[:,None]-xyz[None],axis=-1)
            sv=np.linalg.svd(weights,compute_uv=False);rank=int(np.sum(sv>sv[0]*1e-6))
            norms=np.linalg.norm(weights,axis=1)
            normalized=weights/np.maximum(norms[:,None],1e-300)
            correlation=normalized@normalized.T
            # Full row rank only certifies an unconstrained per-Gaussian complex readout.
            target=data.read_fit(selected)
            flat=target.reshape(len(selected),-1).astype(np.complex128)
            coefficients=np.linalg.pinv(weights,rcond=1e-6)@flat
            reconstruction=weights@coefficients
            unconstrained_residual=float(np.sum(np.abs(reconstruction-flat)**2)/np.sum(np.abs(flat)**2))
            C.atomic_npy(art/f"site{site}_weights.npy",weights)
            summaries.append(dict(site=int(site),rows=selected.tolist(),gaussian_count=weights.shape[1],
                rank=rank,singular_values=sv.tolist(),condition=float(sv[0]/sv[-1]),
                row_cosine=correlation.tolist(),unconstrained_complex_readout_residual_nmse=unconstrained_residual,
                points=[dict(row_id=int(selected[i]),weight_sum=float(weights[i].sum()),
                    weight_L2=float(norms[i]),maximum_weight=float(weights[i].max()),
                    effective_count=float(weights[i].sum()**2/max(np.sum(weights[i]**2),1e-300)),
                    relative_support_count=int(np.sum(weights[i]>=weights[i].max()*1e-4)),
                    nearest_gaussian_distance_m=float(distances[i].min()),
                    maximum_row_cosine_with_another=float(np.max(np.delete(correlation[i],i)))) for i in range(len(index))]))
    after=R.model_hash(model)
    after_rng=dict(cpu=R.tensor_hash(torch.get_rng_state()),cuda=R.tensor_hash(torch.cuda.get_rng_state()))
    assert before==after and before_rng==after_rng
    result=dict(status="complete",source_sha256=C.sha256(Path(__file__)),checkpoint=entry["checkpoint"],
        checkpoint_sha256=entry["checkpoint_sha256"],sites=summaries,
        parameter_updates=False,optimizer_created=False,validation_labels_read=False,outer_truth_read=False,
        actual_state_hash_before=before,actual_state_hash_after=after,RNG_before=before_rng,RNG_after=after_rng,
        device=gpu,scope="8 fixed FIT locations per site; current positive Gaussian weights at lambda1 endpoint",
        unconstrained_projection_note="FIT-only representation diagnostic, not model predictions or deployable scores. Free Gaussian coefficients may violate the native decoder constraints. Never used to replace emitted H.",
        conclusion_limit="Weight rank excludes a singular spatial summation basis only; it does not certify the nonlinear contribution decoder, optimization or generalization.")
    C.atomic_json(out/"results.json",result)
    print(json.dumps(dict(status="complete",sites=[{k:v for k,v in x.items() if k not in ["points","row_cosine"]} for x in summaries])),flush=True)
if __name__=="__main__":main()

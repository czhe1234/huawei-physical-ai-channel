"""Read-only CPU FIT diagnostics for completed native nGRF refit models."""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from round2.paper_baseline_training_v1.common import FoldData,atomic_json,sha256
from round2.paper_baseline_training_v1.native_ngrf import NativeNGRF


def run(outer):
    torch.set_num_threads(4);started=time.monotonic()
    data=FoldData(outer,"refit","ngrf")
    checkpoint=PATHS.artifacts/"round2/paper_four_baselines_v1/ngrf"/f"outer{outer}/refit/complete.pt"
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model=NativeNGRF(data.geometry,data.positions(data.fit_ids),data.sites(data.fit_ids)).eval()
    model.load_state_dict(ck["model"])
    descriptor=PATHS.artifacts/"round2/pre_sionna/round2_train_spectral_descriptors.npz"
    with np.load(descriptor,allow_pickle=False) as z:energy=np.asarray(z["total_power"],np.float64)
    selected=[];quantiles=[.05,.25,.5,.75,.95,.99,1.]
    for s in (0,1):
        ids=data.fit_ids[data.sites(data.fit_ids)==s];ids=ids[energy[ids]>0]
        ids=ids[np.argsort(energy[ids],kind="stable")]
        for q in quantiles:selected.append(dict(site=s,quantile=q,row=int(ids[min(len(ids)-1,int(round(q*(len(ids)-1))))])))
    ids=np.array([x["row"] for x in selected],np.int64)
    frequency=torch.tensor(data.frequencies(),dtype=torch.float64)
    positions=torch.tensor(data.positions(ids),dtype=torch.float64);sites=torch.tensor(data.sites(ids),dtype=torch.int64)
    with torch.no_grad():pred=model(positions,sites,frequency)
    truth=torch.tensor(data.read_fit(ids))
    se=(pred-truth).abs().square().sum((1,2,3),dtype=torch.float64)
    te=truth.abs().square().sum((1,2,3),dtype=torch.float64)
    pe=pred.abs().square().sum((1,2,3),dtype=torch.float64)
    rows=[]
    for i,spec in enumerate(selected):rows.append(dict(**spec,nmse=float(se[i]/te[i]),prediction_to_truth_energy=float(pe[i]/te[i]),
      normalized_truth_mean_energy=float(te[i]/truth[i].numel()),normalized_prediction_mean_energy=float(pe[i]/truth[i].numel())))
    activation={}
    with torch.no_grad():
        for s,field in model.fields.items():
            latent,act=field.native.get_attributes_and_activation(field.fixed_tx.float());scale=field.native.get_scaling
            activation[s]=dict(mean=float(act.mean()),min=float(act.min()),max=float(act.max()),
               logits_L1=float(field.activation_penalty()),positive_output_gain=float(field.log_output_gain.exp()),
               Gaussian_scale_quantiles_m=torch.quantile(scale.flatten(),torch.tensor([0.,.25,.5,.75,1.])).tolist())
    # One strong and one median FIT point per site, four actual frequency tones.
    grad_ids=ids[[2,4,9,11]];ff=np.array([0,64,128,191],np.int64)
    target=torch.tensor(data.read_fit(grad_ids,ff));model.zero_grad(set_to_none=True)
    value=model(torch.tensor(data.positions(grad_ids),dtype=torch.float64),torch.tensor(data.sites(grad_ids),dtype=torch.int64),torch.tensor(data.frequencies(ff),dtype=torch.float64))
    loss=(value-target).abs().square().mean()
    params=list(model.named_parameters());grad=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True)
    groups={}
    for (name,param),g in zip(params,grad):
        if "attribute_network" in name:group=name.split(".")[1]+"/attribute_network"
        elif "contribution_decoder" in name:group=name.split(".")[1]+"/contribution_decoder"
        else:group=name
        record=groups.setdefault(group,dict(squared_norm=0.,elements=0,zero_elements=0,disconnected=0,finite=True))
        if g is None:record["disconnected"]+=param.numel();continue
        record["squared_norm"]+=float(g.double().square().sum());record["elements"]+=g.numel();record["zero_elements"]+=int((g==0).sum());record["finite"]=record["finite"] and bool(torch.isfinite(g).all())
    for value in groups.values():value["norm"]=value.pop("squared_norm")**.5
    out=dict(method="nGRF full-H adaptation",outer=outer,checkpoint=str(checkpoint.relative_to(PATHS.root)),checkpoint_sha256=sha256(checkpoint),
      selected_screen_updates=ck["info"]["selected_screen_updates"],refit_updates=ck["info"]["completed_steps"],
      role="FIT-only read-only diagnostic",frequencies="all192 actual frequencies",port_count=1024,rows=rows,
      aggregate=dict(nmse=float(se.sum()/te.sum()),prediction_to_truth_energy=float(pe.sum()/te.sum()),selected_position_count=len(ids)),
      activation=activation,gradient=dict(FIT_ids=grad_ids.tolist(),frequency_ids=ff.tolist(),mse=float(loss),groups=groups),
      parameter_updates=False,optimizer_created=False,outer_truth_read=False,training_config_changed=False,
      FIT_energy_descriptor=str(descriptor.relative_to(PATHS.root)),scale=data.scale,seconds=time.monotonic()-started,
      peak_CPU_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report=PATHS.analysis/"results/round2_paper_four_baselines_v1/ngrf"/f"outer{outer}/refit_FIT_check.json"
    atomic_json(report,out);print(json.dumps(dict(outer=outer,aggregate=out["aggregate"],activation=activation,seconds=out["seconds"])),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--outers",default="0,1");args=ap.parse_args()
    for outer in map(int,args.outers.split(",")):run(outer)

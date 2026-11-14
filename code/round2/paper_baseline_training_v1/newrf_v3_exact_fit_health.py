"""CPU diagnostic on exact scalar labels previously used by a v3 checkpoint."""
import json
import numpy as np
import torch
from .common import FoldData,seed_all,atomic_json,sha256
from .newrf_runner import GEOMETRY,FitGeometryBank,FitDirectionCache,subset_forward,scalar_target
from .newrf_conditioned_v3 import build_conditioned_model,output_paths_v3
from .newrf_query import subset_query

def main():
    torch.set_num_threads(4)
    data=FoldData(0,"screen",method="newrf")
    path=output_paths_v3("newrf",0,"screen")["artifact"]/"best.pt"
    ck=torch.load(path,map_location="cpu",weights_only=False)
    if ck["config"]["schema"]!="newrf-main-training-config-v3":raise ValueError("Not v3")
    cfg=ck["config"];rng=np.random.default_rng(cfg["seed"])
    latest={};first={}
    for step in range(1,ck["step"]+1):
        rows=rng.choice(data.fit_ids,size=cfg["batch_positions"],replace=True)
        field=(step-1)%8
        ss=np.sort(rng.choice(128,size=cfg["BS_spatial_samples"],replace=False))
        ff=np.sort(rng.choice(192,size=cfg["frequency_samples"],replace=False))
        batch=dict(step=step,rows=rows.copy(),spatial=ss,frequencies=ff)
        first.setdefault(field,batch);latest[field]=batch
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit")
    cache=FitDirectionCache(bank).to("cpu");seed_all(cfg["seed"])
    model=build_conditioned_model(data,cache);model.load_state_dict(ck["state_dict"],strict=True);model.eval()
    output={}
    with torch.no_grad():
        for panel,batches in [("first_training_batch_per_field",first),("last_training_batch_per_field",latest)]:
            records=[];totals={k:dict(error=0.,target_energy=0.,pred_energy=0.) for k in ("cached_FIT","query_prior")}
            for field,b in batches.items():
                rows=b["rows"];ss=b["spatial"];ff=b["frequencies"];p,u=divmod(field,4)
                t=scalar_target(data,rows,field,ss,ff,"cpu")
                h,hc=subset_forward(model,cache,data,rows,field,ss,ff)
                q=subset_query(model,data,rows,p*128+ss,np.array([u]),ff)[:,:,0]
                entry=dict(field=field,training_step=b["step"],rows=rows.tolist(),BS_spatial=ss.tolist(),
                           frequencies=ff.tolist(),target_energy=float(t.abs().square().sum()))
                for name,v in [("cached_FIT",h),("query_prior",q)]:
                    err=float((v-t).abs().square().sum());power=float(v.abs().square().sum());energy=entry["target_energy"]
                    entry[name]=dict(nmse=err/max(energy,1e-30),power_ratio=power/max(energy,1e-30),
                                     exact_zero_fraction=float((v.abs()==0).float().mean()))
                    totals[name]["error"]+=err;totals[name]["target_energy"]+=energy;totals[name]["pred_energy"]+=power
                records.append(entry)
            for v in totals.values():
                v["nmse"]=v["error"]/max(v["target_energy"],1e-30)
                v["power_ratio"]=v["pred_energy"]/max(v["target_energy"],1e-30)
            output[panel]=dict(batches=records,totals=totals)
    out=dict(checkpoint=str(path),checkpoint_sha256=sha256(path),step=ck["step"],CPU_only=True,
             exact_training_scalar_axes_replayed=True,results=output,outer_truth_reads=0,
             used_for_selection=False,complete_H_official_score=False)
    report=output_paths_v3("newrf")["report"]
    atomic_json(report/f"exact_trained_labels_health_step{ck['step']}.json",out);print(json.dumps(out),flush=True)
if __name__=="__main__":main()


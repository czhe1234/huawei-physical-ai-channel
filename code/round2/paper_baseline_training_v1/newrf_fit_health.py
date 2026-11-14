"""CPU-only saved-checkpoint FIT health; no training or outer labels."""
import json,math
from pathlib import Path
import numpy as np
import torch
from .common import FoldData,output_paths,atomic_json,seed_all,sha256,digest_ids
from .newrf_runner import GEOMETRY,FitGeometryBank,FitDirectionCache,subset_forward,scalar_target
from .newrf_latefreq import build_late_model
from .newrf_query import subset_query

def main():
    torch.set_num_threads(4)
    data=FoldData(0,"screen",method="newrf")
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit")
    cache=FitDirectionCache(bank)
    ckpath=output_paths("newrf",0,"screen")["artifact"]/"best.pt"
    ck=torch.load(ckpath,map_location="cpu",weights_only=False)
    seed_all(42);model=build_late_model(data,cache,128,128,frequency_chunk=8)
    model.load_state_dict(ck["state_dict"]);model.eval();cache.to("cpu")
    panel=[]
    for site in (0,1):
        ids=data.fit_ids[data.inputs.site_id[data.fit_ids]==site]
        order=np.lexsort((data.inputs.positions[ids,1],data.inputs.positions[ids,0]))
        panel.extend(ids[order[np.linspace(0,len(ids)-1,4,dtype=int)]].tolist())
    ss=np.linspace(0,127,8,dtype=np.int64);ff=np.array([0,63,127,191],np.int64)
    result=[];totals={k:dict(error=0.,energy=0.,pred_energy=0.) for k in ("cached_FIT","query_prior")}
    with torch.no_grad():
        for field,row in enumerate(panel):
            p,u=divmod(field,4);rows=np.array([row],np.int64)
            target=scalar_target(data,rows,field,ss,ff,"cpu")
            trained,_=subset_forward(model,cache,data,rows,field,ss,ff)
            query=subset_query(model,data,rows,p*128+ss,np.array([u]),ff)[:,:,0]
            entry=dict(field=field,row=row,rays=int(cache.count[row]) or 64,
                       truth_energy=float(target.abs().square().sum()))
            for key,pred in [("cached_FIT",trained),("query_prior",query)]:
                error=float((pred-target).abs().square().sum());power=float(pred.abs().square().sum())
                truth=entry["truth_energy"];total=totals[key]
                total["error"]+=error;total["energy"]+=truth;total["pred_energy"]+=power
                entry[key]=dict(nmse=error/max(truth,1e-30),pred_energy=power,
                                pred_over_truth_power=power/max(truth,1e-30),
                                fraction_exact_zero=float((pred.abs()==0).float().mean()))
            result.append(entry)
    for total in totals.values():
        total["nmse"]=total["error"]/max(total["energy"],1e-30)
        total["pred_over_truth_power"]=total["pred_energy"]/max(total["energy"],1e-30)
    out=dict(checkpoint=str(ckpath),checkpoint_sha256=sha256(ckpath),step=ck["step"],
             input_identity=data.inputs.identity,fit_ids_sha256=digest_ids(data.fit_ids),
             panel_basis="8 geometry-only sorted-position endpoints/quartiles, one per scalar field",
             rows=result,totals=totals,CPU_only=True,training=False,outer_truth_reads=0,
             used_for_frozen_model_selection=False,complete_H_or_official_score=False)
    atomic_json(output_paths("newrf")["report"]/f"FIT_health_checkpoint{ck['step']}.json",out)
    print(json.dumps(out),flush=True)
if __name__=="__main__":main()


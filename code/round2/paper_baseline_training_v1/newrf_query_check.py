"""CPU tensor-axis equivalence for the subset selector; no data or labels."""
import json
import numpy as np
import torch
from round2.paper_baseline_adapters_v1.newrf import NeWRFFullH,SyntheticRayProvider
from .newrf_query import subset_query
from .newrf_precision_probe import apply_streaming
from .common import output_paths,atomic_json

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    class Spec:
        frequencies_hz=6.69e9+(np.arange(192,dtype=np.float64)-96)*30000.
        bs_port_positions_m=np.array([[[[[s*2+h*.03,v*.07,2.] for v in range(8)]
            for h in range(16)] for p in range(2)] for s in range(2)],np.float64)
        def ue_world_positions(self,pos,site):
            off=torch.tensor([[0.,0.,-.01],[0.,0.,.01],[0.,0.,-.01],[0.,0.,.01]],
                             dtype=torch.float64,device=pos.device)
            return pos[:,None]+off[None]
    spec=Spec();provider=SyntheticRayProvider([[1.,0.,0.],[0.,1.,0.]])
    cfg=dict(n_layers=2,hidden_dim=16,skip=(),position_levels=2,direction_levels=1,
             n_samples=5,n_importance=3,far_m=4.,frequency_chunk=2,coordinate_scale_m=1.)
    model=NeWRFFullH(spec,provider,cfg,synthetic_only=True)
    model=apply_streaming(model,False)
    model._qualify=lambda:None
    class Data:
        def positions(self,ids):return np.array([[.6+i,.4,1.] for i in ids],np.float64)
        def sites(self,ids):return np.asarray(ids,dtype=np.int64)%2
        def frequencies(self,ff):return spec.frequencies_hz[ff]
    data=Data();ids=np.array([0,1]);bs=np.array([255,0,129,127,32])
    ue=np.array([3,0,2]);ff=np.array([191,1,97])
    with torch.no_grad():
        ref=model(data.positions(ids),data.sites(ids),data.frequencies(ff))
        ref=ref[:,bs][:,:,ue]
        got=subset_query(model,data,ids,bs,ue,ff)
    error=float((ref-got).abs().max())
    rel=float(torch.linalg.vector_norm(ref-got)/torch.linalg.vector_norm(ref).clamp_min(1e-30))
    if error>1e-7 or rel>1e-4:raise AssertionError((error,rel))
    result=dict(status="passed",artificial_only=True,full_and_subset_shape=list(got.shape),
                max_absolute_error=error,relative_L2=rel,BS_ids=bs.tolist(),UE_ids=ue.tolist(),
                frequency_ids=ff.tolist(),CSI_label_reads=0,geometry_cache_reads=0,
                no_training=True)
    atomic_json(output_paths("newrf")["report"]/"subset_query_CPU_check.json",result)
    print(json.dumps(result))
if __name__=="__main__":main()


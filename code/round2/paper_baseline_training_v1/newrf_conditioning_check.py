"""CPU-only density initialization audit of late-frequency NeWRF."""
import copy,json
import numpy as np
import torch
from .common import FoldData,output_paths,atomic_json,seed_all
from .newrf_runner import GEOMETRY,FitGeometryBank,FitDirectionCache
from .newrf_latefreq import build_late_model,stratified64

def raw_coordinate_condition(model,scale):
    for field in list(model.coarse)+list(model.fine):
        network=field.network;d=field.encoder.d_output
        with torch.no_grad():
            network.layers[0].weight[:,:6].mul_(scale)
            for layer in network.layers[1:]:
                if layer.in_features==field.cfg.hidden_dim+d:
                    layer.weight[:,field.cfg.hidden_dim:field.cfg.hidden_dim+6].mul_(scale)
    return model

def main():
    torch.set_num_threads(4)
    data=FoldData(0,"screen",method="newrf")
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit")
    cache=FitDirectionCache(bank).to("cpu")
    seed_all(42);initial=build_late_model(data,cache,128,128)
    trained=copy.deepcopy(initial)
    ck=torch.load(output_paths("newrf",0,"screen")["artifact"]/"best.pt",map_location="cpu",weights_only=False)
    trained.load_state_dict(ck["state_dict"])
    factor=9./426.831421
    conditioned=raw_coordinate_condition(copy.deepcopy(initial),factor)
    bias=copy.deepcopy(conditioned)
    with torch.no_grad():
        for field in list(bias.coarse)+list(bias.fine):field.network.alpha_out.bias.fill_(.1)
    rows=[1249,3564,1322,1676,496,733,2763,1432]
    result={}
    with torch.no_grad():
        for name,model in [("native_initial",initial),("trained10000",trained),
                           ("coordinate_conditioned_initial",conditioned),
                           ("coordinate_conditioned_bias01_initial",bias)]:
            entries=[]
            for field,row in enumerate(rows):
                p,u=divmod(field,4);pos=torch.tensor(data.positions([row]),dtype=torch.float64)
                site=torch.tensor(data.sites([row]),dtype=torch.long)
                origin=model.spec.ue_world_positions(pos,site)[0];tx=model.bs_ports[site][0]
                ss=torch.tensor([0]);directions=cache.scalar(model,row,field,ss,origin,site[0],tx)[0,:8]
                oo=origin[u].expand(len(directions),3)
                points,z=stratified64(oo,directions,cache.far_m,32)
                view=directions[:,None,:].expand_as(points);tt=tx[p,0].expand_as(points)
                f=torch.full((*points.shape[:-1],1),6.69e9,dtype=torch.float64)
                for stage,nets in [("coarse",model.coarse),("fine",model.fine)]:
                    raw=nets[field](points,view,tt,f)
                    rho=raw[...,-1];signal=raw[...,:2]
                    entries.append(dict(field=field,stage=stage,row=row,
                        density_quantiles=torch.quantile(rho,torch.tensor([0.,.05,.5,.95,1.])).tolist(),
                        positive_density_fraction=float((rho>0).float().mean()),
                        signal_abs_median=float(signal.abs().median()),
                        tanh_saturation_fraction=float((signal.abs()>3).float().mean())))
            result[name]=entries
    out=dict(CPU_only=True,training=False,outer_truth_reads=0,coordinate_condition_factor=factor,
             factor_source="native default far9m / public map maxabsXYZ426.831421m",
             only_raw_coordinate_weight_columns_scaled=True,PE_sinusoidal_inputs_unchanged=True,
             results=result)
    atomic_json(output_paths("newrf")["report"]/"initial_density_CPU_audit.json",out)
    print(json.dumps(out),flush=True)
if __name__=="__main__":main()


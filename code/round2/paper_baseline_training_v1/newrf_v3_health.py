"""v3 numerical invariants and bounded real nonzero-FIT fitting health."""
from __future__ import annotations
import argparse,copy,json,math,time,traceback
import numpy as np
import torch
from .common import FoldData,atomic_json,device_guard,seed_all
from .newrf_runner import (GPU_UUID,GEOMETRY,FitGeometryBank,FitDirectionCache,
                           subset_forward,scalar_target,weighted_loss,UniformRaySynthesizer)
from .newrf_latefreq import LateFrequencyField,stratified64
from .newrf_conditioned_v3 import (ConditionedConfig,ConditionedField,build_conditioned_model,
                                  output_paths_v3)
from .newrf_latefreq_pilot import equivalence

def numerical_checks():
    torch.manual_seed(375)
    cfg=ConditionedConfig(n_layers=3,hidden_dim=16,skip=(1,),position_levels=3,
         direction_levels=2,frequency_levels=3,n_samples=5,n_importance=3,far_m=551.)
    old=LateFrequencyField(cfg);new=ConditionedField(cfg);new.load_state_dict(old.state_dict())
    factor=cfg.raw_coordinate_scale_m
    with torch.no_grad():
        new.network.layers[0].weight[:,:6].mul_(factor)
        for layer in new.network.layers[1:]:
            if layer.in_features==cfg.hidden_dim+new.encoder.d_output:
                layer.weight[:,cfg.hidden_dim:cfg.hidden_dim+6].mul_(factor)
    pt=torch.randn(3,7,3,dtype=torch.float64)*50
    tx=torch.randn_like(pt)*50;view=torch.randn_like(pt);view/=view.norm(dim=-1,keepdim=True)
    freq=torch.full((3,7,1),6.69e9,dtype=torch.float64)
    a=old(pt,view,tx,freq);b=new.raw_forward(pt,view,tx,freq)
    a.square().sum().backward();b.square().sum().backward()
    ga=[];gb=[]
    for (name,p),(name2,q) in zip(old.named_parameters(),new.named_parameters()):
        g=q.grad.detach().clone()
        if name=="network.layers.0.weight":g[:,:6].mul_(factor)
        if ".weight" in name and name.startswith("network.layers.") and p.shape[1]==cfg.hidden_dim+new.encoder.d_output:
            g[:,cfg.hidden_dim:cfg.hidden_dim+6].mul_(factor)
        ga.append(p.grad.reshape(-1));gb.append(g.reshape(-1))
    ga=torch.cat(ga);gb=torch.cat(gb)
    coordinate=dict(raw_output_relative_L2=float((a-b).norm()/a.norm()),
                    chain_gradient_relative_L2=float((ga-gb).norm()/ga.norm()),
                    identity_channels_scaled_only=True,physical_PE_sin_cos_exact=True)
    if coordinate["raw_output_relative_L2"]>1e-5 or coordinate["chain_gradient_relative_L2"]>1e-4:
        raise AssertionError(coordinate)
    # The integrator gets the same strictly positive physical rho; prove its
    # tensor-level output and the smooth reparameterization's chain derivative.
    z=torch.linspace(1e-10,551.,13,dtype=torch.float64).expand(6,13)
    rho=torch.full((6,13),1./551.,dtype=torch.float64,requires_grad=True)
    latent=torch.log(torch.expm1(rho.detach()*551.)).requires_grad_(True)
    signal=torch.randn(6,13,2,dtype=torch.float64)*.1
    raw=torch.cat((signal,rho[...,None]),-1)
    raw2=torch.cat((signal,(torch.nn.functional.softplus(latent)/551.)[...,None]),-1)
    synth=UniformRaySynthesizer();f=torch.full((6,1),6.69,dtype=torch.float64)
    h=synth.synthesize(raw,z,f,[3,3])[0];h2=synth.synthesize(raw2,z,f,[3,3])[0]
    gr=torch.autograd.grad(h.abs().square().sum(),rho)[0]
    gl=torch.autograd.grad(h2.abs().square().sum(),latent)[0]
    expected=gr*torch.sigmoid(latent.detach())/551.
    integral=dict(H_relative_L2=float((h-h2).norm()/h.norm()),
                  chain_gradient_relative_L2=float((gl-expected).norm()/expected.norm()),
                  zero_density_is_limit=True,old_ReLU_network_weight_equivalence_claimed=False)
    if integral["H_relative_L2"]>1e-12 or integral["chain_gradient_relative_L2"]>1e-10:raise AssertionError(integral)
    return dict(coordinate_linear_reparameterization=coordinate,physical_density_integrator=integral)

def optical(model,cache,data,row):
    rows=np.array([row],np.int64);pos=torch.tensor(data.positions(rows),dtype=torch.float64,device="cuda:0")
    site=torch.tensor(data.sites(rows),dtype=torch.long,device="cuda:0")
    origins=model.spec.ue_world_positions(pos,site)[0];tx=model.bs_ports[site][0]
    ss=torch.tensor([0],dtype=torch.long,device="cuda:0");out=[]
    with torch.no_grad():
        for field in range(8):
            p,u=divmod(field,4);dirs=cache.scalar(model,row,field,ss,origins,site[0],tx)[0]
            origin=origins[u].expand(len(dirs),3);point,z=stratified64(origin,dirs,model.config.far_m,128)
            view=dirs[:,None].expand_as(point);t=tx[p,0].expand_as(point)
            dz=torch.cat((z[...,1:]-z[...,:-1],torch.zeros_like(z[...,:1])),-1)
            for stage,nets in [("coarse",model.coarse),("fine",model.fine)]:
                _,rho=nets[field].prepare(point,view,t);rho=rho[...,0]
                depth=(rho*dz).sum(-1);first=1-torch.exp(-rho[:,0]*dz[:,0])
                out.append(dict(field=field,stage=stage,positive_density_fraction=float((rho>0).float().mean()),
                       optical_depth_quantiles=torch.quantile(depth,torch.tensor([0.,.5,1.],device="cuda:0",dtype=torch.float64)).tolist(),
                       first_alpha_max=float(first.max()),first_alpha_median=float(first.median())))
    return out

def main(args):
    torch.set_num_threads(4);seed_all(42);report=output_paths_v3("newrf",0,args.tag)["report"]
    if (report/"results.json").exists():raise FileExistsError("Preserve v3 pilot")
    gold=numerical_checks();atomic_json(report/"CPU_numerical_checks.json",gold)
    data=FoldData(0,"screen",method="newrf")
    bank=FitGeometryBank(GEOMETRY,data.inputs,outer=0,role="screen_fit");cache=FitDirectionCache(bank)
    seed_all(42);model=build_conditioned_model(data,cache)
    guard=device_guard(GPU_UUID);torch.backends.cuda.matmul.allow_tf32=False
    model=model.to("cuda:0");cache.to("cuda:0")
    rows=np.array([350],np.int64);ss=np.linspace(0,127,8,dtype=np.int64);ff=np.array([0,63,127,191],np.int64)
    targets=[scalar_target(data,rows,f,ss,ff,"cuda:0") for f in range(8)]
    initial_optical=optical(model,cache,data,350);atomic_json(report/"initial_optical.json",initial_optical)
    GPUeq=equivalence(model,None,None,None,None,targets[0],data.scale,
              train_callback=lambda m:subset_forward(m,cache,data,rows,0,ss,ff))
    atomic_json(report/"GPU_naive_shared_equivalence.json",GPUeq);model.zero_grad(set_to_none=True)
    def diagnostic():
        out=[]
        with torch.no_grad():
            for field in range(8):
                h,hc=subset_forward(model,cache,data,rows,field,ss,ff);t=targets[field]
                out.append(dict(field=field,nmse=float((h-t).abs().square().sum()/t.abs().square().sum()),
                         coarse_nmse=float((hc-t).abs().square().sum()/t.abs().square().sum()),
                         power_ratio=float(h.abs().square().sum()/t.abs().square().sum()),
                         zero_fraction=float((h.abs()==0).float().mean())))
        return out
    initial=diagnostic();curve=[];zero=[0]*8;first_grad=[]
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-4,weight_decay=0)
    start=time.monotonic()
    for step in range(args.steps):
        field=step%8;h,hc=subset_forward(model,cache,data,rows,field,ss,ff)
        loss,_,_=weighted_loss(h,hc,targets[field],data.scale)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if float(gn)==0:zero[field]+=1
        if step<8:first_grad.append(dict(field=field,norm=float(gn)))
        if not math.isfinite(float(loss)) or not math.isfinite(float(gn)):raise FloatingPointError("v3 health nonfinite")
        optimizer.step()
        if (step+1)%512==0:
            diag=diagnostic();curve.append(dict(step=step+1,fields=diag))
            atomic_json(report/"progress.json",curve[-1]);print(json.dumps(curve[-1]),flush=True)
            if all(v["nmse"]<.05 for v in diag):break
        if time.monotonic()-start>260:break
    final=diagnostic()
    passed=all(v["nmse"]<.8 and v["power_ratio"]>.01 and v["zero_fraction"]<1 for v in final)
    result=dict(status="passed" if passed else "fit_health_failed",optimizer_steps=step+1,
       wall_seconds=time.monotonic()-start,CPU_numerical=gold,GPU_equivalence=GPUeq,
       initialization_optical=initial_optical,final_optical=optical(model,cache,data,350),
       first8_gradient=first_grad,initial=initial,curve=curve,final=final,
       zero_gradient_batches_by_field=zero,GPU=guard,training_health_only=True,
       row=350,BS_ids=ss.tolist(),frequency_ids=ff.tolist(),
       validation_truth_reads=0,outer_truth_reads=0,not_a_formal_checkpoint=True)
    atomic_json(report/"results.json",result);print(json.dumps(result),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--steps",type=int,default=10000);p.add_argument("--tag",default="single_FIT_health")
    args=p.parse_args()
    try:main(args)
    except BaseException:
        atomic_json(output_paths_v3("newrf",0,args.tag)["report"]/"failure.json",dict(traceback=traceback.format_exc()))
        raise


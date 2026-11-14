"""Cheap WRF-GS+ batched-core equivalence checks, without channel labels."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from round2.paper_baseline_adapters_v1.contract import SimulationInputs
from round2.paper_baseline_adapters_v1.wrfgs import _alpha_render
from round2.paper_baseline_training_v1.wrfgs_model import (
    BatchedWRFGS,alpha_render_batch,site_forward_reference,POWER_GAIN,
)
from round2.paper_baseline_training_v1.common import atomic_json,REPORT,sha256


def check():
    torch.set_num_threads(2);torch.manual_seed(2031)
    data=SimulationInputs();geom=data.geometry
    offsets=np.array([[3.,1.,1.],[10.,5.,2.],[-5.,8.,3.],[7.,-6.,-1.]])
    initial=[geom.bs_positions_m[s]+offsets for s in (0,1)]
    model=BatchedWRFGS(geom,data.positions[data.roles(0)["screen_fit"]],count=4,initial=initial,z_scale=10.)
    positions=torch.tensor(np.array(data.positions[[0,2,4]]),dtype=torch.float32)
    sites=torch.tensor(np.array(data.site_id[[0,2,4]]),dtype=torch.long)
    fids=torch.tensor([0,96,191])
    normf=((model.frequencies_hz[fids]-model.frequency_center)/model.frequency_scale).float()
    p=model.forward_pairs(positions,sites,fids)
    q=torch.stack([site_forward_reference(model.fields[int(s)],x,f) for x,s,f in zip(positions,sites,normf)])
    maxerr=float((p-q).abs().max())
    torch.testing.assert_close(p,q,atol=1e-5,rtol=2e-5)
    old=torch.stack([model.fields[int(s)](x,f)*POWER_GAIN for x,s,f in zip(positions,sites,normf)])
    torch.testing.assert_close(p,old,atol=1e-5,rtol=2e-5)
    for field in model.fields:
        torch.nn.init.normal_(field.deformation.gaussian_warp.weight,std=.0001)
    model.zero_grad(set_to_none=True)
    p=model.forward_pairs(positions,sites,fids);p.square().mean().backward()
    grad1={k:v.grad.detach().clone() for k,v in model.named_parameters() if v.grad is not None}
    model.zero_grad(set_to_none=True)
    q=torch.stack([site_forward_reference(model.fields[int(s)],x,f) for x,s,f in zip(positions,sites,normf)])
    q.square().mean().backward()
    torch.testing.assert_close(p,q,atol=1e-5,rtol=2e-5)
    maxgrad=0.
    for k,v in model.named_parameters():
        if k in grad1:
            maxgrad=max(maxgrad,float((v.grad-grad1[k]).abs().max()))
            torch.testing.assert_close(v.grad,grad1[k],atol=1e-5,rtol=3e-4)
    k=4
    pixel=torch.randn(k,2,requires_grad=True)
    cov=(torch.eye(2)[None].repeat(k,1,1)*.7).requires_grad_()
    depth=torch.arange(k,dtype=torch.float32)+2
    opacity=torch.full((k,),.1,requires_grad=True)
    color=torch.rand(k,8,requires_grad=True)
    a=_alpha_render(pixel,cov,depth,opacity,color,8,4)
    b=alpha_render_batch(pixel[None],cov[None],depth[None],opacity,color[None],8,4)[0]
    torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
    assert p.shape==(3,256,4) and torch.isfinite(p).all() and (p>=0).all()
    result=dict(status="passed",maximum_pair_error=maxerr,maximum_gradient_error=maxgrad,
      zero_warp_matches_old=True,alpha_matches_frozen=True,shape=list(p.shape),
      restored_warp_nonzero_checked=True,read_CSI=False,optimizer_steps=0,quality_scored=False,
      model_source_sha256=sha256(Path(__file__).with_name("wrfgs_model.py")))
    atomic_json(REPORT/"wrfgs"/"cpu_equivalence.json",result)
    print(json.dumps(result))
    return result

if __name__=="__main__":check()

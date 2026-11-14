"""Author-based WRF-GS+ power adapter with a differentiable CPU reference renderer.

Pinned original deformation network; spherical projection, SH color and
front-to-back alpha blending follow the author's CUDA equations. The adapter
maps the learned angular field to the common physical32-beam codebook.
This is a measured-task adaptation, not an unchanged CUDA reproduction.
"""
from __future__ import annotations
import ast
from pathlib import Path
import numpy as np
import torch
from torch import nn
from project_paths import PATHS
from .common import C0, check_prediction
from .metrics import codebook

AUTHOR=PATHS.code_root/"third_party/paper_baselines/wrf_gsplus"

def author_network_class():
    source=(AUTHOR/"utils/pos_utils.py").read_text()
    tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    import sys,types
    name="hall_wrf_author_deformation"
    if name not in sys.modules:
        module=types.ModuleType(name);module.__file__=str(AUTHOR/"utils/pos_utils.py")
        module.__dict__.update(torch=torch,nn=nn,F=nn.functional)
        sys.modules[name]=module
        exec(compile(ast.Module(body=nodes,type_ignores=[]),module.__file__,"exec"),module.__dict__)
    return sys.modules[name].DeformNetwork

def author_eval_sh():
    scope={"__name__":"hall_wrf_author_sh"}
    exec(compile((AUTHOR/"utils/sh_utils.py").read_text(),str(AUTHOR/"utils/sh_utils.py"),"exec"),scope)
    return scope["eval_sh"]

def quaternion_matrix(q):
    q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    r,x,y,z=q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-r*z),2*(x*z+r*y),
                        2*(x*y+r*z),1-2*(x*x+z*z),2*(y*z-r*x),
                        2*(x*z-r*y),2*(y*z+r*x),1-2*(x*x+y*y)],-1).reshape(-1,3,3)

def project_gaussians(means,covariance,center,camera_basis,width,height):
    t=(means-center)@camera_basis
    x,y,z=t.unbind(-1);rho2=x*x+z*z;rho=rho2.sqrt();rr=rho2+y*y
    px=(torch.atan2(x,z)/torch.pi+1)*width/2-.5
    py=(torch.asin((y/rr.sqrt().clamp_min(1e-8)).clamp(-.999999,.999999))*2/torch.pi+1)*height/2-.5
    factor_x=width/(2*torch.pi);factor_y=height/torch.pi
    j=torch.stack([factor_x*z/(rho2+1e-7),torch.zeros_like(x),-factor_x*x/(rho2+1e-7),
        -factor_y*x*y/(rho+1e-7)/(rr+1e-7),factor_y*rho/(rr+1e-7),
        -factor_y*z*y/(rho+1e-7)/(rr+1e-7)],-1).reshape(-1,2,3)
    cov_camera=camera_basis.T@covariance@camera_basis
    cov2=j@cov_camera@j.transpose(-1,-2)+.3*torch.eye(2,dtype=means.dtype,device=means.device)
    return torch.stack([px,py],-1),cov2,rr.sqrt()

def alpha_render(pixel,cov2,depth,opacity,color,width,height):
    """Dense reference of forward.cu; no CUDA/tile-pruning equivalence claim."""
    yy,xx=torch.meshgrid(torch.arange(height,device=pixel.device,dtype=pixel.dtype),
                         torch.arange(width,device=pixel.device,dtype=pixel.dtype),indexing="ij")
    grid=torch.stack([xx,yy],-1).reshape(-1,2)
    delta=pixel[:,None,:]-grid[None]
    conic=torch.linalg.inv(cov2)
    exponent=-.5*torch.einsum("kpi,kij,kpj->kp",delta,conic,delta)
    alpha=(opacity[:,None]*exponent.exp()).clamp(max=.99)
    alpha=torch.where((alpha>=1/255)&(depth[:,None]>.2),alpha,torch.zeros_like(alpha))
    order=depth.argsort()
    trans=torch.ones(len(grid),dtype=pixel.dtype,device=pixel.device)
    alive=torch.ones(len(grid),dtype=torch.bool,device=pixel.device)
    result=torch.zeros((len(grid),color.shape[-1]),dtype=pixel.dtype,device=pixel.device)
    for index in order:
        next_trans=trans*(1-alpha[index])
        use=alive & (next_trans>=1e-4)
        alive=use
        result=result+color[index]*torch.where(use,alpha[index]*trans,torch.zeros_like(trans))[:,None]
        trans=torch.where(use,next_trans,trans)
    return result.reshape(height,width,color.shape[-1]).clamp(0,1)

def angular_grid(spec,panel,width,height):
    lon=2*np.pi*((np.arange(width)+.5)/width-.5)
    lat=np.pi*((np.arange(height)+.5)/height-.5)
    lon,lat=np.meshgrid(lon,lat,indexing="xy")
    local=np.stack([np.sin(lon)*np.cos(lat),np.sin(lat),np.cos(lon)*np.cos(lat)],-1)
    # Camera columns are horizontal,up,normal; no perspective-camera substitution.
    camera=spec.axes[panel][:,[1,2,0]]
    direction=np.einsum("ij,hwj->hwi",camera,local)
    # Exact average over ALL specified frequencies using finite geometric sums.
    # Avoid storing32,400 directions x32 ports x1024 frequencies.
    dh,dv=np.meshgrid(np.arange(-7,8),np.arange(-3,4),indexing="ij")
    dh,dv=dh.ravel(),dv.ravel()
    difference=spec.spacing_m*(dh[:,None]*spec.axes[panel][:,1]-dv[:,None]*spec.axes[panel][:,2])
    tau=direction.reshape(-1,3)@difference.T/C0
    df=spec.bandwidth_hz/spec.frequency_count
    fmean=spec.carrier_hz+((spec.frequency_count-1)/2-spec.frequency_count//2)*df
    coherent=np.exp(2j*np.pi*fmean*tau)*np.sinc(spec.frequency_count*df*tau)/np.sinc(df*tau)
    u,v=np.meshgrid(np.arange(8),np.arange(4),indexing="ij")
    transform=np.exp(-2j*np.pi*(dh[:,None]*u.ravel()[None]/8+dv[:,None]*v.ravel()[None]/4))
    multiplicity=(8-np.abs(dh))*(4-np.abs(dv))/32
    response=np.maximum(((coherent*multiplicity)@transform).real,0)
    solid=np.cos(lat).reshape(-1)*(2*np.pi/width)*(np.pi/height)
    return camera,(response*solid[:,None]).astype(np.float32)

class WRFField(nn.Module):
    def __init__(self,initial_xyz,fit_position,spec,panel,width=360,height=90):
        super().__init__();self.width=width;self.height=height
        xyz=np.asarray(initial_xyz,np.float32)
        self.xyz=nn.Parameter(torch.tensor(xyz))
        self.log_scale=nn.Parameter(torch.full_like(self.xyz,np.log(.5)))
        self.quaternion=nn.Parameter(torch.tensor(np.tile([1.,0.,0.,0.],(len(xyz),1)),dtype=torch.float32))
        self.opacity_logit=nn.Parameter(torch.full((len(xyz),),-2.1972246))
        self.sh=nn.Parameter(torch.zeros((len(xyz),16,1)))
        nn.init.normal_(self.sh[:,0],std=.1)
        self.deformation=author_network_class()()
        # Scalar linear-power task head replaces the author example RGB display head.
        self.deformation.gaussian_signal=nn.Linear(self.deformation.W,1)
        self.deformation.gaussian_phase=nn.Linear(self.deformation.W,1)
        # In the pinned author forward, warp is ignored and |real*exp(j phase)| is phase invariant.
        # Keep those inactive heads frozen, rather than count them as useful trainable capacity.
        for name,module in self.deformation.named_children():
            if name in ["gaussian_warp","gaussian_phase"]:
                for p in module.parameters():p.requires_grad_(False)
        camera,response=angular_grid(spec,panel,width,height)
        self.register_buffer("camera",torch.tensor(camera,dtype=torch.float32))
        self.register_buffer("beam_response",torch.tensor(response))
        self.register_buffer("center",torch.tensor(np.array(spec.center[panel]),dtype=torch.float32))
        p=np.asarray(fit_position)
        self.register_buffer("input_center",torch.tensor(p.mean(0),dtype=torch.float32))
        self.register_buffer("input_scale",torch.tensor(np.maximum(p.std(0),.1),dtype=torch.float32))
        self.eval_sh=author_eval_sh()

    def forward(self,position):
        values=[]
        for query in position:
            xyz_input=(self.xyz.detach()-self.input_center)/self.input_scale
            tx_input=((query-self.input_center)/self.input_scale).expand(len(self.xyz),-1)
            _,rotation,scale,signal=self.deformation(xyz_input,tx_input)
            # Valid physical covariance; normalizing deformed quaternions is explicit.
            q=self.quaternion/self.quaternion.norm(dim=-1,keepdim=True).clamp_min(1e-8)+rotation
            r=quaternion_matrix(q)
            scales=(self.log_scale.exp()+scale).abs().clamp_min(1e-3)
            cov=r@torch.diag_embed(scales.square())@r.transpose(-1,-2)
            pixel,cov2,depth=project_gaussians(self.xyz,cov,self.center,self.camera,self.width,self.height)
            direction=nn.functional.normalize(self.xyz-self.center,dim=-1)
            sh=(self.sh+signal[:,None,:]).transpose(1,2)
            color=(self.eval_sh(3,sh,direction)+.5).clamp_min(0)
            image=alpha_render(pixel,cov2,depth,self.opacity_logit.sigmoid(),color,self.width,self.height)
            # A scalar linear-power task replaces the author PNG/RGB display target.
            angular_power=image[...,0].reshape(-1)
            values.append(angular_power@self.beam_response)
        return torch.stack(values)

class WRFGSPlusPower(nn.Module):
    def __init__(self,fit_position,spec,initial_xyz,width=360,height=90):
        super().__init__()
        self.fields=nn.ModuleList([WRFField(initial_xyz,fit_position,spec,p,width,height) for p in range(2)])

    def forward(self,position):
        return torch.stack([field(position) for field in self.fields],dim=1)

    @torch.inference_mode()
    def predict(self,query):
        self.eval();device=self.fields[0].xyz.device
        power=self(torch.tensor(np.array(query.position),dtype=torch.float32,device=device)).cpu().numpy()
        return check_prediction(dict(method="WRF_GSplus_author_based_DFT_power_adapter",power=power,
             diagnostics=dict(original_CUDA_kernel_verified=False,unchanged_paper_reproduction=False,
                              shared_wideband_training_target=True,all_frequency_array_response=True,analytic_frequency_average=True)),query)

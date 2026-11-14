"""Read-only FIT diagnostics for one immutable NeRF2 repair checkpoint."""
from __future__ import annotations
import argparse, hashlib, json, math, signal, time, traceback
from pathlib import Path
import numpy as np
import torch
from . import nerf2_repair as R

BS=[0,17,30,47,64,81,98,127]

def correlation(x,y):
    x=x.reshape(-1).astype(np.complex128);y=y.reshape(-1).astype(np.complex128)
    xx=float(np.vdot(x,x).real);yy=float(np.vdot(y,y).real)
    c=np.vdot(x,y)/math.sqrt(xx*yy) if xx>0 and yy>0 else None
    return dict(real=None if c is None else float(c.real),magnitude=None if c is None else float(abs(c)))

def contrast(x):
    z=np.asarray(x,np.complex128).reshape(2,-1)
    e=float(np.abs(z).square().sum()) if hasattr(np.abs(z),"square") else float(np.square(np.abs(z)).sum())
    return dict(correlation=correlation(z[0],z[1]),
        deviation_energy_fraction=float(np.square(np.abs(z-z.mean(0))).sum()/e) if e else None)

def group(n):
    if n=="log_output_gain":return "global_gain"
    if "attenuation_output" in n or "signal_output" in n:return "frequency_heads"
    if "attenuation_linears" in n:return "medium_trunk"
    if "signal_linears" in n or "feature_layer" in n:return "signal_trunk"
    return "other"

def summarize_gradients(gs):
    answer={}
    for key in ["all","global_gain","frequency_heads","medium_trunk","signal_trunk","other"]:
        names=[n for n in gs[0] if key=="all" or group(n)==key]
        a=sum(float(gs[0][n].square().sum()) for n in names)
        b=sum(float(gs[1][n].square().sum()) for n in names)
        dot=sum(float((gs[0][n]*gs[1][n]).sum()) for n in names)
        joint=sum(float((.5*(gs[0][n]+gs[1][n])).square().sum()) for n in names)
        answer[key]=dict(parameter_tensors=len(names),position_gradient_norm=[math.sqrt(a),math.sqrt(b)],
            cosine=dot/math.sqrt(a*b) if a>0 and b>0 else None,joint_gradient_norm=math.sqrt(joint),
            dot=dot)
    return answer

def adam_projection(model,opt,gs):
    """First-order loss projection of the frozen Adam state; no optimizer.step."""
    projections=[0.,0.];norm=0.
    names={id(p):n for n,p in model.named_parameters()}
    for pg in opt.param_groups:
        b1,b2=pg["betas"];eps=pg["eps"];lr=pg["lr"];wd=pg.get("weight_decay",0.)
        for p in pg["params"]:
            n=names[id(p)]
            if n not in gs[0]:continue
            state=opt.state[p]
            if not state:continue
            k=int(state["step"])+1
            gradient=.5*(gs[0][n]+gs[1][n])+wd*p.detach().double()
            m=b1*state["exp_avg"].double()+(1-b1)*gradient
            v=b2*state["exp_avg_sq"].double()+(1-b2)*gradient.square()
            direction=-lr*(m/(1-b1**k))/(torch.sqrt(v/(1-b2**k))+eps)
            norm+=float(direction.square().sum())
            for i in range(2):projections[i]+=float((gs[i][n]*direction).sum())
    return dict(first_order_loss_change=projections,step_direction_norm=math.sqrt(norm),
        meaning="infinitesimal projection only; negative predicts local decrease; no actual update or line search",
        batch_BS_count=len(BS))

def param_hash(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def run(checkpoint,label):
    start=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError("60s diagnostic cap")))
    signal.alarm(60)
    gpu=R.C.device_guard(R.GPU);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfgroot=json.loads(R.CONFIG.read_text());cfg=cfgroot["nerf2"];src=R.identity()
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if ck["source_identity"]!=src or ck["config"]!=cfg:raise RuntimeError("Checkpoint source/config mismatch")
    data,norm,raw,info=R.inputs(cfg)
    if info!=ck["input_identity"]:raise RuntimeError("Checkpoint inputs mismatch")
    model,opt,_=R.build(data,cfgroot["seed"]);model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
    model.eval();before=param_hash(model)
    targets=torch.tensor(norm,device="cuda:0",dtype=torch.complex64)
    renderers=[R.J.JointRenderer(data,dict(row_id=int(row))) for row in cfg["rows"]]
    pred,stats=R.evaluate(model,renderers,targets,cfg["rows"])
    gs=[];losses=[];values=[]
    for i,renderer in enumerate(renderers):
        value=renderer(model,BS,[0,1]);truth=R.J.label_subset(targets[i],BS,[0,1])
        loss=.5*(value-truth).abs().square().mean()
        gs.append(R.N.grad_map(model,loss));losses.append(float(loss.detach()))
        values.append(value.detach().cpu().numpy())
    gradients=summarize_gradients(gs);projection=adam_projection(model,opt,gs)
    total=sum(losses)
    pred_np=pred.detach().cpu().numpy()
    result=dict(status="passed",checkpoint=R.PATHS.relative(checkpoint),
        checkpoint_sha256=R.C.sha256(checkpoint),checkpoint_step=int(ck["step"]),
        source_identity=src,diagnostic_source_sha256=R.C.sha256(Path(__file__)),
        input_identity=info,GPU=gpu,BS_ids=BS,UE_ids=[0,1],frequency_count=192,
        gradients_scope="one fixed8BS x2UE x192F subtensor at each of2FIT positions",
        full_FIT_scope="2x128BSx2UEx192F",
        full_FIT_stats=stats,prediction_between_positions=contrast(pred_np),
        truth_between_positions=contrast(norm),
        prediction_truth_correlation=[correlation(pred_np[i],norm[i]) for i in range(2)],
        unit_position_losses=losses,position_loss_shares=[x/total for x in losses],
        gradients=gradients,frozen_Adam_direction=projection,optimizer_updates=0,
        actual_UE_world_position_difference_m=float((renderers[0].world-renderers[1].world).norm()),
        original_parameters_sha256=before,after_parameters_sha256=param_hash(model),
        query_truth_read=False,validation_truth_read=False,phase_delay_amplitude_alignment=False,
        seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        no_complete_root_cause_claim=True)
    if result["original_parameters_sha256"]!=result["after_parameters_sha256"]:
        raise RuntimeError("Read-only parameter guard")
    R.C.atomic_json(R.REPORT/f"diagnosis_{label}.json",result)
    signal.alarm(0)
    print(json.dumps({k:result[k]for k in ["status","checkpoint_step","prediction_between_positions",
        "truth_between_positions","prediction_truth_correlation","position_loss_shares","gradients",
        "frozen_Adam_direction","seconds"]},allow_nan=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--label",required=True)
    args=p.parse_args()
    try:run(R.PATHS.root/args.checkpoint,args.label)
    except BaseException:
        with (R.REPORT/f"failure_diagnose_{args.label}.log").open("a")as f:f.write(traceback.format_exc()+"\n")
        raise

"""Short no-optimizer NeWRF frequency sampler QA and timing."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import torch
M=None
B=None
def rel(a,b):
    a=np.asarray(a,np.complex128);b=np.asarray(b,np.complex128)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
def cpu_floor(cfg,scale):
    rng=np.random.default_rng(730);reports=[]
    for case in["normal","mixed_zero","all_zero","subfloor"]:
        t=(rng.normal(size=(2,8,192))+1j*rng.normal(size=(2,8,192))).astype(np.complex64)*scale
        if case=="mixed_zero":t[0]=0
        if case=="all_zero":t[:]=0
        if case=="subfloor":t*=1e-5
        p=torch.tensor((rng.normal(size=t.shape)+1j*rng.normal(size=t.shape)).astype(np.complex64)*scale,requires_grad=True)
        q=torch.tensor((rng.normal(size=t.shape)+1j*rng.normal(size=t.shape)).astype(np.complex64)*scale,requires_grad=True)
        y=torch.tensor(t);den,e,f=M.denominator(t,type("D",(),{"scale":scale})(),cfg)
        full=(.9*abs(p-y).square().sum(dtype=torch.float64)+.1*abs(q-y).square().sum(dtype=torch.float64))/den
        g=torch.autograd.grad(full,[p,q],retain_graph=True)
        record=dict(case=case,D_full=den,true_energy=e,floor=f,partition_gradient_relative_L2={})
        for k in[8,16]:
            accum=[torch.zeros_like(x)for x in g];parts=np.arange(192).reshape(-1,k)
            for ff in parts:
                z=(.9*abs(p[:,:,ff]-y[:,:,ff]).square().sum(dtype=torch.float64)+.1*abs(q[:,:,ff]-y[:,:,ff]).square().sum(dtype=torch.float64))*(192/k)/den
                gg=torch.autograd.grad(z,[p,q],retain_graph=True)
                for a,b in zip(accum,gg):a.add_(b/len(parts))
            err=rel(torch.cat([x.detach().reshape(-1)for x in accum]).numpy(),torch.cat([x.detach().reshape(-1)for x in g]).numpy())
            assert err<1e-6,(case,k,err)
            assert M.denominator(t,type("D",(),{"scale":scale})(),cfg)[0]==den
            record["partition_gradient_relative_L2"][str(k)]=err
        reports.append(record)
    return reports
def run(path,uuid,module):
    global M,B
    M=module;B=M.B
    cfg=M.load_config(path);gpu=B.gpu_setup(uuid);B.C.seed_all(cfg["seed"])
    data=B.prepare_data(4,"screen");start=time.monotonic();bank,base,model=B.build(data,cfg,"cuda:0")
    build_seconds=time.monotonic()-start;parent_sha=B.C.sha256(M.PARENT)
    ck=torch.load(M.PARENT,map_location="cpu",weights_only=False);model.load_state_dict(ck["model"])
    opt=torch.optim.Adam(model.fine[0].parameters(),lr=cfg["learning_rates"]["warm"])
    B.set_stage(model,0,"warm",opt,False)
    sampler=M.ToneSampler(data,cfg["batch_positions"],cfg["seed"]+4000);sampler.restore(ck["sampler"])
    rows,blocks=sampler.draw();full=B.block_labels(data,rows,blocks,0)
    cache=B.QueryDirectionLRU(model,data,cfg["direction_cache_entries"])
    active=[p for p in model.parameters()if p.requires_grad]
    qa_cpu_rng=ck["torch_rng"].clone();qa_cuda_rng=[x.clone()for x in ck["cuda_rng"]]
    def reset_rng():
        torch.set_rng_state(qa_cpu_rng.clone());torch.cuda.set_rng_state_all([x.clone()for x in qa_cuda_rng])
    def gradient(ff,mask=False):
        reset_rng()
        loss,info=M.sample_loss(model,data,cache,rows,blocks,0,np.asarray(ff,np.int64),full,cfg,mask)
        g=torch.autograd.grad(loss,active,allow_unused=True)
        vec=torch.cat([(torch.zeros_like(p)if x is None else x).detach().double().flatten()for x,p in zip(g,active)])
        assert torch.isfinite(vec).all()
        return vec.cpu().numpy(),info
    # Actual physical frequency IDs explicitly cover band edges and centre.
    choices={8:np.array([0,19,47,95,96,123,160,191]),16:np.array([0,9,19,35,47,63,79,95,96,111,127,143,159,175,183,191])}
    forwards={}
    with torch.no_grad():
        for k,ff in choices.items():
            pp=[];cc=[];pr=[];cr=[]
            for row,block in zip(rows,blocks):
                bs=np.arange(int(block)*8,int(block)*8+8)
                reset_rng();h,hc=B.cached_block(model,data,int(row),0,bs,B.FF,cache)
                reset_rng();s,sc=B.cached_block(model,data,int(row),0,bs,ff,cache)
                pp.append(s.cpu().numpy());cc.append(sc.cpu().numpy())
                pr.append(h[:,:,ff].cpu().numpy());cr.append(hc[:,:,ff].cpu().numpy())
            er=dict(fine=rel(np.concatenate(pp),np.concatenate(pr)),coarse=rel(np.concatenate(cc),np.concatenate(cr)))
            assert max(er.values())<=1e-5,(k,er)
            forwards[str(k)]=er
    full_grad,_=gradient(B.FF);checks={};rng=np.random.default_rng(731)
    for k,ff in choices.items():
        g,info=gradient(ff);reference,_=gradient(ff,True);err=rel(g,reference)
        assert err<=1e-4,(k,"masked gradient",err)
        partition=rng.permutation(192).reshape(-1,k);mean=np.zeros_like(full_grad)
        for group in partition:mean+=gradient(group)[0]/len(partition)
        recover=rel(mean,full_grad);assert recover<=1e-4,(k,"partition gradient",recover)
        timing=[]
        for i in range(5):
            group=np.sort(rng.choice(192,k,replace=False))
            torch.cuda.synchronize();tick=time.monotonic();gradient(group);torch.cuda.synchronize()
            if i>=2:timing.append(time.monotonic()-tick)
        checks[str(k)]=dict(actual_Hz_count=k,D_full=info["D_full"],floor_full=info["floor_full"],
          masked_unclipped_gradient_relative_L2=err,partition_mean_gradient_relative_L2=recover,
          median_gradient_only_seconds=float(np.median(timing)),timings=timing)
    full_times=[]
    for _ in range(3):
        torch.cuda.synchronize();tick=time.monotonic();gradient(B.FF);torch.cuda.synchronize();full_times.append(time.monotonic()-tick)
    model.eval();row=int(rows[0])
    # New complete-H wrapper versus old original subset-query, same full192 tones.
    with torch.no_grad():
        reset_rng();new=B.query_full_position(model,data,row);original=np.empty_like(new)
        reset_rng()
        for field in range(8):
            p,u=divmod(field,4);v,_=B.predict_field(model,data,[row],field)
            original[p*128:(p+1)*128,u]=v[0]
    complete=rel(new,original);assert complete<=1e-5,complete
    floors=cpu_floor(cfg,data.scale)
    assert parent_sha==B.C.sha256(M.PARENT)==M.PARENT_SHA
    chosen=min(checks,key=lambda k:checks[k]["median_gradient_only_seconds"])
    result=dict(status="passed",source_identity=M.source_identity(path),qa_source_sha256=B.C.sha256(Path(__file__)),
      GPU=gpu,build_seconds=build_seconds,forward_full192_slice_relative_L2=forwards,
      frequency_cases=checks,full192_gradient_median_seconds=float(np.median(full_times)),
      complete_H_shape=list(new.shape),complete192_H_relative_L2=complete,
      floor_CPU_cases=floors,recommended_K_by_latency_only=int(chosen),
      parent_checkpoint_unchanged=True,formal_resume_started=False,optimizer_updates=0,
      timings_are_forward_backward_only=True,no_validation_or_outer_CSI=True,peak_cuda_bytes=torch.cuda.max_memory_allocated())
    M.ORIG_JSON(M.R/"QA.json",result);print(json.dumps(result))

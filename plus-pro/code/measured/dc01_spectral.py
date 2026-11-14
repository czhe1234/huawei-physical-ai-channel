"""Port the final spectral CNNs and LinkGate to the real dc01 array.

Only class definitions are loaded from the frozen production source, avoiding
its competition-specific datasets and global cache locations. Network bodies,
residual limits, and gate initialization are retained. Geometry sizes and the
number of physical ports are adapted to the measured single-TX panels.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from torch import nn

from project_paths import PATHS
from measured_runtime import aggregate,cosine,metric_points,normalize,pas,pdp,project,render,safe_sqrt
from dc01_state import combine_support


def original_classes():
    root=PATHS.root/'final_code/code/round2/physical_ai_submission/base_runtime'
    namespace=dict(torch=torch,nn=nn,np=np,math=math,dataclass=dataclass,TINY=1e-30)
    selected={'TrainConfig','ResidualBlock1d','ResidualBlock2d','SpectralCalibrator','LinkGate'}
    provenance={}
    for name in ['spectral_calibrator.py','path_link_core.py']:
        source=(root/name).read_text()
        nodes=[n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name in selected]
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),str(root/name),'exec'),namespace)
        provenance[name]=hashlib.sha256(source.encode()).hexdigest()
    return namespace,provenance


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,path)


def atomic_torch(path,value):
    tmp=path.with_suffix('.partial.pt');torch.save(value,tmp);os.replace(tmp,path)


class Bank:
    def __init__(self,rows,base,truth,raw,position,panel_position,device,allow_labels=True):
        self.device=device;self.rows=rows
        self.base=torch.as_tensor(np.asarray(base),device=device).reshape(-1,1,8,4,base.shape[-1])
        self.truth=torch.as_tensor(np.asarray(truth[rows]),device=device).reshape_as(self.base) if allow_labels else None
        self.pas=pas(self.base)[:,0];self.pdp=pdp(self.base)[:,0]
        self.true_pas=pas(self.truth)[:,0] if allow_labels else None
        self.true_pdp=pdp(self.truth)[:,0] if allow_labels else None
        mass=torch.as_tensor(np.array(raw['mass'][rows]).reshape(-1,128),device=device)
        features=torch.as_tensor(np.array(raw['features'][rows]).reshape(-1,128,24),device=device)
        coordinate=torch.as_tensor(np.array(raw['coordinate'][rows]).reshape(-1,128,3),device=device)
        coordinate=coordinate.clone()
        peak=self.pdp.mean((1,2)).argmax(-1)
        coordinate[...,2]=(coordinate[...,2]+peak[:,None])%base.shape[-1]
        self.prior={}
        for name,index in [('base',11),('scatter',12),('both',None)]:
            m=mass if index is None else mass*features[...,index]
            m=normalize(m,(-1,))
            p,d=render(m[:,None],coordinate[:,None],8,4,base.shape[-1])
            self.prior[name]=dict(pas=p[:,0].mean(-1),pdp=d[:,0].mean((1,2)))
        n=len(rows)
        pos=np.repeat(position[rows],2,axis=0)
        panel=np.tile(np.arange(2),n)
        distance=np.linalg.norm(pos-panel_position[panel],axis=-1)
        entropy=-(mass*mass.clamp_min(1e-12).log()).sum(-1).cpu().numpy()
        count=(mass>0).sum(-1).cpu().numpy()
        context=np.c_[pos/30,np.eye(2)[panel],distance/100,entropy/5,count/128]
        self.context=torch.as_tensor(context,device=device,dtype=torch.float32)
        self.frequency=base.shape[-1]
        self.predicted={}

    def sample(self,domain,mode,point,slice_id,mean=None,std=None,include_target=True):
        if domain=='pas':
            anchor=self.pas[point,:,:,slice_id]
            raw_base=self.prior['base']['pas'][point]
            raw_scatter=self.prior['scatter']['pas'][point]
            coordinate=slice_id.float()/self.frequency
            extra=torch.stack((coordinate-0.5,torch.sin(2*torch.pi*coordinate),torch.cos(2*torch.pi*coordinate)),dim=1)
            dims=(1,2)
        else:
            anchor=self.pdp.flatten(1,2)[point,slice_id]
            raw_base=self.prior['base']['pdp'][point]
            raw_scatter=self.prior['scatter']['pdp'][point]
            h=slice_id//4;v=slice_id%4
            extra=torch.stack((h/7.,v/3.,torch.sin(2*torch.pi*h/8),torch.cos(2*torch.pi*h/8),torch.sin(2*torch.pi*v/4),torch.cos(2*torch.pi*v/4)),dim=1)
            dims=(1,)
        raw_base=torch.where(raw_base.sum(dim=dims,keepdim=True)>0,raw_base,anchor)
        raw_scatter=torch.where(raw_scatter.sum(dim=dims,keepdim=True)>0,raw_scatter,anchor)
        if mode=='single':
            baseline=raw_base
            feature=raw_base.clamp_min(1e-8).log()[:,None]
        else:
            baseline=normalize((0.5*safe_sqrt(raw_base)+0.5*safe_sqrt(raw_scatter)).square(),dims)
            feature=torch.stack([raw_base.clamp_min(1e-8).log(),raw_scatter.clamp_min(1e-8).log(),anchor.clamp_min(1e-8).log(),(raw_base-raw_scatter).abs()],dim=1)
        context=torch.cat((self.context[point],extra),dim=1)
        if mean is not None:context=(context-mean)/std
        if include_target:
            if self.truth is None:raise RuntimeError('Held-out labels are unavailable to spectral inference')
            target=self.true_pas[point,:,:,slice_id] if domain=='pas' else self.true_pdp.flatten(1,2)[point,slice_id]
        else:target=None
        return feature,context,baseline,target

    def indices(self,domain):
        slices=self.frequency if domain=='pas' else 32
        point=torch.arange(len(self.base),device=self.device).repeat_interleave(slices)
        slice_id=torch.arange(slices,device=self.device).repeat(len(self.base))
        return point,slice_id


def make_calibrator(classes,domain,channels,context,frequency,device):
    config=classes['TrainConfig']()
    model=classes['SpectralCalibrator'](domain,channels,context,config).to(device)
    if domain=='pas':
        h=torch.arange(8,device=device)*2*torch.pi/8
        v=torch.arange(4,device=device)*2*torch.pi/4
        hh,vv=torch.meshgrid(h,v,indexing='ij')
        model.coordinate=torch.stack((torch.sin(hh),torch.cos(hh),torch.sin(vv),torch.cos(vv)))[None]
        model.output=nn.Conv2d(config.hidden,1,1).to(device)
        nn.init.zeros_(model.output.weight);nn.init.zeros_(model.output.bias)
    else:
        d=torch.arange(frequency,device=device)*2*torch.pi/frequency
        model.coordinate=torch.stack([f(k*d) for k in [1,2,4] for f in [torch.sin,torch.cos]])[None]
    return model,config


def train_calibrator(classes,domain,mode,fit,val,epochs,seed,checkpoint=None):
    torch.manual_seed(seed)
    fp,fs=fit.indices(domain)
    vp,vs=val.indices(domain)
    _,context,_,_=fit.sample(domain,mode,fp,fs)
    mean=context.mean(0);std=context.std(0).clamp_min(1e-3)
    feature,c,_,_=fit.sample(domain,mode,fp[:1],fs[:1],mean,std)
    model,config=make_calibrator(classes,domain,feature.shape[1],c.shape[1],fit.frequency,fit.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    steps=max(1,math.ceil(0.25*len(fp)/512))
    best=-1;best_epoch=0;best_state=None;history=[];first_epoch=0
    if checkpoint is not None and checkpoint.exists():
        saved=torch.load(checkpoint,map_location=fit.device,weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        best=saved['best'];best_epoch=saved['best_epoch'];best_state=saved['best_state'];history=saved['history']
        first_epoch=saved['epoch']+1
        torch.set_rng_state(saved['torch_rng'].cpu())
        if fit.device.type=='cuda':torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),fit.device)
    for epoch in range(first_epoch,epochs+1):
        if epoch:
            model.train()
            for _ in range(steps):
                take=torch.randint(len(fp),(min(512,len(fp)),),device=fit.device)
                x,c,baseline,target=fit.sample(domain,mode,fp[take],fs[take],mean,std)
                prediction,delta=model(x,c,baseline)
                dims=tuple(range(1,prediction.ndim))
                loss=(1-cosine(prediction,target,dims)).mean()+config.hellinger_weight*(safe_sqrt(prediction)-safe_sqrt(target)).square().sum(dims).mean()+config.residual_weight*delta.square().mean()
                if not torch.isfinite(loss):raise RuntimeError('Non-finite spectral calibration loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model.eval();scores=[]
        with torch.no_grad():
            for first in range(0,len(vp),2048):
                x,c,baseline,target=val.sample(domain,mode,vp[first:first+2048],vs[first:first+2048],mean,std)
                prediction,_=model(x,c,baseline)
                scores.append(cosine(prediction,target,tuple(range(1,prediction.ndim))))
        score=float(torch.cat(scores).mean())
        if score>best:
            best=score;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        row=dict(epoch=epoch,validation_cosine=score);history.append(row)
        if checkpoint is not None:
            atomic_torch(checkpoint,dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),
                         best=best,best_epoch=best_epoch,best_state=best_state,history=history,
                         torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(fit.device) if fit.device.type=='cuda' else None))
        if epoch%4==0:print(json.dumps(dict(domain=domain,mode=mode,**row)),flush=True)
    model.load_state_dict(best_state);model.eval()
    return model,mean,std,dict(best_epoch=best_epoch,validation_cosine=best,history=history,steps_per_epoch=steps,
                              parameters=sum(p.numel() for p in model.parameters())),config


@torch.inference_mode()
def predict_calibrator(model,mean,std,bank,domain,mode):
    points,slices=bank.indices(domain);output=[]
    for first in range(0,len(points),2048):
        feature,context,baseline,_=bank.sample(domain,mode,points[first:first+2048],slices[first:first+2048],mean,std,include_target=False)
        prediction,_=model(feature,context,baseline);output.append(prediction)
    value=torch.cat(output)
    if domain=='pas':return value.reshape(len(bank.base),bank.frequency,8,4).permute(0,2,3,1)
    return value.reshape(len(bank.base),8,4,bank.frequency)


def complete_score(pred,truth):
    return float(0.5*(cosine(pas(pred),pas(truth),(2,3)).mean()+cosine(pdp(pred),pdp(truth),(-1,)).mean()))


def link_features(base,raw,context):
    n=len(base);f=base.shape[-1]
    b=base.reshape(n,32,32,f//32).sum(-1);r=raw.reshape(n,32,32,f//32).sum(-1)
    br=safe_sqrt(b);rr=safe_sqrt(r)
    summary=torch.stack((-(base*base.clamp_min(1e-12).log()).sum(-1),-(raw*raw.clamp_min(1e-12).log()).sum(-1),(br*rr).sum(-1)),dim=-1)
    h=torch.arange(8,device=base.device).repeat_interleave(4);v=torch.arange(4,device=base.device).repeat(8)
    position=torch.stack((h/7.,v/3.,torch.sin(2*torch.pi*h/8),torch.cos(2*torch.pi*h/8),torch.sin(2*torch.pi*v/4),torch.cos(2*torch.pi*v/4)),dim=-1)
    return torch.cat((br,rr,rr-br,br*rr,summary,context[:,None].expand(-1,32,-1),position[None].expand(n,-1,-1)),dim=-1)


def main():
    parser=argparse.ArgumentParser()
    for name in ['prepared','baseline','geometry','output']:parser.add_argument(f'--{name}-dir',type=Path,required=True)
    parser.add_argument('--device',default='cuda:3');parser.add_argument('--epochs',type=int,default=16);parser.add_argument('--fold',type=int)
    args=parser.parse_args();torch.set_num_threads(8)
    device=torch.device(args.device);torch.cuda.set_device(device)
    classes,provenance=original_classes()
    h=np.load(args.prepared_dir/'channels.npy',mmap_mode='r');meta=np.load(args.prepared_dir/'selection.npz')
    split=json.loads((args.baseline_dir/'split_manifest.json').read_text())
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(args.baseline_dir/'split_manifest.json',root/'split_manifest.json')
    support=root/'merged_support';combine_support(args.geometry_dir,support,len(h))
    raw={k:np.load(support/f'{k}.npy',mmap_mode='r') for k in ['mass','coordinate','features']}
    for fold in (range(5) if args.fold is None else [args.fold]):
        started=time.time();out=root/f'fold{fold}';out.mkdir(exist_ok=True)
        identity=dict(split=split['sha256'],code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      original=provenance,epochs=args.epochs,
                      baseline=hashlib.sha256((args.baseline_dir/f'fold{fold}'/'baseline_report.json').read_bytes()).hexdigest(),
                      support=hashlib.sha256((support/'support_manifest.json').read_bytes()).hexdigest())
        if (out/'training_identity.json').exists() and json.loads((out/'training_identity.json').read_text())!=identity:raise ValueError('Spectral training identity changed')
        atomic_json(out/'training_identity.json',identity)
        if (out/'baseline_report.json').exists():print('RESUME_SPECTRAL_FOLD',fold,flush=True);continue
        torch.cuda.reset_peak_memory_stats(device)
        ids={k:np.array(split['splits'][fold][k]) for k in ['fit','val','test']}
        banks={k:Bank(rows,np.load(args.baseline_dir/f'fold{fold}'/f'{k}_base.npy',mmap_mode='r'),h,raw,meta['position'],meta['panel_position'],device,allow_labels=k!='test') for k,rows in ids.items()}
        training={};weights={}
        for domain in ['pas','pdp']:
            for mode in ['single','joint']:
                model,mean,std,report,config=train_calibrator(classes,domain,mode,banks['fit'],banks['val'],args.epochs,20261015+fold+100*int(domain=='pdp')+1000*int(mode=='joint'),out/f'{domain}_{mode}_resume.pt')
                training[f'{domain}_{mode}']=report
                weights[f'{domain}_{mode}']=dict(model={k:v.cpu() for k,v in model.state_dict().items()},mean=mean.cpu(),std=std.cpu())
                for bank in banks.values():bank.predicted[f'{domain}_{mode}']=predict_calibrator(model,mean,std,bank,domain,mode)
                del model
        atomic_torch(out/'spectral_models.pt',weights)
        calibrated={}
        for name,bank in banks.items():
            ps=bank.predicted['pas_single'];pj=bank.predicted['pas_joint']
            ds=bank.predicted['pdp_single'];dj=bank.predicted['pdp_joint']
            pt=normalize((0.25*safe_sqrt(ps)+0.75*safe_sqrt(pj)).square(),(1,2))
            dt=normalize((0.50*safe_sqrt(ds)+0.50*safe_sqrt(dj)).square(),(-1,))
            calibrated[name]=project(bank.base,pt[:,None],dt[:,None],0.45,0.35,4)
        raw_val=complete_score(banks['val'].base,banks['val'].truth)
        spectral_val=complete_score(calibrated['val'],banks['val'].truth)
        spectral_accepted=spectral_val>raw_val+1e-7
        if not spectral_accepted:calibrated={k:b.base for k,b in banks.items()}
        fit=banks['fit'];val=banks['val']
        context_mean=fit.context.mean(0);context_std=fit.context.std(0).clamp_min(1e-3)
        gate_input={};gate_raw={};gate_base={};gate_truth={}
        for name,bank in banks.items():
            b=pdp(calibrated[name])[:,0].flatten(1,2)
            r=bank.prior['both']['pdp'][:,None,:].expand(-1,32,-1)
            gate_base[name]=b;gate_raw[name]=r
            if bank.true_pdp is not None:gate_truth[name]=bank.true_pdp.flatten(1,2)
            gate_input[name]=link_features(b,r,(bank.context-context_mean)/context_std)
        torch.manual_seed(20261015+fold)
        gate=classes['LinkGate'](gate_input['fit'].shape[-1],256,0.08).to(device)
        optimizer=torch.optim.AdamW(gate.parameters(),lr=0.001,weight_decay=0.0001)
        best=-1;best_state=None;best_epoch=0
        for epoch in range(args.epochs+1):
            if epoch:
                gate.train();order=torch.randperm(len(fit.base),device=device)
                for first in range(0,len(order),32):
                    index=order[first:first+32];a=0.5*gate(gate_input['fit'][index])
                    predicted=normalize(((1-a[...,None])*safe_sqrt(gate_base['fit'][index])+a[...,None]*safe_sqrt(gate_raw['fit'][index])).square(),(-1,))
                    loss=1-cosine(predicted,gate_truth['fit'][index],(-1,)).mean()
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(gate.parameters(),1.);optimizer.step()
            gate.eval()
            with torch.no_grad():
                a=0.5*gate(gate_input['val'])
                predicted=normalize(((1-a[...,None])*safe_sqrt(gate_base['val'])+a[...,None]*safe_sqrt(gate_raw['val'])).square(),(-1,))
                score=float(cosine(predicted,gate_truth['val'],(-1,)).mean())
            if score>best:best=score;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in gate.state_dict().items()}
        gate.load_state_dict(best_state);gate.eval();linked={}
        with torch.no_grad():
            for name,bank in banks.items():
                a=0.5*gate(gate_input[name])
                target=normalize(((1-a[...,None])*safe_sqrt(gate_base[name])+a[...,None]*safe_sqrt(gate_raw[name])).square(),(-1,)).reshape(-1,1,8,4,h.shape[-1])
                linked[name]=project(calibrated[name],pas(calibrated[name]),target,0,1,4)
        before_link=complete_score(calibrated['val'],val.truth);after_link=complete_score(linked['val'],val.truth)
        gate_accepted=after_link>before_link+1e-7
        if not gate_accepted:linked=calibrated
        atomic_torch(out/'link_gate.pt',dict(model=best_state,context_mean=context_mean.cpu(),context_std=context_std.cpu(),best_epoch=best_epoch))
        for name,value in linked.items():np.save(out/f'{name}_base.npy',value[:,0].reshape(-1,2,8,4,h.shape[-1]).cpu().numpy())
        points={}
        for first in range(0,len(ids['test']),32):
            predicted=linked['test'][:,0].reshape(-1,2,8,4,h.shape[-1])[first:first+32]
            truth=torch.as_tensor(np.asarray(h[ids['test'][first:first+32]]),device=device)
            for key,value in metric_points(predicted,truth).items():points.setdefault(key,[]).append(value)
        points={k:np.concatenate(v) for k,v in points.items()}
        np.savez(out/'frozen_base_metrics.npz',source_id=meta['source_id'][ids['test']],**points)
        metrics=aggregate(points)
        report=dict(identity=identity,selected_on_validation='spectral_link_foundation',
                    frozen_base_test=metrics,test={'spectral_link_foundation':metrics},
                    training=training,spectral_accepted=spectral_accepted,gate_accepted=gate_accepted,
                    validation_raw=raw_val,validation_spectral=spectral_val,validation_before_link=before_link,
                    validation_after_link=after_link,gate_best_epoch=best_epoch,
                    elapsed_s=time.time()-started,peak_gpu_GiB=torch.cuda.max_memory_allocated(device)/1024**3,
                    gpu=torch.cuda.get_device_name(device),original_classes=provenance,
                    note='Four original spectral CNN bodies and the original LinkGate, retrained on dc01; no competition checkpoint is used.')
        atomic_json(out/'baseline_report.json',report)
        print(json.dumps(dict(stage='spectral_fold_complete',fold=fold,test=metrics,spectral=spectral_accepted,link=gate_accepted)),flush=True)
        del banks,calibrated,linked,gate,optimizer,gate_input,gate_raw,gate_base,gate_truth
        torch.cuda.empty_cache()


if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise

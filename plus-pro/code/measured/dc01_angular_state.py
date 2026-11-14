"""Measurement-supported angular path correction with explicit energy field."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import torch
from torch import nn

from project_paths import PATHS
from dc01_state import MeasuredComponentState
from dc01_factorized_field import atomic_json,utility
from measured_runtime import aggregate,cosine,metric_points,normalize,pas,pdp,safe_sqrt,unit_complex


def atomic_torch(path,value):
    temp=path.with_suffix('.partial.pt');torch.save(value,temp);os.replace(temp,path)


def covariance_error(pred,truth):
    a=pred.flatten(2,3);b=truth.flatten(2,3)
    ra=a@a.conj().transpose(-1,-2);rb=b@b.conj().transpose(-1,-2)
    ra=ra/ra.diagonal(dim1=-2,dim2=-1).real.sum(-1)[...,None,None].clamp_min(1e-30)
    rb=rb/rb.diagonal(dim1=-2,dim2=-1).real.sum(-1)[...,None,None].clamp_min(1e-30)
    return (ra-rb).abs().square().sum((-1,-2))/rb.abs().square().sum((-1,-2)).clamp_min(1e-30)


def angular_render(mass,coordinate):
    kernels=[]
    for axis,n in enumerate([8,4]):
        index=torch.arange(n,device=coordinate.device,dtype=torch.float32)
        wave=torch.exp(2j*torch.pi*coordinate[...,axis,None]*index/n)
        kernels.append(normalize(torch.fft.fft(wave).abs().square(),(-1,)))
    return normalize(torch.einsum('bk,bkh,bkv->bhv',mass,*kernels),(1,2))[:,None,...,None]


def angular_project(channel,target,alpha):
    power=channel.abs().square().sum((2,3,4),keepdim=True)
    desired_p=normalize(((1-alpha)*safe_sqrt(pas(channel))+alpha*safe_sqrt(target)).square(),(2,3))
    desired_d=pdp(channel);value=channel
    for _ in range(4):
        angle=torch.fft.fft2(value,dim=(2,3));energy=angle.abs().square().sum((2,3),keepdim=True)
        value=torch.fft.ifft2(safe_sqrt(desired_p*energy)*unit_complex(angle),dim=(2,3))
        delay=torch.fft.ifft(value,dim=-1);energy=delay.abs().square().sum(-1,keepdim=True)
        value=torch.fft.fft(safe_sqrt(desired_d*energy)*unit_complex(delay),dim=-1)
        value=value*safe_sqrt(power/value.abs().square().sum((2,3,4),keepdim=True).clamp_min(1e-30))
    return value


class AngularState(nn.Module):
    def __init__(self):
        super().__init__();self.path=MeasuredComponentState()
        self.gate=nn.Sequential(nn.Linear(120,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,math.log(2/3))

    def forward(self,batch):
        state=self.path(batch)
        mass=normalize(batch['base_log_mass'].exp()*batch['mask'],(-1,))
        context=(mass[...,None]*batch['base']).sum(1)
        alpha=.5*torch.sigmoid(self.gate(context))[:,None,None,None]
        target=angular_render(state['mass'],state['coordinate'])
        value=angular_project(batch['channel'],target,alpha)
        value=torch.where(batch['mask'].any(-1)[:,None,None,None,None],value,batch['channel'])
        return value,state,alpha


def input_bank(rows,base,raw,meta,device,shuffle):
    donor=np.array(rows)
    if shuffle:donor=np.random.default_rng(20261016).permutation(donor)
    channel=torch.as_tensor(np.array(base),device=device)
    coordinate=np.array(raw['coordinate'][donor]);features=np.array(raw['features'][donor])
    mass=np.array(raw['mass'][donor]);mask=np.array(raw['mask'][donor])
    # Invariant features: relative energy and geometry; no absolute RT phase.
    period=np.array([8,4,256]);phase=2*np.pi*coordinate/period
    features[...,:3]=np.sin(phase);features[...,3:6]=np.cos(phase);features[...,6:9]=coordinate/period
    features[...,9]=np.log(np.maximum(mass,1e-30))/10;features[...,16]=np.log(np.maximum(mass,1e-30))/20
    features[...,17:19]=0
    features[...,19:22]=meta['position'][rows,None,None,:]/30
    distance=np.linalg.norm(meta['position'][rows,None,:]-meta['panel_position'][None],axis=-1)
    features[...,15]=distance[...,None]/100
    p=pas(channel).mean(-1).flatten(2);d=pdp(channel).mean((2,3)).reshape(len(rows),2,64,4).sum(-1)
    context=torch.cat([p.clamp_min(1e-8).log(),d.clamp_min(1e-8).log()],-1).cpu().numpy()
    features=np.concatenate([features,np.broadcast_to(context[:,:,None,:],(len(rows),2,128,96))],-1)
    sequence=np.array(raw['sequence'][donor]);depth=np.array(raw['depth'][donor])
    active=np.arange(16)<depth[...,None]
    sequence[...,18:21]=(meta['position'][rows,None,None,None,:]/30-sequence[...,:3])*active[...,None]
    return dict(base=torch.as_tensor(features.reshape(-1,128,120),device=device,dtype=torch.float32),
        sequence_geometry=torch.as_tensor(sequence.reshape(-1,128,16,21),device=device),
        interactions=torch.as_tensor(np.array(raw['interaction'][donor]).reshape(-1,128,16),device=device),
        depth=torch.as_tensor(depth.reshape(-1,128),device=device),
        coordinate=torch.as_tensor(coordinate.reshape(-1,128,3),device=device),
        base_log_mass=torch.as_tensor(np.log(np.maximum(mass,1e-30)).reshape(-1,128),device=device),
        mask=torch.as_tensor(mask.reshape(-1,128),device=device),channel=channel.reshape(-1,1,8,4,256))


@torch.inference_mode()
def validation(model,bank,truth,batch=64,baseline=False):
    model.eval();scores=[]
    for first in range(0,len(truth),batch):
        index=slice(first,first+batch);data={k:v[index] for k,v in bank.items()}
        value=data['channel'] if baseline else model(data)[0]
        target=truth[index];p=pas(value);tp=pas(target)
        structural=.5*(cosine(p,tp,(2,3)).mean((1,2))+cosine(pdp(value),pdp(target),(-1,)).mean((1,2,3)))
        pred_beam=torch.fft.fft2(value,dim=(2,3)).abs().square().mean(-1).flatten(2).argmax(-1)
        actual=torch.fft.fft2(target,dim=(2,3)).abs().square().mean(-1).flatten(2)
        loss=-10*torch.log10((actual.gather(2,pred_beam[...,None])[...,0]/actual.amax(-1).clamp_min(1e-30)).clamp_min(1e-30))
        scores.append(structural-.05*covariance_error(value,target).mean(1)-.02*loss.mean(1))
    return float(torch.cat(scores).mean())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:1');parser.add_argument('--control',action='store_true');parser.add_argument('--fold',type=int)
    args=parser.parse_args();torch.set_num_threads(8);device=torch.device(args.device);torch.cuda.set_device(device)
    prepared=PATHS.artifacts/'measured/dc01_v1';foundation=PATHS.artifacts/'measured/dc01_m3_foundation_v1'
    support=PATHS.artifacts/'measured/dc01_spectral_v1/merged_support'
    out=PATHS.artifacts/('measured/dc01_m3_angular_shuffle_v1' if args.control else 'measured/dc01_m3_angular_v1');out.mkdir(parents=True,exist_ok=True)
    h=np.load(prepared/'channels.npy',mmap_mode='r');meta=np.load(prepared/'selection.npz');split=json.loads((foundation/'split_manifest.json').read_text())
    raw={k:np.load(support/f'{k}.npy',mmap_mode='r') for k in ['coordinate','features','mass','mask','sequence','depth','interaction']}
    for fold in (range(5) if args.fold is None else [args.fold]):
        torch.manual_seed(20261016+fold)
        folder=out/f'fold{fold}';folder.mkdir(exist_ok=True)
        identity=dict(code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            inherited=hashlib.sha256((Path(__file__).parent/'dc01_state.py').read_bytes()).hexdigest(),
            original=hashlib.sha256((PATHS.root/'final_code/V16_reorganized_submission/code/round2/physical_ai_submission/component_state/model.py').read_bytes()).hexdigest(),
            split=split['sha256'],support=hashlib.sha256((support/'support_manifest.json').read_bytes()).hexdigest(),
            foundation=hashlib.sha256((foundation/f'fold{fold}/foundation_report.json').read_bytes()).hexdigest(),
            control=args.control,epochs=48,batch=64,seed=20261016+fold)
        if (folder/'training_identity.json').exists():assert json.loads((folder/'training_identity.json').read_text())==identity
        else:atomic_json(folder/'training_identity.json',identity)
        if (folder/'angular_report.json').exists():print('RESUME_ANGULAR',fold,args.control,flush=True);continue
        started=time.time();torch.cuda.reset_peak_memory_stats(device)
        ids={k:np.asarray(split['splits'][fold][k]) for k in ['fit','val','test']}
        banks={name:input_bank(rows,np.load(foundation/f'fold{fold}/{name}_base.npy',mmap_mode='r'),raw,meta,device,args.control) for name,rows in ids.items()}
        observed=banks['fit']['base'][banks['fit']['mask']];mean=observed.mean(0);std=observed.std(0).clamp_min(1e-3)
        for bank in banks.values():bank['base']=(bank['base']-mean)/std
        train_truth=torch.as_tensor(np.array(h[ids['fit']]),device=device).reshape(-1,1,8,4,256)
        val_truth=torch.as_tensor(np.array(h[ids['val']]),device=device).reshape(-1,1,8,4,256)
        model=AngularState().to(device);optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.001,weight_decay=.0001)
        best=validation(model,banks['val'],val_truth,baseline=True);best_epoch=0;best_state=None;first_epoch=1;history=[]
        if (folder/'resume.pt').exists():
            s=torch.load(folder/'resume.pt',map_location=device,weights_only=False)
            model.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer']);best=s['best'];best_epoch=s['best_epoch'];best_state=s['best_state'];history=s['history'];first_epoch=s['epoch']+1
            torch.set_rng_state(s['cpu_rng'].cpu());torch.cuda.set_rng_state(s['cuda_rng'].cpu(),device)
        for epoch in range(first_epoch,49):
            model.train();order=torch.randperm(len(train_truth),device=device);losses=[]
            for first in range(0,len(order),64):
                index=order[first:first+64];data={k:v[index] for k,v in banks['fit'].items()}
                value,state,alpha=model(data);target=train_truth[index]
                loss=1-cosine(pas(value),pas(target),(2,3)).mean()+.25*covariance_error(value,target).mean()+.001*state['score_residual'].square().mean()+.001*state['coordinate_residual'][...,:2].square().mean()
                if not torch.isfinite(loss):raise RuntimeError('Non-finite angular objective')
                optimizer.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(grad):raise RuntimeError('Non-finite angular gradient')
                optimizer.step();losses.append(float(loss.detach()))
            if epoch==1 or epoch%4==0:
                score=validation(model,banks['val'],val_truth)
                if score>best+1e-7:best=score;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                row=dict(epoch=epoch,loss=float(np.mean(losses)),validation_utility=score,best_epoch=best_epoch);history.append(row)
                atomic_torch(folder/'resume.pt',dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),best=best,best_epoch=best_epoch,best_state=best_state,history=history,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device)))
                print(json.dumps(dict(fold=fold,control=args.control,**row)),flush=True)
        if best_state is not None:model.load_state_dict(best_state)
        model.eval();atomic_torch(folder/'angular_model.pt',dict(model=best_state,selected_epoch=best_epoch,feature_mean=mean.cpu(),feature_std=std.cpu(),identity=identity))
        def predict(name):
            values=[];alphas=[]
            with torch.no_grad():
                bank=banks[name]
                for first in range(0,len(bank['channel']),64):
                    index=slice(first,first+64);data={k:v[index] for k,v in bank.items()}
                    if best_epoch==0:value=data['channel'];alpha=torch.zeros((len(value),1),device=device)
                    else:value,_,alpha=model(data)
                    values.append(value[:,0].cpu().numpy());alphas.append(alpha.flatten().cpu().numpy())
            return np.concatenate(values).reshape(-1,2,8,4,256),np.concatenate(alphas).reshape(-1,2)
        fit_prediction,_=predict('fit');fit_truth=np.asarray(h[ids['fit']])
        coefficient=np.sum(fit_prediction.conj()*fit_truth,axis=(0,2,3,4),dtype=np.complex128)/np.sum(np.abs(fit_prediction)**2,axis=(0,2,3,4),dtype=np.float64)
        prediction,alpha=predict('test');np.save(folder/'test_prediction.npy',prediction);np.save(folder/'test_alpha.npy',alpha)
        diagnostics=prediction*coefficient[None,:,None,None,None]
        points={};mean_points={}
        with torch.no_grad():
            for first in range(0,len(prediction),24):
                target=torch.as_tensor(np.array(h[ids['test'][first:first+24]]),device=device)
                for source,collection in [(prediction,points),(diagnostics,mean_points)]:
                    for k,v in metric_points(torch.as_tensor(source[first:first+24],device=device),target).items():collection.setdefault(k,[]).append(v)
        points={k:np.concatenate(v) for k,v in points.items()};mean_points={k:np.concatenate(v) for k,v in mean_points.items()}
        np.savez(folder/'test_metrics.npz',source_id=meta['source_id'][ids['test']],**points)
        np.savez(folder/'conditional_mean_metrics.npz',source_id=meta['source_id'][ids['test']],**mean_points)
        report=dict(identity=identity,fold=fold,control=args.control,selected_epoch=best_epoch,validation_utility=best,
            test=aggregate(points),conditional_mean_diagnostic=aggregate(mean_points),conditional_coefficient=np.c_[coefficient.real,coefficient.imag].tolist(),
            gate_mean=alpha.mean(0).tolist(),elapsed_s=time.time()-started,gpu=torch.cuda.get_device_name(device),
            peak_gpu_GiB=torch.cuda.max_memory_allocated(device)/1024**3,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        atomic_json(folder/'angular_report.json',report);print(json.dumps(dict(stage='complete',fold=fold,control=args.control,epoch=best_epoch,seconds=report['elapsed_s'])),flush=True)
        del model,optimizer,banks,train_truth,val_truth;torch.cuda.empty_cache()


if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise

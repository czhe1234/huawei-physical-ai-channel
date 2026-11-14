"""Measured energy field + phase-invariant spectra + declared donor phase."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import torch

from project_paths import PATHS
from measured_runtime import aggregate,metric_points,normalize,pas,pdp,project,safe_sqrt,unit_complex


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,path)


def utility(m):
    return m['structural_score']-.05*m['covariance_error']-.02*m['beam_loss_db']


def rescale(h,power):
    current=h.abs().square().sum((2,3,4),keepdim=True)
    return h*safe_sqrt(power[...,None,None,None]/current.clamp_min(1e-30))


class MeasuredField:
    def __init__(self,truth,position,panels,fit,device):
        self.fit=np.asarray(fit);self.position=position;self.panels=panels;self.device=device
        # The only measurement store visible to any prediction routine is fit H.
        self.h=torch.as_tensor(np.array(truth[self.fit]),device=device)
        self.power=self.h.abs().square().sum((2,3,4),dtype=torch.float64).cpu().numpy()
        self.log_power=10*np.log10(self.power)
        self.tree=cKDTree(position[fit,:2]);self.cache={}
        self.joint=torch.fft.ifft(torch.fft.fft2(self.h,dim=(2,3)),dim=-1)
        self.spectra=dict(pas=pas(self.h),pdp=pdp(self.h),
                          joint=normalize(self.joint.abs().square(),(2,3,4)))
        self.sqrt_spectra={k:safe_sqrt(v) for k,v in self.spectra.items()}
        self.pathloss=[]
        for panel in range(2):
            x=self.range_features(position[self.fit],panel)
            coefficient=np.linalg.solve(x.T@x+np.diag([0.,.001]),x.T@self.log_power[:,panel])
            self.pathloss.append((coefficient,self.log_power[:,panel]-x@coefficient))

    def range_features(self,position,panel):
        distance=np.maximum(np.linalg.norm(position-self.panels[panel],axis=-1),.1)
        return np.c_[np.ones(len(position)),np.log10(distance)]

    def weights(self,query,kind,k,exclude=False):
        key=(tuple(np.asarray(query).tolist()),kind,k,exclude)
        if key in self.cache:return self.cache[key]
        distance,ids=self.tree.query(self.position[query,:2],k=k+int(exclude))
        if k+int(exclude)==1:distance=distance[:,None];ids=ids[:,None]
        if exclude:
            distance=np.where(self.fit[ids]==np.asarray(query)[:,None],np.inf,distance)
            take=np.argsort(distance,axis=1)[:,:k]
            ids=np.take_along_axis(ids,take,axis=1);distance=np.take_along_axis(distance,take,axis=1)
        if kind=='idw':
            weight=1/np.maximum(distance,.05)**2;weight/=weight.sum(1,keepdims=True)
        elif kind=='nearest':weight=np.ones((len(query),1))
        else:
            weight=[]
            for donor,d in zip(ids,distance):
                scale=max(float(d[-1]),.1);matrix=cdist(self.position[self.fit[donor],:2],self.position[self.fit[donor],:2])/scale
                system=np.zeros((k+1,k+1));system[:k,:k]=matrix+.003*np.eye(k)
                system[:k,k]=1;system[k,:k]=1
                weight.append(np.linalg.solve(system,np.r_[d/scale,1])[:k])
            weight=np.asarray(weight)
        self.cache[key]=(ids,weight);return ids,weight

    def power_prediction(self,query,spec,exclude=False):
        kind,k=spec
        ids,w=self.weights(query,'idw' if kind=='pathloss' else kind,k,exclude)
        prediction=[]
        for panel in range(2):
            if kind=='pathloss':
                coefficient,residual=self.pathloss[panel]
                value=self.range_features(self.position[query],panel)@coefficient+(w*residual[ids]).sum(1)
            else:value=(w*self.log_power[ids,panel]).sum(1)
            prediction.append(np.clip(value,self.log_power[:,panel].min()-3,self.log_power[:,panel].max()+3))
        return np.stack(prediction,axis=1)

    @torch.inference_mode()
    def prediction(self,query,spectral,power_specs,mode,exclude=False):
        log_power=np.stack([self.power_prediction(query,power_specs[p],exclude)[:,p] for p in range(2)],axis=1)
        target_power=torch.as_tensor(10**(log_power/10),device=self.device,dtype=torch.float32)
        ids,w=self.weights(query,*spectral,exclude)
        donor=torch.as_tensor(ids,device=self.device)
        weight=torch.as_tensor(w,device=self.device,dtype=torch.float32)
        targets={}
        for key,bank in self.sqrt_spectra.items():
            value=torch.einsum('bk,bkphvf->bphvf',weight,bank[donor]).clamp_min(0).square()
            targets[key]=normalize(value,(2,3,4) if key=='joint' else ((2,3) if key=='pas' else (-1,)))
        nearest,_=self.weights(query,'nearest',1,exclude)
        anchor=self.h[torch.as_tensor(nearest[:,0],device=self.device)]
        if mode=='marginal':h=anchor
        else:
            reference=self.joint[torch.as_tensor(nearest[:,0],device=self.device)]
            joint=safe_sqrt(targets['joint'])*unit_complex(reference)
            h=torch.fft.ifft2(torch.fft.fft(joint,dim=-1),dim=(2,3))
        h=rescale(h,target_power)
        if mode!='joint':h=project(h,targets['pas'],targets['pdp'],1,1,4)
        return rescale(h,target_power),log_power


@torch.inference_mode()
def evaluate(truth,rows,predict,device):
    points={}
    for first in range(0,len(rows),24):
        selected=rows[first:first+24]
        value=predict(selected)
        reference=torch.as_tensor(np.array(truth[selected]),device=device)
        for k,v in metric_points(value,reference).items():points.setdefault(k,[]).append(v)
    return {k:np.concatenate(v) for k,v in points.items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:1');parser.add_argument('--fold',type=int)
    args=parser.parse_args();torch.set_num_threads(8);device=torch.device(args.device);torch.cuda.set_device(device)
    prepared=PATHS.artifacts/'measured/dc01_v1';out=PATHS.artifacts/'measured/dc01_m3_foundation_v1'
    original=PATHS.artifacts/'measured/dc01_baselines_v1'
    out.mkdir(parents=True,exist_ok=True);shutil.copyfile(original/'split_manifest.json',out/'split_manifest.json')
    h=np.load(prepared/'channels.npy',mmap_mode='r');meta=np.load(prepared/'selection.npz');split=json.loads((out/'split_manifest.json').read_text())
    code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    runtime=hashlib.sha256((Path(__file__).parent/'measured_runtime.py').read_bytes()).hexdigest()
    power_bank=[('nearest',1)]+[('idw',k) for k in (4,8,16,32)]+[('rbf',k) for k in (16,32,64)]+[('pathloss',k) for k in (8,16)]
    spectra=[('idw',k) for k in (4,8,16,32)]+[('rbf',k) for k in (16,32,64)]
    for fold in (range(5) if args.fold is None else [args.fold]):
        folder=out/f'fold{fold}';folder.mkdir(exist_ok=True)
        identity=dict(code=code,runtime=runtime,split=split['sha256'],preparation=json.loads((prepared/'data_audit.json').read_text())['preparation_sha256'])
        if (folder/'foundation_report.json').exists():
            assert json.loads((folder/'foundation_report.json').read_text())['identity']==identity
            print('RESUME_FOUNDATION',fold,flush=True);continue
        started=time.time();torch.cuda.reset_peak_memory_stats(device)
        ids={k:np.asarray(split['splits'][fold][k]) for k in ['fit','val','test']}
        model=MeasuredField(h,meta['position'],meta['panel_position'],ids['fit'],device)
        val_power=10*np.log10(np.sum(np.abs(np.asarray(h[ids['val']]))**2,axis=(2,3,4),dtype=np.float64))
        errors=np.array([np.abs(model.power_prediction(ids['val'],spec)-val_power).mean(0) for spec in power_bank])
        selected=[power_bank[int(np.argmin(errors[:,p]))] for p in range(2)]
        validation={};config={}
        for spec in spectra:
            for mode in ['marginal','joint','joint_marginal']:
                name=f'{spec[0]}{spec[1]}_{mode}';config[name]=(spec,mode)
                points=evaluate(h,ids['val'],lambda q:model.prediction(q,spec,selected,mode)[0],device)
                validation[name]=aggregate(points)
        winner=max(validation,key=lambda n:utility(validation[n]));spec,mode=config[winner]
        atomic_json(folder/'frozen_selection.json',dict(identity=identity,power=selected,spectral=spec,mode=mode,winner=winner,validation=validation))
        for name,rows in ids.items():
            temp=folder/f'{name}_base.partial.npy'
            data=np.lib.format.open_memmap(temp,mode='w+',dtype=np.complex64,shape=(len(rows),2,8,4,256))
            powers=[]
            for first in range(0,len(rows),24):
                q=rows[first:first+24];value,log=model.prediction(q,spec,selected,mode,exclude=name=='fit')
                data[first:first+len(q)]=value.cpu().numpy();powers.append(log)
            data.flush();del data;os.replace(temp,folder/f'{name}_base.npy')
            np.save(folder/f'{name}_log_power.npy',np.concatenate(powers))
        saved=np.load(folder/'test_base.npy',mmap_mode='r');index={int(k):i for i,k in enumerate(ids['test'])}
        points=evaluate(h,ids['test'],lambda q:torch.as_tensor(np.array(saved[[index[int(k)] for k in q]]),device=device),device)
        np.savez(folder/'test_metrics.npz',source_id=meta['source_id'][ids['test']],**points)
        report=dict(identity=identity,fold=fold,selected_power=selected,selected_spectral=winner,validation=validation,
                    power_validation_mae=errors.tolist(),power_candidates=power_bank,test=aggregate(points),
                    elapsed_s=time.time()-started,gpu=torch.cuda.get_device_name(device),
                    peak_gpu_GiB=torch.cuda.max_memory_allocated(device)/1024**3,
                    peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        atomic_json(folder/'foundation_report.json',report)
        print(json.dumps(dict(fold=fold,power=selected,spectral=winner,seconds=report['elapsed_s'])),flush=True)
        del model;torch.cuda.empty_cache()


if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise

"""Strict spatial splits and measurement-native V16 carrier/spectral baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import torch

from measured_runtime import aggregate, metric_points, normalize, pas, pdp, project


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    os.replace(tmp,path)


def split_manifest(position, block_m=4.0, guard_m=0.30):
    """Positions only. Both panels and all repeated snapshots stay together."""
    tiles=np.floor(position[:,:2]/block_m).astype(np.int64)
    fold=(tiles[:,0]+2*tiles[:,1])%5
    result=[]
    for outer in range(5):
        test=np.flatnonzero(fold==outer)
        val=np.flatnonzero(fold==(outer+1)%5)
        possible=np.flatnonzero((fold!=outer)&(fold!=(outer+1)%5))
        held=np.r_[test,val]
        distance=cKDTree(position[held,:2]).query(position[possible,:2],k=1)[0]
        fit=possible[distance>=guard_m]
        if min(len(fit),len(val),len(test))<16:
            raise ValueError('Insufficient geometry coverage for registered 5-color spatial split')
        assert not set(fit)&set(test) and not set(fit)&set(val) and not set(val)&set(test)
        nearest=cKDTree(position[fit,:2]).query(position[test,:2],k=1)[0]
        result.append(dict(outer=outer,fit=fit.tolist(),val=val.tolist(),test=test.tolist(),
                           min_test_fit_distance=float(nearest.min()),
                           test_fit_distance_quantiles=np.quantile(nearest,[0,.25,.5,.75,1]).tolist()))
    digest=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return dict(block_m=block_m,guard_m=guard_m,fold=fold.tolist(),splits=result,sha256=digest)


def normalize_rows(w):
    total=w.sum(axis=-1,keepdims=True)
    if np.any(np.abs(total)<1e-10):
        raise ValueError('Unstable signed interpolation row')
    return w/total


def affine_weights(delta,distance,k):
    x=np.concatenate([np.ones((*delta[:,:k].shape[:2],1)),
                      delta[:,:k,:2]/np.maximum(distance[:,k-1,None,None],1e-3)],axis=-1)
    w=1/np.maximum(distance[:,:k],0.05)**2
    gram=np.einsum('bki,bk,bkj->bij',x,w,x)
    gram+=np.diag([1e-6,0.05,0.05])[None]
    rhs=np.zeros((len(x),3));rhs[:,0]=1
    solved=np.linalg.solve(gram,rhs)
    return normalize_rows(w*np.einsum('bki,bi->bk',x,solved))


class CarrierBank:
    names=['nearest']+[f'idw{k}' for k in (4,8,16,32)]+[f'radial{k}' for k in (4,8,16,32)]+[f'affine{k}' for k in (8,16,32,64)]+[f'array_affine{k}' for k in (8,16,32,64)]

    def __init__(self,h,position,panels,directions,fit,device,frequency=3.438e9,bandwidth=50e6):
        self.device=device
        self.position=position
        self.fit=np.asarray(fit)
        self.h=torch.as_tensor(np.asarray(h[self.fit]),dtype=torch.complex64,device=device)
        self.local_index=np.full(len(h),-1,np.int64)
        self.local_index[self.fit]=np.arange(len(self.fit))
        self.tree=cKDTree(position[fit,:2])
        self.panels=panels
        self.frequency=frequency
        self.bandwidth=bandwidth
        spacing=0.04364
        self.offset=[]
        for direction in directions:
            normal=direction/np.linalg.norm(direction)
            horizontal=np.cross([0,0,1],normal);horizontal/=np.linalg.norm(horizontal)
            vertical=np.cross(normal,horizontal)
            offset=((np.arange(8)-3.5)[:,None,None]*horizontal[None,None,:]+
                    (1.5-np.arange(4))[None,:,None]*vertical[None,None,:])*spacing
            self.offset.append(offset)
        self.offset=np.array(self.offset)
        self.ridge=None

    def neighbours(self,query,exclude_self=False,k=64):
        query=np.asarray(query)
        count=min(k+int(exclude_self),len(self.fit))
        distance,loc=self.tree.query(self.position[query,:2],k=count)
        if exclude_self:
            ids=self.fit[loc]
            distance=np.where(ids==query[:,None],np.inf,distance)
            order=np.argsort(distance,axis=1)[:,:min(k,count-1)]
            loc=np.take_along_axis(loc,order,1)
            distance=np.take_along_axis(distance,order,1)
        if loc.shape[1]<k:
            raise ValueError('At least 65 training positions are needed')
        return self.fit[loc[:,:k]],distance[:,:k]

    @torch.inference_mode()
    def bank(self,query,exclude_self=False):
        ids,distance=self.neighbours(query,exclude_self)
        donor_pos=self.position[ids]
        q=self.position[query]
        delta=donor_pos-q[:,None,:]
        values=self.h[torch.as_tensor(self.local_index[ids],device=self.device)]
        results=[values[:,0]]
        for k in (4,8,16,32):
            w=normalize_rows(1/np.maximum(distance[:,:k],0.05)**2)
            results.append(torch.einsum('bk,bkphvf->bphvf',torch.as_tensor(w,device=self.device,dtype=torch.complex64),values[:,:k]))
        radius_q=np.linalg.norm(q[:,None,:]-self.panels[None],axis=-1)
        radius_d=np.linalg.norm(donor_pos[:,:,None,:]-self.panels[None,None],axis=-1)
        df=(torch.arange(self.h.shape[-1],device=self.device)-self.h.shape[-1]/2)*(self.bandwidth/self.h.shape[-1])
        wave=2*torch.pi*(self.frequency+df)/299792458.0
        radial_delta=torch.as_tensor(radius_q[:,None,:]-radius_d,dtype=torch.float32,device=self.device)
        radial=values*torch.exp(-1j*radial_delta[:,:,:,None,None,None]*wave)
        for k in (4,8,16,32):
            w=normalize_rows(1/np.maximum(distance[:,:k],0.05)**2)
            results.append(torch.einsum('bk,bkphvf->bphvf',torch.as_tensor(w,device=self.device,dtype=torch.complex64),radial[:,:k]))
        local=[]
        for k in (8,16,32,64):
            w=affine_weights(delta,distance,k)
            local.append(w)
            results.append(torch.einsum('bk,bkphvf->bphvf',torch.as_tensor(w,device=self.device,dtype=torch.complex64),radial[:,:k]))
        uq=(q[:,None,:]-self.panels[None])/np.maximum(radius_q[...,None],1e-6)
        ud=(donor_pos[:,:,None,:]-self.panels[None,None])/np.maximum(radius_d[...,None],1e-6)
        direction_delta=uq[:,None,:,:]-ud
        array_distance=np.einsum('bkpc,phvc->bkphv',direction_delta,self.offset)
        array_phase=torch.exp(1j*torch.as_tensor(array_distance,device=self.device,dtype=torch.float32)[...,None]*wave)
        organized=radial*array_phase
        for k,w in zip((8,16,32,64),local):
            results.append(torch.einsum('bk,bkphvf->bphvf',torch.as_tensor(w,device=self.device,dtype=torch.complex64),organized[:,:k]))
        return torch.stack(results,dim=1)

    @torch.inference_mode()
    def fit_ridge(self,seed=20261015):
        rng=np.random.default_rng(seed)
        rows=rng.choice(self.fit,min(len(self.fit),512),replace=False)
        coefficient=rng.choice(8*4*self.h.shape[-1],128,replace=False)
        gram=torch.zeros(2,17,17,dtype=torch.complex128,device=self.device)
        rhs=torch.zeros(2,17,dtype=torch.complex128,device=self.device)
        for start in range(0,len(rows),16):
            ids=rows[start:start+16]
            values=self.bank(ids,True).flatten(3)[...,coefficient].to(torch.complex128)
            target=self.h[torch.as_tensor(self.local_index[ids],device=self.device)].flatten(2)[...,coefficient].to(torch.complex128)
            gram+=torch.einsum('bepc,bfpc->pef',values.conj(),values)
            rhs+=torch.einsum('bepc,bpc->pe',values.conj(),target)
        ridge=0.01*gram.diagonal(dim1=-2,dim2=-1).real.mean(-1)
        gram+=ridge[:,None,None]*torch.eye(17,device=self.device)[None]
        self.ridge=torch.linalg.solve(gram,rhs).to(torch.complex64)
        return self.ridge.cpu().numpy()

    @torch.inference_mode()
    def combined(self,bank):
        if self.ridge is None:raise ValueError('Ridge must be fitted on training positions')
        return torch.einsum('bephvf,pe->bphvf',bank,self.ridge)

    @torch.inference_mode()
    def rbf_endpoint(self,query,carrier):
        ids,distance=self.neighbours(query)
        neighbours=self.position[ids,:2]
        scale=np.maximum(distance[:,-1],0.1)
        w=[]
        for i in range(len(query)):
            matrix=cdist(neighbours[i],neighbours[i])/scale[i]
            n=len(matrix)
            system=np.zeros((n+1,n+1));system[:n,:n]=matrix+0.003*np.eye(n)
            system[:n,n]=1;system[n,:n]=1
            rhs=np.r_[distance[i]/scale[i],1]
            w.append(np.linalg.solve(system,rhs)[:n])
        weights=torch.as_tensor(np.asarray(w),device=self.device,dtype=torch.float32)
        donor=self.h[torch.as_tensor(self.local_index[ids],device=self.device)]
        shape=donor.shape
        flat=donor.reshape(-1,*shape[2:])
        p=pas(flat).reshape(*shape)
        d=pdp(flat).reshape(*shape)
        target_p=normalize(torch.einsum('bk,bkphvf->bphvf',weights,p.sqrt()).clamp_min(0).square(),(2,3))
        target_d=normalize(torch.einsum('bk,bkphvf->bphvf',weights,d.sqrt()).clamp_min(0).square(),(-1,))
        return project(carrier,target_p,target_d,1,1,4)


def evaluate_predictions(h,ids,predictions,device):
    accumulated={}
    for start in range(0,len(ids),32):
        selected=ids[start:start+32]
        truth=torch.as_tensor(np.asarray(h[selected]),device=device)
        for name,pred in predictions(start,min(start+32,len(ids))).items():
            point=metric_points(pred,truth)
            if name not in accumulated:accumulated[name]={k:[] for k in point}
            for key,value in point.items():accumulated[name][key].append(value)
    result={}
    for name,data in accumulated.items():
        result[name]={k:np.concatenate(v) for k,v in data.items()}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepared-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',default='cuda:3')
    parser.add_argument('--fold',type=int)
    args=parser.parse_args()
    torch.set_num_threads(8)
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    h=np.load(args.prepared_dir/'channels.npy',mmap_mode='r')
    m=np.load(args.prepared_dir/'selection.npz')
    audit=json.loads((args.prepared_dir/'data_audit.json').read_text())
    protocol=split_manifest(m['position'])
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'split_manifest.json',protocol)
    folds=range(5) if args.fold is None else [args.fold]
    for fold in folds:
        out=root/f'fold{fold}';out.mkdir(exist_ok=True)
        identity=dict(source_sha256=audit['source_sha256'],split_sha256=protocol['sha256'],
                      preparation_sha256=audit['preparation_sha256'],
                      selection_sha256=hashlib.sha256((args.prepared_dir/'selection.npz').read_bytes()).hexdigest(),
                      runtime_sha256=hashlib.sha256((Path(__file__).parent/'measured_runtime.py').read_bytes()).hexdigest(),
                      code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if (out/'baseline_report.json').exists():
            old=json.loads((out/'baseline_report.json').read_text())
            if old.get('identity')==identity:
                print('RESUME_FOLD',fold,flush=True);continue
        started=time.time()
        split=protocol['splits'][fold]
        fit,val,test=[np.array(split[k]) for k in ['fit','val','test']]
        bank=CarrierBank(h,m['position'],m['panel_position'],m['panel_direction'],fit,device)
        weights=bank.fit_ridge()
        np.save(out/'carrier_ridge.npy',weights)
        names=bank.names+['carrier_ridge','sqrt_rbf']

        def predict(ids):
            def portion(start,end):
                query=ids[start:end]
                b=bank.bank(query)
                mix=bank.combined(b)
                pred={name:b[:,i] for i,name in enumerate(bank.names)}
                pred['carrier_ridge']=mix
                pred['sqrt_rbf']=bank.rbf_endpoint(query,mix)
                return pred
            return portion

        val_points=evaluate_predictions(h,val,predict(val),device)
        validation={k:aggregate(v) for k,v in val_points.items()}
        winner=max(validation,key=lambda k:validation[k]['structural_score'])
        test_points=evaluate_predictions(h,test,predict(test),device)
        metrics={k:aggregate(v) for k,v in test_points.items()}
        for name,points in test_points.items():np.savez(out/f'{name}_metrics.npz',source_id=m['source_id'][test],**points)
        # Freeze the training-only selected carrier for downstream physical models.
        for split_name,ids in [('fit',fit),('val',val),('test',test)]:
            target=out/f'{split_name}_base.npy'
            tmp=out/f'{split_name}_base.partial.npy'
            result=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.complex64,shape=(len(ids),*h.shape[1:]))
            for start in range(0,len(ids),16):
                query=ids[start:start+16]
                b=bank.bank(query,exclude_self=split_name=='fit')
                mix=bank.combined(b)
                if winner=='sqrt_rbf':
                    # Exclude-self spectra must also exclude the same donor.
                    if split_name=='fit':
                        saved=bank.neighbours
                        bank.neighbours=lambda q,exclude_self=False,k=64:saved(q,True,k)
                        try:chosen=bank.rbf_endpoint(query,mix)
                        finally:bank.neighbours=saved
                    else:chosen=bank.rbf_endpoint(query,mix)
                elif winner=='carrier_ridge':chosen=mix
                else:chosen=b[:,bank.names.index(winner)]
                result[start:start+len(query)]=chosen.cpu().numpy()
            result.flush();del result;os.replace(tmp,target)
        frozen=np.load(out/'test_base.npy',mmap_mode='r')
        frozen_points=evaluate_predictions(h,test,
            lambda start,end:{'frozen':torch.as_tensor(np.asarray(frozen[start:end]),device=device)},device)['frozen']
        np.savez(out/'frozen_base_metrics.npz',source_id=m['source_id'][test],**frozen_points)
        report=dict(identity=identity,fold=fold,fit_count=len(fit),val_count=len(val),test_count=len(test),
                    selected_on_validation=winner,validation=validation,test=metrics,
                    frozen_base_test=aggregate(frozen_points),
                    elapsed_s=time.time()-started,
                    peak_gpu_GiB=torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else 0,
                    gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                    caveat='This is a measurement-native adaptation of the V16 carrier and spectral core, not competition checkpoints.')
        atomic_json(out/'baseline_report.json',report)
        print(json.dumps(dict(fold=fold,selected=winner,test=metrics[winner],seconds=report['elapsed_s'])),flush=True)
        del bank
        if device.type=='cuda':torch.cuda.empty_cache()


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc()
        raise

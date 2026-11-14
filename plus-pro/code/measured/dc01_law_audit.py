"""Identify measured power/coherence laws using fit labels only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree
import torch

from project_paths import PATHS
from dc01_data import read_record
from measured_runtime import cosine, normalize, pas, pdp, render


def atomic_json(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,path)


def interval(delta, positions):
    delta=np.asarray(delta)
    if delta.ndim>1:delta=delta.mean(tuple(range(1,delta.ndim)))
    _,group=np.unique(np.floor(positions[:,:2]/4),axis=0,return_inverse=True)
    keys=np.unique(group);n=np.array([(group==g).sum() for g in keys])
    s=np.array([delta[group==g].sum() for g in keys])
    draw=np.random.default_rng(20261016).integers(len(keys),size=(1000,len(keys)))
    value=s[draw].sum(1)/n[draw].sum(1)
    ci=np.quantile(value,[.025,.975]).tolist()
    return dict(mean=float(delta.mean()),ci95=ci,blocks=len(keys),positive_supported=bool(ci[0]>0))


def pairs(position, rows, limit=4096):
    k=min(13,len(rows));_,near=cKDTree(position[rows,:2]).query(position[rows,:2],k=k)
    source=np.repeat(rows,k-1);target=rows[near[:,1:]].ravel()
    indices=np.unique(np.sort(np.c_[source,target],axis=1),axis=0)
    if len(indices)>limit:indices=indices[np.random.default_rng(20261016).choice(len(indices),limit,replace=False)]
    return indices


@torch.inference_mode()
def pair_metrics(left,right,device):
    results={}
    for first in range(0,len(left),64):
        a=torch.as_tensor(np.asarray(left[first:first+64]),device=device,dtype=torch.complex64)
        b=torch.as_tensor(np.asarray(right[first:first+64]),device=device,dtype=torch.complex64)
        ea=a.abs().square().sum((2,3,4));eb=b.abs().square().sum((2,3,4))
        den=(ea*eb).sqrt().clamp_min(1e-30)
        cross=(a.conj()*b).sum((2,3))
        dot=cross.sum(-1)/den
        aligned=torch.fft.ifft(cross,n=8*a.shape[-1]).abs().amax(-1)*(8*a.shape[-1])/den
        values=dict(correlation=dot,phase_invariant=dot.abs(),phase_delay_invariant=aligned,
                    pas=cosine(pas(a),pas(b),(2,3)).mean(-1),
                    pdp=cosine(pdp(a),pdp(b),(-1,)).mean((2,3)),
                    power_difference_db=(10*torch.log10(ea/eb)).abs())
        for key,value in values.items():results.setdefault(key,[]).append(value.cpu().numpy())
    return {k:np.concatenate(v) for k,v in results.items()}


def phase_law(h,position,panels,fit,device):
    half=np.random.default_rng(20261016).permutation(fit);left,right=np.array_split(half,2)
    train=pairs(position,left);held=pairs(position,right)
    cm=pair_metrics(h[train[:,0]],h[train[:,1]],device)
    ch=pair_metrics(h[held[:,0]],h[held[:,1]],device)
    r=np.linalg.norm(position[:,None,:]-panels[None],axis=-1)
    dt=r[train[:,0]]-r[train[:,1]];dh=r[held[:,0]]-r[held[:,1]]
    beta=np.arange(-200.,200.001,.25);theory=2*np.pi*3.438e9/299792458.
    reports=[]
    for panel in range(2):
        score=(cm['correlation'][:,panel,None]*np.exp(-1j*dt[:,panel,None]*beta[None])).real.mean(0)
        chosen=float(beta[np.argmax(score)]);z=ch['correlation'][:,panel]
        raw=z.real;learned=(z*np.exp(-1j*dh[:,panel]*chosen)).real
        physical=(z*np.exp(-1j*dh[:,panel]*theory)).real
        wrong=(z*np.exp(1j*dh[:,panel]*chosen)).real
        shuffled=(z*np.exp(-1j*np.roll(dh[:,panel],1)*chosen)).real
        reports.append(dict(panel=panel,beta_learned=chosen,beta_theory=theory,
            train_real_coherence=float(score.max()),held_raw=float(raw.mean()),held_learned=float(learned.mean()),
            held_theoretical=float(physical.mean()),held_wrong_sign=float(wrong.mean()),
            held_shuffled_distance=float(shuffled.mean()),
            gain=interval(learned-raw,position[held[:,0]]),
            gain_over_shuffle=interval(learned-shuffled,position[held[:,0]]),
            held_phase_invariant=float(ch['phase_invariant'][:,panel].mean()),
            held_phase_delay_invariant=float(ch['phase_delay_invariant'][:,panel].mean())))
    distance=np.linalg.norm(position[held[:,0],:2]-position[held[:,1],:2],axis=-1)
    bins=[]
    for lo,hi in [(0,.3),(.3,.6),(.6,1.),(1.,2.),(2.,4.),(4.,100.)]:
        take=(distance>=lo)&(distance<hi)
        if take.any():bins.append(dict(distance_m=[lo,hi],pairs=int(take.sum()),
            **{k:np.asarray(v[take].real.mean(0)).tolist() for k,v in ch.items()}))
    log_left=10*np.log10(np.sum(np.abs(np.asarray(h[left]))**2,axis=(2,3,4),dtype=np.float64))
    log_right=10*np.log10(np.sum(np.abs(np.asarray(h[right]))**2,axis=(2,3,4),dtype=np.float64))
    distances,donor=cKDTree(position[left,:2]).query(position[right,:2],k=8)
    w=1/np.maximum(distances,.05)**2;w/=w.sum(1,keepdims=True)
    prediction=(w[...,None]*log_left[donor]).sum(1)
    global_prediction=np.median(log_left,axis=0)
    error=np.abs(prediction-log_right);global_error=np.abs(global_prediction-log_right)
    power_report=dict(log_idw8_mae_db=float(error.mean()),global_median_mae_db=float(global_error.mean()),
                      improvement=interval(global_error-error,position[right]))
    return dict(identification_points=len(left),held_points=len(right),train_pairs=len(train),held_pairs=len(held),
                panels=reports,distance_bins=bins,power=power_report)


@torch.inference_mode()
def geometry_audit(h,position,fit,base,raw,device):
    shuffled=np.random.default_rng(20261016).permutation(fit)
    points={}
    for first in range(0,len(fit),32):
        rows=fit[first:first+32]
        truth=torch.as_tensor(np.asarray(h[rows]),device=device)
        anchor=torch.as_tensor(np.asarray(base[first:first+32]),device=device)
        peak=pdp(anchor).mean((2,3)).argmax(-1)
        for name,ids in [('registered',rows),('shuffled',shuffled[first:first+32])]:
            mass=torch.as_tensor(np.array(raw['mass'][ids]),device=device)
            coordinate=torch.as_tensor(np.array(raw['coordinate'][ids]),device=device)
            coordinate[...,2]=(coordinate[...,2]+peak[...,None])%h.shape[-1]
            p,d=render(mass,coordinate,8,4,h.shape[-1])
            for key,value in dict(pas=cosine(p,pas(truth),(2,3)).mean(-1),
                                  pdp=cosine(d,pdp(truth),(-1,)).mean((2,3))).items():
                points.setdefault(name+'_'+key,[]).append(value.cpu().numpy())
    points={k:np.concatenate(v) for k,v in points.items()}
    return dict(**{k:float(v.mean()) for k,v in points.items()},
        pas_gain=interval(points['registered_pas']-points['shuffled_pas'],position[fit]),
        pdp_gain=interval(points['registered_pdp']-points['shuffled_pdp'],position[fit]))


def recurrence_audit(data,prepared,fit,device):
    metadata=np.load(prepared/'all_metadata.npz');selection=np.load(prepared/'selection.npz')
    pos=metadata['position'];stamp=metadata['time']
    _,group=np.unique(np.floor(pos[:,:2]/.2).astype(np.int64),axis=0,return_inverse=True)
    rows=np.flatnonzero(np.isin(group,selection['group'][fit]))
    local=np.asarray(list(cKDTree(pos[rows]).query_pairs(.02)),dtype=np.int64).reshape(-1,2)
    candidate=rows[local]
    candidate=candidate[np.abs(stamp[candidate[:,0]]-stamp[candidate[:,1]])>10]
    if not len(candidate):return dict(pairs=0,note='No fit-cell recurrence satisfying the registered tolerance')
    if len(candidate)>512:candidate=candidate[np.random.default_rng(20261016).choice(len(candidate),512,replace=False)]
    assignments=np.asarray([a['assignments'] for a in json.loads((data/'spec.json').read_text())['antennas']])
    offset=json.loads((data/'context/data/reftx-offsets-dichasus-dc01.json').read_text())
    correction=np.exp(1j*(np.asarray(offset['cpo'])[:,None]+2*np.pi*np.asarray(offset['sto'])[:,None]*np.arange(1024)/1024)).astype(np.complex64)
    raw=np.memmap(data/'dichasus-dc01.tfrecords',dtype=np.uint8,mode='r')
    cache={}
    for row in np.unique(candidate):
        x=read_record(raw,int(row))['csi'];v=np.fft.fftshift(x[...,0]+1j*x[...,1],axes=-1)*correction
        cache[int(row)]=v[assignments][...,::4].transpose(0,2,1,3)
    a=np.array([cache[int(i)] for i in candidate[:,0]]);b=np.array([cache[int(i)] for i in candidate[:,1]])
    m=pair_metrics(a,b,device)
    cfo=metadata['cfo'][rows]
    return dict(pairs=len(candidate),position_tolerance_m=.02,time_separation_min_s=10,
        distance_quantiles_m=np.quantile(np.linalg.norm(pos[candidate[:,0]]-pos[candidate[:,1]],axis=-1),[0,.5,1]).tolist(),
        time_separation_quantiles_s=np.quantile(np.abs(stamp[candidate[:,0]]-stamp[candidate[:,1]]),[0,.5,1]).tolist(),
        metrics={k:np.asarray(v.real.mean(0)).tolist() for k,v in m.items()},
        cfo_fit_quantiles=np.quantile(cfo,[.01,.5,.99]).tolist(),
        note='Near-position recurrences are not identical channels; phase/time alignment is oracle diagnostic only')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:1');args=parser.parse_args()
    torch.set_num_threads(8);device=torch.device(args.device);torch.cuda.set_device(device)
    prepared=PATHS.artifacts/'measured/dc01_v1';base=PATHS.artifacts/'measured/dc01_baselines_v1'
    support=PATHS.artifacts/'measured/dc01_spectral_v1/merged_support'
    out=PATHS.analysis/'results/measured_dc01_m3_audit_v2';out.mkdir(parents=True,exist_ok=True)
    identity=dict(code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  preparation=json.loads((prepared/'data_audit.json').read_text())['preparation_sha256'],
                  split=hashlib.sha256((base/'split_manifest.json').read_bytes()).hexdigest(),
                  support=hashlib.sha256((support/'support_manifest.json').read_bytes()).hexdigest())
    if (out/'audit.json').exists():
        old=json.loads((out/'audit.json').read_text());assert old['identity']==identity;print('RESUME_COMPLETE');return
    started=time.time();h=np.load(prepared/'channels.npy',mmap_mode='r');meta=np.load(prepared/'selection.npz')
    split=json.loads((base/'split_manifest.json').read_text());raw={k:np.load(support/f'{k}.npy',mmap_mode='r') for k in ['mass','coordinate']}
    reports=[]
    for fold in range(5):
        fit=np.array(split['splits'][fold]['fit']);t=time.time()
        r=dict(fold=fold,law=phase_law(h,meta['position'],meta['panel_position'],fit,device),
               geometry=geometry_audit(h,meta['position'],fit,np.load(base/f'fold{fold}/fit_base.npy',mmap_mode='r'),raw,device))
        r['elapsed_s']=time.time()-t;reports.append(r);atomic_json(out/f'fold{fold}.json',r)
        print(json.dumps(r),flush=True)
    recurrence=recurrence_audit(PATHS.datasets/'dichasus/dc01',prepared,np.array(split['splits'][0]['fit']),device)
    result=dict(status='complete',identity=identity,folds=reports,recurrence=recurrence,elapsed_s=time.time()-started,
                gpu=torch.cuda.get_device_name(device),peak_gpu_GiB=torch.cuda.max_memory_allocated(device)/1024**3,
                peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                interpretation='Fit-only law audit. No test-derived phase/energy/CFO/SNR is available as a predictor input.')
    atomic_json(out/'audit.json',result);print(json.dumps(dict(stage='complete',recurrence=recurrence,seconds=result['elapsed_s'])),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise

"""Measurement adaptation of the frozen V16 ComponentState architecture.

The original sequence/set encoders are imported from the reviewed final source.
Real dc01 has one mobile TX port: candidate mass is learned separately for each
physical RX panel; the competition's eight UE/polarization outputs are removed.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import torch.nn.functional as F

from project_paths import PATHS
from measured_runtime import aggregate,cosine,metric_points,normalize,pas,pdp,project,render

FINAL_SOURCE=PATHS.root/'final_code/V16_reorganized_submission/code'
sys.path.insert(0,str(FINAL_SOURCE))
from round2.physical_ai_submission.component_state.model import ComponentStateModel,ModelConfig


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,path)


def atomic_torch(path,value):
    tmp=path.with_suffix('.partial.pt');torch.save(value,tmp);os.replace(tmp,path)


class MeasuredComponentState(ComponentStateModel):
    def __init__(self):
        super().__init__(120,21,ModelConfig())
        self.port_head=nn.Linear(self.config.hidden,1)
        # These four tables are not referenced by the inherited V16 forward path.
        for module in (self.object_embedding,self.primitive_embedding,self.hash_embedding_a,self.hash_embedding_b):
            for parameter in module.parameters():parameter.requires_grad_(False)

    def forward(self,batch):
        base=self.base_input(batch['base'])
        sequence=self.sequence_state(batch['sequence_geometry'],batch['interactions'],batch['depth'])
        hidden=self.set_encoder(self.fusion(torch.cat((base,sequence),dim=-1)))
        score=self.config.score_limit*torch.tanh(self.score_head(hidden)[...,0])
        coordinate=torch.tanh(self.coordinate_head(hidden))*torch.tensor((2.,4.,2.),device=hidden.device)
        logits=(batch['base_log_mass']+score).masked_fill(~batch['mask'],-10000.)
        mass=torch.softmax(logits,dim=1)
        return dict(mass=mass,coordinate=batch['coordinate']+coordinate,
                    score_residual=score,coordinate_residual=coordinate)


def combine_support(geometry,out,points):
    manifest=json.loads((geometry/'generation_report.json').read_text())
    seeds=manifest['identity']['seeds']
    payload=dict(generator=manifest['identity'],points=points,
                 combiner_sha256=hashlib.sha256(inspect.getsource(combine_support).encode()).hexdigest())
    identity=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    if (out/'support_manifest.json').exists():
        old=json.loads((out/'support_manifest.json').read_text())
        if old['identity']==identity:return
        raise ValueError('Support identity changed; choose a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    shapes=dict(coordinate=(points,2,128,3),mass=(points,2,128),mask=(points,2,128),
                features=(points,2,128,24),sequence=(points,2,128,16,21),
                interaction=(points,2,128,16),depth=(points,2,128))
    dtype={k:np.float32 for k in shapes};dtype.update(mask=bool,interaction=np.int64,depth=np.int64)
    arrays={k:np.lib.format.open_memmap(out/f'{k}.partial.npy',mode='w+',dtype=dtype[k],shape=s) for k,s in shapes.items()}
    batch_size=manifest['identity']['batch']
    for panel in range(2):
        for first in range(0,points,batch_size):
            bundles=[]
            for configuration in ['base','scatter']:
                for seed in seeds:
                    p=geometry/f'panel{panel}'/configuration/f'seed{seed}'/f'batch{first//batch_size:04d}.npz'
                    with np.load(p,allow_pickle=False) as z:bundles.append({k:z[k] for k in z.files})
            for local,row in enumerate(bundles[0]['point_id']):
                bank={}
                for bundle in bundles:
                    for candidate in np.flatnonzero(bundle['mask'][local]):
                        key=int(bundle['identity'][local,candidate])
                        if key not in bank:
                            bank[key]=[0.,0,bundle,candidate]
                        bank[key][0]+=float(bundle['mass'][local,candidate])
                        bank[key][1]+=1
                chosen=sorted(bank.values(),key=lambda x:-x[0])[:128]
                total=sum(v[0] for v in chosen)
                for target,(energy,count,bundle,candidate) in enumerate(chosen):
                    for key in shapes:
                        arrays[key][row,panel,target]=bundle[key][local,candidate]
                    arrays['mass'][row,panel,target]=energy/max(total,1e-30)
                    arrays['features'][row,panel,target,13]=count/(2*len(seeds))
            print(json.dumps(dict(stage='merge_support',panel=panel,first=first)),flush=True)
    for key in list(arrays):
        arrays[key].flush();del arrays[key]
        os.replace(out/f'{key}.partial.npy',out/f'{key}.npy')
    atomic_json(out/'support_manifest.json',dict(identity=identity,points=points,seeds=seeds,
                    note='Global dictionary deduplication; no adjacent-only quantized-cell assumption'))


def peak_labels(truth,coordinate,mask):
    """One-to-one energy-weighted matching, using training H only."""
    h=np.asarray(truth)
    joint=np.abs(np.fft.ifft(np.fft.fft2(h,axes=(0,1)),axis=-1))**2
    local=maximum_filter(joint,size=(3,3,3),mode='wrap')
    indices=np.argwhere((joint>=local)&(joint>0))
    energy=joint[tuple(indices.T)]
    order=np.argsort(-energy)[:24];indices=indices[order];energy=energy[order]
    peaks=indices.astype(np.float64)
    period=np.array(h.shape)
    log=np.log(np.maximum(joint,1e-30))
    for j,index in enumerate(indices):
        for axis in range(3):
            left=index.copy();right=index.copy()
            left[axis]=(left[axis]-1)%period[axis];right[axis]=(right[axis]+1)%period[axis]
            a,b,c=log[tuple(left)],log[tuple(index)],log[tuple(right)]
            denominator=a-2*b+c
            if abs(denominator)>1e-8:peaks[j,axis]+=np.clip(0.5*(a-c)/denominator,-0.5,0.5)
    mass=np.zeros(len(coordinate),np.float32)
    delta=np.zeros_like(coordinate,dtype=np.float32)
    matched=np.zeros(len(coordinate),bool)
    valid=np.flatnonzero(mask)
    if not len(valid) or not len(peaks):return mass,delta,matched
    difference=(peaks[:,None,:]-coordinate[None,valid,:]+period/2)%period-period/2
    distance=np.sum((difference/np.array([1.,1.,2.]))**2,axis=-1)
    # Coverage reward prevents a small candidate set matching only weak peaks.
    cost=(np.minimum(distance,16.)-16.)*np.sqrt(energy/max(float(energy.sum()),1e-30))[:,None]
    rows,cols=linear_sum_assignment(cost)
    for row,col in zip(rows,cols):
        index=valid[col]
        mass[index]=energy[row]
        delta[index]=np.clip(difference[row,col],-np.array([2.,4.,2.]),np.array([2.,4.,2.]))
        matched[index]=True
    mass/=max(float(mass.sum()),1e-30)
    return mass,delta,matched


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepared-dir',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path,required=True)
    parser.add_argument('--geometry-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--support-dir',type=Path)
    parser.add_argument('--device',default='cuda:3')
    parser.add_argument('--epochs',type=int,default=48)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--fold',type=int)
    args=parser.parse_args()
    torch.set_num_threads(8)
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    h=np.load(args.prepared_dir/'channels.npy',mmap_mode='r')
    meta=np.load(args.prepared_dir/'selection.npz')
    split=json.loads((args.baseline_dir/'split_manifest.json').read_text())
    support=args.support_dir or out/'merged_support'
    combine_support(args.geometry_dir,support,len(h))
    raw={k:np.load(support/f'{k}.npy',mmap_mode='r') for k in ['coordinate','mass','mask','features','sequence','interaction','depth']}
    folds=range(5) if args.fold is None else [args.fold]
    for fold in folds:
        torch.manual_seed(20261015+fold);np.random.seed(20261015+fold)
        folder=out/f'fold{fold}';folder.mkdir(exist_ok=True)
        identity=dict(split_sha256=split['sha256'],
                      preparation_sha256=json.loads((args.prepared_dir/'data_audit.json').read_text())['preparation_sha256'],
                      original_model_sha256=hashlib.sha256((FINAL_SOURCE/'round2/physical_ai_submission/component_state/model.py').read_bytes()).hexdigest(),
                      selection_sha256=hashlib.sha256((args.prepared_dir/'selection.npz').read_bytes()).hexdigest(),
                      baseline_report_sha256=hashlib.sha256((args.baseline_dir/f'fold{fold}'/'baseline_report.json').read_bytes()).hexdigest(),
                      support_sha256=hashlib.sha256((support/'support_manifest.json').read_bytes()).hexdigest(),
                      code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      runtime_sha256=hashlib.sha256((Path(__file__).parent/'measured_runtime.py').read_bytes()).hexdigest(),
                      epochs=args.epochs,batch_size=args.batch_size)
        if (folder/'training_identity.json').exists():
            old=json.loads((folder/'training_identity.json').read_text())
            if old!=identity:raise ValueError('Training inputs changed; use a fresh output directory')
        else:atomic_json(folder/'training_identity.json',identity)
        if (folder/'state_report.json').exists():
            old=json.loads((folder/'state_report.json').read_text())
            if old.get('identity')!=identity:raise ValueError('Completed model identity mismatch')
            print('RESUME_STATE_FOLD',fold,flush=True);continue
        started=time.time()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        manifest=split['splits'][fold]
        ids={k:np.array(manifest[k]) for k in ['fit','val','test']}
        assert not set(ids['fit'])&set(ids['test'])
        assert not set(ids['val'])&set(ids['test'])
        base={k:np.load(args.baseline_dir/f'fold{fold}'/f'{k}_base.npy',mmap_mode='r') for k in ids}
        input_bank={}
        for name,rows in ids.items():
            base_h=torch.as_tensor(np.asarray(base[name]),device=device)
            coordinate=np.array(raw['coordinate'][rows],copy=True)
            coarse_delay=pdp(base_h).mean(dim=(2,3)).argmax(dim=-1).cpu().numpy()
            coordinate[...,2]=(coordinate[...,2]+coarse_delay[...,None])%h.shape[-1]
            local=np.array(raw['features'][rows],copy=True)
            period=np.array([8,4,h.shape[-1]])
            local[...,:3]=np.sin(2*np.pi*coordinate/period)
            local[...,3:6]=np.cos(2*np.pi*coordinate/period)
            local[...,6:9]=coordinate/period
            anchor_p=pas(base_h).mean(dim=-1).flatten(2)
            anchor_d=pdp(base_h).mean(dim=(2,3)).reshape(len(rows),2,64,-1).sum(-1)
            anchor=torch.cat([anchor_p.clamp_min(1e-8).log(),anchor_d.clamp_min(1e-8).log()],dim=-1)
            anchor=anchor[:,:,None,:].expand(-1,-1,128,-1).cpu().numpy()
            features=np.concatenate([local,anchor],axis=-1)
            assert features.shape[-1]==120
            input_bank[name]=dict(
                base=torch.as_tensor(features.reshape(-1,128,120),device=device,dtype=torch.float32),
                sequence_geometry=torch.as_tensor(np.array(raw['sequence'][rows]).reshape(-1,128,16,21),device=device),
                interactions=torch.as_tensor(np.array(raw['interaction'][rows]).reshape(-1,128,16),device=device),
                depth=torch.as_tensor(np.array(raw['depth'][rows]).reshape(-1,128),device=device),
                coordinate=torch.as_tensor(coordinate.reshape(-1,128,3),device=device,dtype=torch.float32),
                base_log_mass=torch.as_tensor(np.log(np.maximum(raw['mass'][rows],1e-30)).reshape(-1,128),device=device),
                mask=torch.as_tensor(np.array(raw['mask'][rows]).reshape(-1,128),device=device),
                channel=base_h.reshape(-1,1,8,4,h.shape[-1]))
        valid_features=input_bank['fit']['base'][input_bank['fit']['mask']]
        mean=valid_features.mean(dim=0);std=valid_features.std(dim=0).clamp_min(1e-3)
        for bank in input_bank.values():bank['base']=(bank['base']-mean)/std
        train_truth=torch.as_tensor(np.asarray(h[ids['fit']]),device=device).reshape(-1,1,8,4,h.shape[-1])
        val_truth=torch.as_tensor(np.asarray(h[ids['val']]),device=device).reshape(-1,1,8,4,h.shape[-1])
        label_file=folder/'training_labels.npz'
        if label_file.exists():
            with np.load(label_file) as z:labels={k:torch.as_tensor(z[k],device=device) for k in z.files}
        else:
            mass=[];delta=[];matched=[]
            coordinates=input_bank['fit']['coordinate'].cpu().numpy();masks=input_bank['fit']['mask'].cpu().numpy()
            for j,truth in enumerate(np.asarray(h[ids['fit']]).reshape(-1,8,4,h.shape[-1])):
                m,d,valid=peak_labels(truth,coordinates[j],masks[j]);mass.append(m);delta.append(d);matched.append(valid)
            values=dict(mass=np.array(mass),delta=np.array(delta),matched=np.array(matched))
            np.savez_compressed(label_file,**values)
            labels={k:torch.as_tensor(v,device=device) for k,v in values.items()}
        model=MeasuredComponentState().to(device)
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=0.001,weight_decay=0.0001)

        def subset(bank,index):return {k:v[index] for k,v in bank.items()}

        def prediction(bank,index):
            data=subset(bank,index);state=model(data)
            target_p,target_d=render(state['mass'][:,None],state['coordinate'][:,None],8,4,h.shape[-1])
            value=project(data['channel'],target_p,target_d,0.2,0.2,4)
            value=torch.where(data['mask'].any(-1)[:,None,None,None,None],value,data['channel'])
            return value,state

        @torch.inference_mode()
        def structural_validation():
            model.eval();total=0.;count=0
            for first in range(0,len(val_truth),args.batch_size):
                index=slice(first,first+args.batch_size)
                value,_=prediction(input_bank['val'],index)
                score=0.5*(cosine(pas(value),pas(val_truth[index]),(2,3)).mean()+
                           cosine(pdp(value),pdp(val_truth[index]),(-1,)).mean())
                n=len(value);total+=float(score)*n;count+=n
            return total/count

        with torch.no_grad():
            baseline=0.5*(cosine(pas(input_bank['val']['channel']),pas(val_truth),(2,3)).mean()+
                          cosine(pdp(input_bank['val']['channel']),pdp(val_truth),(-1,)).mean())
        best=float(baseline);best_epoch=0;best_state=None;history=[];first_epoch=1
        if (folder/'state_resume.pt').exists():
            saved=torch.load(folder/'state_resume.pt',map_location=device,weights_only=False)
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            best=saved['best'];best_epoch=saved['best_epoch'];best_state=saved['best_state'];history=saved['history'];first_epoch=saved['epoch']+1
            torch.set_rng_state(saved['torch_rng'].cpu())
            if device.type=='cuda':torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),device)
        for epoch in range(first_epoch,args.epochs+1):
            model.train();order=torch.randperm(len(train_truth),device=device);losses=[]
            for first in range(0,len(order),args.batch_size):
                index=order[first:first+args.batch_size]
                prediction_h,state=prediction(input_bank['fit'],index)
                truth=train_truth[index]
                spectral=2-cosine(pas(prediction_h),pas(truth),(2,3)).mean()-cosine(pdp(prediction_h),pdp(truth),(-1,)).mean()
                target_mass=labels['mass'][index]
                mass_loss=-(target_mass*state['mass'].clamp_min(1e-9).log()).sum(-1).mean()
                matches=labels['matched'][index]
                coordinate_loss=F.smooth_l1_loss(state['coordinate_residual'][matches],labels['delta'][index][matches]) if matches.any() else spectral*0
                regularization=0.0001*state['score_residual'].square().mean()
                loss=spectral+0.1*mass_loss+0.1*coordinate_loss+regularization
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(grad):raise RuntimeError('Non-finite training gradient')
                optimizer.step();losses.append(float(loss.detach()))
            if epoch%4==0 or epoch==1:
                score=structural_validation()
                if score>best+1e-7:
                    best=score;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                row=dict(epoch=epoch,loss=float(np.mean(losses)),validation_structural=score,best_epoch=best_epoch)
                history.append(row);atomic_json(folder/'training_progress.json',dict(fold=fold,elapsed_s=time.time()-started,history=history))
                atomic_torch(folder/'state_resume.pt',dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    best=best,best_epoch=best_epoch,best_state=best_state,history=history,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None))
                print(json.dumps(dict(fold=fold,**row)),flush=True)
        # Model selection is frozen before held-out measurement labels are evaluated.
        if best_state is not None:model.load_state_dict(best_state)
        model.eval()
        atomic_torch(folder/'component_state.pt',dict(model=best_state,best_epoch=best_epoch,feature_mean=mean.cpu(),feature_std=std.cpu(),
                        split_sha256=split['sha256'],epochs=args.epochs))
        outputs=[]
        with torch.no_grad():
            for first in range(0,len(input_bank['test']['channel']),args.batch_size):
                index=slice(first,first+args.batch_size)
                if best_epoch==0:value=input_bank['test']['channel'][index]
                else:value,_=prediction(input_bank['test'],index)
                outputs.append(value[:,0].cpu().numpy())
        predicted=np.concatenate(outputs).reshape(-1,2,8,4,h.shape[-1])
        np.save(folder/'test_prediction.npy',predicted)
        data={}
        for first in range(0,len(predicted),32):
            values=metric_points(torch.as_tensor(predicted[first:first+32],device=device),torch.as_tensor(np.asarray(h[ids['test'][first:first+32]]),device=device))
            for key,value in values.items():data.setdefault(key,[]).append(value)
        data={k:np.concatenate(v) for k,v in data.items()}
        np.savez(folder/'test_metrics.npz',source_id=meta['source_id'][ids['test']],**data)
        physical_values=[]
        with torch.no_grad():
            bank=input_bank['test']
            for first in range(0,len(bank['channel']),args.batch_size):
                index=slice(first,first+args.batch_size)
                masses=normalize(torch.exp(bank['base_log_mass'][index]),(-1,))
                rp,rd=render(masses[:,None],bank['coordinate'][index,None],8,4,h.shape[-1])
                value=project(bank['channel'][index],rp,rd,0.2,0.2,4)
                value=torch.where(bank['mask'][index].any(-1)[:,None,None,None,None],value,bank['channel'][index])
                physical_values.append(value[:,0].cpu().numpy())
        physical=np.concatenate(physical_values).reshape(-1,2,8,4,h.shape[-1])
        physical_points={}
        for first in range(0,len(physical),32):
            points=metric_points(torch.as_tensor(physical[first:first+32],device=device),torch.as_tensor(np.asarray(h[ids['test'][first:first+32]]),device=device))
            for key,value in points.items():physical_points.setdefault(key,[]).append(value)
        physical_points={k:np.concatenate(v) for k,v in physical_points.items()}
        np.savez(folder/'physical_prior_metrics.npz',source_id=meta['source_id'][ids['test']],**physical_points)
        baseline_report=json.loads((args.baseline_dir/f'fold{fold}'/'baseline_report.json').read_text())
        selected=baseline_report['selected_on_validation']
        metrics=aggregate(data);reference=baseline_report['frozen_base_test']
        report=dict(identity=identity,fold=fold,selected_epoch=best_epoch,validation_baseline=float(baseline),validation_selected=best,
                    test=metrics,baseline_method=selected,baseline_test=reference,
                    raw_physical_prior_test=aggregate(physical_points),
                    structural_gain=metrics['structural_score']-reference['structural_score'],
                    split_sha256=split['sha256'],epochs=args.epochs,elapsed_s=time.time()-started,
                    parameters_total=sum(p.numel() for p in model.parameters()),
                    parameters_optimizer=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    peak_gpu_GiB=torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else 0,
                    gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                    adaptations=['Actual 8x4 single-polarization RX panel and one UE TX port',
                                 'Original V16 sequence/set encoder; dynamic spectral renderer',
                                 'Energy-weighted one-to-one FFT-component supervision',
                                 'Measurement-native carrier/sqrt-RBF foundation',
                                 'Fresh spatial split and training-only calibration',
                                 'Physical seed budget recorded separately; no competition weights reused'],
                    limitation='This evaluates the ported carrier + ComponentState + projection core. Competition-specific CNN/LinkGate/lifecycle checkpoints are not transferred.')
        atomic_json(folder/'state_report.json',report)
        print(json.dumps(dict(stage='fold_complete',fold=fold,test=metrics,gain=report['structural_gain'],seconds=report['elapsed_s'])),flush=True)
        del model,optimizer,input_bank,train_truth,val_truth
        if device.type=='cuda':torch.cuda.empty_cache()


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise

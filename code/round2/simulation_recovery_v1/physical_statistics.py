"""Streaming regenerated V16 physical statistics; no channels/GT/models are read.

Keep the original full Train site groups of16 and seed scheduling. Synthesize
only canonical500 CFRs from live V16 components. Welford revisions read an
immutable previous seed, so rerunning a partly committed batch is idempotent.
This is regenerated physical input, not a proof of old-cache numerical identity.
"""
from __future__ import annotations
import argparse,ast,ctypes,hashlib,importlib,importlib.util,json,os,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
from project_paths import PATHS
from round2.paper_baseline_adapters_v1.contract import SimulationInputs,sha256,save_json
from .geometry import VENDOR,OUTPUT,REPORT,vendor_identity

ALLOWED=("GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009","GPU-70b6fceb-c7c4-9467-3c81-097c9284c160")
# Canonical original source list is loaded and checked below, not guessed.
EXPECTED_SEEDS=(42,101,211,307,401,503,601,701,809,907,1009,1103,1201,1301,1409,1511,1601,1709,1801,1901,2003,2111,2203,2309)
SOURCE_CONFIGS={'base':'primary_d16_base_v4lambda_1m','scatter':'primary_d10_scatter015_v4lambda_1m'}
DOMAINS=('pas','pdp','ad','log_power','path_count')
ROOT=OUTPUT/'v16_n24_statistics'
LOG=REPORT/'statistics'
ORIGINAL=PATHS.round2_code/'generate_multiseed_sionna_dataset.py'

def definitions():
    tree=ast.parse(ORIGINAL.read_text());ns={'np':np,'TINY':1e-30,'NUM_SUBCARRIERS':192}
    keep={'score_aligned_statistics','domain_shape','count_shape','expand_count'}
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in keep]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ORIGINAL),'exec'),ns)
    seeds=None
    for x in tree.body:
        if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SEEDS' for t in x.targets):seeds=tuple(ast.literal_eval(x.value))
    if seeds!=EXPECTED_SEEDS:raise ValueError('Original24-seed order changed')
    return ns

def runtime():
    uuid=os.environ.get('CUDA_VISIBLE_DEVICES')
    if uuid not in ALLOWED:raise RuntimeError('Exactly one allocated full UUID must be visible')
    import torch,mitsuba as mi,drjit as dr
    if torch.cuda.device_count()!=1:raise RuntimeError('One logical GPU required')
    torch.cuda.set_device(0);torch.set_num_threads(4)
    if mi.variant() is None:mi.set_variant('cuda_ad_mono_polarized')
    if mi.variant()!='cuda_ad_mono_polarized':raise RuntimeError('Unexpected Mitsuba variant')
    lib=ctypes.CDLL(str(Path(dr.__file__).with_name('libdrjit-core.so')))
    fn=getattr(lib,'_Z19jit_cuda_set_devicei');fn.argtypes=[ctypes.c_int];fn.restype=None;fn(0)
    name='round2.physical_prior'
    if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=(VENDOR/'__init__.py').resolve():raise RuntimeError('Different physical runtime already imported')
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,VENDOR/'__init__.py',submodule_search_locations=[str(VENDOR)])
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return importlib.import_module(name+'.geometry_prior_engine'),importlib.import_module(name+'.scene_prior_data'),mi,dr,torch

def tensor_on_gpu(x,torch):
    # DrJit offers a Torch/DLPack view. CPU fallback only changes transfer cost.
    if hasattr(x,'torch'):return x.torch().detach()
    return torch.as_tensor(np.asarray(x.numpy()),device='cuda:0')

def coherent_cfr(coefficient,tau,frequencies_hz,torch,chunk=32):
    """[B,4,256,P] passband coeff + [B,P] -> [B,256,4,192].

    V16 component_response omits carrier path phase. Apply actual fc+df in
    FP64 exactly once. Element/polarization phases are already in coefficient.
    """
    f=torch.tensor(np.array(frequencies_hz,dtype=np.float64,copy=True),dtype=torch.float64,device=coefficient.device)
    if f.dtype!=torch.float64 or f.shape!=(192,):raise ValueError('Real192-tone physical axis required')
    out=[]
    for lo in range(0,len(f),chunk):
        phase=torch.remainder(-2*torch.pi*tau.double()[...,None]*f[None,None,lo:lo+chunk]+torch.pi,2*torch.pi)-torch.pi
        phasor=torch.complex(phase.cos(),phase.sin()).to(torch.complex64)
        out.append(torch.einsum('bump,bpf->bumf',coefficient,phasor))
    return torch.cat(out,-1).permute(0,2,1,3).contiguous()

def component_channel(components,local_indices,frequency,torch):
    idx=torch.tensor(local_indices,device='cuda:0',dtype=torch.long)
    rr,ii=components.a
    real=tensor_on_gpu(rr,torch).index_select(0,idx)
    imag=tensor_on_gpu(ii,torch).index_select(0,idx)
    if real.ndim!=5 or tuple(real.shape[1:4])!=(4,1,256):raise ValueError('Unexpected component coefficients/port layout')
    coefficient=torch.complex(real[:,:,0],imag[:,:,0]).to(torch.complex64)
    tau=tensor_on_gpu(components.tau,torch).index_select(0,idx)[:,0]
    valid=tensor_on_gpu(components.valid,torch).index_select(0,idx)[:,0].bool()
    if not bool((torch.isfinite(coefficient)|(~valid[:,None,None,:])).all()):raise ValueError('Nonfinite valid geometric coefficient')
    coefficient=torch.where(valid[:,None,None,:],coefficient,torch.zeros_like(coefficient))
    return coherent_cfr(coefficient,tau,frequency,torch),valid.sum(-1).cpu().numpy().astype(np.float32)

def fields(ns,rows=500):
    result={}
    for domain in DOMAINS:
        for kind in ('mean','m2','count'):
            shape=ns['count_shape'](domain,rows) if kind=='count' else ns['domain_shape'](domain,rows)
            result[domain+'.'+kind]=(shape,np.uint16 if kind=='count' else np.float32)
    return result

def load_arrays(directory,ns,create=False,previous=None):
    directory.mkdir(parents=True,exist_ok=True);arrays={}
    for key,(shape,dtype) in fields(ns).items():
        path=directory/(key+'.npy')
        if create:
            a=np.lib.format.open_memmap(path,mode='w+',shape=shape,dtype=dtype)
            a[:]=0 if previous is None else previous[key][:];a.flush()
        else:a=np.load(path,mmap_mode='r+',allow_pickle=False)
        if a.shape!=shape or a.dtype!=np.dtype(dtype):raise ValueError('Statistics revision layout differs')
        arrays[key]=a
    return arrays

def welford_assign(out,previous,rows,values,ns):
    for domain in DOMAINS:
        value=np.asarray(values[domain],np.float32);valid=np.asarray(values[domain+'_valid'],bool)
        oldcount=np.zeros(valid.shape,np.uint16) if previous is None else np.array(previous[domain+'.count'][rows])
        count=oldcount+valid.astype(np.uint16)
        oldmean=np.zeros(value.shape,np.float32) if previous is None else np.array(previous[domain+'.mean'][rows])
        oldm2=np.zeros(value.shape,np.float32) if previous is None else np.array(previous[domain+'.m2'][rows])
        expand=ns['expand_count'];mask=expand(domain,valid);den=expand(domain,np.maximum(count,1)).astype(np.float32)
        delta=value-oldmean;mean=np.where(mask,oldmean+delta/den,oldmean)
        m2=np.where(mask,oldm2+delta*(value-mean),oldm2)
        out[domain+'.mean'][rows]=mean;out[domain+'.m2'][rows]=m2;out[domain+'.count'][rows]=count
    for a in out.values():a.flush()

def original_batches(inputs):
    target=set(map(int,inputs.query_ids));result=[]
    for site in (0,1):
        ids=np.flatnonzero(inputs.site_id==site)
        for start in range(0,len(ids),16):
            group=ids[start:start+16]
            selected=[j for j,row in enumerate(group) if int(row) in target]
            if selected:result.append((site,group,np.asarray(selected,np.int64)))
    return result

def source_run(source,seeds=24,pilot_batches=0):
    if source not in SOURCE_CONFIGS or seeds not in (1,24) or pilot_batches<0:raise ValueError('Source/seed budget')
    inputs=SimulationInputs();ns=definitions();root=ROOT/source;root.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    groups=original_batches(inputs);rowmap={int(v):i for i,v in enumerate(inputs.query_ids)}
    config=dict(name=SOURCE_CONFIGS[source],seed_order=list(EXPECTED_SEEDS),max_depth=16 if source=='base' else 10,
      diffuse_reflection=source=='scatter',scattering_coefficient=0. if source=='base' else .15,
      samples_per_src=1000000,max_components_per_source=1000000,batch_size=16)
    contract=dict(schema='v16-regenerated-physical-source-v1',row_ids=inputs.query_ids.tolist(),config=config,
      input_identity=inputs.identity,geometry_fingerprint=inputs.geometry.geometry_fingerprint,
      source_files=vendor_identity(),statistics_source_sha256=sha256(ORIGINAL),
      CFR_convention='passband component coefficients times exp(-j2pi*(fc+df)*tau), FP64 Hz phase, exactly once',
      canonical_only_statistics=True,original_full16_target_groups=True,original_Train_seed_rule='seed+100000*site+first_original_train_row_in_group',
      original_group_count=len(groups),coordinate_frame='official Round2 map/TrainPos world XYZ metres',
      regenerated=True,old_cache_numerical_equivalence_verified=False,no_CSI=True,no_official_Test=True,
      allocated_GPU_UUID=os.environ.get('CUDA_VISIBLE_DEVICES'))
    key=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest();statepath=root/'state.json'
    if statepath.exists():
        state=json.loads(statepath.read_text())
        if state['contract_hash']!=key:raise ValueError('Preserve existing source revision with a different contract')
    else:
        state=dict(contract_hash=key,contract=contract,completed_seeds=[],current_directory=None,active_index=None)
        save_json(statepath,state)
    engine,data,mi,dr,torch=runtime();started=time.monotonic();batch_reports=[]
    xml=data.prepare_scene_files(PATHS.round2,root/'scene')
    for index,seed in enumerate(EXPECTED_SEEDS[:seeds],1):
        if seed in state['completed_seeds']:continue
        old=root/state['current_directory'] if state['current_directory'] else None
        previous=load_arrays(old,ns) if old is not None else None
        directory=root/f'running_n{index:02d}';init=directory/'initialization.json'
        if not init.exists():
            print(json.dumps(dict(stage='initialize_revision',source=source,index=index)),flush=True)
            arrays=load_arrays(directory,ns,create=True,previous=previous)
            np.save(directory/'done.npy',np.zeros(500,bool));save_json(init,dict(ready=True,index=index,seed=seed))
        else:arrays=load_arrays(directory,ns)
        done=np.load(directory/'done.npy',mmap_mode='r+');completed_groups=0
        state['active_index']=index;save_json(statepath,state)
        for site in (0,1):
            site_completed=0
            scene=engine.load_scene(str(xml));scene.frequency=inputs.geometry.frequency_center_hz
            scene.tx_array=engine.build_transmit_array(4.,.5);scene.rx_array=engine.build_receive_array()
            data.configure_scene_response(scene,config['scattering_coefficient'])
            bs=inputs.geometry.bs_positions_m[site]
            transmitter=engine.Source(name=f'{source}_site{site}',position=mi.Point3f(*bs.tolist()),
              look_at=data.tx_target(bs,data.SITE_YAW_DEG[site],data.SITE_TILT_DEG[site]))
            scene.add(transmitter);builder=engine.ScenePriorBuilder()
            for group_site,group,local in groups:
                if group_site!=site:continue
                rows=np.array([rowmap[int(group[i])] for i in local],np.int64)
                if np.all(done[rows]):continue
                if pilot_batches and site_completed>=max(1,pilot_batches//2):break
                for row in group:scene.add(engine.Target(name=f'train_{row}',position=mi.Point3f(*inputs.positions[row].tolist()),look_at=transmitter))
                tick=time.monotonic();trace_seed=int(seed)+100000*site+int(group[0])
                components=builder(scene,max_depth=config['max_depth'],max_components_per_source=config['max_components_per_source'],
                  samples_per_src=config['samples_per_src'],diffuse_reflection=config['diffuse_reflection'],seed=trace_seed)
                traced=time.monotonic()
                with torch.no_grad():channel,path_count=component_channel(components,local,inputs.geometry.frequencies_hz,torch)
                rendered=time.monotonic();values=ns['score_aligned_statistics'](channel.cpu().numpy())
                values['path_count']=path_count;values['path_count_valid']=np.ones(len(rows),bool)
                welford_assign(arrays,previous,rows,values,ns);done[rows]=True;done.flush()
                report=dict(source=source,seed=seed,site=site,original_group_ids=group.tolist(),canonical_row_ids=inputs.query_ids[rows].tolist(),
                  trace_seed=trace_seed,trace_seconds=traced-tick,CFR_seconds=rendered-traced,total_seconds=time.monotonic()-tick,
                  path_counts=path_count.tolist(),GPU_UUID=os.environ['CUDA_VISIBLE_DEVICES'])
                batch_reports.append(report);completed_groups+=1;site_completed+=1
                for row in group:scene.remove(f'train_{row}')
                del components,channel,values;dr.sync_thread()
                progress=dict(stage='physical_statistics',source=source,seed_index=index,seed=seed,canonical_done=int(done.sum()),
                  canonical_total=500,last_batch=report,elapsed_seconds=time.monotonic()-started)
                save_json(LOG/f'{source}_progress.json',progress);print(json.dumps(progress),flush=True)
            del scene,builder;dr.sync_thread()
        if not np.all(done):
            if not pilot_batches:raise RuntimeError('Incomplete source seed')
            estimate=(sum(x['total_seconds'] for x in batch_reports)/max(len(batch_reports),1))*len(groups)*24
            pilot=dict(status='bounded_physical_statistics_pilot',source=source,seed=seed,groups_completed=completed_groups,
              total_original_groups=len(groups),canonical_rows_completed=int(done.sum()),elapsed_seconds=time.monotonic()-started,
              batches=batch_reports,estimated_24seed_compute_seconds=estimate,full_seed_complete=False,
              state=PATHS.relative(statepath),no_CSI=True,trained=False,scored=False)
            save_json(LOG/f'{source}_pilot.json',pilot);print(json.dumps(pilot),flush=True);return pilot
        state['completed_seeds'].append(seed);state['current_directory']=directory.name;state['active_index']=None;save_json(statepath,state)
        print(json.dumps(dict(stage='seed_complete',source=source,count=index,seed=seed,elapsed_seconds=time.monotonic()-started)),flush=True)
        # Delete only this generator's own obsolete rolling intermediate, after
        # the new complete revision and pointer are durable. Legacy assets stay read-only.
        if old is not None:
            del previous
            if old.parent!=root or not old.name.startswith('running_n'):raise RuntimeError('Unsafe intermediate retirement')
            shutil.rmtree(old)
    result=dict(status='source_n24_complete' if len(state['completed_seeds'])==24 else 'source_n1_complete',
      source=source,seed_count=len(state['completed_seeds']),state=PATHS.relative(statepath),elapsed_seconds=time.monotonic()-started)
    save_json(LOG/f'{source}_result.json',result);print(json.dumps(result),flush=True)
    combine_if_complete(inputs)
    return result

def combine_if_complete(inputs=None):
    inputs=inputs or SimulationInputs();ns=definitions();arrays={};states={}
    for source in SOURCE_CONFIGS:
        statepath=ROOT/source/'state.json'
        if not statepath.exists():return None
        state=json.loads(statepath.read_text())
        if state['completed_seeds']!=list(EXPECTED_SEEDS):return None
        states[source]=state
        for key in fields(ns):
            path=ROOT/source/state['current_directory']/(key+'.npy')
            arrays[source+'.'+key]=dict(path=PATHS.relative(path),sha256=sha256(path))
    geometry_source=PATHS.artifacts/'round2/pre_sionna/round2_geometry_visibility_features.npz'
    with np.load(geometry_source,allow_pickle=False) as z:
        geometry=np.asarray(z['train'][inputs.query_ids],np.float32);names=z['names'].tolist()
    if geometry.shape!=(500,37) or len(names)!=37:raise ValueError('Geometry descriptor contract changed')
    path=ROOT/'geometry.npy';np.save(path,geometry);arrays['geometry']=dict(path=PATHS.relative(path),sha256=sha256(path))
    manifest=dict(schema='v16-n24-physical-statistics-v1',row_ids=inputs.query_ids.tolist(),source_configs=list(SOURCE_CONFIGS.values()),
      seed_count=24,seeds=list(EXPECTED_SEEDS),arrays=arrays,geometry_names=names,
      regenerated=True,old_cache_numerical_equivalence_verified=False,
      source_state_paths={s:PATHS.relative(ROOT/s/'state.json') for s in states},
      no_CSI=True,trained=False,scored=False)
    save_json(ROOT/'manifest.json',manifest);return manifest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',choices=['base','scatter'],required=True)
    parser.add_argument('--seeds',type=int,choices=[1,24],default=24);parser.add_argument('--pilot-batches',type=int,default=0)
    args=parser.parse_args()
    try:source_run(args.source,args.seeds,args.pilot_batches)
    except BaseException:
        LOG.mkdir(parents=True,exist_ok=True);(LOG/f'{args.source}_failure.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()

"""V2 regenerated prior for canonical500 query positions, grouped16 per site.

Same physical budgets/24 seeds as original, explicit changed target grouping.
This applies frozen V16 weights to fresh physical inputs; it is not recovery of
old-cache numbers. Pilot files remain separate from the original-group V1.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,importlib,json,os,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
from project_paths import PATHS
from round2.paper_baseline_adapters_v1.contract import SimulationInputs,sha256,save_json
from .geometry import OUTPUT,REPORT,vendor_identity
from .physical_statistics import definitions,runtime,component_channel,fields,load_arrays,welford_assign,EXPECTED_SEEDS,SOURCE_CONFIGS,ORIGINAL
ROOT=OUTPUT/'v16_n24_statistics_v3_query500_lossless'
LOG=REPORT/'statistics_v3_query500_lossless'

def query_batches(inputs):
    sites=inputs.site_id[inputs.query_ids];groups=[]
    for site in (0,1):
        rows=np.flatnonzero(sites==site)
        for start in range(0,len(rows),16):
            indices=rows[start:start+16];ids=inputs.query_ids[indices]
            groups.append((site,ids,np.arange(len(ids),dtype=np.int64)))
    return groups

def source_run(source,seeds=24,pilot_batches=0,observer=None):
    if source not in SOURCE_CONFIGS or seeds not in (1,24) or pilot_batches<0:raise ValueError('Source/seed budget')
    if not pilot_batches and observer is None:raise RuntimeError('Complete n24 generation requires the live compact-token observer')
    inputs=SimulationInputs();ns=definitions();root=ROOT/source;root.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    groups=query_batches(inputs);rowmap={int(v):i for i,v in enumerate(inputs.query_ids)}
    config=dict(name=SOURCE_CONFIGS[source],seed_order=list(EXPECTED_SEEDS),max_depth=16 if source=='base' else 10,
      diffuse_reflection=source=='scatter',scattering_coefficient=0. if source=='base' else .15,
      samples_per_src=1000000,max_components_per_source=1000000,batch_size=16)
    contract=dict(schema='v16-regenerated-query500-physical-source-v2',row_ids=inputs.query_ids.tolist(),config=config,
      input_identity=inputs.identity,geometry_fingerprint=inputs.geometry.geometry_fingerprint,
      source_files=vendor_identity(),statistics_source_sha256=sha256(ORIGINAL),
      shared_helper_sha256=sha256(Path(__file__).with_name('physical_statistics.py')),
      observer_identity=observer.identity_hash if observer is not None else None,
      CFR_convention='passband component coefficients times exp(-j2pi*(fc+df)*tau), FP64 Hz phase, exactly once',
      canonical_only_statistics=True,original_full16_target_groups=False,canonical_query16_target_groups=True,
      trace_seed_rule='seed+100000*site+first_original_Train_ID_in_new_query_group',
      batching_change='canonical500 positions grouped by site in canonical row order; not original4000 Train batches',
      live_token_policy='frozen Base96/Scatter64 candidates per seed, aggregate128; observer required before complete',
      storage_mode='uncompressed_npz_lossless_full1024',
      query_group_count=len(groups),coordinate_frame='official Round2 map/TrainPos world XYZ metres',
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
        if seed in state['completed_seeds']:
            if observer is None:raise RuntimeError('Committed seed lacks a live observer qualification')
            for site,group,local in groups:
                if not observer.has_batch(point_ids=group,configuration=source,seed=seed,site=site):
                    raise RuntimeError('Committed source seed has a missing/corrupt token shard; preserve it and use an explicit new recovery version')
            continue
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
                if np.all(done[rows]) and (observer is None or observer.has_batch(point_ids=group,configuration=source,seed=seed,site=site)):continue
                if pilot_batches and site_completed>=max(1,pilot_batches//2):break
                for row in group:scene.add(engine.Target(name=f'train_{row}',position=mi.Point3f(*inputs.positions[row].tolist()),look_at=transmitter))
                tick=time.monotonic();trace_seed=int(seed)+100000*site+int(group[0])
                components=builder(scene,max_depth=config['max_depth'],max_components_per_source=config['max_components_per_source'],
                  samples_per_src=config['samples_per_src'],diffuse_reflection=config['diffuse_reflection'],seed=trace_seed)
                traced=time.monotonic()
                with torch.no_grad():channel,path_count=component_channel(components,local,inputs.geometry.frequencies_hz,torch)
                rendered=time.monotonic();values=ns['score_aligned_statistics'](channel.cpu().numpy())
                values['path_count']=path_count;values['path_count_valid']=np.ones(len(rows),bool)
                welford_assign(arrays,previous,rows,values,ns)
                observed=time.monotonic()
                if observer is not None:
                    observer.consume_live(components,point_ids=np.array(group),configuration=source,seed=seed,site=site,replace_uncommitted=True)
                token_seconds=time.monotonic()-observed
                done[rows]=True;done.flush()
                report=dict(source=source,seed=seed,site=site,original_group_ids=group.tolist(),canonical_row_ids=inputs.query_ids[rows].tolist(),
                  trace_seed=trace_seed,trace_seconds=traced-tick,CFR_seconds=rendered-traced,token_seconds=token_seconds,total_seconds=time.monotonic()-tick,
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
              total_query_groups=len(groups),canonical_rows_completed=int(done.sum()),elapsed_seconds=time.monotonic()-started,
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
    combine_if_complete(inputs,observer)
    return result


def combine_if_complete(inputs=None,observer=None):
    inputs=inputs or SimulationInputs();ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'combine.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        arrays={};states={};shared_contract=None
        for source in SOURCE_CONFIGS:
            path=ROOT/source/'state.json'
            if not path.exists():return None
            state=json.loads(path.read_text())
            if state['completed_seeds']!=list(EXPECTED_SEEDS):return None
            c=state['contract']
            if c['input_identity']!=inputs.identity or c['geometry_fingerprint']!=inputs.geometry.geometry_fingerprint or c['row_ids']!=inputs.query_ids.tolist():
                raise ValueError('Physical sources refer to different/current input coordinates')
            common={k:c[k] for k in ('input_identity','geometry_fingerprint','row_ids','source_files','statistics_source_sha256','shared_helper_sha256','observer_identity','CFR_convention','batching_change','live_token_policy','storage_mode')}
            if shared_contract is None:shared_contract=common
            elif common!=shared_contract:raise ValueError('Base/Scatter physical source versions differ')
            if c['config']['name']!=SOURCE_CONFIGS[source] or c['config']['seed_order']!=list(EXPECTED_SEEDS):raise ValueError('Physical configuration/seed identity mismatch')
            states[source]=state
            for key in fields(definitions()):
                p=ROOT/source/state['current_directory']/(key+'.npy')
                arrays[source+'.'+key]=dict(path=PATHS.relative(p),sha256=sha256(p))
        if observer is None:raise RuntimeError('Final statistics require compact-token coverage verification')
        for source in SOURCE_CONFIGS:
            for seed in EXPECTED_SEEDS:
                for site,group,local in query_batches(inputs):
                    if not observer.has_batch(point_ids=group,configuration=source,seed=seed,site=site):
                        raise RuntimeError('Incomplete or corrupt compact-token coverage')
        original=PATHS.artifacts/'round2/pre_sionna/round2_geometry_visibility_features.npz'
        with np.load(original,allow_pickle=False) as z:
            geometry=np.asarray(z['train'][inputs.query_ids],np.float32);names=z['names'].tolist()
        if geometry.shape!=(500,37) or len(names)!=37:raise ValueError('Geometry37 contract')
        p=ROOT/'geometry.npy'
        if p.exists():
            if not np.array_equal(np.load(p),geometry):raise ValueError('Existing geometry descriptor changed')
        else:
            tmp=p.with_suffix('.npy.partial')
            with tmp.open('wb') as stream:np.save(stream,geometry)
            tmp.replace(p)
        arrays['geometry']=dict(path=PATHS.relative(p),sha256=sha256(p))
        result=dict(schema='v16-n24-physical-statistics-v1',row_ids=inputs.query_ids.tolist(),source_configs=list(SOURCE_CONFIGS.values()),
          seed_count=24,seeds=list(EXPECTED_SEEDS),arrays=arrays,geometry_names=names,regenerated=True,
          protocol='new canonical500 query16 physical features; frozen model inference; not old-score reproduction',
          old_cache_numerical_equivalence_verified=False,input_identity=inputs.identity,geometry_fingerprint=inputs.geometry.geometry_fingerprint,
          source_contract_hashes={s:states[s]['contract_hash'] for s in states},shared_physical_contract=shared_contract,
          source_state_paths={s:PATHS.relative(ROOT/s/'state.json') for s in states},
          live_token_root=PATHS.relative(observer.root),live_token_observer_identity=observer.identity_hash,
          compact_token_coverage_complete=True,requires_final_token_aggregation=True,no_CSI=True,trained=False,scored=False)
        save_json(ROOT/'manifest.json',result);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',choices=['base','scatter'],required=True)
    parser.add_argument('--seeds',type=int,choices=[1,24],default=24);parser.add_argument('--pilot-batches',type=int,default=0)
    parser.add_argument('--observer-module')
    args=parser.parse_args();observer=None
    try:
        if args.observer_module:
            sys.path.insert(0,str(Path(__file__).resolve().parent))
            factory=importlib.import_module(args.observer_module)
            ROOT.mkdir(parents=True,exist_ok=True)
            with (ROOT/'observer_init.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                observer=factory.create_observer(ROOT,args.source)
        source_run(args.source,args.seeds,args.pilot_batches,observer)
    except BaseException:
        LOG.mkdir(parents=True,exist_ok=True);(LOG/f'{args.source}_failure.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()

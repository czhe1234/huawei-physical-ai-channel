"""Compare regenerated operators full16 vs canonical-only targets, no GT."""
import argparse,json,time
import numpy as np
from round2.paper_baseline_adapters_v1.contract import SimulationInputs,save_json
from .physical_statistics import runtime,original_batches,component_channel,SOURCE_CONFIGS,ROOT,LOG

def probe(source):
    inputs=SimulationInputs();engine,data,mi,dr,torch=runtime()
    groups=original_batches(inputs);records=[];start=time.monotonic()
    xml=data.prepare_scene_files(__import__('project_paths').PATHS.round2,ROOT/source/'scene')
    for site in (0,1):
        scene=engine.load_scene(str(xml));scene.frequency=inputs.geometry.frequency_center_hz
        scene.tx_array=engine.build_transmit_array(4.,.5);scene.rx_array=engine.build_receive_array()
        data.configure_scene_response(scene,0. if source=='base' else .15)
        bs=inputs.geometry.bs_positions_m[site]
        tx=engine.Source(name=f'probe_bs{site}',position=mi.Point3f(*bs.tolist()),look_at=data.tx_target(bs,data.SITE_YAW_DEG[site],data.SITE_TILT_DEG[site]))
        scene.add(tx);builder=engine.ScenePriorBuilder()
        candidates=[x for x in groups if x[0]==site]
        for group_index in (0,len(candidates)//2):
            _,group,local=candidates[group_index];seed=42+100000*site+int(group[0]);outputs={}
            for mode in ('full16','canonical_subset'):
                ids=group if mode=='full16' else group[local]
                selected=local if mode=='full16' else np.arange(len(ids))
                for row in ids:scene.add(engine.Target(name=f'probe_{row}',position=mi.Point3f(*inputs.positions[row].tolist()),look_at=tx))
                tick=time.monotonic()
                components=builder(scene,max_depth=16 if source=='base' else 10,
                  max_components_per_source=1000000,samples_per_src=1000000,diffuse_reflection=source=='scatter',seed=seed)
                dr.eval(components.a[0],components.a[1],components.tau,components.valid);dr.sync_thread()
                traced=time.monotonic()
                with torch.no_grad():value,count=component_channel(components,selected,inputs.geometry.frequencies_hz,torch)
                value=value.cpu().numpy();torch.cuda.synchronize();rendered=time.monotonic()
                outputs[mode]=(value.copy(),count.copy(),dict(target_count=len(ids),trace_and_response_seconds=traced-tick,
                  CFR_and_transfer_seconds=rendered-traced,total_seconds=rendered-tick,
                  coefficient_shape=list(components.a[0].shape),canonical_path_counts=count.tolist()))
                del components,value;dr.sync_thread()
                for row in ids:scene.remove(f'probe_{row}')
            full,fc,fullcost=outputs['full16'];small,sc,smallcost=outputs['canonical_subset']
            difference=float(np.linalg.norm((full-small).ravel())/max(np.linalg.norm(full.ravel()),1e-30))
            item=dict(source=source,site=site,seed=seed,original_group_ids=group.tolist(),canonical_ids=group[local].tolist(),
              full16=fullcost,subset=smallcost,path_counts_equal=bool(np.array_equal(fc,sc)),
              maximum_absolute_operator_difference=float(np.max(np.abs(full-small))),relative_operator_difference=difference,
              generated_CFR_close=bool(np.allclose(full,small,rtol=1e-4,atol=1e-10)),
              frame='official world XYZ',query_truth_read=False)
            records.append(item);print(json.dumps(item),flush=True)
        del scene,builder;dr.sync_thread()
    result=dict(status='target_subset_numerical_probe',source=source,records=records,group_count=len(groups),
      all_probed_path_counts_equal=all(x['path_counts_equal'] for x in records),
      all_probed_operators_close=all(x['generated_CFR_close'] for x in records),
      scope='bounded observed equivalence only, not all groups/seeds proof',elapsed_seconds=time.monotonic()-start,
      no_CSI=True,trained=False,scored=False)
    save_json(LOG/f'{source}_target_subset_probe.json',result);print(json.dumps(result),flush=True)
    return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',choices=['base','scatter'],required=True);args=parser.parse_args();probe(args.source)

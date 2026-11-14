"""Run the registered measured evaluation through checked resumable stages."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from project_paths import PATHS


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,path)


def validate_geometry(root,reports,points):
    manifest=json.loads((root/'generation_report.json').read_text())
    assert manifest['status']=='complete'
    identity=manifest['identity'];seeds=identity['seeds'];batch=identity['batch']
    assert len(seeds)==24
    expected=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    total=2*2*len(seeds)*math.ceil(points/batch)
    done=0;valid=0;archives={};started=time.time()
    for panel in range(2):
        for config in ['base','scatter']:
            for seed in seeds:
                for first in range(0,points,batch):
                    p=root/f'panel{panel}'/config/f'seed{seed}'/f'batch{first//batch:04d}.npz'
                    rows=np.arange(first,min(points,first+batch))
                    with np.load(p,allow_pickle=False) as z:
                        assert str(z['cache_hash'])==expected
                        assert np.array_equal(z['point_id'],rows)
                        assert z['coordinate'].shape==(len(rows),128,3)
                        assert z['sequence'].shape==(len(rows),128,16,21)
                        assert z['features'].shape==(len(rows),128,24)
                        assert z['mask'].dtype==bool
                        for field in ['coordinate','mass','sequence','features']:
                            if not np.isfinite(z[field]).all():raise ValueError(f'Non-finite physical support: {field}')
                        assert np.all(z['mass']>=0)
                        assert np.all((z['depth']>=0)&(z['depth']<=16))
                        valid+=int(z['mask'].sum())
                    archives[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
                    done+=1
                    if done%128==0:atomic_json(reports/'geometry_validation_progress.json',dict(done=done,total=total,elapsed_s=time.time()-started))
    result=dict(status='complete',identity=identity,archives=archives,files=done,
                valid_retained_candidate_instances=valid,elapsed_s=time.time()-started)
    atomic_json(reports/'geometry_validation.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args()
    root=PATHS.root
    prepared=PATHS.artifacts/'measured/dc01_v1'
    geometry=PATHS.artifacts/'measured/dc01_geometry_v1'
    foundation=PATHS.artifacts/'measured/dc01_baselines_v1'
    spectral=PATHS.artifacts/'measured/dc01_spectral_v1'
    state=PATHS.artifacts/'measured/dc01_state_v1'
    reports=PATHS.analysis/'results/measured_dc01';reports.mkdir(parents=True,exist_ok=True)
    progress=reports/'pipeline_progress.json';started=time.time()
    source=json.loads((PATHS.datasets/'dichasus/dc01/upload_verified.json').read_text())
    audit=json.loads((prepared/'data_audit.json').read_text())
    assert source['sha256']==audit['source_sha256'] and source['size']==16870531926
    assert (prepared/'prepared_verified.json').exists()
    atomic_json(progress,dict(stage='waiting_for_geometry',elapsed_s=0))
    while not (reports/'geometry_full.exitcode').exists():time.sleep(3)
    assert (reports/'geometry_full.exitcode').read_text().strip()=='0','Geometry generation failed'
    assert (reports/'baselines.exitcode').read_text().strip()=='0','Foundation generation failed'
    atomic_json(progress,dict(stage='validate_geometry',elapsed_s=time.time()-started))
    validate_geometry(geometry,reports,audit['selected_points'])

    def run(stage,command):
        atomic_json(progress,dict(stage=stage,elapsed_s=time.time()-started,command=command))
        t=time.time()
        with (reports/f'{stage}.log').open('a',encoding='utf-8') as output:
            result=subprocess.run([sys.executable,*command],cwd=root,stdout=output,stderr=subprocess.STDOUT)
        (reports/f'{stage}.exitcode').write_text(str(result.returncode)+'\n')
        if result.returncode:raise RuntimeError(f'{stage} failed with exit code {result.returncode}; see its log')
        print(json.dumps(dict(stage=stage,status='complete',elapsed_s=time.time()-t)),flush=True)

    rel=lambda p:str(p.relative_to(root))
    run('spectral_training',['code/measured/dc01_spectral.py','--prepared-dir',rel(prepared),
        '--baseline-dir',rel(foundation),'--geometry-dir',rel(geometry),'--output-dir',rel(spectral),
        '--device',args.device,'--epochs','16'])
    run('state_training',['code/measured/dc01_state.py','--prepared-dir',rel(prepared),
        '--baseline-dir',rel(spectral),'--geometry-dir',rel(geometry),'--output-dir',rel(state),
        '--support-dir',rel(spectral/'merged_support'),'--device',args.device,'--epochs','48','--batch-size','64'])
    run('evaluation_report',['code/measured/dc01_report.py','--prepared-dir',rel(prepared),
        '--data-dir',rel(PATHS.datasets/'dichasus/dc01'),'--baseline-dir',rel(foundation),
        '--spectral-dir',rel(spectral),'--state-dir',rel(state),'--output-dir',rel(reports)])
    atomic_json(progress,dict(stage='complete',elapsed_s=time.time()-started))


if __name__=='__main__':
    try:main()
    except Exception:
        p=PATHS.analysis/'results/measured_dc01/pipeline_failure.json'
        atomic_json(p,dict(status='failed',traceback=traceback.format_exc()))
        traceback.print_exc();raise

"""Independent CPU manifest/count/normalization validation, never scores H."""
import json
from pathlib import Path
import numpy as np
from project_paths import PATHS
from round2.paper_baseline_adapters_v1.contract import SimulationInputs,sha256,save_json
from .physical_statistics_v2 import ROOT,LOG
from .physical_statistics import EXPECTED_SEEDS

def run():
    path=ROOT/'manifest.json';manifest=json.loads(path.read_text());inputs=SimulationInputs()
    if manifest['row_ids']!=inputs.query_ids.tolist() or manifest['seeds']!=list(EXPECTED_SEEDS) or manifest['seed_count']!=24:raise ValueError('Canonical/seeds identity')
    if manifest['input_identity']!=inputs.identity or not manifest['compact_token_coverage_complete']:raise ValueError('Incomplete source identity/coverage')
    arrays={};bytes_total=0
    for key,proof in manifest['arrays'].items():
        p=PATHS.root/proof['path']
        if sha256(p)!=proof['sha256']:raise ValueError('Array hash: '+key)
        arrays[key]=np.load(p,mmap_mode='r',allow_pickle=False);bytes_total+=p.stat().st_size
    results={}
    for source in ('base','scatter'):
        result={}
        for domain,axes in [('pas',(1,2,3)),('pdp',(-1,)),('ad',(1,2,3))]:
            mean=arrays[source+'.'+domain+'.mean'];count=arrays[source+'.'+domain+'.count']
            err=0.;negative=0.
            for lo in range(0,500,8):
                x=np.asarray(mean[lo:lo+8],np.float64);c=np.asarray(count[lo:lo+8])
                if not np.isfinite(x).all() or np.any(c>24) or np.any(c<0):raise ValueError('Nonfinite/count range')
                summed=x.sum(axis=axes);expected=(c>0).astype(float)
                err=max(err,float(np.max(np.abs(summed-expected))));negative=min(negative,float(x.min()))
            if err>2e-5 or negative<-1e-7:raise ValueError('Physical distribution normalization')
            result[domain]=dict(maximum_normalization_error=err,minimum_mean=negative,count_min=int(count.min()),count_max=int(count.max()))
        pc=arrays[source+'.path_count.count']
        if not np.array_equal(pc,np.full(500,24,dtype=pc.dtype)):raise ValueError('Zero-path points omitted from path-count statistics')
        for key,a in arrays.items():
            if key.startswith(source+'.'):
                for lo in range(0,len(a),16):
                    x=np.asarray(a[lo:lo+16])
                    if not np.isfinite(x).all():raise ValueError('Nonfinite '+key)
                    if key.endswith('.m2') and np.any(x<0):raise ValueError('Negative Welford M2 '+key)
        result['all500_path_count_denominators_equal24']=True;results[source]=result
    result=dict(status='complete_regenerated_physical_statistics_checks_passed',canonical_positions=500,
      genuine_seeds_per_source=24,sources=['base','scatter'],normalization=results,
      arrays=len(arrays),array_bytes=bytes_total,statistics_manifest=PATHS.relative(path),manifest_sha256=sha256(path),
      compact_token_coverage_complete=True,final128_token_aggregation_is_separate=True,
      old_cache_numerical_equivalence_claimed=False,no_official_CSI_read=True,training=False,scoring=False)
    save_json(LOG/'final_statistics_check.json',result);print(json.dumps(result))
    return result
if __name__=='__main__':run()

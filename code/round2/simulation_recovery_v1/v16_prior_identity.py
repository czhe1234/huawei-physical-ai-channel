"""Validate regenerated query500 prior identity, physical budgets and trace groups."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from v16_recovery import PATHS,ORIGINAL_SEEDS,canonical,sha256,file_identity


def validate_prior_identity(manifest: Path,info: dict):
    from round2.paper_baseline_adapters_v1.contract import SimulationInputs
    inputs=SimulationInputs()
    ids,folds=canonical()
    row_sha=hashlib.sha256(ids.astype("<i8",copy=False).tobytes()).hexdigest()
    if info.get("row_ids")!=ids.tolist() or info.get("input_identity")!=inputs.identity:
        raise ValueError("Physical producer input/query identity differs")
    if info.get("geometry_fingerprint")!=inputs.geometry.geometry_fingerprint:
        raise ValueError("Physical geometry fingerprint differs")
    group_list=[]
    for site in (0,1):
        local=ids[inputs.site_id[ids]==site]
        group_list.extend((site,local[i:i+16].tolist()) for i in range(0,len(local),16))
    expected_groups={(site,tuple(group)) for site,group in group_list}
    states={};contracts={}
    for source,config_name,depth,scatter in (
        ("base","primary_d16_base_v4lambda_1m",16,0.),
        ("scatter","primary_d10_scatter015_v4lambda_1m",10,.15)):
        p=PATHS.resolve(info["source_state_paths"][source])
        state=json.loads(p.read_text());c=state["contract"];cfg=c["config"]
        contract_hash=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
        if state.get("contract_hash")!=contract_hash or info["source_contract_hashes"][source]!=contract_hash:
            raise ValueError("Source producer contract hash differs")
        if state.get("completed_seeds")!=list(ORIGINAL_SEEDS):raise ValueError("Incomplete original24 seed sequence")
        expected=dict(name=config_name,seed_order=list(ORIGINAL_SEEDS),max_depth=depth,
            diffuse_reflection=source=="scatter",scattering_coefficient=scatter,
            samples_per_src=1000000,max_components_per_source=1000000,batch_size=16)
        if cfg!=expected:raise ValueError("Physical source/ray/depth/seed budget differs")
        if c.get("row_ids")!=ids.tolist() or c.get("input_identity")!=inputs.identity or c.get("geometry_fingerprint")!=inputs.geometry.geometry_fingerprint:
            raise ValueError("Source row/input/geometry identity differs")
        if c.get("query_group_count")!=len(group_list) or not c.get("canonical_query16_target_groups") or c.get("original_full16_target_groups"):
            raise ValueError("Regenerated query grouping contract differs")
        if c.get("trace_seed_rule")!="seed+100000*site+first_original_Train_ID_in_new_query_group":
            raise ValueError("Trace seed rule differs")
        if c.get("CFR_convention")!="passband component coefficients times exp(-j2pi*(fc+df)*tau), FP64 Hz phase, exactly once":
            raise ValueError("CFR phase/frequency convention differs")
        if not c.get("no_CSI") or not c.get("regenerated") or c.get("old_cache_numerical_equivalence_verified"):
            raise ValueError("Producer qualification is incorrectly labelled")
        if not c.get("source_files"):raise ValueError("Missing frozen vendor source identity")
        for name,expected_sha in c["source_files"].items():
            if sha256(PATHS.resolve(name))!=expected_sha:raise ValueError("Frozen physical engine source changed")
        if c.get("statistics_source_sha256")!=sha256(PATHS.round2_code/"generate_multiseed_sionna_dataset.py"):
            raise ValueError("Original statistics operator source changed")
        if c.get("shared_helper_sha256")!=sha256(Path(__file__).parent/"physical_statistics.py"):
            raise ValueError("Physical CFR/Welford helper source changed")
        states[source]=file_identity(p);contracts[source]=c
    token_manifest=manifest.parent/"live_tokens"/"tokens"/"manifest.json"
    token=json.loads(token_manifest.read_text())
    if PATHS.resolve(info.get("live_token_root",""))!=token_manifest.parent.parent:
        raise ValueError("Statistics token root differs")
    if info.get("live_token_observer_identity")!=token.get("observer_identity"):
        raise ValueError("Statistics token observer identity differs")
    if any(c.get("observer_identity")!=token.get("observer_identity") for c in contracts.values()):
        raise ValueError("Base/Scatter source observers differ")
    prior=token["prior_identity"]
    if token.get("query_ids")!=ids.tolist() or token.get("original_seeds")!=list(ORIGINAL_SEEDS):
        raise ValueError("Token/statistics query or seed identity differs")
    if prior.get("input_identity")!=inputs.identity or prior.get("geometry_fingerprint")!=inputs.geometry.geometry_fingerprint:
        raise ValueError("Live token physical identity differs")
    generator=Path(__file__).parent/"physical_statistics_v2.py"
    if prior.get("generator_sha256")!=sha256(generator):
        raise ValueError("Regenerated token producer source changed")
    # Each compact shard preserves its real source/base seed/site/target group.
    # The actual trace integer is derived from the frozen producer's explicit
    # call formula, not confused with the base seed. No guessed random values.
    traces=[];seen=set()
    for item in token["compact_shards"]:
        p=PATHS.resolve(item["path"]);m=p.with_suffix(".json")
        shard=json.loads(m.read_text())
        source=shard["source"];seed=int(shard["seed"]);site=int(shard["site"]);group=shard["point_ids"]
        key=(source,seed,site,tuple(group))
        if key in seen or source not in contracts or seed not in ORIGINAL_SEEDS or (site,tuple(group)) not in expected_groups:
            raise ValueError("Compact target-group/source/seed coverage differs")
        if shard.get("sha256")!=item["sha256"] or shard.get("source_identity")!=prior:
            raise ValueError("Compact shard contract differs from finalized tokens")
        seen.add(key)
        traces.append(dict(source=source,base_seed=seed,site=site,query_ids=group,
            trace_seed=seed+100000*site+int(group[0]),compact_shard=item,
            evidence="saved_batch_IDs_plus_frozen_generator_trace_seed_formula"))
    expected={(s,seed,site,tuple(g)) for s in contracts for seed in ORIGINAL_SEEDS for site,g in group_list}
    if seen!=expected:raise ValueError("Not every32group x24seed x2source has compact evidence")
    return dict(schema="v16-regenerated-prior-identity-validation-v1",status="passed",
        source_manifest=file_identity(manifest),canonical_row_sha256=row_sha,query_groups=len(group_list),
        original_seed_sequence=list(ORIGINAL_SEEDS),trace_seed_rows=traces,
        source_states=states,token_manifest=file_identity(token_manifest),
        physical_budget_unchanged=True,grouping_changed_from_legacy=True,
        regenerated=True,strict_canonical_qualification=False,old_score_reproduction=False,
        trace_seed_evidence_boundary="source-derived from saved actual batch IDs; runtime stdout can be cross-checked independently")


def main():
    import argparse
    from v16_recovery import REPORT_ROOT,atomic_json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--statistics",type=Path,required=True)
    a=p.parse_args()
    info=json.loads(a.statistics.read_text());result=validate_prior_identity(a.statistics,info)
    output=REPORT_ROOT/"v16_forward_v3"/"prior_identity.json"
    if output.exists():
        if json.loads(output.read_text())!=result:raise ValueError("Prior verification identity differs")
    else:atomic_json(output,result)
    print(json.dumps(dict(status=result["status"],query_groups=result["query_groups"],
        base_seed_count=24,trace_rows=len(result["trace_seed_rows"]),
        canonical_row_sha256=result["canonical_row_sha256"],report=PATHS.relative(output))))


if __name__=="__main__":main()

"""Frozen production5 PathState inference on regenerated query geometry.

The executed legacy dispatch uses each Train ID's own FULL_FOLD. It is never called
canonical-outer-strict; canonical outer and legacy head fold remain separate.
"""
from __future__ import annotations
import dataclasses
import json
from pathlib import Path
import sys
import types
import numpy as np
from v16_recovery import (PATHS,ARTIFACT_ROOT,BUNDLE,canonical,sha256,file_identity,source_namespace,
    safe_checkpoint,get_torch,operators,endpoint_renderer,render_endpoint_batch,save_stage,
    certified_parent,atomic_json,SOURCE_HASHES)

MODEL_SOURCE="../../final_code/V16_reorganized_submission/code/round2/physical_ai_submission/component_state/model.py"


def model_core(torch):
    return source_namespace(MODEL_SOURCE,["ModelConfig","ComponentStateModel","_mapped_coordinates","_batch"],
        dict(torch=torch,nn=torch.nn,fields=dataclasses.fields))


def predict_new_states(query_input_manifest: Path,device="cpu",batch_size=8):
    torch=get_torch(device);ns=model_core(torch)
    proof=json.loads(query_input_manifest.read_text());ids,canonical_fold=canonical()
    if proof.get("schema")!="v16-recovery-link-inputs-v1" or proof.get("row_ids")!=ids.tolist():
        raise ValueError("Regenerated feature row identity required")
    for key in ("features","auxiliary"):
        p=PATHS.resolve(proof["arrays"][key]["path"])
        if sha256(p)!=proof["arrays"][key]["sha256"]:raise ValueError("Feature/auxiliary bytes changed")
    features=np.load(PATHS.resolve(proof["arrays"]["features"]["path"]),mmap_mode="r")
    with np.load(PATHS.resolve(proof["arrays"]["auxiliary"]["path"]),allow_pickle=False) as z:
        auxiliary={k:np.asarray(z[k]) for k in z.files}
    if features.shape!=(500,128,120):raise ValueError("Native120 features required")
    sys.path.insert(0,str(PATHS.round2_code))
    from round2.data import load_positions
    _,_,_,train_sites,_=load_positions()
    sites=train_sites[ids]
    fold_path=PATHS.artifacts/"round2"/"full_path_pdp_link_gate"/"train_fold_id.npy"
    head_fold=np.asarray(np.load(fold_path,mmap_mode="r")[ids],np.int8)
    if not np.isin(head_fold,np.arange(5)).all():raise ValueError("Invalid legacy full fold")
    root=ARTIFACT_ROOT/"historical_replay"/"production_pathstate_FULL_FOLD"
    root.mkdir(parents=True,exist_ok=True)
    output=root/"states.npz";manifest=root/"manifest.json"
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old.get("query_inputs")!=file_identity(query_input_manifest) or old.get("state_array",{}).get("sha256")!=sha256(output):
            raise ValueError("Complete PathState parent/hash differs")
        if old.get("legacy_fold_source")!=file_identity(fold_path):raise ValueError("Legacy fold source changed")
        for item in old["checkpoints"]:
            q=PATHS.resolve(item["checkpoint"]["path"])
            if file_identity(q)!=item["checkpoint"]:raise ValueError("Frozen PathState weight changed")
        return manifest
    result=dict(mass=np.empty((500,128),np.float32),coordinate=np.empty((500,128,3),np.float32),
        port=np.empty((500,128,2,4),np.float32))
    checkpoints=[]
    for fold in range(5):
        path=BUNDLE/"artifacts/round2/physical_ai_submission/models/component_calibrator"/f"fold_{fold}.pt"
        ck=safe_checkpoint(path,torch)
        if int(ck.get("outer",-1))!=fold:raise ValueError("Legacy head metadata differs")
        cfg=ns.ModelConfig.from_checkpoint(ck["config"])
        model=ns.ComponentStateModel(120,auxiliary["geometry"].shape[-1],cfg).to(device)
        model.load_state_dict(ck["state_dict"]);model.eval()
        coordinate=ns._mapped_coordinates(auxiliary,sites,np.asarray(ck["mapping_latent"],np.float32),torch.device(device))
        mean=np.asarray(ck["feature_mean"],np.float32);std=np.asarray(ck["feature_std"],np.float32)
        if mean.shape!=(120,) or std.shape!=(120,) or np.any(std<=0):raise ValueError("Bad frozen feature normalization")
        rows=np.flatnonzero(head_fold==fold)
        for lo in range(0,len(rows),batch_size):
            local=rows[lo:lo+batch_size]
            with torch.inference_mode():
                batch=ns._batch(local,features,auxiliary,coordinate,mean,std,24,torch.device(device))
                prediction=model(batch)
            for key in result:result[key][local]=prediction[key].cpu().numpy().astype(np.float32)
        checkpoints.append(dict(head_train_fold_id=fold,query_count=len(rows),checkpoint=file_identity(path)))
        del model,ck
        if device=="cuda:0":torch.cuda.empty_cache()
    if not all(np.isfinite(a).all() for a in result.values()):raise ValueError("Nonfinite new PathState output")
    partial=output.with_suffix(".partial.npz")
    with partial.open("wb") as f:np.savez(f,query_ids=ids,canonical_outer_id=canonical_fold,
        head_train_fold_id=head_fold,**result)
    partial.replace(output)
    record=dict(schema="v16-production-pathstate-new-query-v1",status="complete",query_ids=ids.tolist(),
        canonical_outer_id=canonical_fold.tolist(),head_train_fold_id=head_fold.tolist(),
        route="one_head_by_each_Train_ID_FULL_FOLD_not_canonical_fold_not_5head_ensemble",
        query_inputs=file_identity(query_input_manifest),legacy_fold_source=file_identity(fold_path),
        state_array=file_identity(output),checkpoints=checkpoints,sources=dict(SOURCE_HASHES),
        original_weights=True,regenerated_query_prior=True,old_endpoints_used=False,
        full_current_model_forward=True,fresh_pathstate_forward=True,canonical_strict_qualification=False,head_training_exclusion_certified=False,
        training_provenance="E190 config/reports consistent; saved train-ID/optimizer trace unavailable",
        query_CSI_used=False,zero_label_used=False,trained=False,performance_evaluated=False)
    atomic_json(manifest,record)
    return manifest


def synthesize_new_v16(v11_manifest: Path,state_manifest: Path,device="cpu",batch_size=8,resume=False):
    torch=get_torch(device);op=operators(torch);renderer=endpoint_renderer(torch)
    base,parent,p=certified_parent(v11_manifest,"v11_parent")
    state=json.loads(state_manifest.read_text())
    if state.get("schema")!="v16-production-pathstate-new-query-v1" or not state.get("full_current_model_forward") or state.get("old_endpoints_used"):
        raise ValueError("Only freshly inferred production states qualify for full forward")
    sp=PATHS.resolve(state["state_array"]["path"])
    if sha256(sp)!=state["state_array"]["sha256"]:raise ValueError("New state bytes changed")
    with np.load(sp,allow_pickle=False) as z:
        arrays={k:np.asarray(z[k]) for k in ("mass","coordinate","port")}
        if not np.array_equal(z["query_ids"],canonical()[0]):raise ValueError("State row mapping differs")
    # Both the new V11 and new PathState must refer to exactly the same immutable
    # physical-statistics manifest, via their independently checked producer chain.
    physical_path=PATHS.resolve(parent["identity"]["physical"]["path"])
    if file_identity(physical_path)!=parent["identity"]["physical"]:
        raise ValueError("V11 physical input manifest changed")
    physical=json.loads(physical_path.read_text())
    query_path=PATHS.resolve(state["query_inputs"]["path"])
    if file_identity(query_path)!=state["query_inputs"]:raise ValueError("PathState input manifest changed")
    query=json.loads(query_path.read_text())
    if physical.get("statistic_source")!=query.get("statistics"):
        raise ValueError("V11/PathState regenerated physical parent differs")
    if physical.get("raw_link_source")!=state["query_inputs"]:
        raise ValueError("V11/PathState do not share the same token-derived query inputs")
    spec=types.SimpleNamespace(mode="sqrt",pas_beta=.20,pdp_beta=.20,order="pas_pdp",cycles=4)
    identity=dict(kind="V16_regenerated_prior_legacy_FULL_FOLD_forward",
        parent_V11=file_identity(p),parent_manifest=file_identity(v11_manifest),
        production_state_manifest=file_identity(state_manifest),sources=dict(SOURCE_HASHES),
        canonical_outer_id=state["canonical_outer_id"],head_train_fold_id=state["head_train_fold_id"],
        original_weights=True,old_endpoints_used=False,regenerated_query_prior=True,
        full_current_model_forward=True,fresh_pathstate_forward=True,canonical_strict_qualification=False,
        qualification="legacy_FULL_FOLD_dispatch_executed_training_exclusion_not_fully_certified",
        head_training_exclusion_certified=False,
        original_score_reproduction=False,query_CSI_used=False,trained=False)
    def writer(out,start,commit):
        for lo in range(start,500,batch_size):
            hi=min(lo+batch_size,500)
            pas,pdp=render_endpoint_batch(arrays,slice(lo,hi),torch,renderer,device)
            b=torch.as_tensor(np.array(base[lo:hi],copy=True),device=device)
            with torch.inference_mode():
                y=op.make_candidate(b,op.pas_distribution(b),op.pdp_distribution(b),
                    torch.as_tensor(pas,device=device),torch.as_tensor(pdp,device=device),spec)
            out[lo:hi]=y.cpu().numpy().astype(np.complex64);commit(hi)
    return save_stage("legacy_fixed_weights_regenerated_query_v3",identity,(500,256,4,192),np.complex64,writer,resume)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["states","synthesize"])
    p.add_argument("--query-inputs",type=Path);p.add_argument("--base-manifest",type=Path)
    p.add_argument("--state-manifest",type=Path);p.add_argument("--device",default="cpu")
    p.add_argument("--batch-size",type=int,default=8);p.add_argument("--resume",action="store_true")
    a=p.parse_args();SOURCE_HASHES[PATHS.relative(Path(__file__))]=sha256(Path(__file__))
    if a.stage=="states":
        if a.query_inputs is None:p.error("states requires --query-inputs")
        result=predict_new_states(a.query_inputs,a.device,a.batch_size)
        print(json.dumps(dict(state_manifest=PATHS.relative(result))))
    else:
        if a.base_manifest is None or a.state_manifest is None:p.error("synthesize requires both parent manifests")
        _,result=synthesize_new_v16(a.base_manifest,a.state_manifest,a.device,a.batch_size,a.resume)
        print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":main()

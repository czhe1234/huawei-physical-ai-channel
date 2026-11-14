"""New-query V16 token -> native120 features/sequence/raw PDP/143 context.

Only geometry-derived token coefficients/statistics and official positions enter.
No training channel, truth spectrum or query zero-channel flag is read.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import types
import numpy as np
from v16_recovery import (PATHS,ARTIFACT_ROOT,canonical,source_namespace,file_identity,
    sha256,protected_output,atomic_json,get_torch,SOURCE_HASHES)
from v16_physical_inference import Statistics

BUNDLE_SOURCE="../../final_code/V16_reorganized_submission/code/round2/physical_ai_submission/"


def load_query_tokens(manifest: Path,statistics: Statistics):
    proof=json.loads(manifest.read_text());ids,_=canonical()
    if proof.get("schema")!="v16-regenerated-tokens-v1" or proof.get("query_ids")!=ids.tolist():
        raise ValueError("New token row identity differs")
    if file_identity(manifest)!=statistics.provenance["token_manifest"]:
        raise ValueError("Passed token manifest is not the one bound by the statistics producer")
    if statistics.info.get("live_token_observer_identity") is not None and proof.get("observer_identity")!=statistics.info["live_token_observer_identity"]:
        raise ValueError("Observer identity differs from statistics")
    prior=proof.get("prior_identity",{})
    if prior.get("input_identity")!=statistics.info.get("input_identity") or prior.get("geometry_fingerprint")!=statistics.info.get("geometry_fingerprint"):
        raise ValueError("Token/statistics physical input identity differs")
    if proof.get("original_seeds")!=statistics.info.get("seeds"):
        raise ValueError("Token/statistics seeds differ")
    arrays={}
    for name,item in proof["arrays"].items():
        p=PATHS.resolve(item["path"])
        if sha256(p)!=item["sha256"]:raise ValueError(f"Token bytes changed: {name}")
        arrays[name]=np.load(p,mmap_mode="r")
    if arrays["coefficient"].shape!=(500,128,256,4) or arrays["mask"].shape!=(500,128):
        raise ValueError("Full native token layout required")
    return arrays,proof


def geometry_regime(positions,sites,ids):
    path=PATHS.artifacts/"round2"/"regime_lifecycle_support"/"geodesic_corridor_features.npz"
    with np.load(path,allow_pickle=False) as f:
        train=np.asarray(f["train_features"])
        names=tuple(f["feature_names"].astype(str))
    ns=source_namespace("run_v6_regime_component_lifecycle.py",["physical_risk_and_regime"],
        dict(),["REGIME_NAMES"])
    _,_,_,query,_=ns.physical_risk_and_regime(train,train[ids],sites,sites[ids],np.arange(len(positions)),names)
    return query,path


def atomic_array(path: Path,array):
    path=protected_output(path)
    if path.exists():
        old=np.load(path,mmap_mode="r")
        if old.shape!=array.shape or old.dtype!=array.dtype or not np.array_equal(old,array):
            raise ValueError("Existing derived array differs")
        return
    partial=path.with_suffix(".partial.npy")
    with partial.open("wb") as f:np.save(f,array,allow_pickle=False)
    os.replace(partial,path)


def build_inputs(statistics_manifest: Path,token_manifest: Path,device="cpu",batch_size=4):
    stats=Statistics(statistics_manifest)
    tokens,token_proof=load_query_tokens(token_manifest,stats)
    ids,folds=canonical()
    root=protected_output(ARTIFACT_ROOT/"historical_replay"/"regenerated_query_inputs")
    root.mkdir(parents=True,exist_ok=True)
    manifest=root/"manifest.json"
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old.get("statistics")!=file_identity(statistics_manifest) or old.get("tokens")!=file_identity(token_manifest) or old.get("bridge_source_sha256")!=sha256(Path(__file__)):
            raise ValueError("Complete query inputs use another physical prior")
        for item in old["arrays"].values():
            if sha256(PATHS.resolve(item["path"]))!=item["sha256"]:raise ValueError("Derived query input changed")
        return old
    sys.path.insert(0,str(PATHS.round2_code))
    from round2.data import load_positions
    setup,positions,_,sites,_=load_positions()
    regime,regime_path=geometry_regime(positions,sites,ids)
    feature_path=root/"features.npy";partial=root/"features.partial.npy"
    # Exact legacy feature builder, with new row-bound native token cache.
    features_ns=source_namespace("train_explicit_path_pdp_predictor.py",["build_features"],
        dict(PATH_CACHE=token_manifest.parent,load_positions=load_positions))
    features=features_ns.build_features(np.arange(500),ids,regime,partial,
        path_cache=token_manifest.parent,split="train",presence_seed_count=24)
    if features.shape!=(500,128,120) or not np.isfinite(features).all():
        raise ValueError("Native120 feature contract failed")
    if feature_path.exists():
        if not np.array_equal(np.load(feature_path,mmap_mode="r"),features):raise ValueError("Features differ")
        partial.unlink(missing_ok=True)
    else:os.replace(partial,feature_path)
    cfg=types.SimpleNamespace(subcarrier_spacing_hz=30000.,subcarriers=192,tx_horizontal=16,
        tx_vertical=8,components_per_point=128,interaction_steps=16)
    ns=source_namespace(BUNDLE_SOURCE+"scene_prior/component_features.py",
        ["initial_coordinates","sequence_geometry","sequence_hashes"],dict(),["SEQUENCE_HASH_BUCKETS","TINY"])
    coordinate=ns.initial_coordinates(tokens["tau"],tokens["angles"],sites[ids],cfg)
    shaped=np.asarray(tokens["coefficient"]).reshape(500,128,2,16,8,4)
    port=(np.abs(shaped)**2).sum(axis=(3,4),dtype=np.float64)
    port/=np.maximum(port.sum(axis=(2,3),keepdims=True),1e-30)
    sequence=ns.sequence_geometry(tokens["vertices"],tokens["interaction_count"],
        positions[ids],sites[ids],setup.bs_positions,positions,16)
    hashes=ns.sequence_hashes(tokens)
    buckets=np.stack([hashes%np.uint64(32749),(hashes>>np.uint64(32))%np.uint64(32749)],axis=-1).astype(np.int64)
    auxiliary=dict(coordinate=coordinate.astype(np.float32),port=port.astype(np.float32),
        geometry=sequence.astype(np.float32),hash=buckets,depth=np.asarray(tokens["interaction_count"],np.int64),
        interactions=np.asarray(tokens["interactions"],np.int64),objects=np.asarray(tokens["objects"],np.int64),
        primitives=np.asarray(tokens["primitives"],np.int64),fraction=np.asarray(tokens["energy_fraction"],np.float32),
        presence=np.asarray(tokens["presence"],np.float32),angles=np.asarray(tokens["angles"],np.float32))
    if not all(np.isfinite(a).all() for a in auxiliary.values()):raise ValueError("Nonfinite sequence input")
    aux_path=root/"auxiliary.npz"
    if aux_path.exists():
        with np.load(aux_path,allow_pickle=False) as f:
            if set(f.files)!=set(auxiliary) or any(not np.array_equal(f[k],a) for k,a in auxiliary.items()):
                raise ValueError("Existing auxiliary differs")
    else:
        temp=aux_path.with_suffix(".partial.npz")
        with temp.open("wb") as f:np.savez(f,**auxiliary)
        os.replace(temp,aux_path)
    # Exact historical143context: token mean120 +23 scalar columns, using the
    # full4000 position reference, never recentering on the500 query subset.
    mask=np.asarray(tokens["mask"],np.float32)
    token_mean=(features*mask[:,:,None]).sum(1)/mask.sum(1).clip(min=1)[:,None]
    xy=positions[:,:2];center=xy.mean(0);scale=xy.std(0)+1e-6
    relative=xy-setup.bs_positions[sites,:2];relative_scale=relative.std(0)+1e-6
    local=relative[ids];bearing=np.arctan2(local[:,1],local[:,0])
    reliability=stats.reliability("pdp")
    extra=np.concatenate([((xy[ids]-center)/scale).astype(np.float32),
        (local/relative_scale).astype(np.float32),np.sin(bearing)[:,None].astype(np.float32),
        np.cos(bearing)[:,None].astype(np.float32),np.eye(2,dtype=np.float32)[sites[ids]],
        np.eye(4,dtype=np.float32)[regime],reliability],axis=1)
    context=np.concatenate([token_mean,extra],axis=1).astype(np.float32)
    if context.shape!=(500,143) or not np.isfinite(context).all():raise ValueError("Native143 context failed")
    context_path=root/"link_context.npy";atomic_array(context_path,context)
    # Raw-path PDP follows the exact old LinkGate renderer: offset tones only;
    # do not replace this with n24 mean PDP or add an extra carrier phase.
    torch=get_torch(device)
    from v16_recovery import operators
    op=operators(torch)
    raw_path=root/"raw_link_pdp.npy";raw_partial=root/"raw_link_pdp.partial.npy"
    if not raw_path.exists():
        out=np.lib.format.open_memmap(raw_partial,mode="w+",dtype=np.float16,shape=(500,256,4,192))
        freq=torch.arange(-96,96,dtype=torch.float32,device=device)*30000.
        for lo in range(0,500,batch_size):
            hi=min(lo+batch_size,500)
            coeff=torch.as_tensor(np.array(tokens["coefficient"][lo:hi],copy=True),device=device)
            tau=torch.as_tensor(np.array(tokens["tau"][lo:hi],np.float32,copy=True),device=device)
            valid=torch.as_tensor(np.array(tokens["mask"][lo:hi],copy=True),device=device)
            with torch.inference_mode():
                phase=torch.exp(-2j*torch.pi*tau[:,:,None]*freq[None,None])
                channel=torch.einsum("bk,bkmu,bks->bmus",valid.to(coeff.dtype),coeff,phase)
                out[lo:hi]=op.pdp_distribution(channel).cpu().numpy().astype(np.float16)
            out.flush()
        del out;os.replace(raw_partial,raw_path)
    raw=np.load(raw_path,mmap_mode="r")
    if raw.shape!=(500,256,4,192) or raw.dtype!=np.float16:raise ValueError("Raw-link PDP header differs")
    for lo in range(0,500,16):
        if not np.isfinite(raw[lo:lo+16]).all():raise ValueError("Raw-link PDP nonfinite")
    names=[f"native_token_mean_{i:03d}" for i in range(120)]+[
        "x_standard","y_standard","relative_x","relative_y","bearing_sin","bearing_cos","site0","site1",
        "regime0","regime1","regime2","regime3"]+[f"native_pdp_reliability_{i:02d}" for i in range(11)]
    info=dict(schema="v16-recovery-link-inputs-v1",row_ids=ids.tolist(),canonical_outer_id=folds.tolist(),
        arrays=dict(raw_link_pdp=file_identity(raw_path),link_context=file_identity(context_path),
            features=file_identity(feature_path),auxiliary=file_identity(aux_path)),
        link_context_names=names,statistics=file_identity(statistics_manifest),tokens=file_identity(token_manifest),
        geometry_regime_source=file_identity(regime_path),source_hashes=dict(SOURCE_HASHES),
        native_feature_dimensions=120,bridge_source_sha256=sha256(Path(__file__)),regenerated_prior=True,no_query_CSI=True,no_zero_label=True,
        parent_physical_identity=stats.info.get("shared_physical_contract"),performance_evaluated=False)
    atomic_json(manifest,info)
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--statistics",type=Path,required=True);p.add_argument("--tokens",type=Path,required=True)
    p.add_argument("--device",default="cpu");p.add_argument("--batch-size",type=int,default=4)
    a=p.parse_args();SOURCE_HASHES[PATHS.relative(Path(__file__))]=sha256(Path(__file__))
    print(json.dumps(build_inputs(a.statistics,a.tokens,a.device,a.batch_size),ensure_ascii=False))


if __name__=="__main__":main()

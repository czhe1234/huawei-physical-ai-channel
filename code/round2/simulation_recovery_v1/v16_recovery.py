"""Isolated V16 recovery. No scoring; historical and strict identities never mix.

The historical route replays frozen operators and saved selected artifacts.
Missing RT statistics must be supplied with their own row/config provenance.
The strict coordinator accepts only explicitly proved reusable fold artifacts,
and exposes fit labels only through a role-scoped reader to new trainers.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types
from typing import Any, Callable

_CODE = Path(__file__).resolve().parents[2]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
from project_paths import PATHS

import numpy as np

ARTIFACT_ROOT = PATHS.artifacts / "round2" / "simulation_recovery_v1"
REPORT_ROOT = PATHS.analysis / "audits" / "simulation_recovery_v1"
ALLOWED_GPU = "GPU-70b6fceb-c7c4-9467-3c81-097c9284c160"
META = PATHS.artifacts / "round2" / "pre_sionna" / "round2_pre_sionna_oof_foundation_meta.npz"
FOLDS = PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
BUNDLE = PATHS.root / "final_code" / "V16_reorganized_submission"
ENDPOINT_ROOT = PATHS.root / "final_code" / "experiments" / "v16_rt_ablation" / "artifacts" / "strict_pathstate" / "full"
CHANNEL_SHAPE = (500, 256, 4, 192)
SCHEMA = "v16-recovery-v1"
ORIGINAL_SEEDS = (42,101,211,307,401,503,601,701,809,907,1009,1103,1201,1301,1409,1511,1601,1709,1801,1901,2003,2111,2203,2309)
SOURCE_HASHES: dict[str, str] = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def portable(path: Path) -> str:
    return PATHS.relative(path)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def protected_output(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(ARTIFACT_ROOT.resolve()) and not path.is_relative_to(REPORT_ROOT.resolve()):
        raise ValueError("All outputs must be under the new simulation_recovery_v1 roots")
    return path


def atomic_json(path: Path, value: Any) -> None:
    path = protected_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_name(path.name + ".partial")
    with partial.open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(partial, path)


def canonical() -> tuple[np.ndarray, np.ndarray]:
    with np.load(META, allow_pickle=False) as f:
        ids = f["query_ids"].astype(np.int64)
        folds = f["fold_id"].astype(np.int8)
    if ids.shape != (500,) or np.unique(ids).size != 500 or folds.shape != (500,):
        raise ValueError("Invalid canonical500 identity")
    return ids, folds


def get_torch(device: str):
    # Must happen before importing torch or any CUDA-capable source module.
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device == "cuda:0":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != ALLOWED_GPU:
            raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the assigned complete A800 UUID")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    else:
        raise ValueError("Only cpu or assigned cuda:0 is supported")
    import torch
    if device == "cuda:0":
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one assigned device must be visible")
        props = torch.cuda.get_device_properties(0)
        uuid = str(getattr(props, "uuid", ""))
        if uuid and uuid.removeprefix("GPU-") != ALLOWED_GPU.removeprefix("GPU-"):
            raise RuntimeError(f"CUDA UUID mismatch: {uuid}")
    return torch


def source_namespace(filename: str, names: list[str], injected: dict[str, Any], constants=()):
    """Compile selected frozen definitions only; no legacy imports or main run."""
    path = PATHS.round2_code / filename
    source = path.read_text()
    SOURCE_HASHES[portable(path)] = sha256(path)
    tree = ast.parse(source)
    definitions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    chosen = set(names)
    while True:
        deps = {n.id for k in chosen for n in ast.walk(definitions[k]) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        new = chosen | (deps & definitions.keys() - injected.keys())
        if new == chosen:
            break
        chosen = new
    assignments = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets):
            assignments.append(n)
    module_name = "_v16_recovery_" + path.stem
    module = types.ModuleType(module_name)
    module.__dict__.update(dict(np=np, math=math, dataclass=dataclasses.dataclass, Path=Path))
    module.__dict__.update(injected)
    sys.modules[module_name] = module
    selected = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    selected += assignments
    selected += [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in chosen]
    subset = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(subset, str(path), "exec"), module.__dict__)
    return module


def operators(torch):
    ns = source_namespace(
        "analyze_v4_exact_joint_projection.py",
        ["normalized", "preserve_point_power", "pas_distribution", "pdp_distribution",
         "exact_ad_projection", "exact_pas_projection", "exact_pdp_projection", "fixed_blend"],
        dict(torch=torch), ["TINY"])
    core = {n: getattr(ns, n) for n in ["normalized", "preserve_point_power", "pas_distribution",
            "pdp_distribution", "exact_ad_projection", "exact_pas_projection", "exact_pdp_projection", "fixed_blend"]}
    joint = source_namespace("train_full_path_pas_slice_gate.py", ["project_joint"], dict(torch=torch, **core))
    endpoint = source_namespace("analyze_e86_surface_endpoint_v11_blend.py",
        ["endpoint_target", "make_candidate"], dict(torch=torch, project_joint=joint.project_joint, **core),
        ["TINY"])
    core.update(project_joint=joint.project_joint, endpoint_target=endpoint.endpoint_target,
                make_candidate=endpoint.make_candidate)
    return types.SimpleNamespace(**core)


def save_stage(name: str, identity: dict, shape: tuple, dtype, writer: Callable, resume=False):
    root = protected_output(ARTIFACT_ROOT / "historical_replay" / name)
    root.mkdir(parents=True, exist_ok=True)
    contract = dict(schema=SCHEMA, name=name, identity=identity, shape=list(shape), dtype=str(np.dtype(dtype)))
    contract_hash = digest(contract)
    final = root / "channels.npy"
    manifest = root / "manifest.json"
    if final.exists() or manifest.exists():
        if not resume or not manifest.exists() or not final.exists():
            raise FileExistsError(root)
        old = json.loads(manifest.read_text())
        if old.get("contract_hash") != contract_hash or old.get("sha256") != sha256(final):
            raise RuntimeError("Complete stage identity/hash mismatch")
        return np.load(final, mmap_mode="r"), old
    partial = root / "channels.partial.npy"
    state = root / "state.json"
    start = 0
    if partial.exists() or state.exists():
        if not resume or not partial.exists() or not state.exists():
            raise FileExistsError("Incomplete stage requires explicit matching --resume")
        status = json.loads(state.read_text())
        if status.get("contract_hash") != contract_hash:
            raise RuntimeError("Incomplete stage identity mismatch")
        start = int(status["completed"])
        output = np.load(partial, mmap_mode="r+")
        if output.shape != shape or output.dtype != np.dtype(dtype):
            raise ValueError("Resume header differs")
    else:
        output = np.lib.format.open_memmap(partial, mode="w+", shape=shape, dtype=dtype)
        initial=state.with_suffix(".tmp")
        initial.write_text(json.dumps(dict(contract_hash=contract_hash,completed=0)))
        os.replace(initial,state)
    def commit(end):
        output.flush()
        payload = dict(contract_hash=contract_hash, completed=int(end))
        temp = state.with_suffix(".tmp")
        temp.write_text(json.dumps(payload))
        os.replace(temp, state)
        print(json.dumps(dict(stage=name, completed=int(end), total=shape[0])), flush=True)
    started = time.time()
    writer(output, start, commit)
    for i in range(0, len(output), 8):
        if not np.isfinite(output[i:i+8]).all():
            raise ValueError("Non-finite stage output")
    output.flush()
    del output
    os.replace(partial, final)
    result = dict(contract, contract_hash=contract_hash, sha256=sha256(final),
        wall_seconds=time.time()-started, status="complete", sources=dict(SOURCE_HASHES),
        performance_evaluated=False)
    atomic_json(manifest, result)
    return np.load(final, mmap_mode="r"), result


def file_identity(path: Path, full_hash=True):
    value = dict(path=portable(path), bytes=path.stat().st_size)
    if full_hash:
        value["sha256"] = sha256(path)
    return value


def recover_v7(device="cpu", batch_size=4, resume=False):
    """Exact saved historical V7 parent reconstruction, without fitting or truth."""
    torch = get_torch(device)
    op = operators(torch)
    lifecycle = source_namespace("run_v6_regime_component_lifecycle.py",
        ["Candidate", "_candidate_from_name", "candidate_list", "apply_candidate"],
        dict(torch=torch, TARGET_PATHS=dict(template=None,lifecycle_plain=None,lifecycle_visibility=None,lifecycle_shift=None), **vars(op)), ["PROFILE_ALPHA", "PROJECTION_CYCLES", "TINY", "PROJECTION_MODES"])
    base_dir = PATHS.artifacts / "round2" / "complex_carrier_lifecycle_overlay"
    target_dir = PATHS.artifacts / "round2" / "regime_lifecycle_support"
    base_path = base_dir / "oof_stacked_v4_projection_nested_scaled.npy"
    current_path = base_dir / "oof_stacked_current_ad_f16.npy"
    policy_path = base_dir / "stacked_lifecycle_selection.json"
    regime_path = target_dir / "regime_assignments.npz"
    ids, folds = canonical()
    # Position/site metadata only. No channel or cached truth-label reads.
    sys.path.insert(0, str(PATHS.round2_code))
    from round2.data import load_positions
    _, _, _, train_site, _ = load_positions()
    site = train_site[ids].astype(np.int8)
    regime = np.load(regime_path, allow_pickle=False)["oof_regime"].astype(np.int8)
    policy = json.loads(policy_path.read_text())["winner"]
    if policy is None:
        raise ValueError("Historical nested lifecycle policy missing")
    target_path = target_dir / {
        "template": "oof_regime_template_f16.npy",
        "lifecycle_plain": "oof_lifecycle_plain_f16.npy",
        "lifecycle_visibility": "oof_lifecycle_visibility_f16.npy",
        "lifecycle_shift": "oof_lifecycle_shift_f16.npy",
    }[policy["stage"]]
    selected = np.full(500, "identity", dtype="U72")
    assigned = np.zeros(500, dtype=bool)
    for row in policy["nested_decisions"]:
        regime_index = {"clear":0,"light":1,"medium":2,"heavy":3}.get(str(row.get("regime","")), row.get("regime_index",-1))
        mask = (folds == int(row["held_fold"])) & (site == int(row["site"])) & (regime == int(regime_index))
        if np.any(assigned[mask]):
            raise ValueError("Duplicated nested policy assignment")
        assigned[mask] = True
        selected[mask] = str(row["candidate"])
    if not assigned.all():
        raise ValueError("Nested policy does not cover canonical queries")
    base = np.load(base_path, mmap_mode="r")
    current = np.load(current_path, mmap_mode="r")
    target = np.load(target_path, mmap_mode="r")
    if base.shape != CHANNEL_SHAPE or current.shape != (500,16,8,192) or target.shape != current.shape:
        raise ValueError("Historical base/support shape mismatch")
    identity = dict(kind="historical_v7_parent_only", query_ids=ids.tolist(), canonical_fold=folds.tolist(),
        inputs=[file_identity(p) for p in [META,base_path,current_path,target_path,policy_path,regime_path]],
        sources=dict(SOURCE_HASHES), fits_executed=False,
        strict_current_protocol=False, final_v16=False)
    def writer(output, start, commit):
        for lo in range(start, 500, batch_size):
            hi = min(lo+batch_size, 500)
            block = np.array(base[lo:hi], dtype=np.complex64, copy=True)
            for name in np.unique(selected[lo:hi]):
                rows = np.flatnonzero(selected[lo:hi] == name)
                if name == "identity":
                    continue
                with torch.inference_mode():
                    predicted = lifecycle.apply_candidate(
                        torch.as_tensor(block[rows], device=device),
                        torch.as_tensor(np.asarray(current[lo:hi][rows], dtype=np.float32), device=device),
                        torch.as_tensor(np.asarray(target[lo:hi][rows], dtype=np.float32), device=device),
                        regime[lo:hi][rows], site[lo:hi][rows], lifecycle._candidate_from_name(str(name)))
                block[rows] = predicted.cpu().numpy().astype(np.complex64)
            output[lo:hi] = block
            commit(hi)
    return save_stage("v7_parent", identity, CHANNEL_SHAPE, np.complex64, writer, resume)


def load_endpoint_arrays():
    ids, folds = canonical()
    result = dict(mass=np.empty((500,128),np.float32), coordinate=np.empty((500,128,3),np.float32),
                  port=np.empty((500,128,2,4),np.float32))
    lookup = {int(q):i for i,q in enumerate(ids)}
    identities = []
    for outer in range(5):
        path = ENDPOINT_ROOT / f"outer{outer}_endpoint.npz"
        with np.load(path, allow_pickle=False) as f:
            expected = ids[folds == outer]
            if not np.array_equal(f["query_ids"], expected):
                raise ValueError("Historical endpoint IDs/fold differ from exact canonical mapping")
            rows = [lookup[int(q)] for q in expected]
            for name, output in result.items():
                if f[name].shape != output[rows].shape or not np.isfinite(f[name]).all():
                    raise ValueError(f"Bad endpoint {name}")
                output[rows] = f[name]
        identities.append(file_identity(path))
    if np.any(result["mass"] < 0) or np.any(result["port"] < 0):
        raise ValueError("Negative endpoint mass/port")
    return result, identities


def endpoint_renderer(torch):
    return source_namespace("analyze_v4_path_track_spectral_targets.py",
        ["spectral_kernels","render_noncoherent"], dict(torch=torch),
        ["MP","MH","MV","N_UE","S_SC","PORTS","TINY"])


def render_endpoint_batch(arrays, rows, torch, renderer, device):
    mass = arrays["mass"][rows]
    port = arrays["port"][rows]
    coefficient = np.sqrt(np.maximum(mass[:,:,None] * port.reshape(len(mass),128,8), 0)).astype(np.complex64)
    _, pas, pdp = renderer.render_noncoherent(arrays["coordinate"][rows], coefficient, torch.device(device))
    return pas.reshape(len(mass),2,16,8,4,192), pdp


def recover_endpoint_channel(base_manifest: Path, device="cpu", batch_size=4, resume=False, allow_hybrid=False):
    """Explicit sensitivity hybrid only; never a new-prior model forward."""
    if not allow_hybrid:
        raise ValueError("Old endpoints cannot certify regenerated-prior V16; explicit --allow-hybrid required")
    torch = get_torch(device)
    parent = json.loads(base_manifest.read_text())
    if parent.get("name") != "v11_parent" or parent.get("status") != "complete":
        raise ValueError("A completed v11_parent manifest is required; V7/E217 are forbidden")
    base_path = base_manifest.parent / "channels.npy"
    if sha256(base_path) != parent.get("sha256"):
        raise ValueError("V11 parent hash mismatch")
    base = np.load(base_path, mmap_mode="r")
    if base.shape != CHANNEL_SHAPE:
        raise ValueError("V11 parent shape differs")
    arrays, identities = load_endpoint_arrays()
    renderer = endpoint_renderer(torch)
    op = operators(torch)
    spec = types.SimpleNamespace(mode="sqrt", pas_beta=.20, pdp_beta=.20, order="pas_pdp", cycles=4)
    identity = dict(kind="hybrid_new_V11_frozen_old_endpoints", inputs=identities,
        parent=file_identity(base_path), parent_manifest=file_identity(base_manifest),
        query_ids=canonical()[0].tolist(), fits_executed=False, strict_current_protocol=False,
        qualification="hybrid_sensitivity_only_not_complete_new_prior_V16_forward",
        old_endpoints_used=True,full_current_model_forward=False,
        sources=dict(SOURCE_HASHES))
    def writer(output,start,commit):
        for lo in range(start,500,batch_size):
            hi=min(lo+batch_size,500)
            pas,pdp=render_endpoint_batch(arrays,slice(lo,hi),torch,renderer,device)
            b=torch.as_tensor(np.array(base[lo:hi],copy=True),device=device)
            with torch.inference_mode():
                y=op.make_candidate(b,op.pas_distribution(b),op.pdp_distribution(b),
                    torch.as_tensor(pas,device=device),torch.as_tensor(pdp,device=device),spec)
            output[lo:hi]=y.cpu().numpy().astype(np.complex64)
            commit(hi)
    return save_stage("hybrid_frozen_old_endpoints500",identity,CHANNEL_SHAPE,np.complex64,writer,resume)


def validate_external_inputs(manifest: Path):
    """No inference until every RT/output array is explicitly row-bound and hashed."""
    info=json.loads(manifest.read_text())
    if info.get("schema")!="v16-recovery-physical-inputs-v1":
        raise ValueError("Physical input schema mismatch")
    ids,_=canonical()
    if info.get("row_ids")!=ids.tolist():
        raise ValueError("Physical inputs must be in exact canonical query order")
    if info.get("seed_count")!=24 or info.get("seeds")!=list(ORIGINAL_SEEDS):
        raise ValueError("One-seed data cannot impersonate n24 V16 inputs")
    if info.get("source_configs")!=["primary_d16_base_v4lambda_1m","primary_d10_scatter015_v4lambda_1m"]:
        raise ValueError("Physical config differs; cannot call this historical-compatible replay")
    for key,producer_name in dict(old_pas="spectral_pas_endpoint",old_pdp="spectral_pdp_endpoint",
            joint_pas="joint_pas_endpoint",joint_pdp="joint_pdp_endpoint").items():
        producer=info.get("endpoint_producers",{}).get(key)
        if not producer:raise ValueError("Missing fold-matched spectrum producer provenance")
        pp=PATHS.resolve(producer["path"])
        if sha256(pp)!=producer["sha256"]:raise ValueError("Spectrum producer manifest changed")
        pm=json.loads(pp.read_text())
        identity=pm.get("identity",{})
        if pm.get("name")!=producer_name or pm.get("status")!="complete":
            raise ValueError("Production ensemble cannot replace fold-matched OOF spectra")
        if identity.get("canonical_ids")!=ids.tolist() or identity.get("canonical_fold")!=canonical()[1].tolist():
            raise ValueError("Spectrum producer row/fold mapping differs")
        if pm.get("sha256")!=info["arrays"][key]["sha256"] or len(identity.get("checkpoints",[]))!=5:
            raise ValueError("Spectrum array/checkpoint provenance differs")
    arrays={}
    expected={"old_pas":(500,2,16,8,4,192),"joint_pas":(500,2,16,8,4,192),
        "old_pdp":CHANNEL_SHAPE,"joint_pdp":CHANNEL_SHAPE,"base_pas_mean":(500,2,16,8,4,192),
        "scatter_pas_mean":(500,2,16,8,4,192),"raw_link_pdp":CHANNEL_SHAPE,
        "link_context":(500,143)}
    for name,shape in expected.items():
        item=info["arrays"][name]
        p=PATHS.resolve(item["path"])
        if sha256(p)!=item["sha256"]:
            raise ValueError(f"Physical input hash differs: {name}")
        a=np.load(p,mmap_mode="r")
        if a.shape!=shape:
            raise ValueError(f"{name}: {a.shape} != {shape}")
        for lo in range(0,500,16):
            if not np.isfinite(a[lo:lo+16]).all():
                raise ValueError(f"Nonfinite physical input {name}")
        arrays[name]=a
    return arrays,info


def recover_v9(v7_manifest: Path, physical_manifest: Path, device="cpu", batch_size=4, resume=False):
    torch=get_torch(device)
    arrays,physical=validate_external_inputs(physical_manifest)
    parent=json.loads(v7_manifest.read_text())
    p=v7_manifest.parent/"channels.npy"
    if parent.get("name")!="v7_parent" or parent.get("sha256")!=sha256(p):
        raise ValueError("Expected certified historical V7 parent")
    base=np.load(p,mmap_mode="r")
    op=operators(torch)
    v9=source_namespace("run_v9_joint_sionna_migration.py",["apply_candidate"],
        dict(torch=torch,**vars(op)),["TINY"])
    spec=types.SimpleNamespace(name="v9",pas_beta=.75,pdp_beta=.50,alpha_pas=.45,alpha_pdp=.35,cycles=4,order="pas_pdp")
    identity=dict(kind="historical_v9_parent",parent=file_identity(p),physical=file_identity(physical_manifest),
        sources=dict(SOURCE_HASHES),fits_executed=False,strict_current_protocol=False)
    def writer(out,start,commit):
        for lo in range(start,500,batch_size):
            hi=min(lo+batch_size,500)
            t=lambda a:torch.as_tensor(np.array(a[lo:hi],dtype=np.float32,copy=True),device=device)
            with torch.inference_mode():
                y=v9.apply_candidate(torch.as_tensor(np.array(base[lo:hi],copy=True),device=device),
                    t(arrays["old_pas"]),t(arrays["joint_pas"]),t(arrays["old_pdp"]),t(arrays["joint_pdp"]),spec)
            out[lo:hi]=y.cpu().numpy().astype(np.complex64);commit(hi)
    return save_stage("v9_parent",identity,CHANNEL_SHAPE,np.complex64,writer,resume)


def safe_checkpoint(path: Path, torch):
    """Existing local checkpoint, weights_only with narrow NumPy scalar support."""
    try:
        from numpy._core.multiarray import _reconstruct, scalar
    except ImportError:
        from numpy.core.multiarray import _reconstruct, scalar
    allowed=[_reconstruct,scalar,np.ndarray,np.dtype]
    allowed += [type(np.dtype(k)) for k in ("float16","float32","float64","int8","int32","int64")]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path,map_location="cpu",weights_only=True)


def certified_parent(manifest: Path,name: str):
    info=json.loads(manifest.read_text());p=manifest.parent/"channels.npy"
    if info.get("name")!=name or info.get("status")!="complete" or info.get("sha256")!=sha256(p):
        raise ValueError(f"Expected completed {name} parent")
    value=np.load(p,mmap_mode="r")
    if value.shape!=CHANNEL_SHAPE or value.dtype!=np.complex64:
        raise ValueError("Parent shape/dtype differs")
    return value,info,p


def recover_v10(v9_manifest: Path,physical_manifest: Path,device="cpu",batch_size=4,resume=False):
    torch=get_torch(device);op=operators(torch)
    arrays,_=validate_external_inputs(physical_manifest)
    base,_,parent_path=certified_parent(v9_manifest,"v9_parent")
    link=source_namespace("train_path_pdp_link_gate.py",
        ["LinkGate","link_coordinates","spectral_features"],dict(torch=torch,nn=torch.nn),["TINY"])
    ids,_=canonical()
    full_fold_path=PATHS.artifacts/"round2"/"full_path_pdp_link_gate"/"train_fold_id.npy"
    query_fold=np.load(full_fold_path,mmap_mode="r")[ids]
    models={}
    checkpoint_ids=[]
    for outer in range(5):
        p=BUNDLE/"artifacts"/"round2"/"physical_ai_submission"/"models"/"base_field"/"component_link"/f"fold_{outer}.pt"
        ck=safe_checkpoint(p,torch);cfg=ck["config"]
        model=link.LinkGate(int(ck["input_dim"]),int(cfg["hidden"]),float(cfg["dropout"])).to(device)
        model.load_state_dict(ck["state_dict"]);model.eval()
        if np.asarray(ck["context_mean"]).shape!=(143,) or np.asarray(ck["context_std"]).shape!=(143,):
            raise ValueError("Link context checkpoint shape differs")
        models[outer]=(model,ck)
        checkpoint_ids.append(file_identity(p))
    link_tensor=torch.as_tensor(link.link_coordinates(),device=device)
    identity=dict(kind="historical_v10_parent_FULL_FOLD_route",parent=file_identity(parent_path),
        physical=file_identity(physical_manifest),fold_asset=file_identity(full_fold_path),
        checkpoint_inputs=checkpoint_ids,sources=dict(SOURCE_HASHES),fits_executed=False,
        strict_current_protocol=False)
    def writer(out,start,commit):
        for lo in range(start,500,batch_size):
            hi=min(lo+batch_size,500)
            b=torch.as_tensor(np.array(base[lo:hi],copy=True),device=device)
            raw=torch.as_tensor(np.array(arrays["raw_link_pdp"][lo:hi],dtype=np.float32,copy=True),device=device)
            with torch.inference_mode():
                base_pdp=op.pdp_distribution(b)
                # The old replay saved and reloaded this intermediary as float16.
                base_pdp=base_pdp.to(torch.float16).float()
                alpha=torch.empty((hi-lo,1024),dtype=torch.float32,device=device)
                for outer in np.unique(query_fold[lo:hi]):
                    rows=np.flatnonzero(query_fold[lo:hi]==outer)
                    model,ck=models[int(outer)]
                    context=(np.asarray(arrays["link_context"][lo:hi][rows],np.float32)
                        -np.asarray(ck["context_mean"],np.float32))/np.asarray(ck["context_std"],np.float32)
                    features=link.spectral_features(base_pdp[rows].reshape(len(rows),1024,192),
                        raw[rows].reshape(len(rows),1024,192),torch.as_tensor(context,device=device),link_tensor)
                    alpha[rows]=model(features)
                # Match saved old alpha float16 before the 0.5 scaling.
                alpha=(.5*alpha.to(torch.float16).float()).clamp(0,1).reshape(hi-lo,256,4)
                desired=(1-alpha[...,None])*base_pdp+alpha[...,None]*raw
                desired/=desired.sum(dim=3,keepdim=True).clamp_min(1e-30)
                y=op.project_joint(b,op.pas_distribution(b),desired,2,"pas_pdp")
            out[lo:hi]=y.cpu().numpy().astype(np.complex64);commit(hi)
    return save_stage("v10_parent",identity,CHANNEL_SHAPE,np.complex64,writer,resume)


def recover_v11(v10_manifest: Path,physical_manifest: Path,device="cpu",batch_size=4,resume=False):
    torch=get_torch(device);op=operators(torch)
    arrays,_=validate_external_inputs(physical_manifest)
    base,_,parent_path=certified_parent(v10_manifest,"v10_parent")
    low=source_namespace("analyze_v11_low_rank_path_pas_residual.py",
        ["normalize_pas","pas_slices","pas_from_slices"],dict(torch=torch),["TINY"])
    reliable=source_namespace("analyze_v11_reliable_sqrt_pas_gate.py",
        ["percentile_rank"],dict(torch=torch),["TINY"])
    consensus=source_namespace("analyze_v11_sqrt_pas_source_consensus.py",
        ["build_consensus"],dict(torch=torch),["TINY"])
    # Source arrays are already canonical500 ordered. Preserve the historical
    # roll-by-two applied to raw Base/Scatter PAS in E77, even for site1.
    def source_amplitudes(lo,hi):
        result=[]
        for k,name in enumerate(("old_pas","joint_pas","base_pas_mean","scatter_pas_mean")):
            a=np.asarray(arrays[name][lo:hi],np.float32)
            if k>=2:a=np.roll(a,2,axis=3)
            a=torch.as_tensor(np.array(a,copy=True),device=device)
            result.append(torch.sqrt(low.pas_slices(low.normalize_pas(a)).clamp_min(0)))
        return result
    raw=np.empty((500,4,192,2),np.float32)
    for lo in range(0,500,batch_size):
        hi=min(lo+batch_size,500)
        b=torch.as_tensor(np.array(base[lo:hi],copy=True),device=device)
        a=torch.sqrt(low.pas_slices(low.normalize_pas(op.pas_distribution(b))).clamp_min(0))
        sources=source_amplitudes(lo,hi)
        residual=torch.stack([s-a for s in sources])
        raw[lo:hi,:,:,0]=torch.stack([s.square() for s in sources]).std(dim=0).mean(dim=3).cpu().numpy()
        raw[lo:hi,:,:,1]=residual.square().sum(dim=4).sqrt().median(dim=0).values.cpu().numpy()
    # Transductive quantile uses predicted features only, no channel labels.
    rt=torch.as_tensor(raw,device=device)
    rank=.5*(reliable.percentile_rank(rt[...,0])+reliable.percentile_rank(rt[...,1]))
    gate=(rank<=.70).to(torch.float16).float()
    identity=dict(kind="historical_v11_parent",parent=file_identity(parent_path),
        physical=file_identity(physical_manifest),sources=dict(SOURCE_HASHES),fits_executed=False,
        strict_current_protocol=False,rank_scope="all_500_query_features_no_labels")
    def writer(out,start,commit):
        for lo in range(start,500,batch_size):
            hi=min(lo+batch_size,500)
            b=torch.as_tensor(np.array(base[lo:hi],copy=True),device=device)
            with torch.inference_mode():
                a=torch.sqrt(low.pas_slices(low.normalize_pas(op.pas_distribution(b))).clamp_min(0))
                target=consensus.build_consensus(a,source_amplitudes(lo,hi),tuple(range(4)))
                desired=(a+.20*gate[lo:hi,:, :,None]*(target-a)).clamp_min(0).square()
                desired/=desired.sum(dim=3,keepdim=True).clamp_min(1e-30)
                y=op.project_joint(b,low.pas_from_slices(desired),op.pdp_distribution(b),4,"pdp_pas")
            out[lo:hi]=y.cpu().numpy().astype(np.complex64);commit(hi)
    return save_stage("v11_parent",identity,CHANNEL_SHAPE,np.complex64,writer,resume)


def verify_saved_endpoints(device="cpu",batch_size=8):
    """All500 saved endpoint rows, no H truth or final V16 prediction."""
    torch=get_torch(device);renderer=endpoint_renderer(torch)
    arrays,identities=load_endpoint_arrays()
    maximum_pas_normalization=0.;maximum_pdp_normalization=0.
    for lo in range(0,500,batch_size):
        hi=min(lo+batch_size,500)
        pas,pdp=render_endpoint_batch(arrays,slice(lo,hi),torch,renderer,device)
        if pas.shape!=(hi-lo,2,16,8,4,192) or pdp.shape!=(hi-lo,256,4,192):
            raise ValueError("Endpoint renderer shape differs")
        if not np.isfinite(pas).all() or not np.isfinite(pdp).all() or np.min(pas)<0 or np.min(pdp)<0:
            raise ValueError("Endpoint renderer invalid values")
        ps=pas.sum(axis=(1,2,3),dtype=np.float64)
        ds=pdp.sum(axis=3,dtype=np.float64)
        maximum_pas_normalization=max(maximum_pas_normalization,float(np.max(np.minimum(np.abs(ps),np.abs(ps-1)))))
        maximum_pdp_normalization=max(maximum_pdp_normalization,float(np.max(np.minimum(np.abs(ds),np.abs(ds-1)))))
    pas_a,pdp_a=render_endpoint_batch(arrays,slice(0,8),torch,renderer,device)
    one=[render_endpoint_batch(arrays,slice(i,i+1),torch,renderer,device) for i in range(8)]
    pas_b=np.concatenate([v[0] for v in one]);pdp_b=np.concatenate([v[1] for v in one])
    chunk_error=max(float(np.max(np.abs(pas_a-pas_b))),float(np.max(np.abs(pdp_a-pdp_b))))
    if chunk_error>2e-6 or max(maximum_pas_normalization,maximum_pdp_normalization)>3e-5:
        raise AssertionError("Renderer chunk/normalization failure")
    return dict(schema=SCHEMA,status="passed",rows=500,paths_per_row=128,
        all_rows_shape_finite_nonnegative=True,chunk_max_abs=chunk_error,
        pas_normalization_max_abs=maximum_pas_normalization,pdp_normalization_max_abs=maximum_pdp_normalization,
        endpoint_inputs=identities,sources=dict(SOURCE_HASHES),device=device,
        no_truth_channel_read=True,optimizer_steps=0,performance_evaluated=False,final_v16_prediction=False)


def strict_roles(outer: int):
    if outer not in range(5):raise ValueError(outer)
    with np.load(FOLDS,allow_pickle=False) as f:
        refit=f[f"fold{outer}_fit"].astype(np.int64)
        held=f[f"fold{outer}_query"].astype(np.int64)
        validation=f[f"fold{(outer+1)%5}_query"].astype(np.int64)
        screen=np.intersect1d(refit,f[f"fold{(outer+1)%5}_fit"]).astype(np.int64)
    for x in [refit,held,validation,screen]:
        if len(x)!=len(np.unique(x)):raise ValueError("Duplicate role IDs")
    if np.intersect1d(refit,held).size or np.intersect1d(screen,validation).size:
        raise ValueError("Role leakage")
    return dict(screen_fit=screen,validation=validation,refit=refit,outer=held)


class FitOnlyLabels:
    """Capability: trainer receives no truth-array path or outer/guard labels."""
    def __init__(self,allowed_ids: np.ndarray):
        self.allowed_ids=np.asarray(allowed_ids,np.int64).copy()
        self._allowed=set(self.allowed_ids.tolist())
        self.access=[]
    def read(self,ids: np.ndarray):
        ids=np.asarray(ids,np.int64)
        if ids.ndim!=1 or not len(ids) or not set(ids.tolist()).issubset(self._allowed):
            raise PermissionError("Attempted label access outside this fit role")
        truth=np.load(PATHS.round2/"Round2_Train_Channel.npy",mmap_mode="r")
        value=np.array(truth[ids],dtype=np.complex64,copy=True)
        self.access.append(ids.tolist())
        return value


STRICT_COMPONENTS=("carrier_estimation","carrier_stack","rbf_spectral_field","donor_network","group_calibration",
    "projection_lifecycle_policy","spectral_mean_pas","spectral_mean_pdp","spectral_joint_pas",
    "spectral_joint_pdp","link_gate","pathstate")


def validate_reusable_artifact(record: dict,outer: int,role: str,component: str):
    allowed=strict_roles(outer)[role]
    if record.get("schema")!="v16-strict-component-v1" or record.get("outer")!=outer or record.get("role")!=role:
        raise ValueError("Component role provenance missing")
    if record.get("component")!=component or record.get("protocol_sha256")!=sha256(FOLDS):
        raise ValueError("Component/protocol identity mismatch")
    # Require all label-sensitive transitive dependencies, not merely final NN IDs.
    required=("training","normalization","basis","calibration","policy")
    if not all(k in record.get("label_ids",{}) for k in required):
        raise ValueError("Incomplete transitive label provenance")
    for kind,ids in record["label_ids"].items():
        if not set(ids).issubset(set(allowed.tolist())):
            raise ValueError(f"{kind} includes forbidden labels")
    if record.get("fit_ids")!=allowed.tolist():
        raise ValueError("Exact fit set differs; architecture-only reuse is not weight qualification")
    p=PATHS.resolve(record["artifact"]["path"])
    if sha256(p)!=record["artifact"]["sha256"]:raise ValueError("Artifact bytes changed")
    return p


class FoldSafePipeline:
    """Runs explicit component trainers; no automatic 31-model launch or scoring.

    A trainer signature is (component, role, fit_ids, labels, upstream, output_dir).
    Returned records must pass the same qualification as a reused checkpoint.
    Geometry arrays are allowed at all query positions; their channel labels are not.
    """
    def __init__(self,outer: int,run_name: str):
        if not run_name or Path(run_name).name!=run_name:raise ValueError("Invalid run name")
        self.outer=outer;self.roles=strict_roles(outer)
        self.root=protected_output(ARTIFACT_ROOT/"strict"/run_name/f"outer{outer}")
        self.root.mkdir(parents=True,exist_ok=True)
        self.records={}
    def run_component(self,component: str,role: str,*,reuse=None,trainer=None):
        if component not in STRICT_COMPONENTS or role not in {"screen_fit","refit"}:
            raise ValueError("Unknown component/fit role")
        key=f"{role}/{component}"
        if key in self.records:raise ValueError("Stage already completed")
        if reuse is not None:
            record=json.loads(Path(reuse).read_text())
            validate_reusable_artifact(record,self.outer,role,component)
        else:
            if trainer is None:
                raise RuntimeError(f"{key} needs an explicit new fold-safe fit; unproved production reuse denied")
            fit=self.roles[role].copy()
            labels=FitOnlyLabels(fit)
            out=self.root/role/component
            out.mkdir(parents=True,exist_ok=False)
            record=trainer(component,role,fit,labels,dict(self.records),out)
            validate_reusable_artifact(record,self.outer,role,component)
            record=dict(record,label_access=labels.access)
            atomic_json(out/"component.json",record)
        self.records[key]=record
        return record
    def declaration(self):
        return dict(schema="v16-strict-recovery-work-order-v1",outer=self.outer,
            protocol_sha256=sha256(FOLDS),roles={k:v.tolist() for k,v in self.roles.items()},
            required_components=list(STRICT_COMPONENTS),completed=list(self.records),
            status="not_fit_not_a_v16_prediction",performance_evaluated=False)


def inspect_recovery():
    ids,folds=canonical()
    arrays,ep=load_endpoint_arrays()
    roots=["pre_sionna","rbf_spectral_field","v3_anchored_neural_rbf","complex_carrier_bank",
           "complex_carrier_stacking","complex_carrier_lifecycle_overlay","regime_lifecycle_support"]
    inventory=[]
    for name in roots:
        root=PATHS.artifacts/"round2"/name
        for p in sorted(root.glob("*")):
            if p.is_file():inventory.append(file_identity(p,full_hash=p.stat().st_size<(1<<20)))
    return dict(schema=SCHEMA,canonical_ids=ids.tolist(),fold_counts=np.bincount(folds).tolist(),
        endpoint_inputs=ep,retained_inventory=inventory,
        available_no_fit=["historical_v7_parent","saved_local_pathstate_endpoint_renderer"],
        missing_to_v11=["n24_base_scatter_physical_statistics","20_fold_spectral_predictions",
            "raw_link_pdp_and_143context","fold_matched_link_inference"],
        strict_roles=[{k:len(v) for k,v in strict_roles(i).items()} for i in range(5)],
        no_training=True,no_performance_evaluation=True)


def synthetic_checks():
    torch=get_torch("cpu");op=operators(torch);renderer=endpoint_renderer(torch)
    torch.set_num_threads(2)
    rng=np.random.default_rng(20261031)
    base=torch.from_numpy((rng.standard_normal((2,256,4,192))+1j*rng.standard_normal((2,256,4,192))).astype(np.complex64))
    pas=op.pas_distribution(base);pdp=op.pdp_distribution(base)
    reconstructed=op.project_joint(base,pas,pdp,4,"pas_pdp")
    err=float((reconstructed-base).abs().max())
    if err>2e-5:raise AssertionError(("projection identity",err))
    coord=np.zeros((2,128,3),np.float32);coord[:,0]=[3,2,7]
    coeff=np.zeros((2,128,8),np.complex64);coeff[:,0]=1
    _,a,b=renderer.render_noncoherent(coord,coeff,torch.device("cpu"))
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise AssertionError("renderer nonfinite")
    if not np.all(np.argmax(b,axis=-1)==7):raise AssertionError("absolute delay convention")
    labels=FitOnlyLabels(np.array([0,2]))
    try:labels.read(np.array([1]))
    except PermissionError:pass
    else:raise AssertionError("label capability failed")
    # Does not read even fit CSI. Fail before opening channel.
    expected=canonical()[0]
    for o in range(5):
        r=strict_roles(o)
        if not np.array_equal(r["outer"],expected[canonical()[1]==o]):
            raise AssertionError("fold mapping")
    return dict(status="passed",projection_identity_max_abs=err,absolute_delay_bin=7,
        fit_label_denial=True,canonical_fold_binding=True,sources=dict(SOURCE_HASHES),
        real_channel_prediction=False,optimizer_steps=0,performance_evaluated=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["inspect","synthetic","v7","v9","v10","v11","endpoint","endpoint-check","strict-plan"])
    p.add_argument("--device",default="cpu")
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--allow-hybrid",action="store_true")
    p.add_argument("--report",default=None)
    p.add_argument("--base-manifest",type=Path)
    p.add_argument("--physical-manifest",type=Path)
    p.add_argument("--outer",type=int,default=0)
    p.add_argument("--run-name",default="canonical_guard_v1")
    a=p.parse_args()
    SOURCE_HASHES[portable(Path(__file__))]=sha256(Path(__file__))
    if a.batch_size<1:raise ValueError("batch-size must be positive")
    if a.stage=="inspect":result=inspect_recovery()
    elif a.stage=="synthetic":result=synthetic_checks()
    elif a.stage=="endpoint-check":result=verify_saved_endpoints(a.device,a.batch_size)
    elif a.stage=="v7":_,result=recover_v7(a.device,a.batch_size,a.resume)
    elif a.stage=="v9":
        if a.base_manifest is None or a.physical_manifest is None:p.error("v9 needs both manifests")
        _,result=recover_v9(a.base_manifest,a.physical_manifest,a.device,a.batch_size,a.resume)
    elif a.stage in {"v10","v11"}:
        if a.base_manifest is None or a.physical_manifest is None:p.error("stage needs both manifests")
        fn=recover_v10 if a.stage=="v10" else recover_v11
        _,result=fn(a.base_manifest,a.physical_manifest,a.device,a.batch_size,a.resume)
    elif a.stage=="endpoint":
        if a.base_manifest is None:p.error("endpoint needs certified V11 manifest")
        _,result=recover_endpoint_channel(a.base_manifest,a.device,a.batch_size,a.resume,a.allow_hybrid)
    else:result=FoldSafePipeline(a.outer,a.run_name).declaration()
    if a.report:atomic_json(REPORT_ROOT/a.report,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

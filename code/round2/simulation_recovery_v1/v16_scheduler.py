"""Execute frozen-weight regenerated-query V16 recovery, without quality scoring.

Run only after geometry worker releases the assigned A800. Each numerical stage
runs in a fresh process, isolating source namespaces and deterministic resumes.
The saved old endpoint hybrid is deliberately absent from this scheduler.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from v16_recovery import (PATHS,ARTIFACT_ROOT,REPORT_ROOT,ALLOWED_GPU,canonical,
    file_identity,sha256,protected_output,certified_parent,atomic_json)
from v16_physical_inference import assemble_physical_manifest

HERE=Path(__file__).parent
FINAL_NAME="legacy_fixed_weights_regenerated_query_v3"


def command_plan(statistics: Path,tokens: Path,device: str):
    physical=ARTIFACT_ROOT/"historical_replay"/"physical_inputs_v3.json"
    query=ARTIFACT_ROOT/"historical_replay"/"regenerated_query_inputs"/"manifest.json"
    parent=lambda name:ARTIFACT_ROOT/"historical_replay"/name/"manifest.json"
    state=ARTIFACT_ROOT/"historical_replay"/"production_pathstate_FULL_FOLD"/"manifest.json"
    return [
        ("tokens_finalize","v16_tokens_finalize.py",["--statistics",str(statistics)]),
        ("prior_identity","v16_prior_identity.py",["--statistics",str(statistics)]),
        ("query_inputs","v16_query_inputs.py",["--statistics",str(statistics),"--tokens",str(tokens),"--device",device]),
        ("spectral20","v16_physical_inference.py",["--statistics",str(statistics),"--device",device,"--resume"]),
        ("assemble",None,[]),
        ("V9","v16_recovery.py",["v9","--base-manifest",str(parent("v7_parent")),"--physical-manifest",str(physical),"--device",device,"--resume"]),
        ("V10","v16_recovery.py",["v10","--base-manifest",str(parent("v9_parent")),"--physical-manifest",str(physical),"--device",device,"--resume"]),
        ("V11","v16_recovery.py",["v11","--base-manifest",str(parent("v10_parent")),"--physical-manifest",str(physical),"--device",device,"--resume"]),
        ("PathState5","v16_production_pathstate.py",["states","--query-inputs",str(query),"--device",device]),
        ("synthesize","v16_production_pathstate.py",["synthesize","--base-manifest",str(parent("v11_parent")),
            "--state-manifest",str(state),"--device",device,"--resume"]),
    ]


def portable_command(script,args):
    def relative(value):
        if value.startswith(str(PATHS.root)+os.sep):return value[len(str(PATHS.root))+1:]
        return value
    return ["python",PATHS.relative(HERE/script),*[relative(v) for v in args]]


def run(statistics: Path,tokens: Path,device="cpu",plan_only=False):
    statistics=PATHS.resolve(statistics);tokens=PATHS.resolve(tokens)
    plan=command_plan(statistics,tokens,device)
    if plan_only:
        return dict(schema="v16-forward-scheduler-plan-v1",final_name=FINAL_NAME,
            steps=[dict(stage=s,command=portable_command(p,a) if p else "assemble_certified_physical_manifest") for s,p,a in plan],
            no_old_endpoint_hybrid=True,no_training=True,no_quality_score=True,
            canonical_strict_qualification=False,head_training_exclusion_certified=False)
    if device=="cuda:0" and os.environ.get("CUDA_VISIBLE_DEVICES")!=ALLOWED_GPU:
        raise RuntimeError("Assigned A800 UUID must be exclusively released and explicitly selected before starting")
    if device not in {"cpu","cuda:0"}:raise ValueError(device)
    # The previously recovered true V7 is a frozen parent, not the final output.
    v7=ARTIFACT_ROOT/"historical_replay"/"v7_parent"/"manifest.json"
    certified_parent(v7,"v7_parent")
    root=protected_output(REPORT_ROOT/"v16_forward_v3")
    root.mkdir(parents=True,exist_ok=True)
    records=[];started=time.time()
    env=dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"]="" if device=="cpu" else ALLOWED_GPU
    env["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";env.setdefault("OMP_NUM_THREADS","4")
    for stage,script,args in plan:
        tick=time.time()
        print(json.dumps(dict(stage=stage,status="starting",elapsed=time.time()-started)),flush=True)
        if script is None:
            query=ARTIFACT_ROOT/"historical_replay"/"regenerated_query_inputs"/"manifest.json"
            physical=ARTIFACT_ROOT/"historical_replay"/"physical_inputs_v3.json"
            assemble_physical_manifest(statistics,query,physical)
            code=0;log=None
        else:
            log=root/f"{stage}.log"
            with log.open("a") as stream:
                stream.write("\n=== new invocation ===\n");stream.flush()
                code=subprocess.run([sys.executable,str(HERE/script),*args],cwd=PATHS.root,
                    env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        record=dict(stage=stage,exit_code=code,wall_seconds=time.time()-tick,
            log=PATHS.relative(log) if log else None,command=portable_command(script,args) if script else None)
        records.append(record)
        state=dict(schema="v16-forward-scheduler-state-v1",stages=records,
            status="failed" if code else "running",statistics=file_identity(statistics),tokens=file_identity(tokens))
        temp=root/"state.partial.json";temp.write_text(json.dumps(state,indent=2));temp.replace(root/"state.json")
        print(json.dumps(record),flush=True)
        if code:raise RuntimeError(f"Stage {stage} failed; inspect its preserved log")
    manifest=ARTIFACT_ROOT/"historical_replay"/FINAL_NAME/"manifest.json"
    meta=json.loads(manifest.read_text());output=manifest.parent/"channels.npy"
    channel=np.load(output,mmap_mode="r")
    ids,outer=canonical()
    if channel.shape!=(500,256,4,192) or channel.dtype!=np.complex64:raise ValueError("Final H shape/dtype invalid")
    for lo in range(0,500,8):
        if not np.isfinite(channel[lo:lo+8]).all():raise ValueError("Final H nonfinite")
    identity=meta["identity"]
    full=np.load(PATHS.artifacts/"round2/full_path_pdp_link_gate/train_fold_id.npy",mmap_mode="r")[ids]
    if identity.get("canonical_outer_id")!=outer.tolist() or identity.get("head_train_fold_id")!=full.tolist():
        raise ValueError("Final dual fold/row identity differs")
    if identity.get("old_endpoints_used") or identity.get("canonical_strict_qualification") or identity.get("head_training_exclusion_certified"):
        raise ValueError("Final output carries an invalid qualification")
    if meta["sha256"]!=sha256(output):raise ValueError("Final H hash differs")
    result=dict(schema="v16-frozen-weights-regenerated-query-delivery-v1",status="complete",
        method=FINAL_NAME,channel=file_identity(output),manifest=file_identity(manifest),
        shape=list(channel.shape),dtype=str(channel.dtype),finite=True,
        query_ids=ids.tolist(),canonical_outer_id=outer.tolist(),head_train_fold_id=full.tolist(),
        statistics=file_identity(statistics),tokens=file_identity(tokens),
        actual_V11_parent=identity["parent_V11"],production_states=identity["production_state_manifest"],
        stages=records,wall_seconds=time.time()-started,device=device,
        GPU_UUID=ALLOWED_GPU if device=="cuda:0" else None,
        original_weights=True,regenerated_prior=True,old_cache_reproduction=False,
        old_endpoints_used=False,head_training_exclusion_certified=False,canonical_strict_qualification=False,
        new_neural_training=False,new_CSI_parameter_fit=False,new_geometry_quantile_preprocessing=True,
        quality_score_computed=False,official_Test_output_changed=False)
    path=root/"verification.json"
    if path.exists():
        # A completed prior delivery remains immutable. A resume rechecks data
        # but does not overwrite its original timing/command evidence.
        old=json.loads(path.read_text())
        if old["channel"]!=result["channel"] or old["statistics"]!=result["statistics"]:
            raise ValueError("Completed delivery differs")
        return old
    atomic_json(path,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--statistics",type=Path,required=True);p.add_argument("--tokens",type=Path)
    p.add_argument("--device",default="cpu");p.add_argument("--plan-only",action="store_true")
    a=p.parse_args();tokens=a.tokens or a.statistics.parent/"live_tokens"/"tokens"/"manifest.json"
    result=run(a.statistics,tokens,a.device,a.plan_only)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":main()

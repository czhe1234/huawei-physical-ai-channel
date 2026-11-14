"""Scheduling-only NeRF2 entry: reuse frozen mathematics on an allowed GPU.

No frozen file is edited. The legacy module is compiled in an isolated process
with only its top-level GPU_UUID binding changed. Frozen model/training/render
functions are reused directly. Per-fold locks permit disjoint fold queues.
"""
from __future__ import annotations
import argparse, ast, copy, hashlib, importlib, importlib.util, json, marshal, os
from pathlib import Path
import signal, sys, time, types
from contextlib import contextmanager
import numpy as np
import torch
from project_paths import PATHS
from round2.paper_baseline_training_v1 import common as C

BASE = PATHS.code_root/"round2/baseline_formal_20261111/nerf2.py"
LEGACY = PATHS.code_root/"round2/paper_baseline_training_v1/field_runner.py"
BASE_SHA = "d210f6c817141a4055a4a42689cd9b41d316aef5cfcb29dd93a7b1bb10aae626"
LEGACY_SHA = "cdf21f0534b8d616511cebddb09b8b72ca325c3ab7e1e8ef6fb3b28192213ed7"
METHOD_SHA = "c298cb71156e7aa78eec1f374ea8bb36fba47d5ef8a7a54b76dbeee5cf258a35"
CANONICAL_MODULE = "round2.baseline_formal_20261111.nerf2"
RUNTIME_SCHEMA = "nerf2-execution-only-portable-v1"
ROOT = C.REPORT/"formal_20261111/nerf2"

def digest_tree(tree):
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()

def normalized_device_tree(tree):
    tree = copy.deepcopy(tree); found = 0
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id=="GPU_UUID" for t in node.targets):
            node.value = ast.Constant("DEVICE_BINDING_ONLY"); found += 1
    if found != 1:
        raise RuntimeError("Exactly one legacy top-level GPU_UUID assignment required")
    return digest_tree(tree)

def isolated_import(gpu):
    if gpu not in C.ALLOWED:
        raise PermissionError("Only explicitly allowed full UUIDs may be used")
    if C.sha256(BASE) != BASE_SHA or C.sha256(LEGACY) != LEGACY_SHA:
        raise RuntimeError("Frozen source changed")
    name = "round2.paper_baseline_training_v1.field_runner"
    if name in sys.modules:
        raise RuntimeError("Portable entry requires a fresh isolated Python process")
    original = ast.parse(LEGACY.read_text(), filename=str(LEGACY))
    modified = copy.deepcopy(original); changed = 0
    old_uuid = None
    for node in modified.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id=="GPU_UUID" for t in node.targets):
            if not isinstance(node.value, ast.Constant) or node.value.value not in C.ALLOWED:
                raise RuntimeError("Unexpected legacy device binding")
            old_uuid = node.value.value; node.value = ast.Constant(gpu); changed += 1
    ast.fix_missing_locations(modified)
    if changed != 1 or normalized_device_tree(original) != normalized_device_tree(modified):
        raise RuntimeError("Changes outside the device binding are forbidden")
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(LEGACY))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(LEGACY); module.__package__ = name.rsplit(".", 1)[0]
    sys.modules[name] = module
    setattr(importlib.import_module(module.__package__), "field_runner", module)
    exec(compile(modified, str(LEGACY), "exec"), module.__dict__)
    frozen = importlib.import_module(CANONICAL_MODULE)
    functions = {k: hashlib.sha256(marshal.dumps(v.__code__)).hexdigest()
                 for k, v in vars(frozen).items() if isinstance(v, types.FunctionType)
                 and v.__module__==CANONICAL_MODULE}
    before_gpu = frozen.GPU; frozen.GPU = gpu
    after = {k: hashlib.sha256(marshal.dumps(v.__code__)).hexdigest()
             for k, v in vars(frozen).items() if isinstance(v, types.FunctionType)
             and v.__module__==CANONICAL_MODULE}
    if after != functions or module.GPU_UUID != gpu:
        raise RuntimeError("Frozen function code changed")
    proof = dict(frozen_base_sha256=BASE_SHA, frozen_legacy_sha256=LEGACY_SHA,
        legacy_original_AST_sha256=digest_tree(original), legacy_device_bound_AST_sha256=digest_tree(modified),
        legacy_device_normalized_AST_sha256=normalized_device_tree(original),
        only_legacy_AST_change="one top-level GPU_UUID constant",
        legacy_original_GPU_UUID=old_uuid, legacy_runtime_GPU_UUID=gpu,
        frozen_module_original_GPU=before_gpu, frozen_module_runtime_GPU=gpu,
        all_frozen_function_code_sha256=functions, frozen_function_bytecode_unchanged=True,
        no_disk_source_modified=True, mathematical_functions_reused_directly=True)
    return frozen, proof

def read_contract(path):
    runtime = json.loads(path.read_text())
    allowed = {"schema","gpu_uuid","folds","method_config","method_config_sha256",
               "frozen_base_sha256","frozen_legacy_sha256","execution_only","frozen","note","qa_min_free_GiB"}
    if set(runtime)-allowed or runtime["schema"] != RUNTIME_SCHEMA or runtime["execution_only"] is not True:
        raise ValueError("Runtime contract may contain only scheduling/device information")
    gpu = runtime["gpu_uuid"]
    if gpu not in C.ALLOWED:
        raise PermissionError("Runtime GPU not in allowed UUIDs")
    if runtime["frozen_base_sha256"] != BASE_SHA or runtime["frozen_legacy_sha256"] != LEGACY_SHA:
        raise RuntimeError("Runtime frozen source binding differs")
    config_path = PATHS.root/runtime["method_config"]
    if runtime["method_config_sha256"] != METHOD_SHA or C.sha256(config_path) != METHOD_SHA:
        raise RuntimeError("Mathematical training configuration must remain exactly frozen")
    folds = runtime["folds"]
    if not folds or len(set(folds)) != len(folds) or any(type(x) is not int or x not in range(5) for x in folds):
        raise ValueError("Distinct canonical fold integers required")
    cfg = json.loads(config_path.read_text()); cfg["_config_rel"] = PATHS.relative(config_path)
    if not cfg.get("formal_frozen"):
        raise RuntimeError("Original formal configuration must be frozen")
    return runtime, cfg, config_path

def owned_base_worker_queues():
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            args = (entry/"cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0")
            if CANONICAL_MODULE not in args or "--mode" not in args or args[args.index("--mode")+1]!="run":
                continue
            if "--folds" in args:
                text = args[args.index("--folds")+1]
            else:
                eq = [a.split("=",1)[1] for a in args if a.startswith("--folds=")]
                text = eq[0] if eq else "0,1,2,3,4"
            found.append(dict(pid=int(entry.name),folds=list(map(int,text.split(",")))))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found

def report_dir(folds):
    p=ROOT/("portable_"+"_".join(map(str,folds))); p.mkdir(parents=True,exist_ok=True)
    return p

def sources(frozen, config_path, runtime_path):
    d = frozen.identity(config_path)
    d[PATHS.relative(Path(__file__))] = C.sha256(Path(__file__))
    d[PATHS.relative(runtime_path)] = C.sha256(runtime_path)
    return d

def publication_cpu_check(report):
    base=report/("publication_cpu_check_"+C.sha256(Path(__file__))[:12])
    base.mkdir(parents=True,exist_ok=True)
    canonical=base/"canonical_manifest.json";pending=base/"pending_manifest.json"
    unrelated=base/"unrelated_progress.json";original=C.atomic_json
    class ProbeFailure(Exception):
        pass
    try:
        with defer_manifest_publication(canonical,pending) as routed:
            C.atomic_json(canonical,{"stage":"unannotated"})
            C.atomic_json(unrelated,{"event":"progress"})
            if canonical.exists() or not pending.exists() or not unrelated.exists() or len(routed)!=1:
                raise AssertionError("Exact-path routing/ordinary passthrough failed")
            raise ProbeFailure("intentional CPU exception")
    except ProbeFailure:
        pass
    if C.atomic_json is not original:
        raise AssertionError("finally did not restore C.atomic_json")
    fake=types.SimpleNamespace(ART=base/"artifact",REPORT=base/"report",C=C)
    fold=4;target=fake.ART/f"outer{fold}/prediction/manifest.json"
    rr=fake.REPORT/f"outer{fold}/prediction";fixture=base/"fixture"
    rr.mkdir(parents=True,exist_ok=True);fixture.mkdir(exist_ok=True)
    source={"CPU_fixture":"same_frozen_source"};refit={"checkpoint_sha256":"CPU_checkpoint"}
    calls={"fresh":0,"published_reader":0}
    def predictor(_fold,_cfg,_source,_selected,_refit):
        calls["fresh"]+=1
        if calls["fresh"]!=1:
            raise AssertionError("Fresh predictor called more than once")
        raw=target.parent/"channels.npy";rows=target.parent/"row_ids.npy"
        C.atomic_npy(raw,np.zeros((1,256,4,192),np.complex64))
        C.atomic_npy(rows,np.array([7],np.int64))
        manifest=dict(source_identity=_source,checkpoint_sha256=_refit["checkpoint_sha256"],
            path=PATHS.relative(raw),sha256=C.sha256(raw),row_ids=[7],metadata={},
            shape=[1,256,4,192],dtype="complex64")
        C.atomic_json(target,manifest)
        if target.exists():
            raise AssertionError("Fresh writer exposed canonical manifest before annotation")
        C.atomic_json(rr/"result.json",dict(status="complete",manifest=manifest))
        return manifest
    fake.outer_prediction=predictor
    manifest=staged_outer_prediction(fake,fold,{},source,{},refit,fixture)
    if calls["fresh"]!=1 or target.exists() or C.atomic_json is not original:
        raise AssertionError("Fresh staged predictor or writer restoration failed")
    pending_file=fixture/f"outer{fold}_pending_canonical_manifest.json"
    if json.loads(pending_file.read_text())!=manifest:
        raise AssertionError("Fresh pending/return value differ")
    raw=target.parent/"channels.npy"
    stat=lambda p:(p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns)
    raw_before=(C.sha256(raw),stat(raw))
    def prohibited_predictor(*_args):
        raise AssertionError("Pending completed output must not be recomputed")
    fake.outer_prediction=prohibited_predictor
    retry=staged_outer_prediction(fake,fold,{},source,{},refit,fixture)
    if retry!=manifest or target.exists() or (C.sha256(raw),stat(raw))!=raw_before:
        raise AssertionError("Pending resume changed completed raw data")
    if C.atomic_json is not original:
        raise AssertionError("Pending resume changed JSON writer")
    # Direct fresh-stage exception must also restore the original writer.
    try:
        staged_outer_prediction(fake,3,{},source,{},refit,fixture)
    except AssertionError as error:
        if str(error)!="Pending completed output must not be recomputed":
            raise
    else:
        raise AssertionError("Intentional fresh predictor exception did not propagate")
    if C.atomic_json is not original:
        raise AssertionError("Fresh predictor exception leaked writer patch")
    events=[]
    def observed(path,value):
        events.append((str(Path(path).resolve()),copy.deepcopy(value)))
        return original(path,value)
    proof={"CPU_fixture":True};runtime={"CPU_fixture":True}
    device={"GPU_UUID":"CPU_fixture","CUDA_context_UUID":"CPU_fixture"}
    C.atomic_json=observed
    try:
        out=annotate_manifest(fake,fold,retry,runtime,proof,source,device,fixture)
    finally:
        C.atomic_json=original
    writes=[x for x in events if x[0]==str(target.resolve())]
    if len(writes)!=1 or events[-1][0]!=str(target.resolve()) or not writes[0][1].get("runtime_adaptation"):
        raise AssertionError("Annotated canonical publication must occur exactly once and last")
    before=C.sha256(target);stamp=target.stat().st_mtime_ns
    def published_reader(_fold,_cfg,_source,_selected,_refit):
        calls["published_reader"]+=1
        current=json.loads(target.read_text())
        if current["source_identity"]!=_source or C.sha256(raw)!=current["sha256"]:
            raise AssertionError("Published reader failed its identity check")
        return current
    fake.outer_prediction=published_reader
    existing=staged_outer_prediction(fake,fold,{},source,{},refit,fixture)
    again=annotate_manifest(fake,fold,existing,runtime,proof,source,device,fixture)
    if again!=out or C.sha256(target)!=before or target.stat().st_mtime_ns!=stamp:
        raise AssertionError("Published canonical manifest changed on reentry")
    return dict(exact_resolved_path_only=True,unrelated_writes_passthrough=True,
        intentional_exception_finally_restores_writer=True,
        fresh_staged_predictor_calls=calls["fresh"],fresh_return_matches_pending=True,
        fresh_predictor_exception_finally_restores_writer=True,
        pending_resume_frozen_predictor_calls=0,pending_raw_SHA_inode_size_mtime_unchanged=True,
        annotated_report_before_canonical=True,canonical_publish_count=1,
        canonical_is_last_write=True,immutable_reentry_SHA_and_mtime=True,
        published_reader_calls=calls["published_reader"])


def cpu_check(m, runtime, cfg, proof, source, report):
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU check initialized CUDA")
    records=[]
    for fold in runtime["folds"]:
        data=C.FoldData(fold,"screen","nerf2")
        first=m.Schedule(data,cfg["position_batch"],cfg["BS_batch"],cfg["seed"]+1009*fold+97)
        second=m.Schedule(data,cfg["position_batch"],cfg["BS_batch"],cfg["seed"]+1009*fold+97)
        for i in range(32):
            key=m.FIELDS[i%8]
            a,b=first.peek(key),second.peek(key)
            if a != b:
                raise RuntimeError("Same frozen sampler produces different schedule")
            first.commit(key,*a);second.commit(key,*b)
        if first.coverage()!=second.coverage():
            raise RuntimeError("Coverage counters differ")
        records.append(dict(fold=fold,bounds=m.bounds(first,cfg),
                            identical_same_seed_sampler=True,train_labels_read=0,
                            outer_count=len(data.outer_ids)))
    publication=publication_cpu_check(report)
    d=dict(status="passed",source_identity=source,runtime=runtime,device_binding_proof=proof,
           publication_CPU_check=publication,CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),optimizer_updates=0,
           frozen_math_config_sha256=METHOD_SHA,records=records,
           active_base_worker_queues=owned_base_worker_queues(),
           duplicate_queue_guard_enforced_at_real_run=True)
    C.atomic_json(report/"cpu_equivalence.json",d)
    print(json.dumps({k:d[k]for k in ["status","CPU_only","optimizer_updates","records","active_base_worker_queues"]}),flush=True)
    return d

def device_qa(m, runtime, cfg, proof, source, report):
    gpu=C.device_guard(runtime["gpu_uuid"])
    free,_=torch.cuda.mem_get_info()
    if free < runtime["qa_min_free_GiB"]*(1<<30):
        raise RuntimeError("Lightweight zero-update QA free-memory margin unavailable")
    data=C.FoldData(0,"screen","nerf2")
    # The unchanged frozen QA selects the first two allowed FIT rows per site.
    rows=np.concatenate([data.fit_ids[data.sites(data.fit_ids)==s][:2] for s in range(2)])
    class QAOnlyCache:
        def __init__(self):
            self.rows=rows
            self.value=torch.tensor(data.read_fit(rows),device="cuda:0",dtype=torch.complex64)
        def target(self, requested, key, bs):
            _,b,u=m.parts(key)
            ix=torch.tensor([int(np.flatnonzero(self.rows==r)[0])for r in requested],device="cuda:0")
            h=self.value.index_select(0,ix)
            power=h[:,b*128:b*128+128,u*2:u*2+2].abs().square().mean((1,2,3),dtype=torch.float64)
            links=torch.tensor(bs,device="cuda:0",dtype=torch.int64)+b*128
            return h.index_select(1,links)[:,:,u*2:u*2+2],power
    model=m.build(data,cfg["seed"]);cache=QAOnlyCache()
    torch.cuda.reset_peak_memory_stats()
    checks=m.qa_gradients(model,data,cache,cfg["power_floor"])
    with torch.no_grad():
        pred=m.predict_full(model,data,rows[[0,2]],cfg["inference_BS_chunk"])
    if pred.shape!=(2,256,4,192) or not bool(torch.isfinite(pred).all()):
        raise RuntimeError("Portable full-H forward failed")
    d=dict(status="passed",source_identity=source,runtime=runtime,device=gpu,
           device_binding_proof=proof,QA=checks,full_H_shape=list(pred.shape),
           optimizer_updates=0,only_four_allowed_FIT_rows_read=True,data=data.info(),
           peak_cuda_bytes=torch.cuda.max_memory_allocated(),outer_truth_read=False)
    C.atomic_json(report/"device_equivalence_QA.json",d)
    print(json.dumps(dict(status="passed",GPU=runtime["gpu_uuid"],optimizer_updates=0,
       max_forward_relative_L2=max(x["forward_relative_L2"]for x in checks),
       max_gradient_relative_L2=max(x["gradient_relative_L2"]for x in checks),
       full_H_shape=list(pred.shape))),flush=True)
    del pred,model,cache;torch.cuda.empty_cache()
    return d

@contextmanager
def defer_manifest_publication(canonical, pending):
    """Intercept one exact resolved path and always restore the shared writer."""
    target=Path(canonical).resolve()
    original=C.atomic_json;redirected=[]
    def route(path,value):
        if Path(path).resolve()==target:
            redirected.append(PATHS.relative(path))
            return original(pending,value)
        return original(path,value)
    C.atomic_json=route
    try:
        yield redirected
    finally:
        C.atomic_json=original

def staged_outer_prediction(m,fold,cfg,source,selected,refit,report):
    canonical=m.ART/f"outer{fold}/prediction/manifest.json"
    pending=report/f"outer{fold}_pending_canonical_manifest.json"
    if canonical.exists():
        existing=m.outer_prediction(fold,cfg,source,selected,refit)
        if not existing.get("runtime_adaptation"):
            raise RuntimeError("Published portable manifest lacks its final runtime annotation")
        return existing
    if pending.exists():
        manifest=json.loads(pending.read_text())
        if manifest["source_identity"]!=source or manifest["checkpoint_sha256"]!=refit["checkpoint_sha256"]:
            raise RuntimeError("Pending publication lineage differs")
        if C.sha256(PATHS.root/manifest["path"])!=manifest["sha256"]:
            raise RuntimeError("Pending raw output changed")
        rows=np.load(canonical.parent/"row_ids.npy",allow_pickle=False)
        if rows.tolist()!=manifest["row_ids"]:
            raise RuntimeError("Pending row order changed")
    else:
        if m.C is not C:
            raise RuntimeError("Frozen and portable JSON writers do not share one module")
        with defer_manifest_publication(canonical,pending) as redirected:
            manifest=m.outer_prediction(fold,cfg,source,selected,refit)
        if canonical.exists() or len(redirected)!=1:
            raise RuntimeError("Canonical manifest was not deferred exactly once")
        if json.loads(pending.read_text())!=manifest:
            raise RuntimeError("Frozen writer return differs from pending publication")
    return manifest

def annotate_manifest(m,fold,manifest,runtime,proof,source,device,report):
    a=m.ART/f"outer{fold}/prediction";r=m.REPORT/f"outer{fold}/prediction"
    before=report/f"outer{fold}_unannotated_manifest.json"
    if not before.exists():
        C.atomic_json(before,manifest)
    out=copy.deepcopy(manifest)
    out["runtime_adaptation"]=dict(kind="scheduling_and_GPU_binding_only",
        GPU_UUID=device["GPU_UUID"],CUDA_context_UUID=device["CUDA_context_UUID"],
        portable_entry=PATHS.relative(Path(__file__)),runtime_contract=runtime,
        proof=proof,mathematics_config_sha256=METHOD_SHA,
        exact_frozen_build_train_validation_predict_functions=True)
    out["metadata"]["runtime_adaptation"]=out["runtime_adaptation"]
    if (a/"manifest.json").exists():
        existing=json.loads((a/"manifest.json").read_text())
        if existing!=out:
            raise RuntimeError("Published canonical manifest is immutable")
        return existing
    old_result=report/f"outer{fold}_unannotated_prediction_result.json"
    if not old_result.exists():
        base_result=(json.loads((r/"result.json").read_text()) if (r/"result.json").exists()
                     else dict(status="recovered_from_complete_pending_writer",manifest=manifest))
        C.atomic_json(old_result,base_result)
    C.atomic_json(r/"result.json",dict(status="complete",manifest=out))
    # Canonical publication is the final write, exactly once, after all
    # runtime annotations and reports exist. A finalizer may seal immediately.
    C.atomic_json(a/"manifest.json",out)
    return out

def execute(m, runtime, cfg, proof, source, report):
    if runtime.get("frozen") is not True:
        raise RuntimeError("Scheduling contract must be registered and frozen before execution")
    overlap=[x for x in owned_base_worker_queues()if set(x["folds"])&set(runtime["folds"])]
    if overlap:
        raise RuntimeError("Base worker still owns requested folds: "+json.dumps(overlap))
    qa=report/"device_equivalence_QA.json"
    if not qa.exists():
        raise RuntimeError("No zero-update device equivalence QA for the target GPU")
    q=json.loads(qa.read_text())
    if q["status"]!="passed" or q["source_identity"]!=source or q["device"]["GPU_UUID"]!=runtime["gpu_uuid"]:
        raise RuntimeError("Device QA does not bind the frozen portable source/runtime")
    device=C.device_guard(runtime["gpu_uuid"])
    free,_=torch.cuda.mem_get_info()
    if free < cfg["min_free_GiB"]*(1<<30):
        raise RuntimeError("Registered free-memory margin unavailable")
    C.atomic_json(report/"device.json",dict(device,free_bytes_before=free,
                                          active_base_worker_queues=owned_base_worker_queues()))
    def stopping(_signal,_frame):
        m.STOP=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    completed=[]
    import fcntl
    for fold in runtime["folds"]:
        _,fold_report=m.locations(fold)
        with (fold_report/"execution.lock").open("a+")as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            overlap=[x for x in owned_base_worker_queues()if fold in x["folds"]]
            if overlap:
                raise RuntimeError("Duplicate fold became active before claim")
            m.verify_identity(source)
            selected=m.stage(fold,"screen",cfg,source)
            refit=m.stage(fold,"refit",cfg,source,selected)
            manifest=staged_outer_prediction(m,fold,cfg,source,selected,refit,report)
            manifest=annotate_manifest(m,fold,manifest,runtime,proof,source,device,report)
            completed.append(dict(outer=fold,manifest=PATHS.relative(m.ART/f"outer{fold}/prediction/manifest.json"),
                                 channel_sha256=manifest["sha256"]))
            C.atomic_json(report/"result.json",dict(status="running",completed=completed,
                source_identity=source,runtime=runtime,device=device,optimizer_math_unchanged=True))
    C.atomic_json(report/"result.json",dict(status="complete",completed=completed,
          source_identity=source,runtime=runtime,device=device,optimizer_math_unchanged=True,
          outer_truth_read=False,formal500_scored=False))
    return completed

def finalize(m):
    """CPU-only publication after both disjoint queues have finished."""
    if torch.cuda.is_initialized():
        raise RuntimeError("Finalization must be CPU only")
    inputs=C.SimulationInputs();records=[];allrows=[]
    for fold in range(5):
        p=m.ART/f"outer{fold}/prediction/manifest.json"
        d=json.loads(p.read_text());data=C.FoldData(fold,"refit","nerf2")
        if d["method"]!="nerf2" or d["outer"]!=fold or d["row_ids"]!=data.outer_ids.tolist():
            raise RuntimeError("Canonical fold/row mismatch")
        if d["method_config_sha256"]!=METHOD_SHA or d["shape"]!=[len(data.outer_ids),256,4,192]:
            raise RuntimeError("Math config or full-H shape differs")
        if C.sha256(PATHS.root/d["path"])!=d["sha256"]:
            raise RuntimeError("Raw prediction SHA mismatch")
        for rel,sha in d["source_identity"].items():
            if C.sha256(PATHS.root/rel)!=sha:raise RuntimeError("Prediction source changed")
        allrows.extend(d["row_ids"])
        records.append(dict(outer=fold,manifest=PATHS.relative(p),manifest_sha256=C.sha256(p),
                            sha256=d["sha256"]))
    if len(allrows)!=500 or len(set(allrows))!=500 or set(allrows)!=set(inputs.query_ids):
        raise RuntimeError("Incomplete/duplicate canonical500")
    C.atomic_json(m.REPORT/"all_folds_complete.json",dict(status="complete",completed=records,
        disjoint_scheduling_only=True,mathematics_config_sha256=METHOD_SHA,
        formal500_scored=False,outer_truth_read=False))
    print(json.dumps(dict(status="complete",folds=5,rows=500,formal500_scored=False)),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runtime-config",type=Path,required=True)
    ap.add_argument("--mode",choices=["cpu-check","device-qa","run","finalize"],required=True)
    args=ap.parse_args()
    runtime,cfg,config_path=read_contract(args.runtime_config)
    if args.mode in ("cpu-check","finalize") and os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("CPU check/finalization needs empty CUDA mask")
    m,proof=isolated_import(runtime["gpu_uuid"])
    source=sources(m,config_path,args.runtime_config);report=report_dir(runtime["folds"])
    if args.mode=="cpu-check":
        return cpu_check(m,runtime,cfg,proof,source,report)
    if args.mode=="finalize":
        return finalize(m)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=runtime["gpu_uuid"]:
        raise RuntimeError("Full target UUID CUDA mask required")
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.mode=="device-qa":
        return device_qa(m,runtime,cfg,proof,source,report)
    return execute(m,runtime,cfg,proof,source,report)

if __name__=="__main__":
    try:
        main()
    except BaseException:
        import traceback
        ROOT.mkdir(parents=True,exist_ok=True)
        with (ROOT/"portable_failure.log").open("a")as f:f.write(traceback.format_exc()+"\n")
        raise


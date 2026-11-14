"""CPU-only recovery of complete NeRF2 pending prediction metadata."""
from __future__ import annotations
import argparse,copy,fcntl,json,os,subprocess,tempfile,time,traceback
from pathlib import Path
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Publication recovery requires an empty CUDA mask")
import numpy as np
from project_paths import PATHS
from round2.paper_baseline_training_v1 import common as C

R=C.REPORT/"freqminibatch_20261112"
A=C.ART/"freqminibatch_20261112/nerf2"
OUT=R/"publication_recovery"
CONFIG=R/"nerf2/method_config.json"
NERF_MODULE="round2.baseline_freqminibatch_20261112.nerf2"
HELPER_MODULE="round2.baseline_freqminibatch_20261112.recover_publication"
SOURCE=PATHS.code_root/"round2/baseline_freqminibatch_20261112/nerf2.py"
SOURCE_SHA="e919f90db32cb87774f4b4dd8eafd84db1dc29f20ee4d5ae7c09725cf41f9ab2"
CONFIG_SHA="f9b9f94f84c39170e37bae663cdbe2608f8b39db1ac78480d3f7458f03281cf6"
GPU3="GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0"
GPU1="GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009"

def normalized(value):
    return json.loads(json.dumps(value,allow_nan=False))

def read(path):
    return json.loads(path.read_text())

def workers(fold):
    found=[]
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            argv=(p/"cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0")
            if NERF_MODULE not in argv or "--mode" not in argv:continue
            if argv[argv.index("--mode")+1]!="run":continue
            if "--folds" not in argv:raise RuntimeError("Unspecified NeRF2 fold ownership")
            folds={int(x)for x in argv[argv.index("--folds")+1].split(",")}
            if fold in folds:found.append(int(p.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return found

def publish_exclusive(path,value):
    """Publish a complete JSON via same-filesystem link; never replace a target."""
    payload=(json.dumps(normalized(value),ensure_ascii=False,indent=2)+"\n").encode()
    fd,name=tempfile.mkstemp(prefix=".publication-",suffix=".json",dir=path.parent)
    temp=Path(name)
    try:
        with os.fdopen(fd,"wb")as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.link(temp,path)
    finally:
        temp.unlink(missing_ok=True)

def stat_identity(path):
    s=path.stat()
    return [s.st_ino,s.st_size,s.st_mtime_ns]

def check_frequency(refit,cfg):
    freq=refit["frequency_sampling"]
    if freq["training_K"]!=16 or len(freq["per_field"])!=8:raise RuntimeError("Frequency contract")
    for key,v in freq["per_field"].items():
        h=np.asarray(v["tone_step_counts"],dtype=np.int64)
        e=np.asarray(v["tone_complex_supervision_elements"],dtype=np.int64)
        legacy=int(v["full192_pre_migration_updates"]);new=int(v["minibatch_updates"])
        updates=refit["actual_coverage"][key]["actual_optimizer_updates"]
        if h.shape!=(192,) or e.shape!=(192,) or min(h)<=0:raise RuntimeError("Tone coverage")
        if legacy+new!=updates or int(h.sum())!=192*legacy+16*new:raise RuntimeError("Tone update mismatch")
        if not np.array_equal(e,h*cfg["position_batch"]*cfg["BS_batch"]*2):raise RuntimeError("Tone scalar exposure")
        if int(e.sum())!=v["actual_complex_supervision_elements"] or int(e.sum())!=refit["actualsupervision"][key]:
            raise RuntimeError("Actual scalar supervision mismatch")

def verify_base(fold,base,screen,refit,source,cfg):
    inputs=C.SimulationInputs();ids=inputs.roles(fold)["outer"];fit=inputs.roles(fold)["refit"]
    required={"schema":"canonical-baseline-prediction-v1","method":"nerf2","outer":fold,"role":"refit",
              "input_identity":inputs.identity,"dtype":"complex64","kind":"channel","trained":True,
              "query_truth_read":False,"phase_delay_amplitude_alignment":False,"all8fields_trained":True}
    if any(base.get(k)!=v for k,v in required.items()):raise RuntimeError("Pending identity or task mismatch")
    if base["row_ids"]!=ids.tolist() or base["row_ids_sha256"]!=C.digest_ids(ids):raise RuntimeError("Canonical rows")
    if base["shape"]!=[len(ids),256,4,192]:raise RuntimeError("Full H shape")
    if base["source_identity"]!=source or base["metadata"]["source_identity"]!=source:
        raise RuntimeError("Pending source lineage")
    if normalized(base["screen_selection"])!=normalized(screen):raise RuntimeError("Screen selection differs")
    if screen["status"]!="screen_complete" or refit["status"]!="refit_complete":raise RuntimeError("Training incomplete")
    if screen["source_identity"]!=source or refit["source_identity"]!=source:raise RuntimeError("Role source mismatch")
    if base["method_config_sha256"]!=CONFIG_SHA:raise RuntimeError("Config binding")
    for rel,digest in source.items():
        if C.sha256(PATHS.root/rel)!=digest:raise RuntimeError(("Frozen source changed",rel))
    if base["checkpoint"]!=refit["checkpoint"] or base["checkpoint_sha256"]!=refit["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint lineage")
    if C.sha256(PATHS.root/base["checkpoint"])!=base["checkpoint_sha256"]:raise RuntimeError("Checkpoint changed")
    if normalized(base["fit_data"])!=normalized(refit["data"]):raise RuntimeError("Actual refit data")
    fd=base["fit_data"]
    if (fd["fit_ids"]!=fit.tolist() or fd["fit_ids_sha256"]!=C.digest_ids(fit)
        or fd["validation_read_count"]!=0 or fd["validation_rows_actually_read"]!=[]
        or fd["query_truth_read"]is not False or fd["oracle_phase_delay_scale_correction"]is not False
        or not set(fd["fit_rows_actually_read"]).issubset(set(fd["fit_ids"]))):
        raise RuntimeError("Refit data isolation")
    if base["query_context"]["query_truth_read"]is not False:raise RuntimeError("Query truth used")
    directory=A/f"outer{fold}/prediction";raw=directory/"channels.npy"
    if PATHS.root/base["path"]!=raw:raise RuntimeError("Raw output path")
    if (directory/"channels_work.npy").exists():raise RuntimeError("Raw output not committed")
    if not np.array_equal(np.load(directory/"row_ids.npy",allow_pickle=False),ids):raise RuntimeError("Stored IDs")
    array=np.load(raw,mmap_mode="r",allow_pickle=False)
    if array.shape!=tuple(base["shape"])or array.dtype!=np.complex64:raise RuntimeError("Array header")
    for start in range(0,len(array),8):
        if not np.isfinite(array[start:start+8]).all():raise RuntimeError("Nonfinite complete H")
    if C.sha256(raw)!=base["sha256"]:raise RuntimeError("Prediction checksum")
    progress=read(R/f"nerf2/outer{fold}/prediction/prediction_work.json")
    if (progress["completed"]!=len(ids)or progress["source_identity"]!=source
        or progress["checkpoint_sha256"]!=base["checkpoint_sha256"]):raise RuntimeError("Prediction incomplete")
    check_frequency(refit,cfg)
    return raw

def recover(fold,readonly=False):
    if fold not in (2,3,4):raise ValueError("Only registered frequency folds")
    OUT.mkdir(parents=True,exist_ok=True)
    gpu=GPU1 if fold==4 else GPU3
    if C.sha256(SOURCE)!=SOURCE_SHA or C.sha256(CONFIG)!=CONFIG_SHA:raise RuntimeError("Frozen entry/config changed")
    if workers(fold):raise RuntimeError("Corresponding GPU worker is still active")
    from round2.baseline_freqminibatch_20261112 import nerf2 as N
    _,cfg,source,proof=N.configure(CONFIG,gpu)
    if N.torch.cuda.is_initialized():raise RuntimeError("CPU helper initialized CUDA")
    report=R/f"nerf2/outer{fold}/prediction"
    canonical=A/f"outer{fold}/prediction/manifest.json"
    pending=report/f"outer{fold}_pending_canonical_manifest.json"
    if not pending.is_file():raise FileNotFoundError(pending)
    screen=read(R/f"nerf2/outer{fold}/screen/result.json")
    refit=read(R/f"nerf2/outer{fold}/refit/result.json")
    base=read(pending)
    devicefile=R/f"nerf2/device_{fold}.json"
    if fold==2:devicefile=R/"nerf2/device_2_3.json"
    device=read(devicefile)
    if device["GPU_UUID"]!=gpu or device["CUDA_context_UUID"]!=gpu or device["source_identity"]!=source:
        raise RuntimeError("Actual worker device/source provenance")
    raw=verify_base(fold,base,screen,refit,source,cfg);before=stat_identity(raw)
    expected=copy.deepcopy(base)
    expected.update(frequency_sampling=refit["frequency_sampling"],actualsupervision=refit["actualsupervision"],
                    migration=refit.get("migration"))
    expected["metadata"]["adapter_version"]="freqminibatch_native192_eight_fields"
    expected["runtime_adaptation"]=dict(GPU_UUID=device["GPU_UUID"],device_binding=proof,
        training_frequency_minibatch=True,evaluation_frequency_count=192,
        unbiased_data_loss_estimator_not_numerically_identical_training=True)
    expected=normalized(expected)
    writer=read(report/"result.json")["manifest"]
    if normalized(writer)!=normalized(base):
        if not canonical.exists() or normalized(writer)!=expected:
            raise RuntimeError("Normalized pending differs from frozen writer result")
    existed=canonical.exists()
    if existed:
        if read(canonical)!=expected:raise RuntimeError("Existing canonical differs; no overwrite allowed")
    else:
        if readonly:raise RuntimeError("Read-only check expected an existing canonical")
        if workers(fold):raise RuntimeError("Worker reappeared before publication")
        original=C.atomic_json
        def guarded_writer(path,value):
            target=Path(path)
            if target.resolve()==canonical.resolve():
                if normalized(value)!=expected:raise RuntimeError("Frozen annotation differs from verified expectation")
                return publish_exclusive(canonical,value)
            if target.resolve()==(report/"result.json").resolve():
                return original(OUT/f"outer{fold}_annotated_result.json",value)
            raise RuntimeError(("Unexpected write attempted by frozen publication",str(path)))
        def forbidden(*args,**kwargs):raise RuntimeError("Model execution is forbidden in CPU metadata recovery")
        old_model_builder=N.M.build;old_predictor=N.M.outer_prediction
        try:
            C.atomic_json=guarded_writer;N.M.build=forbidden;N.M.outer_prediction=forbidden
            value=N.publish_prediction(fold,cfg,source,screen,refit,device,proof)
        finally:
            C.atomic_json=original;N.M.build=old_model_builder;N.M.outer_prediction=old_predictor
        if normalized(value)!=expected or read(canonical)!=expected:raise RuntimeError("Published content differs")
    if stat_identity(raw)!=before or C.sha256(raw)!=base["sha256"]:raise RuntimeError("Raw prediction changed")
    result=dict(status="verified_existing_readonly"if existed else"published_complete_pending",
        fold=fold,raw_sha256=base["sha256"],canonical_sha256=C.sha256(canonical),
        canonical_manifest=PATHS.relative(canonical),pending_sha256=C.sha256(pending),
        normalized_pending_writer_annotation_consistent=True,raw_inode_size_mtime_unchanged=True,
        source_sha256=C.sha256(Path(__file__)),training_updates=0,inference_calls=0,scoring_calls=0,
        GPU_workers_for_fold=[],CUDA_initialized=N.torch.cuda.is_initialized())
    if not readonly:C.atomic_json(OUT/f"outer{fold}_recovery.json",result)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["qa","recover","watch"],required=True)
    parser.add_argument("--fold",type=int,choices=[2,3,4])
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=="qa":
        sample={"tuple":(1,2),"value":3}
        assert sample!=normalized(sample) and normalized(sample)==normalized(normalized(sample))
        wrong=copy.deepcopy(sample);wrong["value"]=4
        assert normalized(sample)!=normalized(wrong)
        checked=recover(2,readonly=True)
        C.atomic_json(OUT/"QA.json",dict(status="passed",tuple_list_roundtrip_only=True,
            substantive_mismatch_rejected=True,existing_outer2_readonly=checked))
        print(json.dumps(checked),flush=True);return
    if args.mode=="recover":
        if args.fold is None:raise ValueError("A registered fold is required")
        print(json.dumps(recover(args.fold)),flush=True);return
    with (OUT/"watcher.lock").open("a+")as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        qa=read(OUT/"QA.json")
        accepted_sha=qa["existing_outer2_readonly"]["source_sha256"]
        if qa["status"]!="passed" or C.sha256(Path(__file__))!=accepted_sha:
            raise RuntimeError("No matching accepted CPU publication QA")
        deadline=time.monotonic()+24*3600;complete=set();previous=None
        while len(complete)<2:
            if C.sha256(Path(__file__))!=accepted_sha:raise RuntimeError("CPU helper source changed while waiting")
            states={}
            for fold in (3,4):
                if fold in complete:continue
                pending=R/f"nerf2/outer{fold}/prediction/outer{fold}_pending_canonical_manifest.json"
                canonical=A/f"outer{fold}/prediction/manifest.json"
                active=workers(fold);states[str(fold)]=dict(worker_pids=active,pending=pending.exists(),canonical=canonical.exists())
                if not active and (pending.exists()or canonical.exists()):
                    command=[str(PATHS.code_root/"tools/run_in_env.sh"),"python","-u","-m",HELPER_MODULE,
                             "--mode","recover","--fold",str(fold)]
                    subprocess.run(command,cwd=PATHS.root,env=os.environ.copy(),check=True)
                    complete.add(fold)
            if states!=previous:
                C.atomic_json(OUT/"waiting.json",dict(folds=states,completed=sorted(complete)))
                print(json.dumps(dict(event="waiting_for_complete_pending",folds=states,completed=sorted(complete))),flush=True)
                previous=states
            if len(complete)==2:
                C.atomic_json(OUT/"complete.json",dict(status="complete",folds=[3,4]));return
            if time.monotonic()>deadline:raise TimeoutError("Complete pending metadata did not appear")
            time.sleep(30)

if __name__=="__main__":
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True)
        with(OUT/"failure.log").open("a")as stream:stream.write(traceback.format_exc()+"\n")
        raise

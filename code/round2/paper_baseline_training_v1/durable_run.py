"""Durable execution of the already frozen baseline jobs; no model changes."""
from __future__ import annotations
import concurrent.futures,json,os,subprocess,threading,time,shutil,hashlib
from pathlib import Path
ROOT=Path(os.environ["HW_PROJECT_ROOT"]).resolve()
REPORT=ROOT/"analysis/results/round2_paper_four_baselines_v1"
LOCK=threading.Lock();SCORE_LOCK=threading.Lock();STATE={"status":"running","jobs":{}}
GPU={"newrf":"GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009",
     "nerf2":"GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0"}
CFG={"newrf":"analysis/results/round2_paper_four_baselines_v1/newrf/v4/formal_config_v4.json",
     "nerf2":"analysis/results/round2_paper_four_baselines_v1/nerf2/v3/formal_config.json"}
MODULE={"newrf":"round2.paper_baseline_training_v1.newrf_train_v4",
        "nerf2":"round2.paper_baseline_training_v1.nerf2_train_main_v3"}
LOG={"newrf":REPORT/"newrf/v4/formal_stdout.log","nerf2":REPORT/"nerf2/v3/main.log"}
def write_state():
    p=REPORT/"execution_supervisor.json";tmp=p.with_suffix(".json.partial")
    tmp.write_text(json.dumps(STATE,ensure_ascii=False,indent=2)+"\n");tmp.replace(p)
def run_command(label,module,args,gpu,log):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE="1",HW_SKIP_ENV_CHECK="1",
                                    OMP_NUM_THREADS="4" if gpu else "1",MKL_NUM_THREADS="4" if gpu else "1",
                                    OPENBLAS_NUM_THREADS="4" if gpu else "1")
    command=[str(ROOT/"code/tools/run_in_env.sh"),"python","-u","-m",module,*args]
    with log.open("a") as stream:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                              stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        with LOCK:
            STATE["jobs"][label]=dict(status="running",PID=proc.pid,GPU_UUID=gpu,command=command,started_unix=time.time())
            write_state()
        rc=proc.wait()
    with LOCK:
        STATE["jobs"][label].update(status="complete" if rc==0 else "failed",returncode=rc,ended_unix=time.time())
        write_state()
    if rc:raise RuntimeError(label+" exited "+str(rc)+"; see "+str(log.relative_to(ROOT)))
def score():
    with SCORE_LOCK:
        run_command("CPU_score","round2.paper_baseline_training_v1.final_score",[],"",REPORT/"final_score_pipeline.log")
def worker(method):
    for outer in range(5):
        if method=="newrf":
            # Weights-complete is an immutable boundary. Never resave a checkpoint
            # while resuming its partially generated predictions.
            stages=["train","predict"]
        else:stages=["all"]
        for stage in stages:
            args=["--config",CFG[method],"--folds",str(outer),"--stage",stage]
            run_command(method+"_outer"+str(outer)+"_"+stage,MODULE[method],args,GPU[method],LOG[method])
        score()
    return method
def final_record():
    d=json.loads((REPORT/"comparison.json").read_text())
    if not d["all_four_complete"]:raise RuntimeError("All4 canonical500 predictions not complete")
    with (ROOT/"experiment_records/EXPERIMENT_RECORD.md").open("a") as f:
        f.write("\n### Four-baseline pipeline completion\n\n")
        for row in d["rows"]:
            f.write("- "+row["method"]+": "+json.dumps(row["metrics"],ensure_ascii=False)+"\n")
        f.write("- All four task-adapted methods completed canonical500 applicable metrics. Independent saved-stat aggregation and prediction-file hashes verified. V16 archival row unchanged; no retraining or test-time phase/delay/amplitude correction. Single seed; original-paper exact reproduction not claimed.\n")
    p=REPORT/"execution_revision.json";r=json.loads(p.read_text())
    r["completed_methods"]={x["key"]:x["metrics"] for x in d["rows"] if x["key"]!="v16"}
    r["active_jobs"]={};r["all_four_complete"]=True
    p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(worker,m):m for m in ["nerf2","newrf"]}
        for future in concurrent.futures.as_completed(jobs):
            try:future.result()
            except Exception as e:errors.append(str(e))
    if not errors:
        try:
            score()
            run_command("CPU_final_verify","round2.paper_baseline_training_v1.verify_finished",
                        ["--require-all","--full-hash"],"",REPORT/"final_verification.log")
            final_record()
        except Exception as e:errors.append(str(e))
    with LOCK:
        STATE.update(status="complete" if not errors else "needs_attention",errors=errors,finished_unix=time.time())
        write_state()
    if errors:raise RuntimeError(errors)
if __name__=="__main__":main()


"""Queue remaining folds on one allowed GPU with a measured memory reserve."""
from __future__ import annotations
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from project_paths import PATHS
R=PATHS.analysis/"results/round2_paper_four_baselines_v1/wrfgs_full_h_v1_20261110"
UUIDS=("GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009",
       "GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0",
       "GPU-70b6fceb-c7c4-9467-3c81-097c9284c160")
def write(value):
    p=R/"queue_state.json";tmp=p.with_suffix(".partial")
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def done(fold):return (R/f"outer{fold}/refit/complete.json").is_file()
def alive(pid):
    p=Path(f"/proc/{pid}/cmdline")
    return p.exists() and b"round2.wrfgs_full_h_v1.run" in p.read_bytes()
def main():
    active={}
    children={};pending=[f for f in range(5) if not done(f)];completed=[f for f in range(5) if done(f)]
    while pending or active:
        for fold,job in list(active.items()):
            if done(fold):
                child=children.get(fold)
                if child is not None and child.poll() is None:continue
                completed.append(fold);del active[fold]
                print(json.dumps(dict(event="fold_complete",fold=fold)),flush=True)
            elif not alive(job["pid"]):
                write(dict(status="failed",failed_fold=fold,pending=pending,active=active,completed=completed))
                raise RuntimeError(f"Fold {fold} worker ended without completion; see its log")
        result=subprocess.run(["nvidia-smi","--query-gpu=uuid,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
        used={v["uuid"] for v in active.values()}
        for line in result.stdout.splitlines():
            uuid,memory,util=[x.strip() for x in line.split(",")]
            if not pending or active or uuid not in UUIDS or uuid in used:continue
            if int(memory)<48*1024:continue
            fold=pending.pop(0)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1")
            log=R/f"train_outer{fold}.log"
            with log.open("a") as f:
                p=subprocess.Popen([sys.executable,"-m","round2.wrfgs_full_h_v1.run",
                    "--stage","train","--folds",str(fold),"--device-uuid",uuid],
                    cwd=PATHS.root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            children[fold]=p;active[fold]=dict(pid=p.pid,uuid=uuid,adopted=False);used.add(uuid)
            print(json.dumps(dict(event="launched",fold=fold,pid=p.pid,uuid=uuid)),flush=True)
        write(dict(status="running" if pending or active else "complete",pending=pending,
            active=active,completed=sorted(completed),poll_seconds=30,
            maximum_parallel_folds=1,minimum_free_mib_before_launch=48*1024,
            launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        if pending or active:time.sleep(30)
    print(json.dumps(dict(event="all_folds_complete",folds=sorted(completed))),flush=True)
if __name__=="__main__":main()

"""Wait for all planned predictions, then score and verify once. CPU only."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from project_paths import PATHS
from round2.baseline_formal_20261111 import score as S
from round2.baseline_formal_20261111 import verify as V
from round2.baseline_formal_20261111 import score_ready as Ready
from round2.paper_baseline_training_v1 import common as C

R=S.R
def event(**items):
    print(json.dumps({"utc":datetime.now(timezone.utc).isoformat(),**items},ensure_ascii=False),flush=True)

def collect_training_reports():
    records={}
    for method in S.METHODS:
        records[method]=[]
        for fold in range(5):
            for role in ("screen","refit"):
                p=R/method/f"outer{fold}"/role/"result.json"
                if not p.is_file():raise FileNotFoundError(p)
                d=json.loads(p.read_text())
                def one(keys):
                    return next((d[k] for k in keys if k in d),None)
                records[method].append({"fold":fold,"role":role,"report":PATHS.relative(p),"sha256":C.sha256(p),
                    "status":d.get("status"),
                    "wall_seconds":one(("wall_seconds","elapsed_seconds","active_wall_seconds","seconds")),
                    "steps":one(("completed_steps","actual_updates","total_updates","updates","optimizer_steps","total_optimizer_updates")),
                    "selected_steps":one(("selected_steps","selected_updates","selected_optimizer_updates")),
                    "peak_torch_bytes":one(("peak_torch_allocated_bytes","peak_cuda_bytes","peak_cuda_allocated_bytes")),
                    "device":one(("GPU","device","gpu"))})
    return records

def write_completion():
    v=R/"evaluation/completed_verification.json"
    evidence=json.loads(v.read_text())
    if evidence["status"]!="passed":raise RuntimeError("Final verification did not pass")
    c=json.loads((R/"evaluation/comparison.json").read_text())
    train=collect_training_reports()
    C.atomic_json(R/"evaluation/training_report_index.json",train)
    dest=R/"RESULTS.md"
    text=(R/"evaluation/RESULTS.md").read_text()
    text+="\n\n训练及来源索引：evaluation/training_report_index.json。独立CPU复算与保护核验：evaluation/completed_verification.json。\n"
    text+="各方法的原始五折预测及模型保留在对应artifacts目录。修复前成绩另保留，不用新表覆盖旧文件。\n"
    tmp=dest.with_suffix(".md.partial");tmp.write_text(text);tmp.replace(dest)
    summary={"status":"scored_and_verified","result_table":PATHS.relative(dest),"table_sha256":C.sha256(dest),
             "verification":PATHS.relative(v),"verification_sha256":C.sha256(v),
             "formal_models_complete":list(S.METHODS),"retained_models":["V16","WRF-GS+ full H"],
             "all_new_methods_have_500":True,"completed_utc":datetime.now(timezone.utc).isoformat(),
             "rows":c["rows"]}
    C.atomic_json(R/"evaluation/completion.json",summary)
    record=PATHS.root/"experiment_records/EXPERIMENT_RECORD.md"
    marker="- 833正式500点统一评分与独立验证完成"
    if marker not in record.read_text():
        body=marker+"："+ "; ".join(f"{x['method']} PAS={x['metrics']['pas']:.9f}, PDP={x['metrics']['pdp']:.9f}, NMSE={x['metrics']['nmse']:.9f}, score={x['metrics']['score']:.9f}" for x in c["rows"])
        with record.open("a") as f:f.write("\n"+body+"。五fold全完整H、31零点、无真值校正；V16和WRF-GS+沿用保留成绩；训练来源、预算、资源和限制见formal_20261111各组report及evaluation索引。旧文件保护和CPU独立复算已通过。\n")
    p=R/"CHECKLIST.md"
    t=p.read_text()
    for item in ["三方法正式配置、输入和有限吞吐检查","冻结预算和停止规则，登记实验合同",
                 "NeRF²五组正式训练、重新拟合和完整H预测","NeWRF五组正式训练、重新拟合和完整H预测",
                 "nGRF五组正式训练、重新拟合和完整H预测","完整500点统一评分和独立核验"]:
        t=t.replace("- [ ] "+item,"- [x] "+item)
    p.write_text(t)
    event(event="formal500_scored_and_verified",metrics={x["key"]:x["metrics"] for x in c["rows"]})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--wall-hours",type=float,default=120.)
    a=ap.parse_args();deadline=time.monotonic()+a.wall_hours*3600
    out=R/"evaluation";out.mkdir(parents=True,exist_ok=True)
    with (out/"finalization.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        last=None
        while True:
            counts={m:sum((S.A/m/f"outer{k}"/"prediction/manifest.json").is_file() for k in range(5)) for m in S.METHODS}
            if counts!=last:event(event="waiting_for_sealed_predictions",complete_folds=counts);last=counts
            eligible=[method for method,count in counts.items()
                      if count==5 and not (out/method/"independent_verification.json").is_file()]
            if eligible:
                table=Ready.main(eligible)
                event(event="completed_methods_scored",methods=eligible,
                      all_methods_complete=table["all_new_methods_formal500"])
            if all(n==5 for n in counts.values()):break
            if time.monotonic()>=deadline:raise TimeoutError("Formal predictions incomplete within coordinator wall window")
            time.sleep(30)
        S.main();V.main();write_completion()
if __name__=="__main__":
    try:main()
    except BaseException:
        (R/"evaluation").mkdir(exist_ok=True)
        with (R/"evaluation/finalization_failure.log").open("a") as f:f.write(traceback.format_exc()+"\n")
        raise


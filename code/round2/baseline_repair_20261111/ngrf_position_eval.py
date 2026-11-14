"""Read-only fixed-endpoint development validation after nGRF fixed per-position loss comparison."""
from __future__ import annotations
import argparse,json,subprocess,traceback
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from common.official_metrics import point_metric_statistics_torch,aggregate_statistics
from round2.paper_baseline_training_v1 import common as C
from round2.baseline_repair_20261111 import ngrf_repair as R
from round2.baseline_repair_20261111 import ngrf_position_loss as Q

OUT=Q.OUT/"fixed_endpoint_validation"
def restore(model,state):
    model.load_state_dict(state["model"])
    modules=dict(model.named_modules())
    for name,p in state["actual_dropout"].items():
        assert isinstance(modules[name],torch.nn.modules.dropout._DropoutNd)
        modules[name].p=float(p)
    assert Q.actual_dropout(model)==state["actual_dropout"]
    model.eval()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--gpu-uuid",required=True);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    results_file=Q.OUT/"results.json";frozen_sha=C.sha256(results_file)
    frozen=json.loads(results_file.read_text());assert frozen["status"]=="complete"
    for rel,h in frozen["source_identity"].items():assert C.sha256(PATHS.root/rel)==h,rel
    free=int(subprocess.check_output(["nvidia-smi","-i",args.gpu_uuid,"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
    assert free>=24*1024
    gpu=C.device_guard(args.gpu_uuid);device=torch.device("cuda:0");torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=C.FoldData(0,"screen","ngrf");rows=data.probe_ids
    assert len(rows)==16 and not set(rows).intersection(data.outer_ids)
    truth=data.read_validation(rows,normalized=False)
    records=[]
    for endpoint in frozen["variants"]:
        checkpoint=PATHS.root/endpoint["checkpoint"]
        assert C.sha256(checkpoint)==endpoint["checkpoint_sha256"]
        state=torch.load(checkpoint,map_location=device,weights_only=True)
        model=R.model_build(data,device);restore(model,state)
        before=R.model_hash(model)
        pred=R.predict(model,data,rows,device)
        raw=(pred*data.scale).astype(np.complex64)
        assert raw.shape==(16,256,4,192) and np.isfinite(raw).all()
        after=R.model_hash(model);assert before==after
        scores=aggregate_statistics(point_metric_statistics_torch(
            torch.tensor(raw,device=device),torch.tensor(truth,device=device)))
        dest=Q.ART/endpoint["branch"]/"validation_raw_H.npy"
        C.atomic_npy(dest,raw);C.atomic_npy(dest.parent/"validation_row_ids.npy",rows)
        records.append(dict(branch=endpoint["branch"],checkpoint=endpoint["checkpoint"],
            checkpoint_sha256=endpoint["checkpoint_sha256"],actual_dropout=state["actual_dropout"],
            native_original_metadata=state["original_native_metadata"],
            model_state_hash_before=before,model_state_hash_after=after,metrics=scores,
            diagnostics=R.stats(pred,truth/data.scale,data.sites(rows),data.scale,rows),
            raw_prediction=PATHS.relative(dest),raw_prediction_sha256=C.sha256(dest)))
        del model;torch.cuda.empty_cache()
    assert C.sha256(results_file)==frozen_sha
    report=dict(status="complete",source_sha256=C.sha256(Path(__file__)),
        frozen_FIT_result=PATHS.relative(results_file),frozen_FIT_result_sha256=frozen_sha,
        row_ids=rows.tolist(),role="original16-development-validation; endpoints fixed by FIT contract",
        weights_or_checkpoint_selection=False,returned_to_training=False,phase_delay_amplitude_alignment=False,
        formal500_test=False,outer_truth_read=False,variants=records,device=gpu,data=data.info())
    C.atomic_json(OUT/"results.json",report)
    print(json.dumps(dict(status="complete",metrics={x["branch"]:x["metrics"] for x in records})),flush=True)
if __name__=="__main__":
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/"failure.log").open("a") as f:f.write(traceback.format_exc()+"\n")
        raise

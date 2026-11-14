"""Bounded NeRF2 exact-transport/GPU precision cost diagnostic, FIT geometry only."""
import argparse,dataclasses,json,math,time
import numpy as np
import torch
from round2.paper_baseline_training_v1 import field_runner as runner
from round2.paper_baseline_training_v1.field_fast import enable_shared_medium
from round2.paper_baseline_training_v1.common import atomic_json,atomic_npy,output_paths


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--far",type=float);ap.add_argument("--test-tf32",action="store_true")
    args=ap.parse_args();gpu=runner.device_guard(runner.GPU_UUID)
    runner.seed_all(42);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=runner.FoldData(0,"screen","nerf2")
    model=runner.build_model("nerf2",data,medium_rays=324,tx_chunk=64,conditioned=True).eval()
    checkpoint=output_paths("nerf2",0,"pilot_conditioned_v2")["artifact"]/"pilot.pt"
    if checkpoint.exists():model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["model"])
    xyz=data.positions(data.fit_ids);sites=data.sites(data.fit_ids)
    bs=np.asarray(data.geometry.bs_positions_m,dtype=np.float64);distance=np.linalg.norm(xyz-bs[sites],axis=1)
    far=args.far if args.far is not None else math.ceil(float(distance.max())/10.)*10.
    for core in model.fields.values():core.config=dataclasses.replace(core.config,far_m=far)
    selected=int(np.argmin(np.linalg.norm(xyz-xyz.mean(0),axis=1)))
    row=np.array([data.fit_ids[selected]],np.int64)
    p=torch.tensor(data.positions(row),device="cuda:0",dtype=torch.float64)
    s=torch.tensor(data.sites(row),device="cuda:0",dtype=torch.int64)
    f=torch.tensor(data.frequencies(),device="cuda:0",dtype=torch.float64)
    def timed(freq):
        torch.cuda.synchronize();start=time.monotonic()
        with torch.no_grad():value=model(p,s,freq)*model.log_output_gain.exp()
        torch.cuda.synchronize();return value,time.monotonic()-start
    reference,_=timed(f[:8])
    enable_shared_medium(model,medium_ray_chunk=324,tx_chunk=64,factor_transport=True)
    shared,part_seconds=timed(f[:8])
    err=float(torch.linalg.vector_norm(shared-reference)/torch.linalg.vector_norm(reference).clamp_min(1e-30))
    torch.cuda.reset_peak_memory_stats();value,seconds=timed(f)
    paths=output_paths("nerf2",0,"cost_v2")
    atomic_npy(paths["artifact"]/"prediction_fit_fp32.npy",value.cpu().numpy())
    result=dict(status="complete",kind="FIT-only numerical/performance diagnostic",row_ids=row.tolist(),
      far_m=far,max_FIT_BS_distance_m=float(distance.max()),full_H_shape=list(value.shape),
      full_H_seconds=seconds,frequency8_seconds=part_seconds,factor_transport_relative_L2=err,
      peak_cuda_bytes=int(torch.cuda.max_memory_allocated()),GPU=gpu,actual_data_labels_read=False,weights_previously_fit=checkpoint.exists(),
      sample_count=32,ray_count=324,model_depth=8,model_width=256,TF32=False,
      source_identity=runner.source_identity(),checkpoint_loaded=checkpoint.exists())
    if args.test_tf32:
        torch.backends.cuda.matmul.allow_tf32=True
        low,low_seconds=timed(f)
        result["tf32"]=dict(full_H_seconds=low_seconds,
           relative_L2=float(torch.linalg.vector_norm(low-value)/torch.linalg.vector_norm(value).clamp_min(1e-30)),
           maximum_absolute=float((low-value).abs().max()),
           geometry_propagation_precision="FP64 unchanged",accepted=False)
        atomic_npy(paths["artifact"]/"prediction_fit_tf32.npy",low.cpu().numpy())
        torch.backends.cuda.matmul.allow_tf32=False
    atomic_json(paths["report"]/"result.json",result);print(json.dumps(result),flush=True)

if __name__=="__main__":main()

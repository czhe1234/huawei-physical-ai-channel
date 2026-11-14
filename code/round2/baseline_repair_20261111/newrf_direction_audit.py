"""Read-only matched-device direction and physical-axis audit for the NeWRF route."""
from __future__ import annotations
import argparse,json
import numpy as np
import torch
from .newrf_repair import R,A,CONFIG,BS,FREQ,build,gpu_setup,identity
from round2.paper_baseline_training_v1.common import FoldData,atomic_json,atomic_npy,sha256,digest_ids
from project_paths import PATHS
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument("--gpu-uuid",required=True);args=p.parse_args()
    dest=R/"route/direction_audit.json"
    if dest.exists():raise FileExistsError("Keep prior direction audit")
    device=gpu_setup(args.gpu_uuid)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=json.loads(CONFIG.read_text());rows=np.array(cfg["newrf"]["route_rows"],np.int64)
    d=FoldData(0,"screen","newrf")
    if not set(rows).issubset(set(d.fit_ids)):raise PermissionError("FIT only")
    bank,cache,model=build(d);source=identity();source[PATHS.relative(Path(__file__))]=sha256(Path(__file__))
    angular=[];all_query=[];ue_coordinates=[]
    with torch.no_grad():
        for row in rows:
            pos=torch.tensor(d.positions([row]),device="cuda:0",dtype=torch.float64)
            site=torch.tensor(d.sites([row]),device="cuda:0",dtype=torch.long)
            origins=model.spec.ue_world_positions(pos,site);tx=model.bs_ports[site]
            opened=list(bank.opened_ids)
            q=model.ray_provider.directions(origins,site,tx)[0,0,:,0]
            if opened!=bank.opened_ids:raise RuntimeError("Query geometry payload read")
            t=cache.scalar(model,int(row),0,torch.tensor(BS,device="cuda:0"),origins[0],site[0],tx[0])
            if not isinstance(t,torch.Tensor):t=torch.stack(t)
            if t.ndim!=3 or q.shape!=(128,64,3):raise ValueError("Directions layout")
            angle=torch.rad2deg(torch.acos(torch.clamp(torch.einsum("brd,bqd->brq",t,q),-1.,1.))).min(-1).values
            perbs=[]
            for j in range(128):
                a=angle[j].cpu().numpy()
                perbs.append(dict(BS=j,nearest_angle_min_deg=float(a.min()),nearest_angle_median_deg=float(np.median(a)),
                    nearest_angle_max_deg=float(a.max()),training_rays_matched_under_1deg=int((a<1).sum()),
                    training_rays_matched_under_point1deg=int((a<.1).sum())))
            atomic_npy(A/f"route/FIT_directions_row{int(row)}.npy",t.cpu().numpy())
            atomic_npy(A/f"route/nearest_angles_row{int(row)}.npy",angle.cpu().numpy())
            all_query.append(q.cpu().numpy());ue_coordinates.append(origins[0,0].cpu().numpy())
            angular.append(dict(row=int(row),training_rays=t.shape[1],query_rays=64,
                median_nearest_angle_all_BS_deg=float(angle.median()),max_nearest_angle_all_BS_deg=float(angle.max()),
                BS30=perbs[30],per_BS=perbs))
    atomic_npy(A/"route/query_directions.npy",np.stack(all_query))
    axes=dict(rows=rows.tolist(),positions_m=d.positions(rows).tolist(),sites=d.sites(rows).tolist(),
        field=0,polarization=0,UE_port=0,UE_world_positions_m=np.stack(ue_coordinates).tolist(),
        BS_spatial_ids=BS.tolist(),BS_world_positions_m=model.bs_ports[0,0].cpu().numpy().tolist(),
        frequency_ids=FREQ.tolist(),frequencies_hz=d.frequencies(FREQ).tolist(),dtype="complex64",raw_H=True,
        prediction_shape=[2,128,192])
    atomic_json(R/"route/axes.json",axes)
    result=dict(status="completed",device=device,source_identity=source,angular_comparison=angular,
        query_input_available_at_unseen_positions=True,query_geometry_payload_reads=0,
        source_geometry_count=len(bank.source_fit_ids),source_geometry_ids_sha256=digest_ids(bank.source_fit_ids),
        directions_frequency_independent=True,CSI_labels_read=0,optimizer_updates=0,formal_test=False,
        direction_sha256={p.name:sha256(p) for p in (A/"route").glob("*directions*.npy")})
    atomic_json(dest,result)
    print(json.dumps(dict(status="completed",angular=[{k:x[k] for k in ["row","training_rays","query_rays","median_nearest_angle_all_BS_deg","max_nearest_angle_all_BS_deg","BS30"]} for x in angular])),flush=True)
if __name__=="__main__":main()

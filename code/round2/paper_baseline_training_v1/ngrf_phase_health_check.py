"""Read-only signed and elementwise FIT phase diagnostics; never aligns phase."""
import json,time
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from round2.paper_baseline_training_v1.common import FoldData,atomic_json,sha256
from round2.paper_baseline_training_v1.native_ngrf import NativeNGRF


def main():
    torch.set_num_threads(4);data=FoldData(0,"screen","ngrf")
    base=PATHS.artifacts/"round2/paper_four_baselines_v1/ngrf/v2/outer0"
    model=NativeNGRF(data.geometry,data.positions(data.fit_ids),data.sites(data.fit_ids)).eval()
    results=[]
    for role in ("health_pure_magnitude","health"):
        p=base/role/"health.pt";ck=torch.load(p,map_location="cpu",weights_only=False)
        ids=np.array(ck["info"]["FIT_ids"],np.int64);model.load_state_dict(ck["model"])
        with torch.no_grad():
            pred=model(torch.tensor(data.positions(ids),dtype=torch.float64),torch.tensor(data.sites(ids),dtype=torch.int64),torch.tensor(data.frequencies(),dtype=torch.float64)).to(torch.complex128)
            truth=torch.tensor(data.read_fit(ids),dtype=torch.complex128)
            pm=pred.abs();tm=truth.abs();valid=(tm>0)&(pm>0);target_nonzero=tm>0
            cosine=(pred.conj()*truth).real[valid]/(pm[valid]*tm[valid])
            weight=tm[valid].square()
            ep=pm.square().sum();et=tm.square().sum()
            results.append(dict(role=role,checkpoint=str(p.relative_to(PATHS.root)),checkpoint_sha256=sha256(p),
              signed_complex_correlation_real=float((pred.conj()*truth).real.sum()/torch.sqrt(ep*et).clamp_min(1e-30)),
              magnitude_complex_correlation_not_used=True,
              nonzero_target_phase_cosine_mean=float(cosine.mean()),
              nonzero_target_energy_weighted_phase_cosine=float((cosine*weight).sum()/weight.sum()),
              nonzero_target_entries=int(target_nonzero.sum()),phase_defined_entries=int(valid.sum()),
              phase_defined_target_energy_fraction=float(weight.sum()/et),prediction_zero_phase="undefined; excluded only from phase statistic and coverage reported",
              complex_nmse=float((pred-truth).abs().square().sum()/et),prediction_to_truth_energy=float(ep/et),phase_alignment=False))
    out=dict(kind="FIT-only read-only phase supervision diagnostic",outer_truth_read=False,training=False,results=results)
    p=PATHS.analysis/"results/round2_paper_four_baselines_v1/ngrf/v2/phase_health_check.json"
    atomic_json(p,out);print(json.dumps(out),flush=True)

if __name__=="__main__":main()

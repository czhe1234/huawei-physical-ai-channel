"""Independent trained-model parity and evaluation checks on allowed rows only."""
from __future__ import annotations
import time,json,traceback
import numpy as np
import torch
from .common import *
from .wrf_batched import BatchedWRF
from .train_baselines import evaluate
from measured.hall_adapters.metrics import codebook
def run():
    start=time.monotonic();seed();dev=DevelopmentData();bank=dev.fit();d=device()
    records=[]
    for count in [64,128]:
        name="wrf_g%d"%count;m=BatchedWRF(bank.position,bank.spec,count).to(d)
        checkpoint=torch.load(WORK/"models"/name/"best.pt",map_location=d,weights_only=False);m.load_state_dict(checkpoint["state_dict"]);m.eval()
        x=torch.tensor(np.array(bank.position[:2]),dtype=torch.float32,device=d)
        bat=m(x);reference=torch.stack([f(x) for f in m.fields],dim=1)
        abs_error=float((bat-reference).abs().max())
        relative=float(((bat-reference).abs()/reference.abs().clamp_min(1e-7)).max())
        if not torch.allclose(bat,reference,rtol=5e-4,atol=1e-6):raise ValueError("Trained WRF reference/optimized forward mismatch")
        parameters=[m.fields[0].xyz,m.fields[0].log_scale,m.fields[0].opacity_logit,m.fields[0].sh,m.fields[0].deformation.gaussian_signal.weight]
        gb=torch.autograd.grad(bat.square().mean(),parameters,retain_graph=False)
        gr=torch.autograd.grad(reference.square().mean(),parameters)
        ge=[float((a-b).abs().max()/b.abs().max().clamp_min(1e-8)) for a,b in zip(gb,gr)]
        if max(ge)>2e-3:raise ValueError("Trained WRF gradient parity mismatch")
        # Predictions are regenerated without a label argument; saved powers must agree.
        pf=evaluate(m,torch.tensor(np.array(bank.position),dtype=torch.float32,device=d))
        pv=evaluate(m,torch.tensor(np.array(dev.query("validation").position),dtype=torch.float32,device=d))
        with np.load(WORK/"models"/name/"best_power.npz") as z:
            pe=max(float(np.max(np.abs(pf-z["fit"]))),float(np.max(np.abs(pv-z["validation"]))))
        if pe>1e-5:raise ValueError("WRF checkpoint prediction mismatch")
        records.append(dict(name=name,author_dense_reference_max_absolute_error=abs_error,
            maximum_relative_pixel_integrated_power_error=relative,selected_group_relative_gradient_errors=ge,
            regenerated_prediction_max_error=pe,original_CUDA_kernel_parity=False,author_original_training_replication=False))
        del m,bat,reference,gb,gr;torch.cuda.empty_cache()
    save_json(REPORT/"trained_wrf_parity.json",dict(passed=True,records=records,seconds=time.monotonic()-start,formal_test_access=False,north_access=False))
    dev.save_access("trained_parity");print("TRAINED_PARITY",json.dumps(records),flush=True)
if __name__=="__main__":
    try:run()
    except BaseException:
        save_json(REPORT/"trained_parity_failure.json",dict(traceback=traceback.format_exc()));raise

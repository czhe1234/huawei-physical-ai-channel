"""Bounded FIT-only check after correcting native nGRF logit regularization."""
from pathlib import Path
import json,time
import numpy as np
import torch
from round2.paper_baseline_training_v1 import field_runner as runner
from round2.paper_baseline_training_v1.common import atomic_json,atomic_torch_save,output_paths


def main():
    runner.device_guard(runner.GPU_UUID);runner.seed_all(42);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=runner.FoldData(0,"screen","ngrf")
    model=runner.build_model("ngrf",data);opt=runner.optimizer_for("ngrf",model)
    cache=runner.FitGPUCache(data,on_gpu=True);rng=np.random.default_rng(42)
    paths=output_paths("ngrf",0,"pilot_logit_l1_v2")
    rows=np.concatenate([x[np.linspace(0,len(x)-1,8,dtype=int)] for x in
          [data.fit_ids[data.sites(data.fit_ids)==s] for s in (0,1)]])
    ff=np.array([0,32,64,96,128,160,191],np.int64)
    p=torch.tensor(data.positions(rows),device="cuda:0",dtype=torch.float64)
    s=torch.tensor(data.sites(rows),device="cuda:0",dtype=torch.int64)
    f=torch.tensor(data.frequencies(ff),device="cuda:0",dtype=torch.float64)
    target=cache.read(rows,ff)
    def monitor(step):
        model.eval()
        with torch.no_grad():
            value=model(p,s,f)
            return dict(step=step,mse=float((value-target).abs().square().mean()),
              zero_mse=float(target.abs().square().mean()),prediction_energy=float(value.abs().square().mean()),
              activation={site:float(field.native.get_attributes_and_activation(field.fixed_tx.float())[1].mean())
                 for site,field in model.fields.items()},
              logit_l1={site:float(field.activation_penalty()) for site,field in model.fields.items()})
    records=[monitor(0)];started=time.monotonic()
    for step in range(1,1001):
        runner.apply_learning_rate("ngrf",opt,step,15000)
        model.train();opt.zero_grad(set_to_none=True)
        loss,detail=runner.full_training_step("ngrf",model,data,cache,rng,points=32,tones=8)
        if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite health loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
        if step%250==0:
            event={**monitor(step),"training_batch":detail,"seconds":time.monotonic()-started}
            records.append(event);print(json.dumps(event),flush=True)
    result=dict(status="complete",kind="FIT-only implementation health",trained=True,
      updates=1000,source_identity=runner.source_identity(),records=records,
      geometry_fixture_ids=rows.tolist(),frequency_ids=ff.tolist(),outer_truth_used=False,
      validation_used=False,seconds=time.monotonic()-started,config=model.metadata(),GPU_UUID=runner.GPU_UUID)
    atomic_json(paths["report"]/"result.json",result)
    atomic_torch_save(paths["artifact"]/"health.pt",dict(model=model.state_dict(),info=result))
    print(json.dumps(result),flush=True)

if __name__=="__main__":main()

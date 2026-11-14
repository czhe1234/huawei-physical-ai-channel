"""One targeted power-first foundation branch with the same geometric core."""
from __future__ import annotations
import numpy as np
import torch,traceback
from .common import *
from .train_core import train
from measured.hall_adapters.core import batch_from_support
def run():
    seed();dev=DevelopmentData();bank=dev.fit()
    y=normalized_power(directional_power(bank.channel));yv=normalized_power(directional_power(dev.reference("validation")))
    base=np.load(WORK/"foundation_phase_safe.npy",mmap_mode="r")
    with np.load(WORK/"geometry_support.npz") as z:support={k:z[k] for k in ["coordinate","mass","mask","features","sequence","interaction","depth","identity"]}
    data=batch_from_support(base,support);d=device();data={k:v.to(d) for k,v in data.items()}
    target=torch.tensor(np.concatenate([y,yv]).reshape(-1,32),dtype=torch.float32,device=d)
    train("ours_phase_safe",dict(bounds=[.5,.25,0.],gate=.3,lr=.0003),data,target,2*len(y))
    dev.save_access("ours_phase_safe")
if __name__=="__main__":
    try:run()
    except BaseException:
        save_json(REPORT/"ours_phase_safe_failure.json",dict(traceback=traceback.format_exc()));raise

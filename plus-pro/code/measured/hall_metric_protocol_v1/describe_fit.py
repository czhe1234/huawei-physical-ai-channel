"""FIT delay-shape descriptions, without comparing any model or baseline."""
import numpy as np
from measured.hall_structured_v2.data import SouthDevelopment
from measured.hall_adapters.common import save_json
from .stage import REPORT
from .metrics import describe
from .check import stats
def run():
    reader=SouthDevelopment();bank=reader.fit();d=describe(bank.channel,bank.spec.bandwidth_hz)
    q=d["receive_delay_joint"].sum(-2);f=q.shape[-1]
    positive=q>0
    entropy=-np.sum(np.where(positive,q*np.log(np.maximum(q,1e-300)),0),axis=-1)
    within_one=q[...,f//2-1:f//2+2].sum(-1)
    result=dict(scope="Declared FIT only; distribution description, not a model score or physical delay-spread certification.",
        positions=len(bank.row_ids),receive_panels=q.shape[0]*q.shape[1],
        peak_bin_mass=stats(q.max(-1)),effective_delay_bins=stats(np.exp(entropy)),
        mass_within_center_plus_minus_one_bin=stats(within_one),
        by_panel=[dict(peak_bin_mass=stats(q[:,j].max(-1)),
            effective_delay_bins=stats(np.exp(entropy[:,j]))) for j in range(2)],
        cautions=["Numerical reference is not the physical first path.",
            "Full-period weak tails can materially affect RMS moments; noise/window rules remain to be fixed.",
            "Energy concentration does not by itself predict between-method ranking."],
        access=reader.access,trained_models_loaded=False,validation=False,test=False,north=False)
    save_json(REPORT/"delay_shape_description.json",result);print(result)
if __name__=="__main__":run()

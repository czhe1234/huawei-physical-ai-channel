"""Crossfitted measured17-expert cache without any formal-test access."""
from __future__ import annotations
import time,json,traceback
import numpy as np
from .common import *
from measured.hall_adapters.common import TrainingBank
from measured.hall_adapters.foundation import CarrierBank
def run():
    start=time.monotonic();dev=DevelopmentData();bank=dev.fit()
    rows=np.r_[dev.fit_ids,dev.val_ids];shape=(len(rows),17,2,8,4,1024)
    path=WORK/"experts.npy";progress=WORK/"experts_progress.json";audit=[]
    if path.exists():
        cache=np.load(path,mmap_mode="r+");done=json.loads(progress.read_text())["completed"]
        if cache.shape!=shape:raise ValueError("Expert cache shape")
        audit=json.loads(progress.read_text())["source_access"]
    else:cache=np.lib.format.open_memmap(path,mode="w+",dtype=np.complex64,shape=shape);done=0
    block=np.floor(bank.position[:,:2]/4).astype(np.int64)
    for i,row in enumerate(rows):
        if i<done:continue
        if i<len(dev.fit_ids):
            same=np.all(block==np.floor(dev.data.meta["position"][row,:2]/4),axis=-1);take=~same
        else:take=np.ones(len(bank.row_ids),bool)
        source=TrainingBank(bank.row_ids[take],bank.position[take],bank.channel[take],bank.spec)
        query=dev.data.query([row]);carrier=CarrierBank(source);values=carrier.experts(query)
        cache[i]=values[0]
        audit.append(dict(row=int(row),role="fit" if i<len(dev.fit_ids) else "validation",donor_rows=source.row_ids.tolist(),own_block_excluded=bool(i<len(dev.fit_ids))))
        cache.flush();save_json(progress,dict(completed=i+1,source_access=audit))
        if i%12==0:print("EXPERT",i,len(rows),time.monotonic()-start,flush=True)
    # Gram matrices reproduce broadband powers of ANY complex expert mixture.
    gram=np.zeros((len(rows),2,32,17,17),np.complex64);expert_power=np.zeros((len(rows),17,2,32),np.float64)
    for i in range(len(rows)):
        angular=np.fft.fft2(cache[i].astype(np.complex128),axes=(2,3),norm="ortho").reshape(17,2,32,1024)
        angular=angular.transpose(1,2,0,3)
        g=angular@angular.conj().swapaxes(-1,-2)/1024
        scale=np.maximum(np.trace(g,axis1=-2,axis2=-1).real.sum(-1),1e-30)
        gram[i]=g/scale[:,None,None,None]
        expert_power[i]=np.diagonal(g,axis1=-2,axis2=-1).real.transpose(2,0,1)
    np.savez_compressed(WORK/"expert_gram.npz",row_ids=rows,gram=gram,expert_power=expert_power)
    save_json(REPORT/"expert_cache.json",dict(complete=True,shape=shape,bytes=int(cache.nbytes),seconds=time.monotonic()-start,source_access=audit,fit_block_crossfit=True,validation_labels_read=False))
    dev.save_access("experts")
if __name__=="__main__":
    try:run()
    except BaseException:
        save_json(REPORT/"experts_failure.json",dict(traceback=traceback.format_exc()));raise

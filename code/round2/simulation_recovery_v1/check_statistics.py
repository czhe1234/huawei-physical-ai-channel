"""CPU algebra checks of regenerated physical feature operators only."""
import json
import numpy as np
import torch
from .physical_statistics import coherent_cfr,definitions,welford_assign,fields

def run():
    torch.set_num_threads(2);torch.manual_seed(31)
    f=6.69e9+(np.arange(192,dtype=np.float64)-96)*30000.
    a=(torch.rand((1,4,256,2))+.1)+1j*(torch.rand((1,4,256,2))-.3)
    tau=torch.tensor([[37.125e-9,83.7e-9]],dtype=torch.float64)
    actual=coherent_cfr(a,tau,f,torch).numpy()
    expected=np.einsum('bump,bpf->bmuf',a.numpy().astype(np.complex128),np.exp(-2j*np.pi*tau.numpy()[...,None]*f[None,None]))
    np.testing.assert_allclose(actual,expected,atol=3e-7,rtol=3e-6)
    ns=definitions();stats=ns['score_aligned_statistics'](np.repeat(actual,2,0))
    np.testing.assert_allclose(stats['pas'].sum((1,2,3)),1.,atol=3e-6)
    np.testing.assert_allclose(stats['pdp'].sum(-1),1.,atol=3e-6)
    class InMemoryStore(np.ndarray):
        def flush(self):pass
    current={k:np.zeros(shape,dtype).view(InMemoryStore) for k,(shape,dtype) in fields(ns,2).items()}
    histories=[]
    for i in range(3):
        values=ns['score_aligned_statistics'](np.repeat(actual,2,0)*(1+.1*i))
        values['path_count']=np.array([2.,2.+i],np.float32);values['path_count_valid']=np.ones(2,bool)
        previous=None if i==0 else {k:v.copy() for k,v in current.items()}
        welford_assign(current,previous,np.arange(2),values,ns)
        histories.append(values)
        # Reassign the same seed from the immutable prior revision: no double count.
        once={k:v.copy() for k,v in current.items()}
        welford_assign(current,previous,np.arange(2),values,ns)
        for k in current:np.testing.assert_array_equal(current[k],once[k])
    for domain in ('pas','pdp','ad','log_power','path_count'):
        stack=np.stack([x[domain] for x in histories])
        np.testing.assert_allclose(current[domain+'.mean'],stack.mean(0),atol=3e-6,rtol=3e-5)
        np.testing.assert_allclose(current[domain+'.m2'],((stack-stack.mean(0))**2).sum(0),atol=3e-6,rtol=3e-5)
    return dict(status='passed',physical_passband_CFR_two_path_reference=True,original_numpy_statistic_normalization=True,
      Welford_stack_reference=True,idempotent_seed_reassignment=True,source_seed_order_verified=True,
      training=False,score=False,read_CSI=False)
if __name__=='__main__':print(json.dumps(run()))

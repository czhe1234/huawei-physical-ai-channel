"""Small independent checks for the trained measured inference contract."""
import unittest
import numpy as np
import torch
from .method import angular_writeback,DevelopedV16
from measured.hall_adapters.metrics import codebook
class Functional(unittest.TestCase):
    def test_writeback_independent_DFT_and_energy(self):
        torch.manual_seed(2);h=torch.randn(3,1,8,4,1024,dtype=torch.complex64)
        p=torch.rand(3,32);p=p/p.sum(-1,keepdim=True)
        out=angular_writeback(h,p)
        literal=np.einsum("ab,naf->nbf",codebook().conj(),out.numpy().reshape(3,32,1024))
        actual=(np.abs(literal)**2).mean(-1);actual/=actual.sum(-1,keepdims=True)
        np.testing.assert_allclose(actual,p.numpy(),atol=1e-6)
        torch.testing.assert_close(out.abs().square().sum((-3,-2,-1)),h.abs().square().sum((-3,-2,-1)),rtol=2e-6,atol=1e-6)
    def test_zero_carrier(self):
        h=torch.zeros(2,1,8,4,64,dtype=torch.complex64);p=torch.full((2,32),1/32)
        self.assertEqual(int(torch.count_nonzero(angular_writeback(h,p))),0)
    def test_gate_mask_and_zero_carrier(self):
        torch.set_num_threads(1);torch.manual_seed(4);m=DevelopedV16().eval();b,k=3,2
        batch=dict(base=torch.zeros(b,k,120),sequence_geometry=torch.zeros(b,k,16,21),interactions=torch.zeros(b,k,16,dtype=torch.long),
          depth=torch.zeros(b,k,dtype=torch.long),coordinate=torch.zeros(b,k,3),base_log_mass=torch.zeros(b,k),
          mask=torch.tensor([[True,False],[False,False],[True,False]]),channel=torch.randn(b,1,8,4,64,dtype=torch.complex64))
        batch["channel"][2]=0
        p,state=m(batch);h,_=m(batch,full_channel=True)
        self.assertTrue(torch.isfinite(p).all());self.assertTrue(torch.isfinite(h).all())
        self.assertEqual(float(state["gate"][1]),0.);self.assertEqual(float(state["gate"][2]),0.)
        torch.testing.assert_close(h[1],batch["channel"][1],rtol=2e-6,atol=1e-6)
        self.assertEqual(float(p[2].sum()),0.)
if __name__=="__main__":unittest.main()

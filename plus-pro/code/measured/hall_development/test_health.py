"""Independent implementation checks required before optimizer runs."""
import unittest
import numpy as np
import torch
from .wrf_batched import dense_batch
from measured.hall_adapters.wrf import alpha_render
from .common import DevelopmentData
class Checks(unittest.TestCase):
    def test_batched_alpha_value_and_gradient(self):
        torch.manual_seed(7)
        pixel=torch.tensor([[3.,2.],[2.7,2.4],[4.,1.]],requires_grad=True)
        cov=torch.eye(2).repeat(3,1,1).requires_grad_()
        depth=torch.tensor([2.,3.,4.]);opacity=torch.tensor([.2,.4,.3],requires_grad=True)
        color=torch.tensor([[.4],[.7],[.2]],requires_grad=True)
        out=alpha_render(pixel,cov,depth,opacity,color,8,5)
        grad=torch.autograd.grad(out.square().sum(),(pixel,cov,opacity,color),retain_graph=True)
        bat=dense_batch(pixel[None],cov[None],depth[None],opacity,color[None],8,5)[0]
        grad2=torch.autograd.grad(bat.square().sum(),(pixel,cov,opacity,color))
        torch.testing.assert_close(out,bat,rtol=2e-6,atol=2e-7)
        for a,b in zip(grad,grad2):torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-7)
    def test_terminal_alpha_not_counted(self):
        pix=torch.zeros((1,3,2));cov=torch.eye(2).repeat(1,3,1,1)
        opacity=torch.tensor([.99,.99,.99])
        col=torch.tensor([[[.2],[.4],[.9]]]);dep=torch.tensor([[1.,2.,3.]])
        a=alpha_render(pix[0],cov[0],dep[0],opacity,col[0],1,1)
        b=dense_batch(pix,cov,dep,opacity,col,1,1)[0]
        torch.testing.assert_close(a,b)
    def test_no_formal_reference(self):
        d=DevelopmentData()
        with self.assertRaises(ValueError):d.reference("test")
        with self.assertRaises(ValueError):d.query("test")
        with self.assertRaises(ValueError):d.reference("north")
if __name__=="__main__":unittest.main()

"""Synthetic metric identities and failure checks, no fitted model scores."""
import unittest
import numpy as np
import torch
from .metrics import *
def random_h(seed=5,f=64):
    rng=np.random.default_rng(seed)
    return (rng.normal(size=(3,2,8,4,f))+1j*rng.normal(size=(3,2,8,4,f))).astype(np.complex64)
def clock(h):
    rng=np.random.default_rng(7);n,p=h.shape[:2];f=h.shape[-1]
    tau=rng.uniform(-f/2,f/2,(n,p));phase=rng.uniform(-np.pi,np.pi,(n,p));gain=rng.uniform(.4,2.,(n,p))
    return h*gain[...,None,None,None]*np.exp(1j*phase[...,None,None,None]-2j*np.pi*tau[...,None,None,None]*np.arange(f)/f)
def two_channels(tau=(2.,6.),f=64):
    h=np.zeros((1,2,32,f),np.complex128)
    for m,delay in enumerate(tau):h[:,:,m]=np.exp(-2j*np.pi*delay*np.arange(f)/f)
    return h.reshape(1,2,8,4,f)
class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_official_pas_layout_adaptation(self):
        from common.official_metrics import point_metric_statistics_torch,ArrayLayout
        p,t=random_h(),random_h(6)
        axis=lambda x:torch.tensor(x.reshape(3,2,32,64).transpose(0,2,1,3))
        official=point_metric_statistics_torch(axis(p),axis(t),ArrayLayout(1,8,4))
        self.assertAlmostEqual(compare(p,t)["pas_cosine"],float(official.pas.mean()),places=6)
    def test_official_pdp_operator_after_independent_reference(self):
        from common.official_metrics import point_metric_statistics_torch,ArrayLayout
        p,t=random_h(),random_h(6)
        axis=lambda x:torch.tensor(x.reshape(3,2,32,64).transpose(0,2,1,3).copy())
        official=point_metric_statistics_torch(axis(reference(p)["channel"]),axis(reference(t)["channel"]),ArrayLayout(1,8,4))
        self.assertAlmostEqual(compare(p,t)["relative_pdp_cosine"],float(official.pdp.mean()),places=6)
    def test_identical_identity(self):
        h=random_h();r=compare(h,h)
        self.assertAlmostEqual(r["pas_cosine"],1.,places=12)
        self.assertAlmostEqual(r["relative_pdp_cosine"],1.,places=12)
        self.assertLess(r["receive_energy_TV"],1e-12);self.assertLess(r["receive_delay_joint_TV"],1e-12)
    def test_common_fractional_clock_invariance(self):
        h=random_h();s=clock(h);a,b=describe(h),describe(s)
        np.testing.assert_allclose(a["receive_delay_joint"],b["receive_delay_joint"],atol=1e-12,rtol=1e-9)
        r=compare(s,h);self.assertAlmostEqual(r["relative_pdp_cosine"],1.,places=11)
        self.assertLess(r["receive_energy_TV"],1e-12)
    def test_relative_interchannel_delay_not_erased(self):
        a,b=two_channels(),two_channels((2.,10.))
        r=compare(b,a)
        self.assertLess(r["relative_pdp_cosine"],.99);self.assertGreater(r["receive_delay_joint_TV"],.1)
        self.assertLess(r["receive_energy_TV"],1e-12)
    def test_same_marginals_wrong_receive_delay_pairing(self):
        a,b=two_channels(),two_channels((6.,2.))
        da,db=describe(a),describe(b)
        np.testing.assert_allclose(da["receive_delay_joint"].sum(-2),db["receive_delay_joint"].sum(-2),atol=1e-12)
        self.assertLess(compare(a,b)["receive_energy_TV"],1e-12)
        self.assertGreater(compare(a,b)["receive_delay_joint_TV"],.99)
    def test_channel_energy_error_detected(self):
        a=two_channels();b=a.copy();b[:,:,0,0]*=2
        r=compare(b,a)
        self.assertGreater(r["receive_energy_TV"],.2)
    def test_parseval_receive_marginal(self):
        d=describe(random_h())
        np.testing.assert_allclose(d["receive_delay_joint"].sum(-1),d["energy_fraction"],atol=1e-15)
    def test_broadband_match_can_hide_per_frequency_error(self):
        z=np.zeros((1,2,32,64),np.complex128)
        z[:,:,0,:32]=1;z[:,:,1,32:]=1
        w=np.roll(z,32,axis=-1)
        h=np.fft.ifft2(z.reshape(1,2,8,4,64),axes=(-3,-2),norm="ortho")
        t=np.fft.ifft2(w.reshape(1,2,8,4,64),axes=(-3,-2),norm="ortho")
        self.assertAlmostEqual(compare(h,t)["pas_cosine"],0.,places=12)
        p,q=describe(h)["pas"].sum(-1),describe(t)["pas"].sum(-1)
        np.testing.assert_allclose(cosine(p,q,-1),1.,atol=1e-15)
    def test_zero_prediction_fails_on_fixed_reference_mask(self):
        h=random_h();r=compare(np.zeros_like(h),h)
        self.assertEqual(r["pas_cosine"],0.);self.assertEqual(r["relative_pdp_cosine"],0.)
        self.assertEqual(r["receive_energy_TV"],1.);self.assertEqual(r["receive_delay_joint_TV"],1.)
        self.assertEqual(r["failed_prediction_pdp_links"],192)
    def test_zero_reference_is_counted_not_a_score(self):
        h=random_h();r=compare(h,np.zeros_like(h))
        self.assertIsNone(r["pas_cosine"]);self.assertIsNone(r["relative_pdp_cosine"])
        self.assertIsNone(r["receive_energy_TV"]);self.assertEqual(r["reference_valid_energy_panels"],0)
    def test_undefined_predicted_anchor_not_dropped(self):
        h=random_h();bad=np.zeros_like(h);bad[...,0]=1
        r=compare(bad,h)
        self.assertEqual(r["relative_pdp_cosine"],0.);self.assertEqual(r["receive_delay_joint_TV"],1.)
        self.assertEqual(r["failed_prediction_pdp_links"],192)
    def test_equal_panel_weight_with_different_valid_link_counts(self):
        h=two_channels()
        flat=h.reshape(1,2,32,64).copy()
        flat[:,0,1]=0
        for m in range(32):flat[:,1,m]=np.exp(-2j*np.pi*(2 if m%2==0 else 6)*np.arange(64)/64)
        h=flat.reshape(1,2,8,4,64);p=h.copy();p[:,1]=0
        self.assertAlmostEqual(compare(p,h)["relative_pdp_cosine"],.5,places=12)
    def test_nonfinite_rejected(self):
        h=random_h();h[0,0,0,0,0]=np.nan
        with self.assertRaises(ValueError):describe(h)
    def test_current_shared_receive_head_restriction(self):
        from measured.hall_structured_v2.model import render_joint
        g=torch.linspace(-.2,.2,32)[None];a=torch.tensor([[[1.,.5,.2],[3.,1.2,2.5]]])
        _,_,e1=render_joint(torch.tensor([[.8,.2]]),a,g,torch.tensor([0.]),64)
        _,_,e2=render_joint(torch.tensor([[.3,.7]]),a+torch.tensor([.2,.1,3.]),g,torch.tensor([0.]),64)
        torch.testing.assert_close(e1,e2,atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(e1,(2*g).softmax(-1),atol=1e-7,rtol=1e-6)
if __name__=="__main__":unittest.main()

"""Synthetic engineering checks; no optimizer step and no real-data score."""
import unittest,tempfile,json
from pathlib import Path
import numpy as np
import torch
from dataclasses import replace
from measured.hall_adapters.common import ArraySpec,TrainingBank,Query
from measured.hall_adapters.metrics import codebook
from .observables import *
from .config import Config
from .geometry import aggregate,legacy_geometry
from .foundation import prepare
from .model import HallStructured,render_joint
from .synthesis import reconstruct,delay_distribution,element_distribution,canonicalize,restore_reference,angle_project
from .cache import write_cache,load_cache
from .foundation import band_energy
def fixture(f=64,geometry=True):
    rng=np.random.default_rng(4)
    spec=ArraySpec(np.array([[-2.,0.,1.],[2.,0.,1.]]),np.array([[1.,0.,0.],[-1.,0.,0.]]),frequency_count=f)
    ids=np.arange(8);position=np.c_[np.arange(8)*.7,np.sin(np.arange(8)),np.ones(8)]
    channel=(rng.normal(size=(8,2,8,4,f))+1j*rng.normal(size=(8,2,8,4,f))).astype(np.complex64)
    bank=TrainingBank(ids,position,channel,spec);q=Query(ids[:2],position[:2],spec)
    g=None
    if geometry:
        n,p,k=2,2,3
        a=dict(coordinate=np.zeros((n,p,k,3),np.float32),mass=np.ones((n,p,k),np.float32),
          mask=np.ones((n,p,k),bool),features=np.zeros((n,p,k,24),np.float32),
          sequence=np.zeros((n,p,k,16,21),np.float32),interaction=np.zeros((n,p,k,16),np.int64),
          depth=np.ones((n,p,k),np.int64))
        a["coordinate"][...,:2]=np.array([[1.3,.7],[3.2,1.2],[6.1,2.3]])
        a["sequence"][...,0,:3]=np.array([[.1,.2,.03],[.2,.1,.03],[.3,.1,.03]])
        a["interaction"][...,0]=1
        a["features"][...,15]=.05
        a["features"][...,14]=np.array([.073,.12,.16])
        records=[]
        for source in ["base","scatter"]:
          for seed in [1,2]:
            b={key:value.copy() for key,value in a.items()};b["mass"]*=1+.1*seed
            records.append(dict(config=source,seed=seed,support=b,ray_converged=True))
        g=aggregate(records,cap=4)
    prepared=prepare(bank,q,16,"self",g)
    return bank,q,prepared
class Mathematical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_literal_DFT_covariance_and_correlation(self):
        bank,q,d=fixture();h=torch.tensor(bank.channel[:2]);ob=structure(h)
        z=np.einsum("ab,npaf->npbf",codebook().conj(),bank.channel[:2].reshape(2,2,32,-1))
        expected=np.abs(z)**2;expected=expected.sum(-1);expected/=expected.sum(-1,keepdims=True)
        np.testing.assert_allclose(ob["broad"].numpy(),expected,rtol=2e-5,atol=1e-7)
        x=bank.channel[:2].reshape(2,2,32,-1).astype(np.complex128)
        r=x@x.conj().swapaxes(-1,-2);r/=np.trace(r,axis1=-2,axis2=-1)[...,None,None]
        np.testing.assert_allclose(ob["covariance"].numpy(),r,atol=2e-7)
        cross=(x[...,1:]*x[...,:-1].conj()).sum((-2,-1))
        c=np.abs(cross)/np.sqrt((np.abs(x[...,1:])**2).sum((-2,-1))*(np.abs(x[...,:-1])**2).sum((-2,-1)))
        np.testing.assert_allclose(ob["correlation"][...,0].numpy(),c,atol=2e-7)
    def test_common_phase_fractional_delay_invariance(self):
        bank,q,d=fixture();h=torch.tensor(bank.channel[:2]);a=structure(h)
        k=torch.arange(h.shape[-1]);phase=torch.tensor([[.2,-.3],[1.2,.7]])[...,None,None,None]
        tau=torch.tensor([[1.37,2.22],[-3.1,4.7]])[...,None,None,None]
        shifted=h*torch.exp(1j*phase-2j*torch.pi*tau*k/h.shape[-1])
        b=structure(shifted)
        for key in ["q","covariance","correlation","beam_correlation"]:
            torch.testing.assert_close(a[key],b[key],rtol=2e-4,atol=2e-6)
    def test_perfect_loss_and_zero_prediction_policy(self):
        bank,q,d=fixture();h=torch.tensor(bank.channel[:2]);power=FitPower.from_fit(bank.channel,bank.row_ids)
        v=losses(h,h,power)
        self.assertLess(float(v["structural"]),1e-7)
        bad=losses(torch.zeros_like(h),h,power)
        self.assertEqual(int(bad["failed_panels"]),4)
        self.assertGreaterEqual(float(bad["structural"]),.99)
    def test_delay_can_differ_at_identical_broadband_angle(self):
        f=64;k=torch.arange(f);a=torch.ones(8,4,dtype=torch.complex64)
        s1=torch.exp(-2j*torch.pi*.37*k/f)+.7*torch.exp(-2j*torch.pi*2.2*k/f)
        s2=torch.exp(-2j*torch.pi*.37*k/f)+.7*torch.exp(-2j*torch.pi*7.8*k/f)
        x=(a[...,None]*s1).expand(1,2,-1,-1,-1).clone();y=(a[...,None]*s2).expand_as(x).clone()
        o1,o2=structure(x),structure(y)
        torch.testing.assert_close(o1["broad"],o2["broad"],atol=1e-6,rtol=1e-5)
        self.assertGreater(float((o1["correlation"]-o2["correlation"]).abs().max()),.1)
    def test_joint_render_delay_gradient_and_PSD(self):
        mass=torch.tensor([[.4,.6]]);coord=torch.tensor([[[1.3,.4,.37],[3.2,1.6,2.21]]],requires_grad=True)
        gain=torch.zeros(1,32,requires_grad=True)
        joint,r,e=render_joint(mass,coord,gain,torch.tensor([.2]),64)
        torch.testing.assert_close(joint.sum(),torch.tensor(1.))
        torch.testing.assert_close(r,r.conj().transpose(-1,-2))
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(r).min()),-1e-6)
        target=torch.linspace(0,1,64)
        (joint*target).sum().backward()
        self.assertTrue(torch.isfinite(coord.grad).all());self.assertGreater(float(coord.grad[...,2].abs().max()),1e-6)
    def test_reconstruction_identity_and_power(self):
        bank,q,d=fixture();h=torch.tensor(bank.channel[:2])
        out,diag=reconstruct(h,angle_frequency(h,16),delay_distribution(h),element_distribution(h),total_energy(h),cycles=2)
        torch.testing.assert_close(out,h,rtol=2e-5,atol=2e-6)
        self.assertLess(float(diag["power_relative_residual"]),2e-6)
    def test_aggregation_and_unknown_legacy_variance(self):
        bank,q,d=fixture();g=d.geometry
        self.assertEqual(g.quality["source_configs"],["base","scatter"])
        self.assertTrue(g.variance_known[g.arrays["mask"]].all())
        source=g.source_fraction[g.arrays["mask"]]
        np.testing.assert_allclose(source.sum(-1),1.)
        old=legacy_geometry(g.arrays,presence=g.presence*2)
        self.assertFalse(old.variance_known.any());self.assertFalse(old.quality["ray_convergence_certified"])
    def test_canonical_reference_roundtrip_and_invariance(self):
        bank,q,d=fixture();h=torch.tensor(bank.channel[:2])
        k=torch.arange(h.shape[-1])
        phi=torch.tensor([[.2,-.7],[1.3,2.2]])[...,None,None,None]
        tau=torch.tensor([[.37,2.27],[-2.4,3.3]])[...,None,None,None]
        shifted=h*torch.exp(1j*phi-2j*torch.pi*tau*k/h.shape[-1])
        a,r=canonicalize(h);b,_=canonicalize(shifted)
        torch.testing.assert_close(restore_reference(a,r),h,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(a,b,atol=4e-5,rtol=2e-4)
    def test_zero_carrier_and_empty_support_stability(self):
        h=torch.zeros(1,2,8,4,64,dtype=torch.complex64)
        q=torch.full((1,2,32,16),1/(32*16))
        a=angle_project(h,q)
        self.assertTrue(torch.isfinite(a).all());self.assertGreater(float(a.abs().max()),0)
        v,r=canonicalize(h)
        out,diag=reconstruct(v,q,torch.full((1,2,32,64),1/64),torch.full((1,2,32),1/32),torch.zeros(1,2))
        self.assertEqual(float(out.abs().max()),0.);self.assertTrue(torch.isfinite(out).all())
    def test_band_energy_preserves_frequency_and_total_power(self):
        bank,q,d=fixture()
        energy=band_energy(bank.channel,16)
        np.testing.assert_allclose(energy.sum((-2,-1)),np.abs(bank.channel).astype(np.float64).__pow__(2).sum((-3,-2,-1))/64,rtol=3e-7)
        self.assertGreater(float(np.std(energy,axis=-1).max()),.001)
        self.assertEqual(tuple(d.expert_energy.shape),(2,13,2,32,16))
    def test_geometry_cache_roundtrip_and_identity_guards(self):
        bank,q,d=fixture();identity=dict(view="synthetic_south",split="frozen")
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"geometry.npz"
            write_cache(path,d.geometry,q.row_ids,q.position,q.spec,identity,dict(source="synthetic"))
            g=load_cache(path,q,identity)
            np.testing.assert_array_equal(g.arrays["coordinate"],d.geometry.arrays["coordinate"])
            with self.assertRaises(FileExistsError):write_cache(path,g,q.row_ids,q.position,q.spec,identity,{})
            with self.assertRaises(ValueError):load_cache(path,q,dict(view="other"))
            with self.assertRaises(ValueError):load_cache(path,Query(q.row_ids,q.position+.1,q.spec),identity)
    def test_zero_prediction_is_not_a_valid_beam_score(self):
        bank,q,d=fixture();truth=torch.tensor(bank.channel[:2])
        m=metrics(torch.zeros_like(truth),truth,FitPower.from_fit(bank.channel,bank.row_ids))
        self.assertEqual(m["failed_prediction_panels"],4);self.assertIsNone(m["mean_beam_loss_db"])
class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def model(self,with_geometry=True):
        bank,q,data=fixture(geometry=with_geometry);power=FitPower.from_fit(bank.channel,bank.row_ids)
        config=Config(geometry_candidates=4,reserve_candidates=4,projection_cycles=2,use_geometry=with_geometry)
        torch.manual_seed(10);model=HallStructured(bank.spec,power,config)
        model.set_fit_statistics(data);return bank,q,data,model
    def test_no_own_donor_and_FIT_only_statistics(self):
        bank,q,data,m=self.model()
        for rid,donors in zip(q.row_ids,data.source_ids):self.assertNotIn(int(rid),donors)
        wrong=replace(data,row_ids=np.array([100,101]))
        with self.assertRaises(ValueError):m.set_fit_statistics(wrong)
        m.eval()
        with torch.no_grad():a=m(data)["channel"]
        self.assertEqual(tuple(a.shape),(2,2,8,4,64));self.assertTrue(torch.isfinite(a).all())
    def test_measurement_only_pipeline(self):
        bank,q,d,m=self.model(False)
        out=m(d)
        self.assertEqual(tuple(out["channel"].shape),(2,2,8,4,64))
        self.assertEqual(float(out["geometry_quality"].abs().max()),0.)
    def test_restored_heads_and_backbone_gradients_without_update(self):
        bank,q,d,m=self.model()
        # Nonzero fixed residual-head fixture tests encoder connectivity, not training.
        torch.manual_seed(23)
        for head in [m.state.score_head,m.state.coordinate_head,m.state.gate_head,m.state.spectral_head,m.state.receive_head,m.state.power_head,m.state.lifecycle_head]:
            torch.nn.init.normal_(head.weight,std=.002)
        before={name:p.detach().clone() for name,p in m.named_parameters()}
        out=m(d);target=torch.tensor(bank.channel[:2]).roll(1,-1)
        terms=losses(out["channel"],target,m.power_stats())
        objective=terms["structural"]+.1*terms["power"]+1e-4*out["residual"]
        objective.backward()
        keys=["foundation.global_logits","foundation.donor_correction.2.weight","context_encoder.0.weight",
          "state.score_head.weight","state.coordinate_head.weight","state.lifecycle_head.weight","state.spectral_head.weight",
          "state.receive_head.weight","state.power_head.weight","state.gate_head.weight",
          "state.sequence_input.weight","state.set_encoder.layers.0.linear1.weight"]
        parameters=dict(m.named_parameters())
        for key in keys:
            grad=parameters[key].grad
            self.assertIsNotNone(grad,key);self.assertTrue(torch.isfinite(grad).all(),key)
            self.assertGreater(float(grad.abs().max()),0.,key)
        self.assertGreater(float(parameters["state.coordinate_head.weight"].grad[2].abs().max()),0.)
        for name,p in m.named_parameters():torch.testing.assert_close(p,before[name],rtol=0,atol=0)
    def test_inference_identity_contract(self):
        bank,q,d,m=self.model();pred=m.predict(q,d)
        np.testing.assert_array_equal(pred["row_ids"],q.row_ids)
        self.assertEqual(pred["channel"].dtype,np.complex64)
        with self.assertRaises(ValueError):m.predict(Query(np.array([90,91]),q.position,q.spec),d)
        with self.assertRaises(ValueError):m.predict(Query(q.row_ids,q.position+.1,q.spec),d)
        with self.assertRaises(ValueError):prepare(bank,Query(q.row_ids,q.position+.1,q.spec),16,"self")
        other=replace(q.spec,spacing_m=.05)
        with self.assertRaises(ValueError):m.predict(Query(q.row_ids,q.position,other),d)
    def test_model_reference_invariance_for_independent_donor_clocks(self):
        bank,q,d,m=self.model();m.eval()
        torch.manual_seed(67)
        for head in [m.state.score_head,m.state.coordinate_head,m.state.spectral_head,m.state.receive_head,m.state.gate_head]:
            torch.nn.init.normal_(head.weight,std=.002)
        rng=np.random.default_rng(12);phi=rng.uniform(-2,2,(8,2,1,1,1));tau=rng.uniform(-4,4,(8,2,1,1,1))
        k=np.arange(64);shift=bank.channel*np.exp(1j*phi-2j*np.pi*tau*k/64)
        other=TrainingBank(bank.row_ids,bank.position,shift.astype(np.complex64),bank.spec)
        moved=prepare(other,q,16,"self",d.geometry)
        with torch.no_grad():a=m(d)["channel"];b=m(moved)["channel"]
        aa,bb=structure(a),structure(b)
        for key in ["q","covariance","correlation","beam_correlation","energy"]:
            torch.testing.assert_close(aa[key],bb[key],atol=3e-5,rtol=3e-4)
    def test_delay_head_changes_within_beam_frequency_structure(self):
        bank,q,d,m=self.model();m.eval()
        with torch.no_grad():
            a=m(d)["channel"]
            torch.manual_seed(31)
            m.state.coordinate_head.weight[2].normal_(std=.03)
            b=m(d)["channel"]
        ca,cb=structure(a)["beam_correlation"],structure(b)["beam_correlation"]
        self.assertGreater(float((ca-cb).abs().max()),1e-6)
    def test_training_entry_validation_without_running_it(self):
        from .train import parser,validate_args
        args=parser().parse_args(["--name","synthetic_contract","--device","cpu"])
        validate_args(args)
        args.name="../bad"
        with self.assertRaises(ValueError):validate_args(args)
        args.name="good";args.epochs=0
        with self.assertRaises(ValueError):validate_args(args)
if __name__=="__main__":unittest.main()

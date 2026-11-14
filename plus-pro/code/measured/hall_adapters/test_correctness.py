"""Independent small mathematical and interface checks; synthetic labels only."""
from __future__ import annotations
import copy
from pathlib import Path
import inspect
import unittest
import numpy as np
import torch
from .common import ArraySpec, Query, TrainingBank, SouthDataset, check_prediction
from .metrics import (codebook,directional_power,torch_directional_power,normalized_power,
                      primary_metrics,secondary_metrics,aggregate_primary,aggregate_secondary,training_power_loss)
from .baselines import RBFPower,PositionPowerMLP,DirectPointing
from .foundation import CarrierBank
from .core import MeasuredV16,batch_from_support,synthesize,HallPathState
from .wrf import WRFGSPlusPower,alpha_render,project_gaussians,angular_grid,author_network_class

def fixture(n=40,frequency=64):
    rng=np.random.default_rng(23)
    spec=ArraySpec(np.array([[0.,-5.,1.4],[0.,5.,1.4]]),np.array([[0.,1.,0.],[0.,-1.,0.]]),frequency_count=frequency)
    position=np.c_[rng.uniform(-2,2,n),rng.uniform(-2,2,n),np.full(n,.93)]
    h=(rng.normal(size=(n,2,8,4,frequency))+1j*rng.normal(size=(n,2,8,4,frequency))).astype(np.complex64)
    bank=TrainingBank(np.arange(n),position,h,spec)
    query=Query(np.array([1000]),np.array([[.37,.23,.93]]),spec)
    return bank,query

def support_fixture(n=1,k=6,frequency=64):
    rng=np.random.default_rng(113)
    coordinate=rng.uniform(0,1,(n,2,k,3))*np.array([8,4,frequency])
    mass=rng.uniform(.1,1,(n,2,k));mass[...,-1]=0;mass/=mass.sum(-1,keepdims=True)
    mask=mass>0;depth=rng.integers(0,3,(n,2,k));depth[~mask]=0
    return dict(coordinate=coordinate.astype(np.float32),mass=mass.astype(np.float32),mask=mask,
        features=rng.normal(size=(n,2,k,24)).astype(np.float32),
        sequence=rng.normal(size=(n,2,k,16,21)).astype(np.float32),
        interaction=rng.integers(0,4,(n,2,k,16)),depth=depth)

class MetricsChecks(unittest.TestCase):
    def test_codebook_and_literal_complex_projection(self):
        bank,_=fixture(n=1)
        w=codebook();np.testing.assert_allclose(w.conj().T@w,np.eye(32),atol=2e-15)
        literal=np.empty((1,2,32))
        h=bank.channel[0].reshape(2,32,-1).astype(np.complex128)
        for p in range(2):
            for b in range(32):literal[0,p,b]=np.mean(np.abs(w[:,b].conj()@h[p])**2)
        np.testing.assert_allclose(directional_power(bank.channel),literal,rtol=2e-14,atol=2e-14)

    def test_torch_transform_and_parseval(self):
        bank,_=fixture(n=2)
        p=directional_power(bank.channel)
        t=torch_directional_power(torch.tensor(bank.channel,dtype=torch.complex128)).numpy()
        np.testing.assert_allclose(p,t,rtol=2e-14,atol=2e-14)
        np.testing.assert_allclose(p.sum(-1),(np.abs(bank.channel.astype(np.complex128))**2).sum((2,3)).mean(-1),rtol=2e-14)

    def test_known_beam_sign_and_order(self):
        for beam in [0,1,5,17,31]:
            h=np.tile(codebook()[:,beam].reshape(1,1,8,4,1),(1,2,1,1,64))
            p=directional_power(h)
            np.testing.assert_array_equal(p.argmax(-1),[[beam,beam]])
            np.testing.assert_allclose(p[:,:,beam],1,atol=1e-14)

    def test_broadband_energy_precedes_normalization(self):
        h=np.zeros((1,2,8,4,2),complex)
        h[...,0]=10*codebook()[:,0].reshape(1,1,8,4)
        h[...,1]=codebook()[:,1].reshape(1,1,8,4)
        p=directional_power(h)
        np.testing.assert_allclose(p[...,0]/p[...,1],100,rtol=1e-12)

    def test_common_phase_delay_scale_invariance(self):
        bank,_=fixture(n=2)
        h=bank.channel.astype(np.complex128)
        k=np.arange(h.shape[-1]);gain=np.array([.3,2.])[None,:,None,None,None]
        phase=np.exp(1j*(np.array([.2,-1.])[None,:,None,None,None]-2*np.pi*1.37*k/len(k)))
        transformed=h*gain*phase
        np.testing.assert_allclose(normalized_power(directional_power(h)),normalized_power(directional_power(transformed)),atol=2e-15)
        result=primary_metrics({"channel":transformed},directional_power(h))
        np.testing.assert_allclose(result["direction_cosine"],1,atol=1e-14)
        np.testing.assert_allclose(result["beam_loss_db"],0,atol=1e-14)
        second=secondary_metrics({"channel":transformed},h)
        np.testing.assert_allclose(second["covariance_error"],0,atol=1e-25)
        np.testing.assert_allclose(second["frequency_correlation_absolute_error"],0,atol=1e-14)

    def test_beam_selected_without_truth_and_hand_computed_loss(self):
        p=np.ones((1,2,32));p[...,3]=2
        truth=np.ones_like(p);truth[...,5]=4
        r=primary_metrics({"power":p},truth)
        np.testing.assert_array_equal(r["beam_id"],[[3,3]])
        np.testing.assert_allclose(r["beam_loss_db"],10*np.log10(4),rtol=1e-14)
        changed=truth.copy();changed[...,5]=1;changed[...,9]=100
        np.testing.assert_array_equal(primary_metrics({"power":p},changed)["beam_id"],r["beam_id"])

    def test_zero_prediction_and_undefined_truth_are_not_silently_dropped(self):
        p=np.zeros((1,2,32));truth=p.copy();truth[...,5]=1
        r=primary_metrics({"power":p},truth);a=aggregate_primary(r)
        self.assertEqual(a["reference_panels_valid"],2);self.assertEqual(a["zero_prediction_panels"],2)
        self.assertEqual(a["mean_direction_cosine"],0);self.assertTrue(a["infinite_mean_beam_loss"])
        z=aggregate_primary(primary_metrics({"power":p},np.zeros_like(p)))
        self.assertEqual(z["undefined_reference_panels"],2);self.assertIsNone(z["mean_direction_cosine"])

    def test_power_only_secondaries_are_not_fabricated(self):
        bank,_=fixture(n=1)
        self.assertFalse(secondary_metrics({"power":np.ones((1,2,32))},bank.channel)["applicable"])

    def test_failed_complex_prediction_not_dropped_from_secondary_denominator(self):
        bank,_=fixture(n=1)
        points=secondary_metrics({"channel":np.zeros_like(bank.channel)},bank.channel)
        result=aggregate_secondary(points)
        self.assertEqual(result["failed_prediction_correlation_cells"],int(points["reference_valid_correlation"].sum()))
        self.assertIsNone(result["frequency_correlation_MAE"])

    def test_nontrivial_secondary_metrics_have_analytic_values(self):
        reference=np.zeros((1,2,8,4,64),np.complex128)
        predicted=np.zeros_like(reference)
        reference[:,:,0,0]=1;predicted[:,:,0,1]=1
        result=secondary_metrics({"channel":predicted},reference)
        np.testing.assert_allclose(result["covariance_error"],2,atol=1e-14)
        k=np.arange(64)
        reference[:,:,0,1]=np.exp(1j*np.pi*k/2)
        predicted=reference.copy();predicted[:,:,0,1]=np.exp(1j*np.pi*k/4)
        result=secondary_metrics({"channel":predicted},reference)
        np.testing.assert_allclose(result["frequency_correlation_absolute_error"][...,0],np.cos(np.pi/8)-np.cos(np.pi/4),atol=1e-14)

    def test_nonfinite_output_is_rejected(self):
        _,q=fixture();p=np.ones((1,2,32));p[0,0,0]=np.nan
        with self.assertRaises(ValueError):check_prediction({"power":p},q)

class AdapterChecks(unittest.TestCase):
    def test_kernel_normalizer_and_output(self):
        bank,q=fixture(n=8)
        r=RBFPower(bank).predict(q);self.assertEqual(r["power"].shape,(1,2,32))
        self.assertTrue(np.isfinite(r["power"]).all());self.assertTrue((r["power"]>=0).all())
        np.testing.assert_array_equal(r["source_ids"],bank.row_ids)

    def test_position_mlp_fit_only_buffers_and_finite_gradient(self):
        bank,q=fixture(n=8);m=PositionPowerMLP(bank.position)
        np.testing.assert_allclose(m.center.numpy(),bank.position.mean(0),atol=1e-7)
        pred=m(torch.tensor(q.position,dtype=torch.float32))
        loss=training_power_loss(pred,torch.ones_like(pred));loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_seventeen_experts_and_own_row_exclusion(self):
        bank,q=fixture();model=CarrierBank(bank)
        out=model.experts(q);self.assertEqual(out.shape,(1,17,2,8,4,64))
        own=Query(np.array([0]),bank.position[:1],bank.spec)
        value=model.experts(own)[0,0]
        neighbor=np.linalg.norm(bank.position[1:,:2]-bank.position[0,:2],axis=-1).argmin()+1
        np.testing.assert_array_equal(value,bank.channel[neighbor])

    def test_radial_experts_use_actual_frequency_axis(self):
        bank,q=fixture()
        h=np.empty_like(bank.channel)
        for p in range(2):
            radius=np.linalg.norm(bank.position-bank.spec.center[p],axis=1)
            h[:,p]=np.exp(-2j*np.pi*radius[:,None,None,None]*bank.spec.frequencies[None,None,None,:]/299792458.)
        measured=TrainingBank(bank.row_ids,bank.position,h,bank.spec)
        out=CarrierBank(measured).experts(q)
        for p in range(2):
            radius=np.linalg.norm(q.position[0]-bank.spec.center[p])
            expected=np.exp(-2j*np.pi*radius*bank.spec.frequencies/299792458.)
            for e in range(5,13):np.testing.assert_allclose(out[0,e,p,0,0],expected,atol=3e-6)

    def test_direct_pointing_uses_same_codebook(self):
        bank,q=fixture();r=DirectPointing().predict(q);self.assertEqual(r["power"].shape,(1,2,32))
        self.assertTrue((r["power"]>=0).all())

    def test_output_row_identity_and_permutation_are_preserved(self):
        bank,q=fixture(n=8)
        query=Query(np.array([1000,1001]),np.r_[q.position,q.position+np.array([[.1,.2,0]])],bank.spec)
        reversed_query=Query(query.row_ids[::-1],query.position[::-1],bank.spec)
        for model in [RBFPower(bank),PositionPowerMLP(bank.position),DirectPointing()]:
            first=model.predict(query);second=model.predict(reversed_query)
            np.testing.assert_array_equal(first["row_ids"],query.row_ids)
            np.testing.assert_allclose(first["power"],second["power"][::-1],rtol=1e-6,atol=1e-7)
        with self.assertRaises(ValueError):check_prediction(dict(power=np.ones((2,2,32)),row_ids=query.row_ids[::-1]),query)

    def test_blend_zero_is_identity(self):
        bank,_=fixture(n=1);base=torch.tensor(bank.channel.reshape(2,1,8,4,64))
        out=synthesize(base,torch.ones((2,8,4))/32,torch.ones((2,64))/64,blend=0)
        self.assertTrue(torch.equal(base,out))

    def test_core_inherits_original_encoder_and_has_no_fake_UE_head(self):
        from .core import ComponentStateModel,FINAL_SOURCE
        self.assertTrue(issubclass(HallPathState,ComponentStateModel))
        self.assertEqual(Path(inspect.getfile(ComponentStateModel)).resolve(),(FINAL_SOURCE/"component_state/model.py").resolve())

    def test_core_power_preservation_padding_and_finite_backward(self):
        bank,_=fixture(n=1);support=support_fixture()
        batch=batch_from_support(bank.channel,support);model=MeasuredV16().eval()
        model.set_fit_statistics(batch)
        out,state=model(batch)
        before=torch.tensor(bank.channel.reshape(2,1,8,4,64)).abs().square().sum((-3,-2,-1))
        after=out.abs().square().sum((-3,-2,-1))
        torch.testing.assert_close(before,after,rtol=2e-6,atol=1e-6)
        target=torch.arange(1,33).expand(2,1,32).float()
        loss=training_power_loss(torch_directional_power(out),target);loss.backward()
        active=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(active);self.assertTrue(all(torch.isfinite(g).all() for g in active))
        changed={k:v.clone() for k,v in batch.items()};mask=~batch["mask"]
        changed["base"][mask]=1e6;changed["sequence_geometry"][mask]=1e6
        with torch.no_grad():other,_=model(changed)
        torch.testing.assert_close(out.detach(),other,rtol=2e-6,atol=2e-6)

    def test_no_geometry_falls_back_exactly_to_foundation(self):
        bank,_=fixture(n=1);s=support_fixture();s["mask"][:]=False;s["mass"][:]=0
        batch=batch_from_support(bank.channel,s);m=MeasuredV16().eval()
        with torch.no_grad():value,_=m(batch)
        self.assertTrue(torch.equal(value,batch["channel"]))

class WRFChecks(unittest.TestCase):
    def test_two_gaussian_alpha_blend_literal(self):
        pixel=torch.zeros((2,2),dtype=torch.float64);cov=torch.eye(2,dtype=torch.float64).repeat(2,1,1)
        out=alpha_render(pixel,cov,torch.tensor([1.,2.]),torch.tensor([.2,.4]),torch.tensor([[.3],[.9]]),1,1)
        self.assertAlmostEqual(float(out[0,0,0]),.2*.3+.8*.4*.9,places=7)

    def test_alpha_termination_does_not_resume_on_a_later_weak_gaussian(self):
        pixel=torch.zeros((4,2),dtype=torch.float64);cov=torch.eye(2,dtype=torch.float64).repeat(4,1,1)
        depth=torch.arange(1,5,dtype=torch.float64)
        opacity=torch.tensor([.99,.5,.99,.1],dtype=torch.float64)
        color=torch.tensor([[.1],[.2],[.7],[.9]],dtype=torch.float64)
        out=alpha_render(pixel,cov,depth,opacity,color,1,1)
        self.assertAlmostEqual(float(out[0,0,0]),.99*.1+.01*.5*.2,places=12)

    def test_projection_covariance_matches_independent_autograd_jacobian(self):
        x=torch.tensor([[2.,1.,3.]],dtype=torch.float64,requires_grad=True);cov=torch.diag_embed(torch.tensor([[.2,.3,.4]],dtype=torch.float64))
        center=torch.zeros(3,dtype=torch.float64);rotation=torch.eye(3,dtype=torch.float64)
        _,sigma,_=project_gaussians(x,cov,center,rotation,32,16)
        def project(z):
            return torch.stack([(torch.atan2(z[0],z[2])/torch.pi+1)*16-.5,
                                (torch.asin(z[1]/torch.linalg.vector_norm(z))*2/torch.pi+1)*8-.5])
        j=torch.autograd.functional.jacobian(project,x[0])
        torch.testing.assert_close(sigma[0],j@cov[0]@j.T+.3*torch.eye(2),rtol=1e-6,atol=1e-7)

    def test_beam_response_uses_full_spherical_grid_and_physical_array(self):
        bank,_=fixture();_,r=angular_grid(bank.spec,0,32,16)
        lat=np.pi*((np.arange(16)+.5)/16-.5)
        expected=np.repeat(32*np.cos(lat)*(2*np.pi/32)*(np.pi/16),32)
        np.testing.assert_allclose(r.sum(-1),expected,rtol=2e-7,atol=1e-8)

    def test_analytic_band_average_matches_literal_all_frequency_operator(self):
        bank,_=fixture()
        width,height=8,4
        camera,operator=angular_grid(bank.spec,0,width,height)
        lon=2*np.pi*((np.arange(width)+.5)/width-.5)
        lat=np.pi*((np.arange(height)+.5)/height-.5)
        directions=np.stack([np.sin(lon)*np.cos(lat[1]),np.full(width,np.sin(lat[1])),np.cos(lon)*np.cos(lat[1])],-1)@camera.T
        solid=np.cos(lat[1])*2*np.pi/width*np.pi/height
        for col in [1,3,7]:
            phase=2*np.pi/299792458.*np.einsum("i,hvi,f->hvf",directions[col],bank.spec.element_offsets[0],bank.spec.frequencies)
            h=np.exp(1j*phase).reshape(32,-1)
            literal=(np.abs(codebook().conj().T@h)**2).mean(-1)*solid
            np.testing.assert_allclose(operator[width+col],literal,rtol=3e-6,atol=1e-8)

    def test_author_network_and_scalar_power_adapter_finite_backward(self):
        from pathlib import Path
        from .wrf import AUTHOR
        torch.manual_seed(23);bank,q=fixture(n=8)
        xyz=np.random.default_rng(5).uniform([-2,-2,.2],[2,2,2.5],(8,3))
        model=WRFGSPlusPower(bank.position,bank.spec,xyz,width=24,height=12)
        self.assertEqual(Path(inspect.getfile(author_network_class())).resolve(),(AUTHOR/"utils/pos_utils.py").resolve())
        value=model(torch.tensor(q.position,dtype=torch.float32))
        self.assertEqual(tuple(value.shape),(1,2,32));self.assertTrue(torch.isfinite(value).all())
        loss=training_power_loss(value,torch.arange(1,33).expand(1,2,32).float());loss.backward()
        active=[p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in active))
        self.assertTrue(any(float(g.abs().max())>0 for g in active))
        for field in model.fields:
            self.assertTrue(all(not p.requires_grad for p in field.deformation.gaussian_phase.parameters()))

class DataChecks(unittest.TestCase):
    def test_all_five_prepared_splits_independent_pairwise_check(self):
        d=SouthDataset();coverage=np.zeros(627,int)
        for outer in range(5):
            parts={r:d.partition(outer,r) for r in ["fit","validation","test"]}
            for a,b in [("fit","validation"),("fit","test"),("validation","test")]:
                self.assertFalse(set(parts[a])&set(parts[b]))
                pa,pb=d.meta["position"][parts[a],:2],d.meta["position"][parts[b],:2]
                self.assertGreaterEqual(float(np.sqrt(((pa[:,None]-pb[None])**2).sum(-1)).min()),.30)
            coverage[parts["test"]]+=1
        np.testing.assert_array_equal(coverage,np.ones(627,int))

    def test_north_view_is_rejected_before_any_data_load(self):
        with self.assertRaises(ValueError):SouthDataset(Path("artifacts/measured/hall_nw_publisher_v2"))

    def test_query_contains_no_CSI_and_held_out_reads_rejected(self):
        d=SouthDataset();q=d.query(d.partition(0,"test")[:1])
        self.assertEqual(set(q.__dict__),{"row_ids","position","spec"})
        for role in ["validation","test"]:
            with self.assertRaises(ValueError):d.read_fit(0,d.partition(0,role)[:1])
        self.assertEqual(d.channel_reads,[])

if __name__=="__main__":
    from pathlib import Path
    torch.set_num_threads(1)
    unittest.main(verbosity=2)

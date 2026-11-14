"""CPU fixture checks only. No fitted-model prediction on actual query features."""
from pathlib import Path
import types
import unittest

import numpy as np

import v16_recovery as recovery
import v16_physical_inference as bridge


class FixtureStatistics:
    def __init__(self):
        self.ids=np.arange(2,dtype=np.int64)
        self.site=np.array([0,1])
        self.arrays={"geometry":np.zeros((2,20),np.float32)}
        rng=np.random.default_rng(607)
        for s in ("base","scatter"):
            for domain,shape,countshape in (
                ("pas",(2,2,16,8,4,192),(2,4,192)),
                ("pdp",(2,256,4,192),(2,256,4)),
                ("ad",(2,16,8,192),None)):
                a=rng.uniform(.1,1,size=shape).astype(np.float32)
                self.arrays[f"{s}.{domain}.mean"]=a
                if domain!="ad":
                    self.arrays[f"{s}.{domain}.m2"]=a*.02
                    self.arrays[f"{s}.{domain}.count"]=np.full(countshape,24,np.uint8)
    def get(self,s,d,k):return self.arrays[f"{s}.{d}.{k}"]
    def scalar(self,s):return np.zeros((2,6),np.float32)
    def reliability(self,d):return np.zeros((2,11),np.float32)


class RecoveryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch=recovery.get_torch("cpu")
        cls.torch.set_num_threads(2)
        cls.fixture=FixtureStatistics()

    def test_pre_first_batch_failure_resumes_with_identity(self):
        import tempfile
        original=recovery.ARTIFACT_ROOT
        parent=original/"synthetic_fixtures"
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="resume_",dir=parent) as temp:
            recovery.ARTIFACT_ROOT=Path(temp)
            try:
                def fail(output,start,commit):
                    self.assertEqual(start,0)
                    raise RuntimeError("synthetic pre-first-batch failure")
                with self.assertRaisesRegex(RuntimeError,"synthetic"):
                    recovery.save_stage("fixture",dict(kind="synthetic"),(2,3),np.float32,fail)
                state=Path(temp)/"historical_replay/fixture/state.json"
                self.assertTrue(state.exists())
                def finish(output,start,commit):
                    self.assertEqual(start,0)
                    output[:]=np.arange(6,dtype=np.float32).reshape(2,3)
                    commit(2)
                result,record=recovery.save_stage("fixture",dict(kind="synthetic"),(2,3),np.float32,finish,True)
                self.assertTrue(np.array_equal(result,np.arange(6,dtype=np.float32).reshape(2,3)))
                def forbidden(*args):raise AssertionError("complete stage reran")
                _,again=recovery.save_stage("fixture",dict(kind="synthetic"),(2,3),np.float32,forbidden,True)
                self.assertEqual(record["sha256"],again["sha256"])
                with self.assertRaises(RuntimeError):
                    recovery.save_stage("fixture",dict(kind="changed"),(2,3),np.float32,forbidden,True)
            finally:recovery.ARTIFACT_ROOT=original

    def test_source_operators(self):
        self.assertEqual(recovery.synthetic_checks()["status"],"passed")

    def test_all_four_bank_paths_without_truth(self):
        torch=self.torch
        old=bridge.frozen_definitions(torch)
        for kind in ("spectral","joint"):
            for domain in ("pas","pdp"):
                bank=(bridge.old_bank(self.fixture,domain,"cpu",torch,old) if kind=="spectral"
                    else bridge.joint_bank(self.fixture,domain,"cpu",torch))
                mode="mean" if kind=="spectral" else "joint"
                context=bank.point_context(mode)
                point=torch.tensor([0,1])
                slices=torch.tensor([0,117])
                mean=torch.zeros(context.shape[1]);std=torch.ones_like(mean)
                if kind=="spectral":
                    x,c,b,target=bank.sample(point,slices,mode,mean,std)
                    self.assertTrue(torch.count_nonzero(target)==0)
                else:x,c,b=bank.sample_input(point,slices,mode,mean,std)
                self.assertTrue(torch.isfinite(x).all())
                cfg=old.TrainConfig(hidden=8,context_channels=4,blocks=1)
                model=old.SpectralCalibrator(domain,x.shape[1],c.shape[1],cfg)
                y,delta=model(x,c,b)
                self.assertTrue(torch.isfinite(y).all())
                self.assertTrue(torch.allclose(y.reshape(2,-1).sum(1),torch.ones(2),atol=2e-6))
                y.square().sum().backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_existing_20_checkpoint_architectures_metadata_only(self):
        torch=self.torch;old=bridge.frozen_definitions(torch)
        for kind in ("spectral","joint"):
            for domain in ("pas","pdp"):
                for fold in range(5):
                    path=recovery.BUNDLE/"artifacts"/"round2"/"physical_ai_submission"/"models"/"base_field"/kind/f"{domain}_fold_{fold}.pt"
                    ck=recovery.safe_checkpoint(path,torch)
                    cfg=old.TrainConfig(**ck["config"])
                    self.assertEqual(len(ck["context_mean"]),37+(2 if kind=="spectral" else 23))
                    input_channels=ck["state_dict"]["stem.weight"].shape[1]-cfg.context_channels-(4 if domain=="pas" else 6)
                    slice_size=(11 if domain=="pas" else 12) if kind=="spectral" else (13 if domain=="pas" else 12)
                    model=old.SpectralCalibrator(domain,input_channels,len(ck["context_mean"])+slice_size,cfg)
                    model.load_state_dict(ck["state_dict"])
                    self.assertEqual(ck["fold"],fold)
                    # No forward of these retained fitted networks.
                    del model

    def test_strict_ridge_solver_and_transitive_denial(self):
        import v16_strict_fitting as strict
        ns=recovery.source_namespace("analyze_v7_complex_carrier_stacking.py",["fit_weights"],dict(),())
        rng=np.random.default_rng(45)
        coef=(rng.normal(size=(2,8,3))+1j*rng.normal(size=(2,8,3))).astype(np.complex64)
        labels=np.array([0,1,0,1])
        gram=np.broadcast_to(np.eye(3,dtype=np.complex64),(4,8,3,3)).copy()
        cross=coef[labels]
        result=ns.fit_weights(dict(gram=gram,cross=cross),np.arange(4),labels,
            types.SimpleNamespace(scope="site",ports="port8",ridge=0.,sum_penalty=0.))
        self.assertTrue(np.array_equal(result[0],coef[0]))
        self.assertTrue(np.array_equal(result[1],coef[1]))
        outer=0;role="refit";roles=recovery.strict_roles(outer)
        record=dict(schema="v16-strict-carrier-features-v1",outer=outer,role=role,
            protocol_sha256=recovery.sha256(recovery.FOLDS),allowed_label_ids=roles[role].tolist(),
            label_sensitive_dependencies=dict(donor=dict(label_ids=roles["outer"][:1].tolist())))
        with self.assertRaisesRegex(ValueError,"Forbidden transitive"):
            strict.check_feature_provenance(record,outer,role)

    def test_unproved_and_held_label_reuse_denied(self):
        for outer in range(5):
            roles=recovery.strict_roles(outer)
            capability=recovery.FitOnlyLabels(roles["screen_fit"])
            with self.assertRaises(PermissionError):
                capability.read(roles["outer"][:1])
            with self.assertRaises(ValueError):
                recovery.validate_reusable_artifact({},outer,"refit","donor_network")
            proof=dict(schema="v16-strict-component-v1",outer=outer,role="refit",
                component="donor_network",protocol_sha256=recovery.sha256(recovery.FOLDS),
                fit_ids=roles["refit"].tolist(),
                label_ids=dict(training=roles["outer"][:1].tolist(),normalization=[],basis=[],calibration=[],policy=[]))
            with self.assertRaisesRegex(ValueError,"forbidden"):
                recovery.validate_reusable_artifact(proof,outer,"refit","donor_network")


if __name__=="__main__":unittest.main(verbosity=2)

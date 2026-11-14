"""CPU tests of the regenerated-query feature/PathState bridge; no quality score."""
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np
import v16_recovery as r
from v16_production_pathstate import model_core
from v16_live_tokens import frozen_aggregation


class QueryForwardChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch=r.get_torch("cpu");cls.torch.set_num_threads(2)

    def test_production5_architecture_only(self):
        torch=self.torch;ns=model_core(torch)
        for fold in range(5):
            p=r.BUNDLE/"artifacts/round2/physical_ai_submission/models/component_calibrator"/f"fold_{fold}.pt"
            ck=r.safe_checkpoint(p,torch)
            cfg=ns.ModelConfig.from_checkpoint(ck["config"])
            model=ns.ComponentStateModel(120,21,cfg)
            model.load_state_dict(ck["state_dict"])
            self.assertEqual(ck["outer"],fold)
            del model

    def test_synthetic_new_mapping_batch_forward_gradient(self):
        torch=self.torch;ns=model_core(torch);rng=np.random.default_rng(62)
        features=rng.normal(size=(2,128,120)).astype(np.float32)
        aux=dict(fraction=np.full((2,128),1/128,np.float32),presence=np.full((2,128),24,np.float32),
            port=np.full((2,128,2,4),1/8,np.float32),geometry=rng.normal(size=(2,128,16,21)).astype(np.float32),
            depth=np.full((2,128),16,np.int64),interactions=np.zeros((2,128,16),np.int64),
            angles=np.full((2,128,4),.8,np.float32),coordinate=np.full((2,128,3),7,np.float32))
        coord=ns._mapped_coordinates(aux,np.array([0,1]),np.zeros(10,np.float32),torch.device("cpu"))
        shifted=ns._mapped_coordinates(aux,np.array([0,1]),np.ones(10,np.float32)*.2,torch.device("cpu"))
        self.assertFalse(np.array_equal(coord[:,:,:2],shifted[:,:,:2]))
        self.assertTrue(np.array_equal(coord[:,:,2],aux["coordinate"][:,:,2]))
        batch=ns._batch(np.array([0,1]),features,aux,coord,np.zeros(120,np.float32),np.ones(120,np.float32),24,torch.device("cpu"))
        cfg=ns.ModelConfig(hidden=16,heads=4,sequence_layers=1,set_layers=1,feedforward=32,dropout=0.)
        model=ns.ComponentStateModel(120,21,cfg);out=model(batch)
        self.assertEqual(tuple(out["mass"].shape),(2,128))
        self.assertTrue(torch.allclose(out["mass"].sum(1),torch.ones(2),atol=2e-6))
        self.assertTrue(torch.allclose(out["port"].sum((2,3)),torch.ones((2,128)),atol=2e-6))
        sum(x.square().mean() for x in out.values()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_legacy120_equals_packaged120_on_native_fixture(self):
        rng=np.random.default_rng(17);agg=frozen_aggregation()
        cfg=types.SimpleNamespace(components_per_point=128,interaction_steps=16,subcarrier_spacing_hz=30000.,
            subcarriers=192,tx_horizontal=16,tx_vertical=8)
        tokens=agg._empty_arrays(2,cfg)
        tokens["mask"][:]=True;tokens["presence"][:]=rng.integers(1,25,size=(2,128))
        tokens["tau"][:]=rng.uniform(1e-8,2e-6,size=(2,128))
        tokens["angles"][:]=rng.uniform(-1,1,size=(2,128,4))
        tokens["energy"][:]=rng.uniform(1e-10,1e-8,size=(2,128))
        tokens["energy_fraction"][:]=rng.uniform(1e-5,.1,size=(2,128))
        tokens["source"][:]=rng.integers(0,2,size=(2,128))
        tokens["coefficient"][:]=(rng.normal(size=(2,128,256,4))+1j*rng.normal(size=(2,128,256,4))).astype(np.complex64)*1e-6
        positions=rng.normal(size=(6,3));sites=np.array([0,1,0,1,0,1]);ids=np.array([1,4]);regime=np.array([1,2],np.int8)
        setup=types.SimpleNamespace(bs_positions=np.array([[2.,3.,4.],[-2.,0.,5.]]))
        loader=lambda:(setup,positions,positions[:2],sites,sites[:2])
        root=r.ARTIFACT_ROOT/"synthetic_fixtures";root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="features_",dir=root) as temp:
            temp=Path(temp)
            for k,v in tokens.items():np.save(temp/f"{k}.npy",v)
            old=r.source_namespace("train_explicit_path_pdp_predictor.py",["build_features"],
                dict(PATH_CACHE=temp,load_positions=loader))
            legacy=old.build_features(np.arange(2),ids,regime,temp/"legacy_features.npy",path_cache=temp)
            new=r.source_namespace("../../final_code/V16_reorganized_submission/code/round2/physical_ai_submission/scene_prior/component_features.py",
                ["_component_features"],dict(),["TINY"])
            actual=new._component_features(tokens,positions[ids],sites[ids],positions,sites,setup.bs_positions,regime,cfg)
            self.assertTrue(np.array_equal(actual,legacy))
            self.assertEqual(actual.shape,(2,128,120))


if __name__=="__main__":unittest.main(verbosity=2)

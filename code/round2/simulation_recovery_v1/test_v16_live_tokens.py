"""Lossless live-token observer CPU fixtures; no RT/model/CSI scoring."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import v16_recovery as recovery
from v16_live_tokens import LiveTokenObserver


class Wrapped:
    def __init__(self,a):self.a=a
    def numpy(self):return self.a


def fake_components(paths=110):
    rng=np.random.default_rng(97)
    a=(rng.normal(size=(1,4,1,256,paths))+1j*rng.normal(size=(1,4,1,256,paths))).astype(np.complex64)
    class C:pass
    c=C()
    for k,v in dict(a=a,valid=np.ones((1,1,paths),bool),
        tau=np.linspace(1e-8,8e-7,paths,dtype=np.float32)[None,None],
        theta_t=np.full((1,1,paths),1.2,np.float32),phi_t=np.full((1,1,paths),.8,np.float32),
        theta_r=np.full((1,1,paths),1.7,np.float32),phi_r=np.full((1,1,paths),2.1,np.float32),
        interactions=np.zeros((16,1,1,paths),np.uint32),
        objects=np.zeros((16,1,1,paths),np.uint32),
        primitives=np.zeros((16,1,1,paths),np.uint32),
        vertices=np.zeros((16,1,1,paths,3),np.float32)).items():setattr(c,k,Wrapped(v))
    return c


class LiveChecks(unittest.TestCase):
    def test_live_native_candidates_equal_frozen_reader_and_replace(self):
        parent=recovery.ARTIFACT_ROOT/"synthetic_fixtures";parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="live_",dir=parent) as temp:
            obs=LiveTokenObserver(Path(temp),np.array([7]),np.array([0]),dict(kind="CPU_synthetic"))
            comp=fake_components();raw=obs.extract.extract_component_support(comp,np.array([7]),42)
            proof=obs.consume_live(comp,np.array([7]),"base",42,0)
            self.assertTrue(obs.has_batch(np.array([7]),"base",42,0))
            with np.load(recovery.PATHS.resolve(proof["path"]),allow_pickle=False) as z:
                saved={k:np.asarray(z[k]) for k in z.files}
            expected=obs.agg._records_for_point(raw,0,96,0)
            actual=obs.agg._records_for_point(saved,0,96,0)
            self.assertEqual(len(expected),96)
            for a,b in zip(expected,actual):
                for field in ["tau","energy","energy_fraction","seed","interaction_count"]:
                    self.assertEqual(getattr(a,field),getattr(b,field))
                for field in ["coefficient","angles","vertices","interactions","objects","primitives"]:
                    self.assertTrue(np.array_equal(getattr(a,field),getattr(b,field)),field)
            with self.assertRaises(FileExistsError):obs.consume_live(comp,np.array([7]),"base",42,0)
            obs.consume_live(comp,np.array([7]),"base",42,0,replace_uncommitted=True)
            self.assertTrue(obs.has_batch(np.array([7]),"base",42,0))
            # Scatter uses64, without any degree/radian or coefficient guessing.
            second=obs.consume_live(comp,np.array([7]),"scatter",42,0)
            self.assertEqual(second["count"],64)

    def test_all_seed_presence_coverage_and_frozen_final_layout(self):
        parent=recovery.ARTIFACT_ROOT/"synthetic_fixtures";parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="aggregate_",dir=parent) as temp:
            obs=LiveTokenObserver(Path(temp),np.array([7]),np.array([0]),dict(kind="CPU_synthetic"))
            comp=fake_components(2)
            with self.assertRaisesRegex(ValueError,"Missing"):obs.finalize()
            for source in ["base","scatter"]:
                for seed in recovery.ORIGINAL_SEEDS:
                    obs.consume_live(comp,np.array([7]),source,seed,0)
            report=obs.finalize()
            # Crash after array publication but before the final manifest.
            (Path(temp)/"tokens/manifest.json").unlink()
            recovered=obs.finalize()
            self.assertEqual(report["arrays"],recovered["arrays"])
            self.assertEqual(recovered["storage_mode"],"uncompressed_npz_lossless_full1024")
            self.assertEqual(recovered["arrays"],obs.finalize()["arrays"])
            arrays={k:np.load(recovery.PATHS.resolve(v["path"]),allow_pickle=False) for k,v in report["arrays"].items()}
            self.assertEqual(arrays["coefficient"].shape,(1,128,256,4))
            self.assertEqual(int(arrays["mask"].sum()),50)
            self.assertTrue(np.all(arrays["presence"][(arrays["source"]==0)&arrays["mask"]]==24))
            self.assertTrue(np.all(arrays["presence"][(arrays["source"]==1)&arrays["mask"]]==1))
            self.assertTrue(np.isfinite(arrays["coefficient"]).all())
            with self.assertRaises(RuntimeError):
                obs.consume_live(comp,np.array([7]),"base",42,0,replace_uncommitted=True)


if __name__=="__main__":unittest.main(verbosity=2)

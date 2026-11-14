"""CPU replay of actual training block and validation-submatrix indexing, no updates."""
import ast,inspect,json,time,types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from round2.paper_baseline_training_v1 import nerf2_runner_v3 as v3,nerf2_runner_v2 as v2
from round2.paper_baseline_training_v1.field_fast import enable_shared_medium
from round2.paper_baseline_training_v1.common import FoldData,atomic_json,sha256
from round2.paper_baseline_adapters_v1.field_models import CompetitionNeRF2


def main():
    started=time.monotonic();torch.set_num_threads(8)
    d=FoldData(0,"screen","nerf2");p=v3.paths(0,"health")["artifact"]/"health.pt"
    ck=torch.load(p,map_location="cpu",weights_only=False);fixture=np.asarray(ck["info"]["FIT_ids"],np.int64)
    cfg={**v2.CONFIG,"world_center_m":tuple(np.mean(d.positions(d.fit_ids),axis=0))}
    m=CompetitionNeRF2(d.geometry,field_config=cfg)
    enable_shared_medium(m,medium_ray_chunk=324,tx_chunk=32,factor_transport=True)
    m.register_parameter("log_output_gain",nn.Parameter(torch.zeros(())))
    for core in m.fields.values():
        old=core.field.attenuation_output;core.field.attenuation_output=v3.PerMetreExtinction(old.in_features,old.out_features)
        core.field.embed_pts_fn=v3.condition_embedding(core.field.embed_pts_fn)
        core.field.embed_tx_fn=v3.condition_embedding(core.field.embed_tx_fn)
    m.load_state_dict(ck["model"]);m.eval();log={};original={}
    for key,core in m.fields.items():
        original[key]=core.forward
        def trace(self,tx,rx,freq,*,_key=key):
            value=original[_key](tx,rx,freq)
            log.update(key=_key,tx=tx.clone(),rx=rx.clone(),frequency=freq.clone(),value=value.clone())
            return value
        core.forward=types.MethodType(trace,core)
    class Cache:
        def read(self,rows,ff):
            log["rows"]=np.asarray(rows).copy();log["ff"]=np.asarray(ff).copy()
            # Exact CPU equivalent of the actual GPU cache: normalize allF, then indexF.
            value=torch.tensor(np.take(d.read_fit(rows),ff,axis=-1));log["cache_value"]=value.clone();return value
    tree=ast.parse(inspect.getsource(v3.fixture_step))
    class CPU(ast.NodeTransformer):
        def visit_Constant(self,n):
            return ast.copy_location(ast.Constant("cpu"),n) if n.value=="cuda:0" else n
    tree=CPU().visit(tree);ast.fix_missing_locations(tree);ns={**v3.__dict__};exec(compile(tree,"device-only CPU replay of fixture_step","exec"),ns)
    with torch.no_grad():loss,detail=ns["fixture_step"](m,d,Cache(),np.random.default_rng(42),fixture)
    for key,core in m.fields.items():core.forward=original[key]
    key=log["key"];b=int(key.split("_b")[1][0]);u=int(key.split("_u")[1][0]);ff=log["ff"]
    freq_slots=np.array([int(np.flatnonzero(v2.FREQUENCY_IDS==f)[0]) for f in ff])
    with torch.no_grad():
        short=log["value"].permute(1,0,2,3)*m.log_output_gain.exp()
        full=m.fields[key](log["tx"],log["rx"],torch.tensor(d.frequencies(v2.FREQUENCY_IDS),dtype=torch.float64)).permute(1,0,2,3)*m.log_output_gain.exp()
        sliced=full[...,freq_slots]
        actual_target=log["cache_value"][:,b*128+torch.tensor(v2.BS_SPATIAL),u*2:(u+1)*2]
        evaluation_target=torch.tensor(d.read_fit(log["rows"],v2.FREQUENCY_IDS)[:,v2.BS_PORTS])[:,b*8:(b+1)*8,u*2:(u+1)*2][...,freq_slots]
        relative=float(torch.linalg.vector_norm(short-sliced)/torch.linalg.vector_norm(short).clamp_min(1e-30))
        torch.testing.assert_close(actual_target,evaluation_target,rtol=0,atol=0)
        torch.testing.assert_close(short,sliced,rtol=1e-4,atol=1e-5)
        replay=.5*(short-actual_target).abs().square().mean()
        torch.testing.assert_close(loss,replay,rtol=1e-6,atol=1e-7)
    out=dict(status="passed",checkpoint_sha256=sha256(p),FIT_ids=log["rows"].tolist(),core=key,
      actual_BS_ports=(b*128+v2.BS_SPATIAL).tolist(),actual_UE_ports=list(range(u*2,(u+1)*2)),
      actual_frequency_ids=ff.tolist(),actual_frequencies_Hz=log["frequency"].tolist(),
      normalized_target_max_abs_difference=float((actual_target-evaluation_target).abs().max()),
      training4tone_vs_evaluation12tone_sliced_prediction_relative_L2=relative,
      loss=float(loss),reconstructed_loss=float(replay),normalization_FIT_RMS=d.scale,
      source_replay="original fixture_step AST with CUDA device literals changed to CPU only",
      parameter_updates=False,outer_truth_read=False,seconds=time.monotonic()-started)
    atomic_json(v3.paths()["report"]/"training_eval_subset_check.json",out);print(json.dumps(out),flush=True)

if __name__=="__main__":main()

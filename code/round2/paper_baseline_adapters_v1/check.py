"""CPU synthetic correctness only; never benchmark model quality."""
from __future__ import annotations
import hashlib,json,os,sys,time,platform
import numpy as np
import torch
from project_paths import PATHS
from common.official_metrics import ArrayLayout,pas_power,pdp_power
from .contract import *
from .evaluation import pas_from_power,eligibility,evaluate_output,reject_online_test_as_oof
from .registry import SimulationAdapter,make_field_model,make_wrf_model,make_newrf_model
CHECKS=[]
INTEGRITY_ONLY=False
def passed(name,details=None):
    CHECKS.append(dict(name=name,passed=True,details=details));print("PASS",name,flush=True)
def rejects(fn,exceptions=(ValueError,TypeError,PermissionError,RuntimeError)):
    try:fn()
    except exceptions:return
    raise AssertionError("Illegal input accepted")
def close(a,b,atol=1e-6,rtol=1e-5):
    np.testing.assert_allclose(np.asarray(a),np.asarray(b),atol=atol,rtol=rtol)
def digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def dataset_checks():
    src=SimulationInputs();g=src.geometry
    count=[];roles={}
    for outer in range(5):
        r=src.roles(outer);count.append(len(r["outer"]))
        for name,ids in r.items():
            assert ids.ndim==1 and len(np.unique(ids))==len(ids) and np.all((ids>=0)&(ids<4000))
        roles[str(outer)]={k:v.tolist() for k,v in r.items()}
        rejects(lambda:src.read_fit(outer,r["outer"][:1]),PermissionError)
        rejects(lambda:src.read_fit(outer,r["validation"][:1]),PermissionError)
    assert count==[103,71,63,110,153]
    canonical=hashlib.sha256(np.ascontiguousarray(src.query_ids).tobytes()).hexdigest()
    assert canonical=="f115290de0dc5f80cfe304e1f2143696694e66eec10c418e1b086d0b179ed08b"
    rejects(lambda:src.reference(0,"outer"),PermissionError)
    rejects(lambda:src.query(0,"official_test"))
    q=src.query(0,"validation",src.roles(0)["validation"][:1])
    assert set(vars(q))=={"row_ids","ue_position","site_id","input_identity","input_namespace"}
    rejects(lambda:PositionQuery(q.row_ids,q.ue_position,q.site_id,src.identity,"official_test_position_order"))
    # Only two approved screenFIT signals, no validation/outer/test truth.
    rows=src.roles(0)["screen_fit"][:2];fit=src.read_fit(0,rows)
    assert fit.shape==(2,256,4,192) and fit.dtype==np.complex64 and np.isfinite(fit).all()
    passed("canonical500_nested_roles_and_fit_only_data",dict(canonical_ids_sha256=canonical,outer_counts=count,
       sampled_fit_rows=rows.tolist(),sampled_fit_shape=list(fit.shape),heldout_labels_read=False))
    close(g.frequencies_hz,6690000000.+(np.arange(192)-96)*30000.,atol=0,rtol=0)
    assert g.bs_port_positions_m.shape==(2,2,16,8,3)
    close(g.bs_port_positions_m[:,0],g.bs_port_positions_m[:,1],atol=0,rtol=0)
    close(g.ue_port_offsets_m[:,0],g.ue_port_offsets_m[:,1],atol=0,rtol=0)
    close(np.linalg.det(g.bs_rotations),1.,atol=1e-14,rtol=1e-14)
    # Independent NumPy UE look-at: right/up/normal basis and local z offsets.
    x=torch.tensor([[.213,-.419,1.237],[1.731,.583,2.117]],dtype=torch.float64)
    sid=torch.tensor([0,1]);world=g.ue_world_positions(x,sid).numpy()
    for b in range(2):
        d=g.bs_positions_m[b]-x[b].numpy();d=d/np.linalg.norm(d)
        horizontal=np.cross([0.,0.,1.],d);horizontal/=np.linalg.norm(horizontal)
        up=np.cross(d,horizontal)
        expected=np.stack([x[b].numpy()+up*z for z in g.ue_port_offsets_m[b,0,0,:,2]])
        close(world[b,:2],expected,atol=1e-13,rtol=1e-13)
        close(world[b,:2],world[b,2:],atol=0,rtol=0)
    assert not np.allclose(world[0,0]-x[0].numpy(),world[1,0]-x[1].numpy())
    passed("V16_frequency_array_and_dynamic_UE_geometry",g.metadata())
    save_json(REPORT/"data_contract.json",dict(identity=src.identity,canonical_ids_sha256=canonical,roles=roles,
      geometry=g.metadata(),source_files={PATHS.relative(p):sha256(p) for p in [src.setup_path,FOLDS,META]},
      input_information=["FIT CSI only for training/scaling","query position","known site/Setup","known independent geometry"],
      label_access=src.access,official_test_labels_available=False))
    return src

def operator_checks(src):
    rng=np.random.default_rng(31)
    h=(rng.normal(size=(1,256,4,192))+1j*rng.normal(size=(1,256,4,192))).astype(np.complex64)
    actual=pas_power(h)
    shaped=h.astype(np.complex128).reshape(1,2,16,8,4,192)
    a=np.arange(16);b=np.arange(8)
    dft_h=np.exp(-2j*np.pi*a[:,None]*a[None,:]/16)
    dft_v=np.exp(-2j*np.pi*b[:,None]*b[None,:]/8)
    select=shaped[..., [0,96,191]]
    matrix=np.einsum("uh,wv,nphvlf->npuwlf",dft_h,dft_v,select)
    expected=np.abs(matrix)**2
    close(actual.reshape(1,2,16,8,4,192)[..., [0,96,191]],expected,atol=.004,rtol=3e-6)
    close(pdp_power(h),np.abs(np.fft.ifft(h.astype(np.complex128),axis=-1))**2,atol=2e-7,rtol=5e-6)
    # No relative reference: known absolute delay index must remain visible.
    k=np.arange(192);delay=13
    tone=np.exp(-2j*np.pi*delay*k/192).astype(np.complex64)
    one=np.broadcast_to(tone,(1,256,4,192)).copy()
    assert np.argmax(pdp_power(one),axis=-1).min()==delay
    rotated=h*np.complex64(np.exp(.73j))
    close(pas_power(rotated),pas_power(h),atol=.004,rtol=5e-6)
    close(pdp_power(rotated),pdp_power(h),atol=2e-7,rtol=5e-6)
    assert np.sum(np.abs(rotated-h)**2,dtype=np.float64)>0
    # Global NMSE reduction is fixed by energy sums; no per-query phase fit.
    t=np.zeros((2,256,4,192),np.complex64);t[0]=3;t[1]=1
    p=t.copy();p[1]=0
    global_error=float(np.sum(abs(p-t)**2,dtype=np.float64)/np.sum(abs(t)**2,dtype=np.float64))
    assert abs(global_error-.1)<1e-14
    close(pas_from_power(pas_power(h),h),np.ones(1),atol=1e-6,rtol=1e-6)
    close(pas_from_power(np.zeros_like(actual),h),np.zeros(1),atol=0,rtol=0)
    rejects(lambda:reject_online_test_as_oof(prediction_namespace="official_test_position_order",truth_namespace="canonical_train_row_ids"))
    passed("official_PAS_PDP_raw_phase_and_global_error_operators",dict(absolute_delay_bin=delay,
      measured_self_reference_used=False,oracle_alignment=False,model_quality_statistics_computed=False))

def model_checks(src):
    from .field_models import CompetitionNeRF2,CompetitionNGRF
    from .wrfgs import WRFGSPlusPowerAdapter
    from .newrf import NeWRFFullH,SyntheticRayProvider,NeWRFQualificationError,synthetic_self_check as check_newrf
    cfgs=dict(nerf2=dict(depth=2,width=8,skips=(),position_levels=1,direction_levels=1,
                  transmitter_levels=1,n_rays=1,n_samples=3,near_m=.1,far_m=1.,
                  ray_chunk=256,frequency_chunk=96),
              ngrf=dict(num_gaussians=2,latent_dim=4,hidden_dim=8,initial_scale=1.,
                   frequency_encoding_levels=1,frequency_chunk_size=96,pos_encoding_freqs=1,
                   attribute_num_layers=2,decoder_num_layers=2,decoder_dropout_p=0.))
    pos=torch.tensor([[.213,-.419,1.237]],dtype=torch.float64);site=torch.tensor([0])
    assert not any(np.allclose(pos[0].numpy(),x,atol=1e-12,rtol=0) for x in src.positions)
    q=PositionQuery(np.array([-1001]),pos.numpy(),site.numpy(),src.identity)
    freq=torch.tensor(np.array(src.geometry.frequencies_hz),dtype=torch.float64)
    meta={}
    for name in ["nerf2","ngrf"]:
        model=make_field_model(name,src,cfgs[name]);model.eval()
        state=digest(model)
        with torch.no_grad():
            full=model(pos,site,freq)
            few=model(pos,site,freq[[0,95,96,97,191]])
        assert full.shape==(1,256,4,192) and full.dtype==torch.complex64 and torch.isfinite(full).all()
        torch.testing.assert_close(full[..., [0,95,96,97,191]],few,atol=2e-6,rtol=2e-5)
        assert not torch.allclose(full,full[...,96,None].expand_as(full))
        assert not torch.allclose(full[:,:128],full[:,128:])
        assert not torch.allclose(full[:,:,0],full[:,:,2])
        adapter=SimulationAdapter(name,model,src)
        packed=adapter.prediction(q)
        close(packed["channel"],full.numpy(),atol=2e-6,rtol=2e-5)
        rejects(lambda:evaluate_output(packed,q,np.empty((0,))),PermissionError)
        bad={**packed,"prediction_namespace":"official_test_position_order"}
        rejects(lambda:validate_output(bad,q))
        bad={**packed,"method":"v16","prediction_origin":"stored_file"}
        rejects(lambda:validate_output(bad,q))
        rejects(lambda:SimulationAdapter("v16",model,src),TypeError)
        y=model(pos,site,freq[[0,96,191]])
        grad=torch.autograd.grad(y.abs().square().mean(),tuple(model.parameters()),allow_unused=True)
        assert all(g is None or torch.isfinite(g).all() for g in grad)
        assert any(g is not None and (g.abs()>0).any() for g in grad)
        assert state==digest(model)
        passed(name+"_complete_H_port_polarization_frequency_and_provenance",
          dict(shape=list(full.shape),full_frequency=True,subset_parity=True,polarizations_not_copied=True,
           finite_derivatives=True,parameters_unchanged=True,actual_position_model_forward=False))
        meta[name]=dict(configuration=cfgs[name],model=model.metadata(),purpose="synthetic correctness configuration only")
    # WRF uses the same V16 convention and all192 tone identities, power output only.
    g=src.geometry
    synthetic_fit=np.array([[.2,-.4,1.2],[.7,-.2,1.6],[1.2,.5,1.1],[1.7,.8,1.8]])
    initial=[np.stack([center+axes[:,0]*2,center+axes[:,0]*3+axes[:,1]*.2])
             for center,axes in zip(g.bs_positions_m,g.bs_rotations)]
    wrf=make_wrf_model(src,synthetic_fit,initial,width=4,height=3,synthetic_fixture=True);wrf.eval()
    state=digest(wrf)
    with torch.no_grad():power=wrf(pos,site,freq)
    assert power.shape==(1,256,4,192) and torch.isfinite(power).all() and (power>=0).all()
    assert not torch.allclose(power[:,:128],power[:,128:])
    assert not torch.allclose(power[:,:,0],power[:,:,2])
    assert not torch.allclose(power,power[...,96,None].expand_as(power))
    packed=SimulationAdapter("wrfgs",wrf,src).prediction(q)
    assert "channel" not in packed and packed["eligibility"]["H"] is False
    assert packed["eligibility"]["PDP"] is False and packed["eligibility"]["NMSE"] is False
    rejects(lambda:wrf(pos,site,freq.flip(0)))
    assert digest(wrf)==state
    passed("wrfgs_full_frequency_codebook_power_and_partial_metric_scope",
      dict(shape=list(power.shape),nonnegative=True,polarization_and_UE_channels_distinct=True,
           full_H=False,pdp=False,nmse=False,combined_score=False))
    meta["wrfgs"]=dict(eligibility=wrf.eligibility(),render_resolution=[4,3],purpose="synthetic correctness only")
    provider=SyntheticRayProvider([[1.,0.,0.]])
    rejects(lambda:make_newrf_model(src,provider,{}),NeWRFQualificationError)
    # NeWRF's standalone artificial fixture checks author integration/coarse/fine.
    check=check_newrf()
    assert check["dataset_gate_closed"] and not check["trained"] and not check["scored"]
    passed("newrf_synthetic_complete_H_and_real_dataset_gate",check)
    meta["newrf"]=dict(status="synthetic core passes; no qualified canonicalFIT direction provider",
       eligible_for_real_benchmark=False,synthetic_check=check)
    return meta

def preservation():
    global INTEGRITY_ONLY
    INTEGRITY_ONLY=True
    before=json.loads((REPORT/"preservation_before.json").read_text());changed=[]
    for name,v in before.items():
        p=PATHS.root/name
        if not p.is_file():changed.append(name);continue
        st=p.stat()
        if [st.st_size,st.st_mtime_ns,st.st_ino]!=[v["bytes"],v["mtime_ns"],v["inode"]]:
            changed.append(name);continue
        if v["sha256"] and sha256(p)!=v["sha256"]:changed.append(name)
    assert not changed,changed
    assert (PATHS.experiment_records/"EXPERIMENT_RECORD.md").read_bytes().startswith((REPORT/"ledger_before.bin").read_bytes())
    INTEGRITY_ONLY=False
    passed("protected_existing_models_data_results_and_ledger_prefix",dict(files=len(before),changes=changed,checkpoint_deserialization=False))

def run():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("Explicit CPU-only CUDA visibility required")
    torch.set_num_threads(2);torch.manual_seed(20311031);np.random.seed(20311031)
    opened=[]
    def audit(event,args):
        if event=="open" and args and isinstance(args[0],(str,bytes,os.PathLike)):
            name=os.fsdecode(args[0])
            if name.endswith(".tfrecords") or (name.endswith((".pt",".pth")) and not INTEGRITY_ONLY):
                raise PermissionError("No checkpoint/model loads; integrity hashing is isolated")
            if "plus-pro/datasets" in name or "plus-pro/artifacts" in name:raise PermissionError("No measured-data access")
            if name.endswith("Round2_Train_Channel.npy"):opened.append(PATHS.relative(name))
    sys.addaudithook(audit)
    started=time.monotonic()
    src=dataset_checks();operator_checks(src);models=model_checks(src);preservation()
    assert not torch.cuda.is_initialized()
    out=dict(status="correctness_checks_passed_with_explicit_qualification_gates",checks=CHECKS,models=models,
       input_identity=src.identity,input_access=src.access,opened_channel_files=opened,
       training=False,optimizer_updates=0,checkpoint_loads=0,real_position_predictions=0,model_quality_scoring=False,
       raw_H_reference=True,measured_relative_reference=False,oracle_alignment=False,
       runtime=dict(seconds=time.monotonic()-started,device="CPU",torch=torch.__version__,python=platform.python_version(),cuda_initialized=False),
       source_sha256={PATHS.relative(p):sha256(p) for p in sorted(SOURCE.glob("*.py"))})
    save_json(REPORT/"verification.json",out)
    print(json.dumps(dict(status=out["status"],checks=len(CHECKS),runtime=out["runtime"])))
if __name__=="__main__":run()

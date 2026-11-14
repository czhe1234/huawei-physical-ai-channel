"""CPU-only correctness probes; no trained models, real-query predictions or scores."""
from __future__ import annotations
import ast
import dataclasses
import hashlib
import inspect
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from measured.hall_adapters.common import ArraySpec, Query, frozen_array, save_json, sha256
from measured.hall_metric_protocol_v1 import metrics as np_features
from measured.hall_adapters.metrics import codebook
from .contract import SouthInputs, PhysicalContract, FitScale, FullHAdapter, REPORT, SOURCE, file_identity
from .objectives import independent_features, structural_terms, raw_complex_objective

CHECKS=[]
ACCESS=[]
def record(name, detail=None):
    CHECKS.append(dict(name=name,passed=True,detail=detail))
    print("PASS",name,flush=True)
def close(a,b,atol=1e-10,rtol=1e-8):
    np.testing.assert_allclose(np.asarray(a),np.asarray(b),atol=atol,rtol=rtol)
def reject(fn,exception=(ValueError,TypeError,PermissionError)):
    try:fn()
    except exception:return
    raise AssertionError("Invalid input was accepted")
def raw_multipath(f,receiver,n=2):
    # Noninteger path delays, distinct complex coefficients at every receive port.
    m=np.arange(64).reshape(2,32)
    delays=np.stack([43.25e-9+0.11e-9*m,87.7e-9+0.17e-9*m],-1)
    amps=np.stack([(1+.005*m)*np.exp(1j*.04*m),(.55+.002*m)*np.exp(-1j*.13*m)],-1)
    paths=amps[...,None]*np.exp(-2j*np.pi*delays[...,None]*f[None,None,None,:])
    x=paths.sum(-2)
    return np.stack([x*np.exp(1j*.21*i) for i in range(n)]).reshape(n,2,8,4,len(f))

def metadata_and_fit():
    src=SouthInputs();c=src.physical
    prepared=src.data.directory
    contract=json.loads((prepared/"contract.json").read_text())
    provenance={}
    for key in ("offsets","model","spec"):
        entry=contract["inputs"][key];p=PATHS.root/entry["path"]
        assert sha256(p)==entry["sha256"]
        provenance[key]=entry
    spec=json.loads((PATHS.root/contract["inputs"]["spec"]["path"]).read_text())
    assignment=np.array([a["assignments"] for a in spec["antennas"]])
    assert np.array_equal(assignment,src.data.meta["antenna_assignments"])
    assert np.array_equal(c.raw_port_indices,assignment.transpose(0,2,1).reshape(2,32))
    close(c.frequencies_hz[[0,511,512,513,1023]],[3413000000.,3437951171.875,3438000000.,3438048828.125,3462951171.875],atol=0,rtol=0)
    for bad in (c.frequencies_hz[::-1],c.frequencies_hz.astype(np.float32).astype(np.float64)):
        reject(lambda:PhysicalContract(bad,c.receiver_position_m,c.raw_port_indices))
    ports=c.raw_port_indices.copy();ports[ports==29]=9
    reject(lambda:PhysicalContract(c.frequencies_hz,c.receiver_position_m,ports))
    record("data_provenance_frequency_and_port_bijection",dict(provenance=provenance,physical=c.describe()))
    # Check all source spatial roles from metadata only.
    minimum={}
    for pair in (("fit","validation"),("fit","test"),("validation","test")):
        a,b=[src.data.meta["position"][src.data.partition(0,k),:2] for k in pair]
        d=np.sqrt(((a[:,None]-b[None])**2).sum(-1)).min()
        assert d>=.3
        minimum["__".join(pair)]=float(d)
    assert not set(src.fit_ids)&set(src.validation_ids)
    reject(lambda:src.query("test"))
    reject(lambda:src.reference("validation"))
    reject(lambda:src.read_fit(src.validation_ids[:1]))
    reject(lambda:src.query("fit",[src.fit_ids[0],src.fit_ids[0]]))
    query=src.query("validation",src.validation_ids[:1])
    assert not hasattr(query,"channel") and set(vars(query))=={"row_ids","position","spec"}
    badspec=dataclasses.replace(query.spec,carrier_hz=query.spec.carrier_hz+1)
    reject(lambda:c.tensor_inputs(Query(query.row_ids,query.position,badspec)))
    record("spatial_roles_position_only_and_denied_truth",minimum)
    bank=src.read_fit()
    scale=FitScale.from_fit(bank,src.data.identity)
    normalized=scale.normalize(bank.channel)
    restored=scale.restore(normalized)
    close(restored,bank.channel,atol=1e-15,rtol=1e-12)
    assert normalized.dtype==np.complex128
    for p in range(2):
        expected=np.sqrt(np.mean(bank.channel[:,p].real.astype(np.float64)**2+bank.channel[:,p].imag.astype(np.float64)**2))
        close(scale.value[p],expected,atol=1e-15,rtol=1e-13)
    # Liveness is an input diagnostic only; no comparison to model predictions.
    nonzero=(np.abs(bank.channel)>0).sum(axis=(0,1,2,3))
    assert np.isfinite(bank.channel).all()
    record("fit_only_positive_real_scale_roundtrip",dict(scale=scale.describe(),
       fit_channel_shape=list(bank.channel.shape),all_finite=True,
       fully_zero_frequency_indices=np.flatnonzero(nonzero==0).tolist(),
       dc_index512_nonzero_links=int(nonzero[512]),physical_restore_max_error=float(np.max(np.abs(restored-bank.channel)))))
    save_json(REPORT/"data_contract.json",dict(identity=src.data.identity,physical=c.describe(),scale=scale.describe(),
       fit_row_ids=src.fit_ids.tolist(),validation_row_ids=src.validation_ids.tolist(),provenance=provenance,
       interpretation="No absolute mobile clock or interpanel coherent reference certified"))
    return src,scale

def calibration_probe(src):
    # Execute only the existing pure calibration helpers, never the raw-data pipeline.
    path=PATHS.code_root/"measured/prepare_hall_blocker_data.py"
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {"correction","calibrate"}]
    namespace={"np":np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path), "exec"),namespace)
    a=np.arange(64,dtype=np.float64)[:,None]
    k=np.arange(1024,dtype=np.float64)[None,:]
    raw=(1+.01*a+.0001*k)+1j*(.3+.003*a-.00002*k)
    cpo=np.linspace(-.7,.9,64);sto=np.linspace(-1.31,2.17,64)
    assignment=src.data.meta["antenna_assignments"]
    got,_=namespace["calibrate"](np.stack([raw.real,raw.imag],-1),dict(cpo=cpo,sto=sto),assignment)
    # Independent integer source-frequency addressing and scalar phasor arithmetic.
    phase=cpo[:,None]+2*np.pi*sto[:,None]*k/1024.
    shifted=raw[:,(np.arange(1024)+512)%1024]
    expected=shifted*(np.cos(phase)+1j*np.sin(phase))
    expected=expected[assignment].transpose(0,2,1,3)
    close(got,expected,atol=7e-7,rtol=4e-7)
    flat=got.reshape(2,32,1024);back=np.empty((64,1024),np.complex128)
    back[src.physical.raw_port_indices.ravel()]=flat.reshape(64,1024)
    close(back,shifted*(np.cos(phase)+1j*np.sin(phase)),atol=7e-7,rtol=4e-7)
    wrong_center=shifted*np.exp(1j*(cpo[:,None]+2*np.pi*sto[:,None]*(k-512)/1024))
    wrong_sign=shifted*np.exp(-1j*phase)
    assert not np.allclose(back,wrong_center) and not np.allclose(back,wrong_sign)
    double=expected*np.exp(1j*phase)[assignment].transpose(0,2,1,3)
    assert not np.allclose(got,double)
    record("synthetic_calibration_order_sign_index_and_port_inverse",
       dict(existing_helper_sha256=sha256(path),maximum_absolute_error=float(np.max(np.abs(got-expected))),
            wrong_sign_rejected=True,centered_k_rejected=True,double_calibration_rejected=True))

def feature_probes(src):
    c=src.physical;f=c.frequencies_hz
    h=raw_multipath(f,c.receiver_position_m)
    assert np.iscomplexobj(h) and not np.allclose(h,h[...,512,None])
    # Direct unreferenced channel detects nonzero delay and its sign.
    tau=83.125e-9
    direct=np.exp(-2j*np.pi*f*tau)
    step=direct[1:]*direct[:-1].conj()
    close(step,np.exp(-2j*np.pi*np.diff(f)*tau),atol=1e-12,rtol=1e-12)
    assert not np.allclose(direct,np.broadcast_to(direct[512],direct.shape))
    record("analytic_nonzero_delay_and_two_path_raw_complex_probe",
       dict(delay_seconds=tau,frequency_order="ascending",propagation_sign=-1,relative_reference_applied=False))
    # Array geometry -> known planar wave -> exact 2D DFT bin, independent of port labels.
    spec=src.data.spec;lam=299792458./f[512];target=(1,1)
    for p in range(2):
        horizontal=spec.axes[p,:,1];up=spec.axes[p,:,2]
        direction=horizontal*(lam/spec.spacing_m/8)-up*(lam/spec.spacing_m/4)
        direction+=spec.axes[p,:,0]*np.sqrt(1-np.dot(direction,direction))
        x=np.exp(2j*np.pi*np.einsum("hvd,d->hv",spec.element_offsets[p],direction)/lam)
        power=np.abs(np.fft.fft2(x,norm="ortho"))**2
        assert np.unravel_index(power.argmax(),power.shape)==target
    record("physical_port_geometry_plane_wave_to_DFT_bin",dict(horizontal_bin=1,vertical_bin=1))
    # Independent matrix multiplication, not image resampling.
    flat=h.reshape(2,2,32,1024)
    beam=np.einsum("mb,npmf->npbf",codebook().conj(),flat)
    nf=np_features.describe(h)
    close(np.abs(beam)**2,nf["pas"],atol=1e-10,rtol=1e-10)
    th=torch.tensor(h,dtype=torch.complex128)
    tf=independent_features(th)
    total=(np.abs(h)**2).sum(axis=(-3,-2,-1),keepdims=True).reshape(2,2,1,1)
    close(tf["pas"].detach().numpy(),nf["pas"]/total,atol=1e-11,rtol=1e-9)
    tpdp=torch.fft.fftshift(tf["pdp"],dim=-1).detach().numpy()
    close(tpdp,nf["pdp"]/total,atol=1e-11,rtol=1e-9)
    close(tf["share"].detach().numpy(),nf["energy_fraction"],atol=1e-12,rtol=1e-10)
    close(torch.fft.fftshift(tf["joint"],dim=-1).detach().numpy(),nf["receive_delay_joint"],atol=1e-11,rtol=1e-9)
    record("legacy_evaluator_and_differentiable_feature_operator_parity",dict(pdp_order="fftshift reconciled explicitly",no_model_metrics=True))
    # Independent clocks cannot change relative features, but element-wise clocks can.
    phase=np.array([[.3,-.8],[1.4,.6]])
    shift=np.array([[3.7,-2.3],[.7,4.9]])
    gauge=np.exp(1j*phase[...,None,None,None]-2j*np.pi*shift[...,None,None,None]*np.arange(1024)/1024)
    g=np_features.describe(h*gauge)
    close(g["receive_delay_joint"],nf["receive_delay_joint"],atol=1e-11,rtol=1e-8)
    close(g["energy_fraction"],nf["energy_fraction"],atol=1e-12,rtol=1e-10)
    varied=h.copy();varied[:,:,0,0]*=np.exp(-2j*np.pi*9.25*np.arange(1024)/1024)
    bad=np_features.describe(varied)
    assert np.max(np.abs(bad["receive_delay_joint"]-nf["receive_delay_joint"]))>1e-5
    tf2=independent_features(th)
    close(tf2["joint"].detach().numpy(),tf["joint"].detach().numpy(),atol=0,rtol=0)
    assert "truth" not in inspect.signature(independent_features).parameters
    reject(lambda:raw_complex_objective(th,th),PermissionError)
    record("independent_self_reference_and_no_absolute_phase_claim",dict(common_panel_clock_invariant=True,per_link_delay_error_retained=True,paired_alignment=False))
    # Zero/nonfinite failures are never silently masked by a model-specific reference.
    reject(lambda:src.physical.unflatten(torch.full((1,2,32,1024),complex(float("nan"),0),dtype=torch.complex64)))
    zero=structural_terms(torch.zeros_like(th),th)
    assert zero["pas"].item()==1 and zero["relative_pdp"].item()==1 and zero["receive_energy"].item()==1 and zero["receive_delay_joint"].item()==1
    record("synthetic_zero_prediction_penalty_and_nonfinite_rejection")
    return h

def state_digest(model):
    d=hashlib.sha256()
    for name,x in sorted(model.state_dict().items()):
        d.update(name.encode());d.update(x.detach().cpu().numpy().tobytes())
    return d.hexdigest()

def model_checks(src,scale):
    # Full-size frequency/port axes, tiny network/ray counts for numerical checks only.
    from .nerf2 import NeRF2FullH
    from .ngrf import NGRFFullH
    from . import nerf2,ngrf
    return run_models(src,scale,NeRF2FullH,NGRFFullH,nerf2,ngrf)

def run_models(src,scale,NeRF2FullH,NGRFFullH,nerf2_module,ngrf_module):
    rx=torch.tensor(np.array(src.physical.receiver_position_m),dtype=torch.float64)
    f=torch.tensor(np.array(src.physical.frequencies_hz),dtype=torch.float64)
    # Artificial TX positions, unrelated to all measured FIT/validation/test positions.
    tx=torch.tensor([[.213,-.419,1.237],[1.731,.583,2.117]],dtype=torch.float64)
    assert not any(np.allclose(x,tx[0].numpy(),atol=1e-12,rtol=0) or np.allclose(x,tx[1].numpy(),atol=1e-12,rtol=0) for x in src.data.meta["position"])
    configs=dict(nerf2=dict(depth=2,width=16,skips=(),position_levels=2,direction_levels=2,
                 transmitter_levels=2,n_rays=3,n_samples=4,near_m=.1,far_m=2.,
                 ray_chunk=192,frequency_chunk=128),
              ngrf=dict(num_gaussians=5,latent_dim=8,hidden_dim=16,initial_scale=3.,
                frequency_encoding_levels=2,frequency_chunk_size=128,pos_encoding_freqs=2,
                attribute_num_layers=2,decoder_num_layers=2,decoder_dropout_p=0.))
    models={"nerf2":NeRF2FullH(**configs["nerf2"]),
            "ngrf":NGRFFullH(rx,**configs["ngrf"])}
    output={}
    for name,model in models.items():
        model.eval()
        before=state_digest(model)
        with torch.no_grad():
            full=model(tx,rx,f)
            separate=torch.cat([model(tx[i:i+1],rx,f) for i in range(len(tx))],0)
            one=model(tx,rx,f[[0,511,512,513,1023]])
        assert full.shape==(2,2,32,1024) and full.dtype==torch.complex64 and bool(torch.isfinite(full).all())
        assert bool(full.imag.abs().max()>0)
        torch.testing.assert_close(full,separate,atol=3e-6,rtol=1e-5)
        torch.testing.assert_close(full[:,:,:,[0,511,512,513,1023]],one,atol=3e-6,rtol=1e-5)
        assert not torch.allclose(full,full[...,512,None].expand_as(full))
        assert not torch.allclose(full[0],full[1])
        # Deliberately use a synthetic Query rather than any real measurement query.
        q=Query(frozen_array([-1001,-1002],np.int64),frozen_array(tx.numpy()),src.data.spec)
        wrapped=FullHAdapter(model,src.physical,scale)
        packed=wrapped.prediction(q)
        assert packed["channel"].shape==(2,2,8,4,1024)
        close(packed["channel"],full.numpy().reshape(2,2,8,4,1024)*scale.value[None,:,None,None,None],atol=3e-6,rtol=2e-6)
        assert np.array_equal(packed["row_ids"],q.row_ids)
        reject(lambda:model(tx,rx,f.float()),TypeError)
        perm=torch.arange(31,-1,-1)
        if name=="nerf2":
            with torch.no_grad():
                reordered=model(tx,rx[:,perm],f[[0,512,1023]])
            torch.testing.assert_close(reordered,full[:,:,perm][:,:,:,[0,512,1023]],atol=4e-6,rtol=2e-5)
            alternate=NeRF2FullH(**{**configs[name],"ray_chunk":71,"frequency_chunk":2})
        else:
            reject(lambda:model(tx,rx[:,perm],f),ValueError)
            reject(lambda:model(tx,rx.flip(0),f),ValueError)
            alternate=NGRFFullH(rx,**{**configs[name],"frequency_chunk_size":2})
        alternate.load_state_dict(model.state_dict());alternate.eval()
        with torch.no_grad():
            chunked=alternate(tx,rx,f[[0,511,512,513,1023]])
        torch.testing.assert_close(one,chunked,atol=4e-6,rtol=2e-5)
        params=tuple(model.parameters())
        synthetic=model(tx[:1],rx,f[[0,511,512,1023]])
        gradients=torch.autograd.grad(synthetic.abs().square().mean(),params,allow_unused=True)
        assert all(g is None or bool(torch.isfinite(g).all()) for g in gradients)
        active=sum(g is not None and bool(g.abs().max()>0) for g in gradients)
        assert active>0 and state_digest(model)==before
        record(name+"_full_frequency_physical_ports_and_differentiable_forward",
          dict(output_shape=list(full.shape),batch_single_parity=True,tone_subset_parity=True,
           chunk_parity=True,complex_phase_retained=True,middle_tone_copy_rejected=True,
           port_handling="permutation equivariant" if name=="nerf2" else "bound fixed array; permutation rejected",
           parameter_tensors_with_nonzero_synthetic_derivative=active,
           parameter_and_buffer_state_unchanged=True))
        output[name]=dict(config=configs[name],purpose="tiny synthetic correctness only, not selected paper capacity",
          metadata=model.metadata() if hasattr(model,"metadata") else {},state_sha256=before)
    # The NeRF complex integral has an independent two-emitter closed form.
    raw=torch.zeros((1,4,4,4),dtype=torch.float64)
    raw[:,:,0,2]=1.1;raw[:,:,2,2]=.57
    phase0=.37;phase2=-.83
    raw[:,:,0,3]=torch.logit(torch.tensor((phase0+np.pi)/(2*np.pi),dtype=torch.float64))
    raw[:,:,2,3]=torch.logit(torch.tensor((phase2+np.pi)/(2*np.pi),dtype=torch.float64))
    lengths=torch.tensor([1.7,2.1,3.2,1.4],dtype=torch.float64)
    probe_f=torch.tensor([3413e6,3438e6,3438e6+.125,3462e6],dtype=torch.float64)
    actual=nerf2_module.coherent_nerf2_integral(raw,lengths,probe_f,propagation_sign=-1,near_m=.2)
    expected=1.1*torch.exp(1j*(phase0-2*np.pi*probe_f*1.9/299792458.))+             .57*torch.exp(1j*(phase2-2*np.pi*probe_f*7.2/299792458.))
    torch.testing.assert_close(actual[0],expected,atol=2e-12,rtol=2e-12)
    assert not torch.equal(actual[...,1],actual[...,2])
    record("nerf2_two_emitter_closed_form_and_sub_Hz_phase_precision")
    # Compare nGRF adapter to the unmodified official Torch render for real/imag order.
    m=models["ngrf"]
    with torch.no_grad():
        direct=m(tx,rx,f[[0,512,1023]])
        for b in range(2):
            for p in range(2):
                for j,k in enumerate([0,512,1023]):
                    reference=ngrf_module.official_reference_panel(m,tx[b],p,float(f[k]))
                    torch.testing.assert_close(direct[b,p,:,j],reference,atol=2e-6,rtol=1e-5)
    record("ngrf_official_Torch_reference_complex_aggregation_parity",
       dict(synthetic_query_panel_tone_cases=12,nt=1,nr=32,cuda_parity_claimed=False))
    reject(lambda:structural_terms(torch.zeros((1,2,8,4,8),dtype=torch.complex64),torch.zeros((1,2,8,4,8))))
    record("complex_target_required_by_structural_interface")
    return output

def run():
    started=time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES",None)!="":
        raise RuntimeError("Correctness checks require explicit CUDA_VISIBLE_DEVICES empty")
    torch.set_num_threads(2);torch.manual_seed(20261031);np.random.seed(20261031)
    def audit(event,args):
        if event=="open" and args and isinstance(args[0],(str,bytes,os.PathLike)):
            name=os.fsdecode(args[0])
            if any(x in name.lower() for x in ("hall_nw","dichasus-dc20","reftx-offsets-dichasus-dc20","inue-northwest-wall")):
                raise PermissionError("North access is forbidden")
            if name.endswith((".tfrecords",".pt",".pth")):
                raise PermissionError("No raw-frame/checkpoint access in correctness stage")
            if name.endswith("channels.npy"):
                if Path(name).resolve()!=(PATHS.artifacts/"measured/hall_sw_publisher_v2/channels.npy").resolve():
                    raise PermissionError("Unexpected channel source")
                ACCESS.append(dict(file=PATHS.relative(name),purpose="FIT-only array/header access"))
    import sys
    sys.addaudithook(audit)
    src,scale=metadata_and_fit()
    calibration_probe(src)
    feature_probes(src)
    model_result=model_checks(src,scale)
    assert not torch.cuda.is_initialized()
    result=dict(status="correctness_passed_not_performance_validated",checks=CHECKS,models=model_result,
       input_access=dict(fit_reads=src.fit_reads,opened_channel_files=ACCESS,validation_channel_reads=0,
         test_channel_reads=0,north_reads=0,real_position_model_forwards=0),
       training=False,optimizer_updates=0,checkpoint_loads=0,model_quality_scoring=False,
       absolute_phase_reference_verified=False,interpanel_coherence_verified=False,
       environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
          device="CPU",threads=torch.get_num_threads(),cuda_initialized=False),
       wall_seconds=time.monotonic()-started,identity=file_identity(),
       source_sha256={PATHS.relative(p):sha256(p) for p in sorted(SOURCE.glob("*.py"))})
    save_json(REPORT/"verification.json",result)
    print(json.dumps(dict(status=result["status"],passed=len(CHECKS),wall_seconds=result["wall_seconds"]),ensure_ascii=False))
if __name__=="__main__":run()

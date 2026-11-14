"""Numerical feasibility checks: synthetic operators plus south FIT conditioning."""
from __future__ import annotations
import io,json,time,traceback,unittest,resource,importlib.util
import numpy as np
import torch
from project_paths import PATHS
from measured.hall_adapters.common import save_json,sha256
from measured.hall_structured_v2.data import SouthDevelopment
from measured.hall_structured_v2.synthesis import relative_anchor
from .stage import SOURCE,REPORT,init,snapshot
from .metrics import describe,REFERENCE_COHERENCE_MIN
def stats(x):
    x=np.asarray(x).ravel()
    return dict(count=len(x),minimum=float(np.min(x)),p05=float(np.quantile(x,.05)),
        median=float(np.median(x)),p95=float(np.quantile(x,.95)),maximum=float(np.max(x)))
def run():
    init();start=time.monotonic();torch.set_num_threads(2)
    criteria=dict(clock_joint_L1_max=5e-5,FP32_FP64_anchor_difference_ns_max=.02,
        coherence_numeric_min=REFERENCE_COHERENCE_MIN,
        meaning="Numerical gauge/conditioning criteria only, not a measured-noise or absolute-clock certification.")
    save_json(REPORT/"numerical_criteria.json",criteria)
    from . import tests
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromModule(tests)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (REPORT/"operator_checks.log").write_text(stream.getvalue())
    if not result.wasSuccessful():raise AssertionError(stream.getvalue())
    unit=dict(passed=True,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors))
    reader=SouthDevelopment();bank=reader.fit();h=np.array(bank.channel);f=h.shape[-1];bw=bank.spec.bandwidth_hz
    d=describe(h,bw);ref=d["reference"];valid=ref["valid"]
    rng=np.random.default_rng(20261025);n,p=h.shape[:2]
    tau=rng.uniform(-f/2,f/2,(n,p));phi=rng.uniform(-np.pi,np.pi,(n,p));gain=rng.uniform(.4,2.,(n,p))
    factor=gain[...,None,None,None]*np.exp(1j*phi[...,None,None,None]-2j*np.pi*tau[...,None,None,None]*np.arange(f)/f)
    shifted=(h*factor).astype(np.complex64)
    shifted_description=describe(shifted,bw)
    distance=np.abs(d["receive_delay_joint"]-shifted_description["receive_delay_joint"]).sum((-2,-1))
    common_valid=valid&shifted_description["reference"]["valid"]
    clock_error=float(distance[common_valid].max()) if common_valid.any() else None
    with torch.no_grad():bins32=relative_anchor(torch.tensor(h)).numpy()
    difference=(bins32-ref["anchor_bins"]+f/2)%f-f/2
    dtype_ns=np.abs(difference)/bw*1e9
    edge=max(1,int(np.ceil(.05*f)))
    edge_mass=d["receive_delay_joint"][...,:edge].sum((-2,-1))+d["receive_delay_joint"][...,-edge:].sum((-2,-1))
    pi=d["energy_fraction"]
    conditioning=dict(positions=n,panels=int(valid.size),valid_reference_panels=int(valid.sum()),
        bandwidth_hz=bw,frequency_bins=f,frequency_spacing_hz=bw/f,
        nominal_delay_resolution_ns=1e9/bw,delay_period_us=f/bw*1e6,
        reference_coherence=stats(ref["coherence"]),
        reference_coherence_by_panel=[stats(ref["coherence"][:,j]) for j in range(2)],
        clock_transformation_joint_L1=stats(distance[common_valid]),
        FP32_FP64_anchor_difference_ns=stats(dtype_ns[valid]),
        centered_delay_outer10percent_energy=stats(edge_mass),
        receive_energy_data_description=dict(
          maximum_channel_fraction_by_panel=[stats(pi[:,j].max(-1)) for j in range(2)],
          across_position_channel_fraction_std_by_panel=[stats(pi[:,j].std(0)) for j in range(2)]),
        scope="Only measured FIT descriptions and self-transforms; no trained output or method score.")
    usable=bool(valid.all() and clock_error is not None and clock_error<criteria["clock_joint_L1_max"]
        and dtype_ns[valid].max()<criteria["FP32_FP64_anchor_difference_ns_max"])
    before=json.loads((REPORT/"preservation_before.json").read_text());after=snapshot()
    assert before==after,"Prior files changed"
    assert all(r["role"]=="fit" and set(r["rows"])==set(bank.row_ids) for r in reader.access)
    try:reader.reference("validation")
    except ValueError:pass
    else:raise AssertionError("Validation labels exposed")
    # Use the repository's maintained-source pattern definitions on this stage.
    file=PATHS.code_root/"tools/check_hardcoded_paths.py"
    spec=importlib.util.spec_from_file_location("metric_path_checker",file)
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    bad=[]
    for file in SOURCE.glob("*.py"):
        compile(file.read_text(),str(file),"exec")
        for i,line in enumerate(file.read_text().splitlines(),1):
            if any(pattern.search(line) for _,pattern in checker.PATTERNS):bad.append(dict(file=file.name,line=i))
    if bad:raise AssertionError(bad)
    result=dict(implementation_checks_passed=True,operators=unit,FIT_numerical_reference_usable=usable,
        conditioning=conditioning,criteria=criteria,access=reader.access,data_identity=reader.data.identity,
        scope=dict(new_training=False,algorithm_rescoring=False,trained_checkpoints_loaded=False,
            validation_channels=False,test=False,north=False),
        preservation=dict(files=len(before),changed=[],missing=[],old_ledger_prefix=(PATHS.root/"experiment_records/EXPERIMENT_RECORD.md").read_bytes().startswith((REPORT/"ledger_before.bin").read_bytes())),
        source_sha256={file.name:sha256(file) for file in SOURCE.glob("*.py")},new_source_path_failures=bad,
        seconds=time.monotonic()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    save_json(REPORT/"feasibility.json",result)
    print(json.dumps(dict(tests=unit,usable=usable,FIT=conditioning,seconds=result["seconds"]),allow_nan=False),flush=True)
if __name__=="__main__":
    try:run()
    except BaseException:
        REPORT.mkdir(parents=True,exist_ok=True)
        save_json(REPORT/"check_failure.json",dict(traceback=traceback.format_exc()));raise

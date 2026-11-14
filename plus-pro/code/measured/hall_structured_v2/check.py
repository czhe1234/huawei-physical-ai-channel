"""Bounded engineering acceptance. No optimizer, real score, RT, TEST or north."""
from __future__ import annotations
import importlib.util,io,json,platform,resource,time,traceback,unittest
import numpy as np
import torch
from project_paths import PATHS
from measured.hall_adapters.common import save_json,sha256,Query
from .config import Config,REPORT,SOURCE,WORK
from .stage import snapshot
from .data import SouthDevelopment
from .foundation import prepare
from .model import HallStructured
from .observables import FitPower
def static_checks():
    files=sorted(SOURCE.glob("*.py"))
    for file in files:compile(file.read_text(),str(file),"exec")
    path=PATHS.code_root/"tools/check_hardcoded_paths.py"
    spec=importlib.util.spec_from_file_location("hall_scoped_path_check",path)
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    failures=[]
    for file in files:
        for line,text in enumerate(file.read_text().splitlines(),1):
            for label,pattern in checker.PATTERNS:
                if pattern.search(text):failures.append(dict(file=file.name,line=line,label=label))
    if failures:raise AssertionError(failures)
    return dict(compiled_files=len(files),machine_path_failures=failures,source_sha256={f.name:sha256(f) for f in files})
def synthetic_checks():
    from . import test_correctness
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_correctness)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (REPORT/"synthetic_checks.log").write_text(stream.getvalue())
    record=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        skipped=len(result.skipped),success=result.wasSuccessful(),optimizer_steps=0,real_data_score=False)
    if not result.wasSuccessful():raise AssertionError(stream.getvalue())
    return record
def real_fit_interface():
    reader=SouthDevelopment();rows=reader.fit_ids[:12];bank=reader.fit(rows);query=reader.query("fit",rows[:2])
    geometry=reader.geometry(query.row_ids)
    cfg=Config();power=FitPower.from_fit(bank.channel,bank.row_ids)
    data=prepare(bank,query,cfg.subbands,"self",geometry)
    torch.manual_seed(20261024);model=HallStructured(bank.spec,power,cfg)
    model.set_fit_statistics(data)
    out=model.predict(query,data)
    if not all(set(ids).issubset(set(rows)) for ids in data.source_ids):raise AssertionError("Donor outside the bank")
    if any(int(r) in ids for r,ids in zip(query.row_ids,data.source_ids)):raise AssertionError("Self donor")
    if out["diagnostics"]["projection"]["power_relative_residual"]>1e-4:raise AssertionError("Requested/output power inconsistency")
    guards={}
    for role in ["fit","validation","test","north"]:
        try:reader.reference(role)
        except ValueError:guards["reference_"+role]=True
        else:guards["reference_"+role]=False
    try:reader.query("test")
    except ValueError:guards["test_query"]=True
    try:SouthDevelopment(directory=PATHS.artifacts/"measured/hall_nw_publisher_v2")
    except ValueError:guards["north_directory"]=True
    try:model.predict(Query(query.row_ids+10000,query.position,query.spec),data)
    except ValueError:guards["query_identity"]=True
    if not all(guards.values()):raise AssertionError("Data-role guard failed")
    for item in reader.access:
        if item["role"]!="fit" or not set(item["rows"]).issubset(set(rows)):raise AssertionError("Unexpected channel access")
    return dict(input_fit_rows=rows.tolist(),prediction_query_rows=query.row_ids.tolist(),
        output_shape=list(out["channel"].shape),output_dtype=str(out["channel"].dtype),
        all_finite=bool(np.isfinite(out["channel"]).all()),source_ids=out["source_ids"].tolist(),
        per_query_donors=data.source_ids,channel_access=reader.access,guards=guards,
        diagnostics=out["diagnostics"],geometry_variance_unknown=not bool(geometry.variance_known.any()),
        parameters_total=sum(p.numel() for p in model.parameters()),
        parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        optimization_performed=False,real_prediction_metrics_computed=False,
        note="A single untrained CPU interface pass; no query-channel comparison.")
def preservation():
    before=json.loads((REPORT/"preservation_before.json").read_text());after=snapshot()
    missing=sorted(set(before)-set(after));added=sorted(set(after)-set(before))
    changed=[key for key in before.keys()&after.keys() if before[key]!=after[key]]
    old=(REPORT/"ledger_before.bin").read_bytes()
    ledger=(PATHS.root/"experiment_records/EXPERIMENT_RECORD.md").read_bytes()
    result=dict(files_checked=len(before),changed=changed,missing=missing,added_outside_new_version=added,
        historical_ledger_prefix_preserved=ledger.startswith(old),
        small_files_verified_by="size,mtime_ns,SHA256",large_files_verified_by="size,mtime_ns; raw channels not reread for preservation")
    if changed or missing or added or not result["historical_ledger_prefix_preserved"]:raise AssertionError(result)
    return result
def main():
    REPORT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    start=time.monotonic()
    result=dict(scope="engineering correctness only",optimizer_steps=0,new_RT_calls=0,
        formal_test=False,north_channel_reads=0,real_prediction_scores=False,
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
            device="CPU",threads=torch.get_num_threads()))
    try:
        result["static"]=static_checks();result["synthetic"]=synthetic_checks()
        result["real_fit_interface"]=real_fit_interface();result["preservation"]=preservation()
        result["success"]=True
    except BaseException:
        result["success"]=False;result["traceback"]=traceback.format_exc()
        raise
    finally:
        result["wall_seconds"]=time.monotonic()-start
        result["peak_rss_MiB"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        save_json(REPORT/"acceptance.json",result)
        print(json.dumps(dict(success=result["success"],tests=result.get("synthetic",{}),
            output_shape=result.get("real_fit_interface",{}).get("output_shape"),seconds=result["wall_seconds"]),allow_nan=False),flush=True)
if __name__=="__main__":main()

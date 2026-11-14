"""Immutable geometry-cache interchange with row, frame and data identity checks."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from measured.hall_adapters.common import sha256
from .geometry import Geometry,subset
KEYS=("coordinate","mass","mask","features","sequence","interaction","depth")
def spec_signature(spec):
    return dict(center=np.asarray(spec.center).tolist(),direction=np.asarray(spec.direction).tolist(),
        spacing_m=spec.spacing_m,carrier_hz=spec.carrier_hz,frequency_count=spec.frequency_count,
        bandwidth_hz=spec.bandwidth_hz,array_shape=[8,4],coordinate_frame="native_measurement")
def validate_geometry(g,rows):
    a=g.arrays
    n,p,k=a["mass"].shape
    if n!=rows or p!=2 or k<1:raise ValueError("Geometry row/panel/candidate layout mismatch")
    expected=dict(coordinate=(n,p,k,3),mask=(n,p,k),features=(n,p,k,24),
        sequence=(n,p,k,16,21),interaction=(n,p,k,16),depth=(n,p,k))
    for key,shape in expected.items():
        if a[key].shape!=shape:raise ValueError("Geometry shape mismatch: "+key)
    for value in list(a.values())+[g.presence,g.energy_cv,g.source_fraction]:
        if not np.isfinite(value).all():raise ValueError("Nonfinite geometry")
    for value in [g.presence,g.energy_cv,g.variance_known]:
        if value.shape!=(n,p,k):raise ValueError("Reliability layout mismatch")
    if g.source_fraction.shape!=(n,p,k,2):raise ValueError("Source fractions must denote base/scatter")
    if np.any(a["mass"]<0) or np.any(g.energy_cv<0) or np.any((g.presence<0)|(g.presence>1)):
        raise ValueError("Invalid geometry energy/reliability")
    if np.any((a["depth"]<0)|(a["depth"]>16)):raise ValueError("Invalid route depth")
def write_cache(path,g,row_ids,position,spec,data_identity,provenance):
    path=Path(path)
    if path.exists():raise FileExistsError("Geometry caches are immutable")
    row_ids=np.asarray(row_ids,np.int64);position=np.asarray(position)
    if len(np.unique(row_ids))!=len(row_ids) or position.shape!=(len(row_ids),3):raise ValueError("Invalid geometry row identities")
    validate_geometry(g,len(row_ids))
    meta=dict(schema="hall_structured_geometry_v2",spec=spec_signature(spec),data_identity=data_identity,
        quality=g.quality,provenance=provenance,
        certification_note="A cache flag is a recorded claim; training also requires a matching external convergence report.")
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+".partial")
    with temp.open("wb") as f:
        np.savez_compressed(f,**g.arrays,row_ids=row_ids,position=position,presence=g.presence,energy_cv=g.energy_cv,
            variance_known=g.variance_known,source_fraction=g.source_fraction,metadata=np.asarray(json.dumps(meta,sort_keys=True,allow_nan=False)))
    temp.replace(path)
    return sha256(path)
def load_cache(path,query,data_identity):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z["metadata"]))
        if meta["schema"]!="hall_structured_geometry_v2" or meta["data_identity"]!=data_identity:raise ValueError("Geometry data identity mismatch")
        if meta["spec"]!=spec_signature(query.spec):raise ValueError("Geometry physical array/frame mismatch")
        rows=z["row_ids"]
        if len(set(rows))!=len(rows):raise ValueError("Duplicate geometry rows")
        lookup={int(r):i for i,r in enumerate(rows)}
        if any(int(r) not in lookup for r in query.row_ids):raise ValueError("Missing geometry rows; no automatic RT")
        ids=np.array([lookup[int(r)] for r in query.row_ids])
        if not np.allclose(z["position"][ids],query.position,rtol=0,atol=1e-6):raise ValueError("Geometry/query position mismatch")
        arrays={key:np.array(z[key][ids],copy=True) for key in KEYS}
        g=Geometry(arrays,*[np.array(z[key][ids],copy=True) for key in ["presence","energy_cv","variance_known","source_fraction"]],dict(meta["quality"]))
    validate_geometry(g,len(query.row_ids))
    g.quality["cache_sha256"]=sha256(path)
    g.quality["provenance"]=meta["provenance"]
    return g
def convergence_evidence(path,g,data_identity,rows):
    """Verify external evidence identity, without running RT or inventing criteria."""
    report=json.loads(Path(path).read_text())
    if report.get("data_identity")!=data_identity or not set(map(int,rows)).issubset(report.get("row_ids",[])):
        raise ValueError("Convergence report does not cover these data/rows")
    if report.get("geometry_cache_sha256")!=g.quality.get("cache_sha256"):raise ValueError("Convergence report/cache mismatch")
    if report.get("passed") is not True or not report.get("checks") or not report.get("source_record_hashes"):
        raise ValueError("A matching successful report with declared checks and source hashes is required")
    return dict(sha256=sha256(path),checks=report["checks"],source_record_hashes=report["source_record_hashes"])

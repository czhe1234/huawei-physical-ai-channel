"""South-only source reader; correctness does not expose held-out labels."""
from __future__ import annotations
import json
import numpy as np
from project_paths import PATHS
from measured.hall_adapters.common import SouthDataset,SOUTH
from .geometry import legacy_geometry
class SouthDevelopment:
    def __init__(self,allow_validation=False,directory=SOUTH,geometry_cache=None):
        self.data=SouthDataset(directory);self.allow_validation=allow_validation;self.access=[];self.geometry_cache=geometry_cache
        path=PATHS.analysis/"results/hall_south_development_v1/contract.json"
        r=json.loads(path.read_text())
        if r["data_identity"]!=self.data.identity:raise ValueError("Prepared/split identities changed")
        self.fit_ids=np.array(r["fit_rows"],np.int64);self.validation_ids=np.array(r["validation_rows"],np.int64)
        self.fold=int(r["fold"])
    def fit(self,rows=None):
        rows=self.fit_ids if rows is None else np.asarray(rows,np.int64)
        if not set(rows).issubset(set(self.fit_ids)):raise ValueError("Outside the declared FIT bank")
        bank=self.data.read_fit(self.fold,rows);self.access+=self.data.channel_reads;self.data.channel_reads.clear()
        return bank
    def query(self,role,rows=None):
        if role not in ["fit","validation"]:raise ValueError("TEST/north queries are not enabled")
        allowed=self.fit_ids if role=="fit" else self.validation_ids
        rows=allowed if rows is None else np.asarray(rows,np.int64)
        if not set(rows).issubset(set(allowed)):raise ValueError("Query role mismatch")
        return self.data.query(rows)
    def reference(self,role):
        if role!="validation" or not self.allow_validation:raise ValueError("Held-out labels are disabled in correctness mode")
        source=np.load(SOUTH/"channels.npy",mmap_mode="r",allow_pickle=False)
        out=np.array(source[self.validation_ids],copy=True)
        self.access.append(dict(fold=self.fold,role="validation",rows=self.validation_ids.tolist(),purpose="explicit development evaluator"))
        return out
    def geometry(self,rows):
        rows=np.asarray(rows,np.int64)
        if not set(rows).issubset(set(self.fit_ids)|set(self.validation_ids)):raise ValueError("Geometry outside declared development rows")
        if self.geometry_cache is not None:
            from .cache import load_cache
            return load_cache(self.geometry_cache,self.data.query(rows),self.data.identity)
        path=PATHS.artifacts/"measured/hall_sw_development_v1/geometry_support.npz"
        with np.load(path,allow_pickle=False) as z:
            lookup={int(x):i for i,x in enumerate(z["row_ids"])}
            if any(int(x) not in lookup for x in rows):raise ValueError("Missing cached geometry; no automatic RT")
            ids=[lookup[int(x)] for x in rows]
            sp={k:np.array(z[k][ids],copy=True) for k in ["coordinate","mass","mask","features","sequence","interaction","depth"]}
            presence=z["presence"][ids]
        return legacy_geometry(sp,presence,2,self.data.spec.bandwidth_hz)
def slice_prepared(data,ids):
    from .foundation import Prepared
    from .geometry import subset
    rows=np.asarray(ids,dtype=np.int64)
    return Prepared(data.row_ids[rows],data.position[rows],data.spec,data.expert_energy[rows],data.carrier[rows],
      data.donor_distance[rows],data.donor_span[rows],[data.source_ids[i] for i in rows],
      subset(data.geometry,rows) if data.geometry is not None else None)

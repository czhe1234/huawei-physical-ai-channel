"""Versioned recovery entrypoints; old correctness adapters remain frozen."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from project_paths import PATHS
from common.official_metrics import point_metric_statistics, aggregate_statistics
from round2.paper_baseline_adapters_v1.contract import PositionQuery
from round2.paper_baseline_adapters_v1.evaluation import reject_online_test_as_oof
from .ray_provider import FitGeometryBank
from .qualified_newrf import QualifiedNeWRFFullH


@dataclass(frozen=True)
class BoundChannelReference:
    channel: np.ndarray
    row_ids: np.ndarray
    input_identity: str
    namespace: str
    source_path: str
    source_stat: tuple


class RecoveredNeWRFAdapter:
    def __init__(self, inputs, geometry_manifest, *, outer=0, role="screen_fit", config=None,
                 max_query_rays=64, max_training_rays=64):
        self.inputs=inputs
        bank=FitGeometryBank(geometry_manifest,inputs,outer=outer,role=role)
        self.model=QualifiedNeWRFFullH(bank,config,max_query_rays=max_query_rays,max_training_rays=max_training_rays)
        self.outer=outer; self.role=role

    def _validate_query(self, query):
        if type(query) is not PositionQuery or query.input_identity != self.inputs.identity:
            raise ValueError("Query input identity mismatch")
        roles=self.inputs.roles(self.outer)
        allowed=roles["outer"] if self.role == "refit" else np.union1d(roles["outer"],roles["validation"])
        if not set(query.row_ids).issubset(set(allowed)):
            raise PermissionError("Use the FIT interface for FIT rays; refit predictions may only be evaluated on outer rows")
        if not np.array_equal(query.ue_position,self.inputs.positions[query.row_ids]) or not np.array_equal(query.site_id,self.inputs.site_id[query.row_ids]):
            raise ValueError("Query row/coordinate/site mapping changed")
        return self.model._qualify()

    def prediction(self, query):
        self._validate_query(query)
        pos,site=query.tensors()
        with torch.no_grad():
            h=self.model(pos,site).detach().cpu().numpy()
        result=dict(method="newrf",row_ids=np.array(query.row_ids),channel=h,
                    prediction_namespace=query.input_namespace,input_identity=self.inputs.identity,
                    geometry_identity=self.inputs.geometry.geometry_fingerprint,prediction_origin="position_only_forward")
        self.validate_output(result,query)
        return result

    def validate_output(self,result,query):
        self._validate_query(query)
        expected={"method","row_ids","channel","prediction_namespace","input_identity","geometry_identity","prediction_origin"}
        if set(result) != expected or result["method"] != "newrf": raise ValueError("Declared full-H output only")
        if result["input_identity"] != self.inputs.identity or result["geometry_identity"] != self.inputs.geometry.geometry_fingerprint:
            raise ValueError("Prediction input/geometry mismatch")
        if result["prediction_origin"] != "position_only_forward" or result["prediction_namespace"] != query.input_namespace:
            raise ValueError("Prediction namespace/origin mismatch")
        if not np.array_equal(result["row_ids"],query.row_ids):raise ValueError("Prediction row order mismatch")
        h=np.asarray(result["channel"])
        if h.shape != (len(query.row_ids),256,4,192) or h.dtype != np.complex64 or not np.isfinite(h).all():
            raise ValueError("Finite complex64[N,256,4,192] required")
        return result

    def read_reference(self,query,*,explicit_evaluation_enabled=False):
        if not explicit_evaluation_enabled:raise PermissionError("Reference reading remains disabled during recovery")
        self._validate_query(query)
        source=PATHS.round2/"Round2_Train_Channel.npy"
        stat=source.stat()
        array=np.load(source,mmap_mode="r",allow_pickle=False)
        if array.shape != (4000,256,4,192) or array.dtype != np.complex64:raise ValueError("Official channel identity/header")
        channel=np.array(array[query.row_ids],copy=True)
        if not np.isfinite(channel).all():raise ValueError("Nonfinite reference")
        ids=np.array(query.row_ids,copy=True)
        channel.setflags(write=False);ids.setflags(write=False)
        return BoundChannelReference(channel,ids,self.inputs.identity,query.input_namespace,
                                     str(source),(stat.st_size,stat.st_mtime_ns,stat.st_ino))

    def evaluate_output(self,result,query,truth,*,explicit_evaluation_enabled=False):
        if not explicit_evaluation_enabled:raise PermissionError("Scoring remains disabled during recovery")
        self.validate_output(result,query)
        if type(truth) is not BoundChannelReference or truth.input_identity != self.inputs.identity or not np.array_equal(truth.row_ids,query.row_ids):
            raise ValueError("Reference must carry its own canonical row/input identity")
        source=PATHS.round2/"Round2_Train_Channel.npy"
        stat=source.stat()
        if truth.source_path != str(source) or truth.source_stat != (stat.st_size,stat.st_mtime_ns,stat.st_ino):
            raise ValueError("Reference channel source changed")
        if truth.channel.shape != result["channel"].shape or truth.channel.dtype != np.complex64 or not np.isfinite(truth.channel).all():
            raise ValueError("Reference channel shape/type")
        reject_online_test_as_oof(prediction_namespace=result["prediction_namespace"],truth_namespace=truth.namespace)
        return aggregate_statistics(point_metric_statistics(result["channel"],truth.channel))

    def eligibility(self):
        return dict(method="newrf",adaptation="canonical-FIT geometry-assisted multiport full-H",
                    direction_qualified=self.model._qualify()["direction_contract_qualified"],
                    full_H=True,pas=True,pdp=True,nmse=True,score=True,
                    original_paper_reproduction=False,performance_validated=False,
                    metadata=self.model.metadata())

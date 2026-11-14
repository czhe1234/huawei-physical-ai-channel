"""The V16 seventeen-expert families with the actual measured array geometry."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .common import C0, check_prediction

@dataclass(frozen=True)
class Expert:
    name: str
    family: str
    k: int
    ridge: float = .3

def expert_configs():
    out = [Expert("nearest", "nearest", 1)]
    out += [Expert("raw_idw_k"+str(k), "raw", k) for k in [4,8,16,24]]
    out += [Expert("radial_idw_k"+str(k), "radial", k) for k in [4,8,16,24]]
    out += [Expert("radial_affine_k"+str(k), "affine", k) for k in [8,16,24,32]]
    out += [Expert("physical_steered_affine_k"+str(k), "steered", k) for k in [8,16,24,32]]
    return out

def affine_weights(relative, distance, ridge):
    w = 1/np.maximum(distance, 1e-6)**2; w /= w.sum()
    x = np.c_[np.ones(len(w)), relative]
    normal = x.T @ (x*w[:,None])
    regularizer = ridge*max(np.trace(normal)/3, 1e-12)*np.diag([0.,1.,1.])
    return np.linalg.solve(normal+regularizer, x.T*w)[0]

class CarrierBank:
    def __init__(self, bank):
        self.bank = bank
        self.configs = expert_configs()
        self.coefficient = np.zeros((17,2),np.complex128)
        self.coefficient[0] = 1.
        self.coefficients_fitted = False

    def experts(self, query, exclude_same_row=True):
        source = self.bank
        out = np.zeros((len(query.row_ids),17,2,8,4,source.spec.frequency_count), np.complex64)
        frequency = source.spec.frequencies
        for row,(rid,position) in enumerate(zip(query.row_ids,query.position)):
            distance = np.linalg.norm(source.position[:,:2]-position[:2],axis=1)
            allowed = source.row_ids != rid if exclude_same_row else np.ones(len(distance),bool)
            ids = np.flatnonzero(allowed)
            ids = ids[np.lexsort((source.row_ids[ids],distance[ids]))]
            if not len(ids): raise ValueError("No allowed donor after excluding own measurement")
            for e,config in enumerate(self.configs):
                chosen = ids[:min(config.k,len(ids))]
                d = distance[chosen]
                if config.family=="nearest":
                    out[row,e] = source.channel[chosen[0]];continue
                weight = 1/np.maximum(d,1e-6)**2;weight/=weight.sum()
                if config.family in ["affine","steered"]:
                    if len(chosen)<3:raise ValueError("Affine expert requires at least3 distinct donors")
                    weight = affine_weights(source.position[chosen,:2]-position[:2],d,config.ridge)
                for panel in range(2):
                    channel = source.channel[chosen,panel].astype(np.complex128)
                    if config.family!="raw":
                        ranges = np.linalg.norm(source.position[chosen]-source.spec.center[panel],axis=1)
                        qr = np.linalg.norm(position-source.spec.center[panel])
                        channel *= np.exp(2j*np.pi*(ranges-qr)[:,None,None,None]*frequency[None,None,None,:]/C0)
                    if config.family=="steered":
                        unit = (source.position[chosen]-source.spec.center[panel])/ranges[:,None]
                        uq = (position-source.spec.center[panel])/qr
                        phase = 2*np.pi*source.spec.carrier_hz/C0 * np.einsum("ki,hvi->khv",uq[None]-unit,source.spec.element_offsets[panel])
                        channel *= np.exp(1j*phase)[...,None]
                    out[row,e,panel] = np.einsum("k,khvf->hvf",weight,channel)
        return out

    def predict(self, query):
        values = self.experts(query)
        mixed = np.einsum("nephvf,ep->nphvf",values,self.coefficient).astype(np.complex64)
        return check_prediction(dict(method="V16_measured_carrier_bank",channel=mixed,
                    source_ids=np.array(self.bank.row_ids),diagnostics=dict(experts=17,coefficients_fitted=self.coefficients_fitted)),query)

    def fit_synthetic_ridge(self, experts, target, ridge=.01):
        """Numerical helper. Caller must supply OOF fit predictions in real training."""
        x,y=np.asarray(experts),np.asarray(target)
        if x.ndim!=6 or x.shape[1:3]!=(17,2) or y.shape!=(len(x),*x.shape[2:]):
            raise ValueError("Expert fusion layout")
        for p in range(2):
            a=np.moveaxis(x[:,:,p],1,-1).reshape(-1,17).astype(np.complex128)
            b=y[:,p].reshape(-1).astype(np.complex128)
            gram=a.conj().T@a
            scale=max(float(np.trace(gram).real/17),1e-20)
            self.coefficient[:,p]=np.linalg.solve(gram+ridge*scale*np.eye(17),a.conj().T@b)
        self.coefficients_fitted=True

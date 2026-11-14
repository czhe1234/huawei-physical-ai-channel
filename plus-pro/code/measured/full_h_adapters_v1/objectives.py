"""Shared differentiable structural targets, without evaluation or truth alignment.

This is an untrained objective interface, not a new published score. The same
primitive transforms are available to both adapters and to the proposed method.
"""
from __future__ import annotations
import torch
from measured.hall_objective_trial_v1.objectives import describe,metrics

TERM_NAMES={"pas","relative_pdp","receive_energy","receive_delay_joint"}

def structural_terms(prediction,fit_target):
    if prediction.ndim!=5 or tuple(prediction.shape[1:4])!=(2,8,4) or prediction.shape[-1]<2 or prediction.shape!=fit_target.shape or prediction.dtype not in (torch.complex64,torch.complex128) or fit_target.dtype not in (torch.complex64,torch.complex128):
        raise ValueError("Physical complex H prediction/target shape or dtype mismatch")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(fit_target).all()):
        raise ValueError("Nonfinite H")
    # Source primitive independently self-references each panel of each H.
    values=metrics(prediction,fit_target)
    return dict(pas=1-values["pas_cosine"],relative_pdp=1-values["relative_pdp_cosine"],
                receive_energy=values["receive_energy_TV"],
                receive_delay_joint=values["receive_delay_joint_TV"])

def structural_objective(prediction,fit_target,weights):
    # Training weights are deliberately not selected by a correctness-only pass.
    if set(weights)!=TERM_NAMES or any(not torch.isfinite(torch.tensor(v)) or v<0 for v in weights.values()) or sum(weights.values())<=0:
        raise ValueError("Finite nonnegative predeclared structural weights required")
    terms=structural_terms(prediction,fit_target)
    return sum(terms[k]*weights[k] for k in weights)

def raw_complex_objective(prediction,fit_target,*,independent_absolute_reference_verified=False):
    # This stage has no accepted absolute common phase/time target.
    if not independent_absolute_reference_verified:
        raise PermissionError("Raw complex regression requires an independently verified reference contract")
    raise NotImplementedError("An absolute-reference training protocol has not been accepted")

def independent_features(h):
    """Per-H features; accepts no paired truth and does not alter raw output."""
    return describe(h)

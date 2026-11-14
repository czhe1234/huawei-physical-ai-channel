"""Measurement-native spectra, V16-style constrained synthesis, and metrics.

Channel layout is [position, physical_panel, horizontal, vertical, frequency].
Panels are evaluated independently; no inter-panel phase-coherence assumption
is imposed. Ground-truth-assisted phase/delay alignment exists only in metrics.
"""
from __future__ import annotations

import numpy as np
import torch

EPS = 1e-30


def normalize(value, dims):
    value = value.clamp_min(0)
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(total > EPS, value/total.clamp_min(EPS), torch.zeros_like(value))


def pas(channel):
    return normalize(torch.fft.fft2(channel, dim=(2, 3)).abs().square(), (2, 3))


def pdp(channel):
    return normalize(torch.fft.ifft(channel, dim=-1).abs().square(), (-1,))


def cosine(a, b, dims):
    numerator = (a*b).sum(dim=dims)
    denominator = torch.sqrt(a.square().sum(dim=dims)*b.square().sum(dim=dims))
    return torch.where(denominator > 0, numerator/denominator.clamp_min(EPS), torch.zeros_like(numerator))


def spectral_kernels(coordinate, horizontal, vertical, frequency):
    result = []
    for axis, n in enumerate((horizontal, vertical, frequency)):
        index = torch.arange(n, device=coordinate.device, dtype=torch.float32)
        wave = torch.exp((2j if axis != 2 else -2j)*torch.pi*coordinate[...,axis,None]*index/n)
        transform = torch.fft.fft(wave,dim=-1) if axis!=2 else torch.fft.ifft(wave,dim=-1)
        result.append(normalize(transform.abs().square(), (-1,)))
    return result


def render(mass, coordinate, horizontal, vertical, frequency):
    """One UE TX port per panel; candidate mass is not reassigned across panels."""
    h,v,d = spectral_kernels(coordinate,horizontal,vertical,frequency)
    p = normalize(torch.einsum('bpk,bpkh,bpkv->bphv',mass,h,v),(2,3))
    delay = normalize(torch.einsum('bpk,bpkd->bpd',mass,d),(-1,))
    return p[...,None].expand(-1,-1,-1,-1,frequency), delay[:,:,None,None,:].expand(-1,-1,horizontal,vertical,-1)


def unit_complex(value):
    magnitude = value.abs()
    return torch.where(magnitude > EPS, value/magnitude.clamp_min(EPS), torch.ones_like(value))


def safe_sqrt(value):
    """Keep exact zeros while avoiding an infinite derivative at zero power."""
    return torch.where(value>0,value.clamp_min(1e-20).sqrt(),torch.zeros_like(value))


def project(channel, target_pas, target_pdp, alpha_pas=0.2, alpha_pdp=0.2, rounds=4):
    """V16 square-root endpoint blending and alternating exact magnitudes."""
    if alpha_pas == 0 and alpha_pdp == 0:
        return channel
    original_power = channel.abs().square().sum(dim=(2,3,4),keepdim=True)
    desired_pas = normalize(((1-alpha_pas)*safe_sqrt(pas(channel))+alpha_pas*safe_sqrt(target_pas)).square(),(2,3))
    desired_pdp = normalize(((1-alpha_pdp)*safe_sqrt(pdp(channel))+alpha_pdp*safe_sqrt(target_pdp)).square(),(-1,))
    value = channel
    for _ in range(rounds):
        angle = torch.fft.fft2(value,dim=(2,3))
        slice_energy = angle.abs().square().sum(dim=(2,3),keepdim=True)
        value = torch.fft.ifft2(safe_sqrt(desired_pas*slice_energy)*unit_complex(angle),dim=(2,3))
        delay = torch.fft.ifft(value,dim=-1)
        link_energy = delay.abs().square().sum(dim=-1,keepdim=True)
        value = torch.fft.fft(safe_sqrt(desired_pdp*link_energy)*unit_complex(delay),dim=-1)
        current = value.abs().square().sum(dim=(2,3,4),keepdim=True)
        value = value*safe_sqrt(original_power/current.clamp_min(EPS))
    return value


@torch.inference_mode()
def metric_points(pred, truth):
    pred,truth = pred.to(torch.complex64),truth.to(torch.complex64)
    if pred.shape!=truth.shape or pred.ndim!=5:
        raise ValueError('Expected matching [point,panel,H,V,F] channels')
    dims=(2,3,4)
    p,t = pas(pred),pas(truth)
    d,td = pdp(pred),pdp(truth)
    pas_cos=cosine(p,t,(2,3)).mean(dim=-1)
    pdp_cos=cosine(d,td,(-1,)).mean(dim=(2,3))
    pred_energy=pred.abs().square().sum(dim=dims,dtype=torch.float64)
    energy=truth.abs().square().sum(dim=dims,dtype=torch.float64)
    error=(pred-truth).abs().square().sum(dim=dims,dtype=torch.float64)
    # One scalar per whole snapshot: measures ambiguity of the mobile TX phase.
    dot=(pred.conj()*truth).sum(dim=(1,2,3,4))
    aligned=pred*unit_complex(dot)[:,None,None,None,None]
    aligned_error=(aligned-truth).abs().square().sum(dim=dims,dtype=torch.float64)
    # More permissive diagnostic: a scalar per physical panel, still not per antenna.
    dot_panel=(pred.conj()*truth).sum(dim=dims)
    panel_aligned=pred*unit_complex(dot_panel)[:,:,None,None,None]
    panel_error=(panel_aligned-truth).abs().square().sum(dim=dims,dtype=torch.float64)
    # Maximize shared time/phase alignment over the actually sampled DFT grid.
    cross_f=(pred.conj()*truth).sum(dim=(2,3))
    alignment_grid=8*pred.shape[-1]
    best_dot=torch.fft.ifft(cross_f,n=alignment_grid,dim=-1).abs().amax(dim=-1)*alignment_grid
    phase_delay_error=(pred_energy+energy-2*best_dot.to(torch.float64)).clamp_min(0)
    # DFT beam selection is evaluated on measured H, using only prediction to choose.
    pred_beam=torch.fft.fft2(pred,dim=(2,3)).abs().square().mean(dim=-1).flatten(2).argmax(dim=-1)
    true_beam_power=torch.fft.fft2(truth,dim=(2,3)).abs().square().mean(dim=-1).flatten(2)
    selected=true_beam_power.gather(2,pred_beam[...,None])[...,0]
    oracle=true_beam_power.amax(dim=-1)
    beam_loss=-10*torch.log10((selected/oracle.clamp_min(EPS)).clamp_min(EPS))
    power_error=10*torch.log10((pred_energy/energy.clamp_min(EPS)).clamp_min(EPS))
    # Relative PDP alignment is a separate diagnostic, not a deployable correction.
    a=d.mean(dim=(2,3));b=td.mean(dim=(2,3))
    corr=torch.fft.ifft(torch.fft.fft(a,dim=-1).conj()*torch.fft.fft(b,dim=-1),dim=-1).real
    relative_pdp=corr.amax(dim=-1)/(a.square().sum(-1)*b.square().sum(-1)).sqrt().clamp_min(EPS)
    # Array covariance preserves physically coherent relationships inside each panel.
    hp=pred.flatten(2,3);ht=truth.flatten(2,3)
    rp=hp@hp.conj().transpose(-1,-2)/pred.shape[-1]
    rt=ht@ht.conj().transpose(-1,-2)/pred.shape[-1]
    rp=rp/(rp.diagonal(dim1=-2,dim2=-1).real.sum(-1)[...,None,None].clamp_min(EPS))
    rt=rt/(rt.diagonal(dim1=-2,dim2=-1).real.sum(-1)[...,None,None].clamp_min(EPS))
    covariance=(rp-rt).abs().square().sum((-1,-2))/rt.abs().square().sum((-1,-2)).clamp_min(EPS)
    data=dict(pas=pas_cos,pdp=pdp_cos,relative_pdp=relative_pdp,energy=energy,error=error,
              phase_error=aligned_error,panel_phase_error=panel_error,phase_delay_error=phase_delay_error,
              beam_loss_db=beam_loss,power_error_db=power_error,covariance_error=covariance)
    return {k:v.cpu().numpy() for k,v in data.items()}


def aggregate(points):
    energy=float(np.sum(points['energy']))
    result={k:float(np.mean(points[k])) for k in ['pas','pdp','relative_pdp','beam_loss_db','covariance_error']}
    for key,out in [('error','nmse'),('phase_error','phase_aligned_nmse'),
                    ('panel_phase_error','panel_phase_aligned_nmse'),('phase_delay_error','panel_phase_delay_aligned_nmse')]:
        result[out]=float(np.sum(points[key])/max(energy,EPS))
    result['power_mae_db']=float(np.mean(np.abs(points['power_error_db'])))
    result['structural_score']=0.5*(result['pas']+result['pdp'])
    result['adapted_competition_score']=0.4*result['pas']+0.4*result['pdp']+0.2/(1+result['nmse'])
    result['positions']=int(len(points['pas']))
    return result


def check_invariants():
    generator=torch.Generator().manual_seed(20261015)
    h=torch.complex(torch.randn(3,2,8,4,32,generator=generator),torch.randn(3,2,8,4,32,generator=generator))
    assert torch.equal(project(h,pas(h),pdp(h),0,0),h)
    identity=project(h,pas(h),pdp(h),1,1)
    assert torch.allclose(identity,h,atol=8e-6,rtol=8e-6)
    altered=h*torch.exp(torch.tensor(1.2j))
    points=metric_points(altered,h)
    assert np.max(np.abs(points['pas']-1))<2e-6
    assert np.max(np.abs(points['pdp']-1))<2e-6
    assert float(points['error'].sum())>1
    assert float(points['phase_error'].sum())<1e-8
    ramp=torch.exp(2j*torch.pi*torch.arange(32)*5/32)
    shifted=metric_points(h*ramp,h)
    assert float(shifted['phase_delay_error'].sum())/float(shifted['energy'].sum())<3e-6
    return 'zero-action, identity projection, global-phase invariance and shared-delay alignment passed'


if __name__=='__main__':
    print(check_invariants())

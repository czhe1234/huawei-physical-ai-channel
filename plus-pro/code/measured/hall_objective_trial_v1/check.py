"""Forward/gradient checks before objective-only training."""
import json,torch,numpy as np
from measured.hall_metric_protocol_v1.metrics import compare
from .objectives import metrics,KEYS

def checks():
    torch.set_num_threads(2)
    g=torch.Generator().manual_seed(3028)
    p=torch.complex(torch.randn((2,2,8,4,32),generator=g,dtype=torch.float64),torch.randn((2,2,8,4,32),generator=g,dtype=torch.float64))
    t=torch.complex(torch.randn((2,2,8,4,32),generator=g,dtype=torch.float64),torch.randn((2,2,8,4,32),generator=g,dtype=torch.float64))
    expected=compare(p.numpy(),t.numpy());got=metrics(p,t)
    forward={k:abs(float(got[k])-expected[k]) for k in KEYS}
    assert max(forward.values())<1e-11,forward
    def clock(x,shift):
        k=torch.arange(x.shape[-1],dtype=x.real.dtype)
        return x*torch.exp(-2j*torch.pi*shift*k/x.shape[-1])
    loss=lambda x:sum(metrics(x,t)[k]*(1 if k.endswith("TV") else -1) for k in KEYS)
    x=p.clone().requires_grad_();value=loss(x);grad,=torch.autograd.grad(value,x)
    direction=torch.complex(torch.randn(p.shape,generator=g,dtype=torch.float64),torch.randn(p.shape,generator=g,dtype=torch.float64))
    eps=1e-5
    numeric=float((loss(p+eps*direction)-loss(p-eps*direction))/(2*eps))
    analytic=float((grad.conj()*direction).real.sum())
    difference=abs(numeric-analytic)
    assert difference<1e-7,difference
    theta=torch.tensor(.37,dtype=torch.float64,requires_grad=True)
    transformed=loss(clock(p,theta));gauge,=torch.autograd.grad(transformed,theta)
    assert abs(float(transformed-loss(p)))<1e-10
    assert abs(float(gauge))<1e-9,float(gauge)
    broken=torch.zeros_like(p);zero=metrics(broken,t)
    assert float(zero["pas_cosine"])==0 and float(zero["relative_pdp_cosine"])==0
    assert float(zero["receive_energy_TV"])==1 and float(zero["receive_delay_joint_TV"])==1
    empty=torch.zeros_like(p);empty[...,0]=1
    bad=metrics(empty,t);assert float(bad["relative_pdp_cosine"])==0
    # Masks/weights stay identical to NumPy when panels have unequal valid links.
    sparse=t.clone();sparse[:,0,:2]=0
    a=metrics(p,sparse);b=compare(p.numpy(),sparse.numpy())
    assert max(abs(float(a[k])-b[k]) for k in KEYS)<1e-11
    return dict(passed=True,numpy_forward_differences=forward,directional_derivative_difference=difference,
        common_fractional_delay_loss_delta=float(transformed-loss(p)),common_delay_directional_gradient=float(gauge),
        zero_prediction_and_invalid_reference_penalties=True,unequal_link_masks=True)
if __name__=="__main__":print(json.dumps(checks()))

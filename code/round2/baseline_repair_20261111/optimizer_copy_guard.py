"""Deep-copy optimizer loading with immutable-state and storage fingerprint guards."""
from __future__ import annotations
import copy,hashlib,json
import torch
def fingerprint(obj):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(b"tensor");h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            h.update(b"dict")
            for k in sorted(x,key=lambda z:(str(type(z)),repr(z))):visit(k);visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(str(type(x)).encode())
            for v in x:visit(v)
        else:h.update(str(type(x)).encode());h.update(repr(x).encode())
    visit(obj);return h.hexdigest()
def storages(obj):
    result=set()
    def visit(x):
        if torch.is_tensor(x):
            if x.numel():result.add((str(x.device),x.untyped_storage().data_ptr()))
        elif isinstance(x,dict):
            for v in x.values():visit(v)
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
    visit(obj);return result
def steps(state):
    values=[]
    for s in state["state"].values():
        if "step" in s:
            x=s["step"];values.append(float(x.item()) if torch.is_tensor(x) else float(x))
    return sorted(set(values))
def immutable_part(initial):
    return {k:initial[k] for k in ("model","optimizer","torch_rng","cuda_rng") if k in initial}
def assert_immutable(initial,expected):
    actual=fingerprint(immutable_part(initial));assert actual==expected,("Initial state mutated",actual,expected)
def load_optimizer(opt,state):
    expected=fingerprint(state);cloned=copy.deepcopy(state)
    assert not storages(cloned).intersection(storages(state))
    opt.load_state_dict(cloned)
    loaded=opt.state_dict()
    assert not storages(loaded).intersection(storages(state)),"Optimizer shares storage with immutable parent"
    assert fingerprint(loaded)==expected,"Optimizer load changed state content"
    assert fingerprint(state)==expected,"Optimizer load mutated parent"
    return dict(parent_optimizer_fingerprint=expected,loaded_optimizer_fingerprint=fingerprint(loaded),
        CPU_step_and_all_moment_storages_disjoint=True,actual_step_values=steps(loaded))
def cpu_reproduction():
    m=torch.nn.Parameter(torch.tensor([.3]));opt=torch.optim.AdamW([m],lr=.01)
    for _ in range(3):opt.zero_grad();m.square().sum().backward();opt.step()
    initial=copy.deepcopy(opt.state_dict());bad0=fingerprint(initial)
    a=torch.nn.Parameter(torch.tensor([.3]));bad=torch.optim.AdamW([a],lr=.01)
    bad.load_state_dict(initial);before=steps(initial);bad.zero_grad();a.square().sum().backward();bad.step()
    after=steps(initial);assert before==[3.] and after==[4.] and fingerprint(initial)!=bad0
    fixed=copy.deepcopy(opt.state_dict());f=fingerprint(fixed)
    b=torch.nn.Parameter(torch.tensor([.3]));good=torch.optim.AdamW([b],lr=.01)
    evidence=load_optimizer(good,fixed);good.zero_grad();b.square().sum().backward();good.step()
    assert fingerprint(fixed)==f and steps(fixed)==[3.] and steps(good.state_dict())==[4.]
    return dict(status="passed",torch_version=torch.__version__,
        old_load_shares_and_mutates_CPU_state=True,old_parent_steps_before=before,old_parent_steps_after=after,
        deepcopy_parent_preserved=True,fixed_parent_steps=steps(fixed),fixed_live_steps=steps(good.state_dict()),
        evidence=evidence)

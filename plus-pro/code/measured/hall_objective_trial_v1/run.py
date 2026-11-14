"""Three matched finite-budget finetunes; final endpoint is predeclared."""
from __future__ import annotations
import copy,csv,hashlib,json,os,platform,re,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from measured.hall_adapters.common import sha256,save_json
from measured.hall_structured_v2.data import SouthDevelopment,slice_prepared
from measured.hall_structured_v2.foundation import prepare
from measured.hall_structured_v2.observables import FitPower
from measured.hall_structured_v2.model import HallStructured
from measured.hall_structured_v2.config import Config,ALLOWED_GPUS
from measured.hall_structured_v2.train import atomic_torch,atomic_npz,rng_state,restore_rng
from measured.hall_metric_protocol_v1.metrics import compare
from .objectives import objective,metrics,describe,KEYS
from .check import checks
REPORT=PATHS.analysis/"results/hall_south_objective_trial_20261028"
WORK=PATHS.artifacts/"measured/hall_sw_objective_trial_20261028"
SOURCE=PATHS.code_root/"measured/hall_objective_trial_v1"
OLD_REPORT=PATHS.analysis/"results/hall_south_structured_development_v2"
OLD_WORK=PATHS.artifacts/"measured/hall_sw_structured_development_v2"
SEED=20261028
VARIANTS=("old","pdp","joint")
EPOCHS=40

def snapshot():
    out={}
    for base in [PATHS.code_root,PATHS.analysis,PATHS.artifacts/"measured",PATHS.datasets,PATHS.root/"final_code"]:
      for p in base.rglob("*"):
        if not p.is_file() or p.is_symlink() or "__pycache__" in p.parts or any(d in p.parents for d in [REPORT,WORK]):continue
        st=p.stat();out[str(p.relative_to(PATHS.root))]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return out

def state_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def seed():
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);np.random.seed(SEED)
    torch.set_num_threads(2)

@torch.no_grad()
def predict(model,data):
    model.eval();out=[]
    for first in range(0,len(data.row_ids),32):
        out.append(model(slice_prepared(data,np.arange(first,min(first+32,len(data.row_ids)))))["channel"].detach())
    return torch.cat(out)

def main():
    visible=os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in ALLOWED_GPUS:raise RuntimeError("Require one permitted full GPU UUID")
    if torch.cuda.device_count()!=1:raise RuntimeError("Unexpected GPU visibility")
    uuid=str(torch.cuda.get_device_properties(0).uuid)
    assert uuid in (visible,visible.removeprefix("GPU-")),(uuid,visible)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.__version__.split("+")[0]!="2.5.1" or np.__version__!="1.26.4":raise RuntimeError("Runtime changed")
    check=checks();seed();dev=torch.device("cuda:0")
    if not (REPORT/"contract.json").exists():
        REPORT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
        save_json(REPORT/"operator_checks.json",check)
        save_json(REPORT/"preservation_before.json",snapshot())
        ledger=PATHS.root/"experiment_records/EXPERIMENT_RECORD.md";prefix=ledger.read_bytes()
        (WORK/"ledger_before.bin").write_bytes(prefix)
        item=max(map(int,re.findall(r"^## (\d+)\.",prefix.decode(),re.M)))+1
        old=json.loads((OLD_REPORT/"ours_default.json").read_text())
        contract=dict(primary_hypothesis="Direct self-referenced PDP supervision can reduce the measured delay regression without sacrificing PAS/receive shares.",
            type="Bounded matched objective-only optimization diagnostic, not independent confirmation",
            variants={"old":"old structural+0.1power+0.0001residual","pdp":"old+(1-relativePDPcos)",
                "joint":"old+(1-relativePDPcos)+(1-PAScos)+0.25jointTV"},
            initialization="All reload identical old-selected ours_default epoch25 weights and FIT statistics; reset identical AdamW",
            starting_checkpoint_sha256=sha256(OLD_WORK/"models/ours_default/best.pt"),
            epochs=EPOCHS,updates=120,batch=32,lr=1e-4,weight_decay=1e-4,gradient_clip=2,
            selection="Fixed final epoch40 primary; report common initial and all10-epoch trajectories; no new checkpoint selection",
            seed=SEED,data_identity=old["contract"]["data_identity"],fit_rows=old["contract"]["fit_rows"],validation_rows=old["contract"]["validation_rows"],
            screening={"PDP_gain_vs_old_final_min":.01,"PAS_drop_vs_old_final_max":.005,"receiveTV_increase_vs_old_final_max":.005,
                "interpretation":"Engineering screens only; also report initial endpoint and learned foundation to avoid mistaking A regression for recovery."},
            max_model_minutes=8,max_stage_minutes=25,formal_test=False,north=False,new_RT=False,architecture_changed=False,
            references="Frozen new metric protocol; prediction/truth self-references computed separately; prediction arg differentiated, no fitted alignment",
            source_sha256={str(p.relative_to(PATHS.root)):sha256(p) for p in SOURCE.glob("*.py")},
            old_model_sources={k:v for k,v in old["contract"]["source_sha256"].items() if "/hall_structured_v2/" in k},
            ledger_item=item)
        save_json(REPORT/"contract.json",contract)
        with ledger.open("a") as f:f.write(f"\n\n## {item}. 2026-10-28 South matched objective-only finetuning trial\n\n### Contract\n- User authorizes the bounded objective-alignment trial. Same epoch25 start, architecture,96 FIT/48 validation, seed/batch/order,40epochs/120updates and AdamW;3 objectives, fixed final endpoint.\n- No north, formal TEST, new RT, architecture change or checkpoint selection. Old sources and outputs preserved. Contract: analysis/results/hall_south_objective_trial_20261028/contract.json.\n")
    contract=json.loads((REPORT/"contract.json").read_text())
    assert contract["source_sha256"]=={str(p.relative_to(PATHS.root)):sha256(p) for p in SOURCE.glob("*.py")}
    for name,digest in contract["old_model_sources"].items():assert sha256(PATHS.root/name)==digest
    fixed=OLD_WORK/"models/ours_default/best.pt";assert sha256(fixed)==contract["starting_checkpoint_sha256"]
    checkpoint=torch.load(fixed,map_location="cpu",weights_only=False)
    old=json.loads((OLD_REPORT/"ours_default.json").read_text())
    reader=SouthDevelopment();assert reader.data.identity==contract["data_identity"]
    bank=reader.fit();qf=reader.query("fit");qv=reader.query("validation")
    assert qf.row_ids.tolist()==contract["fit_rows"] and qv.row_ids.tolist()==contract["validation_rows"]
    fit=prepare(bank,qf,16,"block",reader.geometry(qf.row_ids)).to(dev)
    val=prepare(bank,qv,16,"self",reader.geometry(qv.row_ids)).to(dev)
    power=FitPower.from_fit(bank.channel,bank.row_ids).to(dev);target=torch.tensor(np.array(bank.channel),device=dev)
    model=HallStructured(bank.spec,power,Config(**old["contract"]["config"])).to(dev)
    model.load_state_dict(checkpoint["state_dict"]);model.eval()
    with torch.no_grad():initial_val=predict(model,val).cpu().numpy()
    assert all(x["role"]=="fit" for x in reader.access)
    reader.allow_validation=True;truth=reader.reference("validation");truth_gpu=torch.tensor(truth,device=dev)
    initial=compare(initial_val,truth)
    start=time.monotonic()
    for variant in VARIANTS:
        resultfile=REPORT/(variant+".json")
        if resultfile.exists() and json.loads(resultfile.read_text()).get("complete"):
            print("REUSE",variant,flush=True);continue
        seed();model=HallStructured(bank.spec,power,Config(**old["contract"]["config"])).to(dev)
        model.load_state_dict(checkpoint["state_dict"])
        # Model construction consumes RNG; all variants follow this same sequence.
        initial_digest=state_digest(model.state_dict())
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
        folder=WORK/variant;folder.mkdir(parents=True,exist_ok=True)
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        curve=[];first=1;steps=0;grad_health={};orders=[]
        if (folder/"last.pt").exists():
            saved=torch.load(folder/"last.pt",map_location=dev,weights_only=False)
            model.load_state_dict(saved["state_dict"]);opt.load_state_dict(saved["optimizer"])
            restore_rng(saved["rng"],dev);first=saved["epoch"]+1;curve=saved["curve"];steps=saved["steps"];orders=saved["orders"]
        else:
            model.eval()
            with torch.no_grad():
                probe=slice_prepared(fit,np.arange(4));out=model(probe)
            for kind in VARIANTS:
                model.zero_grad(set_to_none=True);out=model(probe)
                loss=objective(out["channel"],target[:4],None,power,out["residual"],kind);loss.backward()
                assert torch.isfinite(loss)
                norms={n:float(p.grad.abs().max()) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
                assert norms and all(np.isfinite(x) for x in norms.values()) and max(norms.values())>0
                grad_health[kind]=dict(loss=float(loss.detach()),gradient_tensors=len(norms),max_gradient=max(norms.values()))
            seed()
            # Reset RNG exactly to post-construction state, shared across variants.
            del model
            model=HallStructured(bank.spec,power,Config(**old["contract"]["config"])).to(dev);model.load_state_dict(checkpoint["state_dict"])
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
            save_json(folder/"gradient_health.json",grad_health)
            curve.append(dict(epoch=0,steps=0,validation={k:initial[k] for k in KEYS}))
        for epoch in range(first,EPOCHS+1):
            model.train();losses=[];order=torch.randperm(96);orders.append(order.tolist())
            for ids in order.split(32):
                if time.monotonic()-started>480 or time.monotonic()-start>1500:raise TimeoutError("Bounded run; restart resumes saved epoch")
                model.zero_grad(set_to_none=True);out=model(slice_prepared(fit,ids.numpy()))
                loss=objective(out["channel"],target[ids],None,power,out["residual"],variant)
                if not torch.isfinite(loss):raise FloatingPointError("Nonfinite training objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],2,error_if_nonfinite=True)
                opt.step();steps+=1;losses.append(float(loss.detach()))
            if epoch%10==0 or epoch==1:
                pred=predict(model,val);fit_pred=predict(model,fit)
                with torch.no_grad():
                    vm={k:float(v) for k,v in metrics(pred,truth_gpu).items()}
                    fm={k:float(v) for k,v in metrics(fit_pred,target).items()}
                curve.append(dict(epoch=epoch,steps=steps,train_objective=float(np.mean(losses)),fit=fm,validation=vm,seconds=time.monotonic()-started))
                atomic_torch(folder/"last.pt",dict(epoch=epoch,steps=steps,state_dict=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(dev),curve=curve,orders=orders))
                save_json(REPORT/(variant+"_trajectory.json"),dict(complete=False,curve=curve))
                print("EPOCH",variant,epoch,EPOCHS,"PDP",round(vm["relative_pdp_cosine"],5),"PAS",round(vm["pas_cosine"],5),flush=True)
        pred=predict(model,val).cpu().numpy();fit_pred=predict(model,fit).cpu().numpy()
        final=compare(pred,truth);fit_score=compare(fit_pred,bank.channel)
        with torch.no_grad():torch_score={k:float(v) for k,v in metrics(torch.tensor(pred,device=dev),truth_gpu).items()}
        arithmetic_delta={k:abs(torch_score[k]-final[k]) for k in KEYS}
        assert max(arithmetic_delta.values())<2e-5,arithmetic_delta
        assert steps==120
        atomic_torch(folder/"final.pt",dict(state_dict=model.state_dict(),epoch=EPOCHS,steps=steps,contract=contract,variant=variant))
        atomic_npz(folder/"final_channel.npz",validation=pred,validation_ids=qv.row_ids,fit=fit_pred,fit_ids=qf.row_ids)
        result=dict(complete=True,variant=variant,epoch=EPOCHS,steps=steps,initial_state_sha256=initial_digest,order_sha256=hashlib.sha256(np.array(orders,dtype=np.int64).tobytes()).hexdigest(),
            initial_validation={k:initial[k] for k in KEYS},final_validation=final,final_fit=fit_score,curve=curve,
            torch_numpy_metric_differences=arithmetic_delta,seconds=time.monotonic()-started,
            peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,environment=dict(python=platform.python_version(),
                torch=torch.__version__,numpy=np.__version__,GPU_UUID=visible,GPU_name=torch.cuda.get_device_name(0)),
            architecture_changed=False,formal_test=False,north=False)
        save_json(resultfile,result)
        print("DONE",variant,json.dumps({k:final[k] for k in KEYS}),"seconds",round(result["seconds"],1),flush=True)
    save_json(REPORT/"access.json",dict(access=reader.access,formal_test=False,north=False))
    print("ALL_RUNS_COMPLETE",flush=True)

if __name__=="__main__":
    try:main()
    except BaseException:
        REPORT.mkdir(parents=True,exist_ok=True)
        save_json(REPORT/"failure.json",dict(traceback=traceback.format_exc()))
        raise

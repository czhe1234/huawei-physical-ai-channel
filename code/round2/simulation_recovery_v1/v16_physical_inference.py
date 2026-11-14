"""Frozen V16 20-checkpoint inference from row-bound n24 geometry statistics.

Does not read any query CSI, truth spectrum, zero-channel flag or training label.
All 500 positions are queried. Historical fold routing is not strict qualification.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
import types

import numpy as np
import v16_recovery as recovery
from v16_recovery import (PATHS, ARTIFACT_ROOT, REPORT_ROOT, BUNDLE, SCHEMA, SOURCE_HASHES,
    source_namespace, canonical, sha256, portable, file_identity, get_torch, ORIGINAL_SEEDS,
    safe_checkpoint, protected_output, atomic_json, save_stage)

CONFIGS=("primary_d16_base_v4lambda_1m","primary_d10_scatter015_v4lambda_1m")
ALIASES=dict(zip(CONFIGS,("base","scatter")))


class Statistics:
    def __init__(self,manifest: Path):
        self.path=manifest
        self.info=json.loads(manifest.read_text())
        if self.info.get("schema")!="v16-n24-physical-statistics-v1":
            raise ValueError("Physical statistics schema differs")
        ids,_=canonical()
        if self.info.get("row_ids")!=ids.tolist():raise ValueError("Exact canonical row order required")
        if self.info.get("source_configs")!=list(CONFIGS):raise ValueError("Base/Scatter configuration differs")
        if self.info.get("seed_count")!=24 or self.info.get("seeds")!=list(ORIGINAL_SEEDS):
            raise ValueError("Genuine n24 statistics are required")
        from v16_prior_identity import validate_prior_identity
        self.provenance=validate_prior_identity(manifest,self.info)
        self.arrays={}
        for name,item in self.info["arrays"].items():
            path=PATHS.resolve(item["path"])
            if sha256(path)!=item["sha256"]:raise ValueError(f"Hash differs: {name}")
            a=np.load(path,mmap_mode="r")
            if len(a)!=500:raise ValueError(f"Not row-bound canonical500: {name}")
            self.arrays[name]=a
        self.ids=ids
        for source in ("base","scatter"):
            for domain,shape,countshape in (
                ("pas",(500,2,16,8,4,192),(500,4,192)),
                ("pdp",(500,256,4,192),(500,256,4)),
                ("ad",(500,16,8,192),None),
                ("log_power",(500,),None),("path_count",(500,),None)):
                for stat in (("mean",) if domain=="ad" else ("mean","m2","count")):
                    a=self.get(source,domain,stat)
                    expected=countshape if stat=="count" and countshape else shape
                    if a.shape!=expected:raise ValueError(f"Shape differs: {source}.{domain}.{stat} {a.shape}")
                    for lo in range(0,500,16):
                        if not np.isfinite(a[lo:lo+16]).all():raise ValueError("Nonfinite statistics")
                    if stat=="count" and (np.any(a<0) or np.any(a>24)):
                        raise ValueError("Invalid n24 count")
        geom=self.arrays["geometry"]
        if geom.ndim!=2:raise ValueError("Geometry must be [500,G]")
        # Geometry descriptors have no H dependence; names/provenance must be explicit.
        if len(self.info.get("geometry_names",[]))!=geom.shape[1]:
            raise ValueError("Geometry column names missing")

    def get(self,source,domain,stat):
        if domain=="component_count":domain="path_count"
        return self.arrays[f"{source}.{domain}.{stat}"]

    def scalar(self,source):
        result=[]
        for kind in ("log_power","path_count"):
            mean=np.asarray(self.get(source,kind,"mean"),np.float32)
            m2=np.asarray(self.get(source,kind,"m2"),np.float32)
            count=np.asarray(self.get(source,kind,"count"),np.float32)
            variance=m2/np.maximum(count-1,1)
            if kind=="path_count":
                result.extend([np.log1p(np.maximum(mean,0)),np.log1p(np.sqrt(np.maximum(variance,0))),count/24.])
            else:result.extend([mean,np.sqrt(np.maximum(variance,0)),count/24.])
        return np.column_stack(result).astype(np.float32)

    def reliability(self,domain):
        output=np.empty((500,11),np.float32)
        for lo in range(0,500,4):
            hi=min(lo+4,500);v={}
            for source in ("base","scatter"):
                mean=np.asarray(self.get(source,domain,"mean")[lo:hi],np.float32)
                m2=np.asarray(self.get(source,domain,"m2")[lo:hi],np.float32)
                count=np.asarray(self.get(source,domain,"count")[lo:hi],np.float32)
                expanded=count[:,None,None,None,:,:] if domain=="pas" else count[...,None]
                variance=m2/np.maximum(expanded-1,1)
                flat=mean.reshape(hi-lo,-1).astype(np.float64)
                mass=np.maximum(flat.sum(1),1e-12);p=flat/mass[:,None]
                v[source]=dict(flat=flat,mass=mass,entropy=-np.sum(p*np.log(np.maximum(p,1e-30)),1)/math.log(flat.shape[1]),
                    peak=p.max(1),noise=np.sqrt(np.maximum(variance,0)).reshape(hi-lo,-1).sum(1,dtype=np.float64)/mass,
                    coverage=count.reshape(hi-lo,-1).mean(1)/24.)
            x,y=v["base"],v["scatter"]
            agreement=np.sum(x["flat"]*y["flat"],1)/np.maximum(
                np.sqrt(np.sum(x["flat"]**2,1)*np.sum(y["flat"]**2,1)),1e-12)
            ratio=np.log(np.maximum(x["mass"],1e-12)/np.maximum(y["mass"],1e-12))
            output[lo:hi]=np.column_stack([agreement,ratio,x["noise"],y["noise"],x["coverage"],y["coverage"],
                x["entropy"],y["entropy"],x["peak"],y["peak"],np.abs(ratio)])
        return output


class ZeroTarget:
    def __init__(self,torch,domain,device):self.torch=torch;self.domain=domain;self.device=device
    def __getitem__(self,key):
        n=len(key[0]);shape=(n,2,16,8) if self.domain=="pas" else (n,192)
        return self.torch.zeros(shape,dtype=self.torch.float32,device=self.device)


def frozen_definitions(torch):
    old=source_namespace("train_sionna_spectral_calibrator.py",
        ["TrainConfig","SpectralCalibrator","DomainBank"],dict(torch=torch,nn=torch.nn),["TINY"])
    return old


def old_bank(stats,domain,device,torch,old):
    bank=old.DomainBank.__new__(old.DomainBank)
    bank.domain=domain;bank.device=torch.device(device)
    ids=stats.ids
    rows=len(ids)
    if hasattr(stats,"site"):
        sites=np.asarray(stats.site)
    else:
        sys.path.insert(0,str(PATHS.round2_code))
        from round2.data import load_positions
        _,_,_,all_sites,_=load_positions()
        sites=all_sites[ids]
    bank.site=torch.as_tensor(sites,device=device,dtype=torch.long)
    bank.geometry=torch.as_tensor(np.array(stats.arrays["geometry"],copy=True),device=device,dtype=torch.float32)
    bank.scalar={s:torch.as_tensor(stats.scalar(s),device=device) for s in ("base","scatter")}
    bank.mean={};bank.m2={};bank.count={};bank._point_context={}
    bank.ad={}
    for s in ("base","scatter"):
        for stat,destination in (("mean",bank.mean),("m2",bank.m2)):
            a=torch.as_tensor(np.array(stats.get(s,domain,stat),copy=True),device=device,dtype=torch.bfloat16)
            if domain=="pas":a=a.permute(0,4,5,1,2,3).reshape(rows,768,2,16,8).contiguous()
            else:a=a.reshape(rows,1024,192).contiguous()
            destination[s]=a
        count=torch.as_tensor(np.array(stats.get(s,domain,"count"),copy=True),device=device,dtype=torch.uint8)
        bank.count[s]=count.reshape(len(ids),768 if domain=="pas" else 1024).contiguous()
    bank.truth=ZeroTarget(torch,domain,device)
    bank.slice_count=768 if domain=="pas" else 1024
    return bank


def joint_bank(stats,domain,device,torch):
    class NumpyProxy:
        def __getattr__(self,key):return getattr(np,key)
        def load(self,path,*args,**kwargs):
            if path=="geometry":return dict(test=stats.arrays["geometry"])
            source,split,kind,stat=str(path).split("/")
            if split!="test":raise PermissionError("Joint inference may not request train labels")
            return stats.get(ALIASES[source],kind,stat)
    def running_path(config,split,kind,stat):
        return f"{config}/{split}/{kind}/{stat}"
    ids=stats.ids
    if hasattr(stats,"site"):
        site=np.asarray(stats.site)
    else:
        sys.path.insert(0,str(PATHS.round2_code))
        from round2.data import load_positions
        _,_,_,all_sites,_=load_positions()
        site=all_sites[ids]
    def positions():return (None,None,None,site,site)
    ns=source_namespace("train_v9_joint_sionna_endpoints.py",["JointDomainBank"],
        dict(torch=torch,np=NumpyProxy(),BASE_CONFIG=CONFIGS[0],SCATTER_CONFIG=CONFIGS[1],
            SEEDS=24,MODE="joint",DOMAINS=("pas","pdp"),GEOMETRY_CACHE="geometry",
            running_path=running_path,load_positions=positions,
            spectral_reliability=lambda domain,split,force:stats.reliability(domain),
            source_scalar_matrix=lambda config,split:stats.scalar(ALIASES[config])),["TINY"])
    # Frozen constructor test branch is selected solely to omit truth loading;
    # provider maps its test symbol to explicitly canonical500 physical arrays.
    return ns.JointDomainBank(domain,"test",torch.device(device))


def predict(stats_manifest: Path,device="cpu",slice_batch=2048,resume=False):
    torch=get_torch(device);torch.set_num_threads(4)
    stats=Statistics(stats_manifest)
    old=frozen_definitions(torch)
    ids,folds=canonical();results={}
    for kind in ("spectral","joint"):
        for domain in ("pas","pdp"):
            bank=old_bank(stats,domain,device,torch,old) if kind=="spectral" else joint_bank(stats,domain,device,torch)
            checkpoints=[];models={}
            for fold in range(5):
                path=BUNDLE/"artifacts"/"round2"/"physical_ai_submission"/"models"/"base_field"/kind/f"{domain}_fold_{fold}.pt"
                ck=safe_checkpoint(path,torch)
                expected_mode="mean" if kind=="spectral" else "joint"
                if ck.get("fold")!=fold or ck.get("domain")!=domain or ck.get("mode")!=expected_mode:
                    raise ValueError("Checkpoint fold/domain/mode metadata differs")
                config=old.TrainConfig(**ck["config"])
                mean=np.asarray(ck["context_mean"],np.float32)
                std=np.asarray(ck["context_std"],np.float32)
                if np.any(std<=0):raise ValueError("Invalid context scale")
                # Infer channel count from the frozen stem, avoiding hard-coded
                # joint56/53 values while preserving all model weights exactly.
                weight=ck["state_dict"]["stem.weight"]
                channels=int(weight.shape[1])-int(config.context_channels)-(4 if domain=="pas" else 6)
                point_size=mean.size
                slice_size=bank.slice_context(torch.zeros(1,dtype=torch.long,device=device)).shape[1]
                model=old.SpectralCalibrator(domain,channels,point_size+slice_size,config).to(device)
                model.load_state_dict(ck["state_dict"]);model.eval()
                models[fold]=(model,torch.as_tensor(mean,device=device),torch.as_tensor(std,device=device))
                checkpoints.append(file_identity(path))
            shape=(500,2,16,8,4,192) if domain=="pas" else (500,256,4,192)
            identity=dict(kind="historical_fold_matched_spectral_inference",variant=kind,domain=domain,
                canonical_ids=ids.tolist(),canonical_fold=folds.tolist(),stats=file_identity(stats_manifest),physical_provenance=dict(canonical_row_sha256=stats.provenance["canonical_row_sha256"],query_groups=stats.provenance["query_groups"],source_states=stats.provenance["source_states"]),
                checkpoints=checkpoints,sources=dict(SOURCE_HASHES),fits_executed=False,
                strict_current_protocol=False,query_CSI_used=False,zero_flag_used=False,
                precision="frozen_original_bfloat16_input_buffers_FP32_network")
            def writer(out,start,commit):
                for point in range(start,500):
                    fold=int(folds[point]);model,mean,std=models[fold]
                    pieces=[]
                    for lo in range(0,bank.slice_count,slice_batch):
                        hi=min(lo+slice_batch,bank.slice_count)
                        pi=torch.full((hi-lo,),point,dtype=torch.long,device=device)
                        si=torch.arange(lo,hi,dtype=torch.long,device=device)
                        with torch.inference_mode():
                            if kind=="spectral":feature,context,baseline,_=bank.sample(pi,si,"mean",mean,std)
                            else:feature,context,baseline=bank.sample_input(pi,si,"joint",mean,std)
                            prediction,_=model(feature,context,baseline)
                        pieces.append(prediction.cpu().numpy())
                    value=np.concatenate(pieces)
                    if domain=="pas":value=value.reshape(4,192,2,16,8).transpose(2,3,4,0,1)
                    else:value=value.reshape(256,4,192)
                    out[point]=value.astype(np.float16)
                    if (point+1)%8==0 or point==499:commit(point+1)
            array,manifest=save_stage(f"{kind}_{domain}_endpoint",identity,shape,np.float16,writer,resume)
            results[f"{kind}_{domain}"]=manifest
            del bank,models,array
            if device=="cuda:0":torch.cuda.empty_cache()
    report=dict(schema="v16-spectral-bridge-v1",status="complete",outputs=results,
        source_statistics=file_identity(stats_manifest),performance_evaluated=False,strict_current_protocol=False)
    report_path=REPORT_ROOT/"v16_spectral_bridge.json"
    if report_path.exists():
        if json.loads(report_path.read_text())!=report:raise ValueError("Completed spectral bridge report identity differs")
    else:atomic_json(report_path,report)
    return report


def assemble_physical_manifest(statistics_manifest: Path,link_manifest: Path,output: Path):
    """Join certified fold spectra and independently row-bound raw-link inputs."""
    stat=json.loads(statistics_manifest.read_text())
    links=json.loads(link_manifest.read_text())
    ids,fold=canonical()
    if stat.get("row_ids")!=ids.tolist() or links.get("row_ids")!=ids.tolist():
        raise ValueError("Exact canonical row order required in both input manifests")
    if links.get("schema")!="v16-recovery-link-inputs-v1":
        raise ValueError("Raw-link producer schema missing")
    if len(links.get("link_context_names",[]))!=143:
        raise ValueError("Exact143 link context columns must be declared")
    arrays={};producers={}
    for key,stage in dict(old_pas="spectral_pas_endpoint",old_pdp="spectral_pdp_endpoint",
        joint_pas="joint_pas_endpoint",joint_pdp="joint_pdp_endpoint").items():
        p=ARTIFACT_ROOT/"historical_replay"/stage/"manifest.json"
        record=json.loads(p.read_text())
        if record.get("status")!="complete" or record.get("identity",{}).get("stats")!=file_identity(statistics_manifest):
            raise ValueError("Spectrum producer statistics identity differs")
        arrays[key]=file_identity(p.parent/"channels.npy")
        producers[key]=file_identity(p)
    arrays["base_pas_mean"]=stat["arrays"]["base.pas.mean"]
    arrays["scatter_pas_mean"]=stat["arrays"]["scatter.pas.mean"]
    for k in ["raw_link_pdp","link_context"]:
        arrays[k]=links["arrays"][k]
    result=dict(schema="v16-recovery-physical-inputs-v1",row_ids=ids.tolist(),seed_count=24,
        seeds=list(ORIGINAL_SEEDS),source_configs=list(CONFIGS),arrays=arrays,endpoint_producers=producers,
        statistic_source=file_identity(statistics_manifest),raw_link_source=file_identity(link_manifest),
        regenerated=True,exact_original_cache_equality_claimed=False,performance_evaluated=False)
    # The consumer revalidates producer identities and all bytes/shapes before use.
    if output.exists():
        if json.loads(output.read_text())!=result:raise ValueError("Existing physical manifest differs")
    else:atomic_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics",type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--slice-batch",type=int,default=2048)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    SOURCE_HASHES[portable(Path(__file__))]=sha256(Path(__file__))
    SOURCE_HASHES[portable(Path(recovery.__file__))]=sha256(Path(recovery.__file__))
    result=predict(args.statistics,args.device,args.slice_batch,args.resume)
    print(json.dumps(dict(status=result["status"],outputs=list(result["outputs"])),ensure_ascii=False))


if __name__=="__main__":main()

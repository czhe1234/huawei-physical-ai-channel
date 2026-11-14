"""Live geometry observer retaining exact frozen V16 token inputs.

No RT is run here. Caller supplies live full-port ScenePriorBuilder components.
Compact NPZs retain at most Base96/Scatter64 candidates per position per seed.
Aggregation uses the original identity/presence/scatter-pool/reserve rules.
"""
from __future__ import annotations
import json
import os
import fcntl
from pathlib import Path
import types
import numpy as np
from v16_recovery import (PATHS, ARTIFACT_ROOT, ORIGINAL_SEEDS, BUNDLE, canonical,
    source_namespace, protected_output, sha256, digest, atomic_json, file_identity, SOURCE_HASHES)

SOURCE_ROOT="../../final_code/V16_reorganized_submission/code/round2/"


def frozen_support():
    return source_namespace(SOURCE_ROOT+"physical_prior/scene_prior_data.py",
        ["extract_component_support"],dict())


def frozen_aggregation():
    return source_namespace(SOURCE_ROOT+"physical_ai_submission/scene_prior/component_aggregation.py",
        ["ComponentRecord","_records_for_point","_base_identity","_scatter_identity","_empty_arrays","_write_record"],
        dict())


class LiveTokenObserver:
    def __init__(self,output_dir: Path,query_ids: np.ndarray,site_ids: np.ndarray,prior_identity: dict):
        self.root=protected_output(Path(output_dir))
        self.root.mkdir(parents=True,exist_ok=True)
        self.query_ids=np.asarray(query_ids,np.int64)
        self.sites=np.asarray(site_ids,np.int8)
        if self.query_ids.ndim!=1 or self.sites.shape!=self.query_ids.shape or len(set(self.query_ids.tolist()))!=len(self.query_ids):
            raise ValueError("Unique row/site binding required")
        if not set(self.sites.tolist()).issubset({0,1}):raise ValueError("Two original sites required")
        self.lookup={int(q):i for i,q in enumerate(self.query_ids)}
        self.prior_identity=prior_identity
        self.storage_mode="uncompressed_npz_lossless_full1024"
        self.identity_hash=digest(dict(query_ids=self.query_ids.tolist(),site_ids=self.sites.tolist(),prior=prior_identity,storage_mode=self.storage_mode))
        contract=self.root/"observer_contract.json"
        with (self.root/".observer_contract.lock").open("a") as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            if contract.exists():
                if json.loads(contract.read_text()).get("observer_identity")!=self.identity_hash:
                    raise ValueError("Observer root belongs to a different prior/query contract")
            else:
                atomic_json(contract,dict(schema="v16-live-observer-contract-v1",observer_identity=self.identity_hash,
                    query_ids=self.query_ids.tolist(),site_ids=self.sites.tolist(),prior_identity=prior_identity,storage_mode=self.storage_mode))
        SOURCE_HASHES[PATHS.relative(Path(__file__))]=sha256(Path(__file__))
        self.extract=frozen_support()
        self.cfg=types.SimpleNamespace(components_per_point=128,base_reserve=48,
            base_components_per_seed=96,scatter_components_per_seed=64,scatter_pool=256,
            interaction_steps=16,subcarrier_spacing_hz=30000.,subcarriers=192)
        self.agg=frozen_aggregation()

    def has_batch(self,point_ids,configuration,seed,site):
        group=digest(np.asarray(point_ids,np.int64).tolist())[:16]
        p=self.root/"compact"/configuration/f"seed_{seed}"/f"site_{site}"/f"group_{group}.npz"
        m=p.with_suffix(".json")
        if not p.exists() or not m.exists():return False
        proof=json.loads(m.read_text())
        return proof.get("observer_identity")==self.identity_hash and proof.get("sha256")==sha256(p)

    def consume_live(self,components,point_ids: np.ndarray,configuration: str,seed: int,site: int,replace_uncommitted=False):
        """Call while full a[B,4,1,256,P] and interaction fields are still live."""
        loaded=self.extract.extract_component_support(components,np.asarray(point_ids,np.int64),int(seed))
        return self.consume_support(loaded,configuration,seed,site,replace_uncommitted)

    def consume_support(self,loaded: dict,configuration: str,seed: int,site: int,replace_uncommitted=False):
        """Also accepts the unmodified native extract_component_support dict."""
        if configuration not in {"base","scatter"} or int(seed) not in ORIGINAL_SEEDS:
            raise ValueError("Original Base/Scatter source and exact original seed required")
        ids=np.asarray(loaded["point_ids"],np.int64)
        if not len(ids) or len(np.unique(ids))!=len(ids):raise ValueError("Invalid batch IDs")
        for q in ids:
            if int(q) not in self.lookup or int(self.sites[self.lookup[int(q)]])!=int(site):
                raise ValueError("Live point/site mismatch")
        per_seed=96 if configuration=="base" else 64
        selected=[];offsets=[0]
        for i in range(len(ids)):
            lo=int(loaded["offsets"][i]);hi=int(loaded["offsets"][i+1])
            fraction=np.asarray(loaded["relative_energy"][lo:hi],np.float64)
            count=min(per_seed,len(fraction))
            chosen=np.sort(lo+np.argpartition(fraction,len(fraction)-count)[-count:]) if count else np.empty(0,np.int64)
            selected.extend(chosen.tolist());offsets.append(len(selected))
        selected=np.asarray(selected,np.int64)
        compact=dict(point_ids=ids,offsets=np.asarray(offsets,np.int64))
        for name,array in loaded.items():
            if name not in {"point_ids","offsets"}:compact[name]=np.asarray(array)[selected]
        coef=compact["full_complex_coefficient"]
        if coef.shape!=(len(selected),256,4) or not np.iscomplexobj(coef):
            raise ValueError("Full native 256x4 coefficients required; no repeated polarization/subset substitute")
        for key,a in compact.items():
            if a.dtype.kind in "fc" and not np.isfinite(a).all():raise ValueError(f"Nonfinite compact field {key}")
        group=digest(ids.tolist())[:16]
        out=self.root/"compact"/configuration/f"seed_{seed}"/f"site_{site}"/f"group_{group}.npz"
        out.parent.mkdir(parents=True,exist_ok=True)
        manifest=out.with_suffix(".json")
        if (self.root/"tokens"/"manifest.json").exists():
            raise RuntimeError("Finalized token inputs are immutable")
        if out.exists() or manifest.exists():
            if not replace_uncommitted:
                raise FileExistsError("Driver must skip committed done=True or explicitly replace_uncommitted=True")
            if manifest.exists() and json.loads(manifest.read_text()).get("observer_identity")!=self.identity_hash:
                raise ValueError("Cannot replace a different observer identity")
        partial=out.with_suffix(".partial.npz")
        with partial.open("wb" if replace_uncommitted else "xb") as f:np.savez(f,**compact)
        os.replace(partial,out)
        report=dict(schema="v16-live-compact-shard-v1",observer_identity=self.identity_hash,
            source=configuration,seed=int(seed),site=int(site),point_ids=ids.tolist(),
            count=len(selected),native_port_axes="BS[p,h,v],UE[p,h0,v]",
            source_identity=self.prior_identity,storage_mode=self.storage_mode,sha256=sha256(out),path=PATHS.relative(out),
            sources=dict(SOURCE_HASHES))
        if replace_uncommitted:
            temp=manifest.with_suffix(".partial.json")
            with temp.open("w") as f:json.dump(report,f,indent=2)
            os.replace(temp,manifest)
        else:atomic_json(manifest,report)
        return report

    def finalize(self):
        with (self.root/".finalize.lock").open("a") as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            return self._finalize_unlocked()

    def _finalize_unlocked(self):
        """Require every(query,source,seed) exactly once before final128 tokens."""
        complete=self.root/"tokens"/"manifest.json"
        if complete.exists():
            report=json.loads(complete.read_text())
            if report.get("observer_identity")!=self.identity_hash:raise ValueError("Final token identity differs")
            for item in report["arrays"].values():
                if sha256(PATHS.resolve(item["path"]))!=item["sha256"]:
                    raise ValueError("Final token bytes changed")
            return report
        n=len(self.query_ids);base=[{} for _ in range(n)];scatter=[[] for _ in range(n)]
        coverage=set();shards=[]
        # Preserve original source/site/seed processing order, independent of RT batch order.
        for source,configuration in enumerate(("base","scatter")):
            for site in (0,1):
                for seed in ORIGINAL_SEEDS:
                    root=self.root/"compact"/configuration/f"seed_{seed}"/f"site_{site}"
                    for p in sorted(root.glob("*.npz")):
                        proof=json.loads(p.with_suffix(".json").read_text())
                        if proof.get("observer_identity")!=self.identity_hash or proof.get("sha256")!=sha256(p):
                            raise ValueError("Compact shard identity changed")
                        shards.append(file_identity(p))
                        with np.load(p,allow_pickle=False) as z:loaded={k:np.asarray(z[k]) for k in z.files}
                        for local,q in enumerate(loaded["point_ids"]):
                            row=self.lookup[int(q)];key=(int(q),source,int(seed))
                            if key in coverage:raise ValueError("Duplicate point/source/seed")
                            coverage.add(key)
                            records=self.agg._records_for_point(loaded,local,96 if source==0 else 64,source)
                            if source==0:
                                bank=base[row]
                                for record in records:
                                    identity=self.agg._base_identity(record,site,self.cfg)
                                    previous=bank.get(identity)
                                    if previous is None:bank[identity]=record
                                    else:
                                        presence=previous.presence+1
                                        if record.energy_fraction>previous.energy_fraction:
                                            record.presence=presence;bank[identity]=record
                                        else:previous.presence=presence
                            else:
                                local_best={}
                                for record in records:
                                    identity=self.agg._scatter_identity(record,site,self.cfg)
                                    previous=local_best.get(identity)
                                    if previous is None or record.energy_fraction>previous.energy_fraction:local_best[identity]=record
                                scatter[row].extend(local_best.values())
                                if len(scatter[row])>256:
                                    scatter[row].sort(key=lambda x:x.energy_fraction,reverse=True)
                                    del scatter[row][256:]
        expected={(int(q),source,seed) for q in self.query_ids for source in (0,1) for seed in ORIGINAL_SEEDS}
        if coverage!=expected:raise ValueError(f"Missing {len(expected-coverage)} point/source/seed groups")
        arrays=self.agg._empty_arrays(n,self.cfg)
        for row in range(n):
            records=list(base[row].values())
            records.sort(key=lambda x:x.energy_fraction*(1+.05*(x.presence-1)),reverse=True)
            take=min(48,128,len(records));selected=records[:take]
            if len(selected)<128:
                selected.extend(sorted(scatter[row],key=lambda x:x.energy_fraction,reverse=True)[:128-len(selected)])
            if len(selected)<128:selected.extend(records[take:take+128-len(selected)])
            for slot,record in enumerate(selected[:128]):self.agg._write_record(arrays,row,slot,record)
        output=self.root/"tokens"
        output.mkdir(parents=True,exist_ok=True)
        items={}
        for name,a in arrays.items():
            p=output/f"{name}.npy"
            if p.exists():
                old=np.load(p,mmap_mode="r")
                if old.shape!=a.shape or old.dtype!=a.dtype or not np.array_equal(old,a):
                    raise ValueError("Incomplete final token differs; preserve for explicit repair")
                items[name]=file_identity(p)
                continue
            partial=p.with_suffix(".partial.npy")
            # A prior crash may leave this uncommitted partial. This observer
            # root is contract-bound; final files are never silently replaced.
            with partial.open("wb") as f:np.save(f,a,allow_pickle=False)
            os.replace(partial,p);items[name]=file_identity(p)
        report=dict(schema="v16-regenerated-tokens-v1",observer_identity=self.identity_hash,
            query_ids=self.query_ids.tolist(),site_ids=self.sites.tolist(),original_seeds=list(ORIGINAL_SEEDS),
            prior_identity=self.prior_identity,arrays=items,compact_shards=shards,
            aggregate_rule="frozen_V16_Base96_Scatter64_presence_base48_total128_pool256",storage_mode=self.storage_mode,
            sources=dict(SOURCE_HASHES),regenerated=True,exact_old_cache_equality_claimed=False,
            query_CSI_used=False,performance_evaluated=False)
        atomic_json(output/"manifest.json",report)
        return report


def create_observer(root: Path,source: str):
    """physical_statistics_v2 driver factory; one common identity for both jobs."""
    if source not in {"base","scatter"}:raise ValueError(source)
    from round2.paper_baseline_adapters_v1.contract import SimulationInputs
    inputs=SimulationInputs()
    generator=Path(__file__).parent/"physical_statistics_v2.py"
    prior=dict(schema="v16-query500-prior-v2-common-token-contract",
        input_identity=inputs.identity,geometry_fingerprint=inputs.geometry.geometry_fingerprint,
        source_configs=["primary_d16_base_v4lambda_1m","primary_d10_scatter015_v4lambda_1m"],
        seed_order=list(ORIGINAL_SEEDS),batching="canonical500 grouped by site,16 queries",
        trace_seed_rule="seed+100000*site+first_original_Train_ID_in_new_query_group",
        generator_sha256=sha256(generator),observer_sha256=sha256(Path(__file__)),
        coefficient_representation="lossless native complex64 full256x4",storage_mode="uncompressed_npz_lossless_full1024",
        regenerated=True,exact_old_cache_equality_claimed=False)
    return LiveTokenObserver(Path(root)/"live_tokens",inputs.query_ids,
        inputs.site_id[inputs.query_ids],prior)

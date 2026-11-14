"""FIT-only geometric ray providers; queries never open geometric truth caches.

Training rays use cached approximate RT at allowed FIT rows. Query rays use
specular image-source transforms estimated exclusively from those FIT paths.
Visibility at a new position is unknown. This is a declared geometry-assisted
NeWRF adaptation, not the author's unchanged learned virtual-source search.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from round2.paper_baseline_adapters_v1.contract import SimulationInputs,sha256
from .geometry import OUTPUT,FRAME

def _unit(x):
    norm=np.linalg.norm(x,axis=-1,keepdims=True)
    if np.any(norm<1e-8) or not np.isfinite(x).all():raise ValueError("Degenerate direction vector")
    return x/norm

def reflection_transform(vertices,interaction_type,bs,ue):
    """Return specular image-source affine transform, checked against FIT geometry.

    No assumption that a refraction/diffuse path has a specular image source.
    Returns None for unsupported path physics or a degenerate plane estimate.
    """
    types=np.asarray(interaction_type);v=np.asarray(vertices,np.float64)
    count=int(np.count_nonzero(types))
    if not np.array_equal(np.flatnonzero(types),np.arange(count)):raise ValueError("Interaction ordering changed")
    if count and not np.all(types[:count]==1):return None
    a=np.eye(3);b=np.zeros(3)
    for j in range(count):
        previous=bs if j==0 else v[j-1]
        following=ue if j==count-1 else v[j+1]
        incoming=_unit(v[j]-previous);outgoing=_unit(following-v[j])
        normal=incoming-outgoing
        if np.linalg.norm(normal)<1e-6:return None
        normal=_unit(normal)
        reflection=np.eye(3)-2*np.outer(normal,normal)
        shift=2*normal*np.dot(normal,v[j])
        a,b=reflection@a,reflection@b+shift
    image=a@bs+b
    last=bs if count==0 else v[count-1]
    cosine=float(np.dot(_unit(image-ue),_unit(last-ue)))
    if cosine<1-1e-5:return None
    return a,b,cosine

class FitGeometryBank:
    """Capability-limited pointwise cache. Disallowed IDs fail before file open."""
    def __init__(self,manifest,inputs:SimulationInputs,outer=0,role='screen_fit',source_fit_ids=None):
        if role not in ('screen_fit','refit'):raise ValueError("Only canonical FIT source roles allowed")
        self.inputs=inputs;self.outer=outer;self.role=role
        self.manifest_path=Path(manifest).resolve()
        if not self.manifest_path.is_relative_to(OUTPUT.resolve()):raise ValueError("Geometry bank must belong to the new recovery output")
        self.root=self.manifest_path.parent
        self.manifest=json.loads(self.manifest_path.read_text())
        contract=self.manifest['contract']
        expected=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
        if expected!=self.manifest['contract_hash']:raise ValueError("Geometry contract hash mismatch")
        if contract['input_identity']!=inputs.identity or contract['geometry_fingerprint']!=inputs.geometry.geometry_fingerprint:
            raise ValueError("Geometry cache does not match canonical coordinate/input identity")
        if contract['coordinate_frame']!=FRAME or contract['official_map_sha256']!=sha256(PATHS.round2/'Round2_Map.ply'):
            raise ValueError("Geometry frame/map identity changed")
        if not contract['no_CSI'] or not contract['no_official_Test']:raise ValueError("Unexpected trace information source")
        allowed=np.asarray(inputs.roles(outer)[role],np.int64)
        available=np.array(sorted(map(int,self.manifest['points'])),np.int64)
        selected=np.intersect1d(allowed,available) if source_fit_ids is None else np.asarray(source_fit_ids)
        if selected.ndim!=1 or selected.dtype.kind not in 'iu' or not len(selected) or len(np.unique(selected))!=len(selected):raise ValueError("Unique nonempty FIT IDs required")
        if not set(selected).issubset(set(allowed)) or not set(selected).issubset(set(available)):
            raise PermissionError("Direction sources cross canonical FIT or are unavailable")
        if np.intersect1d(selected,inputs.roles(outer)['outer']).size:raise PermissionError("Outer geometry cannot become a training direction source")
        if role=='screen_fit' and np.intersect1d(selected,inputs.roles(outer)['validation']).size:raise PermissionError("Validation geometry cannot become a screen source")
        self.source_fit_ids=np.sort(selected.astype(np.int64));self.allowed=set(map(int,self.source_fit_ids))
        self.opened_ids=[]
        self.source_fingerprint=hashlib.sha256(json.dumps(dict(manifest=sha256(self.manifest_path),outer=outer,role=role,
          source_fit_ids=self.source_fit_ids.tolist(),input_identity=inputs.identity),sort_keys=True).encode()).hexdigest()
    def get(self,row):
        row=int(row)
        if row not in self.allowed:raise PermissionError("Geometry payload access outside this provider's FIT")
        entry=self.manifest['points'][str(row)];path=(self.root/entry['file']).resolve()
        if not path.is_relative_to(self.root) or path.name!=f'point_{row:05d}.npz':raise ValueError("Pointwise cache path mismatch")
        if sha256(path)!=entry['sha256']:raise ValueError("Geometry payload hash mismatch")
        with np.load(path,allow_pickle=False) as z:point={key:np.array(z[key],copy=True) for key in z.files}
        self.opened_ids.append(row)
        if int(point['point_id'])!=row or int(point['site_id'])!=int(self.inputs.site_id[row]):raise ValueError("Point/site identity mismatch")
        if not np.array_equal(point['ue_position_m'],self.inputs.positions[row]):raise ValueError("FIT position mismatch")
        expected_ports=self.inputs.geometry.ue_world_positions(torch.tensor(self.inputs.positions[row:row+1].copy(),dtype=torch.float64),
          torch.tensor(self.inputs.site_id[row:row+1].copy(),dtype=torch.int64)).numpy()[0]
        if not np.array_equal(expected_ports,point['ue_port_positions_m']):raise ValueError("Dynamic UE port geometry mismatch")
        if not np.array_equal(point['frequencies_hz'],self.inputs.geometry.frequencies_hz):raise ValueError("Physical frequency mismatch")
        return point
    def metadata(self):
        return dict(source_fit_ids=self.source_fit_ids.tolist(),source_fingerprint=self.source_fingerprint,
          canonical_role=self.role,outer=self.outer,canonical_FIT_count=len(self.inputs.roles(self.outer)[self.role]),
          cached_FIT_count=len(self.source_fit_ids),all_FIT_geometry_available=len(self.source_fit_ids)==len(self.inputs.roles(self.outer)[self.role]),
          manifest=PATHS.relative(self.manifest_path),manifest_sha256=sha256(self.manifest_path),
          coordinate_frame=FRAME,geometry_fingerprint=self.inputs.geometry.geometry_fingerprint,
          trace_config=self.manifest['contract']['trace_config'],generated_geometry_count=len(self.manifest['points']),
          generated_role=self.manifest['contract']['generated_role'],geometry_publisher_certified=False,
          approximate=True,read_CSI=False,query_CSI=False,query_path_labels=False)

class TrainingGeometryRayProvider:
    """Ragged exact cache-source directions for explicitly allowed FIT row IDs."""
    def __init__(self,bank):self.bank=bank
    def for_rows(self,row_ids,bs_tx_positions=None):
        result=[]
        for row in row_ids:
            point=self.bank.get(int(row));last=point['last_interaction_m']
            origins=point['ue_port_positions_m'];los=point['depth']==0
            if bs_tx_positions is None:
                vectors=last[:,None]-origins[None]
            else:
                tx=np.asarray(bs_tx_positions[int(point['site_id'])],np.float64).reshape(256,3)
                vectors=np.broadcast_to(last[:,None,None]-origins[None,None],(len(last),256,4,3)).copy()
                vectors[los]=tx[None,:,None]-origins[None,None]
            result.append(dict(point_id=int(row),site_id=int(point['site_id']),directions=_unit(vectors),
              last_interaction_m=last,theta_r=point['theta_r'],phi_r=point['phi_r'],depth=point['depth'],
              array_approximation='non-LoS last interaction fixed from centre synthetic-array trace; LoS uses requested actual TX positions'))
        return result
    def metadata(self):return dict(kind='canonical_FIT_cached_approximate_geometry_training_rays',**self.bank.metadata())

class FitVirtualSourceRayProvider:
    """No tracing, geometry-cache reads or channel inputs at query time.

    Image transforms use only the bank's FIT vertices. Candidate visibility is
    not asserted. Non-specular train paths are retained in the bank/training
    provider, but cannot be converted to specular virtual sources here.
    """
    def __init__(self,bank,max_rays=64,minimum_support=1):
        if max_rays<1 or minimum_support<1:raise ValueError("Positive prior budgets required")
        self.bank=bank;self.max_rays=int(max_rays);self.minimum_support=int(minimum_support)
        groups={};accepted=unsupported=0;errors=[]
        for row in bank.source_fit_ids:
            point=bank.get(row);site=int(point['site_id']);bs=point['bs_position_m'];ue=point['ue_position_m']
            for vertices,types in zip(point['vertices_m'],point['interaction_type']):
                value=reflection_transform(vertices,types,bs,ue)
                if value is None:unsupported+=1;continue
                a,b,cosine=value;accepted+=1;errors.append(np.arccos(np.clip(cosine,-1,1)))
                key=(site,tuple(np.rint(a/1e-4).astype(np.int64).ravel()),tuple(np.rint(b/.05).astype(np.int64)))
                if key not in groups:groups[key]=dict(site=site,a=a,b=b,rows=set(),depth=int(np.count_nonzero(types)))
                groups[key]['rows'].add(int(row))
        self.transforms={};before_cap={}
        for site in (0,1):
            values=[v for v in groups.values() if v['site']==site and len(v['rows'])>=minimum_support]
            values.sort(key=lambda x:(-len(x['rows']),x['depth'],tuple(np.round(x['b'],5))))
            # Known BS location is a legal direct-ray candidate regardless of whether
            # LoS is observed at a training point. It is not a visibility label.
            direct=dict(site=site,a=np.eye(3),b=np.zeros(3),rows=set(),depth=0)
            values=[direct]+[v for v in values if not (np.allclose(v['a'],np.eye(3)) and np.allclose(v['b'],0))]
            before_cap[str(site)]=len(values)
            self.transforms[site]=values[:max_rays]
        self.diagnostic=dict(accepted_fit_specular_paths=accepted,unsupported_or_inconsistent_fit_paths=unsupported,
          maximum_fit_image_direction_error_rad=float(max(errors)) if errors else None,
          source_counts={str(k):len(v) for k,v in self.transforms.items()},quantization_translation_m=.05,
          candidate_visibility_known=False,maximum_rays=max_rays,image_candidates_before_cap=before_cap,
          fallback_rays_per_site={str(k):max_rays-len(v) for k,v in self.transforms.items()},
          discarded_image_candidates={str(k):max(0,before_cap[str(k)]-max_rays) for k in self.transforms})
        self.opened_ids_after_build=list(bank.opened_ids)
    def directions(self,ue_origins,site_ids,bs_tx_positions):
        if ue_origins.shape!=(len(site_ids),4,3) or bs_tx_positions.shape!=(len(site_ids),2,128,3):raise ValueError("Query geometric shapes")
        if not torch.isfinite(ue_origins).all() or not torch.isfinite(bs_tx_positions).all():raise ValueError("Nonfinite query geometry")
        opened_before=list(self.bank.opened_ids)
        device=ue_origins.device;dtype=torch.float64
        k=torch.arange(self.max_rays,dtype=dtype,device=device);z=1-2*(k+.5)/self.max_rays
        az=k*(torch.pi*(3-np.sqrt(5)));rr=torch.sqrt(1-z*z)
        fallback=torch.stack((rr*az.cos(),rr*az.sin(),z),-1)
        out=fallback[None,None,None,None].expand(len(site_ids),2,128,4,-1,-1).clone()
        for i,site_tensor in enumerate(site_ids):
            site=int(site_tensor)
            if site not in self.transforms:raise ValueError("Unknown query site")
            ts=self.transforms[site];a=torch.tensor(np.stack([v['a'] for v in ts]),device=device,dtype=dtype)
            b=torch.tensor(np.stack([v['b'] for v in ts]),device=device,dtype=dtype)
            images=torch.einsum('rij,pmj->rpmi',a,bs_tx_positions[i].to(dtype))+b[:,None,None,:]
            vec=images[:,:,:,None,:]-ue_origins[i,None,None,None].to(dtype)
            norm=torch.linalg.vector_norm(vec,dim=-1,keepdim=True)
            if bool((norm<1e-8).any()):raise ValueError("Query lies on a virtual source")
            out[i,:,:,:,:len(ts),:]=(vec/norm).permute(1,2,3,0,4)
        if self.bank.opened_ids!=opened_before:raise RuntimeError("Unexpected geometry reads during query inference")
        return out
    def metadata(self):
        return dict(kind='FIT_geometry_specular_image_source_prior_plus_fixed_sphere',
          **self.bank.metadata(),**self.diagnostic,query_geometry_cache_reads=False,
          author_ray_search_unchanged=False,trained_count_predictor=False,
          original_paper_reproduction_qualified=False,
          limitations=['one-seed V16 geometry is approximate','new-position visibility and path birth/death are not inferred',
            'refraction/diffuse paths are not specular image sources','not a recovered true multipath DoA set'])

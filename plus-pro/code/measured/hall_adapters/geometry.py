"""Matched south model, fixed RF assumptions and native physical runtime."""
from __future__ import annotations
from pathlib import Path
import json
import os
import numpy as np
from .common import SOUTH, ARTIFACTS, C0, save_json, sha256, check_prediction

MATERIALS = {"metal-reflector":"metal","wood":"wood","paperwall-metal":"metal",
             "ceiling":"ceiling_board","concrete-wall":"concrete","floor":"concrete",
             "mirror":"metal","glass-wall":"glass","plasterboard":"plasterboard"}

def export_scene(work=ARTIFACTS/"scene"):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    objects=json.loads((SOUTH/"model_objects.json").read_text())["objects"]
    lines=['<scene version="2.1.0">'];mapping=[]
    with np.load(SOUTH/"model_measurement_frame.npz",allow_pickle=False) as z:
        for i,obj in enumerate(objects):
            if not obj["include_in_future_rf_scene"]:continue
            material=MATERIALS[obj["material_name"]];name="mesh"+str(i)
            mesh=work/(name+".ply");vertices=z[obj["vertices_key"]];faces=z[obj["faces_key"]]
            header="ply\nformat ascii 1.0\nelement vertex "+str(len(vertices))+"\nproperty float x\nproperty float y\nproperty float z\nelement face "+str(len(faces))+"\nproperty list uchar int vertex_indices\nend_header\n"
            content=header+"".join("%.17g %.17g %.17g\n"%tuple(v) for v in vertices)+"".join("3 %d %d %d\n"%tuple(f) for f in faces)
            if mesh.exists() and mesh.read_text()!=content:raise ValueError("Existing mesh identity changed")
            if not mesh.exists():mesh.write_text(content)
            lines += ['<bsdf type="itu-radio-material" id="radio-'+name+'">',
                      '<string name="type" value="'+material+'"/>',
                      '<float name="thickness" value="0.1"/></bsdf>',
                      '<shape type="ply" id="'+name+'"><string name="filename" value="'+mesh.name+'"/>',
                      '<boolean name="face_normals" value="true"/><ref id="radio-'+name+'" name="bsdf"/></shape>']
            mapping.append(dict(object=obj["name"],visual_material=obj["material_name"],rf_material=material,
                                thickness_m=.1,source="fixed modelling assumption, not measured material calibration",
                                mesh_sha256=sha256(mesh)))
    lines+=["</scene>"];xml=work/"scene.xml";content="\n".join(lines)+"\n"
    if xml.exists() and xml.read_text()!=content:raise ValueError("Scene identity changed")
    if not xml.exists():xml.write_text(content)
    save_json(work/"material_assumptions.json",dict(objects=mapping,rf_parameters_fitted=False,
             scene_source_sha256=sha256(SOUTH/"model_measurement_frame.npz"),
             visual_antennas_excluded=True,previous_scene_or_coordinate_frame_reused=False))
    return xml

def wideband_from_arrays(a,tau,valid,frequency_offsets):
    a=np.asarray(a,np.complex128);tau=np.asarray(tau,np.float64)
    if a.ndim!=3 or tau.shape!=(a.shape[0],a.shape[-1]) or valid.shape!=tau.shape:raise ValueError("Path layout")
    a=np.where(valid[:,None,:],a,0.)
    delay=np.where(valid,tau,0.)
    phase=np.exp(-2j*np.pi*delay[...,None]*np.asarray(frequency_offsets)[None,None,:])
    return np.einsum("nap,npf->naf",a,phase).astype(np.complex64)

class NativeRT:
    def __init__(self,spec,work=ARTIFACTS/"scene"):
        import mitsuba as mi
        if mi.variant() is None:mi.set_variant("llvm_ad_mono_polarized")
        if mi.variant()!="llvm_ad_mono_polarized":raise ValueError("Correctness runtime must remain CPU LLVM")
        from round2.physical_prior.physical_rt import PlanarArray,PathSolver,load_scene
        from measured.dc01_geometry import mesh_normals
        self.mi=mi;self.spec=spec;self.scene=load_scene(str(export_scene(work)),merge_shapes=False)
        self.scene.frequency=spec.carrier_hz
        spacing=spec.spacing_m*spec.carrier_hz/C0
        self.scene.tx_array=PlanarArray(num_rows=4,num_cols=8,vertical_spacing=spacing,
                           horizontal_spacing=spacing,pattern="tr38901",polarization="V")
        self.scene.rx_array=PlanarArray(num_rows=1,num_cols=1,pattern="dipole",polarization="V")
        self.normals=mesh_normals(self.scene);self.solver=PathSolver()
        self.array_check=[]

    def trace(self,query,samples=4096,max_depth=2,seed=20261024,scattering=False):
        from round2.physical_prior.physical_rt import Transmitter,Receiver
        from round2.physical_prior.physical_rt.utils import rotation_matrix
        from measured.dc01_geometry import convert_batch
        import drjit as dr
        h=[];parts=[];diagnostics=[]
        mi=self.mi
        for panel in range(2):
            center=self.spec.center[panel];direction=self.spec.direction[panel]
            tx=Transmitter(name="array",position=mi.Point3f(*center.tolist()),look_at=mi.Point3f(*(center+direction).tolist()))
            self.scene.add(tx)
            local=self.scene.tx_array.positions(C0/self.spec.carrier_hz).numpy().T
            rotation=rotation_matrix(tx.orientation).numpy()
            actual=(local@rotation.T).reshape(8,4,3)
            error=float(np.max(np.abs(actual-self.spec.element_offsets[panel])))
            if error>2e-7:raise ValueError("Native array axes/order differ from measured contract")
            self.array_check.append(dict(panel=panel,maximum_element_position_difference_m=error))
            for i,pos in enumerate(query.position):self.scene.add(Receiver(name="query"+str(i),position=mi.Point3f(*pos.tolist())))
            for material in self.scene.radio_materials.values():material.scattering_coefficient=.15 if scattering else 0.
            paths=self.solver(self.scene,max_depth=max_depth,samples_per_src=samples,
                max_num_paths_per_src=32768,synthetic_array=True,los=True,specular_reflection=True,
                diffuse_reflection=scattering,refraction=True,diffraction=False,edge_diffraction=False,
                seed=seed+panel)
            ar,ai=paths.a
            a=np.asarray(ar.numpy())+1j*np.asarray(ai.numpy())
            valid=np.asarray(paths.valid.numpy(),bool)[:,0]
            tau=np.asarray(paths.tau.numpy(),np.float64)[:,0]
            channel=wideband_from_arrays(a[:,0,0],tau,valid,self.spec.frequencies-self.spec.carrier_hz)
            h.append(channel.reshape(len(query.row_ids),8,4,self.spec.frequency_count))
            support=convert_batch(paths,query.position,center,self.normals,"scatter" if scattering else "base")
            # Old converter's physical path geometry is reused; overwrite all old256-bin features in the new batch builder.
            parts.append(support)
            diagnostics.append(dict(panel=panel,raw_valid_paths=int(valid.sum()),retained_paths=int(support["mask"].sum())))
            for i in range(len(query.row_ids)):self.scene.remove("query"+str(i))
            self.scene.remove("array");del paths;dr.sync_thread()
        support={k:np.stack([x[k] for x in parts],axis=1) for k in parts[0]}
        result=check_prediction(dict(method="fixed_material_RT",channel=np.stack(h,axis=1),
              diagnostics=dict(samples_per_source=samples,max_depth=max_depth,scattering=scattering,
                               source_CSI_read=False,paths=diagnostics)),query)
        return result,support

"""Generate dc01 geometry support using the final algorithm's physical runtime.

The registered NVlabs scene is in the transformed coordinate system used by
dc01_data.py. Raw measurements and V16 frozen assets are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from project_paths import PATHS


def atomic_npz(path,**values):
    tmp=path.with_suffix('.partial.npz')
    np.savez_compressed(tmp,**values)
    os.replace(tmp,path)


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    os.replace(tmp,path)


def scene_xml(data,work):
    work.mkdir(parents=True,exist_ok=True)
    source=data/'context/scenes/inue_simple/meshes'
    lines=['<scene version="2.1.0">']
    for name,material in [('floor','concrete'),('ceiling','ceiling_board'),('walls','plasterboard')]:
        mesh=os.path.relpath(source/f'{name}.ply',work).replace(os.sep,'/')
        lines.extend([f'<bsdf type="itu-radio-material" id="radio-{name}">',
                      f'<string name="type" value="{material}"/>',
                      '<float name="thickness" value="0.1"/>','</bsdf>',
                      f'<shape type="ply" id="{name}">',
                      f'<string name="filename" value="{mesh}"/>',
                      '<boolean name="face_normals" value="true"/>',
                      f'<ref id="radio-{name}" name="bsdf"/>','</shape>'])
    lines.append('</scene>')
    target=work/'measured_scene.xml'
    target.write_text('\n'.join(lines),encoding='utf-8')
    return target


def mesh_normals(scene):
    import drjit as dr
    import mitsuba as mi
    result={}
    for mesh in scene.mi_scene.shapes():
        vertices=dr.unravel(mi.Point3f,mesh.vertex_positions_buffer()).numpy().T
        faces=dr.unravel(mi.Point3u,mesh.faces_buffer()).numpy().T.astype(np.int64)
        normal=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
        normal=normal/np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-30)
        key=int(dr.reinterpret_array(mi.UInt,mi.ShapePtr(mesh)).numpy()[0])
        result[key]=normal.astype(np.float32)
    return result


def convert_batch(paths,positions,source,normals,configuration):
    valid=np.asarray(paths.valid.numpy(),bool)
    tau=np.asarray(paths.tau.numpy(),np.float32)
    interactions=np.asarray(paths.interactions.numpy(),np.uint32)
    vertices=np.asarray(paths.vertices.numpy(),np.float32)
    primitive=np.asarray(paths.primitives.numpy(),np.uint32)
    objects=np.asarray(paths.objects.numpy(),np.uint32)
    ar,ai=paths.a
    coefficients=np.asarray(ar.numpy(),np.float32)+1j*np.asarray(ai.numpy(),np.float32)
    count=len(positions);cap=128;steps=16
    coordinate=np.zeros((count,cap,3),np.float32)
    mass=np.zeros((count,cap),np.float32)
    mask=np.zeros((count,cap),bool)
    features=np.zeros((count,cap,24),np.float32)
    sequence=np.zeros((count,cap,steps,21),np.float32)
    interaction=np.zeros((count,cap,steps),np.int64)
    depths=np.zeros((count,cap),np.int64)
    identity=np.zeros((count,cap),np.uint64)
    for i,receiver in enumerate(positions):
        available=np.flatnonzero(valid[i,0])
        if not len(available):continue
        a=coefficients[i,0,0,:,available]
        if a.shape[0]!=len(available):a=a.T
        a=a.reshape(len(available),8,4)
        energy=np.abs(a).astype(np.float64)**2
        energy=energy.sum((1,2))
        order=np.argsort(-energy,kind='stable')[:cap]
        available=available[order];a=a[order];energy=energy[order]
        delay=tau[i,0,available]
        origin=float(np.min(tau[i,0,np.flatnonzero(valid[i,0])]))
        hd=np.angle(np.sum(a[:,1:,:]*a[:,:-1,:].conj(),axis=(1,2)))*8/(2*np.pi)
        vd=np.angle(np.sum(a[:,:,1:]*a[:,:,:-1].conj(),axis=(1,2)))*4/(2*np.pi)
        coord=np.stack((hd,vd,(delay-origin)*50e6),axis=-1)
        n=len(available)
        coordinate[i,:n]=coord
        mass[i,:n]=energy
        mask[i,:n]=energy>1e-30
        local_kinds=np.zeros((n,steps),np.int64)
        count_steps=interactions.shape[0]
        local_kinds[:,:count_steps]=np.take(interactions[:,i,0],available,axis=1).T
        depth=np.minimum(np.count_nonzero(local_kinds,axis=1),steps)
        route=np.zeros((n,steps,3),np.float32)
        route[:,:count_steps]=np.take(vertices[:,i,0],available,axis=1).transpose(1,0,2)
        active=np.arange(steps)[None,:]<depth[:,None]
        route=np.where(active[...,None],route,0)
        local_objects=np.zeros((n,steps),np.uint32)
        local_faces=np.zeros_like(local_objects)
        local_objects[:,:count_steps]=np.take(objects[:,i,0],available,axis=1).T
        local_faces[:,:count_steps]=np.take(primitive[:,i,0],available,axis=1).T
        normal=np.zeros_like(route)
        found=np.zeros((n,steps),bool)
        for key,bank in normals.items():
            take=active&(local_objects==key)
            if np.any(local_faces[take]>=len(bank)):raise ValueError('Path face index exceeds its true mesh')
            normal[take]=bank[local_faces[take]]
            found|=take
        if np.any(active&~found):raise ValueError('Path object pointer does not match scene meshes')
        previous=np.concatenate((np.broadcast_to(source,(n,1,3)),route[:,:-1]),axis=1)
        following=np.concatenate((route[:,1:],np.broadcast_to(receiver,(n,1,3))),axis=1)
        last=np.arange(steps)[None,:]==depth[:,None]-1
        following=np.where(last[...,None],receiver,following)
        incoming=route-previous;outgoing=following-route
        incoming_length=np.linalg.norm(incoming,axis=-1)
        outgoing_length=np.linalg.norm(outgoing,axis=-1)
        incoming/=np.maximum(incoming_length[...,None],1e-8)
        outgoing/=np.maximum(outgoing_length[...,None],1e-8)
        endpoint=np.where((depth>0)[:,None],route[np.arange(n),np.maximum(depth-1,0)],source)
        length=(incoming_length*active).sum(-1)+np.linalg.norm(receiver-endpoint,axis=-1)
        cumulative=np.cumsum(incoming_length*active,axis=-1)
        repeat=lambda value:np.broadcast_to(value[:,None,None],(n,steps,1))
        sequence[i,:n]=np.concatenate((route/30,normal,incoming,outgoing,
            incoming_length[...,None]/50,outgoing_length[...,None]/50,
            np.abs((normal*incoming).sum(-1))[...,None],
            np.broadcast_to(np.arange(steps)[None,:,None]/16,(n,steps,1)),
            repeat(length/100),cumulative[...,None]/100,(receiver-route)/30),axis=-1)*active[...,None]
        interaction[i,:n]=local_kinds;depths[i,:n]=depth
        period=np.array([8,4,256]);phase=2*np.pi*coord/period
        features[i,:n]=np.column_stack((np.sin(phase),np.cos(phase),coord/period,
            np.log(np.maximum(energy,1e-30))/30,depth/16,
            np.full(n,float(configuration=='base')),np.full(n,float(configuration=='scatter')),
            np.ones(n),length/100,np.full(n,np.linalg.norm(receiver-source)/100),
            np.log(np.maximum(np.abs(a[:,0,0]),1e-30))/20,
            np.cos(np.angle(a[:,0,0])),np.sin(np.angle(a[:,0,0])),
            np.broadcast_to(receiver/30,(n,3)),(depth==0).astype(float),np.ones(n)))
        keys=np.concatenate((np.rint(coord/0.25).astype(np.int64),
            np.rint(route.reshape(n,-1)*100).astype(np.int64),local_kinds),axis=1)
        for j,key in enumerate(keys):identity[i,j]=int.from_bytes(hashlib.sha256(key.tobytes()+configuration.encode()).digest()[:8],'little')
    return dict(coordinate=coordinate,mass=mass,mask=mask,features=features,sequence=sequence,
                interaction=interaction,depth=depths,identity=identity)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=PATHS.datasets/'dichasus/dc01')
    parser.add_argument('--prepared-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,101,211])
    parser.add_argument('--samples-per-source',type=int,default=1000000)
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--probe-points',type=int,default=4)
    args=parser.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    optix=PATHS.root/'.runtime/nvidia-550.135/extracted/libnvoptix.so.550.135'
    if 'DRJIT_LIBOPTIX_PATH' not in os.environ and optix.exists():os.environ['DRJIT_LIBOPTIX_PATH']=str(optix)
    # RTX also loads the matching RTCore library. The ELF loader reads this at
    # process start, so changing os.environ alone after importing CUDA is too late.
    if optix.exists() and str(optix.parent) not in os.environ.get('LD_LIBRARY_PATH','').split(':'):
        os.environ['LD_LIBRARY_PATH']=str(optix.parent)+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
        os.execv(sys.executable,[sys.executable,*sys.argv])
    from round2.physical_prior.physical_rt import PathSolver,PlanarArray,Receiver,Transmitter,load_scene
    from round2.physical_prior.source_port_common import select_drjit_cuda_device
    import mitsuba as mi
    import drjit as dr
    select_drjit_cuda_device(args.gpu)
    meta=np.load(args.prepared_dir/'selection.npz')
    position=meta['position'].astype(np.float32)
    if args.probe:position=position[:min(args.probe_points,len(position))]
    scene=load_scene(str(scene_xml(args.data_dir,out/'scene')),merge_shapes=True,remove_duplicate_vertices=True)
    scene.frequency=3.438e9
    scene.tx_array=PlanarArray(num_rows=4,num_cols=8,vertical_spacing=0.5,horizontal_spacing=0.5,pattern='tr38901',polarization='V')
    scene.rx_array=PlanarArray(num_rows=1,num_cols=1,vertical_spacing=0.5,horizontal_spacing=0.5,pattern='dipole',polarization='V')
    normals=mesh_normals(scene)
    solver=PathSolver()
    identity=dict(source=hashlib.sha256((args.prepared_dir/'selection.npz').read_bytes()).hexdigest(),
                  code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scene_inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((args.data_dir/'context/scenes/inue_simple/meshes').glob('*.ply'))},
                  samples=args.samples_per_source,seeds=args.seeds,batch=args.batch_size)
    cache_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    completed=[]
    for panel in range(2):
        source=meta['panel_position'][panel].astype(np.float32)
        direction=meta['panel_direction'][panel]
        transmitter=Transmitter(name='array',position=mi.Point3f(*source.tolist()),look_at=mi.Point3f(*(source+direction).tolist()))
        scene.add(transmitter)
        for configuration in ['base','scatter']:
            for material in scene.radio_materials.values():material.scattering_coefficient=0.0 if configuration=='base' else 0.15
            for seed in args.seeds:
                folder=out/f'panel{panel}'/configuration/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
                for first in range(0,len(position),args.batch_size):
                    rows=np.arange(first,min(first+args.batch_size,len(position)))
                    path=folder/f'batch{first//args.batch_size:04d}.npz'
                    if path.exists():
                        with np.load(path,allow_pickle=False) as data:
                            if str(data['cache_hash'])==cache_hash and np.array_equal(data['point_id'],rows):continue
                    receiver_names=[]
                    for index in rows:
                        name=f'point{index}'
                        scene.add(Receiver(name=name,position=mi.Point3f(*position[index].tolist()),look_at=transmitter))
                        receiver_names.append(name)
                    t=time.time()
                    paths=solver(scene,max_depth=16 if configuration=='base' else 10,
                        max_num_paths_per_src=1000000,samples_per_src=args.samples_per_source,
                        synthetic_array=True,los=True,specular_reflection=True,
                        diffuse_reflection=configuration=='scatter',refraction=True,
                        diffraction=False,edge_diffraction=False,seed=seed+first+panel*100000)
                    support=convert_batch(paths,position[rows],source,normals,configuration)
                    atomic_npz(path,point_id=rows,cache_hash=np.array(cache_hash),**support)
                    del paths
                    for name in receiver_names:scene.remove(name)
                    dr.sync_thread()
                    status=dict(panel=panel,configuration=configuration,seed=seed,batch=int(first//args.batch_size),
                                points=len(rows),valid=int(support['mask'].sum()),seconds=time.time()-t)
                    completed.append(status)
                    atomic_json(out/'generation_progress.json',dict(identity=identity,elapsed_s=time.time()-started,last=status))
                    print(json.dumps(status),flush=True)
                if args.probe:break
            if args.probe:break
        scene.remove('array')
        if args.probe:break
    atomic_json(out/('probe_report.json' if args.probe else 'generation_report.json'),dict(
        status='complete',identity=identity,elapsed_s=time.time()-started,batches=completed,
        note='Three-seed physical-budget adaptation by default; not the original 24-seed competition asset.'))


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc()
        raise

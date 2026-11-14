"""CPU-only preparation of the two registered hallway blocker recordings.

No neural model, ray tracer, prediction, performance metric, or test alignment
is imported or executed. Raw sources and all previous prepared views are read-only.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import shutil
import struct
import sys
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
from project_paths import PATHS
from dc01_data import example, masked_crc

AUDIT = PATHS.analysis / "audits/hall_blocker_20261024"
REPORT = PATHS.analysis / "results/hall_blocker_preparation_v1"
VIEWS = {"dc10": "hall_sw_publisher_v2", "dc20": "hall_nw_publisher_v2"}
RECORD_BYTES = 525022
CONFIG = dict(voxel_m=.20, block_m=4., guard_m=.30, frequency_bins=1024,
              frequency_stride=1, maximum_points=None,
              coordinate_frame="publisher measurement XYZ, metres",
              gltf_to_measurement="(x,-z,y)",
              pairwise_guard=True, minimum_partition_points=16,
              split_assignment="whole4m_blocks_metadata_count_balancing_v1")
ROTATION = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x
def atomic_json(p, data):
    tmp=p.with_name(p.name+".partial")
    tmp.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    os.replace(tmp,p)
def atomic_npz(p, **data):
    tmp=p.with_name(p.name+".partial")
    with tmp.open("wb") as f: np.savez(f, **data)
    os.replace(tmp,p)
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()
def describe(p):
    return dict(path=PATHS.relative(p), size=p.stat().st_size, sha256=sha256(p))
def emit(**data): print(json.dumps(clean(data)), flush=True)
def quantile(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return None if not len(x) else dict(zip(["min","p01","p50","p95","p99","max"],np.quantile(x,[0,.01,.5,.95,.99,1]).tolist()))
def groups_info(hashes):
    values, inverse, count=np.unique(hashes,return_inverse=True,return_counts=True)
    duplicate=np.flatnonzero(count>1)
    groups=[np.flatnonzero(inverse==g).tolist() for g in duplicate]
    return dict(unique=len(values), duplicate_groups=len(groups),
                duplicate_rows=sum(len(x) for x in groups),
                excess_rows=sum(len(x)-1 for x in groups), groups=groups)
def nearest(a,b):
    if not len(a) or not len(b): return np.full(len(a),np.inf)
    return cKDTree(np.asarray(b)[:,:2]).query(np.asarray(a)[:,:2],k=1)[0]
def select_positions(position, valid, voxel=.20):
    ids=np.flatnonzero(valid)
    cells=np.floor(position[ids,:2]/voxel).astype(np.int64)
    _,groups=np.unique(cells,axis=0,return_inverse=True)
    centers=(cells+.5)*voxel
    distance=((position[ids,:2]-centers)**2).sum(1)
    order=np.lexsort((ids,distance,groups))
    selected=ids[order[np.r_[True,np.diff(groups[order])!=0]]]
    return np.sort(selected)
def block_assignment(position):
    tiles=np.floor(position[:,:2]/CONFIG["block_m"]).astype(np.int64)
    blocks,inverse,count=np.unique(tiles,axis=0,return_inverse=True,return_counts=True)
    if len(blocks)<5:raise ValueError("Fewer than five occupied spatial blocks")
    order=sorted(range(len(blocks)),key=lambda i:(-int(count[i]),int(blocks[i,0]),int(blocks[i,1])))
    totals=np.zeros(5,np.int64);assigned=np.empty(len(blocks),np.int64)
    for i in order:
        outer=int(np.argmin(totals));assigned[i]=outer;totals[outer]+=count[i]
    return assigned[inverse],dict(block_xy=blocks.tolist(),representative_count=count.tolist(),
        block_fold=assigned.tolist(),fold_counts=totals.tolist(),
        rule="Sort whole blocks by descending representative count, then lexicographic XY; assign to smallest current fold count, tie lower fold.")
def split_positions(position):
    color,assignment=block_assignment(position)
    splits=[]
    for outer in range(5):
        test=np.flatnonzero(color==outer)
        candidate_val=np.flatnonzero(color==(outer+1)%5)
        val=candidate_val[nearest(position[candidate_val],position[test])>=CONFIG["guard_m"]]
        candidate_fit=np.flatnonzero((color!=outer)&(color!=(outer+1)%5))
        held=np.r_[test,candidate_val]
        fit=candidate_fit[nearest(position[candidate_fit],position[held])>=CONFIG["guard_m"]]
        if min(len(fit),len(val),len(test))<CONFIG["minimum_partition_points"]:
            raise ValueError("Insufficient metadata support in a spatial partition")
        distances={}
        for a,b in [("fit","validation"),("fit","test"),("validation","test")]:
            aa={"fit":fit,"validation":val,"test":test}[a];bb={"fit":fit,"validation":val,"test":test}[b]
            values=nearest(position[aa],position[bb])
            if values.min()<CONFIG["guard_m"]:raise ValueError("Pairwise spatial guard violated")
            distances[a+"__"+b]=quantile(values)
        splits.append(dict(outer=outer, fit=fit.tolist(), validation=val.tolist(), test=test.tolist(),
            validation_candidates=len(candidate_val), validation_removed_near_test=len(candidate_val)-len(val),
            fit_candidates=len(candidate_fit),fit_removed_near_held=len(candidate_fit)-len(fit),
            pairwise_distance_m=distances))
    return dict(schema="hall-blocker-pairwise-spatial-v2",config=CONFIG,fold=color.tolist(),blocks=assignment,splits=splits,
                test_once=True,sha256=hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest())
def correction(offsets):
    phase=np.asarray(offsets["cpo"],np.float64)[:,None]+2*np.pi*np.asarray(offsets["sto"],np.float64)[:,None]*np.arange(1024)/1024
    return np.exp(1j*phase).astype(np.complex64)
def calibrate(csi, offsets, assignments):
    raw=(csi[...,0]+1j*csi[...,1]).astype(np.complex64)
    shifted=np.fft.fftshift(raw,axes=-1)
    h=shifted*correction(offsets)
    return h[assignments].transpose(0,2,1,3),shifted
def inputs(dataset):
    d=json.loads((AUDIT/"data_calibration_checks.json").read_text())["records"]["dichasus-"+dataset]
    paths={k:PATHS.root/v for k,v in dict(raw=d["raw_identity"]["path"],
        offsets=d["calibration"]["identity"]["path"],model=d["model"],
        spec="datasets/dichasus/dichasus-dcxx/spec.json").items()}
    for k,p in paths.items():
        if not p.is_file():raise FileNotFoundError(k)
    if sha256(paths["offsets"])!=d["calibration"]["identity"]["sha256"]:raise ValueError("Calibration identity changed")
    model_identity=json.loads((AUDIT/"model_checks.json").read_text())["models"][paths["model"].name]
    if sha256(paths["model"])!=model_identity["sha256"]:raise ValueError("Model identity changed")
    expected=dict(frames=d["frame_count_inferred"],size=d["raw_identity"]["size"],
        md5=json.loads((PATHS.root/d["historical_receipt"]).read_text())["md5"],sha256=d["historical_sha256"])
    contract=dict(schema="hall-blocker-publisher-preparation-v2",dataset=dataset,view=VIEWS[dataset],
        inputs={k:dict(path=PATHS.relative(v),sha256=expected["sha256"] if k=="raw" else sha256(v),
                       size=v.stat().st_size) for k,v in paths.items()},
        expected_source=expected, config=CONFIG,
        scientific_protocol=describe(AUDIT/"protocol.json"),
        metadata_split_amendment=describe(REPORT/"split_amendment.json"),
        reader=describe(Path(__file__).with_name("dc01_data.py")),
        pairwise_guard_amendment="Latest request: guard validation from test too; discarded validation candidates are not recycled into fit.")
    key=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    return paths,expected,contract,key
def source_scan(dataset,paths,expected,out,key):
    meta_path=out/"all_metadata.npz";scan_path=out/"source_scan.json"
    scan_code=hashlib.sha256(inspect.getsource(source_scan).encode()).hexdigest()
    stat=paths["raw"].stat()
    signature=[stat.st_size,stat.st_mtime_ns,stat.st_ino]
    if scan_path.exists() and meta_path.exists():
        old=json.loads(scan_path.read_text())
        if old["contract_hash"]!=key or old["raw_stat"]!=signature or old["scan_code_sha256"]!=scan_code or old["metadata_sha256"]!=sha256(meta_path):
            raise ValueError("Completed metadata stage cannot be safely reused")
        emit(dataset=dataset,stage="source_scan",status="reused_verified_stage")
        return old
    count=expected["frames"]
    position=np.empty((count,3),np.float64);timestamp=np.empty(count,np.float64);age=np.empty(count,np.float64)
    snr=np.empty((count,64),np.float32);cfo=np.empty_like(snr)
    csi_hash=np.empty(count,dtype="S64");record_hash=np.empty(count,dtype="S64")
    csi_finite=np.empty(count,bool);csi_zero=np.empty(count,bool)
    samples=set(np.linspace(0,count-1,9,dtype=int).tolist())
    # Cache the CRC32C table for all short length headers; payload samples use the established decoder.
    table=[]
    for value in range(256):
        c=value
        for _ in range(8):c=(c>>1)^(0x82F63B78 if c&1 else 0)
        table.append(c)
    def header_crc(b):
        x=0xffffffff
        for byte in b:x=table[(x^byte)&255]^(x>>8)
        x^=0xffffffff
        return (((x>>15)|(x<<17))+0xa282ead8)&0xffffffff
    md5=hashlib.md5();whole_sha=hashlib.sha256();start=time.monotonic()
    with paths["raw"].open("rb") as f:
        for i in range(count):
            block=f.read(RECORD_BYTES)
            if len(block)!=RECORD_BYTES:raise ValueError("Truncated raw frame")
            length,crc=struct.unpack_from("<QI",block)
            if length!=RECORD_BYTES-16 or header_crc(block[:8])!=crc:raise ValueError("Raw length/CRC error")
            if i in samples and masked_crc(block[12:-4])!=struct.unpack_from("<I",block,RECORD_BYTES-4)[0]:
                raise ValueError("Raw sample payload CRC mismatch")
            md5.update(block);whole_sha.update(block)
            record=example(block[12:-4]);csi=record["csi"]
            if csi.shape!=(64,1024,2) or csi.dtype!=np.float32:raise ValueError("Raw tensor layout changed")
            if np.asarray(record["pos-tachy"]).shape!=(3,) or np.asarray(record["snr"]).shape!=(64,) or np.asarray(record["cfo"]).shape!=(64,):
                raise ValueError("Raw metadata tensor layout changed")
            position[i]=record["pos-tachy"];timestamp[i]=record["time"];age[i]=record["gt-interp-age-tachy"]
            snr[i]=record["snr"];cfo[i]=record["cfo"]
            csi_finite[i]=np.isfinite(csi).all();csi_zero[i]=not np.any(csi)
            csi_hash[i]=hashlib.sha256(csi.tobytes()).hexdigest().encode()
            record_hash[i]=hashlib.sha256(block).hexdigest().encode()
            if (i+1)%2048==0 or i+1==count:
                progress=dict(dataset=dataset,stage="full_source_metadata_and_hash",frames=i+1,total=count,
                              seconds=round(time.monotonic()-start,3))
                atomic_json(out/"scan_progress.json",progress);emit(**progress)
        if f.read(1):raise ValueError("Additional unexpected frames")
    last=paths["raw"].stat()
    if signature!=[last.st_size,last.st_mtime_ns,last.st_ino]:raise ValueError("Source changed during scan")
    if md5.hexdigest()!=expected["md5"] or whole_sha.hexdigest()!=expected["sha256"]:
        raise ValueError("Fresh full-file digest differs from pinned publisher/receipt identity")
    atomic_npz(meta_path,position=position,time=timestamp,age=age,snr=snr,cfo=cfo,
               csi_sha256=csi_hash,record_sha256=record_hash,csi_finite=csi_finite,csi_all_zero=csi_zero)
    scan=dict(contract_hash=key,scan_code_sha256=scan_code,frames=count,raw_stat=signature,
              md5=md5.hexdigest(),sha256=whole_sha.hexdigest(),metadata_sha256=sha256(meta_path),
              all_length_crc_checked=True,payload_crc_sample_indices=sorted(samples),
              all_source_csi_finite=bool(csi_finite.all()),zero_csi_frames=int(csi_zero.sum()),
              seconds=time.monotonic()-start)
    atomic_json(scan_path,scan)
    return scan
def metadata_quality(out):
    with np.load(out/"all_metadata.npz",allow_pickle=False) as z:m={k:z[k] for k in z.files}
    p,t,a=m["position"],m["time"],m["age"]
    valid=np.isfinite(p).all(1)&np.isfinite(t)&np.isfinite(a)&(a>=0)&m["csi_finite"]
    order=np.lexsort((np.arange(len(t)),t))
    delta=np.diff(t[order]);distance=np.linalg.norm(np.diff(p[order],axis=0),axis=1)
    moving=(delta>0)&np.isfinite(delta)&np.isfinite(distance)
    same_position=groups_info(np.ascontiguousarray(p).view("V24").reshape(-1))
    stale=[]
    for ids in same_position["groups"]:
        span=float(np.ptp(t[ids]))
        if span>1:stale.append(dict(source_ids=ids,time_span_s=span,position=p[ids[0]].tolist()))
    exact_record=groups_info(m["record_sha256"]);exact_csi=groups_info(m["csi_sha256"])
    for groups in [exact_record,exact_csi]:
        groups["cross_position_groups"]=[dict(source_ids=ids,max_pair_distance_m=float(np.linalg.norm(p[ids]-p[ids[0]],axis=1).max()),
            time_span_s=float(np.ptp(t[ids]))) for ids in groups["groups"] if not np.array_equal(p[ids],np.broadcast_to(p[ids[0]],(len(ids),3)))]
    audit=dict(frames=len(t),valid_format_and_position_frames=int(valid.sum()),invalid_source_ids=np.flatnonzero(~valid).tolist(),
        validity_rule="finite XYZ/time/age, nonnegative interpolation age, finite CSI; not selected by signal strength or prediction",
        position_bounds=[p[np.isfinite(p).all(1)].min(0).tolist(),p[np.isfinite(p).all(1)].max(0).tolist()],
        nonfinite_position=int((~np.isfinite(p).all(1)).sum()),nonfinite_time=int((~np.isfinite(t)).sum()),
        nonfinite_age=int((~np.isfinite(a)).sum()),negative_age=int((a<0).sum()),
        interpolation_age_s=quantile(a),time_range_s=[float(np.nanmin(t)),float(np.nanmax(t))],
        raw_index_time_nondecreasing=bool(np.all(np.diff(t)>=0)),raw_index_time_nonincreasing=bool(np.all(np.diff(t)<=0)),
        chronological_equal_timestamps=int((delta==0).sum()),positive_time_step_s=quantile(delta[delta>0]),
        chronological_step_distance_m=quantile(distance[moving]),apparent_step_speed_mps=quantile(distance[moving]/delta[moving]),
        exact_position_unique=same_position["unique"],exact_position_repeated_groups=same_position["duplicate_groups"],
        repeated_position_longer_than_1s_candidates=stale,
        repeated_position_note="Revisits/stationary motion are candidates, not automatically tracking loss.",
        nonfinite_snr_elements=int((~np.isfinite(m["snr"])).sum()),nonfinite_cfo_elements=int((~np.isfinite(m["cfo"])).sum()),
        all_zero_csi_frames=int(m["csi_all_zero"].sum()),independent_tachymeter_reference_log_available=False)
    atomic_json(out/"quality_audit.json",audit)
    atomic_json(out/"duplicate_audit.json",dict(exact_record=exact_record,exact_csi=exact_csi,
        exact_position_repetitions_are_not_duplicate_measurement_proof=True))
    atomic_npz(out/"quality_mask.npz",valid=valid,time_order=order)
    emit(stage="full_metadata_quality",view=out.name,valid=int(valid.sum()),frames=len(t),
         age_s=audit["interpolation_age_s"],speed_mps=audit["apparent_step_speed_mps"],
         identical_record_groups=exact_record["duplicate_groups"],identical_csi_groups=exact_csi["duplicate_groups"],
         exact_csi_cross_position_groups=len(exact_csi["cross_position_groups"]),
         stale_position_candidates=len(stale))
    return m,valid,audit
def export_model(source,out,spec):
    b=source.read_bytes()
    magic,version,total=struct.unpack_from("<III",b)
    if magic!=0x46546c67 or version!=2 or total!=len(b):raise ValueError("GLB header")
    offset=12;chunks={}
    while offset<len(b):
        n,kind=struct.unpack_from("<II",b,offset);offset+=8;chunks[kind]=b[offset:offset+n];offset+=n
    d=json.loads(chunks[0x4e4f534a]);binary=chunks[0x004e4942]
    def access(i):
        a=d["accessors"][i];v=d["bufferViews"][a["bufferView"]]
        dtype=np.dtype({5126:"<f4",5123:"<u2",5125:"<u4"}[a["componentType"]])
        width={"SCALAR":1,"VEC3":3}[a["type"]]
        return np.ndarray((a["count"],width),dtype=dtype,buffer=binary,
            offset=v.get("byteOffset",0)+a.get("byteOffset",0),
            strides=(v.get("byteStride",dtype.itemsize*width),dtype.itemsize)).copy()
    arrays={};objects=[];antenna_checks={}
    for ni in d["scenes"][d.get("scene",0)]["nodes"]:
        node=d["nodes"][ni]
        if "mesh" not in node:continue
        if any(k in node for k in ["matrix","translation","rotation","scale"]):raise ValueError("Unexpected mesh node transform")
        for pi,prim in enumerate(d["meshes"][node["mesh"]]["primitives"]):
            if prim.get("mode",4)!=4:raise ValueError("Nontriangle primitive")
            vg=access(prim["attributes"]["POSITION"]).astype(np.float64)
            vertices=vg@ROTATION.T;faces=access(prim["indices"]).reshape(-1,3).astype(np.int64)
            if not np.isfinite(vertices).all() or faces.min()<0 or faces.max()>=len(vertices):raise ValueError("Invalid geometry")
            index=len(objects);arrays["vertices_"+str(index)]=vertices;arrays["faces_"+str(index)]=faces
            material=d["materials"][prim["material"]]["name"]
            objects.append(dict(name=node["name"],primitive=pi,material_name=material,
                 vertices_key="vertices_"+str(index),faces_key="faces_"+str(index),
                 include_in_future_rf_scene=not node["name"].startswith("antenna-array")))
            for label,a in zip(["north","south"],spec["antennas"]):
                if node["name"]=="antenna-array-"+label+"-patches":
                    center=(vertices.min(0)+vertices.max(0))/2
                    antenna_checks[label]=dict(model_center=center.tolist(),spec_location=a["location"],
                        center_difference_m=float(np.linalg.norm(center-np.asarray(a["location"]))),
                        antenna_direction_source="spec, not visual mesh normal")
    if max(x["center_difference_m"] for x in antenna_checks.values())>2e-4:raise ValueError("Model/spec center check failed")
    atomic_npz(out/"model_measurement_frame.npz",**arrays,rotation_gltf_to_measurement=ROTATION)
    atomic_json(out/"model_objects.json",dict(source=describe(source),transform="(x,-z,y)",objects=objects,
        antenna_checks=antenna_checks,rf_materials_fitted=False,
        note="Only coordinate conversion; no ray tracing, map fitting or RF material estimation."))
def prepare(dataset,resume=False,stage="all"):
    start=time.monotonic();paths,expected,contract,key=inputs(dataset)
    out=PATHS.artifacts/"measured"/VIEWS[dataset]
    if out.exists():
        if not resume:raise FileExistsError("Refusing existing view without --resume")
        state=json.loads((out/"run_state.json").read_text())
        if state["contract_hash"]!=key:raise ValueError("Preparation contract changed")
        if (out/"preparation_manifest.json").exists():
            manifest=json.loads((out/"preparation_manifest.json").read_text())
            for value in manifest["outputs"].values():
                if sha256(PATHS.root/value["path"])!=value["sha256"]:raise ValueError("Completed output identity changed")
            emit(dataset=dataset,status="already_complete_integrity_verified");return
    else:
        out.mkdir(parents=True,exist_ok=False);atomic_json(out/"contract.json",contract)
    atomic_json(out/"run_state.json",dict(contract_hash=key,status="running",stage=stage))
    try:
        scan=source_scan(dataset,paths,expected,out,key)
        m,valid,quality=metadata_quality(out)
        if stage=="scan":
            atomic_json(out/"run_state.json",dict(contract_hash=key,status="source_audited",stage=stage))
            return
        selected=select_positions(m["position"],valid)
        p=m["position"][selected];split=split_positions(p)
        assignments=np.asarray([a["assignments"] for a in json.loads(paths["spec"].read_text())["antennas"]],np.int64)
        if assignments.shape!=(2,4,8) or sorted(assignments.ravel().tolist())!=list(range(64)):raise ValueError("Port mapping")
        # Exact CSI duplicates must not create a signal-identical fit/held shortcut.
        hashes=m["csi_sha256"][selected]
        for fold in split["splits"]:
            for aa,bb in [("fit","validation"),("fit","test"),("validation","test")]:
                if np.intersect1d(hashes[fold[aa]],hashes[fold[bb]]).size:
                    raise ValueError("Identical selected CSI crosses spatial partitions; review source duplicates")
        spec=json.loads(paths["spec"].read_text());offsets=json.loads(paths["offsets"].read_text())
        atomic_npz(out/"selection.npz",source_id=selected,position=p,original_position=p,time=m["time"][selected],
            interpolation_age=m["age"][selected],frequency_indices=np.arange(1024,dtype=np.int64),
            frequency_hz=spec["frequency"]+(np.arange(1024)-512)*spec["bandwidth"]/1024,
            antenna_assignments=assignments,panel_position=np.asarray([a["location"] for a in spec["antennas"]],np.float64),
            panel_direction=np.asarray([a["direction"] for a in spec["antennas"]],np.float64),
            measurement_frame_origin=np.zeros(3),measurement_frame_rotation=np.eye(3))
        atomic_json(out/"split_manifest.json",split)
        export_model(paths["model"],out,spec)
        shape=(len(selected),2,8,4,1024);partial=out/"channels.partial.npy";progress=out/"conversion_progress.json"
        row_start=0;max_power_change=0.
        if partial.exists() and progress.exists():
            state=json.loads(progress.read_text())
            if state["contract_hash"]!=key or state["selected_sha256"]!=sha256(out/"selection.npz"):raise ValueError("Conversion resume identity")
            row_start=state["rows"];max_power_change=state["max_power_relative_change"]
            channel=np.lib.format.open_memmap(partial,mode="r+")
            if channel.shape!=shape or channel.dtype!=np.complex64:raise ValueError("Partial layout")
        else:channel=np.lib.format.open_memmap(partial,mode="w+",dtype=np.complex64,shape=shape)
        with paths["raw"].open("rb") as f:
            for row in range(row_start,len(selected)):
                f.seek(int(selected[row])*RECORD_BYTES);record=example(f.read(RECORD_BYTES)[12:-4])
                value,shifted=calibrate(record["csi"],offsets,assignments)
                if not np.isfinite(value).all():raise ValueError("Nonfinite prepared channel")
                raw_panel=shifted[assignments].transpose(0,2,1,3).astype(np.complex128)
                before=(np.abs(raw_panel)**2).sum((1,2,3));after=(np.abs(value.astype(np.complex128))**2).sum((1,2,3))
                diff=np.divide(np.abs(after-before),before,out=np.abs(after-before),where=before>0)
                max_power_change=max(max_power_change,float(diff.max()))
                if max_power_change>1e-6:raise ValueError("Calibration changed per-array total energy")
                channel[row]=value
                if (row+1)%128==0 or row+1==len(selected):
                    channel.flush()
                    state=dict(contract_hash=key,selected_sha256=sha256(out/"selection.npz"),
                        rows=row+1,total=len(selected),max_power_relative_change=max_power_change)
                    atomic_json(progress,state);emit(dataset=dataset,stage="calibration_conversion",rows=row+1,total=len(selected))
        channel.flush();del channel;os.replace(partial,out/"channels.npy")
        files=["contract.json","all_metadata.npz","source_scan.json","quality_audit.json","duplicate_audit.json",
               "quality_mask.npz","selection.npz","split_manifest.json","model_measurement_frame.npz","model_objects.json","channels.npy"]
        manifest=dict(contract_hash=key,schema=contract["schema"],status="complete",dataset=dataset,view=VIEWS[dataset],
            source_records=expected["frames"],selected_points=len(selected),channel_shape=list(shape),dtype="complex64",
            source_scan=scan,conversion=dict(max_power_relative_change=max_power_change,
                formula="fftshift(raw)[a,k]*exp(+j*(cpo[a]+2*pi*sto[a]*k/1024)); all1024 bins, then port reordering",
                no_truth_assisted_alignment=True),
            outputs={n:describe(out/n) for n in files},preparer=describe(Path(__file__)),
            runtime=dict(device="CPU",python=sys.version.split()[0],numpy=np.__version__,
                         elapsed_s=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),
            limits=["public calibration origin/residual accuracy not certified","no original mobile-TX absolute reference recovery",
                    "full positioning audit is diagnostic, not independent tracking certification",
                    "no model training, inference, ray tracing, or performance scoring"])
        atomic_json(out/"preparation_manifest.json",manifest)
        atomic_json(out/"run_state.json",dict(contract_hash=key,status="complete"))
        emit(dataset=dataset,status="prepared",points=len(selected),shape=shape,seconds=time.monotonic()-start)
    except BaseException:
        (out/"failure.txt").write_text(traceback.format_exc())
        atomic_json(out/"run_state.json",dict(contract_hash=key,status="failed"))
        raise
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=["dc10","dc20","both"],default="both")
    parser.add_argument("--stage",choices=["scan","all"],default="all")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    for dataset in (["dc10","dc20"] if args.dataset=="both" else [args.dataset]):
        prepare(dataset,args.resume,args.stage)
if __name__=="__main__":main()

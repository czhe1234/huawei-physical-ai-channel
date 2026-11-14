"""Independent CPU verification of blocker views; never evaluates an algorithm."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np
from project_paths import PATHS
from dc01_data import example
ROOT=PATHS.root
NAMES={"dc10":"hall_sw_publisher_v2","dc20":"hall_nw_publisher_v2"}
REPORT=PATHS.analysis/"results/hall_blocker_preparation_v1"
def require(x,message):
    if not x:raise ValueError(message)
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def save(p,d):
    tmp=p.with_name(p.name+".partial")
    tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n")
    tmp.replace(p)
def minimum_distance(a,b):
    result=np.empty(len(a),np.float64)
    for start in range(0,len(a),128):
        diff=a[start:start+128,None,:2]-b[None,:,:2]
        result[start:start+128]=np.sqrt((diff*diff).sum(-1).min(1))
    return result
def reference_selection(position,valid):
    choices={}
    for i in np.flatnonzero(valid):
        x,y=position[i,:2];key=(int(np.floor(x/.20)),int(np.floor(y/.20)))
        distance=(x-(key[0]+.5)*.20)**2+(y-(key[1]+.5)*.20)**2
        value=(float(distance),int(i))
        if key not in choices or value<choices[key]:choices[key]=value
    return np.array(sorted(x[1] for x in choices.values()),np.int64)
def verify_splits(position,protocol,hashes):
    from collections import Counter
    tiles=[(int(np.floor(p[0]/4)),int(np.floor(p[1]/4))) for p in position]
    count=Counter(tiles);assignment={};totals=[0]*5
    for block in sorted(count,key=lambda b:(-count[b],b[0],b[1])):
        f=min(range(5),key=lambda i:(totals[i],i))
        assignment[block]=f;totals[f]+=count[block]
    color=np.array([assignment[b] for b in tiles],dtype=np.int64)
    require(np.array_equal(color,np.array(protocol["fold"])),"Stored whole-block allocation differs")
    require(len(count)>=5,"Fewer than five occupied blocks")
    coverage=np.zeros(len(position),np.int64);results=[]
    for fold in protocol["splits"]:
        outer=fold["outer"];test=np.flatnonzero(color==outer)
        vc=np.flatnonzero(color==(outer+1)%5)
        val=vc[minimum_distance(position[vc],position[test])>=.30]
        fc=np.flatnonzero((color!=outer)&(color!=(outer+1)%5))
        held=np.r_[test,vc]
        fit=fc[minimum_distance(position[fc],position[held])>=.30]
        arrays={"fit":fit,"validation":val,"test":test}
        for key,value in arrays.items():
            require(np.array_equal(value,np.array(fold[key],dtype=np.int64)),"Partition not reconstructed from coordinates")
            require(len(value)>=16,"Insufficient support")
        distances={}
        for a,b in [("fit","validation"),("fit","test"),("validation","test")]:
            require(not set(arrays[a])&set(arrays[b]),"Partition IDs overlap")
            d=minimum_distance(position[arrays[a]],position[arrays[b]])
            require(float(d.min())>=.30-1e-12,"Independent pairwise spatial guard failed")
            require(np.intersect1d(hashes[arrays[a]],hashes[arrays[b]]).size==0,"Identical raw CSI crosses partitions")
            distances[a+"__"+b]=float(d.min())
        coverage[test]+=1
        results.append(dict(outer=outer,fit=len(fit),validation=len(val),test=len(test),
                           pairwise_minimum_distance_m=distances))
    require(np.array_equal(coverage,np.ones(len(position),np.int64)),"Test coverage is not exactly once")
    return dict(test_once=True,all_pairwise_guards_passed=True,duplicate_CSI_across_partitions=False,folds=results)
def verify_model(out,source,spec,all_positions):
    b=source.read_bytes();offset=12;chunks={}
    while offset<len(b):
        n,t=struct.unpack_from("<II",b,offset);offset+=8;chunks[t]=b[offset:offset+n];offset+=n
    d=json.loads(chunks[0x4e4f534a]);binary=chunks[0x004e4942]
    def acc(index):
        a=d["accessors"][index];v=d["bufferViews"][a["bufferView"]]
        dt=np.dtype({5126:"<f4",5123:"<u2",5125:"<u4"}[a["componentType"]])
        width={"VEC3":3,"SCALAR":1}[a["type"]]
        return np.ndarray((a["count"],width),dtype=dt,buffer=binary,
            offset=v.get("byteOffset",0)+a.get("byteOffset",0),
            strides=(v.get("byteStride",width*dt.itemsize),dt.itemsize)).copy()
    objects=json.loads((out/"model_objects.json").read_text())
    named={node["name"]:node for node in d["nodes"] if "mesh" in node}
    vertex_count=0;bounds_min=np.full(3,np.inf);bounds_max=np.full(3,-np.inf);array_errors={}
    with np.load(out/"model_measurement_frame.npz",allow_pickle=False) as z:
        for x in objects["objects"]:
            prim=d["meshes"][named[x["name"]]["mesh"]]["primitives"][x["primitive"]]
            raw=acc(prim["attributes"]["POSITION"]).astype(np.float64)
            expected=np.column_stack((raw[:,0],-raw[:,2],raw[:,1]))
            require(np.array_equal(expected,z[x["vertices_key"]]),"Model coordinates differ from independent axis conversion")
            require(np.array_equal(acc(prim["indices"]).reshape(-1,3),z[x["faces_key"]]),"Triangle identities changed")
            vertex_count+=len(expected)
            bounds_min=np.minimum(bounds_min,expected.min(0));bounds_max=np.maximum(bounds_max,expected.max(0))
            for label,a in zip(["north","south"],spec["antennas"]):
                if x["name"]=="antenna-array-"+label+"-patches":
                    center=(expected.min(0)+expected.max(0))/2
                    error=float(np.linalg.norm(center-np.asarray(a["location"])))
                    require(error<=2e-4,"Independent model/spec array center mismatch")
                    array_errors[label]=error
        require(np.linalg.det(z["rotation_gltf_to_measurement"])==1,"Coordinate mapping is a reflection")
    require(set(array_errors)=={"north","south"},"Missing antenna center cross-check")
    inside=np.all((all_positions>=bounds_min-.05)&(all_positions<=bounds_max+.05),axis=1)
    require(bool(inside.all()),"Raw measurement positions outside global model bounds")
    return dict(all_vertices_checked=vertex_count,axis_conversion_exact=True,triangle_identities_preserved=True,
                array_center_error_m=array_errors,global_model_bounds_m=[bounds_min.tolist(),bounds_max.tolist()],
                raw_positions_inside_model_global_bounds_5cm=int(inside.sum()),raw_positions_checked=len(all_positions),
                physical_registration_accuracy_independently_certified=False,
                array_direction_source="publisher spec")
def verify(dataset):
    start=time.monotonic();out=PATHS.artifacts/"measured"/NAMES[dataset]
    manifest=json.loads((out/"preparation_manifest.json").read_text())
    contract=json.loads((out/"contract.json").read_text())
    for desc in manifest["outputs"].values():
        p=ROOT/desc["path"]
        require(p.is_file() and p.stat().st_size==desc["size"] and sha256(p)==desc["sha256"],"Output content identity mismatch")
    with np.load(out/"all_metadata.npz",allow_pickle=False) as z:meta={k:z[k] for k in z.files}
    with np.load(out/"selection.npz",allow_pickle=False) as z:selection={k:z[k] for k in z.files}
    with np.load(out/"quality_mask.npz",allow_pickle=False) as z:valid=z["valid"]
    raw_valid=np.isfinite(meta["position"]).all(1)&np.isfinite(meta["time"])&np.isfinite(meta["age"])&(meta["age"]>=0)&meta["csi_finite"]
    require(np.array_equal(raw_valid,valid),"Quality mask differs from registered rule")
    expected_ids=reference_selection(meta["position"],valid)
    require(np.array_equal(expected_ids,selection["source_id"]),"Representative selection differs")
    require(np.array_equal(meta["position"][expected_ids],selection["position"]),"Measurement coordinate positions changed")
    spec=json.loads((ROOT/contract["inputs"]["spec"]["path"]).read_text())
    offsets=json.loads((ROOT/contract["inputs"]["offsets"]["path"]).read_text())
    assignments=np.array([a["assignments"] for a in spec["antennas"]],np.int64)
    require(np.array_equal(assignments,selection["antenna_assignments"]),"Port table differs")
    require(sorted(assignments.ravel().tolist())==list(range(64)),"Port mapping not one-to-one")
    require(np.array_equal(selection["panel_position"],np.array([a["location"] for a in spec["antennas"]])),"Panel centers changed")
    require(np.array_equal(selection["panel_direction"],np.array([a["direction"] for a in spec["antennas"]])),"Panel directions changed")
    require(np.array_equal(selection["frequency_indices"],np.arange(1024)),"Frequency subsampling introduced")
    expected_f=np.array([spec["frequency"]+(i-512)*spec["bandwidth"]/1024 for i in range(1024)],np.float64)
    require(np.array_equal(expected_f,selection["frequency_hz"]),"Physical frequency axis differs")
    split=json.loads((out/"split_manifest.json").read_text())
    spatial=verify_splits(selection["position"],split,meta["csi_sha256"][expected_ids])
    channels=np.load(out/"channels.npy",mmap_mode="r",allow_pickle=False)
    require(channels.shape==(len(expected_ids),2,8,4,1024) and channels.dtype==np.complex64,"Prepared tensor layout differs")
    k=np.arange(1024);raw_index=(k+512)%1024
    phase=np.array(offsets["cpo"])[:,None]+2*np.pi*np.array(offsets["sto"])[:,None]*k/1024
    unit=np.cos(phase)+1j*np.sin(phase)
    max_element_relative_error=0.;max_energy_relative_change=0.
    raw_path=ROOT/contract["inputs"]["raw"]["path"]
    with raw_path.open("rb") as f:
        for row,source_id in enumerate(expected_ids):
            f.seek(int(source_id)*525022);block=f.read(525022)
            require(hashlib.sha256(block).hexdigest().encode()==meta["record_sha256"][source_id],"Selected source record identity differs")
            record=example(block[12:-4]);source=record["csi"]
            require(np.array_equal(record["pos-tachy"],selection["position"][row]),"Selected source position differs")
            require(record["time"]==float(selection["time"][row]),"Selected source time differs")
            # Direct original-bin addressing and cos/sin in float64, independently of production fftshift/exp.
            z=source[...,0].astype(np.float64)+1j*source[...,1].astype(np.float64)
            ref=np.empty((2,8,4,1024),np.complex128)
            raw_energy=np.zeros(2,np.float64)
            for panel in range(2):
                for horizontal in range(8):
                    for vertical in range(4):
                        a=int(assignments[panel,vertical,horizontal])
                        zrow=z[a,raw_index]
                        ref[panel,horizontal,vertical]=zrow*unit[a]
                        raw_energy[panel]+=np.sum(np.abs(zrow)**2)
            current=channels[row].astype(np.complex128)
            require(np.isfinite(current).all(),"Nonfinite prepared data")
            relative=np.abs(current-ref)/np.maximum(np.abs(ref),1e-20)
            max_element_relative_error=max(max_element_relative_error,float(relative.max()))
            require(np.all(np.abs(current-ref)<=5e-7*np.abs(ref)+5e-13),"All-row independent calibration comparison failed")
            energy=np.sum(np.abs(current)**2,axis=(1,2,3))
            diff=np.divide(np.abs(energy-raw_energy),raw_energy,out=np.abs(energy-raw_energy),where=raw_energy>0)
            max_energy_relative_change=max(max_energy_relative_change,float(diff.max()))
            if (row+1)%256==0 or row+1==len(expected_ids):
                print(json.dumps(dict(dataset=dataset,stage="independent_conversion_verification",rows=row+1,total=len(expected_ids))),flush=True)
    require(max_energy_relative_change<=1e-6,"Unit-modulus calibration energy invariant failed")
    model=verify_model(out,ROOT/contract["inputs"]["model"]["path"],spec,meta["position"])
    result=dict(status="passed",dataset=dataset,selected_points=len(expected_ids),source_records=len(meta["time"]),
        all_output_hashes_verified=True,all_selected_source_records_checked=True,
        calibration_elementwise_relative_error_max=max_element_relative_error,
        calibration_array_energy_relative_change_max=max_energy_relative_change,
        all_frequency_bins=1024,port_map_bijective=True,spatial=spatial,model=model,
        seconds=time.monotonic()-start,algorithm_predictions_or_scores_computed=False)
    save(out/"verification.json",result)
    save(REPORT/(dataset+"_verification.json"),result)
    print(json.dumps(dict(dataset=dataset,status="verified",selected_points=len(expected_ids),
        calibration_elementwise_relative_error_max=max_element_relative_error,
        calibration_array_energy_relative_change_max=max_energy_relative_change,
        minimum_pairwise_distance_m=min(v for x in spatial["folds"] for v in x["pairwise_minimum_distance_m"].values()),
        seconds=result["seconds"])),flush=True)
    return result
def cross_source():
    metadata={}
    for name,view in NAMES.items():
        with np.load(PATHS.artifacts/"measured"/view/"all_metadata.npz",allow_pickle=False) as z:
            metadata[name]={k:z[k] for k in ["position","time","record_sha256","csi_sha256"]}
    a,b=metadata["dc10"],metadata["dc20"]
    result=dict(exact_raw_record_hashes_shared=int(len(np.intersect1d(a["record_sha256"],b["record_sha256"]))),
        exact_CSI_hashes_shared=int(len(np.intersect1d(a["csi_sha256"],b["csi_sha256"]))),
        xy_020m_cells_shared=int(len(set(map(tuple,np.floor(a["position"][:,:2]/.2).astype(int)))&
                                       set(map(tuple,np.floor(b["position"][:,:2]/.2).astype(int))))),
        same_physical_environment=True,cross_record_measurement_or_weight_reuse_allowed=False,
        no_joint_guard_claim="Each configuration fits its own partitions; overlap is not independent-site evidence.")
    save(REPORT/"cross_source_check.json",result)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--dataset",choices=["dc10","dc20","both"],default="both")
    args=parser.parse_args()
    for dataset in (["dc10","dc20"] if args.dataset=="both" else [args.dataset]):verify(dataset)
    if args.dataset=="both":cross_source()
if __name__=="__main__":main()

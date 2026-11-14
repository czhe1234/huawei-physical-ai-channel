"""Independent CPU verification of development full-H predictions; no formal test."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from project_paths import PATHS
from .common import FoldData,ART,REPORT
ROOT=REPORT/"fit_repair_dev_20261101"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def stats(pred,target):
 p=np.asarray(pred,np.complex128);t=np.asarray(target,np.complex128)
 te=float(np.vdot(t,t).real);pe=float(np.vdot(p,p).real);se=float(np.vdot(p-t,p-t).real)
 cross=np.vdot(p,t)
 return dict(truth_energy=te,prediction_energy=pe,squared_error=se,
  nmse=se/te if te else None,energy_ratio=pe/te if te else None,
  signed_real_complex_correlation=float(cross.real/math.sqrt(te*pe)) if te*pe else None,
  prediction_mean_power=pe/p.size,truth_mean_power=te/t.size,complex_scalars=p.size)
def main(args):
 panel=json.loads((ROOT/"panel.json").read_text());contract=json.loads((ROOT/"contract.json").read_text())
 rows=np.array(panel["train_ids"],np.int64);data=FoldData(0,"screen",args.method)
 assert not set(rows)&set(data.val_ids) and not set(rows)&set(data.outer_ids)
 art=ART/"fit_repair_dev_20261101"/args.method;report=ROOT/args.method
 predfile=art/("endpoint_fit_prediction.npy" if args.kind=="endpoint" else "final_fit_prediction.npy")
 pred=np.load(predfile,allow_pickle=False)
 ids=np.load(art/"fit_row_ids.npy",allow_pickle=False)
 assert np.array_equal(ids,rows)
 assert pred.shape==(16,256,4,192) and pred.dtype==np.complex64 and np.isfinite(pred).all()
 target=data.read_fit(rows,normalized=False)
 st=stats(pred,target);fields=[]
 for field in range(8):
  if args.method=="nerf2":
   s,b,u=field//4,(field//2)%2,field%2;ix=np.flatnonzero(data.sites(rows)==s)
   pp=pred[ix,b*128:(b+1)*128,u*2:(u+1)*2,:];tt=target[ix,b*128:(b+1)*128,u*2:(u+1)*2,:]
   name=f"s{s}_b{b}_u{u}"
  else:
   b,u=divmod(field,4);pp=pred[:,b*128:(b+1)*128,u:u+1,:];tt=target[:,b*128:(b+1)*128,u:u+1,:]
   name=f"b{b}_u{u}"
  fields.append(dict(field=field,name=name,**stats(pp,tt)))
 positions=[dict(row=int(row),site=int(data.sites([row])[0]),**stats(p,t)) for row,p,t in zip(rows,pred,target)]
 gates={}
 for name,g in contract["acceptance"].items():
  gates[name]=(st["nmse"]<=g["pooled_nmse_max"] and all(
    f["nmse"] is not None and f["nmse"]<=g["all_field_nmse_max"] and
    g["all_field_energy_ratio_min"]<=f["energy_ratio"]<=g["all_field_energy_ratio_max"] for f in fields))
 positive=[p for p in positions if p["truth_energy"]>0];zero=[p for p in positions if p["truth_energy"]==0]
 result=dict(method=args.method,checkpoint_kind=args.kind,status="verified",prediction_sha256=sha(predfile),
  no_formal_test=True,full_axes=True,train_rows=rows.tolist(),pooled=st,fields=fields,positions=positions,
  gates=gates,nonzero_rows_nmse_below1=sum(p["nmse"]<1 for p in positive),nonzero_rows_count=len(positive),
  zero_rows=[dict(row=p["row"],mean_prediction_power=p["prediction_mean_power"],
          leakage_to_parent_FIT_mean_power=p["prediction_mean_power"]/data.scale**2) for p in zero],
  note="Full-H FIT fitting only. Passing aggregate gates does not imply every weak position fits, nor spatial generalization.")
 q=art/"final_fit_query_prediction.npy"
 if args.kind=="selected" and q.exists():
  v=np.load(q,allow_pickle=False);assert v.shape==pred.shape and v.dtype==np.complex64 and np.isfinite(v).all()
  qfields=[]
  for field in range(8):
   if args.method=="nerf2":
    site,b,u=field//4,(field//2)%2,field%2;ix=np.flatnonzero(data.sites(rows)==site)
    pp=v[ix,b*128:(b+1)*128,u*2:(u+1)*2,:];tt=target[ix,b*128:(b+1)*128,u*2:(u+1)*2,:]
   else:
    b,u=divmod(field,4);pp=v[:,b*128:(b+1)*128,u:u+1,:];tt=target[:,b*128:(b+1)*128,u:u+1,:]
   qfields.append(dict(field=field,**stats(pp,tt)))
  qs=stats(v,target)
  qgates={}
  for name,g in contract["acceptance"].items():
   qgates[name]=(qs["nmse"]<=g["pooled_nmse_max"] and all(f["nmse"]<=g["all_field_nmse_max"] and g["all_field_energy_ratio_min"]<=f["energy_ratio"]<=g["all_field_energy_ratio_max"] for f in qfields))
  result["FIT_query_route"]=dict(pooled=qs,fields=qfields,gates=qgates)
 vp=art/"final_validation_prediction.npy";vi=art/"validation_row_ids.npy"
 if not vp.exists() and (art/"final_validation_subset_prediction.npy").exists():vp=art/"final_validation_subset_prediction.npy"
 if args.kind=="selected" and vp.exists() and vi.exists():
  vids=np.load(vi,allow_pickle=False);v=np.load(vp,allow_pickle=False)
  assert np.array_equal(vids,np.array(panel["validation_ids"],np.int64))
  assert v.dtype==np.complex64 and np.isfinite(v).all()
  if v.shape==(len(vids),256,4,192):
   y=data.read_validation(vids,normalized=False);result["validation_full_H"]=stats(v,y)
  else:
   axes_path=art/"validation_axes.json"
   assert axes_path.exists(),"Subtensor validation requires explicit axes"
   axes=json.loads(axes_path.read_text())
   bs=np.array(axes["BS_ids"],np.int64);ue=np.array(axes["UE_ids"],np.int64);ff=np.array(axes["frequency_ids"],np.int64)
   assert v.shape==(len(vids),len(bs),len(ue),len(ff))
   y=data.read_validation(vids,ff,normalized=False)
   y=np.take(np.take(y,bs,axis=1),ue,axis=2)
   result["validation_subtensor"]=dict(axes=axes,metrics=stats(v,y))
 # Frozen training modules, all old checkpoint stats, comparison/V16 source hashes.
 old=json.loads((ROOT/"frozen_sources_and_models_before.json").read_text());unchanged=[]
 for p,h in old["source_sha256"].items():unchanged.append(sha(PATHS.root/p)==h)
 for oldst in old["old_model_stat"]:
  s=(PATHS.root/oldst["path"]).stat()
  unchanged.append(s.st_size==oldst["size"] and s.st_mtime_ns==oldst["mtime_ns"] and s.st_ino==oldst["inode"])
 for p,h in json.loads((ROOT/"protected_before.json").read_text()).items():unchanged.append(sha(PATHS.root/p)==h)
 assert all(unchanged),"Protected old file changed"
 result["protected_old_state_unchanged"]=True
 report.mkdir(parents=True,exist_ok=True);p=report/("independent_endpoint_verification.json" if args.kind=="endpoint" else "independent_verification.json")
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
 with (report/("independent_endpoint_per_position.csv" if args.kind=="endpoint" else "independent_per_position.csv")).open("w") as f:
  w=csv.DictWriter(f,fieldnames=list(positions[0]));w.writeheader();w.writerows(positions)
 print(json.dumps(dict(method=args.method,pooled=st,gates=gates,fields=fields,
          nonzero_rows_improved=result["nonzero_rows_nmse_below1"],old_preserved=True)),flush=True)
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--method",choices=["nerf2","newrf"],required=True);a.add_argument("--kind",choices=["selected","endpoint"],default="selected");main(a.parse_args())

"""Read-only original-data eligibility audit for the L-shaped factory candidate."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime,hashlib,json,math,re,urllib.request
from project_paths import PATHS
REPORT=PATHS.analysis/"audits/factory_lshape_20261029"
GROUPS=[("dichasus-cf0x","2854",290317),("dichasus-cf1x-part1","3150",148656),("dichasus-cf1x-part2","3151",148663)]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fetch(task):
    name,url=task
    try:
        request=urllib.request.Request(url,headers={"User-Agent":"DICHASUS-eligibility-audit/1.0"})
        with urllib.request.urlopen(request,timeout=15) as response:
            content=response.read(2*1024*1024+1)
            if len(content)>2*1024*1024:raise ValueError("Small-metadata size bound exceeded")
            value=json.loads(content)
            return name,dict(ok=True,url=url,final_url=response.geturl(),status=response.status,bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),md5=hashlib.md5(content).hexdigest()),content,value
    except Exception as e:return name,dict(ok=False,url=url,error=repr(e)),None,None
def main():
    if REPORT.exists():raise FileExistsError("Preserve previous audit")
    REPORT.mkdir(parents=True)
    raw=PATHS.datasets/"dichasus"
    ledger=PATHS.root/"experiment_records/EXPERIMENT_RECORD.md";prefix=ledger.read_bytes()
    groups=[x[0] for x in GROUPS]
    receipts=[]
    for p in (PATHS.analysis/"dichasus_acquisition_20261016/server_verification").glob("*.json"):
        d=json.loads(p.read_text())
        if any(d.get("relative_path","").startswith(g+"/") for g in groups):receipts.append(dict(receipt=str(p.relative_to(PATHS.root)),record=d))
    original_files={}
    for group in groups:
        for p in (raw/group).rglob("*"):
            if p.is_file():original_files[str(p.relative_to(PATHS.root))]=dict(bytes=p.stat().st_size,sha256=digest(p))
    raw_candidates=[str(p.relative_to(PATHS.root)) for p in raw.rglob("*") if p.is_file() and re.search(r"dichasus-cf[01].*tfrecord",p.name,re.I)]
    tasks=[]
    for group,doi,spec in GROUPS:
        tasks.append((group+"_metadata.json","https://darus.uni-stuttgart.de/api/datasets/:persistentId/?persistentId=doi:10.18419/darus-"+doi))
        tasks.append((group+"_spec.json","https://darus.uni-stuttgart.de/api/access/datafile/"+str(spec)))
        for p in (raw/group/"context").glob("*.json"):
            tasks.append(("current_"+p.name,"https://dichasus.inue.uni-stuttgart.de/datasets/data/"+group+"/"+p.name))
    fetched={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for name,receipt,content,value in pool.map(fetch,tasks):
            if content is not None:(REPORT/name).write_bytes(content)
            fetched[name]=dict(receipt=receipt,value=value)
            print("FETCH",name,receipt["ok"],flush=True)
    summaries=[]
    for group,doi,specid in GROUPS:
        metadata=fetched[group+"_metadata.json"]["value"]
        spec=fetched[group+"_spec.json"]["value"]
        summary=dict(group=group,persistentId="doi:10.18419/darus-"+doi,metadata_available=metadata is not None,
            current_spec_available=spec is not None,local_CSI_files=[p for p in raw_candidates if "/"+group+"/" in p])
        if metadata is not None:
            version=metadata["data"]["latestVersion"];files=version["files"]
            summary.update(version=str(version["versionNumber"])+"."+str(version["versionMinorNumber"]),
                release_time=version.get("releaseTime"),state=version.get("versionState"),
                official_files=[dict(name=x["dataFile"]["filename"],id=x["dataFile"]["id"],bytes=x["dataFile"]["filesize"],
                    checksum=x["dataFile"].get("checksum")) for x in files])
            spec_meta=next(x["dataFile"] for x in files if x["dataFile"]["filename"]=="spec.json")
            assert spec_meta["id"]==specid
            if spec is not None and spec_meta.get("checksum",{}).get("type")=="MD5":
                assert fetched[group+"_spec.json"]["receipt"]["md5"]==spec_meta["checksum"]["value"]
                summary["spec_publisher_MD5_matches"]=True
        if spec is not None:
            mapping=[];panels=[]
            for a in spec.get("antennas",[]):
                flat=[v for row in a.get("assignments",[]) for v in row]
                valid=[v for v in flat if isinstance(v,int) and v>=0]
                mapping.extend(valid)
                panels.append(dict(name=a.get("name"),assignments=a.get("assignments"),valid_ports=len(valid),
                    location=a.get("location"),spacingX=a.get("spacingX"),spacingY=a.get("spacingY")))
            summary.update(carrier_hz=spec.get("frequency"),bandwidth_hz=spec.get("bandwidth"),antennas=panels,
                mapped_ports=len(mapping),unique_ports=len(set(mapping)),
                port_mapping_complete=(sorted(mapping)==list(range(len(mapping)))))
        calibration=[]
        for p in sorted((raw/group/"context").glob("*.json")):
            d=json.loads(p.read_text());v={k:d.get(k,[]) for k in ["cpo","sto"]}
            n=len(v["cpo"])
            assert n==len(v["sto"]) and n>0 and all(math.isfinite(float(t)) for a in v.values() for t in a)
            rec=next((x for x in receipts if x["record"].get("relative_path")==str(p.relative_to(raw))),None)
            localhash=digest(p)
            if rec is not None:assert localhash==rec["record"]["sha256"]
            current=fetched["current_"+p.name]
            calibration.append(dict(filename=p.name,channels=n,finite=True,CPO_first=v["cpo"][0],STO_first=v["sto"][0],
                local_sha256=localhash,historical_receipt_present=rec is not None,
                publisher_checksum_in_receipt=rec["record"].get("publisher_checksum") if rec else None,
                current_fetch_ok=current["receipt"]["ok"],
                current_matches_local=current["receipt"].get("sha256")==localhash if current["receipt"]["ok"] else None))
        summary["calibrations"]=calibration;summaries.append(summary)
    for path,identity in original_files.items():
        p=PATHS.root/path
        assert p.stat().st_size==identity["bytes"] and digest(p)==identity["sha256"]
    assert ledger.read_bytes()==prefix
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope="Metadata and source-file eligibility only",
        groups=summaries,raw_CSI_candidates_in_project=raw_candidates,historical_receipts=receipts,
        originals_unchanged=original_files,retrievals={k:v["receipt"] for k,v in fetched.items()},
        no_training=True,no_model_inference=True,no_scoring=True,no_RT=True,no_large_download=True,
        raw_frame_CRC_or_complete_position_check=False,
        geometry="Current official metadata files and specs checked; no matching3D model/pointcloud/container geometry identified",
        reference="Receiver compensation does not establish independently known mobile-TX absolute phase/time")
    (REPORT/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    number=max(map(int,re.findall(r"^## (\d+)\.",prefix.decode(),re.M)))+1
    with ledger.open("a") as f:
        f.write(f"\n\n## {number}. 2026-10-29 L-shaped factory candidate eligibility audit\n\n- User accepts the L-shaped container-obstructed factory as the next candidate. This pass checks local availability,current official metadata/specs and calibration only; no training,prediction/scoring,RT or raw-data replacement.\n")
        f.write("- Local project has8 calibration JSONs but no complete cf-series CSI. Historical server_verified receipts concern small calibration files, not CSI; their publisher_checksum fields are null.\n")
        f.write("- First-day full32-port data is the preferred candidate. Day2 parts have24/21 ports, reindexed antennas and a slightly different frequency grid; they cannot be directly pooled or treated as independent blind confirmation.\n")
        f.write("- No matching precise3D geometry identified in checked official manifests/specs. No raw frames were available for full position/repetition/CRC/calibration application checks. No experiment-readiness claim. Evidence: analysis/audits/factory_lshape_20261029/audit.json and saved official metadata/specs.\n")
    print("AUDIT_COMPLETE",json.dumps(summaries),flush=True)
if __name__=="__main__":main()

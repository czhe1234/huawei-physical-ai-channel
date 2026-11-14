"""Upload locally verified data with native SSH, parallel resumable pieces.

No private key is read by this script. Native SSH uses the user's existing host
configuration. Local and remote original arrays are never silently overwritten.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block=f.read(4*1024**2)
            if not block:break
            value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--host',required=True)
    parser.add_argument('--remote-dir',required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    local=args.data_dir.resolve()
    verified=json.loads((local/'download_verified.json').read_text())
    source=local/verified['file']
    if source.stat().st_size!=verified['size'] or digest(source)!=verified['sha256']:
        raise RuntimeError('Local original data must pass identity check before uploading')
    started=time.time()
    remote=args.remote_dir.rstrip('/')

    def ssh(code,extra=()):
        command='python3 -c '+shlex.quote(code)+' '+ ' '.join(shlex.quote(str(x)) for x in extra)
        return subprocess.run(['ssh','-o','BatchMode=yes',args.host,command],check=True,text=True,capture_output=True)

    ssh('from pathlib import Path; import sys; Path(sys.argv[1]).mkdir(parents=True,exist_ok=True)',[remote+'/upload_parts'])
    bundle=local/'upload_parts';bundle.mkdir(exist_ok=True)
    chunk=512*1024**2
    pieces=[]
    with source.open('rb') as stream:
        for index,start in enumerate(range(0,verified['size'],chunk)):
            path=bundle/f'part{index:03d}.bin'
            size=min(chunk,verified['size']-start)
            if path.exists() and path.stat().st_size==size:
                stream.seek(size,1)
            else:
                tmp=path.with_suffix('.partial')
                left=size
                with tmp.open('wb') as out:
                    while left:
                        block=stream.read(min(left,4*1024**2))
                        if not block:raise RuntimeError('Truncated local source')
                        out.write(block);left-=len(block)
                os.replace(tmp,path)
            pieces.append(dict(file=path.name,start=start,size=size,sha256=digest(path)))
    manifest=dict(source=verified,pieces=pieces)
    (bundle/'upload_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

    def upload(piece):
        local_path=bundle/piece['file'];remote_path=remote+'/upload_parts/'+piece['file']
        check="from pathlib import Path; import hashlib,sys; p=Path(sys.argv[1]); print(hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else 'missing')"
        existing=ssh(check,[remote_path]).stdout.strip()
        if existing==piece['sha256']:return piece['size'],'resumed'
        error=None
        for attempt in range(3):
            try:
                subprocess.run(['scp','-q','-C','-o','BatchMode=yes',local_path.as_posix(),f'{args.host}:{remote_path}'],check=True,timeout=1800,capture_output=True,text=True)
                actual=ssh(check,[remote_path]).stdout.strip()
                if actual!=piece['sha256']:raise RuntimeError('Remote piece SHA256 mismatch')
                return piece['size'],'uploaded'
            except Exception as exc:
                error=str(exc);time.sleep(2**attempt)
        raise RuntimeError(f'Upload failed for {piece["file"]}: {error}')

    complete=0;total_bytes=0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(upload,p):p for p in pieces}
        for future in futures.as_completed(pending):
            n,state=future.result();complete+=1;total_bytes+=n
            print(json.dumps(dict(stage='upload',complete=complete,pieces=len(pieces),
                                  verified_bytes=total_bytes,elapsed_s=time.time()-started,state=state)),flush=True)
    subprocess.run(['scp','-q','-o','BatchMode=yes',(bundle/'upload_manifest.json').as_posix(),f'{args.host}:{remote}/upload_manifest.json'],check=True)
    assembly='''from pathlib import Path
import hashlib,json,os,sys,time
r=Path(sys.argv[1]);m=json.loads((r/'upload_manifest.json').read_text());s=m['source'];target=r/s['file']
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
if target.exists():
 if digest(target)!=s['sha256']:raise RuntimeError('Existing original data differs; refusing overwrite')
 print('REMOTE_ORIGINAL_ALREADY_VERIFIED');sys.exit(0)
tmp=target.with_suffix('.assembling');sha=hashlib.sha256();md5=hashlib.md5();size=0
with tmp.open('wb') as out:
 for piece in m['pieces']:
  p=r/'upload_parts'/piece['file'];ph=hashlib.sha256();n=0
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4*1024**2),b''):out.write(b);sha.update(b);md5.update(b);ph.update(b);n+=len(b);size+=len(b)
  if n!=piece['size'] or ph.hexdigest()!=piece['sha256']:raise RuntimeError('Server piece identity failed')
if size!=s['size'] or sha.hexdigest()!=s['sha256'] or md5.hexdigest()!=s['md5']:raise RuntimeError('Server original identity failed')
tmp.rename(target)
(r/'download_verified.json').write_text(json.dumps(s,indent=2))
(r/'upload_verified.json').write_text(json.dumps(dict(status='verified',file=target.name,size=size,sha256=sha.hexdigest(),md5=md5.hexdigest()),indent=2))
print(json.dumps(dict(status='REMOTE_ORIGINAL_VERIFIED',sha256=sha.hexdigest(),md5=md5.hexdigest(),size=size)))
'''
    result=ssh(assembly,[remote])
    print(result.stdout,flush=True)
    # Context files are small; retain their exact directory layout and licenses.
    for name in ['spec.json','source_manifest.json']:
        subprocess.run(['scp','-q','-o','BatchMode=yes',(local/name).as_posix(),f'{args.host}:{remote}/{name}'],check=True)
    subprocess.run(['scp','-q','-r','-o','BatchMode=yes',(local/'context').as_posix(),f'{args.host}:{remote}/'],check=True)
    result=dict(status='complete',file=source.name,sha256=verified['sha256'],bytes=verified['size'],
                elapsed_s=time.time()-started,workers=args.workers)
    (local/'upload_verified.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

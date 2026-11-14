"""Finalize the frozen live observer after both24-seed sources complete."""
from pathlib import Path
import argparse
import json
from v16_live_tokens import create_observer
from v16_recovery import PATHS,sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--statistics",type=Path,required=True)
    a=p.parse_args();path=PATHS.resolve(a.statistics)
    observer=create_observer(path.parent,"base")
    result=observer.finalize()
    print(json.dumps(dict(status="complete",observer_identity=result["observer_identity"],
        token_manifest=PATHS.relative(observer.root/"tokens/manifest.json"),
        array_count=len(result["arrays"]),query_count=len(result["query_ids"]),
        storage_mode=result["storage_mode"])))

if __name__=="__main__":main()

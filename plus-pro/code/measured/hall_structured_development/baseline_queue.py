"""Two bounded spectral MLP comparisons on one allowed device, sequentially."""
import subprocess,sys,json
from .common import REPORT,save_json
def main():
    records=[]
    for weight in [1,4]:
        name="spectral_mlp_w"+str(weight)
        cmd=[sys.executable,"-u","-m","measured.hall_structured_development.runner","--name",name,
            "--kind","mlp","--frequency-weight",str(weight),"--levels","3","--hidden","512",
            "--lr",".001","--batch","32","--epochs","320","--patience","25"]
        with (REPORT/(name+".log")).open("w") as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(name=name,command=cmd,returncode=result.returncode))
        save_json(REPORT/"baseline_queue.json",dict(records=records,complete=len(records)==2 and not result.returncode))
        if result.returncode:raise RuntimeError(name+" failed; see its log")
if __name__=="__main__":main()

"""Sequential jobs only, with separate logs and bounded optional compilation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'


def main():
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
             CCACHE_DISABLE='1',MAX_JOBS='1',TORCHINDUCTOR_COMPILE_THREADS='1',
             TORCHINDUCTOR_CACHE_DIR=str(OUT/'inductor_cache_codex'))
    jobs=[('nudt_components',['run_speed_codex.py','nudt','--components'],600),
          ('nudt_pipeline',['run_speed_codex.py','nudt'],1800),
          ('native_benchmark',['native_benchmark_codex.py'],120),
          ('judge_benchmark',['judge_benchmark_codex.py'],120),
          ('compile_benchmark',['compile_benchmark_codex.py'],180)]
    status=[]
    for name,args,timeout in jobs:
        print('START',name,flush=True);start=time.time()
        try:
            with (OUT/(name+'_codex.log')).open('w') as log:
                result=subprocess.run([sys.executable,'-u',str(HERE/args[0])]+args[1:],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            record=dict(job=name,exit_code=result.returncode,seconds=time.time()-start)
        except subprocess.TimeoutExpired:
            record=dict(job=name,exit_code='timeout',seconds=time.time()-start)
        status.append(record);(OUT/'jobs_codex.json').write_text(json.dumps(status,indent=2))
        if name=='compile_benchmark':(OUT/'compile_status_codex.json').write_text(json.dumps(record,indent=2))
        subprocess.run([sys.executable,str(HERE/'summarise_codex.py')],cwd=ROOT,env=env,check=True)
        print('DONE',record,flush=True)


if __name__=='__main__':main()

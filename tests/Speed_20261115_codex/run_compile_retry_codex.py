"""Avoid the installed Inductor C++ builder's unquoted paths-with-spaces issue.

Only temporary symlink aliases are added. Existing packages/source stay unchanged;
compiled cache artifacts still live in the experiment's results directory.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'


def main():
    alias=Path(tempfile.mkdtemp(prefix='geoloc_speed_compile_codex_'))
    cache=OUT/'inductor_cache_codex';cache.mkdir(exist_ok=True)
    os.symlink(str(ROOT.parent/'venv-geoloc'),alias/'venv')
    os.symlink(str(cache),alias/'cache')
    paths=dict(temp_alias=str(alias),python=str(alias/'venv/bin/python'),cache=str(alias/'cache'))
    (OUT/'compile_retry_paths_codex.json').write_text(json.dumps(paths,indent=2))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
             CCACHE_DISABLE='1',MAX_JOBS='1',TORCHINDUCTOR_COMPILE_THREADS='1',TORCHINDUCTOR_CACHE_DIR=paths['cache'])
    start=time.time()
    try:
        with (OUT/'compile_retry_codex.log').open('w') as log:
            result=subprocess.run([paths['python'],'-u',str(HERE/'compile_benchmark_codex.py')],cwd=ROOT,env=env,
                                  stdout=log,stderr=subprocess.STDOUT,timeout=180)
        status=dict(job='compile_retry',exit_code=result.returncode,seconds=time.time()-start,
                    workaround='Temporary no-space aliases for venv and experiment-owned compiler cache; source and packages unchanged.')
    except subprocess.TimeoutExpired:status=dict(job='compile_retry',exit_code='timeout',seconds=time.time()-start)
    (OUT/'compile_status_codex.json').write_text(json.dumps(status,indent=2));print(status,flush=True)


if __name__=='__main__':main()

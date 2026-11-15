"""Compile a tiny optional C++ kernel and compare with current NumPy implementation."""
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from train.checks.decision_signals import translation_consensus
from train.checks.rich_dump import load_rich


def main():
    command=['g++','-O3','-std=c++17','-shared','-fPIC',str(HERE/'translation_codex.cpp'),'-o',str(OUT/'translation_codex.so')]
    subprocess.run(command,check=True,capture_output=True)
    lib=ctypes.CDLL(str(OUT/'translation_codex.so'));fn=lib.translation_consensus_codex
    ptr=ctypes.POINTER(ctypes.c_double)
    fn.argtypes=[ptr,ctypes.c_int,ctypes.c_double,ctypes.POINTER(ctypes.c_uint8),ptr];fn.restype=ctypes.c_int
    def native(a,b,radius):
        d=np.ascontiguousarray((b-a).astype(np.float64))
        mask=np.zeros(len(a),np.uint8);centre=np.zeros(2,np.float64)
        count=fn(d.ctypes.data_as(ptr),len(d),radius,mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),centre.ctypes.data_as(ptr))
        return count,mask.astype(bool),centre
    rng=np.random.default_rng(151126);cases=[]
    for n in [0,1,2,4,10,50,200,500]:
        a=rng.random((n,2),dtype=np.float32)*200
        b=a+np.concatenate((rng.normal(0,1,(n//2,2)),rng.normal(30,30,(n-n//2,2)))).astype(np.float32)
        cases.append((f'synthetic_{n}',a,b))
    # Exact radius boundaries, repeated displacements and equal support ties.
    for radius in [2.,3.]:
        b=np.array([[0,0],[radius,0],[-radius,0],[0,radius],[0,-radius],[20,20]],np.float64)
        cases.append((f'boundary_{radius}',np.zeros_like(b),b))
    for path in sorted((ROOT/'tests/Test_Iteration_7/results/rich_dump/official150').glob('photo_*.npz'))[:5]:
        data=load_rich(path)
        for q in data['queries'][:2]:
            for match in q['matches']:
                if match is not None:
                    keep=match['confidence']>=.22
                    cases.append((path.name,match['points_a'][keep],match['points_b'][keep]))
    rows=[];mismatches=[]
    for name,a,b in cases:
        for radius in [2.,3.]:
            expected=translation_consensus(a,b,radius);actual=native(a,b,radius)
            same=expected[0]==actual[0] and np.array_equal(expected[1],actual[1]) and np.allclose(expected[2],actual[2],rtol=0,atol=1e-12)
            if not same:mismatches.append(dict(name=name,n=len(a),radius=radius))
            timings={}
            for label,method in [('numpy',translation_consensus),('cpp',native)]:
                values=[]
                for _ in range(7):
                    start=time.perf_counter();method(a,b,radius);values.append(time.perf_counter()-start)
                timings[label]=float(np.median(values))
            rows.append(dict(name=name,n=len(a),radius=radius,same=bool(same),**timings))
    result=dict(command=command,cases=len(rows),mismatches=mismatches,measurements=rows,
                caution='CPU kernel-only timings, ctypes overhead included. No full-pipeline speed claim or production replacement.')
    (OUT/'native_benchmark_codex.json').write_text(json.dumps(result,indent=2))
    print('Native differential cases',len(rows),'mismatches',len(mismatches),flush=True)
    for n in [50,200,500]:
        r=next(r for r in rows if r['name']==f'synthetic_{n}' and r['radius']==3)
        print('N',n,'NumPy',r['numpy'],'C++',r['cpp'],'speedup',r['numpy']/r['cpp'],flush=True)


if __name__=='__main__':main()

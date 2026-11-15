"""Summarise saved CUDA stages and profile CPU signal extraction; no source changes."""
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'results'
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    path=ROOT/'tests/Stage_Evaluation/time_per_query_official_edge_zncc_headings9_scales3_K50_M10.csv'
    table=pd.read_csv(path)
    # Match the notebook's warm-up exclusion, not its extrapolated K/M cost table.
    steady=table[table.photo!=table.photo.min()]
    columns=['render','maps','coarse','semantic','fine','total']
    means=steady[columns].mean()
    shares=means/means['total']
    scenarios={}
    for name,scales in [('fine_2x',{'fine':2}),('fine_4x',{'fine':4}),
                        ('fine_and_semantic_2x',{'fine':2,'semantic':2})]:
        seconds=sum(means[c]/scales.get(c,1) for c in columns if c!='total')
        scenarios[name]=dict(seconds_per_query=seconds,overall_speedup=means['total']/seconds)
    from train.checks.rich_dump import load_rich
    from train.checks.decision_signals import photo_signal_table
    files=sorted((ROOT/'tests/Test_Iteration_7/results/rich_dump/official150').glob('photo_*.npz'))[:5]
    profiles=[]
    profiler=cProfile.Profile()
    for f in files:
        data=load_rich(f)
        data['queries']=data['queries'][:6]
        photo_signal_table(data,0,'profile',with_lo=False) # warm-up
        measurements=[]
        for _ in range(5):
            t=time.perf_counter()
            rows=photo_signal_table(data,0,'profile',with_lo=False)
            measurements.append(time.perf_counter()-t)
        profiles.append(dict(file=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),queries=6,candidates=len(rows),
                             median_seconds=float(np.median(measurements)),max_seconds=max(measurements)))
        profiler.runcall(photo_signal_table,data,0,'profile',with_lo=False)
    stream=io.StringIO()
    pstats.Stats(profiler,stream=stream).sort_stats('cumtime').print_stats(20)
    (OUT/'signal_cpu_profile_codex.txt').write_text(stream.getvalue())
    results=dict(timing_source=str(path),timing_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 all_queries=len(table),warmup_excluded_queries=len(steady),photos=len(steady.photo.unique()),
                 means_seconds=means.to_dict(),shares=shares.to_dict(),hypothetical_amdahl=scenarios,
                 cpu_signal_profiles=profiles,cuda_available=torch.cuda.is_available(),
                 caveat='Historical CUDA stages, not current board timings; CPU profiles use old raw matches without inference/pandas/model cost.')
    (OUT/'speed_analysis_codex.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()

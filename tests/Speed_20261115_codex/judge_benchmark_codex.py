"""Existing frozen judge, offline live-query bookkeeping parity/speed; no fitting."""
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from train.checks.candidate_judge import add_context,add_features,judge_new
from train.checks.decision_signals import photo_signal_table
from train.checks.rich_dump import load_rich


def original(model,blocks):
    prefix=[];predictions=[]
    for block in blocks:
        prefix.append(block)
        t=add_context(add_features(pd.concat(prefix,ignore_index=True)))
        predictions.append(judge_new(model,t))
    return predictions


def incremental(model,blocks):
    parts=[];predictions=[]
    for block in blocks:
        t=add_context(add_features(block.reset_index(drop=True)))
        parts.append(judge_new(model,t))
        predictions.append(np.concatenate(parts))
    return predictions


def main():
    model=joblib.load(ROOT/'tests/Test_Iteration_7/results/judge_t7.joblib')
    rows=[]
    for path in sorted((ROOT/'tests/Test_Iteration_7/results/rich_dump/official150').glob('photo_*.npz'))[:5]:
        photo=load_rich(path);photo['queries']=photo['queries'][:6]
        t=pd.DataFrame(photo_signal_table(photo,0,'live',with_lo=False))
        blocks=[g.copy() for _,g in t.groupby('query',sort=False)]
        a=original(model,blocks);b=incremental(model,blocks)
        exact=all(np.array_equal(x,y) for x,y in zip(a,b))
        times={}
        for label,fn in [('original',original),('incremental',incremental)]:
            measured=[]
            for _ in range(5):
                start=time.perf_counter();fn(model,blocks);measured.append(time.perf_counter()-start)
            times[label]=float(np.median(measured))
        rows.append(dict(photo=path.name,prefixes=len(blocks),probabilities_exact=exact,**times))
    result=dict(rows=rows,all_exact=all(r['probabilities_exact'] for r in rows),
                warning='Frozen thermal-trained T7 model used as historical baseline only. No fitting. Raw signal extraction and neural inference excluded; compares repeated live-prefix Pandas/prediction overhead.')
    (OUT/'judge_benchmark_codex.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

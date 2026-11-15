"""Small CPU reproductions against current source; no data/model fitting or source patches."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from train.checks import candidate_judge as cj
from train.checks import scale_robust as sr
from train.checks.stage_cache import CachedStagePipeline, share_map_fft, unshare_map_fft


def cache_key_collision():
    p = CachedStagePipeline.__new__(CachedStagePipeline)
    a = dict(name='/flight_a/frame.png', rel_height=100, yaw=0)
    b = dict(name='/flight_b/frame.png', rel_height=200, yaw=90)
    key_a = p._key(a, (100,200), 4, None)
    key_b = p._key(b, (100,200), 4, None)
    assert key_a == key_b
    return dict(reproduced=True, key=key_a, inputs=[a,b],
                consequence='Different physical image and pose resolve to the same persistent cache entry.')


def fft_collision():
    import coarse_search.correlation as corr
    torch.manual_seed(15)
    a = torch.rand(8,32,32)
    # At numel=8192 the fingerprint samples every second element. Swap two
    # unsampled values, preserving both sampled and complete sums exactly.
    a.flatten()[1],a.flatten()[3]=.25,.75
    b=a.clone()
    b.flatten()[1],b.flatten()[3]=.75,.25
    kernel=torch.rand(8,4,4)
    cells=torch.ones(4,4)
    expected_a=corr.correlate_everywhere(a,kernel,cells)
    expected_b=corr.correlate_everywhere(b,kernel,cells)
    share_map_fft()
    try:
        actual_a=corr.correlate_everywhere(a,kernel,cells)
        actual_b=corr.correlate_everywhere(b,kernel,cells)
    finally:
        unshare_map_fft()
    initial=float((expected_a['zncc']-actual_a['zncc']).abs().max())
    corrupted=float((expected_b['zncc']-actual_b['zncc']).abs().max())
    assert initial < 1e-5 and corrupted > 1e-3
    return dict(reproduced=True, initial_max_error=initial, changed_window_max_error=corrupted,
                consequence='Distinct equal-sum windows reuse stale FFTs and return incorrect score maps.')


def relative_unavailable_queries():
    # The first query is identical. Only a later, not-yet-executed query changes.
    rows=[dict(data='test',photo=0,query=0,rank=0,inliers=10.,n22=20.,z=2.,coarse=.1,x=0.,y=0.,sigma=8.),
          dict(data='test',photo=0,query=0,rank=1,inliers=1.,n22=2.,z=0.,coarse=.1,x=500.,y=0.,sigma=8.),
          dict(data='test',photo=0,query=26,rank=0,inliers=1.,n22=2.,z=0.,coarse=.1,x=1000.,y=0.,sigma=8.)]
    a=pd.DataFrame(rows)
    b=a.copy()
    b.loc[b['query']==26,['inliers','n22']]=[6.,12.]
    prior=dict(far_inl=.2,far_n22=2.)
    aa=sr.add_relative(a,prior,m=5)
    bb=sr.add_relative(b,prior,m=5)
    va,vb=float(aa.loc[0,'rli']),float(bb.loc[0,'rli'])
    assert va!=vb
    return dict(reproduced=True, first_query_feature_before=va, first_query_feature_after=vb,
                consequence='A feature for an early decision depends on fine matches of an unexecuted query.')


def unsafe_zero_wrong_fallback():
    scan=pd.DataFrame(dict(tau=[.9,.99],data=['d','d'],wrong=[1,1],n_answered=[10,10]))
    threshold=cj.smallest_safe_tau(scan,['d'])
    assert threshold==.99
    return dict(reproduced=True, returned_threshold=threshold,wrong_at_returned_threshold=1,
                consequence='The zero-wrong selector returns an unsafe finite threshold when no threshold meets its condition.')


def early_exit_bypasses_main_cut():
    h,s=cj.STANDARD[0]
    t=pd.DataFrame([dict(data='d',photo=0,rank=0,inliers=10.,z=2.,coarse=.1,heading=h,scale=s,
                         n22=15.,inl_t3=9.,err_sigma=20.,x=0.,y=0.)])
    t=cj.add_features(t)
    out=cj.decide(t,np.array([.985]),.9995,early_tau=.98,override=None)
    assert out.iloc[0]['stage']=='early' and out.iloc[0]['pts']==-20
    return dict(reproduced=True, policy_tau=.9995,early_tau=.98,accepted_probability=.985,
                consequence='API permits early acceptance below the main calibrated cut; T11 final caller guards this with max().')


def augmented_context_reintroduces_unavailable_candidates():
    t=pd.DataFrame([dict(data='d',photo=0,query=0,rank=rank,inliers=10.,n22=15.,z=z,coarse=.1,
                         heading=-5.,scale=1.,inl_t3=8.,err_sigma=.5,x=x,y=0.,sigma=8.)
                    for rank,z,x in [(0,2.5,0.),(7,2.4,500.)]])
    base=cj.add_context(cj.add_features(t),m=5)
    augmented=sr.scale_signals(base,.5) # exact call in fit_judge; m is omitted
    assert base.loc[0,'zgap']>5 and abs(augmented.loc[0,'zgap']-.1)<1e-8
    return dict(reproduced=True, matched_only_zgap=float(base.loc[0,'zgap']),
                augmented_zgap=float(augmented.loc[0,'zgap']),
                consequence='Jetson scale-augmented training copies rebuild context with ranks >=5 despite m=5 base features.')


def main():
    torch.set_num_threads(2)
    output={}
    for fn in [cache_key_collision,fft_collision,relative_unavailable_queries,unsafe_zero_wrong_fallback,early_exit_bypasses_main_cut,
               augmented_context_reintroduces_unavailable_candidates]:
        output[fn.__name__]=fn()
        print(fn.__name__,output[fn.__name__],flush=True)
    (HERE/'results/reproductions_codex.json').write_text(json.dumps(output,indent=2))


if __name__=='__main__':
    main()

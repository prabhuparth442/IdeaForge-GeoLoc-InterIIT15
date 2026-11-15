"""Bounded subprocess benchmark of actual LoFTR CNN with CPU TorchInductor."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import cv2

HERE=Path(__file__).resolve().parent;OUT=HERE/'results';ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from run_speed_codex import load_domain,item_of,WORKSPACE
from image_processing.geometry import render_query
from image_processing.map_levels import level_for_height,map_piece
from fine_matching.loftr_matcher import load_loftr_matcher,candidate_matches,stage2_from_matches


def main():
    torch.set_num_threads(2)
    samples,store,camera=load_domain('official');item=item_of(samples[0]);level=level_for_height(item['rel_height'],camera)
    canvas,valid=render_query(item,camera,level)
    prior_results=ROOT/'tests/Satellite_Only_20261114_codex/results/official_pipeline_codex.jsonl'
    row=json.loads(prior_results.read_text().splitlines()[0]);candidate=row['baseline']
    tile=map_piece(store['levels'],level,candidate['x'],candidate['y'])
    x=torch.from_numpy(np.stack([canvas,tile]).astype(np.float32))[:,None]
    matcher=load_loftr_matcher(WORKSPACE/'Submission_Files/myteam/models/minima_loftr.ckpt',.21,coarse_threshold=.21)
    original=matcher.loftr_model.backbone
    def timed(fn,n=5):
        ts=[];value=None
        for _ in range(n):
            start=time.perf_counter();value=fn();ts.append(time.perf_counter()-start)
        return value,dict(median=float(np.median(ts)),samples=ts)
    with torch.no_grad():
        original(x)
        a,eager=timed(lambda:original(x))
        print('Eager backbone',eager,flush=True)
        start=time.perf_counter()
        compiled=torch.compile(original,backend='inductor',mode='default',dynamic=False)
        b=compiled(x)
        cold=time.perf_counter()-start
        b,warm=timed(lambda:compiled(x))
    peaks=np.array([[.5,candidate['x'],candidate['y'],0.,0.]])
    args=(matcher,canvas,valid,level,peaks,store['levels'])
    raw_a,match_eager=timed(lambda:candidate_matches(*args),n=3)
    matcher.loftr_model.backbone=compiled
    try:raw_b,match_compiled=timed(lambda:candidate_matches(*args),n=3)
    finally:matcher.loftr_model.backbone=original
    aa,bb=raw_a['items'][0],raw_b['items'][0]
    diffs={}
    for key in ['points_a','points_b','confidence']:
        diffs[key]=dict(shape_same=aa[key].shape==bb[key].shape,
                        max_abs=float(np.max(np.abs(aa[key]-bb[key]),initial=0)) if aa[key].shape==bb[key].shape else None)
    cv2.setRNGSeed(7);fit_a=stage2_from_matches(raw_a,level,.21)
    cv2.setRNGSeed(7);fit_b=stage2_from_matches(raw_b,level,.21)
    full_probe=dict(eager=match_eager,compiled=match_compiled,match_counts=[len(aa['confidence']),len(bb['confidence'])],differences=diffs,
                    inliers_same=bool(np.array_equal(fit_a['inliers'],fit_b['inliers'])),
                    shifts_same=bool(np.array_equal(fit_a['shifts'],fit_b['shifts'])),
                    max_shift_delta_m=float(np.max(np.abs(fit_a['shifts']-fit_b['shifts']))))
    result=dict(backend='CPU TorchInductor',device='cpu',batch=2,input_shape=list(x.shape),eager=eager,compiled=warm,cold_compile_seconds=cold,
                feature_max_abs=[float((u-v).abs().max()) for u,v in zip(a,b)],
                feature_allclose=[bool(torch.allclose(u,v,rtol=1e-4,atol=1e-5)) for u,v in zip(a,b)],
                full_match_probe=full_probe,
                warning='Backbone benchmark plus one complete matching-pair probe; not multi-photo deployment validation. Compiled variant excluded from main comparison.')
    (OUT/'compile_benchmark_codex.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

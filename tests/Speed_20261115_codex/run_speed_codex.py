"""Real neural component benchmarks followed by paired image-to-answer research runs."""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WORKSPACE=ROOT.parent
OUT=HERE/'results'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(HERE)]
from optimisations_codex import SpeedAgreementPipeline,batched_dino_scores,cached_photo,cached_query_backbone
from agreement_query_pipeline import AgreementQueryPipeline
from header import settings
from image_processing.geometry import render_query
from image_processing.map_levels import level_for_height
from fine_matching.loftr_matcher import load_loftr_matcher,candidate_matches,stage2_from_matches
from semantic_rerank import dino_rerank as dino
from coarse_search.dino_hint import load_encoder
from train.coarse_methods import build_methods
from train.features.edge import EdgeFamily


def timed(fn,n=3):
    samples=[]
    value=None
    for _ in range(n):
        start=time.perf_counter();value=fn();samples.append(time.perf_counter()-start)
    return value,dict(median=float(np.median(samples)),samples=samples)


def load_domain(domain):
    m=json.loads((ROOT/f'tests/Satellite_Only_20261114_codex/results/{domain}_evaluation_manifest_codex.json').read_text())
    path=WORKSPACE/'Submission_Files/myteam/preprocess' if domain=='official' else ROOT/'tests/Satellite_Only_20261114_codex/results/nudt_store_codex'
    info=json.loads((path/'store.json').read_text())
    store={name:{level:np.load(path/f'{prefix}_{level}.npy',mmap_mode='r') for level in info['levels']}
           for name,prefix in [('levels','level'),('features','features')]}
    settings.LEVELS=info['levels']
    settings.GEOMETRY.update(attitude=domain=='official',yaw_offset=180. if domain=='official' else 0.,scale=1.04 if domain=='official' else .8,height_offset=0.)
    samples=m['samples'][:3]
    meta=samples[0]['meta']
    camera=SimpleNamespace(**{key:float(meta['cam_'+key]) for key in ['fx','fy','cx','cy','k1','k2','k3','p1','p2']},
                           width=320 if domain=='official' else 1280,height=256 if domain=='official' else 1024)
    return samples,store,camera


def item_of(sample):
    m=sample['meta']
    return dict(name=sample['path'],rel_height=float(m['alt']),yaw=float(m['yaw']),pitch=(float(m['pitch'])+180)%360-180,
                roll=(float(m['roll'])+180)%360-180)


def pipeline(kind,domain,store,camera,matcher,encoder):
    cls=AgreementQueryPipeline if kind=='baseline' else SpeedAgreementPipeline
    p=cls(domain,coarse='edge_zncc',matcher='loftr_minima',k_peaks=50,m_fine=10,min_inliers=8,
          matcher_confidence=.21,loftr_coarse_threshold=.21,dino_weight=100,dino_batch=5,
          **(dict(query_backbone=kind=='backbone') if kind!='baseline' else {}))
    p.store=store;p.level_images=store['levels'];p.camera=camera
    p.names=['edge_zncc'];p.coarse_name='edge_zncc';p.coarse_method=build_methods(['edge_zncc'],fused_models={})['edge_zncc']
    p.families={'edge':EdgeFamily(store)};p.matcher=matcher;p.encoder=encoder;p.ready=True
    return p


def serialise(result,truth,sigma):
    error=float(np.hypot(result['x']-truth[0],result['y']-truth[1]))
    ratio=error/sigma
    points=(10 if ratio<=1 else 6 if ratio<=2 else 3 if ratio<=4 else 0 if ratio<=8 else -20) if result['answered'] else 0
    keys=['x','y','answered','inliers','stage','n_queries_run','total','seconds','agreement','escalated','level']
    out={k:result[k] for k in keys}
    out.update(error_m=error,points=points,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    return out


def compare(a,b):
    comparisons=[]
    for qa,qb in zip(a['queries'],b['queries']):
        comparisons.append(dict(peaks_same=bool(np.array_equal(qa['peaks'],qb['peaks'])),
                                max_peak_abs=float(np.max(np.abs(qa['peaks']-qb['peaks']))),
                                z_max_abs=float(np.max(np.abs(qa['z']-qb['z']))),
                                inliers_same=bool(np.array_equal(qa['stage2']['inliers'],qb['stage2']['inliers'])),
                                shifts_same=bool(np.array_equal(qa['stage2']['shifts'],qb['stage2']['shifts']))))
    same=all(a[k]==b[k] for k in ['answered','x','y','inliers','stage','n_queries_run'])
    return dict(final_same=bool(same),same_query_count=len(a['queries'])==len(b['queries']),queries=comparisons)


def components(domain,sample,store,camera,matcher,encoder):
    item=item_of(sample);level=level_for_height(item['rel_height'],camera)
    geometries=[dict(yaw_offset=settings.GEOMETRY['yaw_offset']+h,scale=settings.GEOMETRY['scale']*s)
                for s in [1.,1.05] for h in [-5.,0.,5.]]
    def render():return [render_query(item,camera,level,g) for g in geometries]
    original,rt=timed(render)
    def prepared_render():
        with cached_photo(item):return render()
    prepared,ct=timed(prepared_render)
    np.testing.assert_array_equal(np.stack([x[0] for x in original]),np.stack([x[0] for x in prepared]))
    np.testing.assert_array_equal(np.stack([x[1] for x in original]),np.stack([x[1] for x in prepared]))
    canvas,valid=original[2]
    base=pipeline('baseline',domain,store,camera,matcher,encoder)
    found=base.prepare_query(item,sample['prior_xy'],level,geometries[2])
    peaks=found['shortlist']
    # Real K50 DINO, same model, two transfer/reduction implementations.
    args=(encoder,canvas,valid,store['levels'],level,peaks,5)
    a,dt=timed(lambda:dino.dino_candidate_scores(*args))
    b,vt=timed(lambda:batched_dino_scores(*args))
    dino_error=max(float(np.nanmax(np.abs(x-y))) for x,y in zip(a,b))
    ranking_same=np.array_equal(dino.rerank_peaks(peaks,a[1],keep=10),dino.rerank_peaks(peaks,b[1],keep=10))
    fine_peaks=found['peaks']
    args=(matcher,canvas,valid,level,fine_peaks,store['levels'])
    baseline,ft=timed(lambda:candidate_matches(*args))
    def cached():
        with cached_query_backbone(matcher.loftr_model) as wrapper:
            value=candidate_matches(*args)
            return value,wrapper.encoded_images
    (cached_matches,encodings),qt=timed(cached)
    def raw_diff(other):
        exact=True;counts=True;maximum=0.
        for x,y in zip(baseline['items'],other['items']):
            if x is None or y is None:
                exact &= x is y;counts &= x is y;continue
            for key in ['points_a','points_b','confidence']:
                counts &= x[key].shape==y[key].shape
                if x[key].shape==y[key].shape:
                    exact &= np.array_equal(x[key],y[key]);maximum=max(maximum,float(np.max(np.abs(x[key]-y[key]),initial=0)))
                else:exact=False
        cv2.setRNGSeed(settings.SEED);s1=stage2_from_matches(baseline,level,.21)
        cv2.setRNGSeed(settings.SEED);s2=stage2_from_matches(other,level,.21)
        return dict(raw_exact=bool(exact),raw_shapes_same=bool(counts),raw_max_abs=maximum,
                    inliers_same=bool(np.array_equal(s1['inliers'],s2['inliers'])),shifts_same=bool(np.array_equal(s1['shifts'],s2['shifts'])))
    batches={}
    for batch in [2,4]:
        matcher.batch_size=batch
        value,t=timed(lambda:candidate_matches(*args),n=2)
        batches[str(batch)]=dict(timing=t,parity=raw_diff(value))
    matcher.batch_size=1
    result=dict(domain=domain,photo=sample['id'],render=dict(original=rt,cached=ct,exact=True),
                dino=dict(original=dt,vectorised=vt,max_abs=dino_error,rank_exact=bool(ranking_same)),
                loftr=dict(original=ft,cached=qt,cached_encoded_images=encodings,parity=raw_diff(cached_matches),batches=batches))
    (OUT/f'{domain}_components_codex.json').write_text(json.dumps(result,indent=2))
    print('COMPONENTS',json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('domain',choices=['official','nudt']);parser.add_argument('--components',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    samples,store,camera=load_domain(args.domain)
    weights=WORKSPACE/'Submission_Files/myteam/models'
    matcher=load_loftr_matcher(weights/'minima_loftr.ckpt',.21,coarse_threshold=.21)
    encoder=load_encoder(weights/'dinov2_vits14_timm.safetensors')
    if args.components:
        components(args.domain,samples[0],store,camera,matcher,encoder);return
    component=json.loads((OUT/f'{args.domain}_components_codex.json').read_text())
    # Exclude query-backbone reuse from the primary combined adapter until its
    # raw/fitted parity has been evaluated. Test it as a separate ablation.
    allow_backbone=component['loftr']['parity']['inliers_same'] and component['loftr']['parity']['shifts_same']
    kinds=['baseline','optimised']+(['backbone'] if allow_backbone else [])
    ps={kind:pipeline(kind,args.domain,store,camera,matcher,encoder) for kind in kinds}
    manifest=dict(domain=args.domain,samples=samples,kinds=kinds,geometry=dict(settings.GEOMETRY),levels=list(settings.LEVELS),
                  k=50,m=10,confidence=.21,dino_batch=5,device=settings.DEVICE,threads=2,queries='6/agreement then27/bar11',
                  dependencies={n:bool(importlib.util.find_spec(n)) for n in ['tensorrt','onnx','onnxruntime','numba','torch_tensorrt']})
    (OUT/f'{args.domain}_manifest_codex.json').write_text(json.dumps(manifest,indent=2))
    output=OUT/f'{args.domain}_full_pipeline_codex.jsonl';assert not output.exists()
    with output.open('w') as f:
        for index,sample in enumerate(samples):
            results={};records={}
            # Alternate baseline/primary order to reduce consistent warm-order bias.
            order=kinds if index%2==0 else list(reversed(kinds))
            for kind in order:
                print('START',args.domain,sample['id'],kind,flush=True)
                r=ps[kind].run(item_of(sample),sample['prior_xy'])
                results[kind]=r
                from header.scoring import sigma_m
                sigma=sigma_m(float(sample['meta']['alt']),camera.fx)
                records[kind]=serialise(r,sample['truth_xy'],sigma)
                (OUT/f'{args.domain}_progress_codex.json').write_text(json.dumps(dict(photo=sample['id'],finished_variants=records,planned_frames=3),indent=2))
                print('DONE',args.domain,sample['id'],kind,records[kind]['total'],records[kind]['points'],flush=True)
            comparisons={kind:compare(results['baseline'],results[kind]) for kind in kinds if kind!='baseline'}
            row=dict(domain=args.domain,photo=sample['id'],outputs=records,comparisons=comparisons)
            f.write(json.dumps(row)+'\n');f.flush()


if __name__=='__main__':main()

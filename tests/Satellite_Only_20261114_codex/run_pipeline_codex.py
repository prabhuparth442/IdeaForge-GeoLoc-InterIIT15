"""Actual image-to-answer paired decision pilot; no candidate replay inputs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

from encoder_codex import StructuralEncoder, encode, gate

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
WORKSPACE=HERE.parents[2]
OUT=HERE/'results'
sys.path.insert(0,str(WORKSPACE/'Main_Pipeline_Headless'))
import settings
from geoloc import decision, photo
from geoloc.levels import map_piece, level_gsd
from geoloc.pipeline import Localiser
from geoloc.map_store import MapStore


def verify_frozen():
    manifest=json.loads((OUT/'FROZEN_TRAINING_MANIFEST_codex.json').read_text())
    for name,sha in manifest['files'].items():
        p=HERE/name if (HERE/name).exists() else OUT/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha, 'Frozen file changed: '+str(p)
    return manifest


class InstrumentedLocaliser(Localiser):
    def __init__(self,store,models):
        super().__init__(store,models)
        self.satellite=StructuralEncoder()
        self.satellite.load_state_dict(torch.load(OUT/'encoder_satellite_codex.pt',weights_only=True))
        self.satellite.eval()
        self.captured=[]
        self.encoder_seconds=0.0

    def run_query(self,picture,pose,camera,level,prior,query,seconds,cache):
        result=super().run_query(picture,pose,camera,level,prior,query,seconds,cache)
        start=time.perf_counter()
        canvas,valid=photo.draw_canvas(picture,pose,camera,level,query['heading'],query['scale_factor'])
        windows=[map_piece(self.store.levels[level],level,p[1],p[2]) for p in result['peaks']]
        embeddings=encode(self.satellite,[canvas]+windows,[valid]*(1+len(windows)))
        result['satellite_cosine']=embeddings[1:]@embeddings[0]
        self.encoder_seconds+=time.perf_counter()-start
        self.captured.append(result)
        return result


def sigma_m(alt,fx):
    return math.sqrt(64+(3*max(alt,100)/fx)**2)


def select(results,sigma,cal):
    if not results:
        return {'sat_veto':dict(answered=False), 'sat_select':dict(answered=False)},[]
    peaks,n,shifts,z,owner=decision.pool_queries(results)
    xy=peaks[:,1:3]+shifts
    score=np.concatenate([r['satellite_cosine'] for r in results])
    _,_,_,best,_=decision.choose_candidate(peaks,n,shifts,z)
    query_bests=[]
    for r in results:
        x,y,count,_,_=decision.choose_candidate(r['peaks'],r['stage2']['inliers'],r['stage2']['shifts'],r['z'])
        if count>=3:
            query_bests.append((x,y))
    records=[]
    for i in range(len(peaks)):
        real=(peaks[:,1]<1e8)&np.isfinite(score)
        other=real & (np.linalg.norm(xy-xy[i],axis=1)>sigma)
        margin=float(score[i]-np.max(score[other])) if other.any() else float('-inf')
        agree=sum(math.hypot(x-xy[i,0],y-xy[i,1])<=sigma for x,y in query_bests)
        passed=bool(real[i] and gate(score[i],margin,cal,other.any()))
        records.append(dict(x=float(xy[i,0]),y=float(xy[i,1]),inliers=int(n[i]),z=float(z[i]),query=int(owner[i]),
                            cosine=float(score[i]),margin=margin if np.isfinite(margin) else None,agreement=agree,
                            gate=passed,eligible=bool(n[i]>=8 or (n[i]>=4 and agree>=2))))
    veto=dict(records[best],answered=bool(n[best]>=8 and records[best]['gate']))
    eligible=[i for i,r in enumerate(records) if r['gate'] and r['eligible']]
    if eligible:
        winner=sorted(eligible,key=lambda i:(-(n[i]+settings.GAMMA*z[i]),-peaks[i,0]))[0]
        new=dict(records[winner],answered=True)
    else:
        new=dict(answered=False)
    return {'sat_veto':veto,'sat_select':new},records


def inputs(domain):
    rng=np.random.default_rng(141126)
    if domain=='official':
        root=WORKSPACE/'Submission_Files/dev_data/dataset'
        rows=list(csv.DictReader((root/'metadata.csv').open()))
        selected=sorted(map(int,rng.choice(len(rows),6,replace=False)))
        answer=[]
        for i in selected:
            row=rows[i]
            image=root/'images'/row['filename']
            if not image.exists(): image=root/row['filename']
            truth=[float(row['lat_gtruth']),float(row['lon_gtruth'])]
            meta={k:v for k,v in row.items() if k not in ('lat_gtruth','lon_gtruth','lat_sc','lon_sc')}
            answer.append(dict(id=i,path=str(image),meta=meta,truth_latlon=truth,flight='official_unknown'))
        return answer
    root=WORKSPACE/'Code_Testing/dataset/NUDT/Thermal-UAV/test/changsha'
    all_rows=[]
    for table in sorted(root.glob('*/Thermal_info.csv')):
        for row in csv.DictReader(table.open()):
            all_rows.append((table,row))
    selected=sorted(map(int,rng.choice(len(all_rows),6,replace=False)))
    answer=[]
    for i in selected:
        table,r=all_rows[i]
        fx=12/9.83*1280
        meta=dict(filename=r['filename'],alt=r['rel_height'],yaw=r['yaw'],pitch=0,roll=0,
                  cam_fx=fx,cam_fy=fx,cam_cx=639.5,cam_cy=511.5,cam_k1=0,cam_k2=0,cam_k3=0,cam_p1=0,cam_p2=0)
        answer.append(dict(id=i,path=str(table.parent/'Thermal'/r['filename']),meta=meta,
                           truth_latlon=[float(r['lat']),float(r['lon'])],flight=table.parent.name))
    return answer


def measure(answer,truth,sigma):
    answered=bool(answer.get('answered',False))
    error=math.hypot(answer['x']-truth[0],answer['y']-truth[1]) if answered else None
    ratio=error/sigma if answered else None
    points=(10 if ratio<=1 else 6 if ratio<=2 else 3 if ratio<=4 else 0 if ratio<=8 else -20) if answered else 0
    return dict(answered=answered,error_m=error,points=points,wrong_over4sigma=bool(answered and ratio>4),
                penalised_over8sigma=bool(answered and ratio>8))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('domain',choices=['official','nudt'])
    args=parser.parse_args()
    frozen=verify_frozen()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    settings.MODE='full'
    settings.TIME_BUDGET_S=None
    settings.DINO_BATCH=5 # same individual inference, lower CPU peak memory
    if args.domain=='nudt':
        settings.LEVELS=[3,4]
        settings.GEOMETRY.update(attitude=False,yaw_offset=0.0,scale=.8,height_offset=0.0)
        read_camera=photo.read_camera
        def nudt_camera(meta):
            camera=read_camera(meta)
            camera.update(width=1280,height=1024)
            return camera
        photo.read_camera=nudt_camera
    store_path=WORKSPACE/'Submission_Files/myteam/preprocess' if args.domain=='official' else OUT/'nudt_store_codex'
    samples=inputs(args.domain)
    store=MapStore(store_path)
    rng=np.random.default_rng(141127)
    for sample in samples:
        truth=store.gps_to_metres(*sample['truth_latlon'])
        r=5000*math.sqrt(rng.random())
        angle=rng.uniform(0,2*math.pi)
        prior=(truth[0]+r*math.cos(angle),truth[1]+r*math.sin(angle))
        lat,lon=store.metres_to_gps(*prior)
        sample['meta'].update(lat_sc=lat,lon_sc=lon)
        sample.update(truth_xy=truth,prior_xy=prior)
    manifest=dict(domain=args.domain,samples=samples,settings={k:getattr(settings,k) for k in ['LEVELS','GEOMETRY','DISC_RADIUS','K_PEAKS','M_FINE','LOFTR_CONFIDENCE','LOFTR_COARSE_THRESHOLD','MIN_INLIERS','TIME_BUDGET_S','DINO_BATCH','QUERY_SET','SECOND_PASS']},
                  weights_sha256=frozen['files']['encoder_satellite_codex.pt'],frozen_before_eval=frozen['frozen_at_unix'],started_at_unix=time.time(),
                  source_sha256={str(p.relative_to(WORKSPACE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (WORKSPACE/'Main_Pipeline_Headless').rglob('*.py')})
    (OUT/f'{args.domain}_evaluation_manifest_codex.json').write_text(json.dumps(manifest,indent=2))
    localiser=InstrumentedLocaliser(str(store_path),str(WORKSPACE/'Submission_Files/myteam/models'))
    print('loaded',args.domain,localiser.setup_seconds,flush=True)
    output=OUT/f'{args.domain}_pipeline_codex.jsonl'
    assert not output.exists(),'Do not overwrite an evaluation run'
    with output.open('w') as f:
        for sample in samples:
            start=time.perf_counter()
            localiser.captured=[]
            localiser.encoder_seconds=0.0
            sigma=sigma_m(float(sample['meta']['alt']),float(sample['meta']['cam_fx']))
            row=dict(id=sample['id'],filename=sample['meta']['filename'],domain=args.domain,flight=sample['flight'],sigma_m=sigma)
            try:
                baseline=localiser.locate(sample['path'],sample['meta'])
                alternatives,candidates=select(localiser.captured,sigma,frozen['calibration'])
                row.update(baseline=baseline,alternatives=alternatives,candidates=candidates,
                           scores={name:measure(a,sample['truth_xy'],sigma) for name,a in dict(baseline=baseline,**alternatives).items()},
                           encoder_seconds=localiser.encoder_seconds)
            except Exception as error:
                import traceback
                traceback.print_exc()
                row.update(failure=repr(error),scores={name:measure({},sample['truth_xy'],sigma) for name in ['baseline','sat_veto','sat_select']})
            row['wall_seconds']=time.perf_counter()-start
            f.write(json.dumps(row,allow_nan=False)+'\n')
            f.flush()
            print(args.domain,sample['id'],row['scores'],'seconds',round(row['wall_seconds'],1),flush=True)


if __name__=='__main__':
    main()

"""Post-pilot diagnostic: real image, truth deliberately outside searchable disc.

Not a valid-hint benchmark, not calibration, no parameter changes. Uses the first
preselected official frame and the farthest inset map corner deterministically.
"""
import json
import math
import time

import cv2
import torch

from run_pipeline_codex import (OUT, WORKSPACE, InstrumentedLocaliser, MapStore, measure,
                                select, settings, sigma_m, verify_frozen)


def main():
    frozen=verify_frozen()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    settings.MODE='full'
    settings.TIME_BUDGET_S=None
    settings.DINO_BATCH=5
    sample=json.loads((OUT/'official_evaluation_manifest_codex.json').read_text())['samples'][0]
    store_path=WORKSPACE/'Submission_Files/myteam/preprocess'
    store=MapStore(store_path)
    width,height=store.info['width']*.3,store.info['height']*.3
    corners=[(x,y) for x in (500,width-500) for y in (500,height-500)]
    truth=sample['truth_xy']
    prior=max(corners,key=lambda p:math.dist(p,truth))
    distance=math.dist(prior,truth)
    assert distance>settings.DISC_RADIUS+1000
    sample['meta']['lat_sc'],sample['meta']['lon_sc']=store.metres_to_gps(*prior)
    output=OUT/'negative_control_codex.json'
    assert not output.exists(),'Do not overwrite the negative control'
    manifest=dict(protocol='Post-pilot negative diagnostic, outside valid 5 km hint assumption; no fitting.',
                  source_id=sample['id'],prior=prior,truth_distance_m=distance,
                  weights_sha256=frozen['files']['encoder_satellite_codex.pt'],created_at_unix=time.time())
    (OUT/'negative_control_manifest_codex.json').write_text(json.dumps(manifest,indent=2))
    localiser=InstrumentedLocaliser(str(store_path),str(WORKSPACE/'Submission_Files/myteam/models'))
    baseline=localiser.locate(sample['path'],sample['meta'])
    sigma=sigma_m(float(sample['meta']['alt']),float(sample['meta']['cam_fx']))
    alternatives,candidates=select(localiser.captured,sigma,frozen['calibration'])
    result=dict(manifest=manifest,baseline=baseline,alternatives=alternatives,candidates=candidates,
                scores={name:measure(a,truth,sigma) for name,a in dict(baseline=baseline,**alternatives).items()},
                encoder_seconds=localiser.encoder_seconds)
    output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(scores=result['scores'],n_queries=baseline['n_queries'],
                          second_pass=baseline['second_pass'],seconds=baseline['total_seconds']),indent=2),flush=True)


if __name__=='__main__':
    main()

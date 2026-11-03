import time

import torch

from coarse_search.search import search_frame
from fine_matching.second_stage import second_stage
from header.settings import DEVICE
from image_processing.geometry import render_query
from result_thresholding.decision import final_answer


def run_correlation(testset, store, variants, dino=None):
    assert len(variants) > 0, "need at least one variant"
    disc = {}
    seconds = []
    start = time.time()
    ids = testset.sample_ids
    for count, i in enumerate(ids):
        i = int(i)
        frame_start = time.time()
        level = testset.query_levels[i]
        canvas, valid = render_query(testset.items[i], testset.camera, level)
        found, cutoff = search_frame(canvas, valid, level, testset.priors[i], store, variants=variants, dino=dino)
        disc[i] = dict(peaks=found, cutoff=cutoff)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        seconds.append(time.time() - frame_start)
        if count % 10 == 0 or count == len(ids) - 1:
            print("photo %3d / %d done, %.0f s so far" % (count + 1, len(ids), time.time() - start))
    return disc, seconds


def run_minima(testset, store, matcher, disc):
    assert matcher is not None, "run_minima needs a loaded matcher"
    assert all(int(i) in disc for i in testset.sample_ids), "disc is missing some sample photos (run run_correlation first)"
    stage2 = {}
    final = {}
    start = time.time()
    ids = testset.sample_ids
    for count, i in enumerate(ids):
        i = int(i)
        level = testset.query_levels[i]
        canvas, valid = render_query(testset.items[i], testset.camera, level)
        peaks = disc[i]["peaks"][("zncc", 0.0)]
        stage2[i] = second_stage(matcher, canvas, valid, level, peaks, store["levels"])
        final[i] = final_answer(peaks, stage2[i])
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        if count % 10 == 0 or count == len(ids) - 1:
            print("MINIMA stage: photo %3d / %d, %.0f s so far" % (count + 1, len(ids), time.time() - start))
    per_photo = (time.time() - start) / len(ids)
    print("MINIMA stage: %.2f s per photo" % per_photo)
    return stage2, final, per_photo

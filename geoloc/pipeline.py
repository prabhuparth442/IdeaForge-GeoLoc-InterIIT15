"""The whole pipeline behind one object: Localiser.

    localiser = Localiser(store_dir, models_dir)       # slow, once: opens the map store, loads LoFTR and DINO
    answer = localiser.locate(image_path, meta)       # per photo, meta = the metadata row as a dict of strings

For one photo: read the photo and the pose, pick the zoom level, then for every heading/scale query
  draw the canvas -> coarse search (FFT correlation over the disc) -> DINO re-order -> LoFTR + RANSAC on the best M places.
Then all candidates of all queries are pooled and one is chosen (decision.py).

In MODE "full" every stage is timed (the GPU is synchronised before each reading) and more is returned; in MODE "answer" none of that is done.
"""
import os
import sys
import time

import cv2
import numpy as np
import torch

import settings
from geoloc import decision, fine_matching, photo, queries, semantic_rerank
from geoloc.coarse_search import DEVICE, score_map_for_query
from geoloc.levels import level_for_height
from geoloc.map_store import MapStore

STAGES = ["photo", "render", "coarse", "semantic", "fine", "decision"]


class Localiser:
    def __init__(self, store_dir, models_dir):
        self.timed = settings.MODE == "full"
        self.setup_seconds = {}

        start = time.time()
        self.store = MapStore(store_dir)
        self.setup_seconds["map store"] = time.time() - start

        start = time.time()
        self.loftr = fine_matching.load_loftr(os.path.join(models_dir, settings.LOFTR_FILE))
        self.setup_seconds["LoFTR"] = time.time() - start

        self.encoder = None
        if settings.USE_DINO:
            start = time.time()
            self.encoder = semantic_rerank.load_encoder(os.path.join(models_dir, settings.DINO_FILE))
            self.setup_seconds["DINO"] = time.time() - start
        self.queries = queries.query_list()
        self.extra_queries = queries.second_pass_queries()

    def warm_up(self):
        """One dummy pass through the networks, so the first real photo does not pay for starting the GPU."""
        noise = np.random.default_rng(0).random((settings.TILE, settings.TILE)).astype(np.float32)
        fine_matching.loftr_matches(self.loftr, noise, noise)
        if self.encoder is not None:
            semantic_rerank.token_grids(self.encoder, noise[None])
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    def now(self):
        if self.timed and DEVICE == "cuda":
            torch.cuda.synchronize()
        return time.time()

    def run_query(self, photo_picture, pose, camera, level, prior_m, query, seconds, cache):
        """One query up to the fine stage. Returns dict(peaks, z, stage2)."""
        clock = self.now()
        canvas, valid = photo.draw_canvas(photo_picture, pose, camera, level, query["heading"], query["scale_factor"])
        clock = self.lap(seconds, "render", clock)

        score_map, geometry = score_map_for_query(canvas, valid, level, prior_m, self.store.features, cache)
        clock = self.lap(seconds, "coarse", clock)

        level_image = self.store.levels[level]
        if self.encoder is not None:
            peaks, z = semantic_rerank.semantic_layer(self.encoder, score_map, geometry, canvas, valid, level, level_image)
        else:
            peaks, z = semantic_rerank.coarse_only(score_map, geometry)
        clock = self.lap(seconds, "semantic", clock)

        stage2 = fine_matching.check_places(self.loftr, canvas, valid, level, peaks, level_image)
        self.lap(seconds, "fine", clock)
        return dict(peaks=peaks, z=z, stage2=stage2)

    def run_queries(self, query_list, photo_picture, pose, camera, level, prior_m, meta, seconds, results, used_queries, cache):
        for query in query_list:
            cv2.setRNGSeed(settings.SEED)
            try:
                results.append(self.run_query(photo_picture, pose, camera, level, prior_m, query, seconds, cache))
                used_queries.append(query)
            except ValueError as problem:
                print("%s: query %s skipped: %s" % (meta.get("filename", "?"), query, problem), file=sys.stderr, flush=True)
                continue
            if settings.EARLY_STOP_INLIERS is not None and self.best_inliers(results) >= settings.EARLY_STOP_INLIERS:
                break

    def best_inliers(self, results):
        if len(results) == 0:
            return 0
        peaks, inliers, shifts, z, owner = decision.pool_queries(results)
        return decision.choose_candidate(peaks, inliers, shifts, z)[2]

    def is_answered(self, results):
        return bool(decision.should_answer(self.best_inliers(results)))

    def lap(self, seconds, label, clock):
        if not self.timed:
            return clock
        now = self.now()
        seconds[label] = seconds.get(label, 0.0) + now - clock
        return now

    def locate(self, image_path, meta):
        """Returns dict(answered, lat, lon, confidence, inliers). lat and lon are the best guess even when answered is False
        (None only when the hint point is outside the map). MODE "full" adds x, y, level, seconds, total_seconds, gpu_peak_mb, heading, scale_factor, z."""
        if self.timed and DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = self.now()
        seconds = {}

        camera = photo.read_camera(meta)
        pose = photo.read_pose(meta)
        prior_m = self.store.gps_to_metres(float(meta["lat_sc"]), float(meta["lon_sc"]))
        level = level_for_height(pose["height"], camera)
        clock = self.now()
        photo_picture = photo.prepare_photo(image_path, pose, camera, level)
        self.lap(seconds, "photo", clock)

        results = []
        used_queries = []
        cache = {}          # the map window and its FFTs, shared by all queries of this photo
        self.run_queries(self.queries, photo_picture, pose, camera, level, prior_m, meta, seconds, results, used_queries, cache)
        second_pass = False
        if len(self.extra_queries) > 0 and not self.is_answered(results):
            second_pass = True
            self.run_queries(self.extra_queries, photo_picture, pose, camera, level, prior_m, meta, seconds, results, used_queries, cache)

        answer = dict(answered=False, lat=None, lon=None, confidence=0.0, inliers=0)
        if len(results) == 0:
            return self.finish(answer, start, seconds, level, prior_m, None, None)

        clock = self.now()
        peaks, inliers, shifts, z, owner = decision.pool_queries(results)
        x, y, best_inliers, best, z_best = decision.choose_candidate(peaks, inliers, shifts, z)
        self.lap(seconds, "decision", clock)

        if x < 1e8:
            answer["lat"], answer["lon"] = self.store.metres_to_gps(float(x), float(y))
        answer["inliers"] = int(best_inliers)
        answer["answered"] = bool(decision.should_answer(best_inliers)) and answer["lat"] is not None
        answer["confidence"] = min(1.0, best_inliers / settings.CONFIDENCE_FULL_AT)
        winner = used_queries[int(owner[best])]
        return self.finish(answer, start, seconds, level, prior_m, winner, z_best, (x, y), second_pass)

    def finish(self, answer, start, seconds, level, prior_m, winner, z_best, xy=None, second_pass=False):
        if DEVICE == "cuda":
            if self.timed:
                answer["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.empty_cache()
        if not self.timed:
            return answer
        answer["total_seconds"] = self.now() - start
        answer["seconds"] = seconds
        answer["level"] = level
        answer["prior_m"] = prior_m
        answer["x"], answer["y"] = (None, None) if xy is None else xy
        answer["heading"] = None if winner is None else winner["heading"]
        answer["scale_factor"] = None if winner is None else winner["scale_factor"]
        answer["z"] = z_best
        answer["n_queries"] = len(self.queries) + (len(self.extra_queries) if second_pass else 0)
        answer["second_pass"] = second_pass
        return answer

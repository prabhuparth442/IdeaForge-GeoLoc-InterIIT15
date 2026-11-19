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
from geoloc import agreement, aligned_count, decision, fine_matching, native_fine, photo, photo_view, queries, scoring, semantic_rerank, xoftr_matcher
from geoloc.coarse_search import DEVICE, score_map_for_query
from geoloc.levels import level_for_height, level_gsd
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

        self.xoftr = None
        if settings.XOFTR_FROM_HEIGHT_M is not None or settings.FALLBACK_MATCHER:
            start = time.time()
            self.xoftr = xoftr_matcher.load_xoftr(os.path.join(models_dir, settings.XOFTR_FILE))
            self.setup_seconds["XoFTR"] = time.time() - start

        self.encoder = None
        if settings.USE_DINO:
            start = time.time()
            self.encoder = semantic_rerank.load_encoder(os.path.join(models_dir, settings.DINO_FILE))
            self.setup_seconds["DINO"] = time.time() - start
        self.queries = queries.query_list()
        self.extra_queries = queries.second_pass_queries()
        self.guard = agreement.InflationGuard()      # running mean over the photos of this process (only used with USE_INFLATION_GUARD)
        self.photo_start = 0.0         # clock of the photo that is being answered (for TIME_BUDGET_S)
        self.slowest_query = 0.0       # seconds of the slowest query of that photo so far
        self.copy_queries_run = 0      # queries run on altered copies for the photo that is being answered
        self.native = None             # native-resolution LoFTR context of the photo being answered (geoloc/native_fine.py), None = off
        self.fallback_native = None    # the same for the fallback matcher (FALLBACK_NATIVE_FROM_HEIGHT_M), None = off
        self.fallback_queries = 0      # queries of the photo being answered on which the fallback matcher was tried
        self.photo_view = None         # photo view of the photo being answered (geoloc/photo_view.py), None = off
        self.photo_view_for = None     # the photo picture the photo view belongs to (altered copies are other pictures and use the canvas path)

    def warm_up(self):
        """One dummy pass through the networks, so the first real photo does not pay for starting the GPU."""
        noise = np.random.default_rng(0).random((settings.TILE, settings.TILE)).astype(np.float32)
        fine_matching.loftr_matches(self.loftr, noise, noise)
        if self.xoftr is not None:
            fine_matching.loftr_matches(self.xoftr, noise, noise)
        if self.encoder is not None:
            semantic_rerank.token_grids(self.encoder, noise[None])
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    def fine_model_for(self, pose):
        """The matcher for the fine stage of this photo: XoFTR from XOFTR_FROM_HEIGHT_M up, LoFTR otherwise."""
        if self.xoftr is not None and settings.XOFTR_FROM_HEIGHT_M is not None and pose["height"] >= settings.XOFTR_FROM_HEIGHT_M:
            return self.xoftr
        return self.loftr

    def other_model_for(self, pose):
        """The matcher that is NOT the photo's own: LoFTR for a photo that uses XoFTR, XoFTR otherwise."""
        if self.fine_model_for(pose) is self.xoftr:
            return self.loftr
        return self.xoftr

    def merge_fallback(self, stage2, pose, camera, level, peaks, canvas, valid, level_image, photo_picture, query):
        """The other matcher on the same candidates; every candidate keeps the higher inlier count and its shift."""
        self.fallback_queries += 1
        other = self.other_model_for(pose)
        if self.fallback_native is not None and photo_picture is self.fallback_native["for_picture"]:
            fine_level = self.fallback_native["level"]
            fine_canvas, fine_valid = photo.draw_canvas(self.fallback_native["picture"], pose, camera, fine_level, query["heading"], query["scale_factor"], self.fallback_native["size"])
            second = fine_matching.check_places(other, fine_canvas, fine_valid, fine_level, peaks, self.store.levels[fine_level], self.fallback_native["tile_filter"], level)
        else:
            second = fine_matching.check_places(other, canvas, valid, level, peaks, level_image)
        better = second["inliers"] > stage2["inliers"]
        return dict(inliers=np.where(better, second["inliers"], stage2["inliers"]), shifts=np.where(better[:, None], second["shifts"], stage2["shifts"]), aligned=stage2["aligned"])

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
        use_dino = self.encoder is not None and (settings.DINO_ONLY_BELOW_M is None or pose["height"] < settings.DINO_ONLY_BELOW_M)
        if use_dino:
            peaks, z = semantic_rerank.semantic_layer(self.encoder, score_map, geometry, canvas, valid, level, level_image)
        else:
            peaks, z = semantic_rerank.coarse_only(score_map, geometry)
        clock = self.lap(seconds, "semantic", clock)

        in_view = self.photo_view is not None and photo_picture is self.photo_view_for
        if in_view:
            limit = settings.MAX_SHIFT_CELLS * settings.CELL * level_gsd(level)
            stage2 = self.photo_view.check_places(self.fine_model_for(pose), query, peaks, limit)
        elif self.native is not None and photo_picture is self.native["for_picture"]:
            fine_level = self.native["level"]
            fine_canvas, fine_valid = photo.draw_canvas(self.native["picture"], pose, camera, fine_level, query["heading"], query["scale_factor"], self.native["size"])
            stage2 = fine_matching.check_places(self.fine_model_for(pose), fine_canvas, fine_valid, fine_level, peaks, self.store.levels[fine_level], self.native["tile_filter"], level)
        else:
            stage2 = fine_matching.check_places(self.fine_model_for(pose), canvas, valid, level, peaks, level_image)
        if not in_view and settings.FALLBACK_MATCHER and float(np.max(stage2["inliers"])) < settings.MIN_INLIERS:
            stage2 = self.merge_fallback(stage2, pose, camera, level, peaks, canvas, valid, level_image, photo_picture, query)
        self.lap(seconds, "fine", clock)
        return dict(peaks=peaks, z=z, stage2=stage2)

    def out_of_time(self):
        """True when one more query would probably end after TIME_BUDGET_S."""
        if settings.TIME_BUDGET_S is None:
            return False
        elapsed = time.time() - self.photo_start
        return elapsed + self.slowest_query > settings.TIME_BUDGET_S

    def run_queries(self, query_list, photo_picture, pose, camera, level, prior_m, meta, seconds, results, used_queries, cache):
        for query in query_list:
            if len(used_queries) > 0 and self.out_of_time():
                print("%s: time budget reached, %d queries done, skipping the rest" % (meta.get("filename", "?"), len(used_queries)), file=sys.stderr, flush=True)
                break
            cv2.setRNGSeed(settings.SEED)
            query_start = time.time()
            try:
                results.append(self.run_query(photo_picture, pose, camera, level, prior_m, query, seconds, cache))
                used_queries.append(query)
            except ValueError as problem:
                print("%s: query %s skipped: %s" % (meta.get("filename", "?"), query, problem), file=sys.stderr, flush=True)
                continue
            self.slowest_query = max(self.slowest_query, time.time() - query_start)
            if settings.EARLY_STOP_INLIERS is not None and self.best_inliers(results) >= settings.EARLY_STOP_INLIERS:
                break

    def best_inliers(self, results):
        if len(results) == 0:
            return 0
        peaks, inliers, shifts, z, owner = decision.pool_queries(results)
        return decision.choose_candidate(peaks, inliers, shifts, z)[2]

    def guard_factor(self, results, sigma):
        """1.0 unless the inflation guard is on and the wrong places of this run are stronger than on the development photos."""
        if not settings.USE_INFLATION_GUARD:
            return 1.0
        peaks, inliers, shifts, z, owner = decision.pool_queries(results)
        x, y = decision.choose_candidate(peaks, inliers, shifts, z)[0:2]
        return self.guard.factor_after(agreement.far_level(results, x, y, sigma))

    def bar_factor(self, level, pose):
        """The factor that turns MIN_INLIERS into this photo's bar: PHOTO_VIEW_MIN_INLIERS for a photo that uses the photo view, otherwise the level's (1.0 unless BAR_BY_LEVEL names it)."""
        if self.photo_view is not None:
            return settings.PHOTO_VIEW_MIN_INLIERS / settings.MIN_INLIERS
        return self.level_bar_factor(level)

    def level_bar_factor(self, level):
        """1.0 unless BAR_BY_LEVEL names this level: then the factor that turns MIN_INLIERS into that level's bar."""
        if not settings.BAR_BY_LEVEL or level not in settings.BAR_BY_LEVEL:
            return 1.0
        return settings.BAR_BY_LEVEL[level] / settings.MIN_INLIERS

    def answered_on(self, results, factor, sigma):
        """True when the pooled winner has enough inliers, or (with USE_AGREEMENT) enough queries found the same place."""
        peaks, inliers, shifts, z, owner = decision.pool_queries(results)
        x, y, best_inliers, best, z_best = decision.choose_candidate(peaks, inliers, shifts, z)
        if decision.should_answer(best_inliers, factor):
            return True
        if not settings.USE_AGREEMENT or best_inliers == 0:
            return False
        floor = int(round(settings.AGREE_QUERY_INLIERS * factor))
        count = agreement.agreeing_queries(results, x, y, settings.AGREE_RADIUS_SIGMA * sigma, floor)
        return count >= settings.AGREE_MIN_QUERIES

    def try_altered_copies(self, image_path, pose, camera, level, prior_m, meta, seconds, cache, factor, sigma):
        """Runs the standard queries on each altered copy of the photo. Returns (results, used_queries) of the copy that answers with the most
        inliers, or None when no copy answers. Stops early when the time budget is gone."""
        best = None
        best_inliers = 0
        for kind in settings.ALTERED_COPY_KINDS:
            if self.out_of_time():
                break
            clock = self.now()
            copy_picture = photo.prepare_photo(image_path, pose, camera, level, kind)
            self.lap(seconds, "photo", clock)
            copy_results = []
            copy_queries_used = []
            self.run_queries(self.queries, copy_picture, pose, camera, level, prior_m, meta, seconds, copy_results, copy_queries_used, cache)
            self.copy_queries_run += len(copy_queries_used)
            if len(copy_results) == 0:
                continue
            inliers = self.best_inliers(copy_results)
            if settings.ALTERED_COPY_BAR_ONLY:
                copy_answers = bool(decision.should_answer(inliers, factor))
            else:
                copy_answers = self.answered_on(copy_results, factor, sigma)
            if not copy_answers:
                continue
            if inliers > best_inliers:
                best = (copy_results, copy_queries_used)
                best_inliers = inliers
        return best

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
        self.photo_start = time.time()
        self.slowest_query = 0.0
        self.copy_queries_run = 0
        seconds = {}

        camera = photo.read_camera(meta)
        pose = photo.read_pose(meta)
        prior_m = self.store.gps_to_metres(float(meta["lat_sc"]), float(meta["lon_sc"]))
        level = level_for_height(pose["height"], camera)
        clock = self.now()
        photo_picture = photo.prepare_photo(image_path, pose, camera, level)
        self.native = None
        fine_level = native_fine.fine_level(pose["height"], level, self.store.levels)
        if fine_level is not None:
            fine_gsd = level_gsd(fine_level)
            sigma_tile = native_fine.tile_sigma(pose["height"] / camera["fx"], fine_gsd)
            self.native = dict(for_picture=photo_picture, level=fine_level, size=native_fine.canvas_size(level - fine_level),
                               picture=photo.prepare_photo(image_path, pose, camera, fine_level, None, True),
                               tile_filter=lambda tile: native_fine.clean_tile(tile, sigma_tile))
        self.photo_view = None
        self.photo_view_for = None
        if photo_view.active(pose):
            self.native = None
            self.photo_view = photo_view.PhotoView(image_path, pose, camera, self.store.levels)
            self.photo_view_for = photo_picture
        self.fallback_native = None
        self.fallback_queries = 0
        gate = settings.FALLBACK_NATIVE_FROM_HEIGHT_M
        if settings.FALLBACK_MATCHER and gate is not None and pose["height"] >= gate:
            drop = settings.NATIVE_FINE_LEVELS
            if level - drop >= min(settings.LEVELS) and (level - drop) in self.store.levels:
                fine_gsd = level_gsd(level - drop)
                sigma_tile = native_fine.tile_sigma(pose["height"] / camera["fx"], fine_gsd)
                self.fallback_native = dict(for_picture=photo_picture, level=level - drop, size=native_fine.canvas_size(drop),
                                            picture=photo.prepare_photo(image_path, pose, camera, level - drop, None, True),
                                            tile_filter=lambda tile: native_fine.clean_tile(tile, sigma_tile))
        self.lap(seconds, "photo", clock)

        results = []
        used_queries = []
        cache = {}          # the map window and its FFTs, shared by all queries of this photo
        self.run_queries(self.queries, photo_picture, pose, camera, level, prior_m, meta, seconds, results, used_queries, cache)
        first_pass_answered = False
        factor = self.bar_factor(level, pose)
        sigma = scoring.sigma_m(pose["height"], camera["fx"])
        if len(results) > 0:
            factor = self.guard_factor(results, sigma) * self.bar_factor(level, pose)
            first_pass_answered = self.answered_on(results, factor, sigma)
        copies_first = settings.USE_ALTERED_COPIES and self.photo_view is None and settings.ALTERED_COPIES_ORDER == "before_second_pass"
        copies_last = settings.USE_ALTERED_COPIES and self.photo_view is None and settings.ALTERED_COPIES_ORDER == "after_second_pass"
        copy_used = False
        if copies_first and not first_pass_answered and len(results) > 0:
            rescue = self.try_altered_copies(image_path, pose, camera, level, prior_m, meta, seconds, cache, factor, sigma)
            if rescue is not None:
                results, used_queries = rescue
                first_pass_answered = True
                copy_used = True
        second_pass = False
        if len(self.extra_queries) > 0 and not first_pass_answered:
            second_pass = True
            self.run_queries(self.extra_queries, photo_picture, pose, camera, level, prior_m, meta, seconds, results, used_queries, cache)
            if copies_last and len(results) > 0 and not decision.should_answer(self.best_inliers(results), factor, True):
                rescue = self.try_altered_copies(image_path, pose, camera, level, prior_m, meta, seconds, cache, factor, sigma)
                if rescue is not None:
                    results, used_queries = rescue
                    first_pass_answered = True
                    second_pass = False
                    copy_used = True

        answer = dict(answered=False, lat=None, lon=None, confidence=0.0, inliers=0)
        if len(results) == 0:
            return self.finish(answer, start, seconds, level, prior_m, None, None)

        clock = self.now()
        peaks, inliers, shifts, z, owner = decision.pool_queries(results)
        x, y, best_inliers, best, z_best = decision.choose_candidate(peaks, inliers, shifts, z)
        self.lap(seconds, "decision", clock)
        # runner-up: the best candidate at a clearly different place (a look-alike place has a strong rival); logged only, nothing uses it yet
        second_inliers = 0.0
        if best_inliers > 0:
            far = np.hypot(peaks[:, 1] + shifts[:, 0] - x, peaks[:, 2] + shifts[:, 1] - y) > max(30.0, 3 * sigma)
            if far.any():
                second_inliers = float(inliers[far].max())

        if second_pass:
            answered = bool(decision.should_answer(best_inliers, factor, True))      # the second pass is decided by the bar only (SECOND_PASS_MIN_INLIERS if set)
        else:
            answered = first_pass_answered
        rescued = False
        if not answered and settings.USE_ALIGNED_RESCUE:
            place = aligned_count.pick_place(peaks, shifts, decision.pool_aligned(results), sigma)
            if place is not None:
                x, y = place
                answered = True
                rescued = True
        verify = None
        if answered and self.photo_view is not None and settings.PHOTO_VIEW_VERIFY_BELOW is not None and best_inliers < settings.PHOTO_VIEW_VERIFY_BELOW and x < 1e8 and not copy_used \
                and (settings.PHOTO_VIEW_VERIFY_MAX_HEIGHT_M is None or pose["height"] <= settings.PHOTO_VIEW_VERIFY_MAX_HEIGHT_M):
            verify = self.verify_with_other_matcher(results, used_queries, owner, best, pose, sigma, x, y, level)
            if verify is not None and settings.PHOTO_VIEW_VERIFY_ACT and (verify[0] < settings.PHOTO_VIEW_VERIFY_MIN or verify[1] > 2 * sigma):
                answered = False
        pose_fit = None
        if answered and self.photo_view is not None and settings.PHOTO_VIEW_FIT_POSE and x < 1e8 and not copy_used:
            pose_fit = self.fit_view_pose(results, owner, best, pose, camera, sigma, x, y)
            if pose_fit is not None:
                x, y = pose_fit[0], pose_fit[1]
        refine = None
        if answered and settings.REFINE_FINER_LEVELS > 0 and x < 1e8 and not copy_used and self.photo_view is None:
            refine = self.refine_finer(image_path, pose, camera, level, used_queries[int(owner[best])], x, y, sigma, seconds)
            if refine is not None:
                x, y = refine[0], refine[1]
        if x < 1e8:
            answer["lat"], answer["lon"] = self.store.metres_to_gps(float(x), float(y))
        answer["inliers"] = int(best_inliers)
        answer["answered"] = answered and answer["lat"] is not None
        answer["confidence"] = min(1.0, best_inliers / settings.CONFIDENCE_FULL_AT)
        winner = used_queries[int(owner[best])]
        answer = self.finish(answer, start, seconds, level, prior_m, winner, z_best, (x, y), second_pass)
        if self.timed:
            answer["copy_used"] = copy_used
            answer["rescued_by_alignment"] = rescued
            answer["guard_factor"] = factor
            answer["refined"] = refine is not None
            answer["native_used"] = self.native is not None
            answer["xoftr_used"] = self.xoftr is not None and settings.XOFTR_FROM_HEIGHT_M is not None and pose["height"] >= settings.XOFTR_FROM_HEIGHT_M
            answer["fallback_queries"] = self.fallback_queries
            answer["photo_view_used"] = self.photo_view is not None
            answer["second_inliers"] = second_inliers
            answer["verify_inliers"] = None if verify is None else verify[0]
            answer["verify_distance_m"] = None if verify is None else verify[1]
            answer["pose_fit_used"] = pose_fit is not None
            answer["pose_fit_move_m"] = None if pose_fit is None else pose_fit[2]
            for key in ("rms_px", "dpitch", "droll", "dyaw", "dheight", "rounds"):
                answer["fit_" + key] = None if pose_fit is None else pose_fit[3][key]
            answer["refine_shift_m"] = None if refine is None else refine[2]
            answer["refine_inliers"] = None if refine is None else refine[3]
        return answer

    def verify_with_other_matcher(self, results, used_queries, owner, best, pose, sigma, x, y, level):
        """The other matcher on the winner's candidate in the photo view. Returns (inliers, distance_m from the answer) or None when it cannot run."""
        other = self.other_model_for(pose)
        if other is None:
            return None
        query_index = int(owner[best])
        offset = sum(len(results[i]["peaks"]) for i in range(query_index))
        peaks = np.asarray(results[query_index]["peaks"], dtype=float)[best - offset:best - offset + 1]
        query = used_queries[query_index]
        limit = settings.MAX_SHIFT_CELLS * settings.CELL * level_gsd(level)
        found = self.photo_view.check_places(other, query, peaks, limit)
        inliers = float(found["inliers"][0])
        distance = float(np.hypot(peaks[0, 1] + found["shifts"][0, 0] - x, peaks[0, 2] + found["shifts"][0, 1] - y)) if inliers > 0 else 1e9
        return inliers, distance

    def fit_view_pose(self, results, owner, best, pose, camera, sigma, x, y):
        """The winner's position from a pose fit to its photo-view matches (photo_view.fit_pose), with PHOTO_VIEW_FIT_ROUNDS - 1 more rounds of draw again - match - fit.
        Returns (x, y, move_m, diagnostics) or None when there are no matches kept or the first fit moves the answer more than PHOTO_VIEW_FIT_MAX_MOVE_SIGMA sigma from the median-vote answer (x, y)."""
        query_index = int(owner[best])
        offset = sum(len(results[i]["peaks"]) for i in range(query_index))
        match = results[query_index]["stage2"].get("matches", {}).get(best - offset)
        if match is None:
            return None
        gsd = pose["height"] * settings.GEOMETRY["scale"] / camera["fx"]
        fitted_x, fitted_y, corrections, rms = photo_view.fit_pose(match, pose, camera, gsd)
        move = float(np.hypot(fitted_x - x, fitted_y - y))
        if move > settings.PHOTO_VIEW_FIT_MAX_MOVE_SIGMA * sigma:
            return None
        rounds = 1
        total = np.array(corrections)
        model = self.fine_model_for(pose)
        for _ in range(settings.PHOTO_VIEW_FIT_ROUNDS - 1):
            again = self.photo_view.redraw_and_match(model, match, fitted_x, fitted_y, total)
            if again is None or again["inliers"] < settings.PHOTO_VIEW_MIN_INLIERS:
                break
            x2, y2, step, rms2 = photo_view.fit_pose(again, again["pose"], camera, gsd, prior_centre=total)
            if float(np.hypot(x2 - fitted_x, y2 - fitted_y)) > 1.5 * sigma:
                break
            fitted_x, fitted_y, total, rms, match = x2, y2, total + np.array(step), rms2, again
            rounds += 1
        move = float(np.hypot(fitted_x - x, fitted_y - y))
        return fitted_x, fitted_y, move, dict(rms_px=rms, dpitch=float(total[0]), droll=float(total[1]), dyaw=float(total[2]), dheight=float(np.exp(total[3]) - 1), rounds=rounds)

    def refine_finer(self, image_path, pose, camera, level, query, x, y, sigma, seconds):
        """The answer matched once more REFINE_FINER_LEVELS levels finer. Returns (x, y, shift_m, inliers) or None when the new fit is not good enough."""
        clock = self.now()
        finer = max(min(settings.LEVELS), level - settings.REFINE_FINER_LEVELS)
        if finer >= level:
            return None
        peaks = np.array([[0.0, x, y, 0.0, 0.0]])
        if settings.REFINE_NATIVE:
            # same ground footprint as the canvas of the coarse level, photo and tile cleaned, the matcher that answered
            picture = photo.prepare_photo(image_path, pose, camera, finer, None, True)
            canvas, valid = photo.draw_canvas(picture, pose, camera, finer, query["heading"], query["scale_factor"], native_fine.canvas_size(level - finer))
            sigma_tile = native_fine.tile_sigma(pose["height"] / camera["fx"], level_gsd(finer))
            stage2 = fine_matching.check_places(self.fine_model_for(pose), canvas, valid, finer, peaks, self.store.levels[finer], lambda tile: native_fine.clean_tile(tile, sigma_tile), level)
        else:
            picture = photo.prepare_photo(image_path, pose, camera, finer)
            canvas, valid = photo.draw_canvas(picture, pose, camera, finer, query["heading"], query["scale_factor"])
            stage2 = fine_matching.check_places(self.fine_model_for(pose), canvas, valid, finer, peaks, self.store.levels[finer])
        self.lap(seconds, "refine", clock)
        inliers = float(stage2["inliers"][0])
        shift = stage2["shifts"][0]
        moved = float(np.hypot(shift[0], shift[1]))
        if inliers < settings.REFINE_MIN_INLIERS or moved > settings.REFINE_MAX_SHIFT_SIGMA * sigma:
            return None
        return float(x + shift[0]), float(y + shift[1]), moved, inliers

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
        answer["n_queries"] = len(self.queries) + (len(self.extra_queries) if second_pass else 0) + self.copy_queries_run
        answer["second_pass"] = second_pass
        return answer

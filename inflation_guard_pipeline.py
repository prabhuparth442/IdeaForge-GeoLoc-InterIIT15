"""The agreement pipeline with an inflation guard (Test_Iteration_7, pre-registered in results/preregistered_guard.json).

Problem: if every candidate gets more inliers (a sharper camera, a looser matcher, a new map), false places reach the bars 8 / 11 and the
agreement floor 3 and the pipeline answers them. Measured on our dumps with every inlier count multiplied by f: 2 to 11 wrong answers at f = 1.5 to 2.
Guard: after the 6 standard queries of a photo, measure how many inliers a WRONG place gets on this run (mean inliers of the candidates more than
4 sigma from the chosen place). A running mean over the photos seen so far (weight 30, started at PRIOR) says how strong the false places are on this
camera and map. When that level is more than DEAD_ZONE (1.5) times the prior, the bar 8, the escalation bar 11 and the agreement floor 3 are
multiplied by the ratio. Inside the dead zone nothing changes, so clean data gives the same answers as AgreementQueryPipeline.
No labels, no learned part, no change to any existing file.

Replay on our dumps (level from the 6 standard queries, as here), wrong answers at inlier factor x1 / x1.25 / x1.5 / x2, unguarded -> guarded:
  nudt707        1/1/4/6   -> 1/1/3/3      nudt1010     1/1/2/5   -> 1/1/2/2      official150  0/0/0/0 -> unchanged
  valid1212 (pre-registered confirmation, other flights)  3/4/7/11 -> 3/3/4/3  (with the level from all 27 queries: 3/2/2/2, the pre-registered form)
Clean (x1) answers are identical on all four sets.

The running level lives in the pipeline object: if every photo is run in a fresh process it never leaves the dead zone and the pipeline behaves as
AgreementQueryPipeline (safe, but no guard). Needs state kept between photos of one run.

run(item, prior) returns the keys of AgreementQueryPipeline.run plus guard_factor (1.0 when the guard is silent) and far_level (this photo's level).
"""
import math
import time

import numpy as np
import torch

from agreement_query_pipeline import AgreementQueryPipeline
from header.scoring import sigma_m
from header.settings import DEVICE
from image_processing.map_levels import level_for_height
from semantic_rerank.agreement import decide_with_agreement, query_agreement

PRIOR = 0.169          # mean far-candidate inliers of the dev sets (larger of official150 and nudt707), 6 standard queries
PRIOR_WEIGHT = 30      # the prior counts like this many photos
DEAD_ZONE = 1.5        # act only above this ratio of the running level to the prior (NUDT sets differ by about 1.3x between themselves)
FAR_SIGMA = 4.0        # a candidate this many sigma from the chosen place is called a false place


def far_level(queries, ids, x, y, sigma):
    """Mean inliers of the candidates of the queries in `ids` that lie more than FAR_SIGMA sigma from (x, y); 0.0 when there is none."""
    total = 0.0
    count = 0
    for query_id in ids:
        result = queries[query_id]
        inliers = result["stage2"]["inliers"]
        for k in range(len(result["peaks"])):
            candidate_x = result["peaks"][k][1] + result["stage2"]["shifts"][k][0]
            candidate_y = result["peaks"][k][2] + result["stage2"]["shifts"][k][1]
            if math.hypot(candidate_x - x, candidate_y - y) > FAR_SIGMA * sigma:
                total += float(inliers[k])
                count += 1
    if count == 0:
        return 0.0
    return total / count


def guard_factor_from_level(running_level, prior=PRIOR, dead_zone=DEAD_ZONE):
    """1.0 inside the dead zone, else the ratio running level / prior."""
    ratio = running_level / prior
    if ratio > dead_zone:
        return ratio
    return 1.0


class InflationGuardPipeline(AgreementQueryPipeline):
    def __init__(self, *args, prior=PRIOR, prior_weight=PRIOR_WEIGHT, dead_zone=DEAD_ZONE, **kwargs):
        super().__init__(*args, **kwargs)
        assert prior > 0 and prior_weight > 0, "prior and prior_weight must be above 0"
        assert dead_zone >= 1.0, "the dead zone is a ratio of at least 1"
        self.prior = prior
        self.prior_weight = prior_weight
        self.dead_zone = dead_zone
        self._level_sum = 0.0
        self._photo_count = 0

    def describe(self):
        lines = super().describe()
        lines.append("inflation guard: prior %.3f (weight %d photos), dead zone x%.1f; bars scale with the running far-candidate level" %
                     (self.prior, self.prior_weight, self.dead_zone))
        return lines

    def update_level(self, level_of_photo):
        """Add one photo to the running mean and return the guard factor for it (this photo included, as in the replay)."""
        self._level_sum += level_of_photo
        self._photo_count += 1
        running = (self.prior_weight * self.prior + self._level_sum) / (self.prior_weight + self._photo_count)
        return guard_factor_from_level(running, self.prior, self.dead_zone)

    def run(self, item, prior_m):
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        found = self.run_queries(item, prior_m)
        queries = found["queries"]
        ids = [q["id"] for q in queries]
        answer = self.answer_for(queries, ids)
        sigma = sigma_m(item["rel_height"], self.camera.fx)
        photo_level = far_level(queries, ids, answer["x"], answer["y"], sigma)
        factor = self.update_level(photo_level)
        bar = int(round(self.min_inliers * factor))
        agree_floor = int(round(self.agree_query_inliers * factor))
        escalation_bar = int(round(self.escalation_min_inliers * factor))
        agreement = query_agreement(queries, ids, answer["x"], answer["y"], self.agree_radius_sigma * sigma, agree_floor)
        answered = decide_with_agreement(answer["inliers"], agreement, bar, self.min_agree)
        decision_seconds = time.perf_counter() - start
        for query in queries:
            decision_seconds -= sum(query["seconds"].values())
        if answered:
            if answer["inliers"] >= bar:
                stage = "bar"
            else:
                stage = "agreement"
        else:
            stage = "skip"
        escalated = False
        if not answered and self.escalate:
            level = level_for_height(item["rel_height"], self.camera)
            queries = self.run_query_list(item, prior_m, level, self.wide_queries, queries)
            answer = self.answer_for(queries, [q["id"] for q in queries])
            escalated = True
            answered = bool(answer["inliers"] >= escalation_bar)
            if answered:
                stage = "escalation"
            else:
                stage = "skip"
        gpu_peak_mb = 0.0
        if DEVICE == "cuda":
            gpu_peak_mb = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.empty_cache()
        answer.update(seconds=self.seconds_per_stage(queries, max(decision_seconds, 0.0)), gpu_peak_mb=gpu_peak_mb,
                      answered=answered, agreement=int(agreement), escalated=escalated, stage=stage, sigma_m=float(sigma),
                      level=int(found["level"]), queries=queries, n_queries_run=len(queries), total=time.perf_counter() - start,
                      guard_factor=float(factor), far_level=float(photo_level))
        return answer

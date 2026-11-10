"""The heading-query pipeline with the agreement decision and escalation (tests/Test_Iteration_3/Decision_Report.md, round 2).

Step 1: the standard queries (headings -5, 0, +5 x scales 1.0, 1.05). Answer when the chosen place has >= min_inliers inliers (8) or when
        >= 2 queries' own best candidates lie within 1 sigma of it.
Step 2: a photo still skipped is re-run with the wide query set (headings -10..+10 step 2.5 x scales 0.95, 1.0, 1.05); the queries step 1 already
        ran are reused. Answer only when the choice over all wide queries has >= escalation_min_inliers inliers (11). No agreement here: queries
        2.5 degrees apart are not independent.

Measured from the candidate dumps (0 wrong answers on each set): official 379 photos 49.01 -> 53.54, NUDT 300 photos 53.00 -> 55.56,
NUDT 200 unseen photos 55.38 -> 57.36. Escalation runs on about 10 % of photos and adds about 45 % to the total time.

run(item, prior) returns the keys of HeadingQueryPipeline.run plus agreement, escalated and stage ("bar", "agreement", "escalation" or "skip").
"""
import time

from coarse_search.heading_queries import query_geometry, query_list
from header.scoring import sigma_m
from heading_query_pipeline import HeadingQueryPipeline
from image_processing.map_levels import level_for_height
from semantic_rerank.agreement import (AGREE_MIN_QUERIES, AGREE_QUERY_INLIERS, AGREE_RADIUS_SIGMA, ESCALATION_MIN_INLIERS,
                                       decide_with_agreement, query_agreement)

STANDARD_HEADINGS = [-5.0, 0.0, 5.0]
STANDARD_SCALES = [1.0, 1.05]
WIDE_HEADINGS = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]
WIDE_SCALES = [0.95, 1.0, 1.05]


class AgreementQueryPipeline(HeadingQueryPipeline):
    def __init__(self, *args, headings=STANDARD_HEADINGS, scale_factors=STANDARD_SCALES, wide_headings=WIDE_HEADINGS, wide_scales=WIDE_SCALES,
                 min_agree=AGREE_MIN_QUERIES, agree_radius_sigma=AGREE_RADIUS_SIGMA, agree_query_inliers=AGREE_QUERY_INLIERS,
                 escalation_min_inliers=ESCALATION_MIN_INLIERS, escalate=True, **kwargs):
        super().__init__(*args, headings=headings, scale_factors=scale_factors, **kwargs)
        self.wide_queries = query_list(wide_headings, wide_scales)
        assert min_agree >= 2, "one query always agrees with itself; min_agree must be >= 2"
        assert agree_radius_sigma > 0, "agreement radius must be above 0 sigma"
        assert escalation_min_inliers >= self.min_inliers, "the escalated bar must not be below the normal bar (more queries, more chance hits)"
        self.min_agree = min_agree
        self.agree_radius_sigma = agree_radius_sigma
        self.agree_query_inliers = agree_query_inliers
        self.escalation_min_inliers = escalation_min_inliers
        self.escalate = escalate

    def describe(self):
        lines = super().describe()
        lines.append("decision: answer at >= %d inliers or >= %d queries agreeing within %.1f sigma (each >= %d inliers)"
                     % (self.min_inliers, self.min_agree, self.agree_radius_sigma, self.agree_query_inliers))
        lines.append("escalation: %s; %d wide queries, answer at >= %d inliers" % ("on" if self.escalate else "off", len(self.wide_queries), self.escalation_min_inliers))
        return lines

    def _run_list(self, item, prior_m, level, queries, done):
        """Results for `queries` (ids renumbered 0..n-1); a query with the same heading and scale in `done` is reused, not run again."""
        reuse = {(q["heading"], q["scale_factor"]): q for q in done}
        results = []
        for position, query in enumerate(queries):
            key = (query["heading"], query["scale_factor"])
            if key in reuse:
                found = dict(reuse[key])
            else:
                found = self.candidates_for_query(item, prior_m, level, query_geometry(query["heading"], query["scale_factor"]))
                found.update(query)
            found["id"] = position
            results.append(found)
        return results

    def run(self, item, prior_m):
        start = time.perf_counter()
        found = self.run_queries(item, prior_m)
        queries = found["queries"]
        ids = [q["id"] for q in queries]
        answer = self.answer_for(queries, ids)
        sigma = sigma_m(item["rel_height"], self.camera.fx)
        agreement = query_agreement(queries, ids, answer["x"], answer["y"], self.agree_radius_sigma * sigma, self.agree_query_inliers)
        answered = decide_with_agreement(answer["inliers"], agreement, self.min_inliers, self.min_agree)
        stage = ("bar" if answer["inliers"] >= self.min_inliers else "agreement") if answered else "skip"
        escalated = False
        if not answered and self.escalate:
            level = level_for_height(item["rel_height"], self.camera)
            queries = self._run_list(item, prior_m, level, self.wide_queries, queries)
            answer = self.answer_for(queries, [q["id"] for q in queries])
            escalated = True
            answered = bool(answer["inliers"] >= self.escalation_min_inliers)
            stage = "escalation" if answered else "skip"
        answer.update(answered=answered, agreement=int(agreement), escalated=escalated, stage=stage, sigma_m=float(sigma),
                      level=int(found["level"]), queries=queries, n_queries_run=len(queries), total=time.perf_counter() - start)
        return answer

"""The single-photo pipeline with several queries per photo: one photo goes in, it is drawn once for each heading (and scale) guess, every drawing goes
through coarse search, DINO layer and fine stage, and the answer is chosen among all the candidates.

Same settings as SinglePhotoPipeline (k_peaks and m_fine are per query), plus:
  headings       degrees added to the photo's yaw, one query each (default: -10 to +10 in steps of 2.5, 9 queries)
  scale_factors  the scale of the projection is multiplied by each of these (default: only 1.0); queries = headings x scale_factors
run_queries(item, prior) returns every query's own candidates; answer_for(queries, ids) decides from any subset of them, without running anything again.
run(item, prior) is run_queries + answer_for(all) and returns the same keys as SinglePhotoPipeline.run, plus the per-query results.
"""
import numpy as np
import torch

from coarse_search.heading_queries import DEFAULT_HEADINGS, choose_with_heading_penalty, pool_queries, query_geometry, query_list
from header.settings import DEVICE
from image_processing.map_levels import level_for_height
from semantic_rerank.decision import decide_with_dino
from single_photo_pipeline import SinglePhotoPipeline, STAGES
from train.coarse_methods import build_methods


class HeadingQueryPipeline(SinglePhotoPipeline):
    def __init__(self, *args, headings=None, scale_factors=(1.0,), **kwargs):
        super().__init__(*args, **kwargs)
        self.headings = list(DEFAULT_HEADINGS if headings is None else headings)
        self.scale_factors = list(scale_factors)
        self.queries = query_list(self.headings, self.scale_factors)

    @classmethod
    def from_pipeline(cls, base, headings=None, scale_factors=(1.0,), k_peaks=None, m_fine=None, coarse=None):
        """A heading pipeline that uses the parts an already set-up SinglePhotoPipeline holds (photos, map, models, matcher, DINO): nothing is loaded
        again. k_peaks / m_fine: K and M per query (None = as in `base`). coarse: None = the coarse method of `base`; "edge_zncc" = the cheap one-map
        method, even if `base` uses the fused one (the heading queries are then 9 times the cost of one query, not of six maps)."""
        assert base.ready, "the base pipeline must be set up first"
        assert coarse in (None, "edge_zncc", base.coarse_name), "coarse must be None, 'edge_zncc' or the coarse method of the base pipeline"
        self = cls.__new__(cls)
        self.__dict__.update(base.__dict__)            # shares the big objects (map, encoder, matcher); nothing is copied
        if k_peaks is not None:
            self.k_peaks = k_peaks
        if m_fine is not None:
            self.m_fine = m_fine
        assert 1 <= self.m_fine <= self.k_peaks, "need 1 <= M <= K, got M=%s K=%s" % (self.m_fine, self.k_peaks)
        if coarse == "edge_zncc" and base.coarse_name != "edge_zncc":
            self.coarse = "edge_zncc"
            self.coarse_name = "edge_zncc"
            self.names = ["edge_zncc"]
            self.coarse_method = build_methods(["edge_zncc"], fused_models={})["edge_zncc"]
        self.headings = list(DEFAULT_HEADINGS if headings is None else headings)
        self.scale_factors = list(scale_factors)
        self.queries = query_list(self.headings, self.scale_factors)
        return self

    def describe(self):
        lines = super().describe()
        lines.append("queries per photo: %d (headings %s, scale factors %s); K and M are per query" % (len(self.queries), self.headings, self.scale_factors))
        return lines

    def run_queries(self, item, prior_m):
        """Every query of the photo through the pipeline up to the fine stage. Returns dict(level, queries): one dict per query with its id, heading,
        scale_factor, peaks, z, stage2, seconds and valid_fraction."""
        assert self.ready, "call setup() first"
        assert len(prior_m) == 2 and np.isfinite(prior_m).all(), "prior_m must be (x east, y south) in metres"
        assert item["rel_height"] > 0, "photo height must be above 0 m"
        level = level_for_height(item["rel_height"], self.camera)
        results = []
        for query in self.queries:
            geometry = query_geometry(query["heading"], query["scale_factor"])
            found = self.candidates_for_query(item, prior_m, level, geometry)
            found.update(query)
            results.append(found)
        return dict(level=level, queries=results)

    def answer_for(self, results, ids, heading_penalty=0.0):
        """The decision made from the candidates of the queries `ids` only. heading_penalty: inliers per degree that a candidate pays for the heading of its
        query (0 = the best candidate wins whatever its heading). Returns dict(x, y, answered, inliers, chosen, z_chosen, owner, peaks, stage2):
        chosen is the row in the pooled list and owner[chosen] the query it came from."""
        peaks, stage2, z, owner = pool_queries(results, ids)
        if heading_penalty == 0:
            x, y, inliers, chosen, z_chosen, answered = self.decide_answer(peaks, stage2, z)
        else:
            assert self.use_decision, "a heading penalty needs the DINO decision (semantic_decision=True)"
            headings = [query["heading"] for query in results]
            x, y, inliers, chosen, z_chosen = choose_with_heading_penalty(peaks, stage2, z, owner, headings, self.gamma, heading_penalty)
            answered = decide_with_dino(inliers, z_chosen, self.rescue_inliers, self.rescue_z, min_inliers=self.min_inliers)
        return dict(x=float(x), y=float(y), answered=bool(answered), inliers=int(inliers), chosen=int(chosen), z_chosen=float(z_chosen),
                    owner=owner, peaks=peaks, stage2=stage2)

    def run(self, item, prior_m):
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        found = self.run_queries(item, prior_m)
        clock = self._now()
        answer = self.answer_for(found["queries"], [q["id"] for q in found["queries"]])
        decision_seconds = self._now() - clock

        seconds = {"decision": decision_seconds}
        for label in STAGES:
            if label != "decision":
                seconds[label] = 0.0
                for query in found["queries"]:
                    seconds[label] += query["seconds"][label]
        total = 0.0
        for label in STAGES:
            total += seconds[label]
        gpu_peak_mb = 0.0
        if DEVICE == "cuda":
            gpu_peak_mb = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.empty_cache()
        answer.update(level=int(found["level"]), seconds=seconds, total=total, gpu_peak_mb=gpu_peak_mb, queries=found["queries"])
        return answer

"""The whole pipeline behind one call: one photo in, one answer out. Set up once for a data set, then call run() per photo.

    pipeline = SinglePhotoPipeline("official", coarse="fused")
    pipeline.setup()                                  # slow, once: map store, feature families, models, matcher, DINO
    result = pipeline.run(item, prior_m)              # per photo: render, score maps, coarse, DINO layer, fine, decision

Nothing in run() sees the true position. The caller (a notebook, a flight loop) compares the answer with the truth if it has one.
Every stage of run() is timed (the GPU is synchronised before each reading, so the seconds are real).
Import this file before anything else changes the settings: it imports photo_sets first, which saves the development-set defaults.
"""
import json
import time

import cv2
import numpy as np
import torch

from train.datasets import photo_sets
from header import map_source, settings
from header.settings import DEVICE, SEED
from train import config

from coarse_search.fusion import load_model
from coarse_search.search import pick_peaks
from fine_matching.matcher import load_matcher
from fine_matching.second_stage_any import run_fine_stage
from image_processing.geometry import render_query
from image_processing.map_levels import level_for_height
from image_processing.map_store import build_map_store
from result_thresholding.decision import decide, final_answer
from semantic_rerank.decision import candidate_z, choose_candidate, decide_with_dino
from semantic_rerank import dino_rerank
from semantic_rerank.dino_rerank import coarse_top, load_encoder, semantic_layer
from train.coarse_methods import build_methods
from train.coarse_methods.inputs import make_inputs
from train.features.edge import EdgeFamily
from train.features.learned_family import LearnedFamily
from train.features.mind import MindFamily

OFFICIAL_GEOMETRY = dict(settings.GEOMETRY)      # the development-set projection values, saved at import (before NUDT changes them)

STAGES = ["render", "maps", "coarse", "semantic", "fine", "decision"]


class SinglePhotoPipeline:
    def __init__(self, data_name="official", coarse="fused", semantic_layer=True, semantic_decision=True,
                 gamma=1.0, rescue_inliers=None, rescue_z=None, minima_file=None, k_peaks=50, m_fine=25, matcher="lightglue",
                 min_inliers=None, ransac_pixels=None, matcher_confidence=0.2, satellite=None, map_margin_m=None, loftr_coarse_threshold=None, loftr_upscale=1.0,
                 dino_weight=None, geometry=None):
        assert data_name in ("official", "nudt"), "data_name must be 'official' or 'nudt', got %r" % (data_name,)
        assert coarse in ("fused", "edge_zncc"), "coarse must be 'fused' or 'edge_zncc', got %r" % (coarse,)
        assert gamma >= 0, "gamma must be >= 0"
        assert 1 <= m_fine <= k_peaks, "need 1 <= m_fine (M) <= k_peaks (K), got M=%s K=%s" % (m_fine, k_peaks)
        assert matcher in ("lightglue", "loftr_minima", "roma_minima"), "matcher must be 'lightglue', 'loftr_minima' or 'roma_minima', got %r" % (matcher,)
        assert (rescue_inliers is None) == (rescue_z is None), "give rescue_inliers and rescue_z together"
        assert satellite is None or satellite in map_source.MAPS, "satellite must be one of %s (or None for %r), got %r" % (list(map_source.MAPS), map_source.DEFAULT_MAP, satellite)
        assert min_inliers is None or min_inliers >= 1, "min_inliers must be >= 1 (or None for the settings value)"
        assert ransac_pixels is None or ransac_pixels > 0, "ransac_pixels must be > 0 (or None for the settings value)"
        assert 0 <= matcher_confidence < 1, "matcher_confidence must be in 0..1"
        assert loftr_coarse_threshold is None or 0 < loftr_coarse_threshold < 1, "loftr_coarse_threshold must be between 0 and 1 (or None for the kornia value 0.2)"
        assert loftr_upscale >= 1.0, "loftr_upscale must be 1 or more"
        assert dino_weight is None or dino_weight >= 0, "dino_weight must be >= 0 (or None for the dino_rerank value)"
        assert geometry is None or set(geometry) <= set(settings.GEOMETRY), "unknown geometry key: %s, choose from %s" % (set(geometry or {}) - set(settings.GEOMETRY), list(settings.GEOMETRY))
        self.data_name = data_name
        self.coarse = coarse
        self.use_layer = semantic_layer
        self.use_decision = semantic_layer and semantic_decision
        self.gamma = gamma
        self.rescue_inliers = rescue_inliers
        self.rescue_z = rescue_z
        self.minima_file = minima_file
        self.k_peaks = k_peaks              # K: places the coarse search returns (the shortlist for the DINO layer)
        self.m_fine = m_fine                # M <= K: places sent to the fine stage (after the DINO layer: the top M; without it: the first M)
        self.matcher_name = matcher         # "lightglue" (SuperPoint + LightGlue) or "roma_minima" (RoMa with the MINIMA weights)
        self.min_inliers = settings.ANSWER_MIN_INLIERS if min_inliers is None else min_inliers      # the answer bar: answer when the chosen place has at least this many inliers
        self.ransac_pixels = settings.RANSAC_PIXELS if ransac_pixels is None else ransac_pixels      # a match counts as an inlier when it is within this many pixels of the fitted transform
        self.satellite = map_source.DEFAULT_MAP if satellite is None else satellite      # the base map of the official photos: a name from header.map_source.MAPS (the NUDT photos bring their own map)
        self.map_margin_m = map_margin_m                # metres of map around the photos (None = map_source.MARGIN_M); smaller = faster set-up, map edge closer to the photos
        self.loftr_coarse_threshold = loftr_coarse_threshold    # LoFTR only: its own cut on the coarse match probability (None = 0.2); lower = more, weaker matches
        self.loftr_upscale = loftr_upscale              # LoFTR only: both pictures are enlarged by this factor first (1.0 = not; 1.5 and more were measured worse)
        self.matcher_confidence = matcher_confidence    # LoFTR only: matches below this confidence are dropped before RANSAC
        self.dino_weight = dino_rerank.WEIGHT if dino_weight is None else dino_weight    # DINO's share in the order inside the K shortlist: 0 = coarse order, 100 = DINO decides
        self.geometry = dict(geometry or {})            # overrides of settings.GEOMETRY for the photo-to-map projection: scale, yaw_offset, height_offset (metres), pitch_sign, roll_sign, attitude
        self.setup_seconds = {}
        self.ready = False

    def _now(self):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        return time.time()

    def _timed(self, label, function):
        start = self._now()
        value = function()
        self.setup_seconds[label] = self._now() - start
        return value

    def setup(self, n_photos=500, seed=config.SEED):
        """Everything that depends on the data set and not on one photo. Returns the seconds each step took."""
        self.setup_seconds = {}
        settings.RANSAC_PIXELS = self.ransac_pixels

        def load_data():
            if self.data_name == "official":
                return photo_sets.load_official_photos(n_photos, seed, map_files=self.satellite, margin_m=self.map_margin_m)
            return photo_sets.load_nudt_photos(n_photos, seed)

        loaded = self._timed("photos, camera and map", load_data)
        self.test_set = loaded["test_set"]
        self.camera = loaded["camera"]
        self.satellite_map = loaded["satellite_map"]
        if self.data_name == "official":
            settings.GEOMETRY.update(OFFICIAL_GEOMETRY)         # back to the development values, so a run without overrides is not changed by an earlier one
        settings.GEOMETRY.update(self.geometry)

        self.store = self._timed("map at all zoom levels + edge features", lambda: build_map_store(self.satellite_map))
        self.level_images = self.store["levels"]

        self.names = ["edge_zncc"]
        fused_models = {}
        state_file = config.OUT_DIR / "state.json"
        if self.coarse == "fused":
            if state_file.exists():
                best = json.load(open(state_file))["results"]["fusion_best"]
                model_file = config.OUT_DIR / "models" / "fusion" / (best["tag"] + ".pt")
                model = load_model(model_file).to(DEVICE).eval()
                fused_models["fused"] = (model, best["subset"])
                self.names = list(best["subset"])
            else:
                print("no saved fusion model at %s: using edge_zncc" % state_file)
                self.coarse = "edge_zncc"
        self.coarse_name = self.coarse
        methods = build_methods(["edge_zncc"], fused_models=fused_models)
        self.coarse_method = methods[self.coarse_name]

        def load_families():
            wanted = set()
            for name in self.names:
                wanted.add(name.split("_")[0])
            families = {"edge": EdgeFamily(self.store)}
            if "mind" in wanted:
                families["mind"] = MindFamily(self.store, settings.LEVELS)
            if "learned" in wanted:
                f4_file = config.OUT_DIR / "models" / "f4.pt"
                assert f4_file.exists(), "the fusion model needs the learned features but %s does not exist" % f4_file
                families["learned"] = LearnedFamily(self.store, settings.LEVELS, f4_file)
            return families

        self.families = self._timed("feature families (edge, MIND, learned)", load_families)

        def load_fine_matcher():
            checkpoints = config.find_nudt_dir() / "checkpoint"
            if self.matcher_name == "loftr_minima":
                from fine_matching.loftr_matcher import load_loftr_matcher
                return load_loftr_matcher(checkpoints / "MINIMA" / "minima_loftr.ckpt", self.matcher_confidence, self.loftr_upscale, self.loftr_coarse_threshold)
            if self.matcher_name == "roma_minima":
                from fine_matching.roma_matcher import load_roma_matcher
                return load_roma_matcher(checkpoints / "MINIMA" / "minima_roma.pth", checkpoints / "RoMa" / "dinov2_vitl14_pretrain.pth")
            if self.minima_file is not None:
                settings.MINIMA_FILE = self.minima_file
            else:
                settings.MINIMA_FILE = checkpoints / "MINIMA" / "minima_lightglue.pth"
            return load_matcher()

        if self.matcher_name == "loftr_minima":
            matcher_label = "fine matcher (LoFTR + MINIMA weights)"
        elif self.matcher_name == "roma_minima":
            matcher_label = "fine matcher (RoMa + MINIMA weights)"
        else:
            matcher_label = "keypoint matcher (SuperPoint + LightGlue)"
        self.matcher = self._timed(matcher_label, load_fine_matcher)
        if self.use_layer:
            self.encoder = self._timed("DINO encoder", load_encoder)
        self.ready = True
        return dict(self.setup_seconds)

    def describe(self):
        """A few plain lines about what was set up."""
        lines = []
        lines.append("data: %s | %d photos available | zoom levels %s" % (self.data_name, len(self.test_set.items), list(settings.LEVELS)))
        lines.append("photos: %s | map: %s" % (self.data_name, self.satellite if self.data_name == "official" else "NUDT reference map"))
        lines.append("coarse method: %s | maps: %s" % (self.coarse_name, self.names))
        if self.use_layer:
            lines.append("semantic layer: on (best K=%d -> M=%d) | DINO in the decision: %s (gamma %.2f, rescue %s)" % (
                self.k_peaks, self.m_fine, "on" if self.use_decision else "off", self.gamma, (self.rescue_inliers, self.rescue_z)))
        else:
            lines.append("semantic layer: off (the first M=%d of the K=%d coarse peaks go to the fine stage)" % (self.m_fine, self.k_peaks))
        lines.append("fine matcher: %s | answer bar: %d inliers | RANSAC %.1f px | matcher confidence %.2f | LoFTR coarse threshold %s, upscale %.1f" % (self.matcher_name, self.min_inliers, self.ransac_pixels, self.matcher_confidence, self.loftr_coarse_threshold if self.loftr_coarse_threshold is not None else "0.2 (default)", self.loftr_upscale))
        lines.append("DINO order weight: %s | projection overrides: %s" % (self.dino_weight, self.geometry or "none (settings.GEOMETRY)"))
        lines.append("device: %s" % DEVICE)
        return lines

    def _lap(self, seconds, label, clock):
        now = self._now()
        seconds[label] = now - clock
        return now

    def run(self, item, prior_m):
        """item: the photo record (needs rel_height, the file, the pose). prior_m: (x east, y south) of the rough hint, metres.
        Returns dict(x, y, answered, inliers, chosen, z_chosen, level, peaks, stage2, seconds, total, gpu_peak_mb)."""
        assert self.ready, "call setup() first"
        assert len(prior_m) == 2 and np.isfinite(prior_m).all(), "prior_m must be (x east, y south) in metres"
        assert item["rel_height"] > 0, "photo height must be above 0 m"
        seconds = {}
        cv2.setRNGSeed(SEED)
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        clock = self._now()

        level = level_for_height(item["rel_height"], self.camera)
        canvas, valid = render_query(item, self.camera, level)
        clock = self._lap(seconds, "render", clock)

        inputs = make_inputs(canvas, valid, level, prior_m, self.families, self.names)
        clock = self._lap(seconds, "maps", clock)

        score_map = self.coarse_method(inputs)
        if self.use_layer:
            clock = self._lap(seconds, "coarse", clock)
            peaks, details = semantic_layer(self.encoder, score_map, inputs["geometry"], canvas, valid, level, self.level_images,
                                           shortlist=self.k_peaks, keep=self.m_fine, weight=self.dino_weight)
            z = candidate_z(peaks, details)
            clock = self._lap(seconds, "semantic", clock)
        else:
            peaks = coarse_top(score_map, inputs["geometry"], self.m_fine)
            z = np.zeros(len(peaks))
            clock = self._lap(seconds, "coarse", clock)
            seconds["semantic"] = 0.0

        stage2 = run_fine_stage(self.matcher, canvas, valid, level, peaks, self.level_images)
        clock = self._lap(seconds, "fine", clock)

        if self.use_decision:
            x, y, inliers, chosen, z_chosen = choose_candidate(peaks, stage2, z, self.gamma)
            answered = decide_with_dino(inliers, z_chosen, self.rescue_inliers, self.rescue_z, min_inliers=self.min_inliers)
        else:
            x, y, inliers, chosen = final_answer(peaks, stage2, self.min_inliers)
            answered = decide(inliers, self.min_inliers)
            z_chosen = float(z[chosen])
        self._lap(seconds, "decision", clock)

        gpu_peak_mb = 0.0
        if DEVICE == "cuda":
            gpu_peak_mb = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.empty_cache()
        total = 0.0
        for label in STAGES:
            total += seconds[label]
        return dict(x=float(x), y=float(y), answered=bool(answered), inliers=int(inliers), chosen=int(chosen), z_chosen=float(z_chosen),
                    level=int(level), peaks=peaks, stage2=stage2, seconds=seconds, total=total, gpu_peak_mb=gpu_peak_mb)

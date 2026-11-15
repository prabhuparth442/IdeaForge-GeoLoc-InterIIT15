"""Opt-in research-pipeline acceleration; same edge-ZNCC math and decision rules.

Reuse map FFTs within one photo. Do not calculate unused cosine, robust z-scores
or fusion context for edge_zncc. Fusion falls back to the original implementation.
"""
import math

import cv2
import numpy as np
import torch

import single_photo_pipeline as base
from agreement_query_pipeline import AgreementQueryPipeline
from coarse_search.correlation import fast_length
from coarse_search.search import cut_map_features
from header import settings
from image_processing.map_levels import level_gsd
from train.maps import map_bank


def cached_zncc(window, kernel, cells, cache):
    if cells.sum() <= 0:
        raise ValueError("The photo covers no usable edge cells")
    size = (fast_length(window.shape[1]), fast_length(window.shape[2]))
    if "window_fft" not in cache:
        cache["window_fft"] = torch.fft.rfft2(window, s=size)
        cache["strength_fft"] = torch.fft.rfft2((window ** 2).sum(dim=0), s=size)
    window_fft = cache["window_fft"]
    out_h = window.shape[1] - kernel.shape[1] + 1
    out_w = window.shape[2] - kernel.shape[2] + 1
    n_cells = cells.sum()
    cells_fft = torch.fft.rfft2(cells, s=size)
    energy = torch.fft.irfft2(cache["strength_fft"] * cells_fft.conj(), s=size)[:out_h, :out_w].clamp_min(0)
    kernel_mean = kernel.sum(dim=(1, 2), keepdim=True) / n_cells
    kernel_zero = (kernel - kernel_mean) * cells
    zero_fft = torch.fft.rfft2(kernel_zero, s=size)
    dot_zero = torch.fft.irfft2((window_fft * zero_fft.conj()).sum(dim=0), s=size)[:out_h, :out_w]
    squared_sums = torch.zeros_like(energy)
    # Keep the original summation order to avoid numerical changes to peak ranking.
    for channel in range(window.shape[0]):
        local = torch.fft.irfft2(window_fft[channel] * cells_fft.conj(), s=size)[:out_h, :out_w]
        squared_sums += local ** 2 / n_cells
    variance = (energy - squared_sums).clamp_min(0)
    score = dot_zero / (kernel_zero.pow(2).sum().sqrt() * variance.sqrt() + 1e-6)
    bad = (energy < settings.MIN_ENERGY * n_cells) | (variance < settings.MIN_VARIANCE * n_cells)
    score[bad] = -1.0
    return score


def edge_inputs(canvas, valid, level, prior_m, family, cache):
    gsd = level_gsd(level)
    cell = map_bank.CELL
    side = canvas.shape[0] // cell
    reach = map_bank.DISC_RADIUS / gsd + side * cell / 2
    width = int(math.ceil(2 * reach / cell))
    top = int(round((prior_m[1] / gsd - reach) / cell))
    left = int(round((prior_m[0] / gsd - reach) / cell))
    features = family.map_features(level)
    key = (id(features), level, top, left, width, str(map_bank.DEVICE))
    if cache.get("key") != key:
        cache.clear()
        cache["key"] = key
        piece = cut_map_features(features, top, left, width, width)
        cache["window"] = torch.from_numpy(piece).to(map_bank.DEVICE)
    query_features, query_cells = family.query(canvas, valid)
    kernel = torch.from_numpy(query_features).to(map_bank.DEVICE)
    cells = torch.from_numpy(query_cells.astype(np.float32)).to(map_bank.DEVICE)
    score = cached_zncc(cache["window"], kernel, cells, cache)
    rows = torch.arange(score.shape[0], device=map_bank.DEVICE)
    cols = torch.arange(score.shape[1], device=map_bank.DEVICE)
    y_m = ((top + rows) * cell + side * cell / 2) * gsd
    x_m = ((left + cols) * cell + side * cell / 2) * gsd
    inside = ((x_m[None, :] - float(prior_m[0])) ** 2 +
              (y_m[:, None] - float(prior_m[1])) ** 2) <= map_bank.DISC_RADIUS ** 2
    usable = (score > -1) & inside
    if not usable.any():
        raise ValueError("No usable search positions")
    geometry = dict(level=level, side=side, x_m=x_m, y_m=y_m, inside=inside)
    return dict(maps={"edge_zncc": score}, geometry=geometry, usable=usable)


class FastEdgeAgreementPipeline(AgreementQueryPipeline):
    def run(self, item, prior_m):
        self._edge_cache_codex = {}
        try:
            return super().run(item, prior_m)
        finally:
            self._edge_cache_codex.clear()

    def prepare_query(self, item, prior_m, level, geometry=None):
        if self.coarse_name != "edge_zncc" or self.names != ["edge_zncc"]:
            return super().prepare_query(item, prior_m, level, geometry)
        seconds = {}
        cv2.setRNGSeed(base.SEED)
        clock = self._now()
        canvas, valid = base.render_query(item, self.camera, level, geometry)
        clock = self._lap(seconds, "render", clock)
        cache = getattr(self, "_edge_cache_codex", {})
        inputs = edge_inputs(canvas, valid, level, prior_m, self.families["edge"], cache)
        clock = self._lap(seconds, "maps", clock)
        score_map = self.coarse_method(inputs)
        if self.use_layer:
            clock = self._lap(seconds, "coarse", clock)
            peaks, details = base.semantic_layer(
                self.encoder, score_map, inputs["geometry"], canvas, valid, level, self.level_images,
                shortlist=self.k_peaks, keep=self.m_fine, weight=self.dino_weight, batch=self.dino_batch)
            z = base.candidate_z(peaks, details)
            shortlist = details["shortlist"]
            self._lap(seconds, "semantic", clock)
        else:
            peaks = base.coarse_top(score_map, inputs["geometry"], self.m_fine)
            shortlist = peaks
            z = np.zeros(len(peaks))
            self._lap(seconds, "coarse", clock)
            seconds["semantic"] = 0.0
        return dict(canvas=canvas, valid=valid, peaks=peaks, z=z, shortlist=shortlist,
                    seconds=seconds, valid_fraction=float(valid.mean()))

"""Block: stage caching for the research runner (Test_Iteration_10, 3c), and the candidate selection it makes cheap to vary (3b).

CachedStagePipeline   RichDumpPipeline whose prepare_query (render -> maps -> coarse search -> DINO layer) is stored per (photo, level,
                      heading, scale, hint) in cache_dir, with the WHOLE coarse shortlist (K = 50 places, coarse score) and DINO's dense score
                      of every one of them. A later run with the cache only re-runs fine matching, so matcher / selection experiments cost
                      the fine stage only. `select` chooses which M shortlist places are fine-matched (default: the pipeline's DINO order).
select_dino(m)        the pipeline's choice: re-order by z(coarse) + 100 z(DINO), first m
select_mixture(n_dino, n_coarse)
                      the first n_dino of the DINO order plus the first n_coarse of the plain coarse order not already in (fixed budget
                      n_dino + n_coarse): does DINO ever drop the right place?
share_map_fft()       a patch (no src/ change) of coarse_search.correlation.correlate_everywhere that reuses the map-window FFTs between
                      queries of one photo (they do not depend on the query). Keyed on the window's content fingerprint; the headless
                      pipeline does the same ("shared map-window FFT"). unshare_map_fft() restores the original.
"""
import hashlib
import os

import cv2
import numpy as np
import torch

from header.settings import SEED
from train.checks.rich_dump import RichDumpPipeline


def select_dino(m, weight=100.0):
    from semantic_rerank.dino_rerank import rerank_peaks

    def pick(short, dense):
        return rerank_peaks(short, dense, weight, keep=m)
    return pick


def select_mixture(n_dino, n_coarse, weight=100.0):
    from semantic_rerank.dino_rerank import MISSING, rerank_peaks

    def pick(short, dense):
        first = rerank_peaks(short, dense, weight, keep=len(short))
        chosen = [row for row in first[:n_dino]]
        have = {(float(r[1]), float(r[2])) for r in chosen}
        real = short[short[:, 1] < MISSING]
        for row in real[np.argsort(-real[:, 0], kind="stable")]:
            if len(chosen) >= n_dino + n_coarse:
                break
            if (float(row[1]), float(row[2])) not in have:
                chosen.append(row); have.add((float(row[1]), float(row[2])))
        return np.array(chosen)
    return pick


class CachedStagePipeline(RichDumpPipeline):
    def __init__(self, *args, cache_dir=None, select=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_dir = cache_dir
        self.select = select

    def _key(self, item, prior_m, level, geometry):
        g = geometry or {}
        raw = "%s|%d|%s|%.1f|%.1f" % (os.path.basename(str(item["name"])), level, sorted(g.items()), prior_m[0], prior_m[1])
        return hashlib.sha1(raw.encode()).hexdigest()[:20]

    def prepare_query(self, item, prior_m, level, geometry=None):
        from image_processing.geometry import render_query
        from semantic_rerank.decision import candidate_z
        from semantic_rerank.dino_rerank import dino_candidate_scores, shortlist_peaks
        from train.coarse_methods.inputs import make_inputs
        assert self.use_layer, "the cache stores the DINO layer's scores"
        path = os.path.join(self.cache_dir, self._key(item, prior_m, level, geometry) + ".npz") if self.cache_dir else None
        if path and os.path.exists(path):
            c = np.load(path)
            canvas, valid, short, dense, pooled = c["canvas"], c["valid"], c["shortlist"], c["dense"], c["pooled"]
            seconds = dict(cached=1.0)
        else:
            seconds = {}
            cv2.setRNGSeed(SEED)
            clock = self._now()
            canvas, valid = render_query(item, self.camera, level, geometry)
            clock = self._lap(seconds, "render", clock)
            inputs = make_inputs(canvas, valid, level, prior_m, self.families, self.names)
            clock = self._lap(seconds, "maps", clock)
            score_map = self.coarse_method(inputs)
            clock = self._lap(seconds, "coarse", clock)
            short = shortlist_peaks(score_map, inputs["geometry"], self.k_peaks)
            pooled, dense = dino_candidate_scores(self.encoder, canvas, valid, self.level_images, level, short, self.dino_batch)
            self._lap(seconds, "semantic", clock)
            if path:
                os.makedirs(self.cache_dir, exist_ok=True)
                np.savez_compressed(path, canvas=canvas, valid=valid, shortlist=short, dense=dense, pooled=pooled)
        pick = self.select or select_dino(self.m_fine, self.dino_weight)
        peaks = pick(short, dense)
        details = dict(shortlist=short, dense=dense, pooled=pooled)
        z = candidate_z(peaks, details)
        return dict(canvas=canvas, valid=valid, peaks=peaks, z=z, shortlist=short, dense=dense, seconds=seconds, valid_fraction=float(np.mean(valid)))


_ORIGINAL = {}


def share_map_fft():
    """Patch correlate_everywhere so the window FFTs (window_fft, strength_fft) are computed once per map window."""
    import coarse_search.correlation as corr
    import coarse_search.search as search
    if "fn" in _ORIGINAL:
        return
    _ORIGINAL["fn"] = corr.correlate_everywhere
    store = {}

    def fingerprint(window, size):
        flat = window.reshape(-1)
        step = max(1, flat.numel() // 4096)
        return (tuple(window.shape), size, float(flat[::step].double().sum()), float(window.double().sum()))

    def correlate_shared(window, kernel, cells, min_energy=corr.settings.MIN_ENERGY, min_variance=corr.settings.MIN_VARIANCE):
        n_bins, height, width = window.shape
        kernel_h, kernel_w = kernel.shape[1], kernel.shape[2]
        size = (corr.fast_length(height), corr.fast_length(width))
        out_h, out_w = height - kernel_h + 1, width - kernel_w + 1
        n_cells = cells.sum()
        key = fingerprint(window, size)
        if key not in store:
            store.clear()                                     # one map window at a time (memory)
            store[key] = (torch.fft.rfft2(window, s=size), torch.fft.rfft2((window ** 2).sum(dim=0), s=size))
        window_fft, strength_fft = store[key]
        cells_fft = torch.fft.rfft2(cells, s=size)
        kernel_fft = torch.fft.rfft2(kernel, s=size)
        dot = torch.fft.irfft2((window_fft * kernel_fft.conj()).sum(dim=0), s=size)[:out_h, :out_w]
        energy = torch.fft.irfft2(strength_fft * cells_fft.conj(), s=size)[:out_h, :out_w].clamp_min(0)
        cosine = dot / (kernel.pow(2).sum().sqrt() * energy.sqrt() + 1e-6)
        kernel_mean = kernel.sum(dim=(1, 2), keepdim=True) / n_cells
        kernel_zero = (kernel - kernel_mean) * cells
        zero_fft = torch.fft.rfft2(kernel_zero, s=size)
        dot_zero = torch.fft.irfft2((window_fft * zero_fft.conj()).sum(dim=0), s=size)[:out_h, :out_w]
        squared_sums = torch.zeros_like(energy)
        for b in range(n_bins):
            local_sum = torch.fft.irfft2(window_fft[b] * cells_fft.conj(), s=size)[:out_h, :out_w]
            squared_sums += local_sum ** 2 / n_cells
        variance = (energy - squared_sums).clamp_min(0)
        zncc = dot_zero / (kernel_zero.pow(2).sum().sqrt() * variance.sqrt() + 1e-6)
        no_map = energy < min_energy * n_cells
        too_flat = variance < min_variance * n_cells
        cosine[no_map] = -1.0
        zncc[no_map | too_flat] = -1.0
        return {"cosine": cosine, "zncc": zncc}

    corr.correlate_everywhere = correlate_shared
    search.correlate_everywhere = correlate_shared


def unshare_map_fft():
    import coarse_search.correlation as corr
    import coarse_search.search as search
    if "fn" in _ORIGINAL:
        corr.correlate_everywhere = search.correlate_everywhere = _ORIGINAL.pop("fn")

"""Fusing several score maps into one: the learned fusion function (patch v0.5.5, idea C2).

Three fusion models, from simple to flexible. All three are MONOTONE in every score map
(a higher similarity can never lower the fused score), which keeps them well-behaved on data unlike the training data.

  "linear"    S = sum_k w_k * z_k                              w_k >= 0
  "monotone"  S = sum_k g_k(z_k)                               g_k a monotone piecewise-linear curve per map
  "gated"     S = sum_k a_k(context) * g_k(z_k)                a_k >= 0 chosen by a small neural net from photo-level context

z_k is the per-photo robust z-score of score map k (median/MAD over the search disc).
The context describes the photo (how peaked each map is, the ground pixel size), so the gate can learn
"trust the edge map on this kind of photo, the other map on that kind".
"""
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from coarse_search.correlation import correlate_everywhere
from coarse_search.search import cut_map_features, pick_peaks
from header import settings
from header.settings import CELL, DEVICE, DISC_RADIUS
from image_processing.edge_features import query_edge_features
from image_processing.map_levels import level_gsd

MAP_NAMES = ["zncc", "cosine", "mag"]          # the score maps built from the edge features (dino can be added)
Z_FLOOR = -10.0                                # z value written where a map has no valid score
KNOTS = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 12.0]


# ---------------------------------------------------------------- per-photo normalisation

def robust_z(score_map, usable):
    """(s - median) / (1.4826 * MAD) over the usable positions; unusable positions get Z_FLOOR."""
    assert score_map.shape == usable.shape, "score_map and usable must have the same shape"
    assert usable.any(), "no usable position in the score map"
    values = score_map[usable]
    median = values.median()
    mad = (values - median).abs().median()
    z = (score_map - median) / (1.4826 * mad + 1e-6)
    z = torch.where(usable, z, torch.full_like(z, Z_FLOOR))
    return z


def correlation_maps_multi(map_features, level, query_features, query_cells, prior_m, names=MAP_NAMES):
    """Like search.correlation_maps, but returns several score maps from one window: zncc and cosine on the 8 edge
    layers, and 'mag' = zncc on the single layer 'total edge strength'."""
    assert level in map_features, "no map features for level %s" % level
    gsd = level_gsd(level)
    side = query_features.shape[1]
    reach = DISC_RADIUS / gsd + side * CELL / 2
    cells_wide = int(math.ceil(2 * reach / CELL))
    top = int(round((prior_m[1] / gsd - reach) / CELL))
    left = int(round((prior_m[0] / gsd - reach) / CELL))
    window = torch.from_numpy(cut_map_features(map_features[level], top, left, cells_wide, cells_wide)).to(DEVICE)
    kernel = torch.from_numpy(query_features).to(DEVICE)
    cells = torch.from_numpy(query_cells.astype(np.float32)).to(DEVICE)
    out = correlate_everywhere(window, kernel, cells)
    maps = {"zncc": out["zncc"], "cosine": out["cosine"]}
    if "mag" in names:
        mag = correlate_everywhere(window.sum(0, keepdim=True), kernel.sum(0, keepdim=True), cells)
        maps["mag"] = mag["zncc"]
    shape = maps["zncc"].shape
    row_ids = torch.arange(shape[0], device=DEVICE)
    col_ids = torch.arange(shape[1], device=DEVICE)
    y_m = ((top + row_ids) * CELL + side * CELL / 2) * gsd
    x_m = ((left + col_ids) * CELL + side * CELL / 2) * gsd
    inside = ((x_m[None, :] - float(prior_m[0])) ** 2 + (y_m[:, None] - float(prior_m[1])) ** 2) <= DISC_RADIUS ** 2
    geometry = dict(level=level, side=side, x_m=x_m, y_m=y_m, inside=inside)
    return {n: maps[n] for n in names}, geometry


def z_maps_and_context(maps, geometry, names=MAP_NAMES):
    """maps: dict name -> (R, C) score map. Returns z (R, C, K) tensor and the context vector (2K+1,).
    Context = for each map: peak z and (peak z - runner-up z after suppression); plus log of the ground pixel size."""
    usable = (maps["zncc"] > -1) & geometry["inside"]
    zs, context = [], []
    gap = max(settings.TILE // CELL, geometry["side"] // 2)
    for name in names:
        z = robust_z(maps[name], usable)
        zs.append(z)
        peaks = nms_peaks(z, usable, gap_cells=gap, n=2)
        first = peaks[0][2] if len(peaks) > 0 else Z_FLOOR
        second = peaks[1][2] if len(peaks) > 1 else Z_FLOOR
        context += [first, first - second]
    context.append(math.log(level_gsd(geometry["level"])))
    return torch.stack(zs, dim=-1), np.array(context, dtype=np.float32), usable


def nms_peaks(score, usable, gap_cells, n):
    """Greedy peak picking with a blanked square around each pick. Returns [(row, col, value)]."""
    work = torch.where(usable, score, torch.full_like(score, -1e9))
    peaks = []
    width = score.shape[1]
    for _ in range(n):
        value, flat = work.reshape(-1).max(0)
        if float(value) < -1e8:
            break
        row, col = int(flat) // width, int(flat) % width
        peaks.append((row, col, float(value)))
        work[max(row - gap_cells, 0):row + gap_cells + 1, max(col - gap_cells, 0):col + gap_cells + 1] = -1e9
    return peaks


# ---------------------------------------------------------------- the models

class MonotoneCurves(nn.Module):
    """One monotone increasing curve per map: g(z) = s*z + sum_j p_j*relu(z-t_j) - sum_j q_j*relu(t_j-z), s,p,q >= 0."""

    def __init__(self, n_maps):
        super().__init__()
        self.register_buffer("knots", torch.tensor(KNOTS))
        self.slope = nn.Parameter(torch.full((n_maps,), 0.5413))        # softplus(0.5413) = 1
        self.up = nn.Parameter(torch.full((n_maps, len(KNOTS)), -5.0))
        self.down = nn.Parameter(torch.full((n_maps, len(KNOTS)), -5.0))

    def forward(self, z):                                                # z (..., K) -> (..., K)
        t = self.knots
        up = F.relu(z[..., None] - t)
        down = -F.relu(t - z[..., None])
        return F.softplus(self.slope) * z + (F.softplus(self.up) * up).sum(-1) + (F.softplus(self.down) * down).sum(-1)


class FusionModel(nn.Module):
    def __init__(self, kind, n_maps, n_context=0, hidden=16):
        super().__init__()
        assert kind in ("linear", "monotone", "gated"), "kind must be linear, monotone or gated"
        self.kind, self.n_maps, self.n_context = kind, n_maps, n_context
        if kind == "linear":
            self.weight = nn.Parameter(torch.full((n_maps,), 0.5413))
        else:
            self.curves = MonotoneCurves(n_maps)
        if kind == "gated":
            assert n_context > 0, "the gated model needs context features"
            self.gate = nn.Sequential(nn.Linear(n_context, hidden), nn.Tanh(), nn.Linear(hidden, n_maps))
            nn.init.zeros_(self.gate[2].weight)
            nn.init.constant_(self.gate[2].bias, 0.5413)
            self.register_buffer("context_mean", torch.zeros(n_context))
            self.register_buffer("context_std", torch.ones(n_context))

    def set_context_scale(self, contexts):
        self.context_mean.copy_(torch.as_tensor(contexts.mean(0)))
        self.context_std.copy_(torch.as_tensor(contexts.std(0) + 1e-3))

    def forward(self, z, context=None):
        """z: (..., n, K); context: (..., C) or None. Returns (..., n)."""
        assert z.shape[-1] == self.n_maps, "z must have %d maps, got %d" % (self.n_maps, z.shape[-1])
        if self.kind == "linear":
            return (z * F.softplus(self.weight)).sum(-1)
        curves = self.curves(z)
        if self.kind == "monotone":
            return curves.sum(-1)
        assert context is not None, "the gated model needs the photo context"
        gate = F.softplus(self.gate((context - self.context_mean) / self.context_std))      # (..., K) >= 0
        return (curves * gate[..., None, :]).sum(-1)

    def describe(self):
        if self.kind == "linear":
            return "weights " + str(np.round(F.softplus(self.weight).detach().numpy(), 3).tolist())
        n = sum(p.numel() for p in self.parameters())
        return "%d parameters" % n


# ---------------------------------------------------------------- applying a model to full maps (run time)

@torch.no_grad()
def fuse_maps(model, z, context, chunk=2_000_000):
    """z: (R, C, K) tensor of z maps -> fused score (R, C). Done in chunks to bound memory."""
    assert z.ndim == 3 and z.shape[-1] == model.n_maps, "z must be (R, C, %d)" % model.n_maps
    flat = z.reshape(-1, model.n_maps)
    ctx = torch.as_tensor(context, dtype=torch.float32, device=z.device) if model.kind == "gated" else None
    out = torch.empty(flat.shape[0], device=z.device)
    for first in range(0, flat.shape[0], chunk):
        out[first:first + chunk] = model(flat[first:first + chunk], ctx)
    return out.reshape(z.shape[:2])


def search_frame_fused(canvas, valid, level, prior_m, store, model, names=MAP_NAMES, scale=10.0):
    """Coarse search with a fusion model. Returns the 25 peaks (25, 5) like pick_peaks. The stored score is
    sigmoid(S/scale) in (0, 1) (unusable positions stay -1), so ratio/sharpness are on that scale."""
    assert len(names) == model.n_maps, "model was trained for %d maps, names has %d" % (model.n_maps, len(names))
    qf, qc = query_edge_features(canvas, valid)
    maps, geometry = correlation_maps_multi(store["features"], level, qf, qc, prior_m, names)
    z, context, usable = z_maps_and_context(maps, geometry, names)
    fused = fuse_maps(model.eval(), z, context)
    score = torch.where(usable, torch.sigmoid(fused / scale), torch.full_like(fused, -1.0))
    return pick_peaks(score, geometry)


def save_model(model, path):
    torch.save(dict(kind=model.kind, n_maps=model.n_maps, n_context=model.n_context, state=model.state_dict()), path)


def load_model(path):
    blob = torch.load(path, map_location="cpu")
    model = FusionModel(blob["kind"], blob["n_maps"], blob["n_context"])
    model.load_state_dict(blob["state"])
    return model.eval()

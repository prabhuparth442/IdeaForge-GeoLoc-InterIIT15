"""DINO layer: DINOv2 re-orders the K places of the coarse search before the fine matcher sees them.

For each place a 224 px map window is cut, the photo's valid mask is put on it, and DINO compares it with the photo token by token.
The places are re-ordered by  z(coarse score) + DINO_WEIGHT * z(DINO score)  and the first M go on.
The z-score of the DINO score is also returned, the decision uses it as a small tie-breaker (GAMMA).
"""
import numpy as np
import torch
import torch.nn.functional as F

import settings
from geoloc.coarse_search import DEVICE, pick_peaks
from geoloc.levels import map_piece

PATCH = 14
MISSING = 1e8          # pick_peaks writes x = 1e9 for a place it could not find

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def load_encoder(weights_file):
    """DINOv2 ViT-S/14 from the local weights file (no network)."""
    import timm
    encoder = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                                pretrained_cfg_overlay=dict(file=weights_file), img_size=settings.TILE)
    encoder = encoder.to(DEVICE).eval()
    if DEVICE == "cuda":
        encoder = encoder.half()
    return encoder


@torch.no_grad()
def token_grids(encoder, pictures):
    """pictures (batch, 224, 224) floats 0..1 -> DINO tokens (batch, 384, 16, 16)."""
    x = torch.from_numpy(np.ascontiguousarray(pictures)).float().to(DEVICE).unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - IMAGE_MEAN) / IMAGE_STD
    if DEVICE == "cuda":
        x = x.half()
    tokens = encoder.get_intermediate_layers(x, n=[settings.DINO_LAYER], norm=True)[0]
    side = settings.TILE // PATCH
    tokens = tokens.float().reshape(len(pictures), side, side, -1)
    return tokens.permute(0, 3, 1, 2)


def valid_tokens(valid):
    # token cells that lie fully inside the photographed area
    mask = torch.from_numpy(np.asarray(valid, dtype=np.float32))[None, None]
    share = F.avg_pool2d(mask, PATCH)[0, 0]
    return share > 0.99


def dino_scores(encoder, canvas, valid, level_image, level, peaks):
    """One number per place: mean over valid tokens of the token-to-token cosine between photo and map window. NaN where there is no place."""
    dense = np.full(len(peaks), np.nan)
    real = []
    for i in range(len(peaks)):
        if peaks[i, 1] < MISSING:
            real.append(i)
    if len(real) == 0:
        return dense

    mask = np.asarray(valid, dtype=np.float32)
    query_tokens = token_grids(encoder, (np.asarray(canvas, dtype=np.float32) * mask)[None])[0]
    inside = valid_tokens(valid).to(query_tokens.device)
    query_unit = F.normalize(query_tokens, dim=0)

    for start in range(0, len(real), settings.DINO_BATCH):
        chunk = real[start:start + settings.DINO_BATCH]
        windows = []
        for i in chunk:
            windows.append(map_piece(level_image, level, peaks[i, 1], peaks[i, 2]) * mask)
        tokens = token_grids(encoder, np.stack(windows))
        unit = F.normalize(tokens, dim=1)
        per_token = (unit * query_unit[None]).sum(dim=1)
        for slot in range(len(chunk)):
            dense[chunk[slot]] = float(per_token[slot][inside].mean())
    return dense


def z_score(values):
    """Plain z-score over the finite values; NaN becomes 0."""
    result = np.zeros(len(values))
    good = np.isfinite(values)
    if good.sum() > 1:
        spread = values[good].std()
        if spread < 1e-9:
            spread = 1.0
        result[good] = (values[good] - values[good].mean()) / spread
    return result


def semantic_layer(encoder, score_map, geometry, canvas, valid, level, level_image):
    """Coarse score map in; the best M of the K places out, with the DINO z-score of each. Rows are [score, x, y, sharpness, rank]."""
    shortlist = pick_peaks(score_map, geometry, settings.K_PEAKS)
    dense = dino_scores(encoder, canvas, valid, level_image, level, shortlist)

    real = shortlist[:, 1] < MISSING
    coarse = np.where(real, shortlist[:, 0], np.nan)
    judge = np.where(real, dense, np.nan)
    combined = z_score(coarse) + settings.DINO_WEIGHT * z_score(judge)
    combined[~real] = -1e9
    order = np.argsort(-combined, kind="stable")[:settings.M_FINE]

    peaks = shortlist[order].copy()
    peaks[:, 4] = combined[order]
    return peaks, z_score(dense)[order]


def coarse_only(score_map, geometry):
    """Without the DINO layer: the first M places in coarse order."""
    peaks = pick_peaks(score_map, geometry, settings.M_FINE)
    return peaks, np.zeros(len(peaks))

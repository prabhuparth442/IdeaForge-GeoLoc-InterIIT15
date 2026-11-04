"""F4: learned correlation features.

Two small CNNs turn a picture into the same kind of grid as the edge features (8 numbers per 4x4 block):
  * QueryNet looks at the thermal canvas,   * MapNet looks at the satellite level picture.
They are trained so that the masked ZNCC between the two grids peaks at the true position and nowhere else
(softmax over every offset inside a +-40 cell window = +-136 m at level 3, so repeated city blocks are the negatives).
The FFT scoring code is unchanged: the learned grids simply replace the edge grids (a new family, "learned").

Training data = NUDT train photos rendered through the existing pipeline (render_query) + the map piece around the truth.
"""
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from header.settings import CELL, DEVICE, TILE
from image_processing.edge_features import edge_features, map_edge_features, normalise_cells
from image_processing.geometry import render_query
from image_processing.map_levels import level_gsd
from train import config

SIDE = TILE // CELL            # 56 cells in a query
FLOOR = 0.1                    # cell length floor in the normalisation f / (|f| + FLOOR)


def oriented_filters(count=8, size=5, sigma=1.0):
    """16 filters: for 8 directions an edge detector (1st derivative of a Gaussian) and a line detector (2nd derivative)."""
    ys, xs = np.mgrid[-(size // 2):size // 2 + 1, -(size // 2):size // 2 + 1].astype(np.float32)
    gauss = np.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2))
    filters = []
    for k in range(count):
        theta = np.pi * k / count
        u = xs * np.cos(theta) + ys * np.sin(theta)
        for f in (-u * gauss, (u ** 2 / sigma ** 2 - 1) * gauss):
            f = f - f.mean() if f is not None else f
            filters.append(f / (np.abs(f).sum() + 1e-6))
    return torch.from_numpy(np.stack(filters)[:, None])


def make_net():
    net = nn.Sequential(nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(),
                        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                        nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
                        nn.Conv2d(32, 8, 3, padding=1))
    with torch.no_grad():
        net[0].weight.copy_(oriented_filters())
        net[0].bias.zero_()
    return net


def features_from_net(net, picture):
    """picture (B, 1, H, W) -> (B, 8, H/4, W/4), each cell scaled to length ~1."""
    out = net(picture)
    return out / (out.norm(dim=1, keepdim=True) + FLOOR)


def zncc_conv(window, kernel, cells):
    """Differentiable masked ZNCC (the same formula as coarse_search.correlation, for small windows).
    window (8, H, W), kernel (8, s, s), cells (s, s) float -> (H-s+1, W-s+1)."""
    assert window.ndim == 3 and kernel.ndim == 3 and cells.ndim == 2, "window/kernel are (8, h, w), cells is (s, s)"
    n = cells.sum()
    kz = (kernel - (kernel * cells).sum((1, 2), keepdim=True) / n) * cells
    dot = F.conv2d(window[None], kz[None])[0, 0]
    ones = cells[None, None].expand(window.shape[0], 1, *cells.shape).contiguous()
    sums = F.conv2d(window[None], ones, groups=window.shape[0])[0]
    energy = F.conv2d((window ** 2).sum(0)[None, None], cells[None, None])[0, 0]
    variance = (energy - (sums ** 2).sum(0) / n).clamp_min(0)
    return dot / (kz.pow(2).sum().sqrt() * variance.sqrt() + 1e-6)


# ------------------------------------------------------------------ training pairs

def cell_mask_of(valid):
    safe = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8))
    return safe.reshape(SIDE, CELL, SIDE, CELL).mean(axis=(1, 3)) >= 0.8


def make_pairs(testset, store, ids, margin=None, log=print):
    """For each photo: the rendered canvas, its valid-cell mask, the map piece around the truth, the sub-cell offset."""
    margin = config.F4_MAP_MARGIN_CELLS if margin is None else margin
    crop_px = CELL * (SIDE + 2 * margin)
    canvases, cells, crops, fracs, levels = [], [], [], [], []
    start = time.time()
    for count, i in enumerate(ids):
        i = int(i)
        level = int(testset.query_levels[i])
        gsd = level_gsd(level)
        tx, ty = testset.truths[i][0] / gsd, testset.truths[i][1] / gsd
        cx, cy = int(round(tx)), int(round(ty))
        image = store["levels"][level]
        top, left = cy - crop_px // 2, cx - crop_px // 2
        if top < 0 or left < 0 or top + crop_px > image.shape[0] or left + crop_px > image.shape[1]:
            continue
        crop = image[top:top + crop_px, left:left + crop_px]
        if (crop == 0).mean() > 0.02:
            continue
        canvas, valid = render_query(testset.items[i], testset.camera, level)
        canvases.append((np.clip(canvas, 0, 1) * 255).astype(np.uint8)); cells.append(cell_mask_of(valid))
        crops.append(crop.copy()); fracs.append((tx - cx, ty - cy)); levels.append(level)
        if count % 200 == 0:
            log("F4 pairs: %d / %d photos, %d kept, %.0f s" % (count, len(ids), len(canvases), time.time() - start))
    assert len(canvases) > 0, "no photo had a full map piece around it"
    return dict(canvas=np.stack(canvases), cells=np.stack(cells), crop=np.stack(crops), frac=np.array(fracs, np.float32), level=np.array(levels))


def augment(canvas, crop, rng):
    """canvas, crop: float arrays in [0,1]. Thermal: polarity flip, contrast curve, noise. Map: brightness/contrast jitter."""
    if rng.random() < 0.3:
        canvas = 1.0 - canvas
    canvas = np.clip(canvas, 0, 1) ** rng.uniform(0.7, 1.5)
    canvas = np.clip(canvas + rng.normal(0, 0.02, canvas.shape), 0, 1)
    crop = np.clip((crop - 0.5) * rng.uniform(0.85, 1.15) + 0.5 + rng.uniform(-0.08, 0.08), 0, 1)
    return canvas.astype(np.float32), crop.astype(np.float32)


def batch_logits(query_net, map_net, canvas, cells, crop):
    """canvas (B,1,224,224), cells (B,56,56), crop (B,1,P,P)  ->  list of (Ho, Wo) correlation maps."""
    q = features_from_net(query_net, canvas) * cells[:, None]
    m = features_from_net(map_net, crop)
    return [zncc_conv(m[b], q[b], cells[b]) for b in range(canvas.shape[0])]


def gaussian_target(size, centre_r, centre_c, sigma=1.0):
    rr = torch.arange(size, device=DEVICE, dtype=torch.float32)[:, None]
    cc = torch.arange(size, device=DEVICE, dtype=torch.float32)[None, :]
    g = torch.exp(-((rr - centre_r) ** 2 + (cc - centre_c) ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def pair_loss_and_hit(maps, fracs, margin):
    losses, hits = [], []
    for corr, (fx, fy) in zip(maps, fracs):
        target = gaussian_target(corr.shape[0], margin + fy / CELL, margin + fx / CELL)
        logp = F.log_softmax(corr.reshape(-1) / config.F4_TEMPERATURE, 0)
        losses.append(-(target.reshape(-1) * logp).sum())
        peak = int(corr.reshape(-1).argmax())
        hits.append(abs(peak // corr.shape[1] - margin) <= 2 and abs(peak % corr.shape[1] - margin) <= 2)
    return torch.stack(losses).mean(), hits


@torch.no_grad()
def edge_baseline_hit_rate(pairs, margin, n=200):
    """How often the plain edge-feature ZNCC finds the truth (within 2 cells) on the same pairs: the number to beat."""
    hits = []
    for b in range(min(n, len(pairs["canvas"]))):
        canvas = pairs["canvas"][b] / 255.0
        cells = pairs["cells"][b]
        q = normalise_cells(edge_features(canvas), cells) * cells
        m = map_edge_features(pairs["crop"][b])
        corr = zncc_conv(torch.from_numpy(m).to(DEVICE), torch.from_numpy(q.astype(np.float32)).to(DEVICE), torch.from_numpy(cells.astype(np.float32)).to(DEVICE))
        peak = int(corr.reshape(-1).argmax())
        hits.append(abs(peak // corr.shape[1] - margin) <= 2 and abs(peak % corr.shape[1] - margin) <= 2)
    return float(np.mean(hits))


def train_features(train_pairs, valid_pairs, out_path, minutes=None, batch=None, log=print):
    """Train until the time budget is used. Saves the best (by valid hit rate) to out_path. Returns the history list."""
    minutes = config.F4_MINUTES if minutes is None else minutes
    batch = config.F4_BATCH if batch is None else batch
    margin = (train_pairs["crop"].shape[1] // CELL - SIDE) // 2
    torch.manual_seed(config.SEED)
    rng = np.random.default_rng(config.SEED)
    query_net, map_net = make_net().to(DEVICE), make_net().to(DEVICE)
    params = list(query_net.parameters()) + list(map_net.parameters())
    opt = torch.optim.Adam(params, lr=config.F4_LR)
    deadline = time.time() + minutes * 60
    start = time.time()
    n = len(train_pairs["canvas"])
    baseline = edge_baseline_hit_rate(valid_pairs, margin)
    log("F4: %d train pairs, %d valid pairs; plain edge features find the truth on %.1f %% of the valid pairs (to beat)" % (n, len(valid_pairs["canvas"]), 100 * baseline))
    history, best, step = [], -1.0, 0

    def evaluate():
        query_net.eval(); map_net.eval()
        hits, losses = [], []
        with torch.no_grad():
            for first in range(0, len(valid_pairs["canvas"]), 8):
                sl = slice(first, first + 8)
                c = torch.from_numpy(valid_pairs["canvas"][sl] / 255.0).float()[:, None].to(DEVICE)
                k = torch.from_numpy(valid_pairs["cells"][sl].astype(np.float32)).to(DEVICE)
                m = torch.from_numpy(valid_pairs["crop"][sl] / 255.0).float()[:, None].to(DEVICE)
                loss, h = pair_loss_and_hit(batch_logits(query_net, map_net, c, k, m), valid_pairs["frac"][sl], margin)
                hits += h; losses.append(float(loss))
        query_net.train(); map_net.train()
        return float(np.mean(hits)), float(np.mean(losses))

    while time.time() < deadline:
        for group in opt.param_groups:
            group["lr"] = config.F4_LR * 0.5 * (1 + np.cos(np.pi * min(1.0, (time.time() - start) / (minutes * 60))))
        idx = rng.choice(n, size=min(batch, n), replace=False)
        canvases, crops = [], []
        for b in idx:
            cv, cr = augment(train_pairs["canvas"][b] / 255.0, train_pairs["crop"][b] / 255.0, rng)
            canvases.append(cv); crops.append(cr)
        c = torch.from_numpy(np.stack(canvases))[:, None].to(DEVICE)
        m = torch.from_numpy(np.stack(crops))[:, None].to(DEVICE)
        k = torch.from_numpy(train_pairs["cells"][idx].astype(np.float32)).to(DEVICE)
        flip = rng.random() < 0.5
        if flip:                                                   # mirror query and map piece together: the answer stays at the centre
            c, m, k = c.flip(-1), m.flip(-1), k.flip(-1)
        fr = train_pairs["frac"][idx].copy()
        if flip:
            fr[:, 0] = -fr[:, 0]
        loss, _ = pair_loss_and_hit(batch_logits(query_net, map_net, c, k, m), fr, margin)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        step += 1
        if step % 100 == 0:
            hit, vloss = evaluate()
            history.append((step, float(loss), vloss, hit))
            log("F4 step %5d  train loss %.3f  valid loss %.3f  valid hit rate %.1f %% (edge: %.1f %%)  %.0f min left" % (
                step, float(loss), vloss, 100 * hit, 100 * baseline, max(0, (deadline - time.time()) / 60)))
            if hit > best:
                best = hit
                torch.save(dict(query=query_net.state_dict(), map=map_net.state_dict(), hit=hit, baseline=baseline, step=step), out_path)
    hit, vloss = evaluate()
    history.append((step, float(loss), vloss, hit))
    if hit > best:
        best = hit
        torch.save(dict(query=query_net.state_dict(), map=map_net.state_dict(), hit=hit, baseline=baseline, step=step), out_path)
    log("F4 done: best valid hit rate %.1f %% vs edge %.1f %% after %d steps" % (100 * best, 100 * baseline, step))
    return history, best, baseline

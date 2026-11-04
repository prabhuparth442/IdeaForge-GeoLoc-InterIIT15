"""C1: pre-whitening. A city grid repeats, so its power sits at a few spatial frequencies; dividing by the map's average
power spectrum (to the power alpha/2) flattens them and sharpens the true peak."""
import numpy as np
import torch

from coarse_search.correlation import fast_length
from header.settings import DEVICE

def estimate_power_profile(features, n_patches=24, size=128, bins=32, seed=0):
    """Average radial power spectrum of a feature map: returns (frequency centres, relative power), power mean = 1."""
    assert features.ndim == 3 and features.shape[1] >= size and features.shape[2] >= size, "feature map smaller than the sample patch"
    rng = np.random.default_rng(seed)
    hann = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    power = np.zeros((size, size // 2 + 1))
    used = 0
    for _ in range(n_patches * 20):
        r = rng.integers(0, features.shape[1] - size + 1)
        c = rng.integers(0, features.shape[2] - size + 1)
        patch = features[:, r:r + size, c:c + size].astype(np.float32)
        if (np.abs(patch).sum(axis=0) > 0).mean() < 0.9:
            continue
        patch = (patch - patch.mean(axis=(1, 2), keepdims=True)) * hann
        power += (np.abs(np.fft.rfft2(patch)) ** 2).sum(axis=0)
        used += 1
        if used >= n_patches:
            break
    assert used > 0, "no sample patch of the feature map was fully inside the map"
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.rfftfreq(size)[None, :]
    radius = np.sqrt(fy ** 2 + fx ** 2)
    edges = np.linspace(0, 0.5, bins + 1)
    index = np.minimum(np.digitize(radius.ravel(), edges) - 1, bins - 1)
    profile = np.bincount(index, weights=power.ravel(), minlength=bins) / np.maximum(np.bincount(index, minlength=bins), 1)
    centres = (edges[:-1] + edges[1:]) / 2
    return centres, profile / profile.mean()


_FILTER_CACHE = {}


def whitening_filter(profile, alpha, size, eps=0.02):
    """The filter H(f) = (P(f) + eps)^(-alpha/2) sampled on the rfft grid of `size` (a tensor on DEVICE)."""
    key = (id(profile), round(alpha, 4), size)
    if key not in _FILTER_CACHE:
        centres, power = profile
        fy = np.fft.fftfreq(size[0])[:, None]
        fx = np.fft.rfftfreq(size[1])[None, :]
        radius = np.sqrt(fy ** 2 + fx ** 2)
        gain = np.interp(radius.ravel(), centres, (power + eps) ** (-alpha / 2)).reshape(radius.shape)
        _FILTER_CACHE.clear()
        _FILTER_CACHE[key] = torch.from_numpy(gain.astype(np.float32)).to(DEVICE)
    return _FILTER_CACHE[key]


def whiten(x, profile, alpha, keep):
    """x (C, H, W) tensor; keep (H, W) bool tensor = where real data is. Returns the whitened x, rescaled to the old RMS."""
    assert x.ndim == 3 and keep.shape == x.shape[1:], "keep must be (H, W) of x"
    assert 0 <= alpha <= 1, "alpha must be in 0..1"
    if alpha == 0:
        return x
    size = (fast_length(x.shape[1]), fast_length(x.shape[2]))
    gain = whitening_filter(profile, alpha, size)
    out = torch.fft.irfft2(torch.fft.rfft2(x, s=size) * gain, s=size)[:, :x.shape[1], :x.shape[2]]
    out = out * keep
    before = (x[:, keep] ** 2).mean().sqrt()
    after = (out[:, keep] ** 2).mean().sqrt()
    return out * (before / (after + 1e-8))

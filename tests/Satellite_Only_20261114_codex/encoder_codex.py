"""Scratch-initialised satellite-only structural encoder; no pretrained imports."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class StructuralEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(2, 16, 5, 2, 2), nn.GroupNorm(4, 16), nn.GELU(),
                                  nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.GELU(),
                                  nn.Conv2d(32, 48, 3, 2, 1), nn.GroupNorm(4, 48), nn.GELU(),
                                  nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(48*16, 128))

    def forward(self, x):
        return F.normalize(self.body(x), dim=1)


def structural(image, mask=None):
    x = cv2.resize(np.asarray(image, np.float32), (96, 96), interpolation=cv2.INTER_AREA)
    x = cv2.GaussianBlur(x, (0, 0), .7)
    gx = np.abs(cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3))
    y = np.stack((gx, gy))
    if mask is not None:
        m = cv2.resize(mask.astype(np.uint8), (96, 96), interpolation=cv2.INTER_NEAREST)
        m = cv2.erode(m, np.ones((5, 5), np.uint8))
        y *= m[None]
    return np.clip(y / (np.percentile(y[y > 0], 90) + 1e-6) if np.any(y > 0) else y, 0, 3)


def augment(image, rng):
    n = image.shape[0]
    matrix = cv2.getRotationMatrix2D(((n-1)/2, (n-1)/2), rng.uniform(-10, 10), rng.uniform(.9, 1.1))
    matrix[:, 2] += rng.uniform(-8, 8, 2)
    x = cv2.warpAffine(image, matrix, (n, n), borderMode=cv2.BORDER_REFLECT_101)
    x = np.clip(x, 0, 1) ** rng.uniform(.5, 1.8)
    if rng.random() < .5:
        x = 1-x
    x = cv2.GaussianBlur(x, (0, 0), rng.uniform(.3, 2))
    mask = np.ones((n, n), np.uint8)
    margin = int(rng.uniform(0, .17)*n)
    if margin:
        mask[:margin] = mask[-margin:] = 0
        mask[:, :margin] = mask[:, -margin:] = 0
    if rng.random() < .3:
        side = int(rng.uniform(.05, .2)*n)
        y, z = rng.integers(0, n-side, 2)
        mask[y:y+side, z:z+side] = 0
    return structural(x, mask), mask


@torch.no_grad()
def encode(model, pictures, masks=None, batch=32):
    output = []
    for start in range(0, len(pictures), batch):
        x = np.stack([structural(pictures[i], None if masks is None else masks[i])
                      for i in range(start, min(start+batch, len(pictures)))])
        output.append(model(torch.from_numpy(x)).numpy())
    return np.concatenate(output)


def gate(cosine, margin, calibration, has_competitor=True):
    return bool(has_competitor and np.isfinite(cosine) and np.isfinite(margin)
                and cosine >= calibration['cosine'] and margin >= calibration['margin'])

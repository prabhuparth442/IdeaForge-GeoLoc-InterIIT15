"""Clean a thermal photo before it is stretched and matched: remove the lens vignetting, cover a known artefact, equalise the local contrast.

Why: on the high-altitude photos (above about 800 m) the percentile stretch spends about half of its range on a bright-centre to dark-corner pattern of the lens,
which the satellite map does not have, and the scene itself has very little contrast. The three steps are separate switches in settings.py and all off by default.

radial_surface   the fitted smooth bright-centre-to-dark-corner surface of one frame
remove_vignette  the frame minus that surface (the mean grey stays)
mask_rects       rectangles of the frame (a fixed artefact of a camera) filled with the median grey
local_normalise  grey minus its local mean, divided by its local spread (a floor keeps flat noise from being blown up)
clean            the steps the settings ask for, for a photo taken at this height
"""
import cv2
import numpy as np

import settings


def _radius_squared(shape, cx, cy):
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    dist2 = (cols - cx) ** 2 + (rows - cy) ** 2
    corners = [(0, 0), (shape[1] - 1, 0), (0, shape[0] - 1), (shape[1] - 1, shape[0] - 1)]
    far2 = max((x - cx) ** 2 + (y - cy) ** 2 for x, y in corners)
    return dist2 / far2


def radial_surface(frame, cx, cy, degree):
    """Polynomial of the squared radius (1, r2, r4, ...) fitted to the frame. Pixels far from the fit (edges, hot spots) are left out of the next fit, three rounds."""
    radius2 = _radius_squared(frame.shape, cx, cy)
    values = frame.astype(np.float64)
    basis = np.stack([radius2.ravel() ** power for power in range(degree + 1)], axis=1)
    target = values.ravel()
    keep = np.ones(target.shape, bool)
    for _ in range(3):
        coefficients, *_ = np.linalg.lstsq(basis[keep], target[keep], rcond=None)
        residual = target - basis @ coefficients
        spread = 1.4826 * np.median(np.abs(residual[keep] - np.median(residual[keep]))) + 1e-9
        keep = np.abs(residual) < 2.5 * spread
        if keep.sum() < 0.3 * keep.size:
            break
    return (basis @ coefficients).reshape(frame.shape)


def remove_vignette(frame, cx, cy, degree=3):
    surface = radial_surface(frame, cx, cy, degree)
    return (frame.astype(np.float64) - (surface - surface.mean())).astype(np.float32)


def mask_rects(frame, rects):
    out = frame.astype(np.float32).copy()
    grey = float(np.median(out))
    for x0, y0, x1, y1 in rects:
        out[y0:y1, x0:x1] = grey
    return out


def local_normalise(frame, sigma):
    values = frame.astype(np.float32)
    mean = cv2.GaussianBlur(values, (0, 0), sigma)
    variance = cv2.GaussianBlur((values - mean) ** 2, (0, 0), sigma)
    floor = (0.1 * float(values.std())) ** 2 + 1e-12
    return (values - mean) / np.sqrt(variance + floor)


def clean(frame, pose, camera):
    """The frame as the settings want it for this photo. With every switch at its default the frame comes back as it is."""
    out = frame
    gate = settings.FLAT_FIELD_FROM_HEIGHT_M
    if gate is not None and pose["height"] >= gate:
        if settings.FLAT_FIELD_DEGREE > 0:
            out = remove_vignette(out, camera["cx"], camera["cy"], settings.FLAT_FIELD_DEGREE)
        if settings.LOCAL_NORM_SIGMA is not None:
            out = local_normalise(out, settings.LOCAL_NORM_SIGMA)
    if settings.PHOTO_MASK_RECTS:
        out = mask_rects(out, settings.PHOTO_MASK_RECTS)
    return out

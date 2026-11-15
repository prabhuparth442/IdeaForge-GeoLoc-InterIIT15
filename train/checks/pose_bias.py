"""Block: a constant attitude bias in the final position, fitted on answered dev photos and removed (no GPU, no plotting).

Model (Patch v0.5.6 revised): the signed error of an answer, in metres on the map, is
    e = h * R(yaw) @ (a_f, a_r)  +  h * R(yaw) @ (k_p * tan(pitch), k_r * tan(roll))  +  b
- a_f, a_r: a constant forward / lateral angle (a camera mount or IMU offset, or a roll/pitch sign or convention error), in radians
- k_p, k_r: a scale error on the measured pitch / roll (k = -2 would mean the sign is flipped)
- b: a constant map offset in metres
R(yaw) turns the body frame into the map frame. The map frame here is x east, y south; the handedness and the 180 degree yaw offset are absorbed
by fitting both rotation senses (`mirror`) and keeping the better one on the training photos.

design        the regression matrix for chosen terms
fit / apply   least squares (on photos with error <= max_sigma, so wrong places do not pull the fit) and the corrected positions
"""
import math

import numpy as np

TERMS_ALL = ("angle", "tilt", "const")


def _rot(yaw_deg, mirror):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, -s], [s, c]]) if not mirror else np.array([[c, s], [-s, c]])


def design(h, yaw, pitch, roll, terms=TERMS_ALL, mirror=False):
    """Rows 2n x p: for photo i the x row then the y row."""
    blocks = []
    for hi, yi, pi, ri in zip(h, yaw, pitch, roll):
        R = _rot(yi, mirror)
        cols = []
        if "angle" in terms:
            cols += [hi * R[:, 0], hi * R[:, 1]]
        if "tilt" in terms:
            cols += [hi * math.tan(math.radians(pi)) * R[:, 0], hi * math.tan(math.radians(ri)) * R[:, 1]]
        if "const" in terms:
            cols += [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        blocks.append(np.stack(cols, axis=1))
    return np.concatenate(blocks, axis=0)


def fit(df, terms=TERMS_ALL, max_sigma=4.0):
    """df columns: ex, ey (signed error, metres), err_sigma, h, yaw, pitch, roll. Returns dict(coef, mirror, terms, rms_before, rms_after)."""
    sel = df[df.err_sigma <= max_sigma]
    assert len(sel) >= 10, "need at least 10 answered photos within max_sigma to fit"
    e = np.stack([sel.ex.to_numpy(), sel.ey.to_numpy()], axis=1).reshape(-1)
    best = None
    for mirror in (False, True):
        A = design(sel.h, sel.yaw, sel.pitch, sel.roll, terms, mirror)
        coef, *_ = np.linalg.lstsq(A, e, rcond=None)
        rms = float(np.sqrt(np.mean((e - A @ coef) ** 2)))
        if best is None or rms < best["rms_after"]:
            best = dict(coef=coef, mirror=mirror, terms=terms, rms_before=float(np.sqrt(np.mean(e ** 2))), rms_after=rms)
    return best


def apply(df, model):
    """Corrected signed errors (ex, ey) for every row of df."""
    A = design(df.h, df.yaw, df.pitch, df.roll, model["terms"], model["mirror"])
    pred = (A @ model["coef"]).reshape(-1, 2)
    return df.ex.to_numpy() - pred[:, 0], df.ey.to_numpy() - pred[:, 1]

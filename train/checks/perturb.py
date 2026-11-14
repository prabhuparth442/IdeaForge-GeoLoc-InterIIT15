"""Block: altered copies of a drone photo and of its metadata, to red-team the pipeline and the decision on inputs unlike the dev photos.
No plotting. The pipeline reads a photo through src/image_processing/thermal.load_thermal_frame(item), which joins item["name"] with the
images folder unless the name is an absolute path; perturbed_item writes the altered frame to its own folder and returns a copy of the item
pointing at it, so nothing in src/ changes.

Each alteration is something a new drone, camera or flight could plausibly bring (the score only checks the answer, so these test what the
method does when the input changes, not how a human would see it):
- invert     thermal polarity flipped (hot <-> cold: day vs night, white-hot vs black-hot palette)
- blur       Gaussian blur, sigma 2 px (defocus, motion, a softer lens)
- noise      Gaussian noise, 4 % of the 1-99 % range (a noisier sensor)
- lowres     down to half size and back up (half the ground resolution, as from twice the height, same footprint)
- gamma      tone curve x ** 0.5 on the 1-99 % range (a different automatic gain / palette)
- yaw15      heading metadata off by +15 deg (outside the +-10 deg the queries cover)
- height20   height metadata 20 % too high (scale outside the 0.95-1.05 the queries cover)
The truth and sigma used for scoring are never altered.
"""
import os

import cv2
import numpy as np

KINDS = ["invert", "blur", "noise", "lowres", "gamma", "yaw15", "height20"]
IMAGE_KINDS = {"invert", "blur", "noise", "lowres", "gamma"}


def perturb_frame(frame, kind, seed=0):
    """frame: raw 2-D array (uint8 / uint16 / float). Returns an array of the same dtype and shape."""
    dtype = frame.dtype
    f = frame.astype(np.float64)
    lo, hi = np.percentile(f, [1, 99])
    span = max(hi - lo, 1e-6)
    if kind == "invert":
        g = (f.max() + f.min()) - f
    elif kind == "blur":
        g = cv2.GaussianBlur(f, (0, 0), 2.0)
    elif kind == "noise":
        g = f + np.random.default_rng(seed).normal(0, 0.04 * span, f.shape)
    elif kind == "lowres":
        h, w = f.shape
        g = cv2.resize(cv2.resize(f, (w // 2, h // 2), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    elif kind == "gamma":
        g = lo + span * np.clip((f - lo) / span, 0, 1) ** 0.5
    else:
        raise ValueError(kind)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        g = np.clip(np.rint(g), info.min, info.max)
    return g.astype(dtype)


def perturbed_item(item, kind, out_dir, load_frame, seed=0):
    """A copy of item for `kind`. Image kinds: the altered frame is written (lossless PNG, same dtype) to out_dir and the copy points at it.
    Metadata kinds: yaw or rel_height of the copy is changed. load_frame: image_processing.thermal.load_thermal_frame."""
    new = dict(item)
    if kind in IMAGE_KINDS:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.abspath(os.path.join(out_dir, "%s_%s.png" % (os.path.splitext(os.path.basename(str(item["name"])))[0], kind)))
        if not os.path.exists(path):
            assert cv2.imwrite(path, perturb_frame(load_frame(item), kind, seed)), "could not write " + path
        new["name"] = path
    elif kind == "yaw15":
        new["yaw"] = float(item["yaw"]) + 15.0
    elif kind == "height20":
        new["rel_height"] = float(item["rel_height"]) * 1.2
    else:
        raise ValueError(kind)
    return new

"""Optional rescue of a skipped photo: run it again on altered copies of itself (settings.USE_ALTERED_COPIES, off by default).

Two copies are made from the raw thermal frame, before the pipeline stretches and equalises it:
  "invert"   the polarity flipped (hot becomes cold). A drone that flies with the other palette gives a photo that is the negative of the map's look.
  "sharpen"  an unsharp mask (sigma 2 px, amount 1.5). A soft or defocused photo gets its edges back.
Research repo, Idea5b: 60 official photos x clean / inverted / blurred, first pass only. The copies recovered 14 of 18 skipped inverted photos and 11 of 31
skipped blurred ones, with no wrong answer from a copy.
"""
import cv2
import numpy as np

KINDS = ("invert", "sharpen")
SHARPEN_SIGMA = 2.0
SHARPEN_AMOUNT = 1.5


def keep_type(values, original):
    """Back to the data type of the raw frame (whole numbers for 8 and 16 bit frames)."""
    if np.issubdtype(original.dtype, np.integer):
        limits = np.iinfo(original.dtype)
        values = np.rint(values)
        values = np.clip(values, limits.min, limits.max)
    return values.astype(original.dtype)


def invert_polarity(frame):
    values = frame.astype(np.float64)
    flipped = (values.max() + values.min()) - values
    return keep_type(flipped, frame)


def sharpen(frame):
    values = frame.astype(np.float64)
    soft = cv2.GaussianBlur(values, (0, 0), SHARPEN_SIGMA)
    sharper = values + SHARPEN_AMOUNT * (values - soft)
    return keep_type(sharper, frame)


def make_copy(frame, kind):
    if kind == "invert":
        return invert_polarity(frame)
    if kind == "sharpen":
        return sharpen(frame)
    raise ValueError("unknown copy kind %r, use one of %s" % (kind, KINDS))

"""Camera chain after the optics: blur, noise, stripes, auto gain, optional flattening, vignette, local enhancement (clahe) and grain, 8-bit quantisation. Output pixel values are NOT temperatures."""
import cv2
import numpy as np

DEFAULT_SENSOR = dict(blur_sigma_px=0.7, noise=0.01, stripe=0.004, bits=8, vignette=0.0, contrast=1.0, clahe=0.0, grain=0.0, gamma=1.0)


def apply_sensor(picture, valid, rng, settings_dict=None):
    """picture: float thermal-like image; valid: bool mask. Returns uint8 image (what the camera would save).

    Invalid pixels get the mean of the valid ones, so the automatic gain only reacts to what the camera really sees.
    """
    setting = dict(DEFAULT_SENSOR)
    if settings_dict is not None:
        setting.update(settings_dict)
    picture = picture.astype(np.float32)
    mean_value = float(picture[valid].mean()) if valid.any() else 0.0
    picture = np.where(valid, picture, mean_value)

    if setting["blur_sigma_px"] > 0.05:
        picture = cv2.GaussianBlur(picture, (0, 0), setting["blur_sigma_px"])
    picture = picture + rng.normal(0, setting["noise"], picture.shape).astype(np.float32)
    stripes = rng.normal(0, setting["stripe"], picture.shape[1]).astype(np.float32)
    picture = picture + stripes[None, :]

    low, high = np.percentile(picture[valid], [1, 99]) if valid.any() else (0.0, 1.0)
    stretched = np.clip((picture - low) / (high - low + 1e-6), 0, 1)
    stretched = 0.5 + setting["contrast"] * (stretched - 0.5)                 # 1 = full range, below 1 = flat and foggy
    if setting["vignette"] > 0:
        stretched = stretched * vignette_factor(stretched.shape, setting["vignette"])
    if setting["gamma"] != 1.0:
        stretched = np.power(np.clip(stretched, 0, 1), setting["gamma"])           # above 1: dark body with a bright tail
    if setting["clahe"] > 0:
        stretched = local_enhance(stretched, setting["clahe"])
    if setting["grain"] > 0:
        grain = cv2.GaussianBlur(rng.normal(0, 1, stretched.shape).astype(np.float32), (0, 0), 0.6)
        stretched = np.clip(stretched + setting["grain"] * grain, 0, 1)         # added after the enhancement, as a real chain shows it
    levels = 2 ** setting["bits"] - 1
    quantised = np.round(stretched * levels)
    return quantised.astype(np.uint8 if setting["bits"] <= 8 else np.uint16)


def vignette_factor(shape, strength):
    """1 in the middle, down to 1 - strength in the corners (lens falloff seen in real thermal frames)."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    middle_row = (shape[0] - 1) / 2
    middle_col = (shape[1] - 1) / 2
    distance = np.hypot((rows - middle_row) / middle_row, (cols - middle_col) / middle_col) / np.sqrt(2)
    return 1.0 - strength * distance ** 2


def local_enhance(stretched, clip_limit):
    """Local contrast enhancement (CLAHE) of a 0..1 picture, like the enhancement many thermal cameras apply. clip_limit 2 to 4 is typical."""
    as_16bit = (np.clip(stretched, 0, 1) * 65535).astype(np.uint16)
    enhancer = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return enhancer.apply(as_16bit).astype(np.float32) / 65535

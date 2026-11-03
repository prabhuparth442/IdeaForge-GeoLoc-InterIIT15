import cv2
import numpy as np

from header import settings


def load_thermal_frame(item):
    frame = cv2.imread(str(settings.IMAGES_DIR / item["name"]), cv2.IMREAD_UNCHANGED)
    if frame.ndim == 3:
        frame = frame.mean(axis=2)
    return frame


def normalise_frame(frame, low_pct=1, high_pct=99):
    frame = frame.astype(np.float32)
    low, high = np.percentile(frame, [low_pct, high_pct])
    stretched = np.clip((frame - low) / (high - low + 1e-6), 0, 1)
    as_16bit = (stretched * 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(as_16bit).astype(np.float32) / 65535

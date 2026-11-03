import cv2
import numpy as np
import torch

from header import settings
from header.settings import DEVICE, MATCH_SIZE


class Matcher:
    def __init__(self, keypoint_finder, glue):
        self.keypoint_finder = keypoint_finder
        self.glue = glue


def load_matcher():
    from lightglue import LightGlue, SuperPoint

    if not settings.MINIMA_FILE.exists():
        settings.MINIMA_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(settings.MINIMA_URL, str(settings.MINIMA_FILE))
    keypoint_finder = SuperPoint(max_num_keypoints=2048, detection_threshold=0.0005, nms_radius=4, remove_borders=4).eval().to(DEVICE)
    glue = LightGlue(features="superpoint", n_layers=9, flash=False, depth_confidence=0.95, width_confidence=0.99, filter_threshold=0.1).eval().to(DEVICE)

    weights = torch.load(str(settings.MINIMA_FILE), map_location=DEVICE)
    for layer in range(9):
        weights = {k.replace("self_attn.%d" % layer, "transformers.%d.self_attn" % layer): v for k, v in weights.items()}
        weights = {k.replace("cross_attn.%d" % layer, "transformers.%d.cross_attn" % layer): v for k, v in weights.items()}
    report = glue.load_state_dict(weights, strict=False)
    print("MINIMA weights loaded. File layers not in model:", len(report.unexpected_keys), "| model layers not in file:", len(report.missing_keys))
    if len(report.missing_keys) > 0:
        print("  missing (should be none or only harmless buffers):", report.missing_keys[:6])
    return Matcher(keypoint_finder, glue)


@torch.no_grad()
def find_keypoints(matcher, picture):
    assert picture.ndim == 2 and picture.shape[0] == picture.shape[1], "picture must be square and 2-D, got %s" % (picture.shape,)
    assert picture.min() >= 0 and picture.max() <= 1 + 1e-6, "picture values must be in [0, 1], got %.3f..%.3f" % (picture.min(), picture.max())
    big = cv2.resize(picture.astype(np.float32), (MATCH_SIZE, MATCH_SIZE), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(big)[None, None].to(DEVICE)
    return matcher.keypoint_finder.extract(tensor, resize=None)


@torch.no_grad()
def match_keypoints(matcher, features_a, features_b, picture_size):
    assert picture_size > 0, "picture_size must be positive pixels"
    assert "keypoints" in features_a and "keypoints" in features_b, "features must come from find_keypoints"
    from lightglue.utils import rbd

    result = matcher.glue({"image0": features_a, "image1": features_b})
    features_a, features_b, result = [rbd(x) for x in (features_a, features_b, result)]
    pairs = result["matches"]
    scale = picture_size / MATCH_SIZE
    points_a = features_a["keypoints"][pairs[:, 0]].cpu().numpy() * scale
    points_b = features_b["keypoints"][pairs[:, 1]].cpu().numpy() * scale
    scores = result["scores"].cpu().numpy() if "scores" in result else np.ones(len(points_a))
    return points_a.astype(np.float32), points_b.astype(np.float32), scores

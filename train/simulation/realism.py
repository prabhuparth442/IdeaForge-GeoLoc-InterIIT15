"""DINOv2 features of canvases, for comparing sets of real and simulated photos (see compare.frechet_distance, classifier_auc)."""
import numpy as np
import torch

from train.features.dino_tokens import PATCH, token_grids


def dino_features(encoder, canvases, valids, batch=32):
    """One 384-number vector per canvas: the mean of its DINO tokens over the patches that are mostly valid.

    canvases: list of (224, 224) floats in 0..1; valids: matching bool masks. The encoder comes from train.features.dino_tokens.load_dino(224).
    """
    assert len(canvases) == len(valids) and len(canvases) > 0, "need one valid mask per canvas"
    results = []
    for start in range(0, len(canvases), batch):
        pictures = np.stack(canvases[start:start + batch]).astype(np.float32)
        grids = token_grids(encoder, pictures).cpu().numpy()
        for offset in range(len(pictures)):
            valid = valids[start + offset]
            side = valid.shape[0] // PATCH
            share = valid[:side * PATCH, :side * PATCH].reshape(side, PATCH, side, PATCH).mean(axis=(1, 3))
            good = share > 0.5
            if not good.any():
                good = share >= share.max()
            results.append(grids[offset][:, good].mean(axis=1))
    return np.array(results)

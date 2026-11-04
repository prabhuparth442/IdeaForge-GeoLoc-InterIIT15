"""DINOv2 patch tokens: one 384-number vector per 14 x 14 pixel patch of a picture (layer 8, as src/coarse_search/dino_hint.py).

A 224 px picture gives a 16 x 16 grid of tokens. The tokens keep layout and carry meaning ("roof", "road", "field")
that edge directions lack. This file only runs the network; turning tokens into cells is in dino_dense.py.
"""
import numpy as np
import torch

from header.settings import DEVICE

LAYER = 8
PATCH = 14
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def load_dino(window=224):
    """DINOv2 ViT-S/14 for pictures of window x window pixels (window must be a multiple of 14; the patch suggests 518)."""
    import timm
    assert window % PATCH == 0, "window must be a multiple of %d" % PATCH
    encoder = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=window)
    encoder = encoder.to(DEVICE).eval()
    if DEVICE == "cuda":
        encoder = encoder.half()
    return encoder


@torch.no_grad()
def token_grids(encoder, pictures):
    """pictures (batch, H, W) floats in 0..1, H = W = the window the encoder was made for. Returns (batch, 384, H/14, W/14) floats."""
    batch, height, width = pictures.shape
    x = torch.from_numpy(np.ascontiguousarray(pictures)).float().to(DEVICE).unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - IMAGE_MEAN) / IMAGE_STD
    if DEVICE == "cuda":
        x = x.half()
    tokens = encoder.get_intermediate_layers(x, n=[LAYER], norm=True)[0]
    tokens = tokens.float().reshape(batch, height // PATCH, width // PATCH, -1)
    return tokens.permute(0, 3, 1, 2)

import time

import numpy as np
import torch

from header import settings
from header.settings import DEVICE, DISC_RADIUS, TILE, TILE_STEP
from image_processing.map_levels import level_gsd

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


DINO_FILE = settings.DATASET_DIR / "NUDT" / "checkpoint" / "DINO" / "dinov2_vits14_timm.safetensors"      # local copy of the weights timm would download


def load_encoder(weights_file=DINO_FILE):
    """DINOv2 ViT-S/14. With the local weights file there is no network call at all (the download check of timm is skipped); without it
    timm fetches the weights from the Hugging Face hub (and caches them)."""
    import timm
    if weights_file is not None and weights_file.exists():
        encoder = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, pretrained_cfg_overlay=dict(file=str(weights_file)), img_size=TILE)
    else:
        encoder = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=TILE)
    encoder = encoder.to(DEVICE).eval()
    if DEVICE == "cuda":
        encoder = encoder.half()
    return encoder


def l2_normalise(vectors):
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)


def list_tiles(level_image):
    corners = []
    for top in range(0, level_image.shape[0] - TILE + 1, TILE_STEP):
        for left in range(0, level_image.shape[1] - TILE + 1, TILE_STEP):
            tile = level_image[top:top + TILE, left:left + TILE]
            if (tile > 0).mean() >= 0.5:
                corners.append((top, left))
    return np.array(corners)


class DinoHint:
    def __init__(self, level_images):
        self.encoder = load_encoder()
        self.tiles = {}
        self.tile_centres = {}
        self.gallery = {}
        start = time.time()
        for level in settings.LEVELS:
            image = level_images[level]
            corners = list_tiles(image)
            gsd = level_gsd(level)
            self.tiles[level] = corners
            self.tile_centres[level] = np.stack([(corners[:, 1] + TILE / 2) * gsd, (corners[:, 0] + TILE / 2) * gsd], axis=1)
            pieces = []
            for first in range(0, len(corners), 32):
                batch = np.stack([image[t:t + TILE, l:l + TILE] for t, l in corners[first:first + 32]]).astype(np.float32) / 255
                pieces.append(self.describe(batch, batch > 0))
            self.gallery[level] = l2_normalise(np.concatenate(pieces))
            print("level %d: %d tiles described, %.0f s so far" % (level, len(corners), time.time() - start))

    @torch.no_grad()
    def describe(self, images, pixel_valid):
        assert images.ndim == 3 and images.shape[1:] == (TILE, TILE), "images must be (batch, %d, %d), got %s" % (TILE, TILE, images.shape)
        assert pixel_valid.shape == images.shape, "pixel_valid must be shaped like images"
        batch = len(images)
        x = torch.from_numpy(images).float().to(DEVICE).unsqueeze(1).repeat(1, 3, 1, 1)
        x = (x - IMAGE_MEAN) / IMAGE_STD
        if DEVICE == "cuda":
            x = x.half()
        tokens = self.encoder.get_intermediate_layers(x, n=[8], norm=True)[0]
        real = torch.from_numpy(pixel_valid).float().to(DEVICE)
        real = real.reshape(batch, 16, 14, 16, 14).mean(dim=(2, 4)).reshape(batch, -1)
        weights = (real >= 0.5).float().unsqueeze(-1)
        count = weights.sum(dim=1).clamp(min=1)
        vectors = (tokens.float() * weights).sum(dim=1) / count
        return vectors.cpu().numpy()

    def query_vector(self, canvas, valid):
        vector = self.describe(canvas[None].astype(np.float32), valid[None])
        return l2_normalise(vector)[0]

    def tile_score_raster(self, level, canvas, valid, prior_m):
        assert level in self.gallery, "no DINO gallery for level %s" % level
        assert len(prior_m) == 2, "prior_m must be (x, y) metres"
        centres = self.tile_centres[level]
        tile_m = TILE * level_gsd(level)
        gap_x = centres[:, 0] - prior_m[0]
        gap_y = centres[:, 1] - prior_m[1]
        near = np.where(np.hypot(gap_x, gap_y) <= DISC_RADIUS + tile_m / 2)[0]
        scores = self.gallery[level][near] @ self.query_vector(canvas, valid)
        corners = self.tiles[level]
        raster = np.full((corners[:, 0].max() // TILE_STEP + 1, corners[:, 1].max() // TILE_STEP + 1), scores.mean(), dtype=np.float32)
        raster[corners[near, 0] // TILE_STEP, corners[near, 1] // TILE_STEP] = scores
        return raster, float(np.quantile(scores, 0.3))

    def prior_on_grid(self, raster, geometry):
        assert raster.ndim == 2, "raster must be 2-D"
        raster = torch.from_numpy(raster).to(DEVICE)
        gsd = level_gsd(geometry["level"])
        row_pos = (geometry["y_m"] / gsd / TILE_STEP - 1).clamp(0, raster.shape[0] - 1.001)
        col_pos = (geometry["x_m"] / gsd / TILE_STEP - 1).clamp(0, raster.shape[1] - 1.001)
        r0 = row_pos.floor().long()
        c0 = col_pos.floor().long()
        fr = (row_pos - r0)[:, None]
        fc = (col_pos - c0)[None, :]
        upper = raster[r0][:, c0] * (1 - fc) + raster[r0][:, c0 + 1] * fc
        lower = raster[r0 + 1][:, c0] * (1 - fc) + raster[r0 + 1][:, c0 + 1] * fc
        return upper * (1 - fr) + lower * fr

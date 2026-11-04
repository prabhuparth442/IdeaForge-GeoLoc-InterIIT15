"""The pooled DINO similarity as a score map (the "DINO hint" of src/coarse_search/dino_hint.py, kept as a map of its own).

One DINO vector per 224 px tile of the map and one for the query; their cosine similarity, spread over the search grid.
It is the coarse "does this place look like the same kind of place" signal. It cannot say where inside a tile the photo is.
A map like this is added to the others after compute_maps (see add_pooled_map) and then goes through z_stack like any other.
"""
import torch

from coarse_search.dino_hint import DinoHint


class PooledDinoMap:
    def __init__(self, level_images):
        self.hint = DinoHint(level_images)

    def score_map(self, canvas, valid, level, prior_m, geometry):
        """Cosine similarity at every grid position, -1 outside the search disc."""
        raster, _ = self.hint.tile_score_raster(level, canvas, valid, prior_m)
        grid = self.hint.prior_on_grid(raster, geometry)
        return torch.where(geometry["inside"], grid, torch.full_like(grid, -1.0))


def add_pooled_map(maps, pooled, canvas, valid, level, prior_m, geometry, name="dino_pool"):
    maps[name] = pooled.score_map(canvas, valid, level, prior_m, geometry)

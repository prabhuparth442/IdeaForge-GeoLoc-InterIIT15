"""Block: a coarse method that is a trained fusion model over some of the maps (patch C2)."""
import torch

from coarse_search.fusion import fuse_maps


def make_fused_method(model, subset):
    """model: a FusionModel; subset: the map names it was trained on, in training order."""
    def score_map(inputs):
        names = inputs["names"]
        columns = [names.index(name) for name in subset]
        context_columns = []
        for column in columns:
            context_columns.append(2 * column)
            context_columns.append(2 * column + 1)
        context_columns.append(2 * len(names))                  # the ground pixel size is always the last number
        fused = fuse_maps(model, inputs["z"][..., columns], inputs["context"][context_columns])
        squashed = torch.sigmoid(fused / 10.0)                  # the same squashing as src fusion.search_frame_fused
        return torch.where(inputs["usable"], squashed, torch.full_like(squashed, -1.0))
    return score_map

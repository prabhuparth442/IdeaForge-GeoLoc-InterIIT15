"""Block: the original way of using DINO: a base map plus beta times the pooled DINO similarity (src search_frame, beta = 0.5).
Kept as a method so the learned fusion can be compared with the hand-set beta (patch C2)."""
import torch


def make_hint_method(base="edge_zncc", hint="dino_pool", beta=0.5):
    def score_map(inputs):
        maps = inputs["maps"]
        combined = maps[base] + beta * maps[hint]
        good = inputs["usable"] & (maps[base] > -1)
        return torch.where(good, combined, torch.full_like(combined, -1.0))
    return score_map

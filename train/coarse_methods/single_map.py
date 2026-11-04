"""Block: a coarse method that is just one of the score maps (edge_zncc, mind_zncc, learned_zncc, dino_zncc, dino_pool ...)."""
import torch


def make_single_map_method(name):
    def score_map(inputs):
        scores = inputs["maps"][name]
        good = inputs["usable"] & (scores > -1)
        return torch.where(good, scores, torch.full_like(scores, -1.0))
    return score_map

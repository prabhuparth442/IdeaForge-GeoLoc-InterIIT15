"""R1: learn to rank the strong candidates (not only the fused score).

Trained on the "hard" candidates of each photo (the truth plus the strongest wrong peaks, kind 0 and 1), a list of ~20.
Per candidate the features are: z of every map, the fused score of the chosen fusion model, its rank in the list, its gap to
the list best, and (when the matcher was run) its inlier count. The model is a small monotone-free MLP with a listwise
softmax loss; it is only worth using if it beats the fused score alone by a clear margin on the valid flights.
"""
import numpy as np
import torch
import torch.nn as nn

from train.train_fusion import paired_bootstrap


def list_features(z, fused, kind_mask, extra=None):
    """z (P,n,K), fused (P,n), kind_mask (P,n) bool (True = in the list) -> features (P,n,F)."""
    big_neg = -1e4
    masked = torch.where(kind_mask, fused, torch.full_like(fused, big_neg))
    rank = masked.argsort(dim=1, descending=True).argsort(dim=1).float()
    best = masked.max(dim=1, keepdim=True).values
    second = masked.topk(2, dim=1).values[:, 1:2]
    parts = [z, fused[..., None], rank[..., None] / 10.0, (fused - best)[..., None], (fused - second)[..., None]]
    if extra is not None:
        parts.append(extra)
    return torch.cat(parts, dim=-1)


class Ranker(nn.Module):
    def __init__(self, n_features, hidden=24):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))

    def forward(self, feats):
        return self.net((feats - self.mean) / self.std).squeeze(-1)


def listwise(scores, mask):
    scores = torch.where(mask, scores, torch.full_like(scores, -1e4))
    return -(scores[:, 0] - torch.logsumexp(scores, dim=1)).mean()


def train_ranker(feats, mask, tr, va, epochs=300, lr=0.01, seed=0):
    torch.manual_seed(seed)
    model = Ranker(feats.shape[-1])
    flat = feats[tr][mask[tr]]
    model.mean.copy_(flat.mean(0)); model.std.copy_(flat.std(0) + 1e-3)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    best, state = 1e9, None
    for _ in range(epochs):
        opt.zero_grad()
        loss = listwise(model(feats[tr]), mask[tr]); loss.backward(); opt.step()
        with torch.no_grad():
            v = float(listwise(model(feats[va]), mask[va]))
        if v < best:
            best, state = v, {k: x.clone() for k, x in model.state_dict().items()}
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def top1_flags(scores, mask):
    scores = torch.where(mask, scores, torch.full_like(scores, -1e4))
    return (scores.argmax(1) == 0).numpy()


def compare_to_fused(fused_valid, feats_valid, mask_valid, model, n_boot=2000):
    base = top1_flags(fused_valid, mask_valid)
    ranked = top1_flags(model(feats_valid), mask_valid)
    d, lo, hi = paired_bootstrap(ranked, base, n_boot=n_boot)
    return dict(fused_top1=float(base.mean()), ranker_top1=float(ranked.mean()), diff=d, lo=lo, hi=hi)

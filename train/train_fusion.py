"""Train a fusion model on per-photo candidate sets (listwise softmax loss).

A training example = one photo: z (n, K) z-scores of K maps at n candidate positions (index 0 = the true position),
and the photo's context vector. Loss = -log softmax(S)[0]: the truth must outscore the other candidates.
"""
import numpy as np
import torch

from coarse_search.fusion import FusionModel


def split_by_group(groups, valid_fraction, seed):
    """Split photo indices so that every flight (group) lands wholly in train or in valid."""
    assert len(groups) > 0, "no photos"
    names = sorted(set(groups))
    rng = np.random.RandomState(seed)
    rng.shuffle(names)
    n_valid = max(1, int(round(len(names) * valid_fraction)))
    valid_names = set(names[:n_valid])
    is_valid = np.array([g in valid_names for g in groups])
    assert is_valid.any() and (~is_valid).any(), "need at least two groups to split"
    return np.where(~is_valid)[0], np.where(is_valid)[0]


def listwise_loss(model, z, context):
    """z (B, n, K), context (B, C) -> mean of -log softmax(S)[:, 0]."""
    scores = model(z, context)
    return -(scores[:, 0] - torch.logsumexp(scores, dim=1)).mean()


@torch.no_grad()
def top1_accuracy(model, z, context):
    scores = model(z, context)
    return float((scores.argmax(1) == 0).float().mean())


def train(kind, z, context, train_idx, valid_idx, epochs=300, lr=0.03, weight_decay=1e-4, seed=0, verbose=False):
    """z (P, n, K) float tensor, context (P, C). Returns (model, history)."""
    assert z.ndim == 3, "z must be (photos, candidates, maps)"
    torch.manual_seed(seed)
    model = FusionModel(kind, z.shape[2], context.shape[1])
    if kind == "gated":
        model.set_context_scale(context[train_idx].numpy())
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history, best, best_state = [], 1e9, None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = listwise_loss(model, z[train_idx], context[train_idx])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val = float(listwise_loss(model, z[valid_idx], context[valid_idx]))
        history.append((float(loss.detach()), val))
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and epoch % 50 == 0:
            print("epoch %3d train %.4f valid %.4f" % (epoch, float(loss), val))
    model.load_state_dict(best_state)
    return model.eval(), history


def paired_bootstrap(correct_a, correct_b, n_boot=2000, seed=0):
    """Difference in accuracy (a - b) with a 95% interval, resampling photos."""
    assert len(correct_a) == len(correct_b), "paired arrays must match"
    rng = np.random.RandomState(seed)
    diff = np.asarray(correct_a, float) - np.asarray(correct_b, float)
    boots = [diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


@torch.no_grad()
def correct_flags(model, z, context, idx):
    return (model(z[idx], context[idx]).argmax(1) == 0).numpy()

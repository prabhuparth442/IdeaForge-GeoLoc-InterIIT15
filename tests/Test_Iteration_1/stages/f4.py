"""Stage f4: train the learned features (two small CNNs) on NUDT train photos; weights go to models/f4.pt."""
import numpy as np

from train import config
from train.features.learned_net import make_pairs, train_features


def run_stage(run):
    path = run.out / "models" / "f4.pt"
    if path.exists() and "f4" not in run.args.redo:
        run.log("F4 weights already exist:", path)
        return
    sets, store = run.shared["sets"], run.shared["store"]
    smoke = run.args.smoke
    count = 6 if smoke else config.F4_PAIRS
    rng = np.random.default_rng(config.SEED + 1)
    train_ids = rng.choice(len(sets["train"].items), size=min(count, len(sets["train"].items)), replace=False)
    valid_ids = rng.choice(len(sets["valid"].items), size=min(8 if smoke else 300, len(sets["valid"].items)), replace=False)
    margin = 8 if smoke else None
    train_pairs = make_pairs(sets["train"], store, train_ids, margin, log=run.log)
    valid_pairs = make_pairs(sets["valid"], store, valid_ids, margin, log=run.log)
    minutes = 0.5 if smoke else (run.args.f4_minutes or config.F4_MINUTES)
    history, best, baseline = train_features(train_pairs, valid_pairs, path, minutes=minutes, batch=4 if smoke else None, log=run.log)
    run.set_result("f4", dict(valid_hit_rate=best, edge_hit_rate=baseline, steps=history[-1][0]))

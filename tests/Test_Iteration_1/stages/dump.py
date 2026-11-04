"""Stage dump: for every train and valid photo, all score maps -> one small candidate set (the long stage). Restart-safe."""
import time

import numpy as np

from stages.helpers import build_families, shard_dir
from train.datasets.candidates import dump_split


def run_stage(run):
    store = run.shared["store"]
    names, families = build_families(store, run.out / "models" / "f4.pt", run.log)
    run.shared.update(names=names, families=families)
    run.log("maps used:", names)
    folder = shard_dir(run.out, names)
    total_budget = max(run.deadline - time.time(), 60) * 0.55
    for split, share in (("train", 0.8), ("valid", 0.2)):
        test_set = run.shared["sets"][split]
        deadline = time.time() + total_budget * (share if split == "train" else 0.25)
        dump_split(test_set, store, families, names, np.arange(len(test_set.items)), folder, split, deadline=deadline, log=run.log)

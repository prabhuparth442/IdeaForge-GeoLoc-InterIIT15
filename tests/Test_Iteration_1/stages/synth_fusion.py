"""Stage fusion (synthetic run): train on fake photos only, and on fake + real photos; the real valid photos pick the winners."""
from pathlib import Path

import numpy as np
import torch

from stages.helpers import load_real_candidates, map_names
from train.datasets.candidates import load_shards
from train.datasets.synthetic_candidates import truth_wins
from train.models.fusion_ablation import run_ablation


def mix_with_real(synth, real_train, repeat):
    return dict(z=torch.cat([synth["z"]] + [real_train["z"]] * repeat),
                context=torch.cat([synth["context"]] + [real_train["context"]] * repeat),
                groups=synth["groups"] + real_train["groups"] * repeat)


def run_stage(run):
    real_out = Path(run.args.real_out)
    names, real_train, real_valid = load_real_candidates(real_out)
    assert names == map_names(real_out / "models" / "f4.pt")
    synth = load_shards(run.out / "shards_synth", "train", names)
    run.log("fake photos: %d | real train photos: %d | real valid photos (picks the winner): %d" % (len(synth["z"]), len(real_train["z"]), len(real_valid["z"])))
    run.log("truth wins per map, real train:   %s" % np.round(truth_wins(real_train["z"].numpy()), 2))
    run.log("truth wins per map, fake photos:  %s" % np.round(truth_wins(synth["z"].numpy()), 2))
    mixed = mix_with_real(synth, real_train, run.args.real_repeat)
    for kind, train_data in (("synth", synth), ("mixed", mixed)):
        run.log("== training on %s (%d photos)" % (kind, len(train_data["z"])))
        rows, best = run_ablation(train_data, real_valid, names, run.out / "models" / kind, log=run.log)
        run.set_result("ablation_" + kind, dict(rows=rows, best=best))
        run.log("best %s: %s %s valid top-1 %.3f" % (kind, best["maps"], best["kind"], best["valid_top1"]))

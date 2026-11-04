"""Stage ranker: a small listwise ranker over the strongest candidates (coarse features only)."""
import torch

from coarse_search.fusion import load_model
from stages.helpers import load_real_candidates
from train import config
from train.models import ranker
from train.models.fusion_ablation import columns
from train.train_fusion import split_by_group


def run_stage(run):
    names, train_data, valid_data = load_real_candidates(run.out)
    best = run.state["results"]["fusion_best"]
    model = load_model(run.out / "models" / "fusion" / (best["tag"] + ".pt"))
    subset = best["subset"]

    def prepare(data):
        z, context = columns(data, names, subset)
        fused = model(z, context)
        mask = data["kind"] < 2
        return z, fused.detach(), mask, ranker.list_features(z, fused.detach(), mask)

    z_train, fused_train, mask_train, features_train = prepare(train_data)
    z_valid, fused_valid, mask_valid, features_valid = prepare(valid_data)
    inner_train, inner_valid = split_by_group(train_data["groups"], 0.2, config.SEED)
    trained = ranker.train_ranker(features_train, mask_train, inner_train, inner_valid, epochs=80 if run.args.smoke else 300)
    result = ranker.compare_to_fused(fused_valid, features_valid, mask_valid, trained, n_boot=config.BOOTSTRAP)
    torch.save(trained.state_dict(), run.out / "models" / "ranker.pt")
    run.log("ranker on the hard candidates (valid): fused-score top-1 %.3f, ranker top-1 %.3f, difference %+.3f (95%% %+.3f .. %+.3f)" % (
        result["fused_top1"], result["ranker_top1"], result["diff"], result["lo"], result["hi"]))
    run.set_result("ranker", result)

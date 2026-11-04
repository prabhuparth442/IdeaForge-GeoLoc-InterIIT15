"""Stage fusion: train linear / monotone / gated fusion on several map subsets; the valid photos pick the winner."""
from stages.helpers import load_real_candidates
from train.models.fusion_ablation import run_ablation


def run_stage(run):
    names, train_data, valid_data = load_real_candidates(run.out)
    run.log("fusion: %d train photos, %d valid photos, %d candidates each" % (len(train_data["z"]), len(valid_data["z"]), train_data["z"].shape[1]))
    rows, best = run_ablation(train_data, valid_data, names, run.out / "models" / "fusion", log=run.log)
    run.set_result("fusion_rows", rows)
    run.set_result("fusion_best", best)
    run.log("best by valid top-1:", best["maps"], best["kind"], "%.3f" % best["valid_top1"])

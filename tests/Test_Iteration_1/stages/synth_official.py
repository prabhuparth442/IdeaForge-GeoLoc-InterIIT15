"""Stage official (synthetic run): the three fusion models on the official photos, another city (the generalisation check)."""
from pathlib import Path

from stages.helpers import synthetic_fusion_models
from stages.official_check import evaluate_on_official_set


def run_stage(run):
    load_models = lambda: synthetic_fusion_models(run.args.real_out, run.out, run.state["results"])
    evaluate_on_official_set(run, load_models, "evaluate_official", run.args.n_eval_official, Path(run.args.real_out) / "models" / "f4.pt")

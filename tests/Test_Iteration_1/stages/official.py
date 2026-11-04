"""Stage official: the trained models on the official development photos (the out-of-domain / overfitting check)."""
from stages.helpers import real_fusion_models
from stages.official_check import evaluate_on_official_set


def run_stage(run):
    assert "fusion_best" in run.state["results"], "no trained fusion model yet: run the stages up to 'fusion' first"
    evaluate_on_official_set(run, lambda: real_fusion_models(run.out, run.state["results"]), "official",
                             run.args.n_official or (12 if run.args.smoke else 500), run.out / "models" / "f4.pt")

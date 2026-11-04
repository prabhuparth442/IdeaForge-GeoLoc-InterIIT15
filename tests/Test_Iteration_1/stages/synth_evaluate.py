"""Stage evaluate (synthetic run): real-only, fake-only and mixed fusion on real NUDT test photos."""
from stages.helpers import synthetic_fusion_models
from stages.nudt_check import evaluate_on_photos


def run_stage(run):
    shared = run.shared
    models = synthetic_fusion_models(run.args.real_out, run.out, run.state["results"])
    evaluate_on_photos(run, shared["test_set"], shared["store"], shared["families"], shared["names"], models, "evaluate_nudt", "NUDT test")

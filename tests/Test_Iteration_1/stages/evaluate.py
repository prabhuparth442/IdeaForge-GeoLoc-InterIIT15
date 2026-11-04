"""Stage evaluate: full 5 km map search on NUDT test photos for every single map and the best fusion models."""
from stages.helpers import real_fusion_models
from stages.nudt_check import evaluate_on_photos


def run_stage(run):
    shared = run.shared
    if "families" not in shared:
        from stages.helpers import build_families
        shared["names"], shared["families"] = build_families(shared["store"], run.out / "models" / "f4.pt", run.log)
    models = real_fusion_models(run.out, run.state["results"])
    evaluate_on_photos(run, shared["sets"]["test"], shared["store"], shared["families"], shared["names"], models, "evaluate", "NUDT test")

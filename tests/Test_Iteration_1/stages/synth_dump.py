"""Stage dump (synthetic run): pick how hard the fake photos must be, then make candidate sets from them (restart-safe)."""
import time
from pathlib import Path

from stages.helpers import load_real_candidates
from train.datasets.synthetic_candidates import dump_synthetic, pick_strength, save_sample_picture, truth_wins


def run_stage(run):
    shared = run.shared
    _, real_train, _ = load_real_candidates(Path(run.args.real_out))
    real_rates = truth_wins(real_train["z"].numpy())
    strength = run.state["results"].get("strength")
    if strength is None:
        strength = pick_strength(shared["satellite_map"], shared["families"], shared["names"], shared["picker"], shared["camera"], real_rates, log=run.log)
        run.set_result("strength", strength)
    picture = run.out / "pseudo_samples.png"
    save_sample_picture(picture, shared["satellite_map"], shared["picker"], shared["test_set"], shared["camera"], strength)
    run.log("look at the fake photos next to real ones:", picture)
    deadline = time.time() + 60 * run.args.dump_minutes
    dump_synthetic(shared["satellite_map"], shared["families"], shared["names"], shared["picker"], shared["camera"],
                   run.out / "shards_synth", run.args.n_synth, strength, deadline, log=run.log)

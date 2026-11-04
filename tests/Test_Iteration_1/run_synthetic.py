"""Train the fusion on pseudo-thermal photos cut from the Changsha map, then test it on real thermal photos.

    python run_synthetic.py                       # the whole thing, about 45 minutes
    python run_synthetic.py --dump-minutes 2 --n-synth 100 --n-eval-nudt 6 --n-eval-official 6 --out results/synthetic_smoke   # quick check
    python run_synthetic.py --stages fusion evaluate official report    # only these stages (finished work is kept)

Stages: setup (map, photos, feature families) -> dump (fake photos -> candidate sets) -> fusion (train on fake only, and on
fake + real) -> evaluate (real NUDT test photos) -> official (real photos of the development set on the small_sat_area map,
another city) -> report. Everything goes to results/synthetic/, the earlier run in results/overnight/ is only read.
Not in this run: DINO (the fusion keeps the same six maps).
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO), str(REPO / "src")]

from stages import dev_defaults  # noqa: F401  (must be imported before any stage changes the settings)
from stages import synth_dump, synth_evaluate, synth_fusion, synth_official, synth_report, synth_setup
from stages.driver import run_stages
from stages.run_context import RunContext
from train import config

STAGE_FUNCTIONS = {"setup": synth_setup.run_stage, "dump": synth_dump.run_stage, "fusion": synth_fusion.run_stage,
                   "evaluate": synth_evaluate.run_stage, "official": synth_official.run_stage, "report": synth_report.run_stage}
NEEDS_SETUP = {"dump", "evaluate"}


def read_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="*", help="run only these stages (%s)" % ", ".join(STAGE_FUNCTIONS))
    parser.add_argument("--redo", nargs="*", default=[], help="run these stages again even if finished")
    parser.add_argument("--out", default=str(config.SYNTH_OUT_DIR))
    parser.add_argument("--real-out", default=str(config.OUT_DIR), help="the earlier run: models, geometry, real candidate shards")
    parser.add_argument("--dump-minutes", type=float, default=25, help="time budget of the fake-photo dump")
    parser.add_argument("--n-synth", type=int, default=20000, help="most fake photos to make (the time budget usually stops it first)")
    parser.add_argument("--real-repeat", type=int, default=2, help="in the mixed training set every real photo counts this many times")
    parser.add_argument("--n-eval-nudt", type=int, default=150)
    parser.add_argument("--n-eval-official", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = read_arguments()
    run = RunContext(args, args.out)
    run_stages(run, STAGE_FUNCTIONS, args.stages or list(STAGE_FUNCTIONS), args.redo, NEEDS_SETUP)

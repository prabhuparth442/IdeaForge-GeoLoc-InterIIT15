"""The overnight experiment on real NUDT photos, stage by stage, with a report at the end.

    python run_overnight.py                     full run on NUDT (about 8-10 h with a GPU; see --hours)
    python run_overnight.py --smoke             the same stages on a tiny synthetic world, to check everything works (a few minutes)
    python run_overnight.py --stages check geometry      only some stages
    python run_overnight.py --redo f4 dump               run those stages again even if they finished before
    python run_overnight.py --stages official            after the other stages: the out-of-domain check on the official photos

Stages (in order): check, setup, geometry, f4, dump, fusion, ranker, evaluate, official, report.
Restarting is safe: finished stages are skipped, finished candidate shards are skipped, a crash in one stage is logged and the
next stage still runs when it can. Everything goes to results/overnight/ (run.log, state.json, report.md, models/).
Where is NUDT? Set NUDT_DIR to the folder holding Reference_map/ and Thermal-UAV/ (default: Code_Testing/dataset/NUDT).
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO), str(REPO / "src")]

from stages import dev_defaults  # noqa: F401  (must be imported before any stage changes the settings)
from stages import check, dump, evaluate, f4, fusion, geometry, official, ranker, report, setup_nudt
from stages.driver import run_stages
from stages.run_context import RunContext
from train import config

STAGE_FUNCTIONS = {"check": check.run_stage, "setup": setup_nudt.run_stage, "geometry": geometry.run_stage, "f4": f4.run_stage,
                   "dump": dump.run_stage, "fusion": fusion.run_stage, "ranker": ranker.run_stage, "evaluate": evaluate.run_stage,
                   "official": official.run_stage, "report": report.run_stage}
NEEDS_SETUP = {"geometry", "f4", "dump", "evaluate"}


def read_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny synthetic run to check that everything works")
    parser.add_argument("--hours", type=float, default=10.0, help="total time budget; the candidate dump stops at about 55%% of it")
    parser.add_argument("--stages", nargs="*", help="run only these stages (%s)" % ", ".join(STAGE_FUNCTIONS))
    parser.add_argument("--redo", nargs="*", default=[], help="run these stages again even if finished")
    parser.add_argument("--out", default=str(config.OUT_DIR))
    parser.add_argument("--n-train", type=int, default=0)
    parser.add_argument("--n-valid", type=int, default=0)
    parser.add_argument("--n-test", type=int, default=0)
    parser.add_argument("--f4-minutes", type=float, default=0)
    parser.add_argument("--n-official", type=int, default=0, help="how many official photos to test (default 500 = all)")
    return parser.parse_args()


def prepare_smoke(args):
    """Shrink every number for --smoke and build a fake development-set folder."""
    config.SHARD_SIZE, config.FUSION_EPOCHS, config.BOOTSTRAP = 8, 60, 200
    config.N_RANDOM = 16
    if not os.environ.get("DEV_DATA_DIR"):
        import tempfile
        from train.datasets.smoke_world import write_dev_dir
        os.environ["DEV_DATA_DIR"] = str(write_dev_dir(Path(tempfile.mkdtemp(prefix="smoke_dev_")) / "dataset"))
    if args.out == str(config.OUT_DIR):
        args.out = str(config.OUT_DIR) + "_smoke"
    return args


if __name__ == "__main__":
    args = read_arguments()
    if args.smoke:
        args = prepare_smoke(args)
    run = RunContext(args, args.out, hours=args.hours)
    run_stages(run, STAGE_FUNCTIONS, args.stages or list(STAGE_FUNCTIONS), args.redo, NEEDS_SETUP)

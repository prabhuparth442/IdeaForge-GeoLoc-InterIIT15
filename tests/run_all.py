"""Run every test: the pipeline tests (tests/pipeline) and the tests of the train/ blocks (tests/train).
    python tests/run_all.py                  everything (a few minutes)
    python tests/run_all.py --only ransac    extra arguments (--only, --swap ...) go to the pipeline tests
The experiments in tests/Test_Iteration_1 are not run here: they need the real data (see its README).
"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
bad = []

print("=== pipeline tests")
if subprocess.run([sys.executable, str(here / "pipeline" / "run_all.py")] + sys.argv[1:]).returncode != 0:
    bad.append("pipeline")

for script in ["test_fusion.py", "test_blocks.py", "test_dino_blocks.py"]:
    print("=== train/%s" % script)
    if subprocess.run([sys.executable, str(here / "train" / script)]).returncode != 0:
        bad.append("train/" + script)

print("\nALL TESTS PASSED" if not bad else "\nFAILED: " + ", ".join(bad))
sys.exit(1 if bad else 0)

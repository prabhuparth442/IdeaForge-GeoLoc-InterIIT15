"""Run every test script. Extra arguments (for example --swap ...) are passed to each script.
    python tests/pipeline/run_all.py
    python tests/pipeline/run_all.py --only ransac
    python tests/pipeline/run_all.py --swap fine_matching.ransac.fit_transform=my_fit.py:fit_transform
"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
scripts = ["test_header.py", "test_image_processing.py", "test_coarse_search.py", "test_fine_matching.py",
           "test_result_thresholding.py", "test_evaluation.py", "test_end_to_end.py", "test_roma_and_counts.py", "test_mosaic_map.py", "test_loftr_matcher.py", "test_heading_queries.py"]
bad = []
for script in scripts:
    print("=== %s" % script)
    if subprocess.run([sys.executable, str(here / script)] + sys.argv[1:]).returncode != 0:
        bad.append(script)
print("\nALL SCRIPTS PASSED" if not bad else "\nFAILED: " + ", ".join(bad))
sys.exit(1 if bad else 0)

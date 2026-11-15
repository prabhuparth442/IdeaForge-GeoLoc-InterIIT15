# Isolated IdeaForge review

Read [REPORT_codex.md](REPORT_codex.md) for findings, current values, scores, limitations and priorities. [PROTOCOL_codex.md](PROTOCOL_codex.md) records the fixed experiments and distinguishes regression data from a new blind test.

All new scripts, experiments, fixes and outputs are inside this directory. No existing source, settings, model or submission was changed. Delete only `tests/Review_20261113_codex/` to remove this work. Existing untracked Iteration 6/7 work belongs to the original workspace and should be kept.

## Main findings

- Reconstructed active notebook configuration (0.21 confidence): official150 **54.40/60, 0 wrong**; NUDT707 **54.08/60, 1 wrong**. Older reports compared a 0.22 cut, scoring 52.80/53.84 on these dumps.
- No newly tested decision/refinement variant establishes an accuracy improvement on independent data.
- `FastEdgeAgreementPipeline` reduces coarse-input preparation from 0.670 to 0.316 seconds per six-query CPU benchmark, with exact score maps and top-50 candidates. This is not a full-pipeline or Jetson speed measurement.
- Evaluation fixes cover empty-search reporting, failure to certify any threshold, and M=5 features derived from unavailable candidates.
- NUDT707 excludes earlier photo IDs but shares all four development flights; every location is within 100 m of a previously evaluated location.

## Re-run checks

Run from the IdeaForge repository root. These commands write only to this experiment directory (the existing tests also create temporary synthetic data under `/tmp`). Setting thread counts avoids CPU oversubscription.

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

../venv-geoloc/bin/python tests/Review_20261113_codex/test_review_codex.py
../venv-geoloc/bin/python tests/Review_20261113_codex/test_fast_edge_codex.py
../venv-geoloc/bin/python tests/Review_20261113_codex/run_review_codex.py
../venv-geoloc/bin/python tests/Review_20261113_codex/audit_judge_codex.py
../venv-geoloc/bin/python tests/Review_20261113_codex/run_precision_codex.py
../venv-geoloc/bin/python tests/Review_20261113_codex/benchmark_edge_codex.py
../venv-geoloc/bin/python tests/Review_20261113_codex/inventory_codex.py
```

The original end-to-end test can exercise the report fix using the repository's existing swap mechanism:

```bash
../venv-geoloc/bin/python tests/pipeline/test_end_to_end.py \
  --swap evaluation.report.collect_answers=tests/Review_20261113_codex/evaluation_fixes_codex.py:collect_answers
```

`../venv-geoloc/bin/python tests/run_all.py` runs the original suite, which still contains the known report failure because its source was intentionally left untouched. Run `tests/pipeline/test_iteration7_blocks.py` separately; the aggregate runner does not include it.

## Opt-in acceleration

Use the new class in a separate notebook or runner. Import the pipeline before dataset configuration changes shared settings, as the existing pipeline documentation requires. Supply the real recipe explicitly; bare constructor defaults differ from the notebooks.

```python
import sys
from pathlib import Path

root = Path.cwd()  # IdeaForge repository root
sys.path[:0] = [str(root), str(root / "src"),
                str(root / "tests/Review_20261113_codex")]

from fast_edge_codex import FastEdgeAgreementPipeline

pipeline = FastEdgeAgreementPipeline(
    "official",
    coarse="edge_zncc",
    matcher="loftr_minima",
    satellite="new",
    k_peaks=50,
    m_fine=10,
    min_inliers=8,
    matcher_confidence=0.21,
    loftr_coarse_threshold=0.21,
)
pipeline.setup(n_photos=500)
# Use your existing item and coarse hint, in the map's east/south metre frame:
# answer = pipeline.run(item, prior_m)
```

This is a research agreement-pipeline adapter, not a replacement headless submission. It retains FFTs within a photo, so measure CUDA numerical equivalence, peak memory and latency before deployment. Use `.run()` for complete photos; calls to `.prepare_query()` alone do not establish a shared per-photo cache. The original fused mode remains available through the fallback.

## Opt-in evaluation fixes

`evaluation_fixes_codex.py` contains:

- `collect_answers`: use through the swap command above or import explicitly. Empty coarse searches count as skips and stay in the denominator.
- `available_candidate_features(raw_table, m=5)`: truncate to computed candidates before constructing context. Use on raw signal tables; the frozen model's fit/prediction contract still needs validation on the target pipeline.
- `select_certified_tau(scan, sets, alpha=0.005, delta=0.1)`: returns `None` if no threshold passes. Requires a fixed model and genuinely separate calibration data for a statistical claim.
- `decide_certified(table, probabilities, tau, **policy_kwargs)`: handles that failed certificate by returning only skips. **The old `candidate_judge.decide(..., tau=None)` instead enables its bar-only baseline.** Use the wrapper, and calibrate the same policy kwargs that you will deploy.

The spatial-median adapter is delivered for reproducing the rejected experiment; it is not recommended for adoption. It reduced the official score. No learned model was retrained or installed.

## Evidence files

| File | Contents |
|---|---|
| `results/summary_codex.csv` | Fixed-policy scores, query counts and paired intervals |
| `results/per_photo_codex.csv` | Every photo and policy outcome |
| `results/dump_manifest_codex.csv` | Dump SHA-256 hashes and diagnostic metadata |
| `results/judge_context_summary_codex.csv` | Corrected M=5 context versus previous replay |
| `results/calibration_failure_codex.json` | Reproduced unsafe fallback and isolated fix |
| `results/precision_summary_codex.csv` | Rejected median-refinement results |
| `results/edge_benchmark_codex.json` | Exact-equality benchmark and scoped timing claim |
| `results/split_audit_codex.json` | Photo exclusions, flight overlap and spatial proximity |
| `results/nudt_photo_manifest_codex.csv` | Canonical paths, flights and known prior exposure |
| `results/source_manifest_codex.csv` | Source hashes and shared inode evidence |
| `results/git_state_codex.json` | Branches and pre-existing working-tree changes |
| `results/workspace_inventory_codex.json` | Workspace folder inventory |
| `results/jetson_rescore_codex.json` | Re-scored historical 500-photo output |
| `results/*codex.log` | Test and experiment logs |

`failure_breakdown_codex.csv` is a diagnostic export for this review's stored 0.22 baseline. It uses ground truth only to categorise failures, never to make predictions. Existing/frozen-model outputs and all “fresh” sets already inspected here must now be treated as seen.

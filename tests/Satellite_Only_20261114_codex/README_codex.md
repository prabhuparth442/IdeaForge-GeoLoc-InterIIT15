# Isolated satellite-only end-to-end experiment

Read [REPORT_codex.md](REPORT_codex.md) for measured results and limitations; [PROTOCOL_codex.md](PROTOCOL_codex.md) was written before fitting/evaluation. Production sources are imported, never patched on disk. Remove this entire suffixed directory to revert this experiment. The generated NUDT cache is disposable and git-ignored.

The new CNN starts from scratch and reads only an official satellite GeoTIFF. NUDT satellite pixels and all drone images are excluded from fitting, checkpoint selection and calibration. Existing frozen DINO/MINIMA are separate reference components; this does not certify their original training provenance as satellite-only.

## Run

Use the workspace's `venv-geoloc/bin/python`, from the IdeaForge repository root. Set `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`. For tests also set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`: the host's unrelated ROS pytest plugin otherwise fails importing `lark`.

Scripts, in order:

1. `train_satellite_codex.py`: generates geographic patch manifest, trains, selects with satellite validation, calibrates, then writes `FROZEN_TRAINING_MANIFEST_codex.json`.
2. `build_nudt_store_codex.py`: builds a bounded-memory full-map cache from NUDT's reference raster, without drone coordinates. Uses the actual raster pixel spacing when resampling to the pipeline grid.
3. `run_pipeline_codex.py official`
4. `run_pipeline_codex.py nudt`
5. `summarise_codex.py`

`run_negative_control_codex.py` additionally runs the first official pilot image with the truth deliberately outside the search disc. It was added as a post-pilot diagnostic, is excluded from normal scores, and never changes the frozen model or gate. Run it before the summariser to include that diagnostic in the report.

The paths above are under `tests/Satellite_Only_20261114_codex/`. The pipeline asserts frozen hashes and refuses to overwrite an existing domain's JSONL evaluation. To reproduce from scratch, copy the experiment's source files and protocol into a new sibling `_codex` directory with an empty `results/`; keep these recorded outputs intact. The source uses paths relative to its own directory, so a sibling name works.

Run the isolated tests with:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../venv-geoloc/bin/python -m pytest -q -p no:cacheprovider tests/Satellite_Only_20261114_codex/test_satellite_codex.py
```

## Important interpretation

- This pilot runs all image-to-answer stages on 12 actual frames; it does not run all 500 official plus 2,350 NUDT frames.
- Baseline: current **headless** six-query/fallback pipeline, K50/M5/bar8. It is not a reproduction of the research M10/27-query agreement pipeline.
- Alternatives change only decisions on those same available candidates. Search scheduling follows the reference baseline.
- CPU timing has no deployment time limit; do not compare its algorithmic accuracy directly with a deadline-enforced leaderboard score.
- Zero accepted errors can be obtained by rejecting everything. Always read answers and score alongside wrong-answer counts.
- Satellite synthetic calibration is not a thermal-domain safety certificate. Do not loosen thresholds after looking at these drone outputs and call the same frames held-out validation.

`results/*_evaluation_manifest_codex.json` records exact frames, hints, baseline settings and source hashes. `*_pipeline_codex.jsonl` contains per-photo candidates and stage timing. Frozen weights are `encoder_satellite_codex.pt`; training and calibration details are separate JSON/log files. Any execution failure is retained as zero points in all three compared policies.

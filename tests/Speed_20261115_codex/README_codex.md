# Isolated speed tests

Read REPORT_codex.md for progress/results and PROTOCOL_codex.md for the fixed comparison. All adapters, native code, optional compiled caches and measurements stay in this directory. Original src/, changed_src/, weights and submission files are imported unchanged. Removing only this experiment folder reverts these additions.

## Environment and commands

From the IdeaForge repository root use `../venv-geoloc/bin/python` with `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`. Run heavy jobs sequentially. Baseline and changed versions share the same actual model objects/weights and memory-mapped maps. No training occurs.

1. `tests/Speed_20261115_codex/test_optimisations_codex.py`
2. `tests/Speed_20261115_codex/run_speed_codex.py official --components`
3. `tests/Speed_20261115_codex/run_speed_codex.py official`
4. `tests/Speed_20261115_codex/run_speed_codex.py nudt --components`
5. `tests/Speed_20261115_codex/run_speed_codex.py nudt`
6. `tests/Speed_20261115_codex/native_benchmark_codex.py` (requires g++; builds an optional .so inside results/)
7. `tests/Speed_20261115_codex/judge_benchmark_codex.py` (existing frozen judge, numeric/pandas bookkeeping only)
8. `tests/Speed_20261115_codex/compile_benchmark_codex.py` (optional, heavyweight; see below)
9. `tests/Speed_20261115_codex/summarise_codex.py`

The full runner refuses to overwrite an existing JSONL. For a clean repetition, copy the experiment sources into a new sibling folder with an empty results/ directory; preserve these recorded runs. Component benchmarks include repeats and are not the same as repeating the end-to-end accuracy check. No network installation is needed.

For the optional compiler test, use a bounded subprocess, e.g. `timeout 180s`, with `TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1 TORCHINDUCTOR_CACHE_DIR=<absolute experiment results/inductor_cache_codex path>`. The benchmark records cold compilation separately from warmed inference. It is a backbone-only experiment; do not substitute it into deployment without final match/decision testing. TensorRT/ONNX/Numba variants are unavailable in the current environment.

The installed Inductor C++ command builder initially failed on this workspace's paths with spaces. `run_compile_retry_codex.py` reproduces the successful bounded retry using temporary no-space symlink aliases for the existing venv and the experiment-owned compiler cache. It does not alter packages. The compiler benchmark also probes one complete matching pair; it is excluded from the six-frame full adapter comparison because its outputs are not bit-exact. Both initial failure and retry logs are retained.

## Progress and stage evaluation

- `results/*_pipeline_codex.log`: each variant starts and finishes explicitly.
- `results/*_full_pipeline_codex.jsonl`: append/flush after all variants of each completed frame; contains stage timings, final answers and query parity.
- `results/*_components_codex.json`: per-component repeats, raw match parity and batch-size comparisons.
- `results/summary_codex.json`: aggregate stage means, points, final parity and process peak RSS.
- `Stage_Evaluation_Speed_codex.ipynb`: open/run to inspect saved progress and compare stages with historical Stage_Evaluation output. It reads results, not fitting models.

The three variants are baseline, optimised (photo preprocessing + safe edge specialisation/cache + DINO batched scoring), and backbone (those changes plus LoFTR query-CNN reuse). The last is scheduled only if its component fitted-parity check passes. Larger LoFTR batch variants are component tests, not silently enabled in the full run.

CPU stage measurements do not predict Jetson/GPU speed. The sample contains three official and three NUDT regression frames, with six queries and the unchanged agreement/escalation policy. Output parity is a prerequisite for treating a speed result as an equivalent implementation. Native and compiler experiments remain separate until their entire inference path is validated.

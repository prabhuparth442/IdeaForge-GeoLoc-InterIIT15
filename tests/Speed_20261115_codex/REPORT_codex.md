# Isolated speed experiment — progress and results

No model fitting or threshold tuning. Adapters live in this folder; production sources stay unchanged. These are CPU regression/timing measurements, not target-board or independent generalisation results.

## Component measurements

| Domain | Change | Original median | Changed median | Ratio | Parity |
|---|---|---:|---:|---:|---|
| official | Photo preprocessing (six renders) | 0.0538s | 0.0189s | 2.85x | exact |
| official | DINO K50 | 3.1862s | 3.2421s | 0.98x | exact ranks; score max 0.0 |
| official | LoFTR M10 | 5.6984s | 3.6171s | 1.58x | {'raw_exact': True, 'raw_shapes_same': True, 'raw_max_abs': 0.0, 'inliers_same': True, 'shifts_same': True} |
| official | LoFTR batch2 | 5.6984s | 6.1701s | 0.92x | {'raw_exact': False, 'raw_shapes_same': True, 'raw_max_abs': 1.52587890625e-05, 'inliers_same': True, 'shifts_same': False} |
| official | LoFTR batch4 | 5.6984s | 6.3731s | 0.89x | {'raw_exact': False, 'raw_shapes_same': True, 'raw_max_abs': 1.52587890625e-05, 'inliers_same': True, 'shifts_same': False} |
| nudt | Photo preprocessing (six renders) | 0.3627s | 0.0963s | 3.77x | exact |
| nudt | DINO K50 | 3.1652s | 3.1956s | 0.99x | exact ranks; score max 0.0 |
| nudt | LoFTR M10 | 5.8707s | 3.6405s | 1.61x | {'raw_exact': True, 'raw_shapes_same': True, 'raw_max_abs': 0.0, 'inliers_same': True, 'shifts_same': True} |
| nudt | LoFTR batch2 | 5.8707s | 6.1314s | 0.96x | {'raw_exact': False, 'raw_shapes_same': True, 'raw_max_abs': 1.52587890625e-05, 'inliers_same': True, 'shifts_same': True} |
| nudt | LoFTR batch4 | 5.8707s | 6.6497s | 0.88x | {'raw_exact': False, 'raw_shapes_same': True, 'raw_max_abs': 1.52587890625e-05, 'inliers_same': True, 'shifts_same': True} |

## Real image-to-answer pipeline

Research AgreementQueryPipeline: full5.5km search, K50/M10, DINO batching5, MINIMA.21, six queries plus27-query fallback. baseline is unchanged research code; optimised adds preprocessing/edge/DINO changes; backbone also adds LoFTR query feature reuse. The same maps, hints, camera/geometry, weights and thresholds are used on both sides. No persistent query-result cache is used. The original general edge path computes fusion statistics it does not consume; the adapter skips these.

| Domain | Completed frames | Variant | Mean seconds | Score /60 | Answers | Exact final outputs vs baseline |
|---|---:|---|---:|---:|---:|---|
| official | 3/3 | baseline | 58.65 | 60.00 | 3 | True |
| official | 3/3 | optimised | 57.10 | 60.00 | 3 | True |
| official | 3/3 | backbone | 45.14 | 60.00 | 3 | True |
| nudt | 3/3 | baseline | 63.23 | 52.00 | 3 | True |
| nudt | 3/3 | optimised | 57.82 | 52.00 | 3 | True |
| nudt | 3/3 | backbone | 46.21 | 52.00 | 3 | True |

Stage means and cumulative process peak RSS are in results/summary_codex.json. Individual outputs include every query's candidate-score/rank, inlier and shift comparisons. Process RSS is a high-water mark across variants, not an isolated per-variant allocation measurement. Timings are a small sequential paired pilot, with alternating variant order; they have no reliable deployment confidence interval.

Executed query counts in this pilot: [6]. If only six, full-image timing/parity does not exercise the 27-query escalation branch; the branch remains enabled. Component and unit checks are separate from that coverage limitation.

## Native/compiled experiments

C++ translation-consensus kernel: 194 differential cases; 0 mismatches. Original NumPy semantics and real saved matches were compared, including boundary/tie cases. Kernel-only timings include ctypes overhead; this is not a full-pipeline speedup.
- N=50: NumPy 36.2µs; C++ 17.7µs; 2.04x.
- N=200: NumPy 421.5µs; C++ 293.8µs; 1.43x.
- N=500: NumPy 4094.0µs; C++ 1962.1µs; 2.09x.

CPU TorchInductor LoFTR-backbone-only: eager 0.4900s, compiled warm 0.4216s, cold compile 13.0s. Feature max errors [3.337860107421875e-06, 1.0967254638671875e-05]; allclose [True, True]. Excluded from full-pipeline variants; needs final match validation before adoption.

One complete matching-pair probe: {"eager": {"median": 0.580342051001935, "samples": [0.6094028459992842, 0.56602366199877, 0.580342051001935]}, "compiled": {"median": 0.5265769010002259, "samples": [0.5698128869989887, 0.5265769010002259, 0.5205403189975186]}, "match_counts": [47, 47], "differences": {"points_a": {"shape_same": true, "max_abs": 0.0}, "points_b": {"shape_same": true, "max_abs": 1.52587890625e-05}, "confidence": {"shape_same": true, "max_abs": 6.496906280517578e-06}}, "inliers_same": true, "shifts_same": false, "max_shift_delta_m": 4.3662063148874886e-06}

The initial compile failed because the installed C++ builder split workspace paths containing spaces. Temporary no-space aliases for the existing venv and experiment cache allowed a bounded retry. No package/source installation or modification was required; see compile_retry_paths_codex.json and the two compiler logs.

## changed_src live-judge bookkeeping

Five saved photos, six query prefixes: mean repeated-Pandas/prediction time 0.0352s → 0.0281s (1.25x). All prefix probabilities exact: True. Raw signal extraction and network inference are excluded. The existing frozen T7 judge is a thermal-trained historical baseline, not a newly trained or promoted model.

## Availability, tests and lineage

CUDA is unavailable. TensorRT, ONNX, ONNX Runtime, Torch-TensorRT and Numba are not installed. No engine acceleration or GPU scalar-transfer benefit is claimed. DINO and LoFTR use existing frozen baseline weights; no new encoder or judge was trained.

Unit tests cover backbone scope/restoration, per-photo scope/restoration, DINO missing candidates and vectorised score agreement. See results/unit_tests_codex.log. Raw matches and fitted parity are checked using actual MINIMA inference in the component benchmark. Native code is optional and is not installed into the pipeline.

Read PROTOCOL_codex.md and README_codex.md for exact reproduction and scope.

## What to keep from this experiment

- official: full adapter 1.30x faster (23.0% less wall time) on three frames, exact final outputs. Use `SpeedAgreementPipeline(..., query_backbone=True)` for this variant; default false is the separate preprocessing/edge/DINO ablation.
- nudt: full adapter 1.37x faster (26.9% less wall time) on three frames, exact final outputs. Use `SpeedAgreementPipeline(..., query_backbone=True)` for this variant; default false is the separate preprocessing/edge/DINO ablation.
- Per-photo preparation and scoped query-CNN reuse are the strongest equivalent-implementation candidates from these CPU tests. Retain for larger regression validation, not automatic production promotion.
- DINO vectorisation provides no measured CPU speedup here; its GPU synchronisation benefit remains untested. LoFTR batches2/4 are slower here and are not enabled in the full adapter.
- C++ translation consensus saves time in its kernel, but this kernel is a small fraction of complete inference. The judge incremental optimisation likewise saves milliseconds; it is not the main end-to-end gain.
- The compiled backbone changes floating-point outputs and is excluded from the exact-equivalence full runs. The complete matching-pair probe is only one case, even if its inlier count agrees. Do not treat the compiler result as a fully validated replacement.
- All six pilot frames resolved in the first six queries. Larger datasets, hard/escalated cases, GPU numerical parity, target-board latency and shared memory remain to be measured. No independent accuracy/generalisation gain is claimed.

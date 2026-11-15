# Speed opportunities in src/ and changed_src/

15 November 2026. Inspection and scoped profiling only; no production changes. Start with tensor execution and redundant work, then consider native extensions for a measured remaining bottleneck. Moving Python wrappers to C++ does not itself accelerate the OpenCV/CUDA kernels they already call.

## Timing evidence and its scope

The Stage_Evaluation notebook has useful measurements, but they are historical configurations rather than a profile of today's entire changed_src pipeline. Its original single-query K100/M25 table also scales measured semantic/fine times linearly to the baseline K/M in notebook cell30; its K/M cost table is a model, not a separately timed execution for every setting. Prefer the actual heading-query CSV for identifying stage priorities.

`tests/Stage_Evaluation/time_per_query_official_edge_zncc_headings9_scales3_K50_M10.csv`: 100 photos,27 queries each. Following notebook cell31, exclude the first photo as warm-up:2673 queries over99 photos.

| Stage | Mean/query | Share |
|---|---:|---:|
| render | 5.86 ms | 0.89% |
| maps | 190.46 ms | 28.90% |
| coarse | 0.79 ms | 0.12% |
| semantic / DINO | 107.00 ms | 16.24% |
| fine / LoFTR and fitting | 354.86 ms | 53.85% |
| total | 658.97 ms | 100% |

The expensive FFT correlation is in **maps**, inside `make_inputs → compute_maps → correlate_everywhere`. The tiny `coarse` label mainly applies an already computed score map. It does not mean the entire coarse search is free. Peak extraction belongs to semantic timing in the research pipeline. Do not use this table as a Jetson latency forecast or as a profile of cached execution.

Fresh CPU signal microprofile: five deterministic old official dumps, first six queries,60 candidates, `photo_signal_table(..., with_lo=False)`. Per-photo medians ranged **5.06–10.39ms**. This excludes neural inference, Pandas context creation and judge prediction, and is not a worst-case Jetson bound. `translation_consensus` is the largest signal subroutine in the recorded cProfile. See `results/speed_analysis_codex.json` and `signal_cpu_profile_codex.txt`; reproduce with `profile_speed_codex.py`.

## 1. Highest upside: optimise LoFTR execution, not its Python wrapper

Location: `src/fine_matching/loftr_matcher.py:78`, `run_loftr_chunk`.

- Current batching is implemented: first measure batches1/2/4 under the target board's memory cap. Set the existing `loftr_batch`; writing a second batch API is unnecessary. Batched floating-point execution may change boundary matches, so check final answers as well as speed.
- Benchmark PyTorch compilation on the tensor-heavy network components after warm-up. Whole-pipeline compilation is a poor fit for NumPy/OpenCV/Pandas and data-dependent control flow. Installed Kornia LoFTR has variable matched-point outputs and a data-dependent fine-matching branch; compile the backbone/transformer blocks first if the whole model graph breaks.
- Consider ONNX → TensorRT for the same MINIMA weights on NVIDIA deployment hardware. This is an export/validation project, not a library-name replacement. Fixed224x224 image inputs help, but match-count-dependent output/refinement remains dynamic. A partial TensorRT backbone plus existing matching is a reasonable first prototype.
- The existing loader leaves LoFTR at FP32. Benchmark supported mixed precision carefully, preserving sensitive operations where needed; do not assume every LoFTR operation tolerates FP16. First establish backend parity, then measure a precision variant separately.

Installed Kornia's `LoFTR.forward` runs the CNN backbone on concatenated image0/image1 for every pair. At M10 the identical query is encoded10 times. Refactor an adapter to cache the query's **pre-cross-attention** CNN coarse/fine features once per query; retain independent candidate-specific cross-attention. Image encodings become20→11 per query (M5:10→6), not a claim of a45% full-matcher speedup. The backbone's share is not separately measured. Reuse must preserve eval/BatchNorm behavior, feature copies where downstream modules mutate tensors, and exact query/preprocessing identity. Do not cache post-cross-attention query features across different candidates.

Documentation: [TensorRT ONNX deployment](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-onnx-deployment.html), [TensorRT optimisation and accuracy](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html), [PyTorch compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html). Match actual JetPack/PyTorch/TensorRT versions on the target device; the current laptop has no working CUDA driver.

## 2. High priority: specialised edge-ZNCC map preparation

Locations: `src/coarse_search/correlation.py`, `train/maps/map_bank.py`, `train/coarse_methods/inputs.py` (the latter two are on the src pipeline's execution path).

- Compute only ZNCC when the selected method is edge_zncc. Current general correlation also computes cosine; general inputs additionally compute robust z-scores and fusion context unused by this method.
- Keep the immutable map window and its FFTs on the device across a photo's queries; avoid re-extraction/upload. This research optimisation was already demonstrated in the earlier isolated adapter. Headless already does the analogous reuse.
- Fix the new shared-cache fingerprint defect documented in REPORT_codex.md before trusting it. Use explicit photo/map/level/prior ownership, not equal sums.
- Only after a GPU profile, consider a C++/CUDA or Triton kernel for normalisation/masking and native greedy peak extraction. The FFT itself already calls native backend kernels through PyTorch. Replacing `torch.fft` with another Python FFT library does not inherently improve it.
- Batched per-channel inverse FFTs may reduce launches in the variance calculation, but peak memory and floating-point parity must be measured. Preserve summation/tie-break order where decisions depend on it.

Do not generalise the earlier2.12x CPU coarse-preparation benchmark to end-to-end GPU performance; T10 reported only3% improvement from its shared-FFT-only GPU variant. The specialisation that removes unused work is a different change.

## 3. DINO: fewer synchronisations, then compiled inference

Location: `src/semantic_rerank/dino_rerank.py:73–106`.

`dino_candidate_scores` turns two GPU scalar results into Python floats for every candidate. Reduce the entire batch's valid-token cosine and pooled cosine on GPU, then copy score vectors to CPU once per batch. This keeps the algorithm/weights and removes per-candidate host synchronisations. It is a code change worth testing before TensorRT.

Compile/export a wrapper returning only the needed intermediate tokens and benchmark existing DINO batch sizes. DINO **already uses FP16 on CUDA**, so FP16 is not an untapped optimisation for this component. TensorRT may still fuse operations; no gain is established locally.

`src/coarse_search/search.py:55–84` greedily selects K peaks and repeatedly converts GPU scalars to Python. A native CUDA/C++ implementation could keep the exact sequential square-suppression rule and transfer the final K rows once. Do not replace it with plain `topk` or an unrelated NMS rule: the shortlist changes. Attribute its time separately before prioritising it above the networks.

Do not precompute a global unmasked satellite DINO gallery and assume it equals current scoring: candidate tiles are cut at their exact centres and masked **before** encoding, and the valid mask changes across heading/scale queries. Such a replacement changes the method unless cache identity includes the complete tile/mask/preprocessing/weights contract.

## 4. changed_src/: reduce live bookkeeping; little reason to port whole classes

`JudgeEarlyExitPipeline.early_answer` repeatedly concatenates Pandas tables, recomputes context/features for old queries and predicts probabilities for old rows. Query-local context is independent of other queries: compute a new query's immutable feature block and probabilities once, append numeric arrays, and update the pooled best candidate incrementally. Existing whole-table recomputation is avoidable. Keep policy semantics, candidate availability and model feature order identical.

`query_table` goes through the full offline signal extractor, which calculates several signals absent from FEATURES_CTX (e.g. NFA and extra translation-consensus variants). Create a live extractor for the actual model feature contract and reuse the fine-stage fit/inlier mask where valid. Current early-exit default uses the old thermal-trained judge; accelerating it does not change that training provenance.

If profiling identifies large correspondence sets, `decision_signals.translation_consensus` builds an N-by-N distance matrix. An exact Numba/Cython/C++ loop can count support without that allocation. Preserve distance precision, <= comparisons, earliest-tie selection and the refined centre; changing them can alter features. [Numba performance guidance](https://numba.readthedocs.io/_/downloads/en/stable/pdf/) supports compiled loops; no new Numba speedup is claimed here.

`InflationGuardPipeline.far_level` has a small loop over at most a few hundred candidates; NumPy can simplify it, but the guard/decision wrappers are not the main latency source. RANSAC is already OpenCV native (`estimateAffinePartial2D`); rewriting its Python caller in C++ leaves the expensive fitting kernel the same. Consider further porting only if a target CPU profile says it matters.

## 5. Useful small fix: prepare the photo once

`src/image_processing/geometry.py:35` reloads, percentile-normalises, CLAHE-processes and blurs the same photo for every query. Cache that prepared image within one photo/level and reuse a coordinate mesh, then render each geometry. This can preserve output exactly with the right level/camera/preprocessing lifetime. Headless already has `prepare_photo` separate from `draw_canvas`.

This is worthwhile cleanup, but rendering is under1% in the saved CUDA profile. A wholesale C++ rewrite of projection, CLAHE or image loading is low priority; remap, blur and CLAHE already run in OpenCV.

## Potential whole-pipeline gain

Amdahl calculations using the saved steady-state stage proportions and unchanged query/candidate counts:

- Fine stage2x faster → total **1.37x**, about0.659→0.482s/query.
- Fine stage4x faster → total **1.68x**, about0.659→0.393s/query.
- Fine and semantic stages both2x faster → total **1.54x**, about0.659→0.428s/query.

These are conditional calculations, not measured acceleration forecasts. They show why a2x faster LoFTR is not a2x faster pipeline. Any backend/precision change must be tested on final matches, ranks, inlier counts, answer/skip decisions and geolocation scores, not just tensor closeness. Under the actual deadline, faster execution may allow more queries; that changes the effective policy and needs its own evaluation.

Keep architecture and weights fixed for acceleration comparisons. Swapping LoFTR for EfficientLoFTR/another matcher would require compatible weights and new accuracy/provenance evaluation; it is not an equivalent implementation change. INT8 quantisation also introduces calibration and accuracy questions, so it is not the first choice under the strict satellite-only fitting protocol.

## Recommended sequence

1. Measure a fresh target-board stage profile with the actual deployed configuration, including memory, warm-up and synchronisation/transfer overhead. Historical stage tables guide where to look, not acceptance of a speed claim.
2. Apply per-batch DINO score transfers and per-photo preprocessing; use the existing safe edge-only specialisation/cache after invalidation tests.
3. Benchmark the existing LoFTR batch control and query-backbone reuse; verify identical final behavior on frozen regression inputs.
4. Prototype compiled DINO and LoFTR tensor blocks; then TensorRT where supported. Keep cold setup cost separate from warmed serving time.
5. Port only remaining measured CPU/GPU hotspots. Keep all prototypes under a new `_codex` adapter, preserving src/ and the submission.

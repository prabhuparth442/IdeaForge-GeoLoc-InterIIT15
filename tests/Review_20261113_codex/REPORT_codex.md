# IdeaForge pipeline review — 13 November 2026

The pipeline is a credible working baseline with real thermal-image evidence and a recorded Jetson run. Its biggest weakness is the strength of the generalisation evidence, followed by the mismatch between research, replay and deployment configurations. I did not find a new defensible accuracy improvement in the tested decision/refinement variants. I did implement and verify an isolated speed improvement and evaluation fixes. Nothing was promoted into your source or submission.

**Current notebook configuration reconstructed from saved matches:** 54.40/60 on official150, 54.08/60 on NUDT707; 0 and 1 wrong answers respectively. These are development/regression results, not a new blind benchmark. The older 52.80/53.84 comparison used a different confidence cut.

## What was inspected

- Inventoried all top-level workspace folders, including Code_Testing, Documents, IdeaForge, Main_Pipeline_Headless, Member_Code_Pull, Submission_Files and reports. The inventory records filenames by extension and logical byte counts; it does not imply every image, archive, notebook output or PDF was read in full.
- Read the active research entry points, projection, map/feature generation, candidate retrieval, DINO ranking, fine matching, RANSAC, answer selection, agreement/escalation, evaluation and recent Iteration 6/7 experiments.
- Inspected the headless source, settings, query order, timing, FFT fallback, submission harness and the saved 500-image Jetson run. Compared local `main` to the active research branch without checking out or fetching anything.
- Ran the existing test suite, additional Iteration 7 tests, CPU replays on all 300 locally available rich dumps, frozen-judge checks, a matching-confidence reconstruction, and isolated regression/performance tests.
- Used Python 3.10 in `venv-geoloc`, torch 2.8.0+cu126. CUDA was unavailable and `nvidia-smi` failed. No fresh full neural thermal-to-map sweep or Jetson benchmark was performed.

## Branches and configuration drift

| Location | Actual state |
|---|---|
| IdeaForge repository | `prabhuparth/dev`, commit `f914ca5`; same commit as local `prabhuparth/agreement-decision` |
| Local research `main` | `d3bc95b`; older than the active branch |
| Existing untracked research work | Iteration 6/7, judge/replay/signals modules and the Iteration 7 test; preserved |
| Main_Pipeline_Headless | `master`, `e5e0fd3`, with pre-existing changes in settings, pipeline, coarse search, validation, plus untracked FFT fallback; preserved |
| Headless versus Submission_Files/myteam/code | All 17 compared Python files have identical hashes; **10 are hard-linked**, including settings and pipeline |

The hard links matter: an in-place edit in one tree can also alter its submission counterpart. All this review's new files are in this directory, so that did not happen.

| Parameter | Current Single_Image_Run notebook | Rich dumps used by prior comparisons | Headless settings |
|---|---:|---:|---:|
| Coarse / matcher | edge ZNCC / LoFTR-MINIMA | same | same |
| K / M per query | 50 / 10 | 50 / 10 | 50 / 5 |
| Match confidence cut | **0.21** | **0.22** | 0.21 |
| LoFTR coarse threshold | 0.21 | 0.21 | 0.21 |
| DINO weight / decision gamma | 100 / 1 | 100 / 1 | 100 / 1 |
| First pass | 6 queries; -5 degrees first | 6-query decision replay | 6 queries; zero heading first |
| Answer rule | bar 8 OR agreement | same | bar 8 only |
| Escalation | 27 total, bar 11 | same | up to 18 total, bar 8 |
| Time guard | no headless-style guard | query-count models | 22 seconds |
| Early stopping | separate experimental wrapper | several replay variants | disabled by default |

Shared official geometry: attitude enabled, pitch/roll signs -1, yaw offset 180 degrees, height scale 1.04; map GSD 0.3 m, levels 3–7, canvas 224, search radius 5500 m, RANSAC 3 pixels, scale deviation <=0.15, rotation <=12 degrees, maximum centre correction 12 canvas pixels. Local `main` still has a 5000 m search radius.

`src/header/settings.py` and bare `SinglePhotoPipeline` defaults are **not** the notebook recipe: the constructor defaults include fused coarse search, LightGlue, M=25 and a 12-inlier answer bar. Bare `AgreementQueryPipeline()` also conflicts with its default escalation bar of 11 unless the answer bar is supplied. Keep explicit run manifests instead of assuming one settings file describes every entry point.

## Results reproduced and new experiments

Every sample stays in the score denominator. All new variants below were fixed before their corresponding replay, without a threshold sweep. Existing 707 results had already been examined before this review, so none is a new blind test.

| Rule | official150 score / wrong | NUDT707 score / wrong | Interpretation |
|---|---:|---:|---|
| Agreement baseline, stored 0.22 cut | 52.80 / 0 | 53.84 / 1 | Reproduces existing report |
| **Current notebook, reconstructed 0.21 cut** | **54.40 / 0** | **54.08 / 1** | Current settings check, not a newly discovered setting |
| Eligible-candidate rescue, 0.22 | 52.80 / 0 | 53.84 / 1 | No score gain |
| Inlier-first choice, gamma=0, 0.22 | 52.80 / 0 | 53.84 / 1 | No score gain |
| Valid photographed-mask refit, 0.22 | 52.80 / 0 | 53.84 / 1 | No score gain versus control refit |
| Stop at 12 inliers, zero-heading first, 0.22 | 51.84 / 0 | 53.36 / 1 | Saves queries but loses precision/score |
| Median of nearby query estimates, 0.22 | 52.48 / 0 | 53.84 / 1 | Slightly lower mean error does not improve tiered score |

The 0.21 reconstruction uses stored raw pairs retained down to confidence 0.15, with the same network coarse threshold of 0.21. Re-fitting at the original 0.22 cut reproduced all baseline photo scores, providing a control for dump quantisation. The 0.21 result remains a raw-pair replay, not a fresh network run. Its gains over the older dump recipe are +1.60 [0.32, 3.20] official and +0.24 [0, 0.72] NUDT using paired photo bootstrap; those intervals do not establish a new-site gain.

Early12 reduces queries from 8.38 to 4.74 official and 7.82 to 3.54 NUDT, but official loses 0.96 points [−1.76, −0.32]. I would not adopt it for an accuracy-first objective. The median variant reduces mean accepted error slightly but moves four official photos across score tiers, for a net −0.32. It remains exploratory and disabled.

Headless policies on reconstructed 0.21 matches score 51.76/53.04 at a ten-query cap and 52.16/53.04 when allowed all 18. These are **not Jetson measurements**: the actual code uses elapsed time, not a fixed ten-query cap. The headless second pass does not normally stop as soon as it finds a bar-8 answer; several existing “Jetson” replays do. Query order, matching cut, M, stop rules and timing must match before comparing deployment scores.

Detailed outcomes and confidence intervals are in `results/summary_codex.csv` and `results/per_photo_codex.csv`. Every raw dump has a recorded SHA-256 in `dump_manifest_codex.csv`.

## What prevents a strong generalisation claim

NUDT707 has no photo-index overlap with the excluded 500-photo pool or the earlier fresh-photo list. That exclusion is working. However:

- The 2350-photo NUDT test pool contains four flights, and **all four appear both in the old 500-photo pool and NUDT707**.
- Across the earlier excluded sets, the median distance to a previously evaluated location is **30.55 m**; 139/150 are within 50 m, and **150/150 are within 100 m**. These are approximate local geographic distances, not image-overlap calculations.
- Random fresh frames therefore test new views of familiar areas. They do not independently validate new flights, sites, seasons or cameras. Photo-level bootstrap intervals also do not account for that spatial dependence.
- The judge's official150 gain is in-sample. Previous LOSO experiments split seed-labelled sets, and cross-fitting groups by `data + photo`; that is not a flight/site split. Do not infer actual duplicate leakage from that alone: I found no 101/202 ID intersection in the available decision tables, and the reports explain that IDs from different pool sizes are not comparable.
- The official map crop is constructed from test-photo truth coordinates in `src/header/map_source.py`. This is a possible evaluation shortcut through map extent, even though per-photo matching does not receive truth. Fix the public map area independently of evaluated photos. A truth-centred 6 km margin does not generally cover a 5.5 km search disc centred on a hint displaced by up to 5 km; the worst-case extent is about 10.5 km plus tile context, subject to actual map coverage.
- Existing pose constants were fitted on development data. The implemented levels and existing evidence cover roughly 100–352 m; extrapolation to much higher flights has not been validated.

Use physical-image identities and acquisition/site blocks for splitting. Keep calibration separate from training and final testing; grouped validation is designed for dependent samples ([scikit-learn guidance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)). `results/nudt_photo_manifest_codex.csv` records canonical file paths, flight identities and prior exposure. Its unlisted frames are **not certified blind**.

## Confirmed evaluation defects and isolated fixes

1. **No certified threshold still returns a threshold.** `candidate_judge.ltt_tau` initializes its answer to the strictest cut, returning 0.99 when every cut fails. `smallest_safe_tau` also returns the largest cut when none is safe. I reproduced both with 10 wrong answers in 100. `evaluation_fixes_codex.select_certified_tau` returns `None` when certification fails; `decide_certified` then skips everything, including overrides and early exits. Do not feed that `None` directly into the older `decide`: there it means “use the inlier-bar baseline.” The new selector is an opt-in replacement for the LTT path; the original functions remain unchanged.
2. **Unavailable candidates influence M=5 judge features.** The notebooks compute `add_context` over M=10, then truncate inside `decide`. Consequently `ligap` can use inliers from candidates a true M=5 run never matched. Recomputing features after truncation changes context on 281 official and 1000 NUDT candidate rows; 19 and 109 probabilities change, by as much as 0.319. Final answers happen to remain unchanged here. `available_candidate_features` enforces the correct availability boundary.
3. **Empty coarse shortlist crashes the report.** `src/evaluation/report.py:34` calls `min()` on an empty sequence. The full existing suite reaches this failure. The isolated `collect_answers` wrapper restores empty-search photos as skips, with infinite diagnostic error and zero points, while preserving every nonempty-row result. Its targeted original test passes using `--swap`. It replaces this helper only; other reporting functions with similar assumptions still deserve hardening before use on arbitrary empty inputs.

A risk-control label is not a blanket guarantee. A fixed deployed model, independent calibration evidence and the tested end-to-end selection policy matter; cross-fitted probabilities from several models do not automatically certify a newly refitted model. The underlying framework is described in [Learn then Test](https://arxiv.org/abs/2110.01052). Also validate the policy's early exit, multiple candidates and unconditional overrides as part of the same final rule.

## Implemented speed improvement

`fast_edge_codex.FastEdgeAgreementPipeline` is an optional subclass of the existing agreement pipeline. For edge-ZNCC it:

- reuses the map window and its FFTs for all queries of the same photo;
- avoids computing the unused cosine map;
- avoids robust-z/fusion-context calculations that the edge-only coarse method never consumes;
- retains the original channel summation order, retrieval, DINO, matcher, decision thresholds and query policy;
- invalidates the cache when the prior, level or map object changes, and clears it after each photo. Fused configurations use the original implementation.

CPU benchmark: six synthetic 224-pixel queries, level 7, 5500 m search disc, two threads. Median total coarse-input time over three measured repetitions, after one warm-up repetition: **0.670 s → 0.316 s, 2.12× faster**. Score maps and top-50 candidates are bit-for-bit equal in all 24 comparisons. Three complete synthetic agreement runs with the existing SIFT stand-in matcher produce identical answers, stages, coordinates and inliers. Map changes, prior changes, level changes and map-edge cases also pass.

This is **not** a 2.12× end-to-end speed claim. It has not been timed on CUDA/Jetson. Retaining FFTs increases live memory within each photo, particularly at low-altitude large search windows, so profile peak shared RAM before deploying. The headless pipeline already reuses window FFTs and omits fusion statistics; do not expect this research-only patch to provide the same gain there. It also uses the research torch FFT backend, not the headless cuFFT fallback.

## Failure analysis and next changes worth testing

At the stored 0.22 recipe, all 16 official skips have a candidate within one sigma somewhere in the 27-query top-10 results, but none has more than 10 inliers at such a place. On NUDT, 9 of 12 skips have a candidate within four sigma; 3 do not. This separates weak matching/decision evidence from missing fine-stage candidates. A top-10 miss does not tell us whether the true place was absent from coarse K=50 or was removed by DINO.

The NUDT wrong answer, photo 1181, is about 342 sigma away and has 11 inliers. A near-correct candidate **is present in a standard query**, at 1.25 sigma with 5 inliers. Thus this particular error is not simply “the right place was absent”: a repetitive wrong structure beats the weaker correct match. Raising a bar to remove this one known example would be tuning to the error.

Priority order:

1. **Make evaluation trustworthy first.** One versioned run manifest for research and deployment; physical-image hashes, flight/site blocks, independent calibration, a fixed map area, and a single untouched final test. Freeze rules before inspecting final results. Reuse observed sets for regression, not for renewed blind claims.
2. **Improve weak-match verification on hard photos.** Test a bounded second-stage verifier using spatial support, residuals and uncertainty at the reported camera-centre position. Candidate-level judge features are promising, but the existing judge did not beat agreement on NUDT707 and its current probabilities are not a portable correctness guarantee. Avoid simply reducing the inlier bar or adding an unconditional OR rescue.
3. **Improve retrieval where there is no useful fine candidate.** Save the complete K=50 shortlist and compare edge rank versus DINO rank on missing cases. Then test a fixed small mixture of geometric/coarse and DINO candidates, or a second footprint/level only for unresolved photos, keeping total fine-match budget fixed. This is a hypothesis to test, not an implemented score gain.
4. **Use speed savings to preserve coverage under the real deadline.** Benchmark the isolated cache on the target hardware; audit headless time accounting and query order against a real replay. Conservative early exit remains a useful option, but the new simple 12-inlier version lost score and the judge-based version trades some score for speed. Do not promote modelled “4.5 seconds” as measured Jetson timing.
5. **Validate new operating conditions separately.** Random hints out to 5 km, map boundaries, low texture, camera/pose perturbations, new acquisition dates, and larger heights. Include featureless/out-of-map inputs as skips rather than dropping them from evaluation.

The local 500-image Jetson evidence re-scores to **49.572/60**, with 423 matches and zero wrong answers; mean latency 13.06 s, maximum 21.136 s, no late answers. It used an Orin Nano 8 GB, recorded `power_mode_ok=false`, and had hint coordinates equal to truth. Peak reported shared memory was 7146 MiB. This is valuable in-sample evidence, not validation on the final board or displaced hints.

## Verification and delivered files

- Existing aggregate suite: **73 passed, 1 failed**, the empty-shortlist reporting failure above.
- Existing Iteration 7 tests, which the aggregate runner omits: **10 passed**.
- New integrity regression tests: **9 passed**.
- New fast-edge tests: **2 passed**, including three full synthetic pipeline runs.
- Original end-to-end tests with the isolated report wrapper: **2 passed**.
- 300 real-image candidate dumps replayed through fixed policies; original refit control reproduced all scores. Frozen-judge M=5 context availability corrected and checked.
- Historical 500-image Jetson output independently re-scored with the supplied harness scorer.

Start with `README_codex.md` for commands and opt-in use. Logs, per-photo outcomes, source hashes, branch snapshots, split evidence and benchmark records are under `results/`. Existing branches, source, settings, notebooks, models and submissions were left as found. Remove only `tests/Review_20261113_codex/` to revert this review's workspace additions.

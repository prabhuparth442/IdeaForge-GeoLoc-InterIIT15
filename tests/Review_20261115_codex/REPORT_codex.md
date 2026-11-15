# Current code review — 15 November 2026

The engineering and experimental documentation improved since the previous review, but the new cache and relative-inlier replay have correctness defects. Treat the saved gains as research evidence; do not promote the relative judge or cache to deployment until the issues below are fixed and the affected experiments rerun. The current tracked production pipeline is unchanged.

## Scope and current state

- Research repository: `prabhuparth/dev`, `f914ca5`, unchanged tracked HEAD. Reports saying `palash/dev` describe the other development environment, not the local checkout.
- Main_Pipeline_Headless: `master`, `e5e0fd3`; the same pre-existing dirty settings/pipeline/coarse/validation files and untracked FFT module remain. Its production Python hashes match the previous review snapshot. Submission hard links remain a reason to avoid in-place exploratory edits.
- Of 147 files in the previous source manifest, only `train/checks/candidate_judge.py` changed. Iterations8–11 and additional check modules/tests are new untracked research work.
- Read the new reports, protocols, executed-run outputs, notebook source cells, candidate judge, scale robustness, stage cache, synthetic satellite renderer, holdout/evaluation helpers, perturbations and test runners. Independently re-scored 20 final-policy per-photo CSVs, 300 rows each; these are repeated policies over two draws, not 6,000 independent images.
- GPU unavailable (`nvidia-smi` cannot communicate with driver). No fresh LoFTR/DINO sweep or target-board benchmark was performed. T10/T11 raw dumps, cached arrays and signal tables are absent locally, so their networks, fitted judges and full candidate replays cannot be reproduced from this checkout alone. Final decision CSVs and notebooks are available.

## Findings, in priority order

### 1. [P1] Relative-inlier replay consumes queries that have not run

`train/checks/scale_robust.py:44–75` computes a photo's far-candidate level from **all queries** in the supplied signal table. `running_level` includes that complete current-photo level. T10 final notebook cell2 and T11 final notebook cell2 call `add_relative` once on the full dump before invoking the adaptive/early-exit decision replay.

An early Jetson answer therefore depends on fine matches from queries it purportedly skips; even a PC standard-six decision can depend on the later wide-query results. After that photo, its all-query level also feeds later photos' state. This is an inference-availability problem, not use of truth labels. A deployment that actually computes every required query would lose the claimed early-exit cost reduction; one that does not cannot calculate the same features.

CPU reproduction holds the first query fixed and changes only query26: the first candidate's `rli` changes from **2.194296 to 2.130581**. This proves a causal dependency; it does not quantify how many saved real answers change.

**Fix:** define state updates over actually executed queries and chronological photos. Either score the current photo from a previous-photo state, updating only after its actual run, or update within query execution. Rebuild training features, calibration and replay using that same causal contract. Recheck both accuracy and query counts. S1, which does not use relative features, is not affected by this specific dependency.

### 2. [P1] Persistent stage cache key omits inputs that determine its result

`train/checks/stage_cache.py:57–70` hashes only image **basename**, level, explicit geometry overrides and hint rounded to 0.1m. It excludes the physical image/path/content, camera, altitude/yaw/pitch/roll, base geometry, map identity, coarse configuration, K and DINO model identity. Cached output contains rendered pixels, map search and DINO scores, so those omissions can silently reuse an incompatible result.

Reproduced: `/flight_a/frame.png` at height100/yaw0 and `/flight_b/frame.png` at height200/yaw90 produce the identical key at the same level and prior. Changing K or a model while reusing a cache directory likewise does not invalidate the stored shortlist.

**Fix:** persist and verify a versioned input/configuration manifest with full image identity/content hash, exact pose/camera/prior, map/preprocessing identity and all upstream algorithm/weight settings. M/selection may intentionally stay outside the key because selection is performed after loading the full shortlist. Fine-matcher settings may also stay outside this upstream cache. Write entries atomically. Existing speed/equality checks on unchanged inputs do not test invalidation.

### 3. [P1] Shared FFT fingerprint can identify different windows as identical

`train/checks/stage_cache.py:108–123` keys FFTs by tensor shape, FFT size, a sampled sum and the full sum. Sums do not uniquely identify spatial content. Two unsampled cells can swap values, retaining the entire fingerprint while changing their correlation map.

Reproduced with CPU tensors: first window exactly matches the original correlation; after swapping two unsampled values, the patched function returns a stale result with maximum ZNCC error **0.017955**. This is deterministic, not a speculative hash collision.

**Fix:** tie the cache explicitly to a photo's immutable extracted window plus map/prior/level identity and clear it at the end of that photo, or use a proper content digest when content-based reuse is required. Also include dtype/device if sharing across them. Avoid converting the whole window to float64 and reducing it on every query, which adds allocation and synchronisation overhead. Re-run equality tests on invalidation and adversarial equal-sum windows, not just repeated identical windows.

### 4. [P2] Scale augmentation undoes M=5 context isolation during training

`train/checks/scale_robust.py:90–94` calls `scale_signals(base, f)` without passing M. `scale_signals` reconstructs `add_context(..., m=None)`. In the Jetson training path, the input originally has M=5 context, but every augmented copy except f=1 rebuilds context using all ten stored candidates. The transform supplied by the notebooks only adds relative features; it does not repair `zgap`/`ligap`.

Reproduced: a candidate's matched-only `zgap` is **7.5**; the augmented training copy uses rank7 as a rival and changes it to **0.1**. Deployment/evaluation features continue to use M=5, creating a training/evaluation mismatch. Both T10 V5 and T11 S5 use this augmented training path. The earlier context fix is correct for direct callers, but incomplete for augmentation.

**Fix:** pass the candidate budget through `fit_judge` to every feature-rebuilding call, or filter training to the available candidates before augmentation. Refit the affected models and calibrate their final deployed feature contract.

### 5. [P2] Zero-wrong threshold selector still returns an unsafe fallback

`train/checks/candidate_judge.py:236–244` still returns the largest threshold when **no threshold has zero wrong answers**. T11 It1 cell12 calls this helper to select its zero-wrong variants. The newer `ltt_tau` fix does not repair this separate path.

Reproduced: scans with one wrong answer at every cut return **0.99**, despite 0.99 also being unsafe. This does not prove the delivered T11 design scan hit this fallback; it demonstrates incorrect behavior on a valid failure input.

**Fix:** return an explicit failure and convert it to `NO_ANSWER`, as the LTT callers now do. Do not pass an unconverted `None` into `decide`, because that still intentionally selects the bar baseline.

### 6. [P2] Early-exit API can bypass the selected acceptance cut

`train/checks/candidate_judge.py:168–178` accepts at `early_tau` without enforcing `tau`. A caller supplying main cut0.9995 and early cut0.98 accepts a probability0.985 candidate in the reproduction, including a deliberately incorrect one.

T11 final evaluation correctly uses `max(0.98, cut)`, so this finding is **not an explanation for its delivered final errors**. The shared API and other callers remain vulnerable to that configuration mismatch.

**Fix:** enforce `max(tau, early_tau)` internally, or validate the policy contract before running it. Calibrate the exact early-exit/override/wide policy that will be evaluated; do not use a different policy for calibration and then label the modified one certified.

## Satellite-only lineage and generalisation

T11 is materially better than calling a drone-trained judge synthetic-only: the S1/S5 judge fitting uses satellite-rendered candidates from sceneB/Pleiades. However its documented development process is not the stricter satellite-only fitting/calibration protocol used earlier in this session:

- `Sat_Photo_Dump` calls the official pipeline setup, using the official camera, historically fitted geometry and truth-derived map setup.
- Rendering strength was chosen by comparing synthetic inlier distributions against official/NUDT thermal results (explicitly reported in T11).
- It1 chooses acceptance thresholds on NUDT thermal candidate outcomes; the renderer exclusion region is also derived from official locations.
- Existing DINO/MINIMA feature extraction remains pretrained with separate provenance.

T11's README explicitly treats thermal threshold calibration as allowed. This review records that methodological difference; it does not infer a new user permission from the README. **Satellite-derived weight-training examples** is a supportable claim; **the complete model-development process used no thermal-derived fitting** is not. If thermal calibration is an intentional future protocol, label it separately rather than equating it with the earlier strict protocol.

The official TRY/TEST split is spatially grouped and more useful than random frames. But locking previously examined frames cannot make them historically unseen: the project and prior review already examined official outcomes, and shared footprints remain. NUDT808/909 are fresh photo draws from the same existing flight/site pool, not independent acquisitions or unseen domains. Their exclusion/identity lineage cannot be fully rechecked here without the missing canonical raw dump/signal inputs. Photo bootstrap intervals are conditional regression estimates, not new-flight uncertainty.

Candidate AUC around .998 can coexist with unacceptable mistakes after selecting a winner from many candidates. It measures overall ranking across an imbalanced candidate population, not the wrong-answer rate of the final accepted-photo policy. The provided CSV results, not AUC, should decide promotion.

## Results independently checked

Every policy CSV has 300 unique `(data, photo)` rows. Recomputed score, accepted count and penalised wrong count agree with the delivered final tables for all20 outputs. Here "wrong" is the existing report's negative-score definition (>8 scoring sigma); 4–8 sigma answers receive zero and are not counted as negative mistakes. This is arithmetic verification of saved decisions, not rerunning the neural pipeline.

| Experiment | PC baseline | PC primary | Jetson-model baseline | Jetson-model primary |
|---|---:|---:|---:|---:|
| T10, NUDT808 | 52.08 / 5 wrong | V5 54.48 / 0 | 52.68 / 2 | V5 53.00 / 0 |
| T11, NUDT909 | 54.24 / 2 | S5z 53.52 / 1 | 52.72 / 4 | S5z 53.28 / 1 |

T10 J2 (non-relative, no override, cut.98) scores54.20/0 on PC808. It avoids the relative-feature dependency, but is trained on thermal candidates, so it is not a satellite-only model. T11 non-relative S1z Jetson-model result is52.88/0 versus52.72/4, gain+0.16 with reported95% photo interval[-1.56,+2.04]. That is a useful regression safety observation, not a proven deployment score gain or general zero-error claim. The primary S5z criterion failed in T11, as its report correctly acknowledges.

The headless code's actual scheduling differs from these Jetson-labelled replays: zero-heading-first order, wall-clock22s guard, and no default query-by-query early return on the second pass. The replay has a ten-query cap and stops after qualifying extra queries. Research matches use.22 vs current headless.21. New manifests document several differences, but do not make query-count estimates into target-board measurements. T11 final manifests also omit executed `wide_cut`, `early_exit` and baseline bars despite those being in notebook code; save the actual complete policy kwargs and weight/source hashes.

## Tests and persistent gaps

- Existing aggregate suite: **73 pass, 1 fail**, same `evaluation/report.py:34` empty-sequence `min()` crash as before. T10 fixed candidate-table replay handling, not the original report path.
- Manually ran omitted tests: agreement3, inflation-guard4, Iteration7 ten, Iteration8 three, Iteration10 four: **24 pass**.
- The aggregate pipeline runner's hardcoded script list still omits those files. Run them explicitly or register/discover them; a green aggregate alone cannot exercise the new evaluation helpers.
- Six targeted CPU reproductions confirmed findings1–6. These are expected-bad-behavior reproductions, not claims that those defects were fixed.
- No production edits or new learned fitting. Existing tests internally fit synthetic toy models only. All new review outputs are inside this `_codex` folder. No threshold/model tuning against T10/T11 real outcomes was performed.

## Recommended order

1. Repair cache identity/invalidation and causal query availability before reusing caches or timing V5/S5.
2. Preserve M consistently through augmentation; fix failed zero-wrong selection and the early-exit contract.
3. Register omitted tests and integrate the existing empty-report fix.
4. Rebuild/cache only after a versioned source/map/image/weight manifest is frozen. Refit/recalibrate relative/augmented models under their exact inference contract. Keep the weight and calibration lineage explicit.
5. Retain T11 S1z as a comparison candidate, not a deployed safety guarantee. Validate on the actual headless scheduling, displaced hints and target hardware before promotion. Use existing evaluated sets for regression; reserve a genuinely separate acquisition for final claims.

## Delivered evidence

`reproduce_findings_codex.py` is a standalone CPU reproduction script. `audit_snapshot_codex.py` inventories current sources and independently re-scores the saved final decisions. `results/` contains all logs, six reproduction results,20 re-scores, the current279-file source/report/notebook snapshot, git states and the difference from the previous147-file manifest. Original sources and original result files were preserved.

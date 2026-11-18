# New high-altitude dataset review — 18 November 2026

The new set needs a resolution-aware structural matching path, rather than a generally lower inlier bar. The main change is around800m on this camera; above1600m both retrieval and local matching become unreliable. These are observed operating regions, not universal thresholds or proof that altitude alone causes the failure.

**Follow-up: Iteration29 is now available and rejects the tested coarse structural-channel route.** Coarser orientation, ridge/local-contrast and DoG channels, plus their fusion, were weaker than the existing edge search. The any-channel agreement gate accepted only13/142 development frames at≥800m (9.2%,0 wrong), failing the preregistered40% coverage requirement. It also accepted9/150 old official and50/150 NUDT frames,0 wrong each. This should demote generic channel fusion from the priority recommendations below. Those sections record the original hypotheses, not proof that the newly tested method works. Local phase-congruency/RIFT or held-out-region registration are different, still untested hypotheses; the negative result is not a reason to assume they will succeed.

## What was checked

- Read current repository instructions, handoff and Iteration12 results, including the newest coarse-confidence/evidence/refinement work. Production HEAD is3ecbbe0; existing local edits were preserved.
- Reviewed all500 metadata rows, restricted detailed analysis and image viewing to the332 development photos, and checked that Stage_Evaluation row heights match the development metadata order. The168 locked images were not opened, fitted on or re-evaluated. Previously published locked-test outcomes were not used to select any new rule.
- Actually viewed14 raw development frames: two per existing height band, chosen by altitude quartiles without consulting success/failure. The figure uses independent1–99% display stretching; it does not reconstruct temperatures.
- Executed six analysis cells in Highalt_Review_executed_codex.ipynb, recomputing height summaries and existing coarse/refinement scores from saved CSVs. No neural inference, model fitting, new threshold sweep or new pipeline experiment occurred. The extracted high-altitude map/store is on the server; local satellite files are compressed. This review does not claim to rerun its full geolocation pipeline.
- New analysis lives here. Only the required append-only repository documentation entries were added outside this folder. No src/, changed_src/, weights or original result files were modified. Notion mirroring is pending because no Notion connector is available in this session.

## Measured failure pattern

These are the **single-query Stage_Evaluation K100/M25 diagnostic**, not the different headless500-photo baseline or the six-query K50/M5 deployment recipe. A retrieved target means a shortlist candidate within4 scoring sigma, not an exact image/centre match.

| Height | Development frames | Target retained in M | Answered | Score/60 | Wrong beyond8σ |
|---|---:|---:|---:|---:|---:|
| <800m | 190 | 98.9% | 97.9% | 48.60 | 0 |
| 800–1600m | 100 | 92.0% | 53.0% | 21.90 | 0 |
| 1600–2017m | 42 | 64.3% | 4.8% | 1.29 | 0 |

Single-query total:34.57/60,0 wrong,241/332 answers. Six standard queries:35.49/60,0 wrong,247 answers. Wide27:35.96/60,0 wrong,251 answers. Extra headings therefore do not address most of the remaining loss in this saved experiment. Do not compare these figures directly with36.35/60 from a different headless500-image run.

The separate truth-known one-query diagnostic is even more informative:

| Height | Frames | Median inliers at supplied truth | Zero inliers at supplied truth |
|---|---:|---:|---:|
| 800–1000m | 12 | 26 | 2 |
| 1000–1300m | 43 | 10 | 14 |
| 1300–1600m | 45 | 5 | 19 |
| 1600–2017m | 42 | 0 | 38 |

This supports a matching/representation limit, not just a poor winner selector. The diagnostic uses one satellite tile, a particular preprocessing/store and one query; map boundaries and geometry are potential confounders. It does not prove the network alone is responsible. The full-map Stage_Evaluation independently shows the high-altitude inlier collapse.

## What the images suggest

At low heights, individual dark spots and fine tracks are distinct. At greater heights, many spots become tiny, while broader streaks, drainage-like patterns and regional texture remain. A strong radial shading pattern and a recurring hook-shaped object at the lower-right stay tied to camera pixels across different ground views. They are poor geographic evidence; fixed camera artefacts should be excluded or modelled before structural comparison.

These observations are consistent with loss of repeatable local detail, plus a changed terrain/modality distribution. They are not measurements of tree dimensions, calibrated temperatures, or proof that every dark spot is vegetation. Altitude also correlates with flight position in this acquisition, so scene and altitude effects are not fully separated.

See results/drone_images_by_height_codex.png and the saved filename manifest. No locked image is included.

## Why the height matters physically

The new camera has fx≈202.4px. Nominal ground sampling distance is h/fx: about4.0m/photo-pixel at800m,6.9m at1400m and9.9m at2000m. Small structures can cease to be individually resolved; upsampling cannot restore that information. For comparison, NUDT's higher-resolution camera at300m has approximately0.19m/photo-pixel, so a numerical altitude threshold cannot transfer unchanged between cameras.

At level11, the map canvas has13.58m/pixel. A coarse score cell spans4 canvas pixels, or54.3m. At1400m the scoring sigma is about22.2m: grid spacing alone is about2.4σ. A correct region can therefore still lose precision points until its centre is refined. This does not explain all residual error; map registration, image formation and pose can contribute too.

Nominal level selection requests level12 above≈1924m and level13 above≈2720m for this focal length and320px width. Those thresholds ignore tilt, heading-dependent footprint, the1.04 height scale and mask coverage. They should be calculated using the actual camera/footprint, not fixed from this dataset. The collapse starts near1600m, before the nominal level11 cap becomes limiting: **adding levels alone is insufficient**. There are no real observations above2017m here;3000m feasibility is unproven.

## Recommended design by operating region

### Below roughly800m: retain local matching, improve precision

Keep the present retrieval + DINO + MINIMA LoFTR verification. Retain existing match scale/angle/shift gates. Investigate finer local alignment and camera-centre stability when matches have spatial support; require that refinements improve or preserve the old official and NUDT behavior. Historical pointing-bias fits explain only a modest fraction of the residual error; do not fit a separate position correction to each altitude band.

### Roughly800–1600m: coarse location hypothesis, structural local verification

1. Preserve strong edge-search candidates even when a semantic reranker prefers another location. DINO's raw top-place behavior on the new set differs from the old urban set: saved coarse argmax within4σ is90.1% on new and74.7% on old; DINO top1 is87.3% and88.0%, respectively. Its value is scene-dependent. The existing DINO-off-above800 experiment saved time without increasing score; it is a budget source, not an accuracy improvement by itself.
2. Refine the chosen structural peak locally at a matched/native resolution, using the demonstrated two-level-finer search and subpixel interpolation. Treat the refined coordinate as a hypothesis, not an automatic answer.
3. Verify that hypothesis with **dense, multi-scale structural channels**: gradient orientation, band-pass ridges and/or phase congruency, rather than requiring LoFTR correspondences for every accepted image. The aim is to compare the visible spatial arrangement of larger structures when individual keypoints become weak.
4. Split visible ground into separated regions: estimate the small residual translation/rotation/scale on some regions and check whether it predicts the others. Apply the same procedure to the strongest distant challenger. This is a proposed direct-registration verifier, not the earlier universal held-out-LoFTR-pairs gate that lost too many correct answers. Regions/channels share evidence and must not be counted as statistically independent votes.

RIFT is a relevant non-trained baseline because it uses phase congruency and descriptors designed for multimodal radiometric variation; its sparse detector may still fail on feature-poor high-altitude views. Dense phase/ridge alignment should be tested alongside it, not assumed equivalent. [RIFT paper](https://ljy-rs.github.io/web/Files/TIP2020.pdf). Bounded rotation/scale/translation alignment has a classical Fourier/log-polar route; it can provide another diagnostic or local registration baseline, but does not automatically solve cross-modality mismatch. [scikit-image registration example](https://scikit-image.org/docs/0.25.x/auto_examples/registration/plot_register_rotation.html).

### Above roughly1600m: strengthen retrieval and require regional evidence

Local matching is frequently absent even at the supplied truth, while retrieval also falls. Use a coarse-to-fine hierarchy over larger landform/track/texture arrangements, with an explicit missing-location/ambiguous outcome. Correct the level cap and test resolution-matched map filtering. Reject a result when its alignment is broad/unstable or a distant region explains the image comparably well.

An optional second representation can use the geometry of **clusters** of dark structures and long ridges, rather than matching every tiny spot. This is a hypothesis about stable regional layout, not a recommendation to identify every speck as a tree or assume optical/thermal contrast ordering matches. At the most weakly textured heights there may simply be insufficient evidence in a single320×256 photo.

If ordered frames or independently measured motion are available and allowed, multi-frame geometry can add new constraints; correlated neighbouring crops do not provide independent confidence. The current interface is single-photo, so no temporal method is assumed available for submission.

## What the existing experiments already establish

- **Coarse confidence alone is unsafe across domains.** New-set wrong coarse locations have maximum z5.76, but old official reaches14.96 and NUDT15.92. The apparent success of z≥6 on the desert development set does not transfer. Do not ship a terrain-specific z cutoff or choose it from this review.
- **Two-level-finer coarse refinement is a useful building block, not a safe answer rule.** On142 development photos≥800m, forced coarse answers improve3.13→3.63 points/10 and within1σ24.6%→28.9%. The16 existing wrong answers remain16, with no newly wrong photos. Across all332,34.23/60→36.99/60,16 wrong in both. Old official improves37.58/60→40.56/60,52 wrong in both. These deliberately forced-answer scores are not deployment scores; an acceptance gate is still needed. NUDT has only the parabola variant, not the two-finer-level variant, in these local tables: full three-set validation is incomplete.
- **Four-level refinement is not supported:**20 wrong instead of16 on the new set; four previously non-wrong photos become wrong.
- **Coarse + four matcher inliers is conservative but incomplete:** the existing evidence experiment rescues only11/85 skipped high-altitude photos. It cannot solve frames with no matcher evidence. Its threshold was developed on these sets; it is not an independent guarantee.
- **Generic bigger canvases, cleaning-only and peak averaging are weak bets.** In the supplied-truth diagnostic, t448 reduces the number of142 high-altitude photos with≥8 true-location inliers from52 to30; cleaning/local normalisation improves some coarse behavior but does not increase that true-location match count. The prior coarse-peak averaging experiment was already killed. Avoid repeating these without a distinct hypothesis.

## Next experiments and kill rules, written before any new run

No new method is measured by this review. Proposed order:

1. **Integrate a refined structural fallback with the existing baseline.** Keep its proposal and acceptance separate. Preserve existing answers unless a replacement has stronger validated evidence. Verify the complete policy on old official, NUDT and high-altitude development data, with a separate negative-map test.
2. **Add regional dense verification to missing/weak-LoFTR cases**, comparing structural channels against the existing edge representation at the same search/candidate/runtime budget. Include far challengers and null/no-truth shortlists. Kill if false acceptance occurs in the preregistered wrong-map controls, or if coverage/points do not improve at an equivalent wrong-answer target on any required dataset.
3. **Check the high-resolution limit physically:** correct levels/footprint clipping and simulate reduced resolution using held-out satellite scenes with known transforms. Synthetic altitude doubling is only a stress test, not proof of real3000m performance. Kill if gains require broadening acceptance until the old/NUDT controls fail.

Freeze transforms, thresholds, weights and runtime policy before independent real evaluation. Under the session's strict satellite-only learned-development protocol, any new encoder/ranker and its model selection/calibration must use satellite-derived partitions, not drone outcomes. Current real datasets are regression/development evidence, not a new blind validation set after repeated inspection. Do not use the168 locked photos to choose a routing height, confidence threshold or channel. No perfect zero-error selector is established by finite observed samples.

## Deliverables

- Highalt_Review_codex.ipynb and Highalt_Review_executed_codex.ipynb: actual saved analysis outputs and image contact sheet. Execution used sequential IPython with rich display capture because the kernel-server path did not start here; no outputs were fabricated.
- review_blocks.py: small read-only data/physics aggregation helpers; no production replacement.
- results/: height-band metrics, query variants, viewed-image manifest, resolution table, recomputed coarse results, truth-known matcher bands, figures and input hashes.

This review identifies promising next tests; it does not claim a new measured pipeline gain.

## Follow-up check of reported suggestions: Iteration29

Read the executed `tests/Test_Iteration_12/results/coarse_channels/CH_analysis_executed.ipynb`, its source notebooks and all three saved channel tables. Independently recomputed the agreement gate using the same best-z-per-photo/channel selection and≤1σ agreement rule. Results are saved in `results/iteration29_crosscheck_codex.json` and agree with the executed report: new332 accepted144/332 overall and13/142 at≥800m; old official9/150; NUDT50/150; all0 wrong beyond8σ.

The actual experiment used three headings at scale1.0, channels built from4×4-cell means/std with fixed additional blur, and full-disc masked ZNCC. At800–1300m, within2σ is71% for edge versus29% for the best added channel. Above1300m, edge is52%, while the added individual channels are approximately0–1%. Fusion worsens retrieval. Zero decoy agreements is observed alongside very low acceptance; it is not evidence of a useful safe detector. The decoys blank the supplied truth region in the same search map, rather than using an entirely different map region.

Updated priorities: do not repeat the same orientation/texture/DoG fusion or promote the killed gate. Keep two-level coarse refinement as an existing precision building block, retaining its known wrong-answer limitations; the matcher-support fallback is conservative but rescues only13% of high-altitude skips. Before using any coarse-only acceptance policy, complete a true absent-location/wrong-map control. RIFT/phase registration and local region-prediction checks remain untested in these local reports and should be bounded pilots only, not presented as established improvements.

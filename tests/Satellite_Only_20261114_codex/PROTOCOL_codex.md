# Frozen satellite-only pilot protocol

Written before training or real-image evaluation, 2026-11-14.

Train a small structural encoder from random initialisation using only raw pixels and raster metadata from official satellite scene 268234931/BAND.tif. No drone image, pose, label, candidate table or existing learned judge enters training, model selection or calibration. NUDT satellite imagery is excluded from training and calibration too. Existing frozen DINO/MINIMA remain reference pipeline components; their original training is not claimed to be satellite-only.

Divide the source raster into x stripes: training [0,.55], validation [.60,.75], calibration [.80,1]. Entire sampled footprints must fit inside the stripe, plus a 256-pixel inset. Draw 512/128/256 crops respectively, footprint sides uniformly 700–2400 native pixels, resampled to 224. Reject nodata/near-flat patches using satellite pixels alone. Save each exact source window. This is a small feasibility experiment, not a claim that one map covers thermal diversity.

Train a shared compact CNN on polarity-invariant structural channels with contrastive pairs and in-batch negatives, 12 epochs, batch 32, fixed seed 141126. Synthetic views use fixed rotation, scale, translation, contrast, polarity, blur and masking ranges. Select the checkpoint with lowest satellite validation contrastive loss. No real-image-dependent retries or hyperparameter selection.

Calibrate a joint cosine/margin abstention gate using synthetic positive candidate sets and null sets (true patch omitted), 50 candidates per set. Choose thresholds only on the separate satellite calibration stripe: reject every incorrect top selection and null selection in that finite calibration set, then maximise retained correct selections. This is empirical calibration, not a population guarantee. Do not describe zero synthetic mistakes as zero real-world risk.

Freeze weights, calibration, source hashes and training manifest before reading drone evaluation inputs. Evaluate an outcome-independent seeded pilot of 6 official and 6 NUDT test frames. Generate deterministic priors uniformly in a 5 km disc; never pass truth coordinates to the localisation decision. Run all stages of the existing headless pipeline: projection, full 5.5 km coarse search, DINO K50/M5, MINIMA .21, RANSAC, six queries and twelve-query fallback, bar8. Disable the 22-second guard on CPU; these results do not establish deployment timing. Use memory-mapped official store, and build a full-map NUDT store at levels3/4. NUDT camera dimensions come from its documented sensor; the existing NUDT geometry recipe (scale .8, no attitude) is a frozen historical baseline, not newly fitted here.

Compare on exactly the same computed candidates:

- Baseline: unchanged headless decision and escalation.
- Satellite veto: accept its winner only if the satellite-only cosine/margin gate passes.
- Satellite selection: consider each location with >=8 inliers, or >=4 inliers and agreement from two query bests (each >=3) within one scoring sigma; apply the same satellite gate against other distinct locations, then choose by the existing inliers+DINO ordering. This adds a possible recovery path as well as abstention. No change to baseline query scheduling, so this is a decision-layer experiment, not an independently adaptive search policy.

For the margin, competitors are candidates farther than one scoring sigma from the candidate's refined position. No competitor means abstain. Scores use encoder cosine on each query and its candidate tile with the same valid mask. All gate parameters remain fixed after satellite calibration. Failures/skips stay in the denominator. Report points, answers, >4-sigma inaccurate answers, >8-sigma penalised answers, timing, paired uncertainty and all per-photo results. Six photos per domain cannot justify a score-gain or zero-risk claim; it verifies feasibility and catches major domain-shift failures. Keep all changes under this suffixed folder, leave production files untouched.

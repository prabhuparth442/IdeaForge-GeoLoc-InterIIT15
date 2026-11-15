# Satellite-only encoder: real end-to-end pilot

This is an actual raw-drone-image → answer run, not a replay of previously saved candidate tables. All work is isolated in this folder. No production changes were promoted.

## Results

| Dataset | Decision | Photos | Answers | >4σ wrong | >8σ penalised | Score / 60 | Δ score |
|---|---|---:|---:|---:|---:|---:|---:|
| official | baseline | 6 | 6 | 0 | 0 | 60.00 | +0.00 |
| official | sat_veto | 6 | 1 | 0 | 0 | 10.00 | -50.00 |
| official | sat_select | 6 | 1 | 0 | 0 | 10.00 | -50.00 |
| nudt | baseline | 6 | 5 | 0 | 0 | 46.00 | +0.00 |
| nudt | sat_veto | 6 | 0 | 0 | 0 | 0.00 | -46.00 |
| nudt | sat_select | 6 | 1 | 0 | 0 | 10.00 | -36.00 |

Scores include every scheduled frame; skips and failures receive zero. Wrong here means error >4 times the altitude/focal-length scoring sigma; >8 sigma receives −20 raw points. The scoring sigma (metres) is unrelated to statistical uncertainty in mean score.

## Decision

Do not promote these gates. The experiment demonstrates successful satellite-task learning but poor transfer of its candidate scores and calibrated acceptance gate to actual thermal views. Keeping a low wrong-answer count by discarding correct answers is not an improvement. No threshold was loosened after examining drone results. This rejects this specific small-model/gate configuration, not all possible satellite-only training approaches.

Highest encoder-cosine candidate diagnostics (before the retained geometric safeguard):

- official: 3/6 highest-cosine candidates were >4σ wrong; maximum error 5495.7 m.
- nudt: 4/6 highest-cosine candidates were >4σ wrong; maximum error 3226.8 m.

## What was trained

A 117,952-parameter CNN, randomly initialised. Only raw satellite scene 268234931 was read by training. Geographic stripes contained 512 training, 128 validation and 256 calibration crops with buffered, non-overlapping split footprints. No NUDT map was used to train or calibrate, so NUDT is a cross-map test for these new weights. Official is same-map adaptation, not an unseen-map claim.

Validation contrastive loss: 2.0851 → 0.0678. Selected checkpoint: lowest satellite validation loss over 12 epochs. Synthetic calibration retained 216/256 correct positive sets with no accepted incorrect top selections or null sets among the 256 null sets. Frozen gate: cosine ≥ 0.79, margin ≥ 0.065. This is finite-sample empirical calibration, not a certified thermal risk bound.

The encoder uses polarity-invariant gradient channels; fixed synthetic appearance/geometric augmentations are not real thermal rendering. The full pipeline still contains existing frozen DINO/MINIMA weights with separate, not-certified-satellite-only pretraining. No F4/T7/old synthetic judge weights were used.

## What was run

The current headless pipeline was imported from Main_Pipeline_Headless. It ran projection → full 5.5 km edge search → DINO K50/M5 → MINIMA LoFTR (.21) → RANSAC → decision. Standard six-query search and the existing twelve additional-query fallback were enabled. The new decision variants shared the exact same candidate computations and baseline query scheduling. This is the headless bar8 baseline, not the research AgreementQueryPipeline K50/M10/27-query baseline.

Official uses its existing memory-mapped map store. NUDT uses a new full-reference-map store at levels3/4, generated without drone locations. The adapter fixes NUDT camera width/height to 1280/1024 in this process and uses the historically established NUDT scale .8/no-attitude projection. No geometry was fitted in this experiment.

The selected six images in each domain and synthetic 5 km priors were generated with fixed seeds, independent of outcomes. Only the evaluator sees truth; localisation receives image, camera, attitude, altitude and perturbed hint. These are regression datasets previously used by the project, not a newly blind flight acquisition.

## Limits and timing

CUDA was unavailable. The CPU accuracy run disabled the 22-second guard and used DINO batches of five to bound memory. This is not a deployment-time or 30-second-scored benchmark. The timing includes the small extra encoder; stage timing and encoder timing are separately logged. The host was shared with preprocessing/other work, and the negative-control run overlapped the final NUDT frame; latency is descriptive, not an isolated speed benchmark. Six frames cannot establish zero wrong-answer risk, optimal score, or meaningful cross-flight ±σ uncertainty.

- official: 49.22 seconds/frame mean, added encoder 0.046 seconds/frame, 6/6 over 30 seconds, 0 execution failures.
- nudt: 56.71 seconds/frame mean, added encoder 0.060 seconds/frame, 6/6 over 30 seconds, 0 execution failures.

Paired photo-level uncertainty (descriptive only, correlated frames; not a guarantee):

- official sat_veto: Δ -50.00 ± 10.00 points (one photo-level SE); percentile bootstrap 95% [-60.00, -30.00].
- official sat_select: Δ -50.00 ± 10.00 points (one photo-level SE); percentile bootstrap 95% [-60.00, -30.00].
- nudt sat_veto: Δ -46.00 ± 10.00 points (one photo-level SE); percentile bootstrap 95% [-60.00, -26.00].
- nudt sat_select: Δ -36.00 ± 12.00 points (one photo-level SE); percentile bootstrap 95% [-56.00, -16.00].

## Separate negative control

Added after the valid-hint pilot had started, as a diagnostic only. The first preselected official frame was evaluated with its hint at the farthest inset map corner, placing its truth outside the search disc. This deliberately violates the normal ≤5 km hint assumption; it is excluded from the accuracy table and never used for fitting.

Truth-to-hint distance: 9396.0 m. Queries executed: 18; fallback used: True. CPU seconds: 118.9.

- baseline: abstained; >4σ wrong: False.
- sat_veto: abstained; >4σ wrong: False.
- sat_select: abstained; >4σ wrong: False.

One absent-truth control exercises rejection behavior but cannot establish a false-accept probability.

## Integrity and reproduction

Original source hashes checked: 147; changed: 0. Seven isolated unit tests cover split-footprint separation, frozen hashes, polarity invariance, conservative gates, ambiguity, independent-query agreement and failure scoring. Weights, thresholds and training source hashes remain frozen.

See README_codex.md for commands; results/ contains per-photo candidates, stage timings, manifests, training log, calibration tasks, test log and summary JSON. The large generated NUDT .npy cache is ignored by git and can be rebuilt or removed with this experiment folder.

# Satellite-only constraint — 14 November 2026

User requirement: avoid fitting the official drone dataset; any new encoder, NN or learned decision model must be trained on satellite maps, not drone images. This supersedes the earlier suggestion to train a joint judge on flight-labelled candidate tables. The existing T7 judge and the stricter policy tested in this directory remain historical comparison baselines, not compliant newly learned models.

## Allowed development inputs

- Georeferenced satellite imagery and its raster metadata.
- Synthetic views and labels derived from known transformations of satellite crops.
- Fixed camera-model assumptions or documented physical specifications, not parameters fitted to drone-image ground truth.
- Satellite-only training, validation and calibration partitions, established before model fitting.

No drone images, candidate labels, features, outcomes or ground-truth locations may select weights, checkpoints, augmentation strengths, candidate-mining regions or thresholds. Do not initialise a new strictly map-only model from the existing NUDT-trained F4 or T7 weights. A new strict encoder starts from scratch unless its pretrained-weight lineage independently satisfies the same restriction. Existing DINO/MINIMA may remain frozen reference pipeline components, but a model using them as teachers/features must disclose that its full lineage is not established as satellite-only.

## Why the previous synthetic experiment is not compliant

Verified in the current source:

- `tests/Test_Iteration_1/stages/synth_dump.py` loads real training candidates and uses their success rates to select synthetic augmentation strength.
- `tests/Test_Iteration_1/stages/synth_fusion.py` uses real validation candidates to select the best synthetic-trained model; the mixed branch also directly trains on real candidates.
- `tests/Test_Iteration_1/stages/synth_setup.py` loads `f4.pt` from the real-data run, uses fitted geometry and reads drone locations for sampling exclusions.
- `train/features/learned_net.py` documents F4 training on NUDT thermal photos and satellite crops.

Those experiments may be useful research, but their “synth” labels do not demonstrate the stricter satellite-only protocol. Do not reuse their trained artifacts or real-data-calibrated augmentation settings as if they did.

## Proposed next model

Train a small structural encoder and a candidate-set decision head entirely from satellite-derived tasks:

1. Read map patches from predetermined geographic blocks without consulting drone metadata. Keep complete crop footprints inside their assigned block; reserve buffers at least as large as the maximum transformed footprint to prevent train/validation overlap. Multiple dates of the same place remain in the same partition.
2. Produce paired views with known crop/rotation/scale/perspective transforms. Vary grayscale mapping, local contrast, polarity, blur, resolution, occlusion and edge dropout using ranges fixed independently of drone results. These are appearance augmentations, not measured or physically reconstructed thermal images.
3. Train descriptors to preserve corresponding structural content despite those changes. Include negatives from visually similar but geographically distinct buildings, roads and fields; easy random negatives alone are insufficient. Mine training negatives only within the training partition.
4. Train the decision head on candidate sets, including sets where the true map patch is absent. Let it select one location or an explicit “none” option. Supply relative geometry/appearance evidence; avoid raw coordinate, scene-ID or sample-ID shortcuts.
5. Select checkpoints and all decision thresholds on held-out satellite-derived tasks. Verify that harder corruptions and missing-candidate sets increase abstention rather than confidence.
6. Freeze architecture, weights, transforms and thresholds before evaluating real drone data. Keep the original pipeline as the unchanged reference. An observed drone failure may diagnose a limit but must not trigger retuning and rebranding that same set as an independent test.

A structural descriptor trained across transformed satellite views is a plausible research direction, not evidence that synthetic appearance transfers to real thermal imagery. Self-supervised multi-view learning has established precedents such as [DINO](https://arxiv.org/abs/2104.14294); that paper does not establish satellite-to-thermal transfer for this proposed model.

## Separate claims

- **Same-map adaptation:** training from the deployment map is allowed by the user's constraint. Report it explicitly; it may learn map-specific structure.
- **Cross-map generalisation:** hold out an entire map/site during training and all satellite-based model selection, then evaluate there without adapting weights. Training on both official and NUDT reference maps does not establish unseen-map transfer.
- **Drone generalisation:** no drone-derived fitting. Already-examined official/NUDT frames remain regression data. Independent real acquisitions are required for a strong new-flight claim.
- **Risk calibration:** a wrong-answer bound estimated on synthetic satellite tasks does not automatically apply to real thermal inputs under domain shift. Report real score, coverage and wrong answers separately; do not transfer a synthetic certificate to the drone domain.

## Status

This file records the revised constraint and audited training lineage. No new network has been trained in this step, no old weights are being presented as map-only, and no active pipeline files were changed. The next training implementation must use a separate `_codex` experiment directory and a saved source/split/weight manifest.

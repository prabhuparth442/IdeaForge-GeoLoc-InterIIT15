# Codex reviews of 2026-11-18 and how they were checked

Two independent reviews were made by another AI (Codex) on 2026-11-18 and pasted into the Claude Code session. This file keeps (1) what each review said, copied from the session, (2) what was checked against our own saved data and code, and (3) what was done with it. Full reports: `tests/Review_Highalt_20261118_codex/` and `tests/LoFTR_Inputs_20261118_codex/` (each has `REPORT_codex.md`, an executed notebook and `results/`; both are self-contained and can be deleted alone). The reviews claim no new pipeline gain; every proposal in them was untested when written.

Rule that applies to both: the locked test split (168 photos of the new set) was not opened by either review and must not be used to choose anything (see `REPO_MAP.md` section 5).

---

## Review 1: the high-altitude data set (summary as pasted)

> **The new dataset needs a different verification path above roughly 800 m—not simply a more permissive judge.** I reviewed the saved results and visually inspected 14 development images across the height bands.
>
> The single-query stage diagnostic shows:
>
> | Height | Photos | Correct region retained* | Answers accepted |
> |---|---:|---:|---:|
> | Below 800 m | 190 | 98.9% | 97.9% |
> | 800–1,600 m | 100 | 92.0% | 53.0% |
> | Above 1,600 m | 42 | 64.3% | 4.8% |
>
> \*Within four scoring sigmas; this diagnostic uses K100/M25. Accepted answers had zero errors beyond eight sigmas.
>
> The strongest clue is the separate truth-position test: **above 1,600 m, LoFTR produced zero inliers on 38/42 images even when supplied the true location.** That diagnostic has map/geometry limitations, but it shows why changing the final acceptance threshold cannot solve most failures.
>
> In the actual images, individual dark structures become tiny while broad tracks, streaks and regional texture remain. Strong edge shading and a recurring lower-right obstruction also remain fixed in camera coordinates.
>
> I would prioritise these changes:
>
> 1. **Below approximately 800 m: retain the existing matcher and improve position precision.** Retrieval and acceptance already work well. Spend effort on accurate local refinement and checking whether the matches support the reported camera centre.
> 2. **Around 800–1,600 m: use a refined coarse location followed by dense structural verification.** Match larger spatial patterns using gradient orientation, band-pass ridges or phase congruency. Test direct local translation/rotation/scale alignment rather than requiring LoFTR correspondences in every case. RIFT is a relevant non-trained comparison method, although its sparse features may also struggle here.
> 3. **Check the proposed alignment across separated image regions and against distant challengers.** A candidate should explain the visible spatial arrangement, not just one patch. This needs wrong-map controls; several correlated checks agreeing is not a zero-error guarantee.
> 4. **Above 1,600 m: strengthen regional retrieval and preserve abstention.** Broader landform and texture arrangements become more useful than individual small landmarks. Some images may contain insufficient distinctive information for reliable single-photo localisation.
> 5. **Calculate map levels from camera resolution and footprint.** This camera nominally needs level 12 above about **1,924 m**, and level 13 above **2,720 m**. Raising the cap is necessary for those heights, but cannot explain or fix the earlier collapse.
>
> Two cautions matter:
>
> - Existing two-level-finer refinement improved high-altitude within-one-sigma accuracy from **24.6% to 28.9%**, but its forced-answer experiment still contained **16 wrong answers**. Refinement helps precision; it does not certify the location.
> - A coarse-score threshold that looks safe on this desert set fails on old official and NUDT. Avoid hard-coding a desert-specific acceptance rule.
>
> **Route by ground resolution (`height / focal length`) and usable structural evidence.** The 800 m transition is specific to this camera and terrain. These are proposed next tests, not newly demonstrated pipeline gains.
>
> No model was trained, and no locked-test images were opened.

### How it was checked (our data, 2026-11-18)

Recomputed exactly from our files: the stage table by height (190 / 100 / 42 development photos; retained 98.9 / 92.0 / 64.3 %; answered 97.9 / 53.0 / 4.8 %; 48.60 / 21.90 / 1.29 of 60; 0 wrong); the query variants (one query 34.57, standard six 35.49, wide 27 queries 35.96 of 60); the truth-supplied LoFTR diagnostic (zero inliers at the true place for 2/12, 14/43, 19/45 and 38/42 photos in the bands 800-1000, 1000-1300, 1300-1600 and 1600-2100 m); the 448 px canvas result (photos with at least 8 true-place inliers 52 to 30); the level 12 / 13 heights (1924 m, 2720 m); the 2.4 sigma coarse grid at 1400 m; our own refinement and fallback numbers. All 14 viewed images are in the development split. What the review missed: a map one level finer gave 58 instead of 52 photos with at least 8 true-place inliers above 800 m with about the same retrieval (79.6 % against 80.3 %).

### What was done with it

- Its idea of dense structural channels with a wrong-map (decoy) control was tested as iteration 29 (`tests/Test_Iteration_12/notebooks/Coarse_Channels*.ipynb`): the channels were weaker than the edge search at height; the agreement gate was safe (0 wrong, 0 false decoy agreements on three sets) but answered only 9 % of the photos above 800 m.
- The per-matcher bar from wrong-place counts (user's idea, in the spirit of its wrong-map control) was built into iteration 36 (`Matchers_At_Height.ipynb`).

---

## Review 2: what is lost before LoFTR (summary as pasted)

> Yes—some visible detail is lost before LoFTR, and some can be preserved. I reconstructed and inspected the actual inputs for four development images at roughly 600–1900 m.
>
> What I found:
>
> * Small spots and gaps get smoothed together. At 1402 m, the original samples about 6.9 m/pixel, while the baseline canvas samples 13.6 m/pixel, after preblur.
> * The fixed canvas also excludes peripheral content: approximately 9.6–25.7% of source pixel centres in these four examples. The existing 448 option keeps the same ground footprint, so those edges remain excluded.
> * Some differences are inherent: sharp satellite tracks remain weak or invisible in the original thermal image. Preprocessing cannot recover an absent thermal signature.
>
> Below, compare the actual input with direct rendering from the original and the cleaned pair:
>
> The strongest supported approach is direct rendering from raw at 448, a finer satellite tile covering the same area, and compatible cleaning on both sides. Simply enlarging the existing 224 image cannot recover discarded detail.
> Your existing diagnostic supports this combination: 57/142 → 72/142 frames reached eight true-location inliers. Higher resolution alone fell to 44/142. These are diagnostic matches, not a validated pipeline score gain or guarantee against wrong selections.
> The additional experiment worth testing is a larger fine-stage footprint or overlapping windows, with validity masks, to retain useful peripheral features.
> Everything is saved separately in the report and executed notebook. Production code and settings were unchanged.

### How it was checked (our data and code, 2026-11-18)

- Sampling table: nominal source m/px = height / fx (2.98, 4.95, 6.93, 9.32), baseline canvas 4.80, 9.60, 13.58, 13.58 m/px, preblur sigma in source pixels 0.556, 0.719, 0.730, 0.478 from `geoloc/photo.prepare_photo` (`0.5 x canvas pixel / source pixel - 0.25`): all recomputed.
- Iteration 34 counts: 57, 62, 44, 72 and 64 of 142 photos with at least 8 true-place inliers, 0 wrong-place at 8: exactly our `tests/Test_Iteration_12/results/edge_features/loftr_photo_strength.csv`.
- 448 path keeps the same footprint, and the grey padding reaches LoFTR without a validity mask: true in `geoloc/native_fine.py` and `fine_matching.grey_canvas`.
- Source pixels outside the 224 canvas (22.5, 12.0, 9.6, 25.7 %): recomputed independently with the pipeline's own camera geometry as 21.2, 11.8, 9.3 and 25.3 %. The pipeline's own `draw_canvas` shows the same cropping: a 448 canvas at the same pixel size (4 times the area) holds 16 to 40 % more photographed ground than the 224 canvas. (A first quick check with an interpolated index image gave 0.3 % and was wrong, because interpolating indices creates false hits.)
- Not re-verified: the gradient-RMS retention percentages.
- Its remark that the headless-native and matcher analyses were not yet populated was true when written; they were produced later (iterations 35 to 38, `results/native_fine/`, `results/matchers/`).
- Caution on its proposal (larger fine-stage footprint): Fable's earlier test at the same pixel size with a larger area lowered the photos with at least 8 true-place inliers from 52 to 30, so more coverage alone is not promising; it is untested with native resolution and a validity mask.

### What was done with it

The native-resolution path it recommends (canvas two levels finer, photo and tile cleaned) is the switch `NATIVE_FINE_FROM_HEIGHT_M` in `Main_Pipeline_Headless` (iteration 35: 332 working photos 35.64 to 37.79 of 60, 0 wrong). A separate swap of LoFTR for XoFTR from 800 m (iterations 36 to 38) gave larger gains: 35.64 to 39.76 on the working photos and, run once on the locked test split, 38.00 to 40.61 with 0 wrong. See `docs/HANDOFF_2026-11-17.md` addenda 11 to 14 and `CHANGELOG.md`.

---

Notion mirror: results page "Second-list ideas checked in the submission pipeline" (id `41a0a439-e483-81ff-92c5-cb8ad7dd2ffd`), entries "Codex review of the high-altitude set (2026-11-18), checked" and "Codex review of the LoFTR inputs (2026-11-18), checked".

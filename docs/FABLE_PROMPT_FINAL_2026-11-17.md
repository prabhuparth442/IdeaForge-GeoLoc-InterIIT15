# Final prompt for Fable (2026-11-17): the whole state of the work, and what we need decided

This is our last request for ideas before we freeze. Please think hard and answer in full. Everything below was measured; "in-sample" is marked where it applies. Notion has the numbers in more detail: the page "Second-list ideas checked in the submission pipeline: results and verdicts (2026-11-16)" (iterations 1-16), your two high-altitude pages of 2026-11-17, and `tests/Test_Iteration_7/OURS_CHECK.md`. Code: `Main_Pipeline_Headless` (submission) and the research repo (`tests/Test_Iteration_12/`). Nothing is committed.

## 1. The task and the pipeline

Thermal UAV photo (320 x 256, camera metadata: height `alt` relative to take-off, yaw, pitch, roll, a GPS hint that may be 5 km off) -> position on a satellite map (0.28-0.29 m per pixel). The organisers' hidden test set has heights of 100 to 3000 m. Score per photo: error within 1 sigma +10, 2 sigma +6, 4 sigma +3, 8 sigma 0, beyond 8 sigma -20, no answer 0 (sigma depends on height and focal length). A photo that takes longer than 30 s counts as no answer; our guard is 22 s. The Jetson Orin is about 3 times slower than our laptop (13 s per photo against 4-5 s).

Pipeline: the photo is drawn north-up on a 224 px canvas at the map level whose pixel (0.3 * 2^(L/2) m: 0.85 m at level 3, 3.4 m at 7, 6.8 m at 9, 13.6 m at 11) fits the photo footprint -> edge-orientation ZNCC by FFT over a 5.5 km disc, K = 50 peaks -> DINOv2 ViT-S/14 (layer 8) re-orders them -> LoFTR (MINIMA weights) on M = 5 places per query + RANSAC (3 px, scale +-15 %, rotation 12 deg) -> answer if the best place has at least 8 inliers. 6 standard queries (headings -5/0/+5 x scales 1.0/1.05); if unanswered: an agreement rule (two queries within 1 sigma, each with at least 3 inliers), then 12 more queries. Optional switches we added (all off by default): altered copies (polarity flipped and sharpened copies of skipped photos, answered by the bar only), an aligned-match rescue (dropped, see below), `BAR_BY_LEVEL`, and photo cleaning (`FLAT_FIELD_FROM_HEIGHT_M`, `FLAT_FIELD_DEGREE`, `LOCAL_NORM_SIGMA`, `PHOTO_MASK_RECTS`). Only pretrained weights, no training on our data so far.

## 2. The three data sets (and how different they are)

| | old official dev set | NUDT Thermal-UAV | new high-altitude set |
|---|---|---|---|
| photos | 500 | valid split of 1425 frames (we used 60-300) | 500 |
| place | Mumbai area, urban, fields, roads, a river | Changsha | Rajasthan desert: shrubs and trees, dune streaks, no man-made structure |
| sensor | 320 x 256, fx 200.3 | 1280 x 1024, fx 1562 | 320 x 256, fx 202.4 |
| height | 96-347 m (median 148) | 197-318 m | 100-2017 m (median 709; 168 photos above 1000 m) |
| ground size of a photo pixel | 0.5-1.75 m | about 0.2 m | 0.5-10 m |

Similarity measured (notebook `Dataset_Similarity.ipynb`): the pipeline's own DINOv2 separates old from new with AUC 1.00, also for the new photos at 100-350 m (two random halves of the old set give 0.57); no new photo has an old photo among its 5 nearest neighbours. The new photos have a lens vignette 2.6 times stronger (explained variance 0.30 against 0.11, 0.53 above 1300 m), fewer edges (share 0.34 at 100-350 m against 0.52, then 0.26, 0.14, 0.13 with height) and a dark hook artefact in the bottom-right corner (x 303-312, y 225-255) plus a dark bottom-left corner. The dark spots are trees (cooler than sunlit ground; polarity agrees with the map) and they shrink from 10 pixels at 100 m to 1-2 pixels above 800 m. So the new set is a different domain; anything tuned on the old set must be checked on the other two.

Splits: the new set is split into 332 working photos and 168 locked test photos (spatial cells of 0.004 degrees, balanced by height band; the photos were flown in a climb, so a north/south split would confound height). You evaluated frozen thresholds once on the locked test split; treat it as used once for the coarse rule. Every earlier high-altitude number of ours used all 500 photos (in-sample).

## 3. Results so far

Old official set (500 photos, headless pipeline): base 52.69 /60 (454 answered, 1 wrong); + altered copies 53.60 (+0.9, 8 extra photos, all right). Wider search (K 100/200, M 10/20) and a bigger second pass: no gain. Mega evaluation (80 photos, 143 queries each, the heading x scale map peaks at heading 0 and scale 1.00-1.05): the 6 standard queries equal 143 queries; blur (2 px) costs 23 points (49.8 -> 26.4 with the real decision), polarity inversion 11; the true place is still found but too few inliers reach the bar (matcher failure, not search). Agreement rule: +2.3 clean, +4.7 blur, +4.2 invert on the old set.

NUDT: altered copies on 300 photos: clean +0.44, inverted +1.96, blurred +0.20, with 2 wrong copy answers in 600 altered photos. Lowering the bar below 8 loses points on NUDT (clean 52.7 at bar 8 against 48.8 at bar 4 with 14 wrong; wrong places there have 4-7 inliers), although it looks like +6 on the old set. Agreement answers on NUDT were wrong 11 % of the time (3 of 27) against 0.8 % for bar answers (net still positive). An aligned-match rescue (idea 1 of the second list) gave +2.9 on the old set and 25 of 25 right, but 1 wrong in 3 rescues on fresh NUDT photos and no safe setting across five sets: dropped. Parts maps, part-peak agreement, large-window LoFTR, scale/heading refinement: dead.

New high-altitude set, headless pipeline, map levels 3-11 (score /60 over 500 photos, 0 wrong answers everywhere): base 36.35; + copies 37.55; bar 4 at map levels 9-11 38.54; both 39.48 (in-sample; a re-run of bar 4 gave one wrong answer). By height band on the 332 working photos, mean points of 10 (answered share):

| height (m) | photos | base | + copies | bar 4 at levels 9-11 | both | median inliers |
|---|---|---|---|---|---|---|
| 100-300 | 74 | 8.05 (98 %) | 8.14 | 8.05 | 8.05 | 57 |
| 300-500 | 48 | 8.08 (100 %) | 8.08 | 8.08 | 8.08 | 96 |
| 500-800 | 68 | 8.35 (98 %) | 8.50 | 8.50 | 8.50 | 66 |
| 800-1000 | 12 | 4.50 (66 %) | 5.00 | 5.25 | 6.00 | 43 |
| 1000-1300 | 43 | 4.12 (58 %) | 4.77 | 5.56 | 5.65 | 18 |
| 1300-1600 | 45 | 4.00 (51 %) | 4.44 | 4.58 | 5.16 | 8 |
| 1600-2100 | 42 | 0.21 (7 %) | 0.21 | 0.50 | 0.50 | 0 |

Working set 35.64 /60 base; points lost (of 60): 100-300 m 2.6, 300-500 m 1.7, 500-800 m 2.0, 800-1000 m 1.2, 1000-1300 m 4.6, 1300-1600 m 4.9, 1600-2100 m 7.4. A wider second pass above 1000 m (scales 0.7-1.3, headings +-10) is worse (3.83 -> 3.50 per photo, 2 wrong): the failures are not a scale or heading error.

Photo cleaning (your change 1, implemented, headless, 332 working photos, bar 8): vignette removed from 800 m 35.64 -> 36.49; + local normalisation 35.57 (one wrong); + hook mask 36.56; + both corner masks and local norm 36.09. Your falsification test fails for LoFTR (answered share above 1400 m 19 % -> 22 %, median inliers 9 -> 10, and 12 only in the all-masks variant): vignette removal helps the coarse stage, not LoFTR, as your second page said.

Your second page (one query, one tile, true place known): the true place is the best coarse cell for 0.96-1.00 up to 800 m, 0.84-0.96 up to 1600 m and 0.60 above; LoFTR reaches 8 inliers there for 0.78 at 800-1000 m, 0.58 at 1200-1400 m, 0.38 at 1400-1600 m and 0.03 above 1600 m. Your coarse-answer rule (z of the best coarse peak over its disc, threshold 6.0, refine within +-6 canvas pixels) on the 72 test photos above 800 m: LoFTR 38 answered, 3.33 points per photo; coarse rule on the cleaned photo 61 answered, 0 wrong, 5.19; LoFTR then coarse rule 63 answered, 0 wrong, 5.44. NOT implemented yet in the pipeline; not tested with six queries on the three-tile store; not tested on a second region; not safe below 800 m (3 wrong in 83). Your other ideas: separate coarse and fine photo preparation; a camera pointing offset (+3 of 60 below 1200 m, camera specific, hurts above); skip LoFTR above 1600 m; one finer level at 1200-1600 m.

Running now on the GPU server (RTX 4000 Ada, 20 GB, shared with another job): the real `Stage_Evaluation.ipynb` (six stage tables, recall at K, DINO layer, fine matcher, decision, funnel with ceilings, heading queries, time, sweeps, failure samples) on the 332 working photos of the new set, then 150 old-set photos and 150 NUDT valid photos. Results in a few hours.

## 4. What the organisers said about the midterm submission (still open)

Camera constants (yaw offset 180 deg, scale 1.04, roll bias 0.98) were fitted on the dev set and the NUDT scale came out at 0.80: derive the warp from the published convention and plot signed error against pitch, roll and altitude; most no_match frames came from the 22 s clock, not the verifier (re-measure on the Orin NX at 15 W, order second-pass queries by the first pass's best heading); the agreement rule (49.0 -> 53.5) and its 11-inlier bar were tuned on the same photos; the pyramid stopped below the final altitude range (we now have levels 3-11, not yet on the Jetson); DECLARATIONS.md and the freeze file are incomplete. The Jetson has not been measured with any of the new switches.

## 5. What we need from you (please be concrete)

1. A final recommended configuration, by height band (below 500 m, 500-800 m, 800-1300 m, 1300-2000 m, 2000-3000 m), using our existing switches and names or new ones you specify: which of copies, photo cleaning, bar by level, agreement, second pass, aligned rescue to switch on, with expected gain and risk of a wrong answer (-20) for each, and which to leave off. We can run any test on the GPU server tonight.
2. The coarse-answer rule as an exact specification we can implement: where it sits in the flow, what z threshold and margin, how to make the threshold transfer to another region and to heights up to 3000 m (a threshold fitted on one desert region is not safe), how to guard against the true place missing from the map or the hint being near the map edge, the refinement step, and the exact test that would kill it. Please also say what to do above 1600 m, where the coarse stage is right for only 0.60 of the photos and above 2000 m where we have almost no data.
3. The hidden test set reaches 3000 m and we have no real photo above 2017 m. How should we prepare for 2000-3000 m (levels, canvas, queries, decision, time) and how do we test it? We have a map-based simulation option (down-sampling photos and map together), and a block-average of old and NUDT photos by 2-3x with the measured vignette; tell us which is worth building and what it can and cannot show.
4. Blur and low detail: the matcher gives too few inliers on blurred photos (23 points lost on the old set) and on the high-altitude photos. Is there a change for the fine stage (a different matcher or input scale, matching on the edge maps, a sharpening or denoising step for all photos, a score that is not an inlier count) that you expect to work in both cases? Name the cheapest test first.
5. A decision rule that transfers across the three domains (the bar and the agreement rule are tuned on the old set and misbehave on NUDT). Give the form of the rule, the quantities it should be normalised by, and how we fit and test it so that wrong answers stay near zero on a domain we have not seen.
6. A time plan for the Jetson: with the high levels, the 143-query-free standard pipeline, copies and a coarse path, what is the expected seconds per photo by height band, and what should the guard and the query order be.
7. A prioritised list of the experiments to run in the remaining hours, each with the exact falsification threshold, the data it needs (old, NUDT, new working set, new locked test set) and its cost. Mark the one thing you would do if we could only do one.
8. What we should NOT do, and what you are least sure about. List the questions you would want answered by the organisers or by us before you would trust the answer.

Please give numbers (expected gains with a range, not only a direction) and say which of them are measured and which are guesses.

# High-altitude review, 18 November 2026

Open REPORT_codex.md for findings and proposals. Open Highalt_Review_executed_codex.ipynb for executed calculations, raw-image examples and saved outputs. Highalt_Review_codex.ipynb is the rerunnable source notebook. review_blocks.py contains read-only helpers; no pipeline changes or trained weights are introduced.

Inputs are the current Iteration12 CSVs and metadata. Image viewing is restricted to the332 development photos. The notebook validates row alignment, displays14 altitude-stratified development frames, recomputes stage/refinement scores and saves hashes. The168 locked photos are not opened or evaluated.

The saved notebook was executed with a sequential IPython shell and captured real rich outputs; the notebook metadata records that method. A kernel-server attempt did not start in this environment and was stopped. Analyses are lightweight CPU work, with no GPU jobs, map extraction or neural training/inference. Notion updating is unavailable in this session.

| File | Result |
|---|---|
| results/stage_by_height_codex.csv | single-query K100/M25 stage diagnostics by height |
| results/query_variants_codex.csv | saved query-policy summaries, with wrong counts |
| results/drone_images_by_height_codex.png | viewed development images, per-image1–99% stretch |
| results/resolution_by_height_codex.csv | physical GSD, level and grid-spacing calculation |
| results/coarse_refinement_rescore_codex.csv | forced-answer coarse refinement results across available variants |
| results/coarse_confidence_transfer_codex.csv | wrong-location confidence on all three domains |
| results/truth_known_matcher_by_height_codex.csv | matching when the true location is supplied |
| results/inputs_codex.json | source CSV/image hashes |

Remove only this folder to remove the review artifacts. The required local index/handoff/changelog additions identify this review explicitly and can be reverted separately.

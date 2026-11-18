# tests/Test_Iteration_12/scripts

| path | what |
|---|---|
| `server_chains/` | the shell chains that ran on the GPU server, assuming the layout in `docs/HANDOFF_2026-11-17.md` (`/mnt/e/IdeaForge/...`). Download and copy of the new set (`dl_highalt.sh`, `copy_highalt.sh`), store build and runs (`highalt_chain*.sh`, `highalt_bar.sh`), Stage_Evaluation (`stageeval_chain.sh`), DINO and refine runs (`dino_chain.sh`, `refine_chain.sh`), `resume2.sh`, and the coarse-stage experiments of 2026-11-17: `coarse_conf_chain.sh` (iteration 25), `coarse_evidence_chain.sh` (26), `coarse_refine_chain.sh` (28), `coarse_channels_chain.sh` (29), `precision_chain.sh` (40: XoFTR from 800 m with refit variants), `synth_chain.sh` (40: the pipeline on the synthetic photos) |
| `analysis/` | small analysis scripts: `band_cmp.py` (per height band), `avg_test.py`, `spots_look.py` (dark spots against the map), `ff_look.py` (flat-field look) |
| `highalt_truth_diag/` | Fable's map and store builders and runners (own README) |
| `nudt_headless_dataset.py` | converts a random sample of the NUDT valid split to the headless format (320x256, fx 390.64): `python nudt_headless_dataset.py NUDT_DIR OUT_DIR N` |
| `tick.sh`, `overnight_supervisor.sh`, `overnight_wrapper.sh` | the 15-minute status signal and the overnight supervisor of 2026-11-15 (no longer running) |

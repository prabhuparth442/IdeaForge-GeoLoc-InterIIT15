# High-altitude truth-place diagnostic (2026-11-17)

One query (heading 0, scale 1.0) per photo through the submission pipeline's own stages (Main_Pipeline_Headless/geoloc), with the true place known,
so that the coarse search, the DINO re-order and LoFTR can be judged separately. Results: ../../results/highalt_truth_diag/, notebook: ../../notebooks/HighAlt_Truth_Diagnostic.ipynb.

WORK is a scratch folder (about 2 GB): WORK/new/images = the unpacked Dataset_Official_3/d1.zip.

1. `unzip -p .../sat_imagery/2684532111.zip 2684532111/BAND.tif | python build_map.py WORK/map_1p12.npy`  (streams the 11 GB tile, 4x block mean, 1.12 m per pixel, about 3 minutes)
2. `python build_store.py WORK`, then set "crs": "EPSG:32642" in WORK/store/store.json (the tile is UTM 42N)
3. `python run_base.py WORK base`  -> diag_base.csv (every photo above 170 m)
4. `python run_var.py WORK <tag>`  -> var_<tag>.csv (dev photos above 800 m; HX_SPLIT=test for the test split, HX_REFINE=1 adds the pixel-step position refinement)
   tags: base flat flatnorm noclahe lm1 lp1 t336 t336flat t448 l12 l12flatnorm lm1flatnorm
5. `python clip.py WORK`  (geometry only)

Run each through tools/run_capped.sh. Limits: one satellite tile only (the disc is cut for photos near the tile's south edge), one query per photo, map levels 5-12 from a 1.12 m copy.

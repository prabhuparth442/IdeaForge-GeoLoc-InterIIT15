# Test_Iteration_5: drone-view simulator

Map in, simulated thermal drone photo out (camera perspective), with validity mask and metadata.

- Blocks: `train/simulation/` (camera_model, pose, ground_rays, map_sampler, appearance, sensor, render, dataset, compare). Swap points: `appearance` (any function `(grey, rng, strength, pixel_m) -> picture`), `sensor`, and the `terrain` function for non-flat ground.
- Unit tests (synthetic GeoTIFF, no dataset): `python tests/train/test_simulation.py`
- Notebook (executed, small 320 x 256 images, 8 NUDT and 6 official poses): `notebooks/Drone_View_Simulator.ipynb`; saved examples with json metadata in `results/examples/`.
- Main finding: geometry is verified; appearance is the open part. See section 9 of the notebook.
- Realism and downstream metrics (canvas space, with a real-vs-real floor): `notebooks/Realism_and_Downstream.ipynb`; blocks `compare.py`, `realism.py`, `downstream.py`. Finding: every fake is easier than real downstream; the blobby low-`edge_keep` looks are closest in difficulty but far in realism (too little grain and edge detail).

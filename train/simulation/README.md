# train/simulation

Drone-view simulator: a satellite map goes in, a simulated thermal drone photo (camera perspective) comes out. Pose, camera model, ground rays, appearance, sensor chain, realism checks.

| file | what |
|---|---|
| `appearance.py` | What the ground looks like in thermal. One contract: appearance(grey, rng, strength, pixel_m) -> float picture of the same shape. |
| `camera_model.py` | Camera numbers for the simulator. Same attribute names as header.data.Camera, so either can be used. |
| `compare.py` | Numbers for comparing a real photo with a simulated one (or either with the map). No plotting here. |
| `dataset.py` | Random poses and saving examples with their metadata. |
| `downstream.py` | Downstream check: how well does the existing coarse search find the truth from this canvas? Same function for real and simulated queries. |
| `ground_rays.py` | For every camera pixel, the ground point it sees. This is the map-to-camera direction of the geometry. |
| `map_sampler.py` | Read the needed piece of the satellite map and sample it into the camera image. |
| `pose.py` | Where the drone is and where the camera points. |
| `realism.py` | DINOv2 features of canvases, for comparing sets of real and simulated photos (see compare.frechet_distance, classifier_auc). |
| `render.py` | render_view: pose + camera + map -> simulated thermal drone image with a validity mask and full metadata. |
| `sensor.py` | Camera chain after the optics: blur, noise, stripes, auto gain, optional flattening, vignette, local enhancement (clahe) and grain, 8-bit quantisation. Output pixel values are NOT temperatures. |

Related: tests/Test_Iteration_5/README.md. Repo map: `REPO_MAP.md` at the repo root.

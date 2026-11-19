# Parameter distributions of the metadata (2026-11-19)
Working = 332 development photos of the new Rajasthan set; old official = 500 photos. Full table with min, 5/25/75/95 %, max and number of distinct values: parameter_distributions.csv. Angles in degrees, wrapped to -180..180 (the files store pitch and roll as 0..360). tilt = angle between the optical axis and straight down. Derived: ground pixel = height / fx, footprint = height x 320 / fx, sigma = sqrt(8^2 + (3 x ground pixel)^2).


## working (332)

| parameter | mean | std | min | p5 | median | p95 | max |
|---|---|---|---|---|---|---|---|
| alt | 852.9 | 588.5 | 100.6 | 116.2 | 695.4 | 2011 | 2017 |
| yaw_wrapped | -10.33 | 117.7 | -179.7 | -172.1 | -13.29 | 172.5 | 179.9 |
| pitch_w | 2.154 | 3.017 | -8.251 | -4.896 | 3.094 | 4.896 | 9.11 |
| roll_w | -1.617 | 8.655 | -20.57 | -12.21 | -3.667 | 15.49 | 21.83 |
| tilt_deg | 8.305 | 4.689 | 0.2431 | 3.005 | 7.263 | 19.17 | 21.92 |
| cam_fx | 202.4 | 2.847e-14 | 202.4 | 202.4 | 202.4 | 202.4 | 202.4 |
| cam_cx | 163.5 | 0 | 163.5 | 163.5 | 163.5 | 163.5 | 163.5 |
| cam_cy | 121.7 | 0 | 121.7 | 121.7 | 121.7 | 121.7 | 121.7 |
| cam_k1 | -0.2776 | 0 | -0.2776 | -0.2776 | -0.2776 | -0.2776 | -0.2776 |
| cam_k2 | 0.06491 | 1.39e-17 | 0.06491 | 0.06491 | 0.06491 | 0.06491 | 0.06491 |
| fov_h_deg | 76.65 | 1.423e-14 | 76.65 | 76.65 | 76.65 | 76.65 | 76.65 |
| ground_pixel_m | 4.214 | 2.907 | 0.497 | 0.5742 | 3.436 | 9.937 | 9.963 |
| footprint_w_m | 1348 | 930.4 | 159.1 | 183.8 | 1099 | 3180 | 3188 |
| sigma_m | 15.75 | 7.193 | 8.138 | 8.183 | 13.05 | 30.87 | 30.94 |
| tilt_shift_per_deg_m | 14.89 | 10.27 | 1.756 | 2.029 | 12.14 | 35.11 | 35.2 |

## old official (500)

| parameter | mean | std | min | p5 | median | p95 | max |
|---|---|---|---|---|---|---|---|
| alt | 187.9 | 89.19 | 100 | 100.3 | 148.5 | 350.2 | 350.7 |
| yaw_wrapped | 8.4 | 107.3 | -179.9 | -146.7 | -6.073 | 173.6 | 179.8 |
| pitch_w | -6.598 | 6.036 | -16.27 | -14.78 | -6.933 | 3.438 | 9.225 |
| roll_w | 0.4678 | 3.299 | -10.03 | -4.99 | 0.3438 | 6.666 | 15.41 |
| tilt_deg | 8.348 | 4.608 | 0.0573 | 1.145 | 8.058 | 15.49 | 18.35 |
| cam_fx | 200.3 | 5.69e-14 | 200.3 | 200.3 | 200.3 | 200.3 | 200.3 |
| cam_cx | 158.9 | 0 | 158.9 | 158.9 | 158.9 | 158.9 | 158.9 |
| cam_cy | 131.7 | 2.845e-14 | 131.7 | 131.7 | 131.7 | 131.7 | 131.7 |
| cam_k1 | -0.257 | 5.557e-17 | -0.257 | -0.257 | -0.257 | -0.257 | -0.257 |
| cam_k2 | 0.04832 | 6.946e-18 | 0.04832 | 0.04832 | 0.04832 | 0.04832 | 0.04832 |
| fov_h_deg | 77.24 | 1.423e-14 | 77.24 | 77.24 | 77.24 | 77.24 | 77.24 |
| ground_pixel_m | 0.9383 | 0.4453 | 0.4993 | 0.5008 | 0.7415 | 1.749 | 1.751 |
| footprint_w_m | 300.2 | 142.5 | 159.8 | 160.2 | 237.3 | 559.5 | 560.4 |
| sigma_m | 8.57 | 0.5037 | 8.139 | 8.14 | 8.303 | 9.566 | 9.571 |
| tilt_shift_per_deg_m | 3.28 | 1.557 | 1.746 | 1.751 | 2.592 | 6.113 | 6.122 |

## Notes
- Height bands, working: <150 m 31, 150-250 20, 250-350 35, 350-500 36, 500-1000 80, 1000-1300 43, 1300-1600 45, 1600-2100 42, above 2100 0. Old official: <150 m 288, 150-250 59, 250-350 115, 350-500 38, nothing above 500 m.
- One camera per set: fx 202.4 (working) and 200.3 (old official), k1 -0.278 and -0.257, principal point (163.5, 121.7) and (158.9, 131.7); both 320 x 256, about 77 degrees wide. Intrinsics do not vary inside a set.
- Tilt: median 7.3 degrees (up to 21.9) on working, 8.1 (up to 18.4) on old official. Pitch is centred near +3 degrees on working and -7 degrees on old official, so the sign convention or the mounting differs between the sets.
- Ground pixel 0.5 to 10 m (working) against 0.5 to 1.75 m (old official). The evaluation goes to 3000 m: 14.8 m per pixel, a 4.7 km wide footprint, sigma 45 m.
- Location: working photos lie in a 2.8 x 13.6 km strip near 27.08 N, 71.48 E; old official in about 1.5 x 1.5 km near 19.13 N, 73.09 E.
- The takeoff-point elevation (alt_takeoff_pt_dem) is one constant per set (191 and 4).

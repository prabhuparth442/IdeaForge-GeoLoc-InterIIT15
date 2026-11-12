# From a satellite map to a simulated thermal drone view

## The idea

We want to start with a geographically referenced satellite map, choose a drone camera position and height, and produce an image that resembles what a thermal camera **might** see from that position. Because we choose the position, we also know the ground-truth location of the simulated image.

This could help us create training data for a thermal-to-satellite localisation model, test image projection at heights for which we have no drone photos, and explore how camera angle, resolution and image quality affect matching.

**The result is a simulated or pseudo-thermal image, not a measurement of the real temperature or a reconstruction of the exact infrared image a drone would have captured.** A visible satellite map does not contain enough information to determine the thermal state of a scene at a particular time.

```text
Georeferenced satellite map + band information
                   |
Choose location, height, camera and attitude
                   |
Work out the ground area visible to the camera
                   |
Create plausible thermal appearance(s) for the scene
                   |
Project the ground into the camera's perspective
                   |
Simulate sensor resolution, blur and noise
                   |
Simulated thermal drone image + location and metadata
```

The order can vary by implementation. The important point is to account for geometry, scene appearance and camera sampling consistently; simply recolouring a satellite crop does not simulate a drone camera.

## 1. What is in the input map?

A map image can have one band or many. A **band** is a measurement over a particular wavelength range. We must inspect the map's band descriptions and metadata before choosing a conversion method.

| Input | What it gives us | What it does not automatically give us |
|---|---|---|
| Panchromatic / single visible band | A detailed grayscale view of reflected visible light. | Colour, material identity or surface temperature. |
| RGB | Reflected visible red, green and blue. | A direct thermal measurement. |
| Multispectral or other N-band imagery | Measurements at the wavelengths specified by the sensor; some bands may help distinguish vegetation, water or surface types. | Thermal information unless suitable thermal bands are actually present; even then check their resolution, calibration, time and wavelength. |
| Thermal-infrared band | An observation related to emitted and reflected thermal radiation at the satellite's acquisition time. | The exact thermal image at a different time, viewpoint, altitude or sensor. |

**Infrared is a broad wavelength range.** Near-infrared bands can largely measure reflected sunlight and are not equivalent to the thermal infrared measured by a typical thermal camera. Check the drone sensor's spectral band—such as LWIR or MWIR—and the satellite band's actual wavelength and calibration before claiming they correspond.

In our project, the newer official map is described as Cartosat panchromatic imagery in the [stage evaluation analysis](../tests/Parameter_Test/Analysis_2026-11-06_Stage_Evaluation_and_Strategies.md). A model trained on RGB photos may require three input channels; copying the grayscale band three times satisfies its input shape but does **not** recover missing colour information.

## 2. Define where the drone is and how its camera looks

For each simulated image we select or record:

- **Location:** camera latitude/longitude or position in the map's coordinate system.
- **Height above the photographed ground:** not merely height above sea level. If elevation varies, height may differ across the scene.
- **Orientation:** yaw, pitch and roll, together with the camera's mounting orientation relative to the aircraft.
- **Camera intrinsics:** image width and height in pixels, focal lengths or field of view, principal point and lens distortion, if known.
- **Thermal sensor properties:** spectral band, approximate blur and noise, and how the camera outputs its images.
- **Conditions:** plausible time of day, weather and scene variation, if we are modelling them.

These are *simulation inputs*. The desired inference pipeline can later be tested with deliberately imperfect metadata, because real height and attitude estimates may be uncertain.

### Example: what changes at 2.5 km?

For a near-downward pinhole camera, the approximate ground distance per image pixel is `height / focal_length_in_pixels`. Suppose a camera is 2,500 m above flat ground, its horizontal focal length is 320 pixels, and its image is 320 pixels wide:

```text
Approximate ground size per pixel = 2,500 / 320 = 7.8 m/pixel
Approximate ground width          = 320 x 7.8 = 2,500 m
```

This is an **illustration, not our camera's verified calibration**. It assumes a near-nadir view and a flat ground plane; camera tilt, lens distortion and terrain change the footprint. A different focal length or image width can produce a very different footprint.

At high altitude, small roads or buildings may cover very few pixels. A realistic simulation must remove detail the real camera could not resolve; resizing a sharp satellite crop alone is usually insufficient.

## 3. Work out the ground footprint

For a flat-ground approximation, project each camera pixel through the lens into a 3D ray and intersect that ray with the ground plane. This tells us which ground point the pixel sees.

```text
                      Camera
                      /  |  \
                     /   |   \   camera rays
                    /    |    \
          ---------/-----+-----\---------- ground
                 visible ground footprint
```

For a tilted camera, the footprint is generally not a north-up rectangle. A flat plane can be related to an undistorted camera image by a projective transformation; lens distortion is handled separately. See [OpenCV's homography overview](https://docs.opencv.org/4.10.0/d9/dab/tutorial_homography.html).

For hills or steep terrain, a digital elevation model can be used to intersect rays with terrain instead of one plane. A 2D map plus terrain elevations still does not fully represent buildings, trees, occlusions or their thermal properties. More detailed 3D data may help when those effects matter.

The input map must cover the full projected footprint. Pixels that look outside the map or outside the modelled ground should be marked **invalid**, rather than quietly treated as real image content.

## 4. Create a plausible thermal appearance

The satellite map mainly tells us about **where structures are** and how they appeared in the satellite sensor's bands. We need an additional model to approximate **how those structures could appear thermally**.

### Approach A: rule-based pseudo-thermal appearance

Create several variants of a map crop by changing region intensities, local contrast and polarity, and adding plausible variations such as sensor noise or changed areas. If material labels are available, randomise likely appearances by broad surface type—for example, water, vegetation, roofs and roads.

This approach is inexpensive and gives direct control over variation. Its limitations are equally important: arbitrary brightness changes are not a physical temperature simulation, and the generated images can preserve map patterns too perfectly.

The project's existing [pseudo-thermal generator](../train/datasets/pseudo_thermal.py) uses randomized region appearance, noise, small changed patches and rotation/scale variation. It produces a **north-up 224 x 224 query canvas with a mask**, not a validated raw drone-camera view at arbitrary height.

### Approach B: learned visible-to-thermal translation

A trained image-to-image model can take visible imagery and generate a thermal-like image. [IRFormer](https://github.com/CXH-Research/IRFormer) is an example presented by its authors as a lightweight visible-to-infrared translation model. [ThermalGen](https://github.com/arplaboratory/ThermalGen) is another research project whose repository lists pretrained models; its suitability and runtime must be measured for our use.

These are **candidates to investigate**, not models verified for our project. Check checkpoint availability, licence, input bands, training imagery, hardware requirements and whether the model works on overhead satellite scenes. A model trained on RGB street imagery may not transfer to a panchromatic satellite map.

Generated imagery can look realistic while shifting, removing or inventing structures. For localisation, preserving the position of roads, edges and buildings matters more than visual plausibility alone.

### Approach C: physics-informed simulation

With surface materials, weather, sun position, scene geometry and a thermal sensor model, a more detailed simulator can approximate heating, radiation and sensing. This can represent some physical effects more explicitly, but it requires substantial inputs that a map alone normally does not provide. Its accuracy still depends on assumptions and validation against real images.

For a practical first experiment, use several rule-based or learned variants and treat each as a *possible appearance*, not as ground-truth heat.

## 5. Render the simulated drone image

For a chosen location and pose:

1. Determine the required satellite map area, including a margin for camera tilt and sampling.
2. For each camera-image pixel, determine its ground intersection.
3. Sample the simulated ground appearance at that position into the camera image.
4. Apply the camera's distortion model if producing a raw distorted view.
5. Track a validity mask for pixels that cannot be generated reliably.

This is the **map-to-camera** direction: the drone image remains in camera perspective. The present localisation pipeline normally goes the other way, projecting a real drone image into a north-up canvas. The simulated raw image can be passed through that existing projection to test the full pipeline.

When resampling, account for the area covered by each output pixel. Apply appropriate blur or area averaging before downsampling to reduce aliasing. Tilt can make the ground area per pixel vary across an image, so a single global blur is only an approximation.

## 6. Simulate the sensor and image-processing chain

After geometry and appearance, approximate the effects of the actual sensor and saved image format:

- Limited pixel resolution and optical blur.
- Sensor noise, possible fixed-pattern noise and quantisation.
- Contrast and gain differences, including polarity when appropriate for the camera output.
- Image compression or resizing, if the real data undergoes them.
- Atmospheric effects only when they are relevant and can be reasonably modelled.

Use the same preprocessing as the real pipeline when comparing results. For example, our current thermal preprocessing applies percentile normalization and CLAHE. Avoid applying enhancement twice by accident.

Do not present the output pixel values as temperatures unless the whole chain is radiometrically modelled and calibrated for that claim.

## 7. What should each generated example contain?

Keep the generated image together with metadata so it can be used and audited:

```text
simulated image and validity mask
camera location and coordinate reference system
height above the modelled ground
yaw, pitch, roll and camera mounting convention
camera resolution, intrinsics, distortion and spectral band
map source, acquisition date, bands and resolution
terrain source / flat-ground assumption
appearance model, conditions, random seed and parameters
simulation version and any preprocessing applied
```

The camera's ground projection should be labelled separately from the image's visual centre: on a tilted image, these need not refer to the same ground point. Validate the coordinate conventions and axis directions with known points before generating a large dataset.

## 8. Where this fits our localisation project

### A. Test geometry

Generate images at known positions, heights and attitudes, then check whether map search and coordinate recovery return the known ground point. This can catch projection, scale, coordinate-system and sign errors.

**Limit:** a simulator and the tested pipeline can share the same mistaken assumption. Check the simulator with independent test cases and real image–map correspondences as well.

### B. Train a map-retrieval encoder

Use a generated query and its map location as a positive training pair. Other locations provide negative examples. Train across heights, camera angles and appearance variants, then retrieve candidate map regions from a real thermal query.

```text
Simulated thermal view ── query encoder ──┐
                                         ├─ train matching representations
Matching map crop ────── map encoder ────┘
```

The retrieved regions are approximate. Fine matching and geometric verification are still needed to estimate a more precise location and decide whether to answer.

### C. Test an IR-like reference map

Generate a thermal-like version of selected candidate satellite crops and compare it against the real thermal photo. Evaluate it alongside the original satellite crop; keep the original available because image translation may distort landmarks.

Generating map representations ahead of time may reduce online work for a fixed map. But scale, viewpoint, crop boundaries and consistency across neighbouring crops need to be tested.

## 9. What we already learned

The [Iteration 1 report](../tests/Test_Iteration_1/REPORT_seven_methods_and_synthetic_training.md) describes a synthetic-data experiment that trained a fusion model and evaluated it on real photos. It found that the fake queries were much easier to match than real thermal queries, and the measured improvement from mixing fake and real data was not conclusive.

That does **not** prove synthetic data cannot help. It means a new generator should be judged by performance on held-out **real** thermal imagery, not by the appearance of its outputs or its score on images generated from the same map.

## 10. Suggested first experiment

1. Select a small collection of real thermal photos with known locations, camera intrinsics, height and attitude, plus the corresponding satellite map.
2. Generate simulated views at the same or nearby conditions using the existing rule-based method and, if practical, a learned translator.
3. Check projection and ground-truth coordinates using independently verified map features. Inspect visual examples for missing, moved or invented structures.
4. Compare real and simulated image properties and matching difficulty—not just how thermal the images look.
5. Compare downstream localisation using real queries: existing baseline versus any model trained or augmented with synthetic images. Measure candidate recall, position error, score, severe wrong answers, time and memory.
6. Hold out real flights and geographic regions for evaluation. A fixed reference map may be indexed for the intended task, but test queries and threshold tuning must remain separate.
7. For the 2.5 km use case, test on real images at that height or comparable camera ground resolution when available. Until then, results at 2.5 km are simulation results only.

## Further reading

- [OpenCV: planar homographies and perspective geometry](https://docs.opencv.org/4.10.0/d9/dab/tutorial_homography.html)
- [IRFormer: visible-to-infrared translation repository](https://github.com/CXH-Research/IRFormer)
- [ThermalGen: RGB-to-thermal translation repository](https://github.com/arplaboratory/ThermalGen)
- [XoFTR: matching thermal and visible images](https://openaccess.thecvf.com/content/CVPR2024W/IMW/html/Tuzcuoglu_XoFTR_Cross-modal_Feature_Matching_Transformer_CVPRW_2024_paper.html) — a matcher, not an image generator.

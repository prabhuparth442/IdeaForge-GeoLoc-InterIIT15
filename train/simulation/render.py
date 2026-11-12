"""render_view: pose + camera + map -> simulated thermal drone image with a validity mask and full metadata."""
import numpy as np

from train.simulation.appearance import METHODS
from train.simulation.ground_rays import ground_hits
from train.simulation.map_sampler import footprint_metres, read_ground_crop, sample_into_view
from train.simulation.sensor import apply_sensor

MARGIN_PIXELS = 4               # extra map around the footprint, in crop pixels


def crop_box(east, north, valid, base_gsd):
    margin = MARGIN_PIXELS * base_gsd
    return east[valid].min() - margin, east[valid].max() + margin, north[valid].min() - margin, north[valid].max() + margin


def render_view(satellite_map, camera, pose, rng, appearance="regional_detail", strength=1.0, sensor_settings=None,
                terrain=None, map_name="", extra_meta=None):
    """Simulate one drone image.

    satellite_map: header.satellite_map.SatelliteMap. camera: SimCamera or any camera object. pose: see simulation.pose.
    appearance: name in appearance.METHODS or a function (grey, rng, strength, pixel_m) -> picture. terrain: see ground_rays.ground_hits.
    Returns dict(image uint8, valid bool, view float (appearance before the sensor), east, north, meta dict).
    """
    east, north, ground_height, ray_valid = ground_hits(camera, pose, terrain)
    assert ray_valid.any(), "no pixel of this camera sees the ground for this pose"
    footprint = footprint_metres(east, north, ray_valid)
    base_gsd = max(float(np.nanpercentile(footprint, 10)), satellite_map.gsd)

    east_min, east_max, north_min, north_max = crop_box(east, north, ray_valid, base_gsd)
    crop = read_ground_crop(satellite_map, east_min, east_max, north_min, north_max, base_gsd)

    if isinstance(appearance, str):
        appearance_name = appearance
        appearance_function = METHODS[appearance]
    else:
        appearance_name = getattr(appearance, "__name__", "custom")
        appearance_function = appearance
    thermal_crop = appearance_function(crop["image"], rng, strength, base_gsd)
    known_crop = crop["image"] > 0
    if known_crop.any():
        fill_value = float(thermal_crop[known_crop].mean())
    else:
        fill_value = 0.0
    thermal_crop = np.where(known_crop, thermal_crop, fill_value).astype(np.float32)
    view, known = sample_into_view(crop, east, north, ray_valid, footprint, values=thermal_crop)

    valid = ray_valid & (known > 0.99)
    image = apply_sensor(view, valid, rng, sensor_settings)

    centre_col = int(min(max(round(camera.cx), 0), camera.width - 1))
    centre_row = int(min(max(round(camera.cy), 0), camera.height - 1))
    meta = dict(
        camera=dict(fx=camera.fx, fy=camera.fy, cx=camera.cx, cy=camera.cy, width=camera.width, height=camera.height,
                    k1=camera.k1, k2=camera.k2, k3=camera.k3, p1=camera.p1, p2=camera.p2),
        pose=pose,
        camera_ground_point=dict(east=pose["east"], north=pose["north"], note="point straight below the camera"),
        image_centre_ground_point=dict(east=float(east[centre_row, centre_col]), north=float(north[centre_row, centre_col]),
                                       note="ground point seen by the pixel at the principal point"),
        map=dict(name=map_name, crs=str(satellite_map.file.crs), gsd=float(satellite_map.gsd), bands=list(satellite_map.bands)),
        ground_model="flat" if terrain is None else "terrain function",
        crop_gsd_m=float(base_gsd), pixel_footprint_m=dict(median=float(np.nanmedian(footprint)), min=float(np.nanmin(footprint)), max=float(np.nanmax(footprint))),
        appearance=dict(name=appearance_name, strength=strength), sensor=sensor_settings or "default",
        valid_fraction=float(valid.mean()), simulator_version="0.1",
    )
    if extra_meta:
        meta.update(extra_meta)
    return dict(image=image, valid=valid, view=view, east=east, north=north, meta=meta)

"""Block: drone-view "photos" rendered from SATELLITE imagery only (Test_Iteration_11), so a learned decision can be trained without any
thermal photo (the team's satellite-only rule). No GPU, no plotting.

The pipeline's map is Cartosat scene A (268234931). A training photo is cut from a DIFFERENT satellite image of the same ground: Cartosat scene
B (268234921, another pass, 08-Feb-2025) or Pleiades Neo (small_sat_area, Jan 2026, another sensor), so photo and map really differ (date,
sun, sensor, buildings), as a thermal photo and the map do. Then thermal-like changes: random polarity flip (hot / cold), tone curve, blur to
the drone's ground resolution, sensor noise.

Geometry is the exact inverse of image_processing.geometry.make_query for a level camera (pitch = roll = 0): photo pixel (u, v) -> distorted
normalised (x_d, y_d) -> undistorted (x, y) (fixed-point inverse of image_processing.camera.distort) -> ground offset (east, north) at the
given height and yaw -> UTM -> raster pixel. So the pipeline, given the same camera, height and yaw, sees the ground the photo shows.

Source(path, bands)                         a GeoTIFF (UTM 43N) and the bands averaged into one grey image
undistort(camera, xd, yd)                   inverse of camera.distort
render(source, east, north, height, yaw, camera, rng, size, thermal)
                                            one uint16 photo (rows, cols) = (256, 320)
sample_places(rng, n, box, exclude)          uniform UTM centres inside box (x0, x1, y0, y1), outside the exclude box
make_set(out_dir, specs, camera, seed, geometry)
                                            renders every photo, writes PNGs and items.csv; returns the items (absolute file names, lat / lon of
                                            the true centre, rel_height, metadata yaw with a heading error, pitch = roll = 0)
"""
import math
import os

import cv2
import numpy as np
import pandas as pd

from image_processing.camera import distort


class Source:
    def __init__(self, path, bands=(1,)):
        import rasterio
        self.path = str(path)
        self.ds = rasterio.open(self.path)
        self.bands = list(bands)
        self.res = float(self.ds.res[0])

    def crop(self, x0, x1, y0, y1):
        """Grey float32 crop covering UTM [x0, x1] x [y0, y1]; returns (array, left, top) of the crop."""
        from rasterio.windows import from_bounds
        w = from_bounds(x0, y0, x1, y1, self.ds.transform).round_offsets().round_lengths()
        a = self.ds.read(self.bands, window=w, boundless=True, fill_value=0).astype(np.float32).mean(axis=0)
        left, top = self.ds.transform * (w.col_off, w.row_off)
        return a, left, top


def undistort(camera, xd, yd, rounds=20):
    x, y = xd.copy(), yd.copy()
    for _ in range(rounds):
        fx, fy = distort(camera, x, y)
        x, y = x - (fx - xd), y - (fy - yd)
    return x, y


def render(source, east, north, height, yaw_deg, camera, rng, size=(256, 320), thermal=True, strength=0.0):
    """strength: 0 = mild changes (polarity, tone, light blur and noise); higher adds a random non-monotonic grey-level map (probability
    0.5 x strength) and more blur and noise, to bring the matching difficulty towards real thermal photos."""
    rows, cols = size
    u, v = np.meshgrid(np.arange(cols, dtype=np.float64), np.arange(rows, dtype=np.float64))
    x, y = undistort(camera, (u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy)
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    xc, yc = x * height, y * height                       # camera-frame offsets at the ground (z_cam = height)
    de, dn = c * xc - s * yc, -s * xc - c * yc            # inverse of camera_axes(yaw, 0, 0) (the matrix is its own inverse)
    radius = float(np.hypot(de, dn).max()) + 30
    a, left, top = source.crop(east - radius, east + radius, north - radius, north + radius)
    if (a > 0).mean() < 0.98:
        return None                                       # outside the raster (no-data)
    gsd = height / camera.fx                              # the drone's metres per photo pixel
    sigma = 0.5 * gsd / source.res
    if sigma > 0.3:
        a = cv2.GaussianBlur(a, (0, 0), sigma)
    map_x = ((east + de - left) / source.res).astype(np.float32)
    map_y = ((top - (north + dn)) / source.res).astype(np.float32)
    img = cv2.remap(a, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    info = dict(flip=False, gamma=1.0, lut=False, blur=0.0, noise=0.0, strength=float(strength))
    if thermal:
        if rng.random() < 0.35:
            img = 1 - img; info["flip"] = True
        info["gamma"] = float(np.exp(rng.uniform(np.log(0.6), np.log(1.6)))); img = img ** info["gamma"]
        if strength > 0 and rng.random() < min(1.0, 0.5 * strength):
            # thermal brightness is not optical brightness: a random, non-monotonic grey-level map (materials swap order)
            knots = rng.uniform(0, 1, 2 + int(2 * strength)); knots[0], knots[-1] = rng.uniform(0, 0.4), rng.uniform(0.6, 1)
            img = np.interp(img, np.linspace(0, 1, len(knots)), knots); info["lut"] = True
        info["blur"] = float(rng.uniform(0.3, 1.2) * (1 + 0.6 * strength)); img = cv2.GaussianBlur(img, (0, 0), info["blur"])
        info["noise"] = float(rng.uniform(0.005, 0.03) * (1 + strength)); img = img + rng.normal(0, info["noise"], img.shape)
    img = np.clip(img, 0, 1)
    return (18000 + img * 4000).astype(np.uint16), info     # the dev photos' raw range is about 19000-22000


def sample_places(rng, n, box, exclude=None):
    out = []
    while len(out) < n:
        e, nn = rng.uniform(box[0], box[1]), rng.uniform(box[2], box[3])
        if exclude is not None and exclude[0] <= e <= exclude[1] and exclude[2] <= nn <= exclude[3]:
            continue
        out.append((e, nn))
    return out


def make_set(out_dir, specs, camera, seed=11, heights=(100, 350), yaw_error_deg=4.0, geometry=None, strength=0.0):
    """specs: list of dict(name, source=Source, n, box, exclude). geometry: the pipeline's settings.GEOMETRY for these photos (official:
    yaw_offset 180, scale 1.04): the photo is rendered as the pipeline will un-render it (height x scale, metadata yaw + yaw_offset).
    Returns a DataFrame of items (one row per rendered photo)."""
    g = {**dict(yaw_offset=0.0, scale=1.0, height_offset=0.0), **(geometry or {})}
    from pyproj import Transformer
    to_gps = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for spec in specs:
        made = 0
        for e, n in sample_places(rng, spec["n"] * 3, spec["box"], spec.get("exclude")):
            if made >= spec["n"]:
                break
            h = float(rng.uniform(*heights)); yaw = float(rng.uniform(0, 360))
            st = float(rng.uniform(*strength)) if isinstance(strength, (tuple, list)) else float(strength)     # a range: one strength per photo
            out = render(spec["source"], e, n, (h + g["height_offset"]) * g["scale"], yaw, camera, rng, strength=st)
            if out is None:
                continue
            img, info = out
            name = os.path.abspath(os.path.join(out_dir, "%s_%04d.png" % (spec["name"], made)))
            cv2.imwrite(name, img)
            lon, lat = to_gps.transform(e, n)
            meta_yaw = yaw - g["yaw_offset"] + float(np.clip(rng.normal(0, yaw_error_deg), -10, 10))   # what the pipeline reads
            rows.append(dict(name=name, source=spec["name"], east=e, north=n, lat=lat, lon=lon, rel_height=h, yaw=meta_yaw, true_yaw=yaw,
                             pitch=0.0, roll=0.0, **info))
            made += 1
    items = pd.DataFrame(rows)
    items.to_csv(os.path.join(out_dir, "items.csv"), index=False)
    return items

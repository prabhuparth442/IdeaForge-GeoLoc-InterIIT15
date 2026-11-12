"""Tests for the drone-view simulator (train/simulation) on a small synthetic map written as a GeoTIFF, no dataset needed.
Run: python tests/train/test_simulation.py"""
import sys
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests" / "pipeline"))
import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from common import run_tests

from header.satellite_map import SatelliteMap
from image_processing.geometry import make_query
from train.simulation.camera_model import SimCamera
from train.simulation.dataset import generate, random_poses
from train.simulation.ground_rays import ground_hits, project_to_pixel
from train.simulation.pose import NEUTRAL_GEOMETRY, make_pose
from train.simulation.render import render_view

ORIGIN_EAST, ORIGIN_NORTH, MAP_PIXELS = 500000.0, 3000000.0, 2400      # 720 m x 720 m at 0.3 m
QUIET_SENSOR = dict(blur_sigma_px=0.0, noise=0.0, stripe=0.0)
_cache = {}


def keep_grey(grey, rng, strength=1.0, pixel_m=1.0):
    return grey


def make_map():
    """Random smooth texture plus a bright disc at a known place. Returns (SatelliteMap, disc_east, disc_north)."""
    if "map" in _cache:
        return _cache["map"]
    rng = np.random.default_rng(1)
    texture = np.zeros((MAP_PIXELS, MAP_PIXELS), np.float32)
    for sigma, weight in ((2, 1.0), (6, 1.5), (18, 2.0)):
        texture += weight * cv2.GaussianBlur(rng.normal(0, 1, texture.shape).astype(np.float32), (0, 0), sigma) * sigma
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    texture = 40 + 160 * texture
    disc_east, disc_north = ORIGIN_EAST + 360.0, ORIGIN_NORTH - 330.0
    cv2.circle(texture, (int((disc_east - ORIGIN_EAST) / 0.3), int((ORIGIN_NORTH - disc_north) / 0.3)), 20, 255, -1)
    data = np.clip(texture, 1, 255).astype(np.uint8)
    folder = tempfile.mkdtemp()
    path = Path(folder) / "world.tif"
    with rasterio.open(path, "w", driver="GTiff", height=MAP_PIXELS, width=MAP_PIXELS, count=3, dtype="uint8",
                       crs="EPSG:32649", transform=from_origin(ORIGIN_EAST, ORIGIN_NORTH, 0.3, 0.3)) as handle:
        for band in range(3):
            handle.write(data, band + 1)
    _cache["map"] = (SatelliteMap(path), disc_east, disc_north, data)
    return _cache["map"]


def camera_with_lens():
    return SimCamera(fx=200.0, fy=200.0, cx=159.5, cy=127.5, width=320, height=256, k1=-0.25, k2=0.05, p1=0.0001, p2=0.0015)


def test_rays_and_projection_agree():
    """pixel_rays (undistort) and project_to_pixel (forward lens model) are written separately and must invert each other."""
    camera = camera_with_lens()
    for pose in (make_pose(0, 0, 120, 30, 0, 0), make_pose(0, 0, 120, 200, 12, -9), make_pose(5, -3, 80, 310, -25, 20)):
        east, north, _, valid = ground_hits(camera, pose)
        cols, rows, in_front = project_to_pixel(camera, pose, east, north)
        cols_want, rows_want = np.meshgrid(np.arange(camera.width), np.arange(camera.height))
        error = np.hypot(cols - cols_want, rows - rows_want)[valid]
        print("   pose tilt (%.0f, %.0f): worst round-trip error %.5f px" % (pose["pitch"], pose["roll"], error.max()))
        assert error.max() < 0.01


def test_nadir_footprint_matches_pinhole_formula():
    camera = SimCamera(200.0, 200.0, 159.5, 127.5, 320, 256)
    pose = make_pose(100, 200, 105, yaw=0)
    east, north, _, valid = ground_hits(camera, pose)
    assert valid.all()
    pixel_size = 105 / 200.0
    step = east[127, 160] - east[127, 159]
    assert abs(step - pixel_size) < 1e-6, step
    assert abs((north[127, 159] - north[128, 159]) - pixel_size) < 1e-6    # image down = south at yaw 0
    centre_east = 0.5 * (east[127, 159] + east[127, 160])
    assert abs(centre_east - 100) < 1e-6                                   # the principal point sees the point below the camera


def test_yaw_turns_the_picture_up_direction():
    camera = SimCamera(200.0, 200.0, 159.5, 127.5, 320, 256)
    east, north, _, _ = ground_hits(camera, make_pose(0, 0, 100, yaw=90))
    # yaw 90 = picture up points east: going up the image increases east
    assert east[10, 160] > east[240, 160] + 50
    assert abs(north[10, 160] - north[240, 160]) < 1e-6


def test_known_bright_disc_lands_where_projection_says():
    """A bright disc of known ground position must appear at the pixel that project_to_pixel gives, also with tilt and lens distortion."""
    satellite_map, disc_east, disc_north, _ = make_map()
    camera = camera_with_lens()
    for pose in (make_pose(disc_east - 30, disc_north + 20, 110, 15, 0, 0), make_pose(disc_east + 40, disc_north - 50, 100, 250, 15, -10)):
        result = render_view(satellite_map, camera, pose, np.random.default_rng(0), appearance=keep_grey, sensor_settings=QUIET_SENSOR)
        col, row, in_front = project_to_pixel(camera, pose, disc_east, disc_north)
        assert 20 < col < camera.width - 20 and 20 < row < camera.height - 20, "test disc should be in view: %.0f %.0f" % (col, row)
        saturated = (result["view"] >= 0.999).astype(np.uint8)       # the disc is the biggest saturated blob
        count, labels, stats, centres = cv2.connectedComponentsWithStats(saturated)
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        found_col, found_row = centres[biggest]
        print("   disc predicted at (%.1f, %.1f), found at (%.1f, %.1f)" % (col, row, found_col, found_row))
        assert abs(found_col - col) < 1.5 and abs(found_row - row) < 1.5


def shift_between(first, second):
    window = cv2.createHanningWindow(first.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(first.astype(np.float32), second.astype(np.float32), window)
    return dx, dy, response


def test_round_trip_through_the_pipeline_projection():
    """simulated photo -> pipeline's make_query (photo to north-up canvas) must land on the map crop it came from (shift ~ 0)."""
    satellite_map, _, _, data = make_map()
    camera = SimCamera(200.0, 200.0, 159.5, 127.5, 320, 256)
    folder = Path(tempfile.mkdtemp())
    for tilt in (0.0, 12.0):
        pose = make_pose(ORIGIN_EAST + 330, ORIGIN_NORTH - 360, 110, yaw=40, pitch=tilt, roll=-tilt / 2)
        result = render_view(satellite_map, camera, pose, np.random.default_rng(0), appearance=keep_grey, sensor_settings=QUIET_SENSOR)
        cv2.imwrite(str(folder / "sim.png"), result["image"])
        item = dict(name=str(folder / "sim.png"), rel_height=pose["height"], yaw=pose["yaw"], pitch=pose["pitch"], roll=pose["roll"])
        canvas, valid = make_query(item, camera, 160, 1.0, geometry=NEUTRAL_GEOMETRY)
        # the direct map crop around the point below the camera, same 1 m pixels, north-up
        col = (pose["east"] - ORIGIN_EAST) / 0.3
        row = (ORIGIN_NORTH - pose["north"]) / 0.3
        window = data[int(row - 267):int(row + 266), int(col - 267):int(col + 266)]     # 533 map pixels = 160 m
        direct = cv2.resize(window, (160, 160), interpolation=cv2.INTER_AREA).astype(np.float32)
        inside = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        both = np.where(inside, 1.0, 0.0).astype(np.float32)
        raw = result["image"].astype(np.float32)
        canvas_raw, _ = make_query(dict(item, name=str(folder / "sim.png")), camera, 160, 1.0, geometry=NEUTRAL_GEOMETRY)
        dx, dy, response = shift_between(canvas_raw * both, ((direct - direct.mean()) / direct.std()) * both)
        print("   tilt %.0f: canvas-to-map shift (%.2f, %.2f) px of 1 m, response %.2f" % (tilt, dx, dy, response))
        assert abs(dx) < 1.0 and abs(dy) < 1.0, "projection round trip is off by more than 1 m"


def test_far_view_equals_area_average_of_the_map():
    """From far away (1.5 m per pixel, map has 0.3 m detail) the view must be the area average of the map, not single map pixels."""
    satellite_map, _, _, data = make_map()
    camera = SimCamera(100.0, 100.0, 159.5, 127.5, 320, 256)
    pose = make_pose(ORIGIN_EAST + 360, ORIGIN_NORTH - 360, 150, yaw=0, pitch=0)
    result = render_view(satellite_map, camera, pose, np.random.default_rng(0), appearance=keep_grey, sensor_settings=QUIET_SENSOR)
    assert result["meta"]["crop_gsd_m"] > 1.0
    assert result["valid"].mean() > 0.99
    half_cols = int(160 * 1.5 / 0.3)                  # 800 map pixels = 240 m on each side
    half_rows = int(128 * 1.5 / 0.3)
    middle_col = int((pose["east"] - ORIGIN_EAST) / 0.3)
    middle_row = int((ORIGIN_NORTH - pose["north"]) / 0.3)
    window = data[middle_row - half_rows:middle_row + half_rows, middle_col - half_cols:middle_col + half_cols]
    reference = cv2.resize(window, (320, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    naive = cv2.resize(window, (320, 256), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    view = result["view"]
    correlation = np.corrcoef(view.ravel(), reference.ravel())[0, 1]
    naive_correlation = np.corrcoef(naive.ravel(), reference.ravel())[0, 1]
    print("   correlation with area average %.4f (single-pixel sampling would give %.4f)" % (correlation, naive_correlation))
    # not 1.0: the crop grid and the camera grid are offset by a fraction of a pixel, so bilinear lookup blurs a little
    assert correlation > 0.95 and correlation > naive_correlation + 0.02


def test_invalid_outside_map_and_horizon():
    satellite_map, _, _, _ = make_map()
    camera = SimCamera(200.0, 200.0, 159.5, 127.5, 320, 256)
    edge_pose = make_pose(ORIGIN_EAST + 20, ORIGIN_NORTH - 360, 150, yaw=0)         # the left of the view is outside the map
    result = render_view(satellite_map, camera, edge_pose, np.random.default_rng(0), appearance=keep_grey, sensor_settings=QUIET_SENSOR)
    fraction = result["valid"].mean()
    print("   valid fraction near the map edge: %.2f" % fraction)
    assert 0.2 < fraction < 0.95
    steep = make_pose(ORIGIN_EAST + 360, ORIGIN_NORTH - 360, 150, pitch=60)         # the top of the view is above the horizon
    east, north, _, ray_valid = ground_hits(camera, steep)
    assert not ray_valid.all() and ray_valid.any()


def test_same_seed_same_image_and_files_written():
    satellite_map, disc_east, disc_north, _ = make_map()
    camera = SimCamera(200.0, 200.0, 79.5, 63.5, 160, 128)
    rng = np.random.default_rng(3)
    poses = random_poses(rng, 2, (ORIGIN_EAST + 200, ORIGIN_EAST + 500), (ORIGIN_NORTH - 500, ORIGIN_NORTH - 200), (80, 150), tilt_deg=8)
    folder = Path(tempfile.mkdtemp())
    first = generate(satellite_map, camera, poses, folder, seed=5)
    second = generate(satellite_map, camera, poses, folder / "again", seed=5)
    for one, two in zip(first, second):
        assert (one["image"] == two["image"]).all()
    for name in ("sim_0000.png", "sim_0000_valid.png", "sim_0000.json", "sim_0001.png"):
        assert (folder / name).exists(), name
    meta = first[0]["meta"]
    for key in ("camera", "pose", "camera_ground_point", "image_centre_ground_point", "map", "appearance", "seed", "simulator_version"):
        assert key in meta, key


def test_all_appearance_methods_run():
    satellite_map, disc_east, disc_north, _ = make_map()
    camera = SimCamera(200.0, 200.0, 79.5, 63.5, 160, 128)
    pose = make_pose(ORIGIN_EAST + 360, ORIGIN_NORTH - 360, 100, yaw=20)
    for name in ("plain_invert", "rule_based", "regional_detail"):
        result = render_view(satellite_map, camera, pose, np.random.default_rng(1), appearance=name)
        assert result["image"].dtype == np.uint8 and result["image"].std() > 5
        assert result["meta"]["appearance"]["name"] == name


def test_frechet_and_classifier_separate_different_sets():
    from train.simulation.compare import classifier_auc, frechet_distance
    rng = np.random.default_rng(0)
    same_a = rng.normal(0, 1, (300, 40))
    same_b = rng.normal(0, 1, (300, 40))
    shifted = rng.normal(0.8, 1, (300, 40))
    near = frechet_distance(same_a, same_b)
    far = frechet_distance(same_a, shifted)
    auc_same = classifier_auc(same_a, same_b)[0]
    auc_far = classifier_auc(same_a, shifted)[0]
    print("   frechet same %.2f vs shifted %.2f; auc same %.2f vs shifted %.2f" % (near, far, auc_same, auc_far))
    assert far > 3 * near and auc_same < 0.6 and auc_far > 0.9


def test_summary_vector_sees_blur_and_noise():
    from train.simulation.compare import SUMMARY_NAMES, summary_vector
    satellite_map, _, _, data = make_map()
    picture = data[800:1024, 800:1024].astype(np.float32) / 255
    valid = np.ones(picture.shape, bool)
    sharp = summary_vector(picture, valid)
    blurred = summary_vector(cv2.GaussianBlur(picture, (0, 0), 3.0), valid)
    noisy = summary_vector(picture + np.random.default_rng(0).normal(0, 0.1, picture.shape).astype(np.float32), valid)
    names = SUMMARY_NAMES
    assert len(sharp) == len(names) and np.isfinite(sharp).all()
    assert blurred[names.index("fine_detail")] < sharp[names.index("fine_detail")] < noisy[names.index("fine_detail")]
    assert noisy[names.index("power_slope_high")] > sharp[names.index("power_slope_high")] + 1.0      # noise flattens the high-frequency spectrum


def test_downstream_metrics_find_the_truth_on_the_synthetic_world():
    from image_processing.geometry import render_query
    from train.features.edge import EdgeFamily
    from train.simulation.downstream import coarse_truth_metrics
    from common import world
    w = world()
    test_set = w["testset"]
    level = int(test_set.query_levels[0])
    canvas, valid = render_query(test_set.items[0], test_set.camera, level)
    prior = test_set.truths[0] + np.array([300.0, -200.0])
    result = coarse_truth_metrics(canvas, valid, level, test_set.truths[0], prior, {"edge": EdgeFamily(w["store"])})
    print("   ", {key: round(value, 2) for key, value in result.items()})
    assert result["peak_error_m"] < 20 and result["margin"] > 0 and result["truth_percentile"] < 0.01


if __name__ == "__main__":
    run_tests(globals())

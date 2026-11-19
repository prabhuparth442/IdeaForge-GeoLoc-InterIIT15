"""Tests of the photo view (geoloc/photo_view.py): the map drawn into the photo's camera view and the position correction read from the matches.
No data, no weights: a smooth random texture stands in for the map level, and the 'matcher' finds the pixel shift by brute force, so the signs and
sizes of the correction are checked against a known truth. Run: python tests/test_photo_view.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cv2
import numpy as np
import torch

import settings
from geoloc import photo_view
from geoloc.levels import level_gsd

CAMERA = dict(fx=200.0, fy=200.0, cx=159.5, cy=127.5, k1=0, k2=0, k3=0, p1=0, p2=0, width=320, height=256)
POSE = dict(height=1400.0, yaw=0.0, pitch=0.0, roll=0.0)
LEVEL = 9


def texture():
    rng = np.random.default_rng(3)
    base = rng.random((700, 700)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 2.0)
    base = (base - base.min()) / (base.max() - base.min())
    return (base * 250).astype(np.uint8)


class ShiftMatcher:
    """Finds the integer pixel shift s with tile(p + s) = photo(p) by brute force and returns a grid of matches. Looks like the matchers of fine_matching."""
    min_confidence = 0.0

    def __call__(self, data):
        a = data["image0"][0, 0].cpu().numpy()
        b = data["image1"][0, 0].cpu().numpy()
        best, best_s = 1e9, (0, 0)
        core = a[60:196, 80:240]
        for sy in range(-12, 13):
            for sx in range(-12, 13):
                cost = float(np.mean((core - b[60 + sy:196 + sy, 80 + sx:240 + sx]) ** 2))
                if cost < best:
                    best, best_s = cost, (sx, sy)
        ys, xs = np.meshgrid(np.arange(70, 190, 12), np.arange(90, 230, 12))
        p = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
        q = p + np.array(best_s, np.float32)
        return dict(keypoints0=torch.from_numpy(p), keypoints1=torch.from_numpy(q), confidence=torch.ones(len(p)))


def build_view(image):
    view = photo_view.PhotoView.__new__(photo_view.PhotoView)
    view.pose, view.camera, view.level_images, view.level = POSE, CAMERA, {LEVEL: image}, LEVEL
    view.rays = photo_view.undistorted_rays(CAMERA)
    view.tile_sigma = 3.0
    view.grids = {}
    return view


def test_off_by_default():
    assert settings.PHOTO_VIEW_FROM_HEIGHT_M is None
    assert not photo_view.active(POSE)


def test_gate():
    old = settings.PHOTO_VIEW_FROM_HEIGHT_M
    try:
        settings.PHOTO_VIEW_FROM_HEIGHT_M = 1000.0
        assert photo_view.active(dict(POSE, height=1200.0)) and not photo_view.active(dict(POSE, height=900.0))
    finally:
        settings.PHOTO_VIEW_FROM_HEIGHT_M = old


def test_level_follows_photo_pixel():
    gsd = POSE["height"] * settings.GEOMETRY["scale"] / CAMERA["fx"]
    level = photo_view.level_like_photo(POSE["height"], CAMERA, [3, 5, 7, 9, 11])
    assert abs(np.log(level_gsd(level) / gsd)) < 0.4


def test_ground_grid_nadir_camera():
    view = build_view(texture())
    east, north = view.grid(dict(heading=0.0, scale_factor=1.0))
    g = settings.GEOMETRY
    expected = POSE["height"] * g["scale"] / CAMERA["fx"]
    # with no tilt the ground pixel is height / fx everywhere: neighbours are one pixel apart
    step = float(np.hypot(east[128, 161] - east[128, 160], north[128, 161] - north[128, 160]))
    assert abs(step - expected) / expected < 0.01, (step, expected)


def test_correction_recovers_known_offset():
    image = texture()
    view = build_view(image)
    query = dict(heading=0.0, scale_factor=1.0)
    gsd = level_gsd(LEVEL)
    truth = np.array([2400.0, 2300.0])                         # east, south metres of the camera's nadir point
    drawn, share = photo_view.draw_map(image, LEVEL, view.grid(query), truth[0], truth[1])
    assert share > 0.99
    view.picture = drawn.astype(np.float32)                    # the photo is the map seen from the true place
    offset = np.array([42.0, -28.0])                           # the candidate place is off by this (east, south)
    peaks = np.array([[1.0, truth[0] + offset[0], truth[1] + offset[1], 0.0, 0.0]])
    # draw_map's clean_tile step is part of check_places; the shift matcher works on the cleaned tile
    found = view.check_places(ShiftMatcher(), query, peaks, limit_m=500.0)
    assert found["inliers"][0] >= 20, found
    shift = found["shifts"][0]                                 # east, south metres to ADD to the candidate to reach the truth
    assert np.hypot(*(shift + offset)) < 1.5 * gsd, (shift, offset)


def test_shift_limit_rejects():
    image = texture()
    view = build_view(image)
    query = dict(heading=0.0, scale_factor=1.0)
    truth = np.array([2400.0, 2300.0])
    drawn, _ = photo_view.draw_map(image, LEVEL, view.grid(query), truth[0], truth[1])
    view.picture = drawn.astype(np.float32)
    peaks = np.array([[1.0, truth[0] + 42.0, truth[1] - 28.0, 0.0, 0.0]])
    found = view.check_places(ShiftMatcher(), query, peaks, limit_m=10.0)
    assert found["inliers"][0] == 0


def test_no_map_is_skipped():
    image = texture()
    view = build_view(image)
    peaks = np.array([[1.0, 99999.0, 99999.0, 0.0, 0.0], [0.5, 1e9, 1e9, 0.0, 0.0]])
    found = view.check_places(ShiftMatcher(), dict(heading=0.0, scale_factor=1.0), peaks, limit_m=500.0)
    assert found["inliers"].sum() == 0


def test_pose_fit_recovers_a_tilt_error_that_the_median_cannot():
    """Photo = the map seen by a camera with a 1.5 degree tilt error from the true place; the drawn map (metadata pose) has no such error. The matches are exact (found by the geometry,
    not by a matcher). The median vote is off by about height * tilt; the pose fit with its prior brings the position much closer."""
    image = texture()
    pose_true = dict(POSE, pitch=0.0, roll=0.0)
    old = settings.GEOMETRY
    try:
        settings.GEOMETRY = dict(old, attitude=True, pitch_sign=1, roll_sign=1, yaw_offset=0.0, scale=1.0, height_offset=0.0)
        rays = photo_view.undistorted_rays(CAMERA)
        truth = np.array([2400.0, 2300.0])
        # the camera really looks 1.5 degrees off: its ground grid, the metadata says 0
        real = photo_view.ground_grid(rays, dict(pose_true, roll=1.5), CAMERA)
        photo_pic, _ = photo_view.draw_map(image, LEVEL, real, truth[0], truth[1])
        assumed = photo_view.ground_grid(rays, pose_true, CAMERA)
        cand = truth + np.array([30.0, -20.0])
        # exact matches: photo pixel p sees the ground point truth + real(p); find the pixel q of the drawn map with assumed(q) = real(p) + truth - cand (nearest pixel of a dense lookup)
        ys, xs = np.meshgrid(np.arange(20, 236, 12), np.arange(30, 290, 12))
        pa = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
        want_e = real[0][pa[:, 1].astype(int), pa[:, 0].astype(int)] + (truth[0] - cand[0])
        want_n = real[1][pa[:, 1].astype(int), pa[:, 0].astype(int)] - (truth[1] - cand[1])
        flat_e, flat_n = assumed[0].ravel(), assumed[1].ravel()
        pb = np.array([[i % 320, i // 320] for i in (int(np.argmin((flat_e - e) ** 2 + (flat_n - n) ** 2)) for e, n in zip(want_e, want_n))], np.float32)
        match = dict(pa=pa, pb=pb, grid=assumed, heading=0.0, scale_factor=1.0, cand=(float(cand[0]), float(cand[1])))
        e0, n0 = photo_view.ground_at(pa, CAMERA, pose_true, 0.0, 1.0, 0, 0, 0, 0)
        votes = np.array([photo_view.sample(assumed[0], pb) - photo_view.sample(assumed[0], pa), photo_view.sample(assumed[1], pb) - photo_view.sample(assumed[1], pa)])
        median_xy = cand + np.array([np.median(votes[0]), -np.median(votes[1])])
        gsd = POSE["height"] / CAMERA["fx"]
        x, y, corrections, rms = photo_view.fit_pose(match, pose_true, CAMERA, gsd)
        median_error = float(np.hypot(*(median_xy - truth)))
        fit_error = float(np.hypot(x - truth[0], y - truth[1]))
        assert median_error > 15.0, median_error                 # the tilt really moves the median
        assert fit_error < 0.6 * median_error, (fit_error, median_error)
    finally:
        settings.GEOMETRY = old


def test_second_opinion_is_off_by_default_and_limited_by_height():
    assert settings.PHOTO_VIEW_VERIFY_BELOW is None and settings.PHOTO_VIEW_VERIFY_ACT is False and settings.PHOTO_VIEW_VERIFY_MAX_HEIGHT_M is None
    assert settings.PHOTO_VIEW_FIT_POSE is False and settings.PHOTO_VIEW_FIT_ROUNDS == 1


def test_fit_pose_prior_centre_keeps_the_prior_on_the_metadata():
    """A later round of the loop passes the corrections already applied; the prior then pulls toward the metadata, not toward the corrected pose."""
    image = texture()
    view = build_view(image)
    query = dict(heading=0.0, scale_factor=1.0)
    grid = view.grid(query)
    ys, xs = np.meshgrid(np.arange(20, 236, 24), np.arange(30, 290, 24))
    pa = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
    match = dict(pa=pa, pb=pa.copy(), grid=grid, heading=0.0, scale_factor=1.0, cand=(2400.0, 2300.0))
    x1, y1, c1, r1 = photo_view.fit_pose(match, POSE, CAMERA, POSE["height"] / CAMERA["fx"])
    x2, y2, c2, r2 = photo_view.fit_pose(match, POSE, CAMERA, POSE["height"] / CAMERA["fx"], prior_centre=(0.0, 0.0, 0.0, 0.0))
    assert abs(x1 - x2) < 1e-6 and abs(y1 - y2) < 1e-6              # a zero centre is the same as none
    assert abs(x1 - 2400.0) < 1.0 and abs(y1 - 2300.0) < 1.0        # identical pictures: nothing to correct


if __name__ == "__main__":
    failed = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                function()
                print("pass", name)
            except Exception as problem:
                failed += 1
                print("FAIL", name, repr(problem))
    sys.exit(1 if failed else 0)

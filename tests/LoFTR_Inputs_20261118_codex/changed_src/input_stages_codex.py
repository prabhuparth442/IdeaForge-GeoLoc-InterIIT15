"""Reconstruct input stages from a frame, metadata row and saved map windows.

Inputs are explicit paths, one metadata row and two stored satellite tiles.
Returns named image arrays and sampling measurements. No model is trained or run.
The imported headless settings must retain their default cleaning switches.
"""
import hashlib

import cv2
import numpy as np

import settings
from geoloc import fine_matching, native_fine, photo
from geoloc.levels import level_gsd


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stretch_for_display(frame):
    low, high = np.percentile(frame, [1, 99])
    return np.clip((frame.astype(np.float32) - low) / (high - low + 1e-6), 0, 1)


def draw_input(picture, pose, camera, level, size):
    canvas, valid = photo.draw_canvas(picture, pose, camera, level, size=size)
    gray = fine_matching.grey_canvas(canvas, valid)
    return np.clip(gray, 0, 1).astype(np.float32), valid


def gradient_rms(picture):
    horizontal = np.diff(picture[4:-4, 4:-4], axis=1)
    vertical = np.diff(picture[4:-4, 4:-4], axis=0)
    return float(np.sqrt((np.mean(horizontal ** 2) + np.mean(vertical ** 2)) / 2))


def source_inside_canvas(pose, camera, level, size):
    geometry = settings.GEOMETRY
    intrinsic = np.array([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]], dtype=float)
    distortion = np.array([camera["k1"], camera["k2"], camera["p1"], camera["p2"], camera["k3"]])
    cols, rows = np.meshgrid(np.arange(camera["width"]), np.arange(camera["height"]))
    points = np.stack([cols, rows], axis=-1).astype(np.float64)
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 50, 1e-10)
    undistorted = cv2.undistortPointsIter(points.reshape(-1, 1, 2), intrinsic, distortion, None, None, criteria).reshape(-1, 2)
    distorted_x, distorted_y = photo.distort(camera, undistorted[:, 0], undistorted[:, 1])
    restored = np.column_stack([distorted_x * camera["fx"] + camera["cx"], distorted_y * camera["fy"] + camera["cy"]])
    assert np.max(np.abs(restored - points.reshape(-1, 2))) < 1e-5
    pitch = 0.0
    roll = 0.0
    if geometry["attitude"]:
        pitch = geometry["pitch_sign"] * pose["pitch"]
        roll = geometry["roll_sign"] * pose["roll"]
    axis_x, axis_y, axis_z = photo.camera_axes(pose["yaw"] + geometry["yaw_offset"], pitch, roll)
    rays = undistorted[:, :1] * axis_x + undistorted[:, 1:] * axis_y + axis_z
    height = (pose["height"] + geometry["height_offset"]) * geometry["scale"]
    distance = -height / rays[:, 2]
    ground = rays[:, :2] * distance[:, None]
    half_width = size * level_gsd(level) / 2
    inside = (np.abs(ground) < half_width).all(axis=1) & (distance > 0)
    return inside.reshape(camera["height"], camera["width"])


def reconstruct_inputs(frame_path, metadata, level, base_tile, finer_tile):
    camera = photo.read_camera(metadata)
    pose = photo.read_pose(metadata)
    raw = photo.load_photo(str(frame_path))
    normalized = photo.normalise_frame(raw)
    prepared = photo.prepare_photo(str(frame_path), pose, camera, level)
    native = photo.prepare_photo(str(frame_path), pose, camera, level - 2)
    cleaned = photo.prepare_photo(str(frame_path), pose, camera, level - 2, native_clean=True)
    arrays = {
        "raw_uint16": raw,
        "raw_display_only": stretch_for_display(raw),
        "normalized_unblurred": normalized,
        "prepared_blurred": prepared,
    }
    arrays["base_photo"], arrays["base_valid"] = draw_input(prepared, pose, camera, level, 224)
    arrays["no_blur_photo"], _ = draw_input(normalized, pose, camera, level, 224)
    arrays["native_photo"], arrays["native_valid"] = draw_input(native, pose, camera, level - 2, 448)
    arrays["clean_native_photo"], _ = draw_input(cleaned, pose, camera, level - 2, 448)
    arrays["base_map"] = base_tile.astype(np.float32)
    arrays["native_map"] = finer_tile.astype(np.float32)
    photo_gsd = pose["height"] / camera["fx"]
    tile_sigma = native_fine.tile_sigma(photo_gsd, level_gsd(level - 2))
    arrays["clean_native_map"] = native_fine.clean_tile(finer_tile, tile_sigma)
    arrays["source_inside_canvas"] = source_inside_canvas(pose, camera, level, 224)
    arrays["upscaled_photo"] = cv2.resize(arrays["base_photo"], (448, 448), interpolation=cv2.INTER_LINEAR)
    arrays["upscaled_map"] = cv2.resize(base_tile, (448, 448), interpolation=cv2.INTER_LINEAR)
    measurements = {
        "filename": metadata["filename"],
        "height_m": pose["height"],
        "level": level,
        "photo_m_per_pixel": photo_gsd,
        "base_m_per_pixel": level_gsd(level),
        "native_m_per_pixel": level_gsd(level - 2),
        "base_blur_sigma_photo_px": max(0, 0.5 * level_gsd(level) / photo_gsd - 0.25),
        "native_blur_sigma_photo_px": max(0, 0.5 * level_gsd(level - 2) / photo_gsd - 0.25),
        "gradient_rms_retained_after_preblur": gradient_rms(prepared) / gradient_rms(normalized),
        "base_valid_pixels": int(arrays["base_valid"].sum()),
        "native_valid_pixels": int(arrays["native_valid"].sum()),
        "source_centres_outside_canvas_percent": 100 * (1 - arrays["source_inside_canvas"].mean()),
        "raw_sha256": file_digest(frame_path),
    }
    assert base_tile.shape == (224, 224)
    assert finer_tile.shape == (448, 448)
    assert np.isclose(224 * level_gsd(level), 448 * level_gsd(level - 2))
    for name, value in arrays.items():
        assert np.isfinite(value).all(), name
    return arrays, measurements

import math

import cv2
import numpy as np

from header import settings
from image_processing.camera import distort
from image_processing.map_levels import level_gsd
from image_processing.thermal import load_thermal_frame, normalise_frame


def camera_axes(yaw_deg, pitch_deg, roll_deg):
    assert all(math.isfinite(a) for a in (yaw_deg, pitch_deg, roll_deg)), "yaw, pitch and roll must be finite (degrees)"
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    forward = np.array([math.sin(yaw), math.cos(yaw), 0.0])
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    forward1 = forward * math.cos(pitch) - down * math.sin(pitch)
    down1 = down * math.cos(pitch) + forward * math.sin(pitch)
    right2 = right * math.cos(roll) + down1 * math.sin(roll)
    down2 = down1 * math.cos(roll) - right * math.sin(roll)
    return right2, -forward1, down2


def make_query(item, camera, canvas_size=settings.TILE, pixel_size=1.0, geometry=None):
    assert canvas_size > 0 and pixel_size > 0, "canvas_size (pixels) and pixel_size (metres) must be positive"
    assert item["rel_height"] > 0, "photo height must be above 0 m"
    assert camera.fx > 0 and camera.fy > 0, "camera focal lengths must be positive"
    assert geometry is None or set(geometry) <= set(settings.GEOMETRY), "unknown geometry key: %s" % (set(geometry) - set(settings.GEOMETRY),)
    setting = dict(settings.GEOMETRY)
    if geometry is not None:
        setting.update(geometry)

    photo = normalise_frame(load_thermal_frame(item))
    photo_gsd = item["rel_height"] / camera.fx
    blur = max(0.0, 0.5 * pixel_size / photo_gsd - 0.25)
    if blur > 0.2:
        photo = cv2.GaussianBlur(photo, (0, 0), blur)

    height = (item["rel_height"] + setting["height_offset"]) * setting["scale"]
    yaw = item["yaw"] + setting["yaw_offset"]
    pitch = 0.0
    roll = 0.0
    if setting["attitude"]:
        pitch = setting["pitch_sign"] * item["pitch"]
        roll = setting["roll_sign"] * item["roll"]
    axis_x, axis_y, axis_z = camera_axes(yaw, pitch, roll)

    middle = (canvas_size - 1) / 2
    cols, rows = np.meshgrid(np.arange(canvas_size), np.arange(canvas_size))
    east = (cols - middle) * pixel_size
    north = -(rows - middle) * pixel_size
    below = -height

    x_cam = axis_x[0] * east + axis_x[1] * north + axis_x[2] * below
    y_cam = axis_y[0] * east + axis_y[1] * north + axis_y[2] * below
    z_cam = axis_z[0] * east + axis_z[1] * north + axis_z[2] * below
    in_front = z_cam > 1e-3
    z_safe = np.where(in_front, z_cam, 1.0)
    x_real, y_real = distort(camera, x_cam / z_safe, y_cam / z_safe)
    map_x = (camera.fx * x_real + camera.cx).astype(np.float32)
    map_y = (camera.fy * y_real + camera.cy).astype(np.float32)

    canvas = cv2.remap(photo, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inside = cv2.remap(np.ones_like(photo), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.99
    valid = inside & in_front
    canvas = np.where(valid, canvas, photo.mean())
    return canvas, valid


def render_query(item, camera, level, geometry=None):
    assert level in settings.LEVELS, "level %s is not one of %s" % (level, settings.LEVELS)
    return make_query(item, camera, settings.TILE, level_gsd(level), geometry)

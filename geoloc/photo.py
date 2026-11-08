"""From one thermal photo and its metadata row to the north-up canvas the rest of the pipeline works on."""
import math

import cv2
import numpy as np

import settings
from geoloc.levels import level_gsd


def wrap_angle(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0


def read_camera(meta):
    """The camera numbers of one metadata row (strings from the CSV are fine)."""
    camera = {}
    for key in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2"):
        camera[key] = float(meta["cam_" + key])
    camera["width"] = 320
    camera["height"] = 256
    return camera


def read_pose(meta):
    return dict(height=float(meta["alt"]), yaw=float(meta["yaw"]),
                pitch=wrap_angle(float(meta["pitch"])), roll=wrap_angle(float(meta["roll"])))


def load_photo(path):
    frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise FileNotFoundError("cannot read the photo " + path)
    if frame.ndim == 3:
        frame = frame.mean(axis=2)
    return frame


def normalise_frame(frame):
    """Stretch the 16-bit thermal values to 0..1 and apply local contrast (CLAHE)."""
    frame = frame.astype(np.float32)
    low, high = np.percentile(frame, [1, 99])
    stretched = np.clip((frame - low) / (high - low + 1e-6), 0, 1)
    as_16bit = (stretched * 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(as_16bit).astype(np.float32) / 65535


def prepare_photo(path, pose, camera, level):
    """Photo ready for drawing at this zoom level. Done once per photo, every heading query reuses it."""
    photo = normalise_frame(load_photo(path))
    photo_gsd = pose["height"] / camera["fx"]
    blur = max(0.0, 0.5 * level_gsd(level) / photo_gsd - 0.25)      # the canvas pixel is bigger than the photo pixel
    if blur > 0.2:
        photo = cv2.GaussianBlur(photo, (0, 0), blur)
    return photo


def camera_axes(yaw_deg, pitch_deg, roll_deg):
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


def distort(camera, x, y):
    r2 = x * x + y * y
    radial = 1 + camera["k1"] * r2 + camera["k2"] * r2 ** 2 + camera["k3"] * r2 ** 3
    x_d = x * radial + 2 * camera["p1"] * x * y + camera["p2"] * (r2 + 2 * x * x)
    y_d = y * radial + camera["p1"] * (r2 + 2 * y * y) + 2 * camera["p2"] * x * y
    return x_d, y_d


def draw_canvas(photo, pose, camera, level, heading=0.0, scale_factor=1.0):
    """The photo drawn north-up on a TILE x TILE canvas whose pixels are level_gsd(level) metres.
    heading (degrees) is added to the yaw and scale_factor multiplies the projection scale: this is one heading query.
    Returns canvas (floats 0..1) and valid (bool: pixels that the photo really covers)."""
    geometry = settings.GEOMETRY
    size = settings.TILE
    pixel_size = level_gsd(level)

    height = (pose["height"] + geometry["height_offset"]) * geometry["scale"] * scale_factor
    yaw = pose["yaw"] + geometry["yaw_offset"] + heading
    pitch = 0.0
    roll = 0.0
    if geometry["attitude"]:
        pitch = geometry["pitch_sign"] * pose["pitch"]
        roll = geometry["roll_sign"] * pose["roll"]
    axis_x, axis_y, axis_z = camera_axes(yaw, pitch, roll)

    middle = (size - 1) / 2
    cols, rows = np.meshgrid(np.arange(size), np.arange(size))
    east = (cols - middle) * pixel_size
    north = -(rows - middle) * pixel_size
    below = -height

    x_cam = axis_x[0] * east + axis_x[1] * north + axis_x[2] * below
    y_cam = axis_y[0] * east + axis_y[1] * north + axis_y[2] * below
    z_cam = axis_z[0] * east + axis_z[1] * north + axis_z[2] * below
    in_front = z_cam > 1e-3
    z_safe = np.where(in_front, z_cam, 1.0)
    x_real, y_real = distort(camera, x_cam / z_safe, y_cam / z_safe)
    map_x = (camera["fx"] * x_real + camera["cx"]).astype(np.float32)
    map_y = (camera["fy"] * y_real + camera["cy"]).astype(np.float32)

    canvas = cv2.remap(photo, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inside = cv2.remap(np.ones_like(photo), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.99
    valid = inside & in_front
    canvas = np.where(valid, canvas, photo.mean())
    return canvas, valid

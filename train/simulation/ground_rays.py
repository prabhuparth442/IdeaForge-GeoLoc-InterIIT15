"""For every camera pixel, the ground point it sees. This is the map-to-camera direction of the geometry."""
import cv2
import numpy as np

from image_processing.geometry import camera_axes

MIN_DOWNWARD = 0.05            # a ray must point at least this much downward (sine of the angle below the horizon)


def pixel_rays(camera, pose):
    """Unit-free ray directions in world axes (east, north, up) for every pixel, shape (height, width, 3).

    The lens distortion is removed first, so the rays are what the real camera would see.
    """
    cols, rows = np.meshgrid(np.arange(camera.width), np.arange(camera.height))
    pixels = np.stack([cols, rows], axis=-1).reshape(-1, 1, 2).astype(np.float64)
    matrix = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]], dtype=np.float64)
    distortion = np.array([camera.k1, camera.k2, camera.p1, camera.p2, camera.k3], dtype=np.float64)
    stop_rule = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 1e-10)
    normalised = cv2.undistortPointsIter(pixels, matrix, distortion, None, None, stop_rule)
    normalised = normalised.reshape(camera.height, camera.width, 2)

    axis_x, axis_y, axis_z = camera_axes(pose["yaw"], pose["pitch"], pose["roll"])
    x = normalised[:, :, 0][:, :, None]
    y = normalised[:, :, 1][:, :, None]
    return x * axis_x + y * axis_y + axis_z


def ground_hits(camera, pose, terrain=None):
    """Where each pixel's ray meets the ground.

    terrain: None for flat ground, or a function terrain(east, north) -> ground height in metres relative to the ground under
    the camera (arrays in, array out). With terrain the hit is refined by a few fixed-point steps.
    Returns east, north (metres, map CRS, shape (height, width)), ground_height, valid (bool: the ray hits ground in front of the camera).
    """
    rays = pixel_rays(camera, pose)
    down = rays[:, :, 2]
    valid = down < -MIN_DOWNWARD
    safe_down = np.where(valid, down, -1.0)

    ground_height = np.zeros_like(down)
    east = pose["east"] + rays[:, :, 0] * (ground_height - pose["height"]) / safe_down
    north = pose["north"] + rays[:, :, 1] * (ground_height - pose["height"]) / safe_down
    if terrain is not None:
        for _ in range(4):
            ground_height = terrain(east, north)
            east = pose["east"] + rays[:, :, 0] * (ground_height - pose["height"]) / safe_down
            north = pose["north"] + rays[:, :, 1] * (ground_height - pose["height"]) / safe_down
    east = np.where(valid, east, pose["east"])
    north = np.where(valid, north, pose["north"])
    return east, north, ground_height, valid


def project_to_pixel(camera, pose, east, north, ground_height=0.0):
    """The other direction, written separately from pixel_rays so the two can check each other: a ground point -> pixel (col, row).

    Returns (col, row, in_front). Uses the forward lens model (radial + tangential) directly.
    """
    axis_x, axis_y, axis_z = camera_axes(pose["yaw"], pose["pitch"], pose["roll"])
    east, north, ground_height = np.broadcast_arrays(np.asarray(east, dtype=np.float64), np.asarray(north, dtype=np.float64),
                                                     np.asarray(ground_height, dtype=np.float64))
    relative = np.stack([east - pose["east"], north - pose["north"], ground_height - pose["height"]], axis=-1)
    x_cam = relative @ axis_x
    y_cam = relative @ axis_y
    z_cam = relative @ axis_z
    in_front = z_cam > 1e-6
    z_safe = np.where(in_front, z_cam, 1.0)
    x = x_cam / z_safe
    y = y_cam / z_safe
    r2 = x * x + y * y
    radial = 1 + camera.k1 * r2 + camera.k2 * r2 ** 2 + camera.k3 * r2 ** 3
    x_d = x * radial + 2 * camera.p1 * x * y + camera.p2 * (r2 + 2 * x * x)
    y_d = y * radial + camera.p1 * (r2 + 2 * y * y) + 2 * camera.p2 * x * y
    return camera.fx * x_d + camera.cx, camera.fy * y_d + camera.cy, in_front

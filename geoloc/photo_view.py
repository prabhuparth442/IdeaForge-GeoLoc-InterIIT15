"""Photo view: the fine stage for high photos with the photo left as it is and the MAP drawn into the photo's own camera view.

Why (iteration 43, 2026-11-18): today the photo is warped onto a north-up 224 px canvas of the coarse level (a canvas pixel is up to 2 times the photo's pixel at 1400 m, 9 to 26 % of the photo's
pixels fall outside the canvas) and LoFTR/XoFTR matches that against a map tile. Turned round: the photo stays 320 x 256 at its own resolution (vignette removed, local normalisation, stretch + CLAHE)
and the map level whose pixel is closest to the photo's ground pixel is drawn into the photo's view with the photo's pose (the projection of the synthetic geometry audit), so the matcher sees the
map with the photo's perspective, tilt and heading. At the true place on the 130 working photos from 1000 m (Test_Iteration_12/results/photo_view): XoFTR recall at a bar set from wrong places 0.60 -> 0.95,
0.52 -> 0.98 above 1600 m; LoFTR 0.43 -> 0.70. Only the fine stage changes: the coarse search, the DINO order and the decision are the canvas path's.

Matches give the position correction directly. A photo pixel p sees the ground point  truth + G(p)  (G = east/north metres from the camera's nadir point, from the projection); the map drawn for the
candidate place c shows  c + G(q)  at pixel q. A match p <-> q therefore says  truth = c + G(q) - G(p). The RANSAC similarity (as in the canvas path) picks the inliers, the correction is the median of
their votes. Pipeline shifts are (east, south) metres, so the north part is negated.
"""
import math

import cv2
import numpy as np

import settings
from geoloc import fine_matching, flat_field, native_fine, photo
from geoloc.levels import level_gsd


def active(pose):
    """True when this photo uses the photo view."""
    return settings.PHOTO_VIEW_FROM_HEIGHT_M is not None and pose["height"] >= settings.PHOTO_VIEW_FROM_HEIGHT_M


def level_like_photo(height_m, camera, store_levels):
    """The stored level whose pixel is closest (in ratio) to the photo's ground pixel."""
    gsd = height_m * settings.GEOMETRY["scale"] / camera["fx"]
    return min((abs(math.log(level_gsd(level) / gsd)), level) for level in store_levels)[1]


def own_photo(path, camera):
    """The photo at its own resolution, cleaned."""
    frame = photo.load_photo(path)
    frame = flat_field.remove_vignette(frame, camera["cx"], camera["cy"], 3)
    frame = flat_field.local_normalise(frame, settings.PHOTO_VIEW_LOCAL_NORM_PX)
    return photo.normalise_frame(frame)


def undistorted_rays(camera):
    """Undistorted normalised image coordinates of every pixel centre of the 320 x 256 frame, (256 * 320, 2). Same for every query of a photo."""
    intrinsic = np.array([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]], float)
    dist = np.array([camera["k1"], camera["k2"], camera["p1"], camera["p2"], camera["k3"]])
    cols, rows = np.meshgrid(np.arange(camera["width"]), np.arange(camera["height"]))
    points = np.stack([cols, rows], -1).reshape(-1, 1, 2).astype(np.float64)
    return cv2.undistortPointsIter(points, intrinsic, dist, None, None, (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 50, 1e-10)).reshape(-1, 2)


def ground_grid(rays, pose, camera, heading=0.0, scale_factor=1.0):
    """G: (east, north) metres from the camera's nadir point to the ground point under every pixel, two (256, 320) arrays. heading and scale_factor are one query's, as in photo.draw_canvas."""
    geometry = settings.GEOMETRY
    pitch = roll = 0.0
    if geometry["attitude"]:
        pitch = geometry["pitch_sign"] * pose["pitch"]
        roll = geometry["roll_sign"] * pose["roll"]
    axis_x, axis_y, axis_z = photo.camera_axes(pose["yaw"] + geometry["yaw_offset"] + heading, pitch, roll)
    height = (pose["height"] + geometry["height_offset"]) * geometry["scale"] * scale_factor
    ray = rays[:, :1] * axis_x + rays[:, 1:] * axis_y + axis_z
    distance = -height / ray[:, 2]
    shape = (camera["height"], camera["width"])
    return (ray[:, 0] * distance).reshape(shape), (ray[:, 1] * distance).reshape(shape)


def draw_map(level_image, level, grid, x_m, y_m):
    """The map level drawn into the photo's view for a camera whose nadir point is (x_m, y_m) (map metres, y southwards). Returns grey 0..1 (0 where there is no map) and the share of pixels with map."""
    east, north = grid
    gsd = level_gsd(level)
    map_x = ((x_m + east) / gsd - 0.5).astype(np.float32)
    map_y = ((y_m - north) / gsd - 0.5).astype(np.float32)
    x0 = int(max(map_x.min() - 4, 0))
    x1 = int(min(map_x.max() + 5, level_image.shape[1]))
    y0 = int(max(map_y.min() - 4, 0))
    y1 = int(min(map_y.max() + 5, level_image.shape[0]))
    if x1 <= x0 + 8 or y1 <= y0 + 8:
        return None, 0.0
    window = np.asarray(level_image[y0:y1, x0:x1]).astype(np.float32) / 255
    drawn = cv2.remap(window, map_x - x0, map_y - y0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    has = cv2.remap((window > 0).astype(np.float32), map_x - x0, map_y - y0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.99
    return drawn * has, float(has.mean())


def sample(grid, points):
    """Bilinear value of a (256, 320) array at pixel positions (n, 2) as (x, y)."""
    return cv2.remap(grid.astype(np.float32), points[:, :1].astype(np.float32), points[:, 1:].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1)


class PhotoView:
    """Everything of one photo that the photo view needs; built once per photo, every query reuses it."""

    def __init__(self, image_path, pose, camera, level_images):
        self.pose = pose
        self.camera = camera
        self.level_images = level_images
        self.level = level_like_photo(pose["height"], camera, [lv for lv in level_images if lv in settings.LEVELS] or list(level_images))
        self.picture = own_photo(image_path, camera)
        self.rays = undistorted_rays(camera)
        self.tile_sigma = native_fine.tile_sigma(pose["height"] * settings.GEOMETRY["scale"] / camera["fx"], level_gsd(self.level))
        self.grids = {}

    def grid(self, query):
        key = (query["heading"], query["scale_factor"])
        if key not in self.grids:
            self.grids[key] = ground_grid(self.rays, self.pose, self.camera, query["heading"], query["scale_factor"])
        return self.grids[key]

    def redraw_and_match(self, model, match, x, y, corrections):
        """One more round of the loop: draw the map again with the pose corrected by `corrections` (dpitch, droll, dyaw, dlogh) for a camera at (x, y), match it with the photo.
        Returns a match dict for fit_pose (its base pose is the corrected one) or None when the picture has too little map or too few inliers."""
        geometry = settings.GEOMETRY
        dp, dr, dyaw, dlogh = corrections
        base = dict(self.pose)
        if geometry["attitude"]:
            base["pitch"] = self.pose["pitch"] + dp / geometry["pitch_sign"]
            base["roll"] = self.pose["roll"] + dr / geometry["roll_sign"]
        base["yaw"] = self.pose["yaw"] + dyaw
        base["height"] = (self.pose["height"] + geometry["height_offset"]) * math.exp(dlogh) - geometry["height_offset"]
        grid = ground_grid(self.rays, base, self.camera, match["heading"], match["scale_factor"])
        drawn, share = draw_map(self.level_images[self.level], self.level, grid, x, y)
        if drawn is None or share < settings.PHOTO_VIEW_MIN_MAP_SHARE:
            return None
        points_a, points_b, confidence = fine_matching.loftr_matches(model, self.picture, native_fine.clean_tile(drawn, self.tile_sigma))
        keep = confidence >= getattr(model, "min_confidence", settings.LOFTR_CONFIDENCE)
        matrix, picked = fine_matching.fit_similarity(points_a[keep], points_b[keep])
        if matrix is None or picked.sum() == 0:
            return None
        return dict(pa=points_a[keep][picked], pb=points_b[keep][picked], grid=grid, heading=match["heading"], scale_factor=match["scale_factor"], cand=(float(x), float(y)), pose=base, inliers=int(picked.sum()))

    def check_places(self, model, query, peaks, limit_m):
        """The fine stage for every candidate of one query: same return as fine_matching.check_places (inliers (M,), shifts (M, 2) east/south metres, aligned (M,))."""
        grid = self.grid(query)
        inliers = np.zeros(len(peaks))
        shifts = np.zeros((len(peaks), 2))
        aligned = np.full(len(peaks), -np.inf)
        keep_from = getattr(model, "min_confidence", settings.LOFTR_CONFIDENCE)
        kept = {}
        for k in range(len(peaks)):
            if peaks[k, 1] > 1e8:
                continue
            drawn, share = draw_map(self.level_images[self.level], self.level, grid, peaks[k, 1], peaks[k, 2])
            if drawn is None or share < settings.PHOTO_VIEW_MIN_MAP_SHARE:
                continue
            tile = native_fine.clean_tile(drawn, self.tile_sigma)
            points_a, points_b, confidence = fine_matching.loftr_matches(model, self.picture, tile)
            keep = confidence >= keep_from
            matrix, picked = fine_matching.fit_similarity(points_a[keep], points_b[keep])
            if matrix is None or picked.sum() == 0:
                continue
            a = points_a[keep][picked]
            b = points_b[keep][picked]
            east = float(np.median(sample(grid[0], b) - sample(grid[0], a)))
            north = float(np.median(sample(grid[1], b) - sample(grid[1], a)))
            shift = np.array([east, -north])
            if math.hypot(shift[0], shift[1]) <= limit_m:
                inliers[k] = int(picked.sum())
                shifts[k] = shift
                if settings.PHOTO_VIEW_FIT_POSE:
                    kept[k] = dict(pa=a, pb=b, grid=grid, heading=query["heading"], scale_factor=query["scale_factor"], cand=(float(peaks[k, 1]), float(peaks[k, 2])))
        return dict(inliers=inliers, shifts=shifts, aligned=aligned, matches=kept)


# ---------------------------------------------------------------- pose fit (iteration 44)
def ground_at(points, camera, pose, heading, scale_factor, dp, dr, dyaw, dlogh):
    """G at pixel positions (n, 2): (east, north) metres from the nadir point for the pose with the given corrections (degrees; dlogh = log of the height factor)."""
    geometry = settings.GEOMETRY
    intrinsic = np.array([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]], float)
    dist = np.array([camera["k1"], camera["k2"], camera["p1"], camera["p2"], camera["k3"]])
    und = cv2.undistortPointsIter(np.asarray(points, np.float64).reshape(-1, 1, 2), intrinsic, dist, None, None,
                                  (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 50, 1e-10)).reshape(-1, 2)
    pitch = roll = 0.0
    if geometry["attitude"]:
        pitch = geometry["pitch_sign"] * pose["pitch"] + dp
        roll = geometry["roll_sign"] * pose["roll"] + dr
    axis_x, axis_y, axis_z = photo.camera_axes(pose["yaw"] + geometry["yaw_offset"] + heading + dyaw, pitch, roll)
    height = (pose["height"] + geometry["height_offset"]) * geometry["scale"] * scale_factor * math.exp(dlogh)
    ray = und[:, :1] * axis_x + und[:, 1:] * axis_y + axis_z
    distance = -height / ray[:, 2]
    return ray[:, 0] * distance, ray[:, 1] * distance


def robust_least_squares(residual, start, steps, iterations=25, f_scale=2.0):
    """Levenberg-Marquardt with soft-L1 weights (IRLS) and a numerical Jacobian; numpy only (scipy is not a dependency of the pipeline)."""
    p = np.array(start, float)

    def weighted(q):
        r = residual(q)
        return r * (1 + (r / f_scale) ** 2) ** -0.25

    def cost(r):
        return float(np.sum(r * r))
    r = weighted(p)
    current = cost(r)
    damping = 1e-3
    for _ in range(iterations):
        jacobian = np.empty((len(r), len(p)))
        for j in range(len(p)):
            q = p.copy()
            q[j] += steps[j]
            jacobian[:, j] = (weighted(q) - r) / steps[j]
        gradient = jacobian.T @ r
        hessian = jacobian.T @ jacobian
        improved = False
        for _ in range(8):
            try:
                move = np.linalg.solve(hessian + damping * np.diag(np.diag(hessian) + 1e-9), -gradient)
            except np.linalg.LinAlgError:
                damping *= 10
                continue
            trial = weighted(p + move)
            if cost(trial) < current:
                p, r, current = p + move, trial, cost(trial)
                damping = max(damping / 3, 1e-9)
                improved = True
                break
            damping *= 4
        if not improved or np.max(np.abs(move / steps)) < 1e-2:
            break
    return p


def fit_pose(match, pose, camera, gsd_photo, prior_centre=None):
    """The camera position (map metres, y south) from one candidate's inlier matches, with the metadata's noise as a prior on pitch, roll, yaw and height.
    Every match gives the ground point the photo pixel sees (from the drawn map); the fit moves position, pitch, roll, yaw and height so that all pixels land on them.
    prior_centre: corrections already applied to the pose that was drawn (a later round of the loop): the prior stays centred on the metadata, not on the corrected pose.
    Returns (x, y, corrections, rms_px) with corrections = (dpitch, droll, dyaw degrees, log of the height factor) relative to `pose` and rms_px the match residual after the fit in photo pixels."""
    grid, pa, pb, heading, scale_factor = match["grid"], match["pa"], match["pb"], match["heading"], match["scale_factor"]
    cx, cy = match["cand"]
    e0, n0 = ground_at(pa, camera, pose, heading, scale_factor, 0, 0, 0, 0)
    mx = cx + e0 + (sample(grid[0], pb) - sample(grid[0], pa))
    my = cy - (n0 + (sample(grid[1], pb) - sample(grid[1], pa)))
    prior = np.array([settings.PHOTO_VIEW_PRIOR_TILT_DEG, settings.PHOTO_VIEW_PRIOR_TILT_DEG, settings.PHOTO_VIEW_PRIOR_YAW_DEG, settings.PHOTO_VIEW_PRIOR_HEIGHT])
    scale = 1.5 * gsd_photo
    centre = np.zeros(4) if prior_centre is None else np.asarray(prior_centre, float)

    def residual(p):
        e, n = ground_at(pa, camera, pose, heading, scale_factor, p[2], p[3], p[4], p[5])
        return np.concatenate([(p[0] + e - mx) / scale, (p[1] - n - my) / scale, (p[2:6] + centre) / prior])
    start = np.array([np.median(mx - e0), np.median(my + n0), 0.0, 0.0, 0.0, 0.0])
    fitted = robust_least_squares(residual, start, np.array([0.1, 0.1, 0.01, 0.01, 0.01, 1e-4]))
    e, n = ground_at(pa, camera, pose, heading, scale_factor, fitted[2], fitted[3], fitted[4], fitted[5])
    rms_px = float(np.sqrt(np.mean((fitted[0] + e - mx) ** 2 + (fitted[1] - n - my) ** 2)) / gsd_photo)
    return float(fitted[0]), float(fitted[1]), (float(fitted[2]), float(fitted[3]), float(fitted[4]), float(fitted[5])), rms_px

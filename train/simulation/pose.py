"""Where the drone is and where the camera points.

A pose is a plain dict, all in the map's own metric coordinates:
  east, north  camera position (metres, map CRS)
  height       camera height above the ground under it (metres)
  yaw          degrees, compass direction of the picture's up (0 = north, 90 = east)
  pitch, roll  degrees of tilt away from straight down (0, 0 = looking straight down); same meaning as image_processing.geometry.camera_axes
"""
import math

NEUTRAL_GEOMETRY = {"attitude": True, "pitch_sign": 1, "roll_sign": 1, "yaw_offset": 0.0, "scale": 1.0, "height_offset": 0.0}


def make_pose(east, north, height, yaw=0.0, pitch=0.0, roll=0.0):
    for name, value in (("east", east), ("north", north), ("height", height), ("yaw", yaw), ("pitch", pitch), ("roll", roll)):
        assert math.isfinite(value), "pose value %s must be finite, got %s" % (name, value)
    assert height > 0, "height must be above the ground (> 0 m)"
    assert abs(pitch) < 80 and abs(roll) < 80, "tilt of 80 degrees or more is not supported (the camera would see the horizon)"
    return dict(east=float(east), north=float(north), height=float(height), yaw=float(yaw), pitch=float(pitch), roll=float(roll))


def pose_from_item(item, satellite_map, geometry=None):
    """Pose from a pipeline photo item (lat, lon, rel_height, yaw, pitch, roll).

    geometry has the keys of settings.GEOMETRY. None means neutral: the numbers are taken as the truth.
    Pass settings.GEOMETRY to simulate what the pipeline believes about the real photos (yaw offset, scale, tilt signs).
    """
    setting = dict(NEUTRAL_GEOMETRY)
    if geometry is not None:
        setting.update(geometry)
    east, north = satellite_map.to_map_crs.transform(item["lon"], item["lat"])
    height = (item["rel_height"] + setting["height_offset"]) * setting["scale"]
    yaw = item["yaw"] + setting["yaw_offset"]
    pitch = 0.0
    roll = 0.0
    if setting["attitude"]:
        pitch = setting["pitch_sign"] * item["pitch"]
        roll = setting["roll_sign"] * item["roll"]
    return make_pose(east, north, height, yaw, pitch, roll)

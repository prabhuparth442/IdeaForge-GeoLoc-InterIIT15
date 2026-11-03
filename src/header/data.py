import numpy as np
import pandas as pd

from header import settings

CAMERA_COLUMNS = ["cam_fx", "cam_fy", "cam_cx", "cam_cy", "cam_k1", "cam_k2", "cam_k3", "cam_p1", "cam_p2"]


class Camera:
    def __init__(self, table):
        for column in CAMERA_COLUMNS:
            spread = table[column].max() - table[column].min()
            assert spread < 1e-6, "the camera numbers differ between photos: " + column
        self.fx, self.fy = table["cam_fx"][0], table["cam_fy"][0]
        self.cx, self.cy = table["cam_cx"][0], table["cam_cy"][0]
        self.k1, self.k2, self.k3 = table["cam_k1"][0], table["cam_k2"][0], table["cam_k3"][0]
        self.p1, self.p2 = table["cam_p1"][0], table["cam_p2"][0]
        self.width = 320
        self.height = 256


def wrap_angle(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0


def read_metadata():
    return pd.read_csv(settings.IMAGES_DIR / "metadata.csv")


def make_items(table):
    items = []
    for index in range(len(table)):
        row = table.iloc[index]
        item = {}
        item["name"] = row["filename"]
        item["lat"] = row["lat_gtruth"]
        item["lon"] = row["lon_gtruth"]
        item["rel_height"] = row["alt"]
        item["yaw"] = row["yaw"]
        item["pitch"] = wrap_angle(row["pitch"])
        item["roll"] = wrap_angle(row["roll"])
        items.append(item)
    return items


def assign_flights(items):
    heights = np.array([item["rel_height"] for item in items])
    flight_of_frame = np.array([settings.FLIGHT_NAMES[int(np.searchsorted(settings.BAND_EDGES, h, side="right")) - 1] for h in heights])
    for item, flight in zip(items, flight_of_frame):
        item["flight"] = flight
    return flight_of_frame


def find_map_file():
    for path in sorted(settings.MAP_DIR.iterdir()):
        if path.suffix == ".tif":
            return path
    raise FileNotFoundError("no .tif map in " + str(settings.MAP_DIR))

import math

import numpy as np

from header.scoring import sigma_m
from image_processing.map_levels import level_for_height


def truth_in_metres(item, satellite_map):
    row, col = satellite_map.gps_to_map_pixel(item["lat"], item["lon"])
    return np.array([col * satellite_map.gsd, row * satellite_map.gsd])


def random_prior(truth, rng, radius=5000):
    distance = radius * math.sqrt(rng.random())
    angle = 2 * math.pi * rng.random()
    return truth + distance * np.array([math.cos(angle), math.sin(angle)])


class TestSet:
    def __init__(self, items, camera, satellite_map, flight_of_frame, sample_ids):
        self.items = items
        self.camera = camera
        self.flight_of_frame = flight_of_frame
        self.sample_ids = sample_ids

        prior_rng = np.random.default_rng(0)
        self.truths = np.array([truth_in_metres(item, satellite_map) for item in items])
        self.priors = np.array([random_prior(truth, prior_rng) for truth in self.truths])
        self.sigmas = np.array([sigma_m(item["rel_height"], camera.fx) for item in items])
        self.query_levels = np.array([level_for_height(item["rel_height"], camera) for item in items])

        heights = np.array([item["rel_height"] for item in items])
        self.city = heights <= 200
        self.sample_city = self.city[sample_ids]
        self.sample_sigma = self.sigmas[sample_ids]

    def error_m(self, i, x, y):
        return math.hypot(x - self.truths[i][0], y - self.truths[i][1])

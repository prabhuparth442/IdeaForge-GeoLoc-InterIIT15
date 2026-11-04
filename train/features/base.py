"""A feature family turns a picture into a grid of 8-number cells (one cell = 4x4 pixels), for the map and for the query.
All families give (8, rows, cols) for the map and (8, 56, 56) + a valid-cell mask for the query, so the same FFT correlation
(coarse_search.correlation) scores all of them."""
from train.features.whitening import estimate_power_profile


class Family:
    """name, map_features(level) -> (8, R, C) numpy, query(canvas, valid) -> (8, s, s), (s, s) bool."""
    name = "?"

    def __init__(self):
        self._profiles = {}

    def profile(self, level):
        if level not in self._profiles:
            self._profiles[level] = estimate_power_profile(self.map_features(level))
        return self._profiles[level]

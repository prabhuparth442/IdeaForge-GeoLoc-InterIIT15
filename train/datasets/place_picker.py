"""Where to cut fake photos from: random places on the map, away from the photos we test on, spread over scene types."""
import numpy as np
from scipy.ndimage import binary_dilation


class PlacePicker:
    """Random places on the map: where there is map data, away from the photos we test on, spread evenly over scene types.

    Scene type = how much the grey values vary inside a 100 m cell (flat fields and water, mixed, dense city).
    """

    def __init__(self, satellite_map, avoid_points_m, cell_m=100.0, avoid_m=500.0):
        self.cell_m = cell_m
        self.satellite_map = satellite_map
        width_m = satellite_map.width * satellite_map.gsd
        height_m = satellite_map.height * satellite_map.gsd
        columns = int(width_m // cell_m)
        rows = int(height_m // cell_m)
        pixels_per_cell = 10
        overview = satellite_map.read_grey(out_shape=(rows * pixels_per_cell, columns * pixels_per_cell))
        blocks = overview.reshape(rows, pixels_per_cell, columns, pixels_per_cell)
        has_data = (blocks > 0).mean(axis=(1, 3))
        texture = blocks.std(axis=(1, 3))

        avoid = np.zeros((rows, columns), bool)
        for x_m, y_m in avoid_points_m:
            row, column = int(y_m // cell_m), int(x_m // cell_m)
            if 0 <= row < rows and 0 <= column < columns:
                avoid[row, column] = True
        avoid = binary_dilation(avoid, iterations=int(avoid_m / cell_m))
        data_around = binary_dilation(has_data < 0.95, iterations=2)
        allowed = ~avoid & ~data_around

        edges = np.quantile(texture[allowed], [1 / 3, 2 / 3])
        scene = np.digitize(texture, edges)
        self.cells_of_scene = []
        for kind in range(3):
            found_rows, found_columns = np.where(allowed & (scene == kind))
            self.cells_of_scene.append(np.stack([found_rows, found_columns], axis=1))
        self.allowed_km2 = allowed.sum() * cell_m ** 2 / 1e6
        self.avoided_km2 = avoid.sum() * cell_m ** 2 / 1e6

    def pick(self, rng):
        """Returns x (east, m), y (south, m) and a block name (1 km square) used to keep validation places apart from training places."""
        kind = int(rng.integers(0, 3))
        cell = self.cells_of_scene[kind][int(rng.integers(0, len(self.cells_of_scene[kind])))]
        y_m = (cell[0] + rng.random()) * self.cell_m
        x_m = (cell[1] + rng.random()) * self.cell_m
        return x_m, y_m, "b%d_%d" % (int(x_m // 1000), int(y_m // 1000))

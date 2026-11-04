"""The learned features (F4) used at search time: the trained nets turn the map and the query into cells like the edge family does."""
import numpy as np
import torch

from header.settings import CELL, DEVICE
from image_processing.edge_features import no_data_cells
from train.features.base import Family
from train.features.learned_net import cell_mask_of, features_from_net, make_net


class LearnedFamily(Family):
    name = "learned"

    def __init__(self, store, levels, checkpoint):
        super().__init__()
        blob = torch.load(checkpoint, map_location=DEVICE)
        self.query_net, self.map_net = make_net().to(DEVICE).eval(), make_net().to(DEVICE).eval()
        self.query_net.load_state_dict(blob["query"]); self.map_net.load_state_dict(blob["map"])
        self.hit, self.baseline = blob.get("hit", 0), blob.get("baseline", 0)
        self.features = {level: self._map_features(store["levels"][level]) for level in levels}

    @torch.no_grad()
    def _map_features(self, level_image, block=512, columns=2048, pad=16):
        """The map net over the level picture in blocks (block rows x columns columns, each with a pad of context), so GPU memory stays small.
        The net sees about 10 px around a cell, so a pad of 16 px makes every block give the same cells as one pass over the whole picture."""
        height, width = level_image.shape
        features = np.zeros((8, height // CELL, width // CELL), dtype=np.float32)
        for top in range(0, height, block):
            first_row, last_row = max(top - pad, 0), min(top + block + pad, height)
            for left in range(0, width, columns):
                first_col, last_col = max(left - pad, 0), min(left + columns + pad, width)
                piece = torch.from_numpy(level_image[first_row:last_row, first_col:last_col] / 255.0).float()[None, None].to(DEVICE)
                out = features_from_net(self.map_net, piece)[0].cpu().numpy()
                skip_rows = (top - first_row) // CELL
                skip_cols = (left - first_col) // CELL
                rows = min(block, height - top) // CELL
                cols = min(columns, width - left) // CELL
                features[:, top // CELL:top // CELL + rows, left // CELL:left // CELL + cols] = out[:, skip_rows:skip_rows + rows, skip_cols:skip_cols + cols]
        border = no_data_cells(level_image)[:features.shape[1], :features.shape[2]]
        features[:, border] = 0
        return features

    def map_features(self, level):
        return self.features[level]

    @torch.no_grad()
    def query(self, canvas, valid):
        picture = torch.from_numpy(np.clip(canvas, 0, 1).astype(np.float32))[None, None].to(DEVICE)
        cells = cell_mask_of(valid)
        features = features_from_net(self.query_net, picture)[0].cpu().numpy() * cells
        return features.astype(np.float32), cells

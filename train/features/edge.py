"""The existing edge-direction cells (built by image_processing.map_store; nothing new here)."""
from image_processing.edge_features import query_edge_features
from train.features.base import Family


class EdgeFamily(Family):
    name = "edge"

    def __init__(self, store):
        super().__init__()
        self.store = store

    def map_features(self, level):
        return self.store["features"][level]

    def query(self, canvas, valid):
        return query_edge_features(canvas, valid)

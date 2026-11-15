"""Block: a "rich" candidate dump. Same queries and answers as HeadingQueryPipeline, but every candidate also keeps its raw LoFTR matches, so any
fit, verifier or decision can be re-made offline (no GPU). Also keeps the photo's attitude and every query's seconds per stage.

RichDumpPipeline      HeadingQueryPipeline whose queries also carry `matches` (the dict of fine_matching.loftr_matcher.candidate_matches)
pack_photo            one photo's queries -> dict of numpy arrays for np.savez_compressed
load_rich             a saved photo file -> dict(truth, sigma, height, level, yaw, pitch, roll, queries=[...]) with matches decoded per candidate

Storage of the matches: points as uint16 (pixel * 64, so 1/64 px steps, up to 1023 px) and confidence as uint16 (* 65535); one flat array per query
with an offset per candidate. Matches below `min_keep_confidence` are dropped to save disk (the pipeline cuts at 0.22).
No plotting here.
"""
import cv2
import numpy as np

from fine_matching.loftr_matcher import candidate_matches, stage2_from_matches
from header.settings import SEED
from heading_query_pipeline import HeadingQueryPipeline

POINT_STEP = 64.0
MIN_KEEP_CONFIDENCE = 0.15


class RichDumpPipeline(HeadingQueryPipeline):
    """Only for the LoFTR matcher. candidates_for_query returns the usual keys plus `matches` (raw matches of every candidate)."""

    def candidates_for_query(self, item, prior_m, level, geometry=None):
        assert hasattr(self.matcher, "loftr_model"), "the rich dump needs the LoFTR matcher"
        found = self.prepare_query(item, prior_m, level, geometry)
        clock = self._now()
        stored = candidate_matches(self.matcher, found["canvas"], found["valid"], level, found["peaks"], self.level_images)
        cv2.setRNGSeed(SEED)
        stage2 = stage2_from_matches(stored, level, self.matcher.min_confidence)
        self._lap(found["seconds"], "fine", clock)
        return dict(peaks=found["peaks"], z=found["z"], stage2=stage2, seconds=found["seconds"], valid_fraction=found["valid_fraction"],
                    shortlist=found["shortlist"], matches=stored)


def _pack_matches(stored, min_keep_confidence):
    rows, offsets = [], [0]
    for entry in stored["items"]:
        if entry is not None:
            keep = entry["confidence"] >= min_keep_confidence
            a, b, c = entry["points_a"][keep], entry["points_b"][keep], entry["confidence"][keep]
            assert (a >= 0).all() and (b >= 0).all() and a.max(initial=0) < 1023 and b.max(initial=0) < 1023, "points outside 0..1023 px"
            block = np.zeros((len(a), 5), np.uint16)
            block[:, 0:2] = np.round(a * POINT_STEP)
            block[:, 2:4] = np.round(b * POINT_STEP)
            block[:, 4] = np.round(np.clip(c, 0, 1) * 65535)
            rows.append(block)
            offsets.append(offsets[-1] + len(a))
        else:
            offsets.append(offsets[-1])
    flat = np.concatenate(rows) if rows else np.zeros((0, 5), np.uint16)
    return flat, np.asarray(offsets, np.int32)


def pack_photo(item, truth, sigma, level, queries, min_keep_confidence=MIN_KEEP_CONFIDENCE):
    """Arrays for one photo. queries: the list run_queries returned (each with peaks, z, stage2, seconds, heading, scale_factor, matches)."""
    arrays = dict(truth=np.asarray(truth, float), sigma=float(sigma), height=float(item["rel_height"]), level=int(level),
                  yaw=float(item["yaw"]), pitch=float(item["pitch"]), roll=float(item["roll"]), n_queries=len(queries))
    for q, query in enumerate(queries):
        p = "q%d_" % q
        arrays[p + "peaks"] = np.asarray(query["peaks"], float)
        arrays[p + "z"] = np.asarray(query["z"], float)
        arrays[p + "inliers"] = np.asarray(query["stage2"]["inliers"], float)
        arrays[p + "shifts"] = np.asarray(query["stage2"]["shifts"], float)
        arrays[p + "heading"] = float(query["heading"])
        arrays[p + "scale"] = float(query["scale_factor"])
        arrays[p + "valid_fraction"] = float(query["valid_fraction"])
        labels = sorted(query["seconds"])
        arrays[p + "seconds_labels"] = np.asarray(labels)
        arrays[p + "seconds"] = np.asarray([query["seconds"][k] for k in labels], float)
        stored = query["matches"]
        flat, offsets = _pack_matches(stored, min_keep_confidence)
        arrays[p + "matches"] = flat
        arrays[p + "match_offsets"] = offsets
        arrays[p + "inside"] = np.packbits(stored["inside"])
        arrays[p + "inside_shape"] = np.asarray(stored["inside"].shape)
        arrays[p + "margin"] = int(stored["margin"])
        arrays[p + "border_px"] = int(stored["border_px"])
    return arrays


def load_rich(path, with_matches=True):
    """Decoded photo: dict(truth, sigma, height, level, yaw, pitch, roll, queries); each query has peaks, z, inliers, shifts, heading, scale,
    valid_fraction, seconds (dict), inside (bool mask), margin, border_px and matches (list per candidate: dict(points_a, points_b, confidence) or None
    when the candidate kept no match). with_matches=False skips decoding the matches (all None): much faster for decision-only work."""
    data = np.load(path)
    out = dict(truth=data["truth"], sigma=float(data["sigma"]), height=float(data["height"]), level=int(data["level"]),
               yaw=float(data["yaw"]), pitch=float(data["pitch"]), roll=float(data["roll"]), queries=[])
    for q in range(int(data["n_queries"])):
        p = "q%d_" % q
        shape = tuple(data[p + "inside_shape"])
        inside = np.unpackbits(data[p + "inside"])[:shape[0] * shape[1]].reshape(shape).astype(bool)
        flat, offsets = (data[p + "matches"], data[p + "match_offsets"]) if with_matches else (None, [])
        matches = [] if with_matches else [None] * len(data[p + "inliers"])
        for k in range(len(offsets) - 1):
            block = flat[offsets[k]:offsets[k + 1]]
            if len(block) == 0:
                matches.append(None)
                continue
            matches.append(dict(points_a=block[:, 0:2].astype(np.float32) / POINT_STEP, points_b=block[:, 2:4].astype(np.float32) / POINT_STEP,
                                confidence=block[:, 4].astype(np.float64) / 65535.0))
        out["queries"].append(dict(peaks=data[p + "peaks"], z=data[p + "z"], inliers=data[p + "inliers"], shifts=data[p + "shifts"],
                                   heading=float(data[p + "heading"]), scale=float(data[p + "scale"]), valid_fraction=float(data[p + "valid_fraction"]),
                                   seconds=dict(zip([str(s) for s in data[p + "seconds_labels"]], data[p + "seconds"].tolist())),
                                   inside=inside, margin=int(data[p + "margin"]), border_px=int(data[p + "border_px"]), matches=matches))
    return out

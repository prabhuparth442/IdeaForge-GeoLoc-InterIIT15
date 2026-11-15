"""CPU equality and timing check on six synthetic 224px queries over a 5.5km disc."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from fast_edge_codex import edge_inputs
from train.features.edge import EdgeFamily
from train.coarse_methods.inputs import make_inputs
from train.coarse_methods.single_map import make_single_map_method
from semantic_rerank.dino_rerank import shortlist_peaks


def main():
    rng = np.random.default_rng(13)
    family = EdgeFamily(dict(features={7: rng.uniform(0, 1, (8, 1024, 1024)).astype(np.float32)}))
    valid = np.zeros((224, 224), bool)
    valid[20:204, 30:194] = True
    pictures = rng.random((6, 224, 224)).astype(np.float32)
    method = make_single_map_method("edge_zncc")
    records = []
    for repeat in range(4):
        cache = {}
        for index, picture in enumerate(pictures):
            start = time.perf_counter()
            original = make_inputs(picture, valid, 7, [6000, 6000], {"edge": family}, ["edge_zncc"])
            old_seconds = time.perf_counter() - start
            start = time.perf_counter()
            fast = edge_inputs(picture, valid, 7, [6000, 6000], family, cache)
            new_seconds = time.perf_counter() - start
            old_score, new_score = method(original), method(fast)
            assert torch.equal(old_score, new_score), "Score maps differ"
            for field in ("x_m", "y_m", "inside"):
                assert torch.equal(original["geometry"][field], fast["geometry"][field]), field
            old_peaks = shortlist_peaks(old_score, original["geometry"], 50)
            new_peaks = shortlist_peaks(new_score, fast["geometry"], 50)
            np.testing.assert_array_equal(old_peaks, new_peaks)
            records.append(dict(repeat=repeat, query=index, original_s=old_seconds, optimized_s=new_seconds,
                                maps_exact=True, peaks_exact=True))
    import pandas as pd
    table = pd.DataFrame(records)
    table.to_csv(HERE / "results/edge_benchmark_codex.csv", index=False)
    measured = table[table.repeat > 0].groupby("repeat")[["original_s", "optimized_s"]].sum().median()
    summary = dict(device="cpu", workload="six synthetic queries, level7, 5500m search radius, 2 threads",
                   original_s=float(measured.original_s), optimized_s=float(measured.optimized_s),
                   speedup=float(measured.original_s / measured.optimized_s), comparisons=len(table), exact=True,
                   caveat="coarse input preparation only, not end-to-end or Jetson timing")
    (HERE / "results/edge_benchmark_codex.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

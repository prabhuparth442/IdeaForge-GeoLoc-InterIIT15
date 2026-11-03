import numpy as np

from header.settings import SEED

from result_thresholding.pipeline import localise_photo


def check_independence(testset, store, matcher, ids):
    assert len(ids) >= 2, "need at least 2 photos to test independence"
    run = lambda j: localise_photo(testset.items[j], testset.priors[j], testset.camera, store, matcher)
    forward = {j: run(j) for j in ids}
    backward = {j: run(j) for j in reversed(ids)}
    alone = run(ids[-1])
    for j in ids:
        a, b = forward[j], backward[j]
        same = (abs(a["x"] - b["x"]) < 1e-3 and abs(a["y"] - b["y"]) < 1e-3 and a["inliers"] == b["inliers"]
                and np.allclose(a["peaks"], b["peaks"], atol=1e-4))
        print("photo %3d: forward and reverse order give the same answer: %s (error %.0f m, %d inliers)" % (
            j, same, testset.error_m(j, a["x"], a["y"]), a["inliers"]))
        assert same, "photo %d: the answer depends on what was processed before it!" % j
    assert abs(alone["x"] - forward[ids[-1]]["x"]) < 1e-3 and alone["inliers"] == forward[ids[-1]]["inliers"]
    print("ok: every photo gives the same answer whatever was processed before it. Median time per photo: %.2f s" % np.median([r["seconds"] for r in forward.values()]))


def check_replay(testset, store, matcher, final):
    rng = np.random.default_rng(SEED + 5)
    ids = [int(j) for j in rng.choice(testset.sample_ids, size=min(4, len(testset.sample_ids)), replace=False)]
    for j in ids:
        r = localise_photo(testset.items[j], testset.priors[j], testset.camera, store, matcher)
        assert abs(r["x"] - final[j][0]) < 1e-3 and abs(r["y"] - final[j][1]) < 1e-3, "photo %d: batch run and single run disagree" % j
    print("serial replay ok: running photos %s one by one reproduces the batch results" % ids)

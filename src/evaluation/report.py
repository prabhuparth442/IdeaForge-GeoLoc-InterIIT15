import numpy as np

from evaluation.metrics import auc, best_gate, gated_curve, honest_gate
from header.scoring import final_score, points_for_error
from header.settings import MIN_INLIERS, NUM_PEAKS
from result_thresholding.decision import final_answer


def print_table(header, rows):
    widths = [max(len(str(row[k])) for row in [header] + rows) for k in range(len(header))]
    for row in [header] + rows:
        print("  ".join(str(cell).ljust(widths[k]) for k, cell in enumerate(row)))
    print()


def collect_answers(testset, disc, stage2, final, use_minima):
    answers = {"correlation": []}
    if use_minima:
        answers["+MINIMA re-rank"] = []
        answers["+MINIMA re-rank + refine"] = []
    closest, ratios, inlier_counts = [], [], []
    for i in testset.sample_ids:
        i = int(i)
        peaks = disc[i]["peaks"][("zncc", 0.0)]
        answers["correlation"].append((peaks[0, 1], peaks[0, 2]))
        if use_minima:
            order = np.lexsort((-peaks[:, 0], -stage2[i]["inliers"]))
            best = int(order[0])
            if stage2[i]["inliers"][best] < MIN_INLIERS:
                best = 0
            answers["+MINIMA re-rank"].append((peaks[best, 1], peaks[best, 2]))
            answers["+MINIMA re-rank + refine"].append((final[i][0], final[i][1]))
            inlier_counts.append(final[i][2])
        closest.append(min(testset.error_m(i, p[1], p[2]) for p in peaks if p[1] < 1e8))
        ratios.append(peaks[0, 0] / max(peaks[1, 0], 1e-6))

    results = {}
    for name, xy in answers.items():
        errors = np.array([testset.error_m(int(i), x, y) for (x, y), i in zip(xy, testset.sample_ids)])
        points = [points_for_error(e, testset.items[int(i)]["rel_height"], testset.camera.fx) for e, i in zip(errors, testset.sample_ids)]
        results[name] = dict(errors=errors, points=np.array(points))
    return results, np.array(closest), np.array(ratios), np.array(inlier_counts)


def print_results(testset, results, closest, ratios, inlier_counts, use_minima):
    sigma = testset.sample_sigma
    city = testset.sample_city

    def three(flags):
        return "%4.2f %4.2f %4.2f" % (flags.mean(), flags[city].mean(), flags[~city].mean())

    rows = []
    for name, r in results.items():
        rows.append([name, three(r["errors"] / sigma <= 1), three(r["errors"] / sigma <= 4), "%.1f" % final_score(list(r["points"]))])
    print_table(["answer", "<=1 sigma a/l/h", "<=4 sigma a/l/h", "always"], rows)

    ceiling = closest / sigma <= 8
    print("any of the %d candidates <= 8 sigma (ceiling): %s" % (NUM_PEAKS, three(ceiling)))
    best_name = list(results)[-1]
    good = results[best_name]["errors"] / sigma <= 4
    print("AUC of the correlation ratio: %.2f" % auc(ratios, results["correlation"]["errors"] / sigma <= 4))
    if use_minima:
        print("AUC of the MINIMA inlier count: %.2f" % auc(inlier_counts, good))
    print()
    return good


def print_gated(results, ratios, inlier_counts, use_minima):
    best_name = list(results)[-1]
    curves = [("correlation ratio", ratios, results["correlation"]["points"])]
    if use_minima:
        curves.append(("MINIMA inliers (final answer)", inlier_counts + 0.01 * ratios, results[best_name]["points"]))
    for label, signal, points in curves:
        shares, scores = gated_curve(signal, points)
        k = int(np.argmax(scores))
        print("%-30s best: %.1f / 60 answering %.0f%% of the photos (answer everything: %.1f)" % (label, scores[k], 100 * shares[k], final_score(list(points))))
    print()


def print_final_scores(testset, results, closest, ratios, inlier_counts, stage2, disc, use_minima):
    names = list(results)
    sigma = testset.sample_sigma
    low = testset.sample_city
    high = ~low
    best_name = names[-1]
    signals = {n: (ratios if n == "correlation" else inlier_counts + 0.01 * ratios) for n in names}

    def score_of(points, mask=None):
        points = np.asarray(points)
        return final_score(list(points if mask is None else points[mask]))

    labels = ["within 4 sigma (all)", "  low photos (<= 200 m)", "  high photos", "within 1 sigma (+10)", "within 2 sigma (+6 or better)",
              "worse than 8 sigma (-20)", "median error (sigma)", "score, every photo answered (of 60)", "  low photos only", "  high photos only",
              "score, confident only (optimistic)", "score, confident only (honest, half/half)"]
    columns = {}
    for n in names:
        err = results[n]["errors"] / sigma
        pts = results[n]["points"]
        within4 = err <= 4
        best_s, _, best_share = best_gate(signals[n], pts)
        h_mean, h_std, h_share = honest_gate(signals[n], pts)
        columns[n] = ["%.0f%%" % (100 * within4.mean()), "%.0f%%" % (100 * within4[low].mean()), "%.0f%%" % (100 * within4[high].mean()),
                      "%.0f%%" % (100 * (err <= 1).mean()), "%.0f%%" % (100 * (err <= 2).mean()), "%.0f%%" % (100 * (err > 8).mean()),
                      "%.1f" % np.median(err), "%.1f" % score_of(pts), "%.1f" % score_of(pts, low), "%.1f" % score_of(pts, high),
                      "%.1f (%.0f%% answered)" % (best_s, 100 * best_share), "%.1f +- %.1f (%.0f%% answered)" % (h_mean, h_std, 100 * h_share)]
    print("Final scores on %d photos (%d low, %d high)" % (len(testset.sample_ids), low.sum(), high.sum()))
    print_table([""] + names, [[label] + [columns[n][k] for n in names] for k, label in enumerate(labels)])

    tiers = [(10, "<= 1 sigma"), (6, "<= 2 sigma"), (3, "<= 4 sigma"), (0, "<= 8 sigma"), (-20, "> 8 sigma")]
    rows = [["%+d (%s)" % (value, text)] + [str(int((results[n]["points"] == value).sum())) for n in names] for value, text in tiers]
    print("How the points are made up (number of photos)")
    print_table(["points per photo"] + names, rows)

    if not use_minima:
        return

    rows = []
    for K in sorted(set([1, 3, 5, NUM_PEAKS])):
        xs, ys, near = [], [], []
        for i in testset.sample_ids:
            i = int(i)
            peaks = disc[i]["peaks"][("zncc", 0.0)][:K]
            x, y, _, _ = final_answer(peaks, dict(inliers=stage2[i]["inliers"][:K], shifts=stage2[i]["shifts"][:K]))
            xs.append(x)
            ys.append(y)
            near.append(min(testset.error_m(i, p[1], p[2]) for p in peaks if p[1] < 1e8) / testset.sigmas[i])
        err = np.array([testset.error_m(int(i), x, y) for x, y, i in zip(xs, ys, testset.sample_ids)])
        pts = [points_for_error(e, testset.items[int(i)]["rel_height"], testset.camera.fx) for e, i in zip(err, testset.sample_ids)]
        rows.append([K, "%.0f%%" % (100 * (err / sigma <= 4).mean()), "%.1f" % final_score(pts), "%.0f%%" % (100 * (np.array(near) <= 8).mean())])
    print("How many candidates to check (top K of the correlation peaks)")
    print_table(["K", "within 4 sigma", "score, all answered", "ceiling (any of K within 8 sigma)"], rows)

    pts = results[best_name]["points"]
    rows = []
    for T in (0, 8, 12, 16, 20, 25, 30, 40):
        answered = inlier_counts >= T
        rows.append([T, "%.0f%%" % (100 * answered.mean()), int((answered & (pts == -20)).sum()), "%.1f" % final_score(list(np.where(answered, pts, 0)))])
    print("Fixed per-photo answer rule: answer if inliers >= T")
    print_table(["T", "photos answered", "wrong answers (-20)", "score of 60"], rows)

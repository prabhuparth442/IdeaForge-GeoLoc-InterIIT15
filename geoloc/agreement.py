"""Two optional decision helpers, both off by default (settings.USE_AGREEMENT, settings.USE_INFLATION_GUARD).

Agreement: a photo with too few inliers at the winner is still answered when several of the standard queries, each on its own, found the same place.
Two queries rarely make the same chance hit, so this is safer than lowering the bar.

Inflation guard: if the camera or the map gives every candidate more inliers than the development photos did, false places start to reach the bar.
The guard measures how many inliers a wrong place gets (the mean over the candidates that are more than FAR_SIGMA sigma from the answer), keeps a running
mean over the photos of the run and, when that is more than GUARD_DEAD_ZONE times the development value, multiplies the bars by the ratio.
It needs one process for all photos (the running mean lives in the Localiser); with one process per photo it never leaves the dead zone and does nothing.

Measured offline on stored candidates of four photo sets (tests/Test_Iteration_7/OURS_CHECK.md in the research repo): agreement + escalation 49.0 -> 53.5 on 379
official photos, 0 wrong; guard: no change on clean data, with every inlier count multiplied by 1.5 to 2 the wrong answers fell from 2-11 to 2-4.
"""
import math

import settings

FAR_SIGMA = 4.0


def best_of_query(result):
    """Position (metres) and inliers of the query's own best candidate, or None when no candidate has an inlier."""
    inliers = result["stage2"]["inliers"]
    best = -1
    for k in range(len(inliers)):
        if inliers[k] > 0 and (best < 0 or inliers[k] > inliers[best]):
            best = k
    if best < 0:
        return None
    x = result["peaks"][best][1] + result["stage2"]["shifts"][best][0]
    y = result["peaks"][best][2] + result["stage2"]["shifts"][best][1]
    return x, y, float(inliers[best])


def agreeing_queries(results, x, y, radius_m, floor):
    """How many queries found a best candidate with at least floor inliers within radius_m of (x, y)."""
    count = 0
    for result in results:
        found = best_of_query(result)
        if found is None:
            continue
        if found[2] >= floor and math.hypot(found[0] - x, found[1] - y) <= radius_m:
            count += 1
    return count


def far_level(results, x, y, sigma):
    """Mean inliers of the candidates more than FAR_SIGMA sigma from (x, y); 0.0 when there is none."""
    total = 0.0
    count = 0
    for result in results:
        inliers = result["stage2"]["inliers"]
        for k in range(len(inliers)):
            candidate_x = result["peaks"][k][1] + result["stage2"]["shifts"][k][0]
            candidate_y = result["peaks"][k][2] + result["stage2"]["shifts"][k][1]
            if math.hypot(candidate_x - x, candidate_y - y) > FAR_SIGMA * sigma:
                total += float(inliers[k])
                count += 1
    if count == 0:
        return 0.0
    return total / count


class InflationGuard:
    def __init__(self):
        self.level_sum = 0.0
        self.photo_count = 0

    def factor_after(self, photo_level):
        """Adds one photo to the running mean and returns the factor for the bars: 1.0 inside the dead zone, else the ratio to the prior."""
        self.level_sum += photo_level
        self.photo_count += 1
        weight = settings.GUARD_PRIOR_WEIGHT
        running = (weight * settings.GUARD_PRIOR + self.level_sum) / (weight + self.photo_count)
        ratio = running / settings.GUARD_PRIOR
        if ratio > settings.GUARD_DEAD_ZONE:
            return ratio
        return 1.0

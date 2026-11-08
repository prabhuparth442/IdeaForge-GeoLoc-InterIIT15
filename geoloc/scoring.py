"""The official scoring, for checking results when the true position is known (MODE "full" on the development set)."""
import math


def distance_m(lat1, lon1, lat2, lon2):
    # haversine on a sphere, good to a fraction of a percent at these distances
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def sigma_m(altitude_m, focal_px):
    height = max(altitude_m, 100.0)
    gsd = height / focal_px
    return math.sqrt(8.0 ** 2 + (3.0 * gsd) ** 2)


def points_for_error(error_m, altitude_m, focal_px):
    ratio = error_m / sigma_m(altitude_m, focal_px)
    if ratio <= 1:
        return 10
    if ratio <= 2:
        return 6
    if ratio <= 4:
        return 3
    if ratio <= 8:
        return 0
    return -20


def final_score(points_list):
    """Out of 60. A no_match counts as 0 points, so it has to be in the list."""
    accuracy = sum(points_list) / (10 * len(points_list))
    return 60 * max(accuracy, 0)

import math


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
    accuracy = sum(points_list) / (10 * len(points_list))
    return 60 * max(accuracy, 0)


def check_scoring():
    print("sigma at 1000 m with a 200-pixel focal length:", round(sigma_m(1000, 200), 1), "m (statement: about 17 m)")
    example = [10] * 40 + [6] * 20 + [-20] * 2 + [0] * 38
    print("statement's example score:", round(final_score(example), 1), "(statement: 28.8)")

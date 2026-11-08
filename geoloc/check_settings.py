"""Stops a run early when a value in settings.py is outside the range written next to it."""
import settings


def need(condition, message):
    if not condition:
        raise ValueError("settings.py: " + message)


def check_settings():
    need(settings.MODE in ("answer", "full"), "MODE must be 'answer' or 'full', got %r" % (settings.MODE,))
    need(0.25 <= settings.MAP_GSD <= 0.4, "MAP_GSD must be 0.25 to 0.4")
    need(len(settings.LEVELS) > 0 and all(isinstance(level, int) and 0 <= level <= 10 for level in settings.LEVELS), "LEVELS must be whole numbers 0 to 10")
    need(settings.TILE % 56 == 0, "TILE must be a multiple of 56")
    need(settings.QUERY_SET in ("standard", "all", "custom"), "QUERY_SET must be 'standard', 'all' or 'custom'")
    need(0.0 in settings.HEADINGS, "HEADINGS must contain 0.0")
    need(all(-15 <= heading <= 15 for heading in settings.HEADINGS), "HEADINGS must be between -15 and 15 degrees")
    need(1.0 in settings.SCALE_FACTORS, "SCALE_FACTORS must contain 1.0")
    need(all(0.5 <= factor <= 1.5 for factor in settings.SCALE_FACTORS), "SCALE_FACTORS must be between 0.5 and 1.5")
    need(5000 <= settings.DISC_RADIUS <= 7000, "DISC_RADIUS must be 5000 to 7000 metres")
    need(1 <= settings.M_FINE <= settings.K_PEAKS <= 200, "need 1 <= M_FINE <= K_PEAKS <= 200")
    need(0 <= settings.DINO_WEIGHT <= 100, "DINO_WEIGHT must be 0 to 100")
    need(0.2 <= settings.LOFTR_CONFIDENCE < 1, "LOFTR_CONFIDENCE must be 0.2 to below 1")
    need(0 < settings.LOFTR_COARSE_THRESHOLD < 1, "LOFTR_COARSE_THRESHOLD must be between 0 and 1")
    need(0 < settings.RANSAC_PIXELS <= 8, "RANSAC_PIXELS must be above 0 and at most 8")
    need(1 <= settings.MAX_SHIFT_CELLS <= 6, "MAX_SHIFT_CELLS must be 1 to 6")
    need(settings.GAMMA >= 0, "GAMMA must be 0 or more")
    need(settings.MIN_INLIERS >= 1, "MIN_INLIERS must be 1 or more")
    for key in ("attitude", "pitch_sign", "roll_sign", "yaw_offset", "scale", "height_offset"):
        need(key in settings.GEOMETRY, "GEOMETRY has no key %r" % key)

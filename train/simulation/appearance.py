"""What the ground looks like in thermal. One contract: appearance(grey, rng, strength, pixel_m) -> float picture of the same shape.

grey is the satellite crop in 0..1 (north-up), pixel_m the ground size of one crop pixel in metres. All sizes inside a look are in
metres, so the same look works at 0.4 m or at 2 m per pixel. The result is NOT a temperature: it is one possible thermal-looking
appearance of the same scene, so every output must be treated as "a possible look".
To add a method, write a function with this contract and give it to render_view(appearance=...).
"""
import cv2
import numpy as np

from train.datasets.pseudo_thermal import blur, local_contrast, make_pseudo_thermal

APPEARANCE_GSD = 1.0             # metres per pixel the old pseudo-thermal generator was written for


def plain_invert(grey, rng, strength=1.0, pixel_m=1.0):
    """Cheapest rule: random polarity, random gamma, a little contrast change. Keeps every map pattern exactly (too easy to match)."""
    gamma = rng.uniform(0.6, 1.6)
    picture = np.power(np.clip(grey, 0, 1), gamma)
    if rng.random() < 0.5:
        picture = 1.0 - picture
    return (picture * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1)).astype(np.float32)


def rule_based(grey, rng, strength=1.0, pixel_m=1.0):
    """The project's pseudo-thermal generator (train/datasets/pseudo_thermal.py), run at 1 m per pixel as it was written.
    Smooth blobs: it throws away most fine detail, so it is a coarse-search look, not a close-up look."""
    if pixel_m >= APPEARANCE_GSD:
        return make_pseudo_thermal(grey, rng, strength)
    height, width = grey.shape
    shrink = pixel_m / APPEARANCE_GSD
    small = cv2.resize(grey, (max(2, int(width * shrink)), max(2, int(height * shrink))), interpolation=cv2.INTER_AREA)
    result = make_pseudo_thermal(small, rng, strength)
    return cv2.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)


def pixels(metres, pixel_m):
    return max(0.5, metres / pixel_m)


def region_temperature(picture, pixel_m, rng):
    """One random temperature per region (a grey band that is flat or textured), smoothed over 1.5 m. Sizes in metres."""
    base = blur(picture, pixels(4.0, pixel_m))
    texture = blur(local_contrast(picture, pixels(3.0, pixel_m)), pixels(5.0, pixel_m))
    band_count = int(rng.integers(3, 6))
    cut_levels = np.sort(rng.uniform(0.1, 0.9, band_count - 1))
    band = np.digitize(base, np.quantile(base, cut_levels))
    textured = texture > np.quantile(texture, rng.uniform(0.4, 0.7))
    region = band * 2 + textured.astype(int)
    temperature_of_region = rng.uniform(0, 1, 2 * band_count)
    return blur(temperature_of_region[region].astype(np.float32), pixels(1.5, pixel_m))


def add_spots_and_patches(thermal, pixel_m, rng, strength):
    """Hot spots (vehicles, vents; 1-6 m) and changed patches (new roofs, 3-15 m); counts grow with the area seen."""
    height, width = thermal.shape
    area_m2 = height * width * pixel_m * pixel_m
    spots = np.zeros_like(thermal)
    for _ in range(int(rng.poisson(strength * area_m2 / 30000.0))):
        centre = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        radius = max(1, int(rng.uniform(1.0, 6.0) / pixel_m))
        cv2.circle(spots, centre, radius, float(rng.uniform(-0.4, 0.4)), -1)
    thermal = thermal + blur(spots, pixels(0.7, pixel_m))
    for _ in range(int(rng.poisson(strength * area_m2 / 50000.0))):
        patch_height = max(2, int(rng.uniform(3.0, 15.0) / pixel_m))
        patch_width = max(2, int(rng.uniform(3.0, 15.0) / pixel_m))
        if patch_height >= height or patch_width >= width:
            continue
        top = int(rng.integers(0, height - patch_height))
        left = int(rng.integers(0, width - patch_width))
        thermal[top:top + patch_height, left:left + patch_width] = rng.uniform(0, 1)
    return thermal


def heavy_tail(thermal, power):
    """Squash the cool end so most ground is dark and a few hot things stand out (power 1 = unchanged)."""
    low, high = np.percentile(thermal, [1, 99])
    unit = np.clip((thermal - low) / max(high - low, 1e-6), 0.0, 1.5)
    return (unit ** power).astype(np.float32)


def regional_detail(grey, rng, strength=1.0, pixel_m=1.0, edge_keep=0.5, tail=1.0):
    """Region temperatures + the map's own structure with random polarity + sharpened edges + spots, patches and slow drift.

    edge_keep (0..1) is the main knob for how hard the result is to match with the map: 1 keeps every map edge (as easy as
    plain_invert), 0 keeps only the region blobs (like rule_based). Real thermal photos sit well below 1 (see the notebook sweep).
    tail (>1): heavy-tailed grey values, mostly dark with a few bright things, like real frames.
    strength (0..2): more regional temperature, more spots and patches.
    """
    picture = blur(grey, pixels(0.6, pixel_m))
    temperature = region_temperature(picture, pixel_m, rng)
    body = blur(picture, pixels(1.2, pixel_m))
    fine = picture - blur(picture, pixels(2.0, pixel_m))
    polarity = rng.choice([-1.0, 1.0])
    if polarity > 0:
        structure = body
    else:
        structure = 1.0 - body
    region_share = float(np.clip(0.2 + 0.15 * strength + rng.uniform(-0.05, 0.05), 0.1, 0.6))
    detail_gain = rng.uniform(1.2, 2.5) / (1.0 + 0.3 * strength)
    structure = edge_keep * structure + (1.0 - edge_keep) * temperature
    thermal = region_share * temperature + (1 - region_share) * structure + edge_keep * polarity * detail_gain * fine
    thermal = add_spots_and_patches(thermal.astype(np.float32), pixel_m, rng, strength)

    drift = rng.uniform(0.85, 1.15, (4, 4)).astype(np.float32)
    drift = cv2.resize(drift, (thermal.shape[1], thermal.shape[0]), interpolation=cv2.INTER_CUBIC)
    thermal = (thermal * drift + rng.uniform(-0.1, 0.1)).astype(np.float32)
    if tail != 1.0:
        thermal = heavy_tail(thermal, tail)
    return thermal


METHODS = {"regional_detail": regional_detail, "rule_based": rule_based, "plain_invert": plain_invert}


def regional_detail_with(edge_keep, tail=1.0):
    """regional_detail with a fixed edge_keep, as a function with the plain appearance contract (and a readable name)."""
    def look(grey, rng, strength=1.0, pixel_m=1.0):
        return regional_detail(grey, rng, strength, pixel_m, edge_keep, tail)
    look.__name__ = "regional_detail_keep%.2f" % edge_keep + ("" if tail == 1.0 else "_tail%.1f" % tail)
    return look

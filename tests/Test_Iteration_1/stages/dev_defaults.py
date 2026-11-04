"""The development-set settings, saved at import time, before the NUDT setup changes them. Import this module early."""
from header import settings

DEV_DEFAULTS = dict(levels=list(settings.LEVELS), geometry=dict(settings.GEOMETRY), images=settings.IMAGES_DIR, map_dir=settings.MAP_DIR)


def apply_dev_settings(dev_dir):
    settings.LEVELS = list(DEV_DEFAULTS["levels"])
    settings.GEOMETRY.clear()
    settings.GEOMETRY.update(DEV_DEFAULTS["geometry"])
    settings.IMAGES_DIR = dev_dir / "d1" / "images"
    settings.MAP_DIR = dev_dir / "small_sat_area"

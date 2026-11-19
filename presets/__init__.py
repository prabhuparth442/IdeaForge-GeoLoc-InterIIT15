"""Named configurations. settings.PRESET (or ./run ... --preset NAME) applies one before --set overrides. None = the submitted behaviour."""
import json
import os

import settings
from presets import pv_x, pv_x_r2

PRESETS = {"pv_x": pv_x.PV_X, "pv_x_r2": pv_x_r2.PV_X_R2}


def apply_preset(name):
    """Sets the values of preset `name` in settings and returns a list of warnings. LEVELS is not taken from the preset when the map store (settings.PREPROCESS_DIR) holds other levels: it is
    set to the levels the store has, so a store built with levels 3 to 7 still runs (photos above about 470 m then use level 7, which the red team showed costs precision)."""
    if name not in PRESETS:
        raise SystemExit("--preset %s: choose one of %s" % (name, ", ".join(sorted(PRESETS))))
    for key, value in PRESETS[name].items():
        setattr(settings, key, value)
    warnings = []
    info = os.path.join(settings.PREPROCESS_DIR, "store.json")
    if os.path.exists(info):
        with open(info) as handle:
            stored = sorted(json.load(handle)["levels"])
        wanted = list(PRESETS[name]["LEVELS"])
        if stored != sorted(wanted):
            settings.LEVELS = stored
            if max(stored) < max(wanted):
                warnings.append("preset %s: the map store has levels %d to %d, the preset was measured with %d to %d; photos above about %d m are drawn at level %d (build the store with --set LEVELS=%s to match)"
                                % (name, min(stored), max(stored), min(wanted), max(wanted), 470, max(stored), wanted))
    xoftr = os.path.join(settings.MODELS_DIR, settings.XOFTR_FILE)
    if not os.path.exists(xoftr):
        raise SystemExit("preset %s needs the XoFTR weights %s" % (name, xoftr))
    return warnings

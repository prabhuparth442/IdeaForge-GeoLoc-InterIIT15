"""Presets pv_x and pv_x_r2: names exist in settings.py, applying passes check_settings, the store's levels win over the preset's, a missing XoFTR file stops the run, the defaults stay off, --preset works in the command line.
Run: python tests/test_pv_x_preset.py"""
import argparse
import importlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import settings
from geoloc import cli
from geoloc.check_settings import check_settings
import presets

DEFAULTS = {name: getattr(settings, name) for name in list(presets.pv_x.PV_X) + list(presets.pv_x_r2.PV_X_R2) + ["PRESET", "PREPROCESS_DIR", "MODELS_DIR"]}


def reset():
    for name, value in DEFAULTS.items():
        setattr(settings, name, value)


def make_dirs(levels):
    root = tempfile.mkdtemp()
    models, store = os.path.join(root, "models"), os.path.join(root, "store")
    os.makedirs(models)
    os.makedirs(store)
    open(os.path.join(models, settings.XOFTR_FILE), "w").close()
    with open(os.path.join(store, "store.json"), "w") as handle:
        json.dump(dict(levels=levels), handle)
    return models, store


def test_defaults_are_off():
    reset()
    assert settings.PRESET is None and settings.PHOTO_VIEW_FROM_HEIGHT_M is None and settings.XOFTR_FROM_HEIGHT_M is None and not settings.PHOTO_VIEW_FIT_POSE
    check_settings()


def test_every_name_exists_and_values_pass_checks():
    for name, values in presets.PRESETS.items():
        reset()
        models, store = make_dirs(list(range(3, 12)))
        settings.MODELS_DIR, settings.PREPROCESS_DIR = models, store
        for key in values:
            assert hasattr(settings, key), key
        assert presets.apply_preset(name) == []
        settings.PRESET = name
        check_settings()
        assert settings.PHOTO_VIEW_MIN_INLIERS == 12 and settings.PHOTO_VIEW_FIT_POSE is True and settings.LEVELS[-1] == 11
    assert settings.PHOTO_VIEW_FIT_ROUNDS == 2 and settings.PHOTO_VIEW_PRIOR_TILT_DEG == 2.0 and settings.PHOTO_VIEW_FIT_MAX_MOVE_SIGMA == 6.0


def test_store_levels_win_and_warn():
    reset()
    models, store = make_dirs([3, 4, 5, 6, 7])
    settings.MODELS_DIR, settings.PREPROCESS_DIR = models, store
    warnings = presets.apply_preset("pv_x")
    assert settings.LEVELS == [3, 4, 5, 6, 7] and len(warnings) == 1 and "level 7" in warnings[0]


def test_missing_xoftr_stops():
    reset()
    models, store = make_dirs([3, 4, 5, 6, 7])
    os.remove(os.path.join(models, settings.XOFTR_FILE))
    settings.MODELS_DIR, settings.PREPROCESS_DIR = models, store
    try:
        presets.apply_preset("pv_x_r2")
    except SystemExit as error:
        assert "XoFTR" in str(error)
    else:
        raise AssertionError("no stop")


def test_command_line_and_set_after_preset():
    reset()
    models, store = make_dirs(list(range(3, 12)))
    args = argparse.Namespace(map=None, models=models, preprocess=store, dataset=None, mode=None, preset="pv_x_r2", set=["PHOTO_VIEW_MIN_INLIERS=16"])
    cli.apply_options(args)
    assert settings.PRESET == "pv_x_r2" and settings.PHOTO_VIEW_FIT_ROUNDS == 2 and settings.PHOTO_VIEW_MIN_INLIERS == 16
    reset()


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("pass", name)
    reset()

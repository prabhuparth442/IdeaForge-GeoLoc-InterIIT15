"""Frozen configuration PV-X (photo view + pose fit + XoFTR for every photo). Written down 2026-11-19; nothing here may change after a result is seen.
Not applied by default: settings.py is the submitted behaviour. To use it, call apply() before the pipeline starts, or copy these values into settings.py.
Results and caveats: IdeaForge-GeoLoc-InterIIT15/tests/Test_Iteration_12/results/photo_view/README.md and FREEZE_PV-X.md. Needs minima_xoftr.ckpt in MODELS_DIR."""
import settings

PV_X = dict(
    XOFTR_FROM_HEIGHT_M=0,
    PHOTO_VIEW_FROM_HEIGHT_M=0,
    PHOTO_VIEW_MIN_INLIERS=12,
    PHOTO_VIEW_FIT_POSE=True,
    PHOTO_VIEW_PRIOR_TILT_DEG=0.4,
    PHOTO_VIEW_PRIOR_YAW_DEG=0.8,
    PHOTO_VIEW_PRIOR_HEIGHT=0.008,
    PHOTO_VIEW_FIT_ROUNDS=1,
    LEVELS=[3, 4, 5, 6, 7, 8, 9, 10, 11],  # dev runs passed this on the command line; apply_preset() uses the levels of the store when they differ
)
# NUDT headless runs additionally used LEVELS=[3,4,5] and GEOMETRY attitude False, yaw_offset 0, scale 0.8 (set by hand for that camera; nothing finds them automatically).


def apply():
    for name, value in PV_X.items():
        setattr(settings, name, value)

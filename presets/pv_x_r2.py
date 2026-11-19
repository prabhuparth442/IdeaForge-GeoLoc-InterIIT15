"""PV-X-R2: PV-X plus the robust pose fit found by the red team (2026-11-19): two render-match-fit rounds, wide priors, move limit 6 sigma.
PV-X itself (presets/pv_x.py) is unchanged. Evidence: results/photo_view/red_team/ROBUSTNESS.md in the research repo. Off by default; call apply()."""
import settings
from presets import pv_x

PV_X_R2 = dict(pv_x.PV_X)
PV_X_R2.update(
    PHOTO_VIEW_FIT_ROUNDS=2,
    PHOTO_VIEW_PRIOR_TILT_DEG=2.0,
    PHOTO_VIEW_PRIOR_YAW_DEG=4.0,
    PHOTO_VIEW_PRIOR_HEIGHT=0.05,
    PHOTO_VIEW_FIT_MAX_MOVE_SIGMA=6.0,
)


def apply():
    for name, value in PV_X_R2.items():
        setattr(settings, name, value)

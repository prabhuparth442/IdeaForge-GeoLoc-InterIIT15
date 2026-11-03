import numpy as np

from header import settings


def pick_sample_ids(flight_of_frame, n_photos, seed):
    rng = np.random.default_rng(seed)
    per_group = max(1, n_photos // len(settings.FLIGHT_NAMES))
    ids = []
    for group in settings.FLIGHT_NAMES:
        members = np.where(flight_of_frame == group)[0]
        chosen = rng.choice(members, size=min(per_group, len(members)), replace=False)
        ids.extend(int(m) for m in chosen)
    return np.array(sorted(ids))

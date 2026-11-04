import numpy as np
from common import run_tests


def test_scoring_example_from_statement():
    from header.scoring import final_score
    assert abs(final_score([10] * 40 + [6] * 20 + [-20] * 2 + [0] * 38) - 28.8) < 1e-9


def test_sigma_values():
    from header.scoring import sigma_m
    assert abs(sigma_m(100, 200) - 8.14) < 0.01 and abs(sigma_m(1000, 200) - 17.0) < 0.01


def test_tiers():
    from header.scoring import points_for_error, sigma_m
    s = sigma_m(350, 200)
    assert [points_for_error(k * s, 350, 200) for k in (0.9, 1.9, 3.9, 7.9, 8.1)] == [10, 6, 3, 0, -20]


def test_skipped_photos_stay_in_denominator():
    from header.scoring import final_score
    assert abs(final_score([10, 0, 0, 0]) - 15.0) < 1e-9


def test_floor_at_zero():
    from header.scoring import final_score
    assert final_score([-20, -20, 10]) == 0


def test_sampling_is_balanced_and_repeatable():
    from header.sampling import pick_sample_ids
    from header import settings
    names = settings.FLIGHT_NAMES
    groups = np.array([names[0]] * 50 + [names[1]] * 50 + [names[2]] * 50)
    ids = pick_sample_ids(groups, 12, 7)
    assert [int((groups[ids] == g).sum()) for g in names] == [4, 4, 4]
    assert (ids == pick_sample_ids(groups, 12, 7)).all()


def test_bad_input_is_rejected():
    from header.scoring import sigma_m
    try:
        sigma_m(-5, 200)
    except AssertionError:
        return
    raise AssertionError("negative altitude was accepted")


run_tests(globals())

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch

from coarse_search.dino_hint import DinoHint
from coarse_search.self_tests import fast_equals_slow, map_tests
from evaluation.geometry_sweep import sweep_geometry
from evaluation.independence_check import check_independence, check_replay
from evaluation.report import collect_answers, print_final_scores, print_gated, print_results
from evaluation.run_sample import run_correlation, run_minima
from evaluation.sensitivity import run_sensitivity
from evaluation.testset import TestSet
from fine_matching.matcher import load_matcher
from header import settings
from header.data import Camera, assign_flights, find_map_file, make_items, read_metadata
from header.sampling import pick_sample_ids
from header.satellite_map import SatelliteMap
from header.scoring import check_scoring
from image_processing.map_store import build_map_store


def read_arguments():
    parser = argparse.ArgumentParser(description="UAV visual geo-localisation: run the pipeline on the development photos and score it")
    parser.add_argument("--n-photos", type=int, default=settings.N_PHOTOS, help="how many photos to test (spread over the height groups)")
    parser.add_argument("--no-minima", action="store_true", help="skip the MINIMA matcher (correlation only)")
    parser.add_argument("--dino", action="store_true", help="add the DINO hint to the correlation variants")
    parser.add_argument("--sweep", action="store_true", help="re-derive the camera geometry before the run (6.1)")
    parser.add_argument("--sensitivity", action="store_true", help="heading and scale error test after the run (6.6)")
    parser.add_argument("--tests", action="store_true", help="run the correlation self-tests first (3.3)")
    parser.add_argument("--independence", action="store_true", help="check that photos do not influence each other (5.3)")
    return parser.parse_args()


def main():
    args = read_arguments()
    use_dino = settings.USE_DINO or args.dino
    use_minima = settings.USE_MINIMA and not args.no_minima

    random.seed(settings.SEED)
    np.random.seed(settings.SEED)
    torch.manual_seed(settings.SEED)
    print("running on:", settings.DEVICE)
    check_scoring()

    table = read_metadata()
    items = make_items(table)
    camera = Camera(table)
    flight_of_frame = assign_flights(items)
    for name in settings.FLIGHT_NAMES:
        print("%-10s: %d photos" % (name, (flight_of_frame == name).sum()))

    satellite_map = SatelliteMap(find_map_file())
    print("map: %d x %d pixels, %.2f m per pixel" % (satellite_map.width, satellite_map.height, satellite_map.gsd))
    if abs(satellite_map.gsd - settings.MAP_GSD) > 1e-3:
        print("WARNING: the map pixel size differs from settings.MAP_GSD (%.2f)" % settings.MAP_GSD)

    sample_ids = pick_sample_ids(flight_of_frame, args.n_photos, settings.SEED)
    testset = TestSet(items, camera, satellite_map, flight_of_frame, sample_ids)
    print("testing on %d photos: %d low (<= 200 m), %d high" % (len(sample_ids), testset.sample_city.sum(), (~testset.sample_city).sum()))

    store = build_map_store(satellite_map)

    dino = None
    if use_dino:
        try:
            dino = DinoHint(store["levels"])
        except Exception as error:
            print("Could not load DINO (%s). Continuing without it." % error)

    matcher = None
    if use_minima:
        try:
            matcher = load_matcher()
        except Exception as error:
            print("Could not load MINIMA (%s: %s). Continuing without it." % (type(error).__name__, str(error)[:200]))

    if args.tests:
        fast_equals_slow()
        map_tests(store, testset.truths.mean(axis=0))

    if args.independence:
        check_ids = [int(j) for j in sample_ids[:: max(1, len(sample_ids) // 6)][:6]]
        check_independence(testset, store, matcher, check_ids)

    if args.sweep:
        settings.GEOMETRY.update(sweep_geometry(testset, store))

    variants = [("cosine", 0.0), ("zncc", 0.0)]
    if dino is not None:
        variants += [("cosine", settings.DINO_BETA), ("zncc", settings.DINO_BETA)]
    disc, seconds = run_correlation(testset, store, variants, dino)
    print("median time per photo: %.2f s (all %d variants together)" % (np.median(seconds), len(variants)))

    stage2, final = {}, {}
    if matcher is not None:
        stage2, final, _ = run_minima(testset, store, matcher, disc)
        check_replay(testset, store, matcher, final)

    results, closest, ratios, inlier_counts = collect_answers(testset, disc, stage2, final, matcher is not None)
    print_results(testset, results, closest, ratios, inlier_counts, matcher is not None)
    print_gated(results, ratios, inlier_counts, matcher is not None)
    print_final_scores(testset, results, closest, ratios, inlier_counts, stage2, disc, matcher is not None)

    if args.sensitivity:
        run_sensitivity(testset, store)


if __name__ == "__main__":
    main()

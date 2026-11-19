"""The three ways to run the pipeline: serve (the evaluation protocol), batch (a folder of photos), preprocess (build the map store)."""
import argparse
import ast
import csv
import os
import sys
import time
import traceback

import settings
from geoloc.check_settings import check_settings


def add_common_options(parser):
    parser.add_argument("--mode", choices=["answer", "full"], help="override settings.MODE")
    parser.add_argument("--preset", choices=["pv_x", "pv_x_r2"], help="a named configuration (settings.PRESET): pv_x = photo view + pose fit + XoFTR, pv_x_r2 = the same, robust to worse metadata; --set values are applied after it")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="override one value of settings.py, e.g. --set K_PEAKS=100 --set HEADINGS=[-5,0,5]")
    parser.add_argument("--map", help="folder with the GeoTIFF scenes")
    parser.add_argument("--models", help="folder with the model weights")
    parser.add_argument("--preprocess", help="folder of the map store")
    parser.add_argument("--dataset", help="folder with images/ and metadata.csv")


def apply_options(args):
    if args.map:
        settings.MAP_DIR = args.map
    if args.models:
        settings.MODELS_DIR = args.models
    if args.preprocess:
        settings.PREPROCESS_DIR = args.preprocess
    if args.dataset:
        settings.DATASET_DIR = args.dataset
    if args.mode:
        settings.MODE = args.mode
    if args.preset:
        settings.PRESET = args.preset
    if settings.PRESET:
        from presets import apply_preset
        for warning in apply_preset(settings.PRESET):
            print("WARNING: " + warning, file=sys.stderr, flush=True)
    for text in args.set:
        name, _, value = text.partition("=")
        if not hasattr(settings, name):
            raise SystemExit("--set %s: settings.py has no value of that name" % name)
        setattr(settings, name, ast.literal_eval(value))
    check_settings()


def log(text):
    print(text, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def result_line(name, meta_answer):
    if meta_answer is None or meta_answer["lat"] is None:
        return "RESULT,%s,no_match,,," % name
    status = "match" if meta_answer["answered"] else "no_match"
    return "RESULT,%s,%s,%.8f,%.8f,%.3f" % (name, status, meta_answer["lat"], meta_answer["lon"], meta_answer["confidence"])


def info_line(name, answer):
    parts = ["INFO", name, "answered=%s" % answer["answered"], "inliers=%d" % answer["inliers"]]
    if "total_seconds" in answer:
        parts.append("level=%s" % answer["level"])
        parts.append("queries=%s" % answer["n_queries"])
        parts.append("heading=%s" % answer["heading"])
        parts.append("scale=%s" % answer["scale_factor"])
        parts.append("total=%.2fs" % answer["total_seconds"])
        for stage, seconds in answer["seconds"].items():
            parts.append("%s=%.2fs" % (stage, seconds))
        if "gpu_peak_mb" in answer:
            parts.append("gpu_peak=%.0fMB" % answer["gpu_peak_mb"])
    return " ".join(parts)


def serve(args):
    from geoloc.pipeline import Localiser

    localiser = Localiser(settings.PREPROCESS_DIR, settings.MODELS_DIR)
    localiser.warm_up()
    log("start-up seconds: %s" % {key: round(value, 1) for key, value in localiser.setup_seconds.items()})
    print("READY", flush=True)

    header = next(csv.reader([sys.stdin.readline()]))
    for line in iter(sys.stdin.readline, ""):
        if line.strip() == "":
            continue
        meta = dict(zip(header, next(csv.reader([line]))))
        name = meta["filename"]
        answer = None
        try:
            answer = localiser.locate(os.path.join(settings.DATASET_DIR, "images", name), meta)
        except Exception:
            log("%s: failed, answering no_match\n%s" % (name, traceback.format_exc()))
        print(result_line(name, answer), flush=True)
        if answer is not None and settings.MODE == "full":
            log(info_line(name, answer))


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

def read_rows(dataset_dir):
    with open(os.path.join(dataset_dir, "metadata.csv"), newline="") as handle:
        return list(csv.DictReader(handle))


def move_hint(meta, index, radius_m, seed):
    """Development only: the hint point of the real evaluation is up to 5 km from the truth, the development set has it exactly on the truth."""
    import math
    import numpy as np
    rng = np.random.default_rng([seed, index])
    distance = radius_m * math.sqrt(rng.random())
    angle = 2 * math.pi * rng.random()
    lat = float(meta["lat_gtruth"])
    lon = float(meta["lon_gtruth"])
    moved = dict(meta)
    moved["lat_sc"] = str(lat + distance * math.sin(angle) / 111320.0)
    moved["lon_sc"] = str(lon + distance * math.cos(angle) / (111320.0 * math.cos(math.radians(lat))))
    return moved


def batch(args):
    from geoloc import scoring
    from geoloc.pipeline import Localiser

    rows = read_rows(settings.DATASET_DIR)
    rows = rows[args.first:]
    if args.limit:
        rows = rows[:args.limit]
    os.makedirs(args.out, exist_ok=True)

    localiser = Localiser(settings.PREPROCESS_DIR, settings.MODELS_DIR)
    localiser.warm_up()
    log("start-up seconds: %s" % {key: round(value, 1) for key, value in localiser.setup_seconds.items()})
    full = settings.MODE == "full"

    results_file = open(os.path.join(args.out, "results.csv"), "w", newline="")
    results = csv.writer(results_file)
    results.writerow(["filename", "status", "lat", "lon", "confidence"])
    details = None
    if full:
        details_file = open(os.path.join(args.out, "details.csv"), "w", newline="")
        details = csv.writer(details_file)
        details.writerow(["filename", "status", "inliers", "level", "heading", "scale_factor", "z", "total_seconds"]
                         + ["seconds_" + stage for stage in ["photo", "render", "coarse", "semantic", "fine", "decision"]]
                         + ["gpu_peak_mb", "error_m", "sigma_m", "points", "n_queries", "second_pass", "copy_used", "rescued_by_alignment", "refined", "refine_shift_m", "refine_inliers", "native_used", "xoftr_used", "fallback_queries", "photo_view_used", "pose_fit_used", "pose_fit_move_m", "fit_rms_px", "fit_dpitch", "fit_droll", "fit_dyaw", "fit_dheight", "fit_rounds", "second_inliers", "verify_inliers", "verify_distance_m"])

    points_list = []
    times = []
    for index, meta in enumerate(rows):
        name = meta["filename"]
        if args.hint_jitter_m > 0:
            meta = move_hint(meta, index + args.first, args.hint_jitter_m, args.seed)
        answer = None
        try:
            answer = localiser.locate(os.path.join(settings.DATASET_DIR, "images", name), meta)
        except Exception:
            log("%s: failed, counted as no_match\n%s" % (name, traceback.format_exc()))

        line = result_line(name, answer).split(",")
        results.writerow(line[1:])
        results_file.flush()

        if full and answer is not None:
            error_m, sigma, points = "", "", 0
            if "lat_gtruth" in meta and meta["lat_gtruth"] != "" and answer["lat"] is not None:
                error_m = scoring.distance_m(answer["lat"], answer["lon"], float(meta["lat_gtruth"]), float(meta["lon_gtruth"]))
                sigma = scoring.sigma_m(float(meta["alt"]), float(meta["cam_fx"]))
                if answer["answered"]:
                    points = scoring.points_for_error(error_m, float(meta["alt"]), float(meta["cam_fx"]))
            points_list.append(points)
            times.append(answer["total_seconds"])
            seconds = answer["seconds"]
            details.writerow([name, line[2], answer["inliers"], answer["level"], answer["heading"], answer["scale_factor"], answer["z"], "%.3f" % answer["total_seconds"]]
                             + ["%.3f" % seconds.get(stage, 0.0) for stage in ["photo", "render", "coarse", "semantic", "fine", "decision"]]
                             + ["%.0f" % answer.get("gpu_peak_mb", 0.0), error_m if error_m == "" else "%.1f" % error_m, sigma if sigma == "" else "%.1f" % sigma, points]
                             + [answer.get("n_queries", ""), answer.get("second_pass", ""), answer.get("copy_used", ""), answer.get("rescued_by_alignment", ""), answer.get("refined", ""), answer.get("refine_shift_m", ""), answer.get("refine_inliers", ""), answer.get("native_used", ""), answer.get("xoftr_used", ""), answer.get("fallback_queries", ""), answer.get("photo_view_used", ""), answer.get("pose_fit_used", ""), answer.get("pose_fit_move_m", "")] + [answer.get("fit_" + key, "") for key in ("rms_px", "dpitch", "droll", "dyaw", "dheight", "rounds")] + [answer.get("second_inliers", ""), answer.get("verify_inliers", ""), answer.get("verify_distance_m", "")])
            details_file.flush()
            log("%d/%d %s" % (index + 1, len(rows), info_line(name, answer)))
        elif not full:
            log("%d/%d %s" % (index + 1, len(rows), line[2]))

    results_file.close()
    if full:
        details_file.close()
        if len(points_list) > 0 and any(row.get("lat_gtruth") for row in rows):
            log("score %.1f / 60 on %d photos (a no_match counts 0)" % (scoring.final_score(points_list), len(points_list)))
        if len(times) > 0:
            log("seconds per photo: mean %.2f, max %.2f" % (sum(times) / len(times), max(times)))
    log("results written to " + args.out)


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

def find_scenes(map_dir):
    scenes = []
    for folder, _, files in os.walk(map_dir):
        for filename in files:
            if filename.lower().endswith((".tif", ".tiff")):
                scenes.append(os.path.join(folder, filename))
    return sorted(scenes)


def preprocess(args):
    from geoloc.map_store import build_map_store

    scenes = find_scenes(settings.MAP_DIR)
    if len(scenes) == 0:
        raise SystemExit("no GeoTIFF found in " + settings.MAP_DIR)
    region = settings.MAP_REGION
    if args.region:
        region = tuple(float(part) for part in args.region.split(","))
    os.makedirs(settings.PREPROCESS_DIR, exist_ok=True)
    log_file = open(os.path.join(settings.PREPROCESS_DIR, "preprocess.log"), "a")

    def write_log(text):
        stamped = "%s %s" % (time.strftime("%H:%M:%S"), text)
        print(stamped, file=sys.stderr, flush=True)
        log_file.write(stamped + "\n")
        log_file.flush()

    write_log("scenes: %s" % scenes)
    build_map_store(scenes, settings.PREPROCESS_DIR, region, write_log)
    log_file.close()


# ---------------------------------------------------------------------------

def main(argv):
    parser = argparse.ArgumentParser(prog="run")
    commands = parser.add_subparsers(dest="command", required=True)

    serve_parser = commands.add_parser("serve", help="the evaluation protocol on stdin / stdout")
    add_common_options(serve_parser)

    batch_parser = commands.add_parser("batch", help="answer every photo of a dataset folder and write results.csv")
    add_common_options(batch_parser)
    batch_parser.add_argument("--out", default="results", help="folder for results.csv (and details.csv in full mode)")
    batch_parser.add_argument("--limit", type=int, default=0, help="only this many photos (0 = all)")
    batch_parser.add_argument("--first", type=int, default=0, help="skip this many photos at the start")
    batch_parser.add_argument("--hint-jitter-m", type=float, default=0.0, help="development only: move the hint point randomly up to this far (metres) from the truth; 5000 imitates the evaluation")
    batch_parser.add_argument("--seed", type=int, default=0, help="seed of the hint jitter")

    preprocess_parser = commands.add_parser("preprocess", help="build the map store from the GeoTIFFs")
    add_common_options(preprocess_parser)
    preprocess_parser.add_argument("--region", help="west,south,east,north in degrees: build only this part of the map (for tests)")

    args = parser.parse_args(argv)
    apply_options(args)
    if args.command == "serve":
        serve(args)
    elif args.command == "batch":
        batch(args)
    else:
        preprocess(args)

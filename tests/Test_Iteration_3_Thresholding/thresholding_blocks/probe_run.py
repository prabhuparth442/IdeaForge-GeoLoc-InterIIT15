"""One photo through Step 1: the unchanged pipeline, then the extra tiles. Needs a set-up SinglePhotoPipeline and the repo's src/ on sys.path.
It reads the pipeline; it does not change it or anything in src/. The true position is used only to choose tiles for the diagnosis."""
import math

from image_processing.geometry import render_query
from image_processing.map_levels import level_gsd
from thresholding_blocks.probe_blocks import offset_label, offset_positions
from thresholding_blocks.probe_matcher import probe_positions


def probe_photo(pipeline, index, group, offsets_px, seed, keep_labels=("truth", "chosen_candidate")):
    """pipeline: a SinglePhotoPipeline after setup(). index: photo number in pipeline.test_set. group: a label ('skipped' or 'control').
    offsets_px: list of (dx, dy) map pixels around the truth, the first must be (0, 0). keep_labels: tiles and matches handed back for pictures.
    Returns dict(probe_rows, photo_row, canvas, valid, tiles, matches)."""
    assert pipeline.ready, "call pipeline.setup() first"
    item = pipeline.test_set.items[index]
    truth_x, truth_y = pipeline.test_set.truths[index]
    result = pipeline.run(item, pipeline.test_set.priors[index])            # the unchanged pipeline
    level = result["level"]
    gsd = level_gsd(level)
    canvas, valid = render_query(item, pipeline.camera, level)              # the picture the fine stage was given
    peaks, chosen = result["peaks"], result["chosen"]
    candidate_x, candidate_y = float(peaks[chosen, 1]), float(peaks[chosen, 2])
    positions = offset_positions(truth_x, truth_y, gsd, offsets_px)
    positions.append(("chosen_candidate", candidate_x, candidate_y))
    positions.append(("coarse_peak0", float(peaks[0, 1]), float(peaks[0, 2])))
    rows, tiles, matches = probe_positions(pipeline.matcher, canvas, valid, level, pipeline.level_images, positions, seed)
    distance = {offset_label(dx, dy): math.hypot(dx, dy) for dx, dy in offsets_px}
    for row in rows:
        row.update(photo=index, group=group, level=level, offset_px=distance.get(row["label"], float("nan")))
    photo_row = dict(photo=index, group=group, level=level, height=item["rel_height"], sigma_m=float(pipeline.test_set.sigmas[index]),
                     answered_in_run=bool(result["answered"]), pipeline_inliers=int(result["inliers"]), pipeline_chosen=int(chosen),
                     z_chosen=float(result["z_chosen"]),
                     candidate_offset_px=math.hypot(candidate_x - truth_x, candidate_y - truth_y) / gsd,
                     peak0_offset_px=math.hypot(float(peaks[0, 1]) - truth_x, float(peaks[0, 2]) - truth_y) / gsd,
                     pipeline_inliers_of_chosen_candidate=float(result["stage2"]["inliers"][chosen]))
    return dict(probe_rows=rows, photo_row=photo_row, canvas=canvas, valid=valid,
                tiles={label: tiles[label] for label in keep_labels}, matches={label: matches[label] for label in keep_labels})

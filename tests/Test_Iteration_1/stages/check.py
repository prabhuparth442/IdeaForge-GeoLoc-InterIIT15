"""Stage check: GPU, packages, NUDT files, split sizes."""
from pathlib import Path

import torch

from header import settings
from train import config
from train.datasets.nudt_data import find_reference_map, read_split


def run_stage(run):
    run.log("torch", torch.__version__, "| device:", settings.DEVICE, "| gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    if settings.DEVICE != "cuda":
        run.log("WARNING: no GPU. The full run will take days on CPU. Use --smoke to check the code, or run on a GPU machine.")
    import cv2, pandas, rasterio  # noqa: F401
    if run.args.smoke:
        run.log("smoke mode: synthetic world, no dataset needed")
        return
    nudt = config.find_nudt_dir()
    reference = find_reference_map(nudt)
    run.log("NUDT folder:", nudt, "| map file:", reference, "| %.1f GB" % (reference.stat().st_size / 1e9))
    for split in ("train", "valid", "test"):
        items = read_split(nudt, split)
        run.log("  %-5s: %5d frames in %d flights" % (split, len(items), len({item["flight"] for item in items})))
    missing = [item["name"] for item in read_split(nudt, "valid")[:30] if not Path(item["name"]).exists()]
    assert not missing, "frame files listed in the csv are missing, e.g. %s" % missing[:2]

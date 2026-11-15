"""Record workspace/code provenance and physical-photo split evidence without altering inputs."""
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORKSPACE = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def git(path, *args):
    result = subprocess.run(["git", "-C", str(path), *args], text=True, capture_output=True)
    return result.stdout.strip()


def inventory():
    folders = []
    for folder in sorted(WORKSPACE.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        sizes, extensions = 0, Counter()
        for current, directories, files in os.walk(folder):
            directories[:] = [name for name in directories if name not in (".git", "__pycache__", ".pytest_cache")]
            for name in files:
                path = Path(current) / name
                if path.is_symlink():
                    continue
                sizes += path.stat().st_size
                extensions[path.suffix or "(no suffix)"] += 1
        folders.append(dict(folder=folder.name, files=sum(extensions.values()), bytes=sizes, extensions=dict(extensions)))
    (HERE / "results/workspace_inventory_codex.json").write_text(json.dumps(folders, indent=2) + "\n")


def source_manifest():
    sources = []
    for folder in (ROOT, WORKSPACE / "Main_Pipeline_Headless", WORKSPACE / "Submission_Files/myteam/code"):
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*.py")):
            relative = path.relative_to(folder)
            if relative.parts[0] in ("tests", "dataset", "third_party", ".git"):
                continue
            sources.append(dict(tree=str(folder.relative_to(WORKSPACE)), path=str(relative),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), inode=path.stat().st_ino))
    pd.DataFrame(sources).to_csv(HERE / "results/source_manifest_codex.csv", index=False)
    states = {}
    for folder in (ROOT, WORKSPACE / "Main_Pipeline_Headless"):
        states[folder.name] = dict(head=git(folder, "rev-parse", "HEAD"), branch=git(folder, "branch", "--show-current"),
                                   status=git(folder, "status", "--short"), branches=git(folder, "branch", "-avv"))
    states[ROOT.name]["main_code_changes"] = git(ROOT, "diff", "--stat", "main...HEAD", "--", "src", "train", "*.py")
    (HERE / "results/git_state_codex.json").write_text(json.dumps(states, indent=2) + "\n")


def split_audit():
    from train import config
    from train.datasets.nudt_data import read_split, spread_sample

    items = read_split(config.find_nudt_dir(), "test")
    pool = [items[index] for index in spread_sample(items, 2400, 7)]
    source = ROOT / "tests/Test_Iteration_7/results"
    selected = sorted(int(path.stem.split("_")[1]) for path in (source / "rich_dump/nudt_fresh707").glob("photo_*.npz"))
    old_pool = set(int(index) for index in spread_sample(pool, 500, 7))
    prior_fresh = set(int(index) for index in pd.read_csv(source / "exclude_t7_seen_fresh.csv").photo)
    seen = old_pool | prior_fresh
    rows = []
    for index, item in enumerate(pool):
        role = "not_in_listed_evaluations"
        if index in seen:
            role = "prior_seen"
        if index in selected:
            role = "nudt707_already_seen"
        rows.append(dict(photo=index, canonical_path=item["name"], flight=item["flight"], lat=item["lat"], lon=item["lon"], role=role))
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "results/nudt_photo_manifest_codex.csv", index=False)
    # Local tangent-plane distance is enough to diagnose adjacent-scene dependence.
    latitude = np.radians(table.lat.mean())
    xy = np.column_stack([table.lon * 111320 * np.cos(latitude), table.lat * 111320])
    tree = cKDTree(xy[sorted(seen)])
    distance, _ = tree.query(xy[selected])
    summary = dict(total_test_pool=len(pool), selected=len(selected), old_pool=len(old_pool),
                   exact_id_overlap_old=len(set(selected) & old_pool), exact_id_overlap_fresh=len(set(selected) & prior_fresh),
                   selected_flights=dict(Counter(pool[index]["flight"] for index in selected)),
                   old_pool_flights=dict(Counter(pool[index]["flight"] for index in old_pool)),
                   nearest_seen_m_median=float(np.median(distance)), within_50m=int((distance < 50).sum()),
                   within_100m=int((distance < 100).sum()),
                   note="No listed frame overlap, but shared flights and nearby scenes. Unlisted frames are not certified blind.")
    (HERE / "results/split_audit_codex.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    inventory()
    source_manifest()
    split_audit()

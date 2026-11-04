"""Small pieces several stages need: map names, feature families, shard folders, fusion models, a result line."""
import hashlib
import json
from pathlib import Path

from header import settings
from train import config
from train.datasets.candidates import load_shards
from train.features.edge import EdgeFamily
from train.features.mind import MindFamily


def map_names(f4_file):
    names = list(config.MAP_NAMES_BASE)
    if Path(f4_file).exists():
        names.append("learned_zncc")
    return names


def build_families(store, f4_file, log=print):
    """The edge, MIND and (if trained) learned feature families for every map level in settings.LEVELS. Returns (names, families)."""
    families = {"edge": EdgeFamily(store), "mind": MindFamily(store, settings.LEVELS)}
    if Path(f4_file).exists():
        from train.features.learned_family import LearnedFamily
        families["learned"] = LearnedFamily(store, settings.LEVELS, f4_file)
        log("learned family loaded (valid hit rate %.1f %% vs edge %.1f %%)" % (100 * families["learned"].hit, 100 * families["learned"].baseline))
    return map_names(f4_file), families


def shard_dir(out_dir, names):
    return Path(out_dir) / ("shards_" + hashlib.md5(",".join(names).encode()).hexdigest()[:8])


def load_real_candidates(out_dir):
    names = map_names(Path(out_dir) / "models" / "f4.pt")
    folder = shard_dir(out_dir, names)
    return names, load_shards(folder, "train", names), load_shards(folder, "valid", names)


def apply_yaw_sign(test_sets, sign):
    """NUDT yaw may run the other way round: the sign is applied to the photo items themselves (yaw = sign * yaw_raw)."""
    for test_set in test_sets:
        for item in test_set.items:
            item["yaw"] = sign * item.get("yaw_raw", item["yaw"])


def real_fusion_models(out_dir, state_results):
    """The fusion models of the real-photo run: the best one and the best linear one. label -> (model, maps)."""
    from coarse_search.fusion import load_model
    best = state_results["fusion_best"]
    fusion_dir = Path(out_dir) / "models" / "fusion"
    models = {"fused:" + best["tag"]: (load_model(fusion_dir / (best["tag"] + ".pt")), best["subset"])}
    linear_rows = [row for row in state_results["fusion_rows"] if row["kind"] == "linear"]
    if linear_rows:
        top = max(linear_rows, key=lambda row: row["valid_top1"])
        tag = "%s__linear" % top["maps"]
        models["fused:" + tag] = (load_model(fusion_dir / (tag + ".pt")), top["maps"].split("+"))
    return models


def synthetic_fusion_models(real_out, synth_out, synth_results):
    """The model trained on real photos only (earlier run) and the two trained with pseudo-thermal photos."""
    from coarse_search.fusion import load_model
    real_results = json.load(open(Path(real_out) / "state.json"))["results"]
    best = real_results["fusion_best"]
    models = {"fused:real_only": (load_model(Path(real_out) / "models" / "fusion" / (best["tag"] + ".pt")), best["subset"])}
    for kind in ("synth", "mixed"):
        best = synth_results["ablation_" + kind]["best"]
        models["fused:" + kind] = (load_model(Path(synth_out) / "models" / kind / (best["tag"] + ".pt")), best["subset"])
    return models


def log_rows(run, title, rows):
    for r in rows:
        run.log("%s %-52s top-1 within 4 sigma %.3f (1s %.3f, 2s %.3f) ceiling@25 %.3f vs edge_zncc %+.3f [%+.3f, %+.3f]" % (
            title, r["map"][:52], r["top1_within_4sigma"], r["top1_within_1sigma"], r["top1_within_2sigma"], r["ceiling25_within_4sigma"],
            r["vs_baseline_4sigma"]["diff"], r["vs_baseline_4sigma"]["lo"], r["vs_baseline_4sigma"]["hi"]))

"""report.md for the synthetic-training run: what was done, and the real-photo results side by side."""
import json
from pathlib import Path


def table(rows):
    lines = ["| model | top-1 <= 1 sigma | <= 2 sigma | <= 4 sigma | best of 25 <= 4 sigma | vs edge_zncc (4 sigma) | 95% interval |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        change = r["vs_baseline_4sigma"]
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %+.3f | %+.3f .. %+.3f |" % (
            r["map"][:60], r["top1_within_1sigma"], r["top1_within_2sigma"], r["top1_within_4sigma"], r["ceiling25_within_4sigma"],
            change["diff"], change["lo"], change["hi"]))
    return "\n".join(lines)


def best_fused(rows):
    fused = [r for r in rows if r["map"].startswith("fused:")]
    if not fused:
        return None
    return max(fused, key=lambda r: r["top1_within_4sigma"])


def verdict(rows, name):
    synth = [r for r in rows if r["map"].startswith("fused:synth")]
    mixed = [r for r in rows if r["map"].startswith("fused:mixed")]
    real = [r for r in rows if r["map"].startswith("fused:real_only")]
    if not (synth and mixed and real):
        return "%s: not all three fused models were evaluated." % name
    parts = []
    for label, row in (("synthetic only", synth[0]), ("synthetic + real", mixed[0])):
        gap = row["top1_within_4sigma"] - real[0]["top1_within_4sigma"]
        parts.append("%s %+.3f against the model trained on real photos only" % (label, gap))
    return "%s (top-1 within 4 sigma): %s. Differences smaller than the 95%% intervals above can be chance." % (name, "; ".join(parts))


def write_report(out_dir, state, args):
    out_dir = Path(out_dir)
    results = state["results"]
    text = ["# Fusion trained on pseudo-thermal photos", ""]
    text.append("Fake photos were cut from the Changsha map (away from the valid/test photos), turned into pseudo-thermal canvases and "
                "pushed through the same maps and candidate sets as real photos. Three fusion models are compared on REAL thermal photos:")
    text.append("- `fused:real_only`: trained on the real NUDT train photos (the earlier run)")
    text.append("- `fused:synth`: trained on fake photos only")
    text.append("- `fused:mixed`: fake photos plus the real train photos (each real photo counted %d times)" % args.real_repeat)
    text.append("")
    text.append("Fake-photo strength chosen by comparing with real training photos: **%s**. Pictures: `pseudo_samples.png`." % results.get("strength", "not run"))
    text.append("")
    for kind in ("synth", "mixed"):
        if "ablation_" + kind in results:
            best = results["ablation_" + kind]["best"]
            text.append("Best %s model, chosen on the real valid photos: `%s` (%s), valid top-1 %.3f." % (kind, best["maps"], best["kind"], best["valid_top1"]))
    text.append("")
    for key, title in (("evaluate_nudt", "Real NUDT test photos (same city as the fake photos' map)"),
                       ("evaluate_official", "Official development photos on the small_sat_area map (another city: the generalisation check)")):
        if key in results:
            text.append("## " + title)
            text.append("")
            text.append(table(results[key]))
            text.append("")
            text.append(verdict(results[key], title.split(" (")[0]))
            text.append("")
    if state["failed"]:
        text.append("## Failed stages")
        text.append(json.dumps(state["failed"], indent=1))
    text.append("")
    text.append("## Caveats")
    text.append("- Fake photos come from the very map they are matched against, so they are easier than real photos; the strength step "
                "only matches how often the truth wins, not every kind of difficulty.")
    text.append("- Only the official-set table says anything about another place. A gain on NUDT alone could still be Changsha.")
    text.append("- DINO is not part of this run: the fusion still has the same six maps.")
    (out_dir / "report.md").write_text("\n".join(text) + "\n")

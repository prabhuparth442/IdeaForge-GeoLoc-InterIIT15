"""Writes report.md from the results saved in state.json."""
import time


def pct(x):
    return "%.1f %%" % (100 * x)


def write_report(out, state, args):
    r = state["results"]
    lines = ["# Overnight run report", "", "Written %s. Mode: %s." % (time.strftime("%Y-%m-%d %H:%M"), "SMOKE (synthetic world; numbers mean nothing)" if args.smoke else "NUDT"), ""]
    if state["failed"]:
        lines += ["## Stages that failed", ""] + ["- **%s**: %s" % (k, v) for k, v in state["failed"].items()] + [""]
    if "geometry" in r:
        g = r["geometry"]
        lines += ["## Photo-to-map geometry", "", "yaw sign %+d, yaw offset %.0f deg, scale %.2f; %s of the check photos land within 4 sigma with it." % (g.get("yaw_sign", 1), g["yaw_offset"], g["scale"], pct(g["share_within_4_sigma"])),
                  "If this is below 30 %% nothing downstream can be trusted.", ""]
    if "f4" in r:
        f = r["f4"]
        lines += ["## F4, learned features", "", "On held-out pairs (truth within 2 cells inside a +-40 cell window): learned **%s**, plain edge features **%s** (%d steps)." % (pct(f["valid_hit_rate"]), pct(f["edge_hit_rate"]), f["steps"]),
                  "Pass rule of the patch document: a clear gain here is what justifies using it in the full search.", ""]
    if "fusion_rows" in r:
        lines += ["## C2, fusion (valid flights, truth against the candidate list)", "", "| maps | model | valid top-1 | vs edge_zncc alone (95 % interval) |", "|---|---|---|---|"]
        for row in r["fusion_rows"]:
            v = row["vs_edge_zncc"]
            lines.append("| %s | %s | %.3f | %+.3f [%+.3f, %+.3f] |" % (row["maps"], row["kind"], row["valid_top1"], v["diff"], v["lo"], v["hi"]))
        b = r["fusion_best"]
        lines += ["", "Best on valid: **%s / %s** (%.3f)." % (b["maps"], b["kind"], b["valid_top1"]), ""]
    if "ranker" in r:
        k = r["ranker"]
        lines += ["## R1, ranker on the hard candidates", "", "fused score alone %.3f, ranker %.3f, difference %+.3f (95 %% %+.3f .. %+.3f). Use it only if the interval is above 0." % (k["fused_top1"], k["ranker_top1"], k["diff"], k["lo"], k["hi"]), ""]
    if "evaluate" in r:
        lines += ["## Final check on the full score maps (NUDT test flights, never used for choosing)", "",
                  "| map / model | top-1 within 1 sigma | 2 sigma | 4 sigma | best of 25 within 4 sigma | median error (sigma) | 4-sigma change vs edge_zncc |", "|---|---|---|---|---|---|---|"]
        for row in r["evaluate"]:
            v = row["vs_baseline_4sigma"]
            lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.1f | %+.3f [%+.3f, %+.3f] |" % (row["map"], row["top1_within_1sigma"], row["top1_within_2sigma"], row["top1_within_4sigma"],
                                                                                              row["ceiling25_within_4sigma"], row["median_error_sigma"], v["diff"], v["lo"], v["hi"]))
        fused = [x for x in r["evaluate"] if x["map"].startswith("fused:")]
        singles = [x for x in r["evaluate"] if not x["map"].startswith("fused:")]
        if fused and singles:
            best_single = max(singles, key=lambda x: x["ceiling25_within_4sigma"])
            best_fused = max(fused, key=lambda x: x["ceiling25_within_4sigma"])
            gain = best_fused["ceiling25_within_4sigma"] - best_single["ceiling25_within_4sigma"]
            lines += ["", "Pass rule C2 (fused >= best single by 0.03 at the 25-candidate ceiling): best fused %s vs best single %s (%s): gain %+.3f -> **%s**." % (
                best_fused["map"], best_single["map"], pct(best_single["ceiling25_within_4sigma"]), gain, "PASS" if gain >= 0.03 else "not passed")]
    if "official" in r:
        nudt = {x["map"]: x for x in r.get("evaluate", [])}
        lines += ["", "## Official development set (d1 photos, small_sat_area map): the overfitting / domain check", "",
                  "Models were fitted on NUDT only. Same score maps and fusion models, different city, camera, sensor and map.", "",
                  "| map / model | NUDT test 4-sigma | official 4-sigma | official 1-sigma | official best of 25 | official change vs edge_zncc (95 % interval) |", "|---|---|---|---|---|---|"]
        for row in r["official"]:
            v = row["vs_baseline_4sigma"]
            n_t = nudt.get(row["map"])
            lines.append("| %s | %s | %.3f | %.3f | %.3f | %+.3f [%+.3f, %+.3f] |" % (row["map"], "%.3f" % n_t["top1_within_4sigma"] if n_t else "-", row["top1_within_4sigma"],
                                                                                   row["top1_within_1sigma"], row["ceiling25_within_4sigma"], v["diff"], v["lo"], v["hi"]))
        fused = [x for x in r["official"] if x["map"].startswith("fused:")]
        if fused:
            b = max(fused, key=lambda x: x["top1_within_4sigma"])
            lo = b["vs_baseline_4sigma"]["lo"]
            lines += ["", "Reading: if a fused model beats edge_zncc on NUDT test but its interval on the official set includes 0 or is negative, the gain was specific to Changsha (overfitting to the domain). "
                      "Best fused here: %s, change %+.3f (lower end %+.3f) -> **%s**." % (b["map"], b["vs_baseline_4sigma"]["diff"], lo, "holds on the official set" if lo > 0 else "does NOT clearly hold on the official set")]
    lines += ["", "## Files", "", "run.log (everything that happened), state.json (all numbers), models/ (f4.pt, fusion/*.pt, ranker.pt), shards_*/ (candidate sets)."]
    (out / "report.md").write_text("\n".join(lines))

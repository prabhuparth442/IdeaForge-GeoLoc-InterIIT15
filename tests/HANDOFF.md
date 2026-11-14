# Handoff: research state at 2026-11-14 16:30 IST (T6 → T10)

**Update 16:30: T9 and T10 are finished** (see their README / REPORT). Final T10 result on the fresh NUDT 808 draw (pre-registered, scored once):
PC V5 (relative inliers + scale augmentation, no override) +2.40 [+0.52, +4.64], 0 wrong at every inlier scale x0.5-x2; Jetson V5 / J1 with
early exit: 0 wrong at every scale, +0.32 (not proven), ~2.2 queries per photo. NUDT 808 is now seen data. Sections 4-5 below describe the
state before the update and are kept for history. Next ideas: section 8.

For whoever continues on another machine. Everything below is on branch `palash/dev` except the large data (section 3).

## 1. Rules (from the user; follow exactly)

- **Never edit `src/`.** New code goes in `train/checks/` (blocks: one job each, docstring, no plotting) with unit tests in `tests/pipeline/`.
- **Notebooks are the face, blocks do the work.** Experiments are executed notebooks saved as `<name>_executed*.ipynb`. No `run_*.py` scripts
  for experiments (small `run_*.sh` drivers for GPU dumps are used).
- **No multi-query agreement** in any decision (user ruled it too risky).
- **0 wrong answers on every set** checked. A gain counts only on data nothing was tuned on, with a paired-bootstrap 95 % interval above 0.
- **Protocol since T8:** build on the NUDT design sets (101, 202, 404, 505, 606); check on official TRY (185 photos); nothing tuned on official
  TEST (194) or NUDT 707 (both are now **seen**). Any new final claim needs a **fresh NUDT draw (seed 808, excluding all earlier photos) dumped
  before choosing any rule**, and **pre-registered finalists**. Report score, wrong, answered, queries, paired bootstrap vs the base, versus the
  plain bar that answers as many photos (`holdout.versus_frontier`), and `holdout.answer_change`. No threshold sweeps aimed at known mistakes.
- Long jobs in the background; no sleep-polling. Pin CPU work with `taskset -c 0-9,12-31` (CPUs 10/11 on the old laptop had machine-check
  errors; on a new machine pin to its own cores). Set `OMP_NUM_THREADS` to the allowed cores (oversubscription made jobs ~100x slower).
- **Commit and push only to `palash/dev`**, only the user's own work. Other members' branches (e.g. `prabhuparth/dev`) are import-only. The old
  laptop's checkout stays on `prabhuparth/dev`; commits to `palash/dev` were made with a temporary git index (see section 6).
- Notion: a page per iteration plus a summary under **InterIIT_UAV_GeoLoc** (page id `41642f5fcd9481e49ff3dd66d6fef3f6`).
- Keep the user's email and personal details out of any external request.
- **Answered 2026-11-14: satellite-only** (no NUDT thermal training for learned components; NUDT / official photos MAY set cut thresholds and test; see section 8).
- Time classes for a method's Jetson cost: < 4 s good, 4-6 s viable, >= 6 s slow (model: 2.2 s per LoFTR query at M = 5).

## 2. Scoring and pipeline (for context)

Per photo: <= 1 sigma +10, <= 2 sigma +6, <= 4 sigma +3, <= 8 sigma 0, worse -20, skip 0; score /60 = mean x 6. Pipeline: thermal photo ->
north-up canvas -> edge ZNCC FFT coarse search over a 5.5 km disc -> K = 50 peaks -> DINO re-rank -> LoFTR-MINIMA on M = 10 (Jetson 5) ->
RANSAC similarity -> choice by inliers + DINO z -> answer or skip. Queries: standard 6 (heading -5/0/+5 x scale 1.0/1.05), wide 27, Jetson second
pass (9 headings x 2 scales, cap 10 in the replay).

## 3. Data that is NOT in git (copy it, or regenerate)

| What | Path (repo-relative) | Size | Regenerate |
|---|---|---|---|
| T6 rich dumps (raw LoFTR matches of every candidate, all 27 queries) | `tests/Test_Iteration_6/results/rich_dump/{official, nudt_seed101, nudt_seed202, nudt_fresh404, nudt_fresh505, nudt_fresh606, nudt_fresh707}` | 251 MB | GPU, ~50-75 min per 300 photos (`tests/Test_Iteration_6/notebooks/Rich_Dump_Run.ipynb`, drivers `run_dumps*.sh`, `Test_Iteration_7/run_blind_dump.sh`, `Test_Iteration_8/run_blind_dump.sh`) |
| Signal tables (per candidate, every signal) | `tests/Test_Iteration_7/results/signals/*.csv.gz`, `tests/Test_Iteration_8/results/signals/nudt707.csv.gz` | 54 MB | CPU minutes: T7 `It1_Signal_Table`, T8 `It1_Blind_Signal_Table` |
| T9 altered-photo dumps (partial) | `tests/Test_Iteration_9/results/rich_dump/official_*`, `perturbed_photos/` | 64 MB | `tests/Test_Iteration_9/run_perturbed_dumps.sh` (resumes) |
| Datasets / weights | `dataset/` (see `dataset/README.md`) | ~30 GB | team Drive |

A tarball of the first three rows was made on the old laptop: `~/GeoLoc_research_data_2026-11-14.tar` (unpack in the repo root).
Venv on the old laptop: `~/Ideaforge/GeoLoc/.venv-geoloc` (Python 3.10, torch 2.8; `requirements.txt`).

## 4. Results so far (one line each; details in each iteration's README / REPORT)

- **T6** (`tests/Test_Iteration_6`): rich-dump replay; attitude bias ~0.98 deg; XoFTR second opinion +2.7 official but a second network; DINO-z veto
  removes fresh-NUDT wrong answers; learned verifier on the chosen candidate does not transfer.
- **T7** (`tests/Test_Iteration_7`): candidate-level monotone GBM judge on every fine-matched candidate (19 features, `FEATURES_CTX`), Learn-then-Test
  cut; early exit (P >= 0.98) cuts Jetson queries ~6.3 -> ~2 (14 s -> ~4.5 s). Blind 606: gain not proven.
- **T8** (`tests/Test_Iteration_8`): the team's review (judge fitted to the data it improved) confirmed: ~3 official points were memory. Official split
  by place into TRY / TEST (`results/official_split.csv`); blind NUDT 707. Pre-registered finalists scored once: **the T7 judge trained on NUDT
  only** = official TEST +5.13 [+2.97, +7.61] (PC) / +3.59 (Jetson), NUDT 707 +1.40 [+0.20, +3.00] / +0.80, **0 wrong** on both; Jetson + early
  exit +3.46 / +0.80 at 2-3 queries. Two-tier rule (3 numbers) weaker; attitude correction not proven. Commit `5f6c5bc`.
- **T9** (red team: "is the method input-invariant?") — **answer: NO.** Offline (`It1_Offline_Red_Team`, `It2_Hardening`):
  - NUDT vs official candidates are separable (AUC 0.988); biggest shift is the coarse ZNCC score, then ratio, spread, sharpness, z.
  - P is not calibrated across sets: wrong rate at P 0.9-0.95 is 15 % NUDT (out-of-fold) / 12 % NUDT 707 / 1 % official; at P 0.8-0.9, 43 % on 707.
  - Margin: on 707 a cut of 0.8 already gives a wrong answer (0.85+ is clean); official is clean down to 0.5.
  - By height band (official 100-351 m): 0 wrong, gains in every band but 175-275 (flat). Heights > 351 m: no data.
  - **Uniform inlier inflation** (every candidate x1.25, as a sharper camera / looser matcher would give): the judge with its 12-inlier override
    makes 2-3 wrong answers on 707, today's bars 5-6 (x2: bars 14-18). **Without the override** the judge makes 1-2 (cut 0.9) and **0 at cut 0.98
    up to x2** (cost ~2 points on official at x1). Removing the override costs nothing on clean data.
  - Feature hardening (drop coarse/sharp, monotone gaps, relative-only features) does not remove the inflation failure.
  - **Not done:** `It3_Altered_Photos` (real pipeline runs on altered photos: polarity flip, blur, noise, half resolution, tone curve, heading +15,
    height +20 %). Dumps: official 5 of 7 alterations complete (invert, blur, noise, lowres, gamma: 60 photos each), yaw15 4/60, height20 0/60,
    NUDT 707 0 of 7. Smoke check: polarity flip 51 -> 37 inliers, heading +15 deg 51 -> 3. T9 README / REPORT / Notion not written yet.
- **T10** (in progress, brief in section 5):
  - 1a done: Jetson context now from rank < 5 only (`candidate_judge.add_context(m=5)`). Effect small: official TEST unchanged (+3.46), NUDT 707
    +0.80 -> +1.00, design sets +-0.3 (`results/it1_jetson_context_fix.csv`).
  - 1b/1c done: empty shortlist -> skip row (never dropped / exception); failed Learn-then-Test -> `None` -> `NO_ANSWER` (answers nothing, override
    included). Unit tests `tests/pipeline/test_iteration10_blocks.py` (4 pass; T7 10 pass, T8 3 pass).
  - 1d done: manifest block `train/checks/eval_tools.py` + comparison with the headless submission (from Notion; code not on the old laptop).
    Differences: research hint = 5 km random vs submission Jetson run hint = truth; research Jetson replay caps 10 queries vs 22 s budget; PC
    replay's 27 queries include scale 0.95 (submission 18 do not); matcher confidence 0.22 (+ coarse 0.21) vs one documented value 0.21 (check
    which in the headless code); judge / override / early exit only in research.
  - 1e done: failure log (`results/it1_failure_log.csv`): official is never "never retrieved"; its loss is "retrieved but rejected" (Jetson today
    28 on TEST, judge 16); NUDT loses 4-14 per set to "never retrieved".
  - 3a done (headroom only): two plausible distant candidates are rare and almost always already answered right; a perfect tie-breaker would
    win <= 0.4 points per 60, 0 on official TRY. Not built (`It3_Targeted_Verification`).
  - 2 (scale robustness) **written, not finished:** `It2_Scale_Robustness.ipynb` + block `train/checks/scale_robust.py` (variants V0 override,
    V1 no override, V2 relative inliers vs running far-candidate level, V3 scale augmentation, V4 safety switch, V5 relative + augmentation;
    scales x0.5 ... x2; leave-one-NUDT-set-out + official TRY; PC and Jetson). The run was killed by the session end.
  - 3b/3c **written, not run:** block `train/checks/stage_cache.py` (cache canvas + full K = 50 shortlist + DINO scores per query; mixture
    selection; optional shared map-window FFT patch) and notebook `Rich_Dump_Cached.ipynb`; driver `tests/Test_Iteration_10/run_gpu.sh` (cache
    identity check on 5 official photos, then the fresh **NUDT 808** dump, then the mixture runs). Note: the old dumps store only the M = 10 kept
    peaks, not K = 50, so 3b needs these new runs.

## 5. The current task (T10 brief, from the user) and what is left

0. Finish T9: run `tests/Test_Iteration_9/run_perturbed_dumps.sh` (resumes), then `It3_Altered_Photos.ipynb`; write T9 README / REPORT (honest:
   NOT input-invariant, list what breaks it), Notion page, push.
1. Evaluation fixes: a-e **done** (above).
2. Inlier-scale robustness: run `It2_Scale_Robustness.ipynb` (CPU, ~10-20 min with 12 processes); then evaluate the variants also on the T9
   altered photos. A variant counts only with 0 wrong answers over the whole scale range; report score kept vs x1. 2a (remove the override) is
   already supported (`override=None`).
3. Accuracy experiments after 1 and 2: 3a done (headroom), 3b mixture coarse/DINO (needs `run_gpu.sh`), 3c stage caching (verify identical
   outputs vs T6 dumps on 5 photos; time; adopt the FFT sharing only if identical and memory fits).
4. Final claim: NUDT 808 dumped first (in `run_gpu.sh`), signals built, finalists pre-registered (like T8 `preregistered_t8.json`), scored once.
5. T10 README / REPORT, Notion page per iteration + summary, push.

**Why the old laptop kept dying (journal, 2026-11-14):** (1) 09:08 and 13:27 — **out of RAM** (14 GiB): the kernel OOM killer killed the
Claude desktop app (and so every background job) 3-4 min after the scale sweep started 12 parallel processes (~1.4 GB each: full signal table
plus scaled / augmented copies) next to a GPU dump. Run `It2_Scale_Robustness` with `n_jobs` <= 4 on a 14-16 GB machine (or 1 GB per process
budget), and not alongside a GPU dump. (2) 01:48 — an abrupt freeze with no OOM and no shutdown log, after machine-check (MCE hardware) errors
earlier in that boot: the known CPU fault (CPUs 10/11), not RAM or GPU. No GPU (Xid / NVRM) errors in any boot.

Order on the new machine: GPU `run_perturbed_dumps.sh` -> `run_gpu.sh` (it waits for the first); meanwhile CPU `It2_Scale_Robustness`.
Run long jobs in `tmux` / `systemd-run --user` — detached `nohup` jobs died with the Claude session twice on the old laptop.

## 6. Code map

- Blocks (`train/checks/`): `rich_dump.py`, `replay.py` (T6); `decision_signals.py`, `candidate_judge.py` (judge, `decide`, `baseline`,
  `ltt_tau`, `NO_ANSWER`, `add_context(m)`), `second_look.py` (T7); `holdout.py` (split, `load_t8_table(locked=...)`, bar frontier,
  `answer_change`, attitude helpers), `two_tier.py` (T8); `perturb.py` (T9); `eval_tools.py`, `scale_robust.py`, `stage_cache.py` (T10).
- Tests: `tests/pipeline/test_iteration{6,7,8,10}_blocks.py` (`python tests/pipeline/<file>`).
- Commit to `palash/dev` without switching branches (what was used):

```bash
export GIT_INDEX_FILE=$(mktemp -u); git read-tree palash/dev
git add -f <files>; TREE=$(git write-tree)
C=$(git commit-tree $TREE -p palash/dev -m "message"); git update-ref refs/heads/palash/dev $C; unset GIT_INDEX_FILE
git push origin palash/dev
```
On a fresh clone simply `git checkout palash/dev` and commit normally.

## 7. Notion pages written (under InterIIT_UAV_GeoLoc)

T6 summary + 8 iteration pages; T7 Iterations 1-6 + summary (with the team's review note and the T8 result appended); T8 Iterations 0-1, 2, 3, 4
+ summary. Not yet: T9, T10.

## 8. After T10: open directions

- **Answered 2026-11-14: satellite-only.** Learned components must be trained on satellite imagery only, so every NUDT-trained judge of
  T7-T10 (T8 NUDT-only judge, T10 V5 / J1 / J2) is research evidence, not shippable. Shippable now: non-learned changes (remove the 12-inlier
  override; bars). Next: a judge trained on simulated drone views made from satellite map crops, tested on NUDT / official.
- Remove the 12-inlier override in the submission; consider V5 (needs the running far-candidate level and the scale-augmented judge).
- Upstream losses dominate now: NUDT "never retrieved" 8-17 photos per 300 (coarse / DINO); official "retrieved but rejected"; T9 metadata
  failures (height +20 %, heading beyond +-10 deg) and blur / resolution (relevant above 351 m, where no data exists).
- A coarse share in the M = 10 shortlist (It7) recovered 1 NUDT photo; check on a fresh draw using the 808 cache (full K = 50 shortlist cached).
- Any new final claim needs another fresh NUDT draw (all of 101-808 are seen) or official-like photos never used.

## 9. T11 (2026-11-14 evening): satellite-only judge

`tests/Test_Iteration_11`: judges trained only on 378 drone-view photos rendered from Cartosat scene B / Pleiades Neo (block
`train/checks/sat_photos.py`). Ranking on thermal photos is nearly as good as the NUDT-trained judge (AUC 0.998 / 0.997), but the confidence is
off: only a cut of 0.999+ gives 0 wrong on the NUDT design sets, and then there is no gain. Pre-registered final on fresh NUDT 909: primary
S5z 1 wrong (base 2 PC / 4 Jetson), no gain; Jetson S1z 0 wrong at every inlier scale, +0.16. Satellite-only training gives safety, not
score, so far. NUDT 909 is now seen data (dump in `tests/Test_Iteration_10/results/rich_dump/nudt_fresh909`). Disk was full (125 GB);
~1.1 GB free after deleting regenerable caches with the user's OK.

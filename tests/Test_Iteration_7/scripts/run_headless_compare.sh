#!/usr/bin/env bash
# Chain 3 (2026-11-16): the submission pipeline (Main_Pipeline_Headless) with and without the new switches, mode full (timings per photo), hint jittered by 5 km.
# Waits for chain 2 (one heavy job at a time). Each run: tools/run_capped.sh (6G, 8 cores) and a hard time limit. Results: results/headless/<name>/details.csv.
HERE="$(cd "$(dirname "$0")/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; ROOT="$(cd "$REPO/.." && pwd)"
LOGS="$HERE/results/logs"; STATUS="$LOGS/headless_status.txt"; OUT="$HERE/results/headless"
SETS="/tmp/claude-1000/-home-voidness-Storage-Synced-Folders-InterIIT-Tech-15/325ac346-d563-455a-92b0-739fdfebd19f/scratchpad/headless_sets"
DEV="$ROOT/Submission_Files/dev_data/dataset"; STORE="$ROOT/Submission_Files/dev_store"; MODELS="$ROOT/Main_Pipeline_Headless/models"
cd "$ROOT/Main_Pipeline_Headless" || exit 1
run() {   # run <name> <dataset> <max minutes> [--set ...]
  name=$1; data=$2; limit=$3; shift 3
  echo "$(date +%H:%M:%S) start $name" >> "$STATUS"
  timeout --signal=TERM --kill-after=60 "${limit}m" "$REPO/tools/run_capped.sh" -m 6G -c 0-7 -t 6 -- "$ROOT/venv-geoloc/bin/python" ./run batch --dataset "$data" --preprocess "$STORE" --models "$MODELS" \
      --out "$OUT/$name" --mode full --hint-jitter-m 5000 "$@" > "$LOGS/headless_$name.txt" 2>&1
  code=$?; echo "$(date +%H:%M:%S) end $name exit $code (check $LOGS/headless_$name.txt)" >> "$STATUS"
}
echo "chain started $(date)" >> "$STATUS"
run smoke_clean2 "$SETS/clean60" 15 --limit 2 --set USE_ALIGNED_RESCUE=True --set LOFTR_COARSE_THRESHOLD=0.05 --set USE_ALTERED_COPIES=True
run base_clean60 "$SETS/clean60" 40
run base_invert60 "$SETS/invert60" 40
run base_blur60 "$SETS/blur60" 40
for kind in clean invert blur; do
  run copiesbar_before_${kind}60 "$SETS/${kind}60" 50 --set USE_ALTERED_COPIES=True --set ALTERED_COPIES_ORDER=before_second_pass --set ALTERED_COPY_BAR_ONLY=True
  run copiesagree_before_${kind}60 "$SETS/${kind}60" 50 --set USE_ALTERED_COPIES=True --set ALTERED_COPIES_ORDER=before_second_pass --set ALTERED_COPY_BAR_ONLY=False
  run copiesbar_after_${kind}60 "$SETS/${kind}60" 50 --set USE_ALTERED_COPIES=True --set ALTERED_COPIES_ORDER=after_second_pass --set ALTERED_COPY_BAR_ONLY=True
done
run rescue_official500 "$DEV" 120 --set USE_ALIGNED_RESCUE=True --set LOFTR_COARSE_THRESHOLD=0.05
run copies_rescue_official500 "$DEV" 150 --set USE_ALIGNED_RESCUE=True --set LOFTR_COARSE_THRESHOLD=0.05 --set USE_ALTERED_COPIES=True --set ALTERED_COPY_BAR_ONLY=True
echo "chain finished $(date)" >> "$STATUS"

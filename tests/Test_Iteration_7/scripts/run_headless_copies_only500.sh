#!/usr/bin/env bash
# Chain 8 (2026-11-16): altered copies only (bar-only, before second pass) on all 500 official photos.
HERE="$(cd "$(dirname "$0")/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; ROOT="$(cd "$REPO/.." && pwd)"
LOGS="$HERE/results/logs"; STATUS="$LOGS/headless4_status.txt"; OUT="$HERE/results/headless"
SETS="/tmp/claude-1000/-home-voidness-Storage-Synced-Folders-InterIIT-Tech-15/325ac346-d563-455a-92b0-739fdfebd19f/scratchpad/headless_sets"
STORE="$ROOT/Submission_Files/dev_store"; MODELS="$ROOT/Main_Pipeline_Headless/models"

cd "$ROOT/Main_Pipeline_Headless" || exit 1
run() {   # run <name> <dataset> <max minutes> [--set ...]
  name=$1; data=$2; limit=$3; shift 3
  echo "$(date +%H:%M:%S) start $name" >> "$STATUS"
  timeout --signal=TERM --kill-after=60 "${limit}m" "$REPO/tools/run_capped.sh" -m 6G -c 0-7 -t 6 -- "$ROOT/venv-geoloc/bin/python" ./run batch --dataset "$data" --preprocess "$STORE" --models "$MODELS" \
      --out "$OUT/$name" --mode full --hint-jitter-m 5000 "$@" > "$LOGS/headless4_$name.txt" 2>&1
  code=$?; echo "$(date +%H:%M:%S) end $name exit $code" >> "$STATUS"
}
echo "chain started $(date)" >> "$STATUS"
run copies_official500 "$ROOT/Submission_Files/dev_data/dataset" 90 --set USE_ALTERED_COPIES=True
echo "chain finished $(date)" >> "$STATUS"

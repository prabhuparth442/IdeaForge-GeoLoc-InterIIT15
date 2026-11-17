#!/usr/bin/env bash
# Chain 5 (2026-11-16): the altered-copy runs of chain 3 again (they failed at once: a string value in --set needs its own quotes). Waits for chain 4 (one heavy job at a time).
HERE="$(cd "$(dirname "$0")/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; ROOT="$(cd "$REPO/.." && pwd)"
LOGS="$HERE/results/logs"; STATUS="$LOGS/headless2_status.txt"; OUT="$HERE/results/headless"
SETS="/tmp/claude-1000/-home-voidness-Storage-Synced-Folders-InterIIT-Tech-15/325ac346-d563-455a-92b0-739fdfebd19f/scratchpad/headless_sets"
STORE="$ROOT/Submission_Files/dev_store"; MODELS="$ROOT/Main_Pipeline_Headless/models"
until grep -q "chain finished" "$LOGS/followup3_status.txt" 2>/dev/null; do sleep 60; done
cd "$ROOT/Main_Pipeline_Headless" || exit 1
run() {   # run <name> <dataset> <max minutes> [--set ...]
  name=$1; data=$2; limit=$3; shift 3
  echo "$(date +%H:%M:%S) start $name" >> "$STATUS"
  timeout --signal=TERM --kill-after=60 "${limit}m" "$REPO/tools/run_capped.sh" -m 6G -c 0-7 -t 6 -- "$ROOT/venv-geoloc/bin/python" ./run batch --dataset "$data" --preprocess "$STORE" --models "$MODELS" \
      --out "$OUT/$name" --mode full --hint-jitter-m 5000 "$@" > "$LOGS/headless2_$name.txt" 2>&1
  code=$?; echo "$(date +%H:%M:%S) end $name exit $code" >> "$STATUS"
}
echo "chain started $(date)" >> "$STATUS"
for kind in clean invert blur; do
  run copiesbar_before_${kind}60 "$SETS/${kind}60" 50 --set USE_ALTERED_COPIES=True --set 'ALTERED_COPIES_ORDER="before_second_pass"' --set ALTERED_COPY_BAR_ONLY=True
  run copiesagree_before_${kind}60 "$SETS/${kind}60" 50 --set USE_ALTERED_COPIES=True --set 'ALTERED_COPIES_ORDER="before_second_pass"' --set ALTERED_COPY_BAR_ONLY=False
  run copiesbar_after_${kind}60 "$SETS/${kind}60" 50 --set USE_ALTERED_COPIES=True --set 'ALTERED_COPIES_ORDER="after_second_pass"' --set ALTERED_COPY_BAR_ONLY=True
done
echo "chain finished $(date)" >> "$STATUS"

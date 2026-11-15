#!/usr/bin/env bash
# waits for the ideas 9/1 run, then idea 5 on the valid-split draw (GPU, one job at a time, memory-capped)
HERE="$(cd "$(dirname "$0")/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; V="$REPO/../venv-geoloc/bin"
cd "$HERE/notebooks" || exit 1
until grep -q "^end exit" "$HERE/results/logs/run_ideas_9_1.out"; do sleep 20; done
"$REPO/tools/run_capped.sh" -m 6G -c 0-7 -t 6 -- env GATE_DATA=nudt NUDT_SPLIT=valid POOL_PHOTOS=1425 TAG=nudt_valid1212 SEED_IDS_FILE="$HERE/results/never_retrieved_valid1212.csv" CONTROL_N=10 \
  "$V/jupyter" nbconvert --to notebook --execute Idea5_Deep_Shortlist_DINO.ipynb --output Idea5_Deep_Shortlist_DINO_executed.ipynb --ExecutePreprocessor.timeout=-1 > "$HERE/results/logs/log_idea5.txt" 2>&1
echo "end exit $?"

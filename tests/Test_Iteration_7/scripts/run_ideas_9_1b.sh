#!/usr/bin/env bash
HERE="$(cd "$(dirname "$0")/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; V="$REPO/../venv-geoloc/bin"
cd "$HERE/notebooks" || exit 1
until grep -q "^end exit" "$HERE/results/logs/run_idea1.out"; do sleep 20; done
"$REPO/tools/run_capped.sh" -m 6G -c 0-7 -t 6 -- env GATE_DATA=official "$V/jupyter" nbconvert --to notebook --execute Ideas_9_1_Finer_Level_Pose_Feedback.ipynb \
  --output Ideas_9_1_Finer_Level_Pose_Feedback_executed.ipynb --ExecutePreprocessor.timeout=-1 > "$HERE/results/logs/log_ideas_9_1_rerun.txt" 2>&1
echo "end exit $?"

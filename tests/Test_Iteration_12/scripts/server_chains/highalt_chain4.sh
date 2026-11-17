#!/usr/bin/env bash
# Fable's change 1 (photo cleaning) on the 332 working photos of the high-altitude set, one run at a time, no copies, bar 8, nothing else changed.
# V1 vignette removed (photos from 800 m); V2 + local normalisation 15 px; V3 V1 + mask of the dark hook (x 303-312, y 225-255);
# V4 V2 + the hook + the dark bottom-left corner (x 0-63, y 186-255).
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_highalt4.txt
while pgrep -f "ipykernel_launcher" >/dev/null || pgrep -f "[M]ega_Dump_official150_invert" >/dev/null; do sleep 20; done
cd $W/Main_Pipeline_Headless
run() { name=$1; shift
  echo "$(date +%m-%d_%H:%M:%S) start $name" >> $S
  $PY ./run batch --dataset $H/dataset_dev --preprocess $H/store --models $W/Main_Pipeline_Headless/models --out $W/out_dev_$name --mode full --hint-jitter-m 5000 --set "LEVELS=[3,4,5,6,7,8,9,10,11]" "$@" > $W/log_dev_$name.txt 2>&1
  echo "$(date +%m-%d_%H:%M:%S) end $name exit $?" >> $S; }
HOOK="(301,223,314,256)"; CORNER="(0,186,64,256)"
run v1_flat --set FLAT_FIELD_FROM_HEIGHT_M=800.0
run v2_flat_local --set FLAT_FIELD_FROM_HEIGHT_M=800.0 --set LOCAL_NORM_SIGMA=15.0
run v3_flat_hook --set FLAT_FIELD_FROM_HEIGHT_M=800.0 --set "PHOTO_MASK_RECTS=[$HOOK]"
run v4_flat_local_masks --set FLAT_FIELD_FROM_HEIGHT_M=800.0 --set LOCAL_NORM_SIGMA=15.0 --set "PHOTO_MASK_RECTS=[$HOOK,$CORNER]"
echo "$(date +%m-%d_%H:%M:%S) CHAIN4 DONE" >> $S

#!/usr/bin/env bash
# Second-matcher fallback runs (iteration 42), frozen base = XoFTR from 800 m. usage: fallback_chain.sh dev|official|nudt NAME "extra --set args"
# dev = the 332 working photos of the new set, official = old official 500, nudt = NUDT in headless format (300).
# Own status file and output folders (out_fb_SET_NAME); nothing existing is overwritten.
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_fallback.txt
cd $W/Main_Pipeline_Headless
echo "$(date +%m-%d_%H:%M:%S) start $1_$2" >> $S
case $1 in
  dev) $PY ./run batch --dataset $H/dataset_dev --preprocess $H/store --models models --out ../out_fb_$1_$2 --mode full --hint-jitter-m 5000 \
        --set "LEVELS=[3,4,5,6,7,8,9,10,11]" --set XOFTR_FROM_HEIGHT_M=800 $3 > ../log_fb_$1_$2.txt 2>&1 ;;
  official) $PY ./run batch --dataset $W/Submission_Files/dev_data/dataset --preprocess $W/Submission_Files/dev_store --models models --out ../out_fb_$1_$2 --mode full --hint-jitter-m 5000 \
        --set XOFTR_FROM_HEIGHT_M=800 $3 > ../log_fb_$1_$2.txt 2>&1 ;;
  nudt) $PY ./run batch --dataset $W/nudt_hl/dataset --preprocess $W/nudt_hl/store --models models --out ../out_fb_$1_$2 --mode full --hint-jitter-m 5000 \
        --set "LEVELS=[3,4,5]" --set 'GEOMETRY={"attitude": False, "pitch_sign": -1, "roll_sign": -1, "yaw_offset": 0.0, "scale": 0.8, "height_offset": 0.0}' \
        --set XOFTR_FROM_HEIGHT_M=800 $3 > ../log_fb_$1_$2.txt 2>&1 ;;
esac
echo "$(date +%m-%d_%H:%M:%S) end $1_$2 exit $?" >> $S

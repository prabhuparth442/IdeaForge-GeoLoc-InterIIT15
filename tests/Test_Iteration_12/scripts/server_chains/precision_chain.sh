#!/usr/bin/env bash
# Precision variants on the 332 working photos (iteration 40): XoFTR from 800 m plus a different final fit.
# usage: precision_chain.sh NAME "extra --set args"
# Writes its own status file and output folder (out_prec_NAME); nothing existing is overwritten.
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_precision.txt
cd $W/Main_Pipeline_Headless
echo "$(date +%m-%d_%H:%M:%S) start $1" >> $S
$PY ./run batch --dataset $H/dataset_dev --preprocess $H/store --models models --out ../out_prec_$1 --mode full --hint-jitter-m 5000 \
  --set "LEVELS=[3,4,5,6,7,8,9,10,11]" --set XOFTR_FROM_HEIGHT_M=800 $2 > ../log_prec_$1.txt 2>&1
echo "$(date +%m-%d_%H:%M:%S) end $1 exit $?" >> $S

#!/usr/bin/env bash
# Pipeline on the synthetic photos of the geometry audit (iteration 40). usage: synth_chain.sh exact|noisy
# The synthetic photos are rendered from the map store by notebooks/Synthetic_Geometry_Audit.ipynb. Frozen configuration (XoFTR from 800 m).
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_synth.txt
cd $W/Main_Pipeline_Headless
echo "$(date +%m-%d_%H:%M:%S) start $1" >> $S
$PY ./run batch --dataset $W/synth_geo/$1 --preprocess $H/store --models models --out ../out_synth_$1 --mode full --hint-jitter-m 5000 \
  --set "LEVELS=[3,4,5,6,7,8,9,10,11]" --set XOFTR_FROM_HEIGHT_M=800 > ../log_synth_$1.txt 2>&1
echo "$(date +%m-%d_%H:%M:%S) end $1 exit $?" >> $S

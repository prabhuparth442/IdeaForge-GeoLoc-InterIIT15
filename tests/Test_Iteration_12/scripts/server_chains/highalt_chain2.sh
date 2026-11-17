#!/usr/bin/env bash
# high-altitude follow-ups (two GPU jobs at most: the official dump is the other one):
#  A) copies + bar 4 at levels 9-11, all 500 photos
#  B) the 230 photos above 1000 m only: wider second pass (scales 0.7-1.3, headings -10..10) with bar 4, to see whether a scale/heading error is what the failing photos have
W=/mnt/e/IdeaForge/work; H=/mnt/e/IdeaForge/datasets/highalt_v2; PY=~/training/ideaforge/venv-geoloc/bin/python; S=$W/status_highalt.txt
until grep -q "end highalt_bar4_lvl9up" $S 2>/dev/null; do sleep 30; done
cd $W/Main_Pipeline_Headless
mkdir -p $H/dataset_above1000/images
$PY - <<"E"
import csv, shutil
H="/mnt/e/IdeaForge/datasets/highalt_v2"
rows=list(csv.DictReader(open(H+"/dataset/metadata.csv")))
keep=[r for r in rows if float(r["alt"])>=1000]
w=csv.DictWriter(open(H+"/dataset_above1000/metadata.csv","w",newline=""), fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(keep)
for r in keep: shutil.copy(H+"/dataset/images/"+r["filename"], H+"/dataset_above1000/images/"+r["filename"])
print(len(keep))
E
run() { name=$1; data=$2; shift 2
  echo "$(date +%m-%d_%H:%M:%S) start $name" >> $S
  $PY ./run batch --dataset $data --preprocess $H/store --models $W/Main_Pipeline_Headless/models --out $W/out_$name --mode full --hint-jitter-m 5000 --set "LEVELS=[3,4,5,6,7,8,9,10,11]" --set "BAR_BY_LEVEL={9:4,10:4,11:4}" "$@" > $W/log_$name.txt 2>&1
  echo "$(date +%m-%d_%H:%M:%S) end $name exit $?" >> $S; }
run highalt_copies_bar4 $H/dataset --set USE_ALTERED_COPIES=True
run above1000_bar4_ref $H/dataset_above1000
run above1000_wide_second $H/dataset_above1000 --set "SECOND_PASS_HEADINGS=[-10.0,-5.0,0.0,5.0,10.0]" --set "SECOND_PASS_SCALES=[0.7,0.8,0.9,1.0,1.1,1.2,1.3]"
echo "$(date +%m-%d_%H:%M:%S) HIGHALT CHAIN2 DONE" >> $S

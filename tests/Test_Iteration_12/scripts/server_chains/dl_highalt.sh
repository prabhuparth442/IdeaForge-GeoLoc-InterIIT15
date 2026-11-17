#!/usr/bin/env bash
# downloads the five files of the shared Drive folder with curl (resumable); each file retried until complete
D=/mnt/e/IdeaForge/datasets/highalt_v2; S=/mnt/e/IdeaForge/work/status_download.txt
mkdir -p $D/sat_imagery
get() { id=$1; out=$2
  [ -f $out.done ] && return 0
  for i in $(seq 1 30); do
    curl -sL -C - --retry 3 -o $out "https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t"
    code=$?; if [ $code = 0 ] && ! head -c 200 $out | grep -qi "<html"; then touch $out.done; echo "$(date +%H:%M:%S) done $out $(stat -c %s $out)" >> $S; return 0; fi
    echo "$(date +%H:%M:%S) retry $i $out code $code" >> $S; sleep 20
  done; }
get 1QJSbQm9viR6uo34h8iMvb_7qoyK9D0MO $D/d1.zip
get 1j4r7eO_knF93wE79TexTwezhyKq4TNnk $D/sat_imagery/2684532111.zip
get 1KKfxwTSOFCV-xhZ9y3aUjstwj35zRySr $D/sat_imagery/2684532121.zip
get 1SXrzIdaxiZ59g6QkNvnxj32tjFuRnk-R $D/sat_imagery/2684532171.zip
get 1hKNAHotISN0TCkt-M0RvTuF5uwT8kvjy $D/sat_imagery/268453231.zip
echo "ALL DOWNLOADS DONE $(date)" >> $S

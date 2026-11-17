#!/usr/bin/env bash
# wait for the second map copy to end, then copy the laptop's high-altitude zips to the server (size-only compare: the first zip is already there)
while pgrep -f "rsync.*Dataset_Official_2/268234921" >/dev/null; do sleep 30; done
SRC="/home/voidness/Storage/Synced Folders/InterIIT Tech 15/Code_Testing/dataset/Dataset_Official_3/"
DST=workstation@100.105.160.107:/mnt/e/IdeaForge/datasets/highalt_v2/
ssh -o BatchMode=yes workstation@100.105.160.107 'pkill -f "[d]l_highalt.sh"; pkill -f "[c]url.*highalt_v2"' 2>/dev/null
for i in 1 2 3 4 5; do
  rsync -rt --no-perms --size-only --partial --info=stats1 -e ssh --exclude='*.part' "$SRC" "$DST" && { echo "copy exit 0"; exit 0; }
  echo "retry $i"; sleep 30
done
echo "copy exit 1"

#!/bin/bash
# Incremental durability sync: pull each box's results JSON to the Mac (and Pi).
# Runs every ~5 min. Small files (a few MB), so the pull is cheap.
REPO=/Users/tvgdeena/nova-nanobody/scoring_backfill
MAC_DURABLE=/Users/tvgdeena/nova-nanobody/data/backfill_results
PI=thomasg@raspberrypi
PI_DIR=/home/thomasg/backfill_results
SSHO="-o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
BOXES=( "0 53449398 ssh2.vast.ai 13756" "1 53449401 ssh4.vast.ai 18324"
        "2 53449408 211.72.13.201 40157" "3 53449412 ssh1.vast.ai 25066"
        "4 53449418 ssh3.vast.ai 34196" "5 53449420 ssh6.vast.ai 27242"
        "6 53449423 ssh3.vast.ai 31808" "7 53449426 ssh8.vast.ai 36372"
        "8 53449432 217.138.104.229 10427" "9 53449435 ssh3.vast.ai 20070" )

mkdir -p $REPO/results $MAC_DURABLE
ssh $SSHO $PI "mkdir -p $PI_DIR" 2>/dev/null
for b in "${BOXES[@]}"; do
  set -- $b; idx=$1; id=$2; host=$3; port=$4
  scp $SSHO -P $port root@$host:/root/results/box$idx.json $REPO/results/box$idx.json >/dev/null 2>&1
  [ -f $REPO/results/box$idx.json ] && cp $REPO/results/box$idx.json $MAC_DURABLE/box$idx.json
done
# push to Pi (durable off-box insurance)
scp $SSHO $REPO/results/box*.json $PI:$PI_DIR/ >/dev/null 2>&1
echo "$(date +%H:%M:%S) SYNC_DONE boxes=$(ls $REPO/results/box*.json 2>/dev/null | wc -l | tr -d ' ')"

#!/bin/bash
REPO=/Users/tvgdeena/nova-nanobody/scoring_backfill
MAC_DURABLE=/Users/tvgdeena/nova-nanobody/data/backfill_children_results
PI=thomasg@raspberrypi
PI_DIR=/home/thomasg/backfill_children_results
SSHO="-o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
BOXES=( "b0 118.163.199.123 12945" "b1 60.250.87.179 58248" "b2 185.229.26.83 41335"
        "b3 118.163.199.123 56759" "b4 ssh4.vast.ai 18324" "b5 ssh1.vast.ai 25066"
        "b6 ssh3.vast.ai 34196" "b7 ssh6.vast.ai 27242" "b8 ssh3.vast.ai 20070"
        "b9 211.72.13.201 41764" "b10 31.13.223.140 34721" )
mkdir -p $REPO/results_children9 $MAC_DURABLE
ssh $SSHO $PI "mkdir -p $PI_DIR" 2>/dev/null
for b in "${BOXES[@]}"; do
  set -- $b; sh=$1; host=$2; port=$3
  scp $SSHO -P $port root@$host:/root/results_children9/$sh.json $REPO/results_children9/$sh.json >/dev/null 2>&1
  scp $SSHO -P $port root@$host:/root/results_children9/steal.json $REPO/results_children9/steal_$sh.json >/dev/null 2>&1
  [ -f $REPO/results_children9/$sh.json ] && cp $REPO/results_children9/$sh.json $MAC_DURABLE/$sh.json
done
scp $SSHO $REPO/results_children9/*.json $PI:$PI_DIR/ >/dev/null 2>&1
echo "$(date +%H:%M:%S) CHILD_SYNC_DONE boxes=$(ls $REPO/results_children9/*.json 2>/dev/null | wc -l | tr -d ' ')"

#!/bin/bash
# Runs ON a box: kill stale scorers and relaunch the merged children worker.
IDX=$1
pkill -9 -f "score_worker" 2>/dev/null
pkill -9 -f "fast_score_batch" 2>/dev/null
sleep 2
mkdir -p /root/results_children /root/bgen_shards_children
nohup python3 /root/score_worker.py --shard /root/box$IDX.json --results /root/results_children/box$IDX.json --work-root /root/bgen_shards_children --batch-size 256 --gpus 0,1,2,3,4,5,6,7 > /root/results_children/box$IDX.worker.log 2>&1 < /dev/null &
echo "RELAUNCHED box$IDX"

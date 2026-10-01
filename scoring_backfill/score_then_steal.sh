#!/bin/bash
# Runs ON a box: score the assigned shard, then steal remaining work until empty.
SHARD=$1
python3 /root/score_worker.py --shard /root/$SHARD.json --results /root/results_children9/$SHARD.json \
  --work-root /root/bgen_shards_children9 --batch-size 256 --gpus 0,1,2,3,4,5,6,7
python3 /root/work_steal.py --gpus 0,1,2,3,4,5,6,7 --batch-size 256 --quiet-polls 10

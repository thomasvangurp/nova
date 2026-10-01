#!/bin/bash
# Deploy a shard + worker to one box and launch the scoring worker in the background.
SHARD_IDX=$1; HOST=$2; PORT=$3
SSHO="-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
REPO=/Users/tvgdeena/nova-nanobody/scoring_backfill
BOXNAME=box$SHARD_IDX
scp $SSHO -P $PORT $REPO/shards/$BOXNAME.json $REPO/score_worker.py root@$HOST:/root/ 2>&1
ssh $SSHO -p $PORT root@$HOST "mkdir -p /root/results /root/bgen_shards && nohup python3 /root/score_worker.py --shard /root/$BOXNAME.json --results /root/results/$BOXNAME.json --work-root /root/bgen_shards --batch-size 256 --gpus 0,1,2,3,4,5,6,7 > /root/results/$BOXNAME.worker.log 2>&1 & echo WORKER_LAUNCHED" 2>&1

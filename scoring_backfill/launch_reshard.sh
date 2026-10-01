#!/bin/bash
# Runs ON a box: launch the re-sharded children worker + work-stealing tail.
SHARD=$1
mkdir -p /root/results_children9 /root/bgen_shards_children9
nohup bash /root/score_then_steal.sh "$SHARD" > /root/results_children9/$SHARD.worker.log 2>&1 < /dev/null &
echo "LAUNCHED $SHARD"

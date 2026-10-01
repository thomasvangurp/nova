#!/bin/bash
# Deploy shards + worker and launch scoring on all boxes (resumable, idempotent).
REPO=/Users/tvgdeena/nova-nanobody/scoring_backfill
declare -a BOXES=(
  "0 53449398 ssh2.vast.ai 13756"
  "1 53449401 ssh4.vast.ai 18324"
  "2 53449408 211.72.13.201 40157"
  "3 53449412 ssh1.vast.ai 25066"
  "4 53449418 ssh3.vast.ai 34196"
  "5 53449420 ssh6.vast.ai 27242"
  "6 53449423 ssh3.vast.ai 31808"
  "7 53449426 ssh8.vast.ai 36372"
  "8 53449432 217.138.104.229 10427"
  "9 53449435 ssh3.vast.ai 20070"
)
for b in "${BOXES[@]}"; do
  set -- $b; IDX=$1; ID=$2; HOST=$3; PORT=$4
  SSHO="-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
  scp $SSHO -P $PORT $REPO/shards/box$IDX.json $REPO/score_worker.py root@$HOST:/root/ >/dev/null 2>&1
  ssh $SSHO -p $PORT root@$HOST "mkdir -p /root/results /root/bgen_shards; nohup python3 /root/score_worker.py --shard /root/box$IDX.json --results /root/results/box$IDX.json --work-root /root/bgen_shards --batch-size 256 --gpus 0,1,2,3,4,5,6,7 > /root/results/box$IDX.worker.log 2>&1 < /dev/null & echo LAUNCHED box$IDX" 2>&1 | tail -1
done
echo "START_SCORING_DONE"

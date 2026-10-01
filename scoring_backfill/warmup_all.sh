#!/bin/bash
declare -a BOXES=(
  "53449398 ssh2.vast.ai 13756"
  "53449401 ssh4.vast.ai 18324"
  "53449408 211.72.13.201 40157"
  "53449412 ssh1.vast.ai 25066"
  "53449418 ssh3.vast.ai 34196"
  "53449420 ssh6.vast.ai 27242"
  "53449423 ssh3.vast.ai 31808"
  "53449426 ssh8.vast.ai 36372"
  "53449432 217.138.104.229 10427"
  "53449435 ssh3.vast.ai 20070"
)
for b in "${BOXES[@]}"; do
  set -- $b; ID=$1; HOST=$2; PORT=$3
  (
    SSHO="-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
    scp $SSHO -P $PORT warmup_hf.sh root@$HOST:/root/ >/dev/null 2>&1
    ssh $SSHO -p $PORT root@$HOST "bash /root/warmup_hf.sh" > /tmp/warmup_$ID.log 2>&1
    echo "WARMUP_DONE $ID rc=$?" >> /tmp/warmup_$ID.log
  ) &
done
wait
echo "WARMUP_ALL_DONE"

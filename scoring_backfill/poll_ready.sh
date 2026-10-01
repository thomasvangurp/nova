#!/bin/bash
# Poll all 10 boxes until every one is provisioned (bgen env + fastscore + pixi present).
BOXES=(
  "53449398 ssh2.vast.ai 13756"
  "53449401 ssh4.vast.ai 18324"
  "53449408 ssh1.vast.ai 38978"
  "53449412 ssh1.vast.ai 25066"
  "53449418 ssh3.vast.ai 34196"
  "53449420 ssh6.vast.ai 27242"
  "53449423 ssh3.vast.ai 31808"
  "53449426 ssh8.vast.ai 36372"
  "53449432 217.138.104.229 10427"
  "53449435 ssh3.vast.ai 20070"
)
check_one() {
  local host=$1 port=$2
  ssh -n -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $port root@$host \
    '[ -x /root/.pixi/envs/bgen/bin/python ] && [ -f /root/fastscore/pinned-530f5dcbc54d/fast_score_batch.py ] && [ -x /usr/local/bin/pixi ] && [ -f /root/pixi.toml ] && echo READY' 2>/dev/null
}
while true; do
  all=1
  status=""
  for b in "${BOXES[@]}"; do
    set -- $b; id=$1; host=$2; port=$3
    r=$(check_one "$host" "$port")
    if [ "$r" = "READY" ]; then
      status="$status $id:Y"
    else
      status="$status $id:N"
      all=0
    fi
  done
  if [ $all -eq 1 ]; then
    echo "ALL_10_READY"
    exit 0
  fi
  echo "$(date +%H:%M:%S)$status"
  sleep 30
done

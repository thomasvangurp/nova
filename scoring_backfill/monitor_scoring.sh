#!/bin/bash
# Emit per-box scoring progress; exit when all scoring boxes are complete.
BOXES=( "3 53449412 ssh1.vast.ai 25066" "4 53449418 ssh3.vast.ai 34196"
        "5 53449420 ssh6.vast.ai 27242" "6 53449423 ssh3.vast.ai 31808"
        "7 53449426 ssh8.vast.ai 36372" "8 53449432 217.138.104.229 10427"
        "9 53449435 ssh3.vast.ai 20070" "2 53449408 211.72.13.201 40157" )
declare -A last
while true; do
  all_done=1; line=""
  for b in "${BOXES[@]}"; do
    set -- $b; idx=$1; id=$2; host=$3; port=$4
    c=$(ssh -n -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $port root@$host \
      'python3 -c "import json,os; p=\"/root/results/box'$idx'.json\"; d=json.load(open(p)) if os.path.exists(p) else {}; print(sum(1 for e in d.values() if all(f\"p{i}\" in e for i in range(1,9))))"' 2>/dev/null)
    c=${c:-ERR}
    err=$(ssh -n -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $port root@$host \
      'grep -cE "FAILED|Traceback|MERGE LOST" /root/results/box'$idx'.worker.log 2>/dev/null' 2>/dev/null)
    line="$line box$idx=$c(err=${err:-0})"
    [ "$c" != "1226" ] && [ "$c" != "1225" ] && all_done=0
  done
  echo "$(date +%H:%M:%S)$line"
  [ $all_done -eq 1 ] && echo "ALL_SCORING_COMPLETE" && exit 0
  sleep 90
done

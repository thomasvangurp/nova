#!/bin/bash
BOXES=( "1 53449401 ssh4.vast.ai 18324 1226" "4 53449418 ssh3.vast.ai 34196 1226" "8 53449432 217.138.104.229 10427 1225" )
while true; do
  all_done=1; line=""
  for b in "${BOXES[@]}"; do
    set -- $b; idx=$1; id=$2; host=$3; port=$4; tgt=$5
    c=$(ssh -n -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $port root@$host \
      "python3 -c \"import json,os;d=json.load(open('/root/results/box$idx.json')) if os.path.exists('/root/results/box$idx.json') else {};print(sum(1 for e in d.values() if all(f'p{i}' in e for i in range(1,9))))\" 2>/dev/null" 2>/dev/null)
    c=${c:-NA}
    line="$line box$idx=$c/$tgt"
    [ "$c" = "$tgt" ] || all_done=0
  done
  echo "$(date +%H:%M:%S)$line"
  [ $all_done -eq 1 ] && echo "RESUME_COMPLETE" && exit 0
  sleep 120
done

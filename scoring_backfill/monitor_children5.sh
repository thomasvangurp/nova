#!/bin/bash
BOXES=( "1 53449401 ssh4.vast.ai 18324 6290" "3 53449412 ssh1.vast.ai 25066 6290"
        "4 53449418 ssh3.vast.ai 34196 6289" "5 53449420 ssh6.vast.ai 27242 6288"
        "9 53449435 ssh3.vast.ai 20070 6288" )
while true; do
  all_done=1; line=""
  for b in "${BOXES[@]}"; do
    set -- $b; idx=$1; id=$2; host=$3; port=$4; tgt=$5
    c=$(ssh -n -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $port root@$host \
      "python3 -c \"import json,os;d=json.load(open('/root/results_children/box$idx.json')) if os.path.exists('/root/results_children/box$idx.json') else {};print(sum(1 for e in d.values() if all(f'p{i}' in e for i in range(1,9))))\" 2>/dev/null" 2>/dev/null)
    c=${c:-NA}
    line="$line box$idx=$c/$tgt"
    [ "$c" = "$tgt" ] || all_done=0
  done
  echo "$(date +%H:%M:%S)$line"
  [ $all_done -eq 1 ] && echo "CHILDREN_COMPLETE" && exit 0
  sleep 180
done

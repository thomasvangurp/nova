#!/bin/bash
BOXES=( "0 53449398 ssh2.vast.ai 13756 3145" "1 53449401 ssh4.vast.ai 18324 3145"
        "2 53449408 211.72.13.201 40157 3145" "3 53449412 ssh1.vast.ai 25066 3145"
        "4 53449418 ssh3.vast.ai 34196 3145" "5 53449420 ssh6.vast.ai 27242 3144"
        "6 53449423 ssh3.vast.ai 31808 3144" "7 53449426 ssh8.vast.ai 36372 3144"
        "8 53449432 217.138.104.229 10427 3144" "9 53449435 ssh3.vast.ai 20070 3144" )
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
  sleep 120
done

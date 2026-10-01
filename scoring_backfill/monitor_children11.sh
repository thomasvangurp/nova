#!/bin/bash
BOXES=( "b0 118.163.199.123 12945 2859" "b1 60.250.87.179 58248 2859"
        "b2 185.229.26.83 41335 2859" "b3 118.163.199.123 56759 2859"
        "b4 ssh4.vast.ai 18324 2859" "b5 ssh1.vast.ai 25066 2859"
        "b6 ssh3.vast.ai 34196 2859" "b7 ssh6.vast.ai 27242 2858"
        "b8 ssh3.vast.ai 20070 2858" "b9 211.72.13.201 41764 2858"
        "b10 31.13.223.140 34721 2858" )
while true; do
  all_done=1; line=""
  for b in "${BOXES[@]}"; do
    set -- $b; sh=$1; host=$2; port=$3; tgt=$4
    c=$(ssh -n -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $port root@$host \
      "python3 -c \"import json,os;d=json.load(open('/root/results_children9/$sh.json')) if os.path.exists('/root/results_children9/$sh.json') else {};print(sum(1 for e in d.values() if all(f'p{i}' in e for i in range(1,9))))\" 2>/dev/null" 2>/dev/null)
    c=${c:-NA}
    line="$line $sh=$c/$tgt"
    [ "$c" = "$tgt" ] || all_done=0
  done
  echo "$(date +%H:%M:%S)$line"
  [ $all_done -eq 1 ] && echo "CHILDREN_COMPLETE" && exit 0
  sleep 180
done

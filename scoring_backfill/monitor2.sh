#!/bin/bash
BOXES=( "0 53449398 ssh2.vast.ai 13756" "1 53449401 ssh4.vast.ai 18324"
        "2 53449408 211.72.13.201 40157" "3 53449412 ssh1.vast.ai 25066"
        "4 53449418 ssh3.vast.ai 34196" "5 53449420 ssh6.vast.ai 27242"
        "6 53449423 ssh3.vast.ai 31808" "7 53449426 ssh8.vast.ai 36372"
        "8 53449432 217.138.104.229 10427" "9 53449435 ssh3.vast.ai 20070" )
while true; do
  all_done=1; line=""
  for b in "${BOXES[@]}"; do
    set -- $b; idx=$1; id=$2; host=$3; port=$4
    info=$(ssh -n -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $port root@$host \
      "python3 -c \"import json,os;d=json.load(open('/root/results/box$idx.json')) if os.path.exists('/root/results/box$idx.json') else {};c=sum(1 for e in d.values() if all(f'p{i}' in e for i in range(1,9)));t=len(d);print(c,t)\" 2>/dev/null" 2>/dev/null)
    if [ -n "$info" ] && [ "$info" != " " ]; then
      c=$(echo $info | awk '{print $1}'); t=$(echo $info | awk '{print $2}')
      line="$line box$idx=${c}/${t}"
      # target complete: 1226 for idx0-3, 1225 for idx4-9
      if [ "$idx" -le 3 ]; then tgt=1226; else tgt=1225; fi
      [ "$c" = "$tgt" ] || all_done=0
    else
      line="$line box$idx=NA"
      all_done=0
    fi
  done
  echo "$(date +%H:%M:%S)$line"
  [ $all_done -eq 1 ] && echo "ALL_COMPLETE" && exit 0
  sleep 120
done

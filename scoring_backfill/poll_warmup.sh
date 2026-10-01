#!/bin/bash
BOXES=(
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
check() {
  ssh -n -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $2 root@$1 \
    's=$(du -sm /root/.cache/huggingface/hub/models--boltzgen--boltzgen-1 2>/dev/null | awk "{print \$1}"); [ "${s:-0}" -ge 5000 ] && echo WARM' 2>/dev/null
}
while true; do
  all=1; st=""
  for b in "${BOXES[@]}"; do
    set -- $b; id=$1; h=$2; p=$3
    r=$(check "$h" "$p")
    if [ "$r" = "WARM" ]; then st="$st $id:Y"; else st="$st $id:N"; all=0; fi
  done
  if [ $all -eq 1 ]; then echo "ALL_WARM"; exit 0; fi
  echo "$(date +%H:%M:%S)$st"
  sleep 45
done

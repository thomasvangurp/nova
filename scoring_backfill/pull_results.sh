#!/bin/bash
REPO=/Users/tvgdeena/nova-nanobody/scoring_backfill
mkdir -p $REPO/results
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
  scp -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $PORT root@$HOST:/root/results/box$IDX.json $REPO/results/box$IDX.json >/dev/null 2>&1
  echo "box$IDX: $(python3 -c "import json;d=json.load(open('$REPO/results/box$IDX.json'));print(len(d),'seqs,',sum(1 for e in d.values() if all(f'p{i}' in e for i in range(1,9))),'complete')" 2>/dev/null || echo 'no results yet')"
done

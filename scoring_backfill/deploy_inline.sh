#!/bin/bash
# Deploy the inline-analysis scorer to all 10 boxes (replaces the deadlocking mp version).
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
  scp -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $PORT \
    /Users/tvgdeena/nova-nanobody/scoring_backfill/fast_score_batch_inline.py \
    root@$HOST:/root/fastscore/pinned-530f5dcbc54d/fast_score_batch.py >/dev/null 2>&1 &
done
wait
echo "INLINE_DEPLOYED_ALL"

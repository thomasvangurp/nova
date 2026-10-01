#!/bin/bash
# Launch provisioning for all 10 scoring boxes in parallel.
cd /Users/tvgdeena/nova-nanobody/scoring_backfill
declare -a BOXES=(
  "53449398 ssh2.vast.ai 13756"
  "53449401 ssh4.vast.ai 18324"
  "53449408 ssh1.vast.ai 38978"
  "53449412 ssh1.vast.ai 25066"
  "53449418 ssh3.vast.ai 34196"
  "53449420 ssh6.vast.ai 27242"
  "53449423 ssh3.vast.ai 31808"
  "53449426 ssh8.vast.ai 36372"
  "53449432 ssh6.vast.ai 15504"
  "53449435 ssh3.vast.ai 20070"
)
pids=()
for b in "${BOXES[@]}"; do
  set -- $b
  ./provision_one.sh "$1" "$2" "$3" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
echo "ALL_PROVISION_DONE"

#!/bin/bash
# Incremental merge: pull results from boxes -> push to prod -> ingest to DB (single writer).
set -uo pipefail
REPO=/Users/tvgdeena/nova-nanobody/scoring_backfill
PROD_HOST=ssh3.vast.ai
PROD_PORT=11806
DB=/root/artifacts/nanobody_mutations.sqlite
ARCHIVE=/root/live/epoch25422/archive.csv
SSHO="-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
BOXES=( "0 53449398 ssh2.vast.ai 13756" "1 53449401 ssh4.vast.ai 18324"
        "2 53449408 211.72.13.201 40157" "3 53449412 ssh1.vast.ai 25066"
        "4 53449418 ssh3.vast.ai 34196" "5 53449420 ssh6.vast.ai 27242"
        "6 53449423 ssh3.vast.ai 31808" "7 53449426 ssh8.vast.ai 36372"
        "8 53449432 217.138.104.229 10427" "9 53449435 ssh3.vast.ai 20070" )

mkdir -p $REPO/results
for b in "${BOXES[@]}"; do
  set -- $b; idx=$1; id=$2; host=$3; port=$4
  scp $SSHO -P $port root@$host:/root/results/box$idx.json $REPO/results/box$idx.json >/dev/null 2>&1
done
# count complete sequences across all results
total_complete=$(python3 -c "
import json,glob,os
tot=0
for p in glob.glob('$REPO/results/box*.json'):
    d=json.load(open(p))
    tot += sum(1 for e in d.values() if all(f'p{i}' in e for i in range(1,9)))
print(tot)")
echo "complete sequences so far: $total_complete"

# push to prod and ingest (single writer)
ssh $SSHO -p $PROD_PORT root@$PROD_HOST "mkdir -p /root/backfill_results" 2>/dev/null
for b in "${BOXES[@]}"; do
  set -- $b; idx=$1; id=$2; host=$3; port=$4
  [ -f $REPO/results/box$idx.json ] && scp $SSHO -P $PROD_PORT $REPO/results/box$idx.json root@$PROD_HOST:/root/backfill_results/ >/dev/null 2>&1
done
scp $SSHO -P $PROD_PORT $REPO/ingest_backfill.py root@$PROD_HOST:/root/ >/dev/null 2>&1
ssh $SSHO -p $PROD_PORT root@$PROD_HOST "/root/bgen_python.sh /root/ingest_backfill.py /root/backfill_results $ARCHIVE $DB" 2>&1 | tail -3
echo "MERGE_DONE"

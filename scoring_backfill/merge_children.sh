#!/bin/bash
set -uo pipefail
REPO=/Users/tvgdeena/nova-nanobody/scoring_backfill
PROD=root@ssh3.vast.ai
PROD_PORT=11806
DB=/root/artifacts/nanobody_mutations.sqlite
ARCHIVE=/root/live/epoch25422/archive.csv
SSHO="-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
BOXES=( "b0 118.163.199.123 12945" "b1 60.250.87.179 58248" "b2 185.229.26.83 41335"
        "b3 118.163.199.123 56759" "b4 ssh4.vast.ai 18324" "b5 ssh1.vast.ai 25066"
        "b6 ssh3.vast.ai 34196" "b7 ssh6.vast.ai 27242" "b8 ssh3.vast.ai 20070"
        "b9 211.72.13.201 41764" "b10 31.13.223.140 34721" )
mkdir -p $REPO/results_children9
for b in "${BOXES[@]}"; do
  set -- $b; sh=$1; host=$2; port=$3
  scp $SSHO -P $port root@$host:/root/results_children9/$sh.json $REPO/results_children9/$sh.json >/dev/null 2>&1
  scp $SSHO -P $port root@$host:/root/results_children9/steal.json $REPO/results_children9/steal_$sh.json >/dev/null 2>&1
done
total=$(python3 -c "
import json,glob
tot=0
for p in glob.glob('$REPO/results_children9/*.json'):
    try: d=json.load(open(p))
    except: continue
    tot += sum(1 for e in d.values() if all(f'p{i}' in e for i in range(1,9)))
print(tot)")
echo "complete children so far: $total"
ssh $SSHO -p $PROD_PORT $PROD "mkdir -p /root/backfill_children_results" 2>/dev/null
for b in "${BOXES[@]}"; do
  set -- $b; sh=$1; host=$2; port=$3
  [ -f $REPO/results_children9/$sh.json ] && scp $SSHO -P $PROD_PORT $REPO/results_children9/$sh.json $PROD:/root/backfill_children_results/ >/dev/null 2>&1
  [ -f $REPO/results_children9/steal_$sh.json ] && scp $SSHO -P $PROD_PORT $REPO/results_children9/steal_$sh.json $PROD:/root/backfill_children_results/ >/dev/null 2>&1
done
ssh $SSHO -p $PROD_PORT $PROD "rm -f /root/backfill_children_results/box*.json" 2>/dev/null
scp $SSHO -P $PROD_PORT $REPO/ingest_backfill.py $PROD:/root/ >/dev/null 2>&1
ssh $SSHO -p $PROD_PORT $PROD "/root/bgen_python.sh /root/ingest_backfill.py /root/backfill_children_results $ARCHIVE $DB children_8x" 2>&1 | tail -2
# refresh + push the work-stealing ledger to prod and every box
python3 $REPO/refresh_ledger.py
scp $SSHO -P $PROD_PORT $REPO/children_remaining.json $PROD:/root/children_remaining.json >/dev/null 2>&1
for b in "${BOXES[@]}"; do
  set -- $b; sh=$1; host=$2; port=$3
  scp $SSHO -P $port $REPO/children_remaining.json root@$host:/root/children_remaining.json >/dev/null 2>&1
done
echo "CHILDREN_MERGE_DONE"

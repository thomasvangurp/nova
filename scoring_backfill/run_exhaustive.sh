#!/bin/bash
# Exhaustive single-edit search: shard + 8x-score + ingest (source='exhaustive_8x').
# Run AFTER the children bootstrap completes.
set -uo pipefail
REPO=/Users/tvgdeena/nova-nanobody/scoring_backfill
SSHO="-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
BOXES11=( "b0 118.163.199.123 12945" "b1 60.250.87.179 58248" "b2 185.229.26.83 41335"
        "b3 118.163.199.123 56759" "b4 ssh4.vast.ai 18324" "b5 ssh1.vast.ai 25066"
        "b6 ssh3.vast.ai 34196" "b7 ssh6.vast.ai 27242" "b8 ssh3.vast.ai 20070"
        "b9 211.72.13.201 41764" "b10 31.13.223.140 34721" )

python3 - "$REPO" <<'PY'
import json, sys, os
repo = sys.argv[1]
d = json.load(open(f"{repo}/exhaustive_children.json"))
children = [{"sequence": c["sequence"], "epoch": 25423} for c in d["children"]]
os.makedirs(f"{repo}/exhaustive_shards", exist_ok=True)
buckets = [[] for _ in range(11)]
for i, c in enumerate(children):
    buckets[i % 11].append(c)
for i, b in enumerate(buckets):
    json.dump(b, open(f"{repo}/exhaustive_shards/b{i}.json", "w"))
print(f"sharded {len(children)} children across 11 boxes")
PY

for b in "${BOXES11[@]}"; do
  set -- $b; sh=$1; host=$2; port=$3
  scp $SSHO -P $port $REPO/exhaustive_shards/$sh.json score_worker.py root@$host:/root/ >/dev/null 2>&1
  ssh $SSHO -p $port root@$host "mkdir -p /root/results_exhaustive /root/bgen_shards_exhaustive; nohup python3 /root/score_worker.py --shard /root/$sh.json --results /root/results_exhaustive/$sh.json --work-root /root/bgen_shards_exhaustive --batch-size 256 --gpus 0,1,2,3,4,5,6,7 > /root/results_exhaustive/$sh.worker.log 2>&1 < /dev/null & echo LAUNCHED $sh" 2>/dev/null
done
echo "EXHAUSTIVE_LAUNCHED"

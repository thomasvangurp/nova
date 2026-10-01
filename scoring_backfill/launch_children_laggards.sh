#!/bin/bash
SSHO="-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
launch() {
  local idx=$1 host=$2 port=$3
  scp $SSHO -P $port children_shards/box$idx.json score_worker.py root@$host:/root/ >/dev/null 2>&1
  ssh $SSHO -p $port root@$host "mkdir -p /root/results_children /root/bgen_shards_children; nohup python3 /root/score_worker.py --shard /root/box$idx.json --results /root/results_children/box$idx.json --work-root /root/bgen_shards_children --batch-size 256 --gpus 0,1,2,3,4,5,6,7 > /root/results_children/box$idx.worker.log 2>&1 < /dev/null & echo LAUNCHED box$idx" 2>/dev/null
}
idx_for() { case $1 in 53449398) echo 0;; 53449408) echo 2;; 53449423) echo 6;; 53449426) echo 7;; 53449432) echo 8;; esac; }
launched=""
while true; do
  for id in 53449398 53449408 53449423 53449426 53449432; do
    echo "$launched" | grep -q "$id" && continue
    st=$(vastai show instance $id --raw 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); i=d if isinstance(d,dict) else d; print(i.get('actual_status'))" 2>/dev/null)
    if [ "$st" = "running" ]; then
      ep=$(vastai show instance $id --raw 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); i=d if isinstance(d,dict) else d; print(i.get('ssh_host'), i.get('ssh_port'))" 2>/dev/null)
      set -- $ep; host=$1; port=$2
      [ "$id" = "53449432" ] && { host=217.138.104.229; port=10427; }
      idx=$(idx_for $id)
      echo "$(date +%H:%M:%S) launching box$idx ($id) on $host:$port"
      launch $idx $host $port
      launched="$launched $id"
    fi
  done
  n=$(echo $launched | wc -w | tr -d ' ')
  [ "$n" -ge 5 ] && echo "ALL_LAGGARDS_LAUNCHED" && exit 0
  sleep 60
done

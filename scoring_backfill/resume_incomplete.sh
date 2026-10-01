#!/bin/bash
# Resume the 3 incomplete boxes once they reach running; re-launch their score_worker.
SSHO="-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
launch() {
  local idx=$1 host=$2 port=$3
  scp $SSHO -P $port shards/box$idx.json score_worker.py root@$host:/root/ >/dev/null 2>&1
  ssh $SSHO -p $port root@$host "mkdir -p /root/results /root/bgen_shards; nohup python3 /root/score_worker.py --shard /root/box$idx.json --results /root/results/box$idx.json --work-root /root/bgen_shards --batch-size 256 --gpus 0,1,2,3,4,5,6,7 > /root/results/box$idx.worker.log 2>&1 < /dev/null & echo LAUNCHED box$idx" 2>/dev/null
}
# resolve endpoints at launch time
while true; do
  ids=$(vastai show instances --raw 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin); insts=d if isinstance(d,list) else d.get('instances',[])
out={}
for i in insts:
    if str(i.get('id')) in ('53449401','53449418','53449432') and str(i.get('actual_status'))=='running':
        h=i.get('ssh_host'); p=i.get('ssh_port')
        if h and p: out[i.get('id')]=(h,p)
print(json.dumps(out))")
  echo "$(date +%H:%M:%S) running: $ids"
  if echo "$ids" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if len(d)==3 else 1)"; then
    # launch each
    python3 - "$ids" <<'PY'
import json,sys
ids=json.loads(sys.argv[1])
mapping={'53449401':'1','53449418':'4','53449432':'8'}
print(json.dumps(ids))
PY
    # simpler: parse via shell
    for id in 53449401 53449418 53449432; do
      ep=$(echo "$ids" | python3 -c "import json,sys; d=json.load(sys.stdin); h,p=d.get('$id',(None,None)); print(h,p)")
      set -- $ep; host=$1; port=$2
      [ -n "$host" ] && [ "$host" != "None" ] || continue
      case $id in 53449401) idx=1;; 53449418) idx=4;; 53449432) idx=8;; esac
      echo "launching box$idx on $host:$port"
      launch $idx $host $port
    done
    echo "RESUME_LAUNCHED"
    exit 0
  fi
  sleep 60
done

#!/bin/bash
# Launch box0/box1 scoring as soon as their HF cache has the affinity checkpoint.
SSHO="-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
launch() {
  local idx=$1 host=$2 port=$3
  scp $SSHO -P $port shards/box$idx.json score_worker.py root@$host:/root/ >/dev/null 2>&1
  ssh $SSHO -p $port root@$host "mkdir -p /root/results /root/bgen_shards; nohup python3 /root/score_worker.py --shard /root/box$idx.json --results /root/results/box$idx.json --work-root /root/bgen_shards --batch-size 256 --gpus 0,1,2,3,4,5,6,7 > /root/results/box$idx.worker.log 2>&1 < /dev/null & echo LAUNCHED box$idx"
}
done0=0; done1=0
while true; do
  for spec in "0 53449398 ssh2.vast.ai 13756" "1 53449401 ssh4.vast.ai 18324"; do
    set -- $spec; idx=$1; id=$2; host=$3; port=$4
    [ $idx -eq 0 ] && [ $done0 -eq 1 ] && continue
    [ $idx -eq 1 ] && [ $done1 -eq 1 ] && continue
    aff=$(ssh -n $SSHO -p $port root@$host 'ls /root/.cache/huggingface/hub/models--boltzgen--boltzgen-1/blobs/6dc13d488015666d3c3fdffd29fab54d72e4f2597b654f996cdcf5937feab090 2>/dev/null && echo Y' 2>/dev/null)
    if [ "$aff" = "Y" ]; then
      echo "box$idx affinity ready; launching"
      launch $idx $host $port
      [ $idx -eq 0 ] && done0=1 || done1=1
    fi
  done
  [ $done0 -eq 1 ] && [ $done1 -eq 1 ] && echo "LAGGARDS_LAUNCHED" && exit 0
  sleep 60
done

#!/bin/bash
REPO=/Users/tvgdeena/nova-nanobody
declare -a BOXES=(
  "53449398 ssh2.vast.ai 13756"
  "53449401 ssh4.vast.ai 18324"
  "53449408 ssh1.vast.ai 38978"
  "53449412 ssh1.vast.ai 25066"
  "53449418 ssh3.vast.ai 34196"
  "53449420 ssh6.vast.ai 27242"
  "53449423 ssh3.vast.ai 31808"
  "53449426 ssh8.vast.ai 36372"
  "53449432 217.138.104.229 10427"
  "53449435 ssh3.vast.ai 20070"
)
pids=()
for b in "${BOXES[@]}"; do
  set -- $b; ID=$1; HOST=$2; PORT=$3
  (
    LOG=/tmp/bootstrap_${ID}.log
    SSHO="-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
    echo "$(date +%s) START $ID $HOST:$PORT" > $LOG
    scp $SSHO -P $PORT $REPO/pixi.toml $REPO/pixi.lock $REPO/config/boltzgen_config.yaml $REPO/scoring_backfill/ckpt_specs.json $REPO/scoring_backfill/bootstrap_remote.sh root@$HOST:/root/ >> $LOG 2>&1
    scp $SSHO -r -P $PORT $REPO/ansible/files/root/bgen_shim root@$HOST:/root/ >> $LOG 2>&1
    scp $SSHO -P $PORT $REPO/ansible/files/root/bgen_python.sh root@$HOST:/root/ >> $LOG 2>&1
    ssh $SSHO -p $PORT root@$HOST "mkdir -p /root/fastscore/pinned-530f5dcbc54d /root/nanobody_opt /root/config /root/.boltzgen" >> $LOG 2>&1
    scp $SSHO -r -P $PORT $REPO/external_tools/fast_nanobody_scorer/ root@$HOST:/root/fastscore/pinned-530f5dcbc54d/ >> $LOG 2>&1
    scp $SSHO -P $PORT $REPO/nanobody_opt/fetch_checkpoints.py root@$HOST:/root/nanobody_opt/ >> $LOG 2>&1
    echo "$(date +%s) SCP_DONE" >> $LOG
    ssh $SSHO -p $PORT root@$HOST "bash /root/bootstrap_remote.sh" >> $LOG 2>&1
    echo "$(date +%s) FINISHED $ID" >> $LOG
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
echo "BOOTSTRAP_ALL_DONE"

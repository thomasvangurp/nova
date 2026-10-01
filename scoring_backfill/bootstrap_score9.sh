#!/bin/bash
set -uo pipefail
REPO=/Users/tvgdeena/nova-nanobody
ID=53449432; HOST=217.138.104.229; PORT=10427
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

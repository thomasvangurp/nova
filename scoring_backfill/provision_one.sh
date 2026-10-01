#!/bin/bash
# Relay-copy the fastscore runtime from the production box (read-only) to one
# scoring box, through this Mac. Uses tar pipes (no agent forwarding through
# the vast.ai proxy). Idempotent: re-running re-copies and re-verifies.
set -uo pipefail
ID=$1; HOST=$2; PORT=$3
SRC_PORT=11806
SRC_HOST=ssh3.vast.ai
LOG=/tmp/provision_${ID}.log
SSHOPTS="-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
echo "$(date +%s) START $ID" > $LOG

# 1) small files (pixi binary, manifest, lock, shim, fastscore tree, config)
ssh $SSHOPTS -p $SRC_PORT root@$SRC_HOST \
  "tar -C / -cf - usr/local/bin/pixi root/pixi.toml root/pixi.lock root/bgen_shim root/fastscore root/config/boltzgen_config.yaml" 2>/dev/null \
  | ssh $SSHOPTS -p $PORT root@$HOST "tar -C / -xf -" 2>>$LOG
echo "$(date +%s) SMALL_DONE rc=${PIPESTATUS[0]}" >> $LOG

# 2) bgen pixi environment (~6.5 GB)
ssh $SSHOPTS -p $SRC_PORT root@$SRC_HOST \
  "tar -C / -cf - root/.pixi/envs/bgen" 2>/dev/null \
  | ssh $SSHOPTS -p $PORT root@$HOST "tar -C / -xf -" 2>>$LOG
echo "$(date +%s) BGEN_DONE rc=${PIPESTATUS[0]}" >> $LOG

# 3) boltz weights (~5.6 GB)
ssh $SSHOPTS -p $SRC_PORT root@$SRC_HOST \
  "tar -C / -cf - root/.boltzgen" 2>/dev/null \
  | ssh $SSHOPTS -p $PORT root@$HOST "tar -C / -xf -" 2>>$LOG
echo "$(date +%s) BOLTZ_DONE rc=${PIPESTATUS[0]}" >> $LOG

# verify
ssh $SSHOPTS -p $PORT root@$HOST \
  "export PYTHONPATH=/root/bgen_shim:/root/fastscore/release/boltzgen_src; /usr/local/bin/pixi run --manifest-path /root/pixi.toml --environment bgen env python -c 'import torch, boltzgen; print(\"VERIFY_OK\", torch.cuda.device_count(), boltzgen.__file__)' 2>&1 | tail -2" >> $LOG 2>&1
echo "$(date +%s) VERIFY_DONE" >> $LOG
echo "PROVISION_FINISHED $ID" >> $LOG

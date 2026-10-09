#!/bin/bash
# Supervise the chain-state miner, resolving its signing hotkey each start.
#
# This was `exec`, i.e. a launcher, not a supervisor: if the miner exited,
# nothing brought it back. That matters more than it looks, because the miner
# is the only writer of /root/artifacts/chain_state.json, and a chain_state
# that stops being written does not look broken -- it looks unchanged. The
# publisher goes on reading it and assigning sequences to uids that may no
# longer be ours.
#
# The hotkey is resolved against the live metagraph on every start rather than
# hardcoded. A hardcoded name is what turned a routine deregistration into an
# unstartable miner on 2026-09-21.
set -u
cd /root
set -a; . /root/nanobody.env; set +a

PYTHON=/root/.pixi/envs/chain310/bin/python
LOG=/root/artifacts/miner_supervisor.log
# 2026-10-09: the master box signs with nova68-slot-330 (slot-87 is gone).
PREFER=${NOVA_MINER_HOTKEY:-nova68-slot-330}

log() { echo "$(date -Is) $*" >> "$LOG"; }

while true; do
    HOTKEY=$("$PYTHON" /root/deploy/pick_registered_hotkey.py \
        --network finney --netuid 68 \
        --wallet-path /root/.bittensor/wallets --wallet-name mining_ops \
        --prefer "$PREFER" 2>>"$LOG")
    if [ -z "$HOTKEY" ]; then
        # No registered hotkey, or the chain was unreachable. Restarting on a
        # tight loop would just burn the log; wait and try again.
        log "no signing hotkey resolved - retrying in 120s"
        sleep 120
        continue
    fi
    log "starting miner with hotkey $HOTKEY"
    "$PYTHON" neurons/miner/miner.py \
        --network finney --netuid 68 \
        --wallet.path /root/.bittensor/wallets --wallet.name mining_ops \
        --wallet.hotkey "$HOTKEY" --chain-state-only --logging.debug
    log "miner exited ($?) - restarting in 15s"
    sleep 15
done

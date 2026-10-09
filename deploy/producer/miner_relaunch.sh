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
# No default slot name. Until 2026-10-09 this read
# PREFER=${NOVA_MINER_HOTKEY:-nova68-slot-330}; slot-330 then deregistered and
# the "preference" went stale silently. The picker now selects among the
# hotkeys the live chain lists under our coldkey; NOVA_MINER_HOTKEY remains an
# explicit override only, and the sticky file keeps restarts on one signer.
PREFER=${NOVA_MINER_HOTKEY:-}
COLDKEY=${NOVA_COLDKEY_SS58:-5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ}
STICKY=/root/artifacts/miner_signer_hotkey

log() { echo "$(date -Is) $*" >> "$LOG"; }

while true; do
    HOTKEY=$("$PYTHON" /root/deploy/pick_registered_hotkey.py \
        --network finney --netuid 68 \
        --wallet-path /root/.bittensor/wallets --wallet-name mining_ops \
        --coldkey "$COLDKEY" --prefer "$PREFER" --sticky-file "$STICKY" 2>>"$LOG")
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

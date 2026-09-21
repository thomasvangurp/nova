#!/bin/bash
# Push newly purchased hotkeys from the purchase host to whichever GPU box is
# rented right now.
#
# Why this exists: the purchaser creates hotkeys where the coldkey is, and the
# publisher needs them where the miner is. Between deploys nothing moved them, so
# the fleet drifted silently -- the Pi held three hotkeys while the miner had two,
# and the publisher was preparing a submission for a uid whose key was not on the
# box. That failure is invisible from the leaderboard: it looks exactly like not
# having bought the slot at all.
#
# Only hotkeys move. The coldkey stays here; both GPU playbooks assert it is
# absent on the rental, and this script never names it.
set -euo pipefail

WALLET="${NOVA_WALLET_NAME:-mining_ops}"
PREFIX="${NOVA_HOTKEY_PREFIX:-nova68-slot}"
LABEL="${NOVA_VAST_LABEL:-nova-nanobody-pixi}"
HOTKEYS="$HOME/.bittensor/wallets/$WALLET/hotkeys"
VASTAI="${NOVA_VASTAI:-/opt/nova-nanobody/deploy/slot_host/.pixi/envs/default/bin/vastai}"
REMOTE_WALLET="/root/.bittensor/wallets/$WALLET/hotkeys"

[ -d "$HOTKEYS" ] || { echo "no hotkey directory at $HOTKEYS"; exit 0; }
shopt -s nullglob
files=("$HOTKEYS/$PREFIX"*)
[ ${#files[@]} -gt 0 ] || { echo "no hotkeys matching $PREFIX yet"; exit 0; }

# Discover the box the same way the Ansible inventory does: by label, so a
# rotated SSH port is never mistaken for an absent host.
# Every match, not the first: during a producer handover the outgoing and
# incoming rentals both carry the label for a while, and `break` would put 186
# hotkeys on whichever one vast.ai happened to list first. Keys on the wrong
# rental is a far worse outcome than a sync that refuses to run.
matched="$("$VASTAI" show instances --raw 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.loads(sys.stdin.read())
except Exception:
    sys.exit(1)
for item in data:
    if item.get('label') == '$LABEL' and item.get('actual_status') == 'running':
        print(item.get('ssh_host'), item.get('ssh_port'))
")" || true

if [ "$(printf '%s\n' "$matched" | grep -c '[^[:space:]]')" -gt 1 ]; then
  echo "more than one running instance is labelled $LABEL; refusing to guess which one gets the hotkeys" >&2
  exit 1
fi
read -r host port <<<"$matched"

if [ -z "${host:-}" ] || [ -z "${port:-}" ]; then
  # No box rented is the normal state between runs, not an error: the purchaser
  # keeps buying while no GPU is rented, which is the whole point of it living
  # here rather than on the box.
  echo "no running instance labelled $LABEL; nothing to sync"
  exit 0
fi

SSH_OPTS=(-o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
ssh "${SSH_OPTS[@]}" -p "$port" "root@$host" "mkdir -p $REMOTE_WALLET && chmod 700 $REMOTE_WALLET"

# rsync rather than scp: only what changed crosses, and mode 600 is preserved.
rsync -e "ssh ${SSH_OPTS[*]} -p $port" -a --chmod=F600 \
      "${files[@]}" "root@$host:$REMOTE_WALLET/"

count=$(ssh "${SSH_OPTS[@]}" -p "$port" "root@$host" "ls -1 $REMOTE_WALLET | grep -c '^$PREFIX' || true")
echo "synced ${#files[@]} local file(s); box now holds $count matching $PREFIX"

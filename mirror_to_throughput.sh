#!/bin/bash
# Box-to-box mirror: prod (this host) -> nova-nanobody-throughput.
# Runs ON prod. Uses /root/.ssh/throughput_sync to reach the new box.
# Scorer box only: NO bittensor wallet, NO miner/publisher artifacts.
set -euo pipefail

SSH_HOST="${SSH_HOST:?set SSH_HOST}"
SSH_PORT="${SSH_PORT:?set SSH_PORT}"

SSHOPT=(-i /root/.ssh/throughput_sync -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -p "$SSH_PORT")
R="rsync -a --info=progress2 -e ssh"
SSH="ssh ${SSHOPT[*]}"

# Ensure rsync exists on the destination (base image may lack it).
$SSH "root@$SSH_HOST" "command -v rsync >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync)" >/dev/null 2>&1 || true

echo "== dirs =="
for d in nanobody_opt config data utils bgen_shim metanano_shim fastscore external_tools .boltzgen; do
  echo "  /root/$d"
  $R -e "ssh ${SSHOPT[*]}" "/root/$d/" "root@$SSH_HOST:/root/$d/"
done

echo "== files =="
$R -e "ssh ${SSHOPT[*]}" \
  /root/pixi.toml \
  /root/pixi.lock \
  /root/.sn68_mysql_creds \
  /root/nanobody.env \
  /root/.hftoken \
  /root/bgen_python.sh \
  /root/metanano_python.sh \
  /root/sync_structures.sh \
  /root/sync_loop.sh \
  /root/multiscore_hf_top200.py \
  /root/multipass_all_rivals.py \
  "root@$SSH_HOST:/root/"

echo "== scorer settings =="
$SSH "root@$SSH_HOST" "mkdir -p /root/artifacts"
$R -e "ssh ${SSHOPT[*]}" /root/artifacts/nanobody_fast_scorer.json "root@$SSH_HOST:/root/artifacts/"

echo "== fix perms on dest (creds + hftoken 600) =="
$SSH "root@$SSH_HOST" "chmod 600 /root/.sn68_mysql_creds /root/.hftoken"

echo "MIRROR_DONE"

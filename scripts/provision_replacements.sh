#!/usr/bin/env bash
# Fast-provision replacement scoring boxes from a running box that already has
# the full scorer environment. Encodes every fix learned on 2026-09-29 so a
# replacement is a one-command run with no re-discovery.
#
# Usage:
#   provision_replacements.sh <source_host:port> <offer_id> [offer_id ...]
#
# - <source_host:port> is any RUNNING box that has the complete setup
#   (HF cache + bgen env + scorer + patch). It is the rsync origin.
# - offer ids are vast.ai 8-GPU offers (RTX 4090 or A100; do NOT use RTX 5090,
#   which needs CUDA 12.8+ that the bgen torch 2.6 cu126 env does not provide).
#
# The data path is source-box -> new-box over DIRECT ssh (public_ipaddr:direct_port),
# bypassing the flaky vast.ai proxy, with agent forwarding so the source box can
# authenticate. Agent forwarding only works on a FRESH connection, so every ssh
# below passes -o ControlMaster=no -o ControlPath=none.
set -euo pipefail

SOURCE="${1:?usage: provision_replacements.sh <source_host:port> <offer_id>...}"
shift
OFFERS=("$@")
(( ${#OFFERS[@]} >= 1 )) || { echo "no offers given" >&2; exit 2; }

IMAGE="nvidia/cuda:12.6.0-devel-ubuntu22.04"
DISK=300
LABEL="nova-nanobody-score-repl"
# The exact set of paths a fresh box needs. Order matters (large first so the
# slow transfer overlaps the later small ones).
DIRS=(
  /root/.cache/huggingface   # model weights (10 GB) -- HF_HUB_OFFLINE=1 makes the scorer use this
  /root/.pixi/envs/bgen      # the boltzgen pixi env (6.5 GB)
  /root/fastscore            # pinned scorer + boltzgen_src + refdesign (already patched)
  /root/nanobody_opt         # scoring workers + adapters
  /root/pixi.toml            # pixi manifest (release_python.sh needs it)
  /root/pixi.lock
  /root/bgen_shim            # shim that keeps boltzgen_wrapper from importing the validator stack
)

echo "== creating ${#OFFERS[@]} instances =="
NEW_IDS=()
for offer in "${OFFERS[@]}"; do
  id=$(vastai create instance "$offer" \
      --image "$IMAGE" --disk "$DISK" --ssh --direct --label "$LABEL" --raw \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["new_contract"])')
  NEW_IDS+=("$id")
  echo "  offer $offer -> instance $id"
done

echo "== waiting for running + resolving direct endpoints =="
declare -a TARGETS
for id in "${NEW_IDS[@]}"; do
  for _ in $(seq 1 60); do
    info=$(vastai show instance "$id" --raw 2>/dev/null)
    st=$(python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status') or d.get('status'))" <<<"$info" 2>/dev/null)
    if [ "$st" = "running" ]; then
      ep=$(python3 -c "import json,sys; d=json.load(sys.stdin); print(d['public_ipaddr']+':'+str(d['direct_port_start']))" <<<"$info")
      TARGETS+=("$ep")
      echo "  $id -> $ep (running)"
      break
    fi
    sleep 10
  done
done
(( ${#TARGETS[@]} == ${#NEW_IDS[@]} )) || { echo "not all instances reached running" >&2; exit 3; }

echo "== fan-out rsync: source $SOURCE -> ${#TARGETS[@]} boxes =="
FANOUT=/root/fanout_replacements.sh
scp -P 11806 -o StrictHostKeyChecking=no /dev/null root@ssh3.vast.ai:/dev/null >/dev/null 2>&1 || true
# Build the fan-out on the source box itself (it has the source dirs local).
ssh -o ControlMaster=no -o ControlPath=none -o StrictHostKeyChecking=no "root@${SOURCE%:*}" -p "${SOURCE#*:}" \
  "cat > $FANOUT" <<'FANOUT_EOF'
#!/usr/bin/env bash
set -u
DIRS=(/root/.cache/huggingface /root/.pixi/envs/bgen /root/fastscore /root/nanobody_opt /root/pixi.toml /root/pixi.lock /root/bgen_shim)
for b in "${TARGETS[@]:-}"; do :; done
FANOUT_EOF
echo "  (fan-out script placeholder written; wiring targets next)"

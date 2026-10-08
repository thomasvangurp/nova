#!/usr/bin/env bash
set -euo pipefail

cache_root=${NOVA_CACHE_ROOT:-/mnt/nova-cache}

if [[ ! -d "$cache_root" || ! -w "$cache_root" ]]; then
  printf 'Persistent cache is not mounted read/write at %s\n' "$cache_root" >&2
  exit 1
fi

# A Vast local volume is mounted before the on-start script runs. Requiring a
# real mount guards against accidentally placing multi-gigabyte checkpoints in
# disposable container storage when attachment failed.
if ! findmnt -n --target "$cache_root" >/dev/null; then
  printf '%s is a directory, but not a mounted filesystem\n' "$cache_root" >&2
  exit 1
fi

ensure_cache_link() {
  local relative_path=$1
  local destination=$2
  local target="${cache_root}/${relative_path}"

  mkdir -p "$target" "$(dirname "$destination")"

  if [[ -L "$destination" ]]; then
    if [[ "$(readlink "$destination")" != "$target" ]]; then
      printf 'Refusing to replace unexpected symlink %s -> %s\n' \
        "$destination" "$(readlink "$destination")" >&2
      return 1
    fi
    return 0
  fi

  if [[ -e "$destination" ]]; then
    if [[ ! -d "$destination" || -n "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      printf 'Refusing to replace non-empty cache path %s\n' "$destination" >&2
      return 1
    fi
    rmdir "$destination"
  fi

  ln -s "$target" "$destination"
}

ensure_cache_link boltzgen /root/.boltzgen
ensure_cache_link huggingface /root/.cache/huggingface
ensure_cache_link triton-bgen /root/.cache/triton-bgen
ensure_cache_link cuda-compute /root/.nv/ComputeCache

printf '%s\n' \
  'layout_version=1' \
  'checkpoints=boltzgen' \
  'huggingface_cache=huggingface' \
  'triton_cache=triton-bgen' \
  'cuda_cache=cuda-compute' \
  > "${cache_root}/.nova-nanobody-cache"

printf 'NOVA nanobody cache ready at %s\n' "$cache_root"

# ---------------------------------------------------------------------------
# SN68 Hetzner MySQL writer + .cif structure sync.
#
# A freshly-spawned vast.ai box must, before any producer/chain writer runs:
#   1. have the system build deps for the PyPI mysqlclient driver,
#   2. declare + install mysqlclient in the `bgen` pixi environment,
#   3. have /root/.sn68_mysql_creds (0600) so nanobody_opt/db.py can reach the
#      central Hetzner MySQL,
#   4. have /root/.ssh/sn68_sync so the .cif rsync can authenticate to Hetzner,
#   5. run the nohup .cif sync loop (vast.ai containers have no systemd/cron).
#
# Every step is idempotent and safe to re-run on each onstart.
# ---------------------------------------------------------------------------

# (a) mysqlclient build prerequisites.
apt-get update -y
apt-get install -y pkg-config default-libmysqlclient-dev

# (b) Declare the PyPI mysqlclient driver in the validator-bgen feature (the
# feature the `bgen` environment is built from), then install the environment.
# Guarded so `pixi add` is skipped once the dependency is already declared --
# re-running `pixi add` for an existing dependency is an error.
if ! grep -q '^mysqlclient' /root/pixi.toml; then
  cd /root
  pixi add --manifest-path /root/pixi.toml --pypi mysqlclient --feature validator-bgen
fi
pixi install --manifest-path /root/pixi.toml --environment bgen

# (c) MySQL credentials file. The password is provisioned from the secret
# store / macOS Keychain at deploy time; this bootstrap only asserts the file
# exists and tightens its mode. Never write the value here.
if [[ ! -s /root/.sn68_mysql_creds ]]; then
  printf 'WARNING: /root/.sn68_mysql_creds missing or empty; populate from the secret store/Keychain before DB writers run\n' >&2
else
  chmod 600 /root/.sn68_mysql_creds
fi

# (d) Box -> Hetzner .cif rsync key.
if [[ ! -s /root/.ssh/sn68_sync ]]; then
  printf 'WARNING: /root/.ssh/sn68_sync missing; .cif sync to Hetzner will fail\n' >&2
else
  chmod 600 /root/.ssh/sn68_sync
fi

# (e) .cif sync loop (nohup background process). The two helper scripts are
# materialised below if absent so the loop is self-contained on a fresh box.
if [[ ! -f /root/sync_loop.sh ]]; then
  cat > /root/sync_loop.sh <<'SYNC_LOOP_EOF'
#!/bin/bash
# Continuous .cif sync loop: push new captured structures to Hetzner every 10 min.
# Survives transient rsync/network failures (retries next tick). One instance at a time.
LOG=/tmp/sync_loop.log
echo "$(date +%F_%T) sync loop started (pid $$)" >> "$LOG"
while true; do
  bash /root/sync_structures.sh >> "$LOG" 2>&1
  sleep 600
done
SYNC_LOOP_EOF
  chmod +x /root/sync_loop.sh
fi

if [[ ! -f /root/sync_structures.sh ]]; then
  cat > /root/sync_structures.sh <<'SYNC_STRUCTURES_EOF'
#!/bin/bash
# Push captured Boltz-2 structures to Hetzner /mnt/sx-data (persistent).
# Only .cif files, preserving the /root/live tree so a local path
# /root/live/epochN/.../structures/H.cif maps to /mnt/sx-data/structures/epochN/.../structures/H.cif.
# Add-only (no --delete): files survive local pruning.
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
rsync -azm \
  --include='*/' --include='*.cif' --exclude='*' \
  -e "ssh -i /root/.ssh/sn68_sync $SSHOPT" \
  /root/live/ sxbot@46.225.101.210:/mnt/sx-data/structures/ 2>/tmp/sync_structures.err
rc=$?
if [ $rc -eq 0 ]; then
  echo "$(date +%F_%T) synced $(ssh -i /root/.ssh/sn68_sync $SSHOPT sxbot@46.225.101.210 'find /mnt/sx-data/structures -name "*.cif" | wc -l') cif files to hetzner"
else
  echo "$(date +%F_%T) sync FAILED rc=$rc"; tail -3 /tmp/sync_structures.err
fi
SYNC_STRUCTURES_EOF
  chmod +x /root/sync_structures.sh
fi

running_pids=$(pgrep -f 'bash /root/sync_loop.sh' || true)
if [[ -n "$running_pids" ]]; then
  printf 'sync loop already running (pid %s)\n' "$(printf '%s' "$running_pids" | tr '\n' ' ')"
else
  nohup bash /root/sync_loop.sh >/dev/null 2>&1 &
  printf 'Started .cif sync loop (pid %s)\n' "$!"
fi

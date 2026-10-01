#!/usr/bin/env bash
# Pull the mutation database off the rented GPU box and keep it here.
#
# The box is the one machine in this system that is expected to die. It died
# twice in one day: once when vast.ai credit ran out and once when the host
# stopped being available, and `vastai start` then answers "Required resources
# are currently unavailable, state change queued" -- that instance is gone and
# nothing on it is coming back.
#
# Almost everything on it is reconstructible. The code is in git, the hotkeys
# live here and are pushed by sync_hotkeys.sh, the target archive re-downloads,
# and the public half of the mutation database rebuilds from the leaderboard API
# with `pixi run seed-mutation-database`. The one artifact that is not is this
# file: it also carries every exact screen we ever ran locally, thousands per
# epoch against the ~40 scored submissions per epoch the public API can return.
# When the box died that history went with it, and no backup existed.
#
# The copy goes to the purchase host because that is the machine that survived
# both deaths -- it holds the coldkey, so it is never a rental.
set -euo pipefail

LABEL="${NOVA_VAST_LABEL:-nova-nanobody-pixi}"
VASTAI="${NOVA_VASTAI:-/opt/nova-nanobody/deploy/slot_host/.pixi/envs/default/bin/vastai}"
DISCOVERY_HOST="${NOVA_VAST_DISCOVERY_HOST:-}"
REMOTE_DB="${NOVA_MUTATION_DATABASE:-/root/artifacts/nanobody_mutations.sqlite}"
DEST="${NOVA_BACKUP_DIR:-$HOME/nova-backups}"
KEEP="${NOVA_BACKUP_KEEP:-24}"
MIN_FREE_GIB="${NOVA_BACKUP_MIN_FREE_GIB:-20}"
MIN_KEEP="${NOVA_BACKUP_MIN_KEEP:-2}"

case "$KEEP:$MIN_FREE_GIB:$MIN_KEEP" in
  *[!0-9:]*|:*|*::*)
    echo "backup retention settings must be non-negative integers" >&2
    exit 2
    ;;
esac
if (( KEEP < 1 || MIN_KEEP < 1 || MIN_KEEP > KEEP )); then
  echo "backup retention requires 1 <= MIN_KEEP <= KEEP" >&2
  exit 2
fi

mkdir -p "$DEST"

# Discover the box the same way the Ansible inventory and sync_hotkeys.sh do:
# by label, so a rotated SSH port is never mistaken for an absent host. The Mac
# may delegate only this discovery call to the Pi; the provider credential and
# raw response remain there, while the database travels directly from the GPU
# to the Mac.
if [ -n "$DISCOVERY_HOST" ]; then
  if [[ ! "$DISCOVERY_HOST" =~ ^[A-Za-z0-9_.@:-]+$ ]] \
      || [[ ! "$VASTAI" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
    echo "unsafe discovery host or client path" >&2
    exit 2
  fi
  provider_json=$(ssh -o BatchMode=yes -o ConnectTimeout=15 \
    "$DISCOVERY_HOST" "$VASTAI" show instances --raw 2>/dev/null || true)
else
  provider_json=$("$VASTAI" show instances --raw 2>/dev/null || true)
fi
read -r host port <<<"$(printf '%s' "$provider_json" | python3 -c "
import sys, json
try:
    data = json.loads(sys.stdin.read())
except Exception:
    sys.exit(1)
for item in data:
    if item.get('label') == '$LABEL' and item.get('actual_status') == 'running':
        print(item.get('ssh_host'), item.get('ssh_port'))
        break
")" || true
unset provider_json

if [ -z "${host:-}" ] || [ -z "${port:-}" ]; then
  # No box rented is the normal state between rentals, not an error.
  echo "no running instance labelled $LABEL; nothing to back up"
  exit 0
fi

SSH_OPTS=(-o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

if ! ssh "${SSH_OPTS[@]}" -p "$port" "root@$host" "test -s '$REMOTE_DB'" 2>/dev/null; then
  # A freshly provisioned box has no database until the producer has run, and a
  # backup job is not the thing that should complain about that.
  echo "no database at $REMOTE_DB on the box yet; nothing to back up"
  exit 0
fi

# SQLite in WAL mode, which this store uses, must not be copied with cp or
# rsync: the -wal file holds committed pages the main file does not yet have, so
# a straight copy taken mid-write restores as a torn database. `.backup` takes
# the same consistent snapshot a reader would see and is safe against a producer
# writing throughout.
# Python's sqlite3, not the sqlite3 CLI: the GPU image does not ship the CLI,
# and the first scheduled run failed on exactly that.
SNAPSHOT="/tmp/nanobody_mutations.backup.sqlite"
ARCHIVE="$SNAPSHOT.gz"
if ! ssh "${SSH_OPTS[@]}" -p "$port" "root@$host" \
    "rm -f '$SNAPSHOT' '$ARCHIVE' && python3 -c \"
import sqlite3
source = sqlite3.connect('$REMOTE_DB')
target = sqlite3.connect('$SNAPSHOT')
source.backup(target)
target.close()
source.close()
check = sqlite3.connect('$SNAPSHOT')
valid = check.execute('pragma integrity_check').fetchone()[0] == 'ok'
check.close()
raise SystemExit(0 if valid else 1)
\" && gzip -1 -f '$SNAPSHOT'" 2>/dev/null; then
  echo "could not snapshot the database on the box; leaving the last backup in place" >&2
  exit 1
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
target="$DEST/nanobody_mutations.$stamp.sqlite.gz"
partial="$target.partial"

if ! scp -C "${SSH_OPTS[@]}" -P "$port" \
    "root@$host:$ARCHIVE" "$partial" >/dev/null 2>&1; then
  echo "could not fetch the snapshot; leaving the last backup in place" >&2
  rm -f -- "$partial"
  exit 1
fi
ssh "${SSH_OPTS[@]}" -p "$port" "root@$host" \
  "rm -f '$SNAPSHOT' '$ARCHIVE'" 2>/dev/null || true

# The database passed SQLite integrity_check before compression. Verify the
# gzip CRC after transport so truncation or corruption cannot become latest.
if ! /usr/bin/gzip -t "$partial" 2>/dev/null; then
  echo "fetched archive failed gzip verification; discarding it" >&2
  rm -f -- "$partial"
  exit 1
fi

mv "$partial" "$target"
ln -sfn "$target" "$DEST/nanobody_mutations.latest.sqlite.gz"
size=$(du -h "$target" | cut -f1)
echo "backed up $size to $target"

# A fixed snapshot count is not a fixed storage budget: this database grows
# every epoch. Restore a free-space floor by dropping only the oldest
# timestamped snapshots, never the live database, the latest symlink, or the
# final MIN_KEEP independent restore points. Python keeps this compatible with
# macOS's Bash 3.2, which has no mapfile.
python3 - "$DEST" "$KEEP" "$MIN_FREE_GIB" "$MIN_KEEP" <<'PY'
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1])
keep, min_free_gib, min_keep = map(int, sys.argv[2:])
snapshots = {
    path for pattern in ("nanobody_mutations.*.sqlite", "nanobody_mutations.*.sqlite.gz")
    for path in root.glob(pattern)
    if not path.is_symlink()
}
retained = sorted(snapshots, key=lambda path: path.stat().st_mtime, reverse=True)
for old in retained[keep:]:
    old.unlink()
    print(f"pruned {old} above the {keep}-snapshot retention window")
retained = retained[:keep]
reserve = min_free_gib * 1024 ** 3
while shutil.disk_usage(root).free < reserve and len(retained) > min_keep:
    old = retained.pop()
    old.unlink()
    print(f"pruned {old} to restore {min_free_gib} GiB free-space reserve")
if shutil.disk_usage(root).free < reserve:
    print(
        f"backup reserve remains below {min_free_gib} GiB with "
        f"{len(retained)} snapshots retained",
        file=sys.stderr,
    )
PY

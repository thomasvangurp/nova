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
REMOTE_DB="${NOVA_MUTATION_DATABASE:-/root/artifacts/nanobody_mutations.sqlite}"
DEST="${NOVA_BACKUP_DIR:-$HOME/nova-backups}"
KEEP="${NOVA_BACKUP_KEEP:-24}"

mkdir -p "$DEST"

# Discover the box the same way the Ansible inventory and sync_hotkeys.sh do:
# by label, so a rotated SSH port is never mistaken for an absent host.
read -r host port <<<"$("$VASTAI" show instances --raw 2>/dev/null | python3 -c "
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
SNAPSHOT="/tmp/nanobody_mutations.backup.sqlite"
if ! ssh "${SSH_OPTS[@]}" -p "$port" "root@$host" \
    "rm -f '$SNAPSHOT' && sqlite3 '$REMOTE_DB' \".backup '$SNAPSHOT'\"" 2>/dev/null; then
  echo "could not snapshot the database on the box; leaving the last backup in place" >&2
  exit 1
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
target="$DEST/nanobody_mutations.$stamp.sqlite"

if ! scp "${SSH_OPTS[@]}" -P "$port" "root@$host:$SNAPSHOT" "$target" >/dev/null 2>&1; then
  echo "could not fetch the snapshot; leaving the last backup in place" >&2
  rm -f "$target"
  exit 1
fi
ssh "${SSH_OPTS[@]}" -p "$port" "root@$host" "rm -f '$SNAPSHOT'" 2>/dev/null || true

# Verify what landed before it is allowed to displace anything: a truncated
# transfer is worse than no backup, because it looks like one.
if ! sqlite3 "$target" "pragma integrity_check;" 2>/dev/null | grep -qx "ok"; then
  echo "fetched file failed integrity_check; discarding it" >&2
  rm -f "$target"
  exit 1
fi

ln -sfn "$target" "$DEST/nanobody_mutations.latest.sqlite"
size=$(du -h "$target" | cut -f1)
echo "backed up $size to $target"

# Keep a window rather than one copy: corruption discovered late is still
# recoverable if an older good snapshot survives.
ls -1t "$DEST"/nanobody_mutations.*.sqlite 2>/dev/null \
  | grep -v 'latest' \
  | tail -n +$((KEEP + 1)) \
  | while read -r old; do rm -f "$old"; done

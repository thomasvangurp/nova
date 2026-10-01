#!/bin/bash
# Take a consistent snapshot of the Pi's live compound bank and retain it on
# the Mac. The Pi holds only the live database and a short-lived /tmp snapshot.
set -euo pipefail

PI_HOST="${NOVA_PI_HOST:-thomasg@raspberrypi}"
REMOTE_DB="${NOVA_BANK_DB:-/opt/nova-submit/bank.sqlite}"
REMOTE_TMP="/tmp/nova-bank-backup.sqlite"
DEST="${NOVA_BACKUP_DIR:-$HOME/nova-backups}"
KEEP="${NOVA_BANK_BACKUP_KEEP:-24}"

if [[ ! "$PI_HOST" =~ ^[A-Za-z0-9_.@:-]+$ ]] \
    || [[ ! "$REMOTE_DB" =~ ^/[A-Za-z0-9_./-]+$ ]] \
    || [[ ! "$KEEP" =~ ^[1-9][0-9]*$ ]]; then
  echo "unsafe bank-backup configuration" >&2
  exit 2
fi

mkdir -p "$DEST"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
target="$DEST/bank.$stamp.sqlite"
partial="$target.partial"

cleanup() {
  python3 - "$partial" <<'PY'
from pathlib import Path
import sys
try:
    Path(sys.argv[1]).unlink()
except FileNotFoundError:
    pass
PY
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$PI_HOST" \
    "python3 -c 'from pathlib import Path; Path(\"$REMOTE_TMP\").unlink(missing_ok=True)'" \
    >/dev/null 2>&1 || true
}
trap cleanup EXIT

ssh -o BatchMode=yes -o ConnectTimeout=15 "$PI_HOST" \
  python3 - "$REMOTE_DB" "$REMOTE_TMP" <<'PY'
import sqlite3
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
if not source.is_file() or source.stat().st_size == 0:
    raise SystemExit("live bank database is missing")
destination.unlink(missing_ok=True)
with sqlite3.connect(source) as origin, sqlite3.connect(destination) as backup:
    origin.backup(backup)
with sqlite3.connect(destination) as check:
    if check.execute("pragma integrity_check").fetchone()[0] != "ok":
        destination.unlink(missing_ok=True)
        raise SystemExit("remote bank snapshot failed integrity check")
PY

scp -q -C -o BatchMode=yes -o ConnectTimeout=15 \
  "$PI_HOST:$REMOTE_TMP" "$partial"

python3 - "$partial" <<'PY'
import sqlite3
import sys
from pathlib import Path

path = Path(sys.argv[1])
with sqlite3.connect(path) as connection:
    if connection.execute("pragma integrity_check").fetchone()[0] != "ok":
        raise SystemExit("transferred bank snapshot failed integrity check")
PY
mv "$partial" "$target"
ln -sfn "$target" "$DEST/bank.latest.sqlite"

python3 - "$DEST" "$KEEP" <<'PY'
from pathlib import Path
import sys

root, keep = Path(sys.argv[1]), int(sys.argv[2])
snapshots = sorted(
    (p for p in root.glob("bank.*.sqlite") if not p.is_symlink()),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
for old in snapshots[keep:]:
    old.unlink()
PY

echo "backed up Pi bank to $target"

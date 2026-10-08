#!/bin/bash
# Fallback: restore the ORIGINAL producer publisher when the children lane
# fails.  The producer publisher is the pre-lane strategy: it de-novo generates
# and submits through gate="validator-novelty" (is_duplicate replay against the
# top-50), so it is an independent source of VALID (>=7-edit, novel-CDR3)
# sequences at ~174-tier quality.  It does not depend on the children bank.
#
# Idempotent: safe to run manually, or from the lane supervisor after N stale
# epochs.  Restoring publisher_argv.bin and killing the session makes the
# session watchdog relaunch the producer publisher within 60s.
set -u
ART=/root/artifacts
SRC="$ART/publisher_argv.PRODUCER-BACKUP.bin"
DST="$ART/publisher_argv.bin"

if [ ! -s "$SRC" ]; then
  echo "ROLLBACK_FAIL: producer backup argv missing at $SRC" >&2
  exit 1
fi
cp "$SRC" "$DST"
tmux kill-session -t nova-nanobody-publisher 2>/dev/null || true
echo "ROLLBACK_OK: restored producer publisher argv; watchdog relaunches within 60s"

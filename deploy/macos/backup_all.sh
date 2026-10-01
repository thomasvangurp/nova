#!/bin/bash
# Mac-owned backup entry point. Provider and SSH credentials stay in their
# normal client stores; this wrapper carries no authentication material.
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
status=0
/bin/bash "$ROOT/deploy/slot_host/backup_artifacts.sh" || status=1
/bin/bash "$ROOT/deploy/macos/backup_pi_bank.sh" || status=1
exit "$status"

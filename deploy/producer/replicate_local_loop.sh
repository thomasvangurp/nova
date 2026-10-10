#!/bin/bash
# Long-running Hetzner replicator for master's local SQLite primary
# (ARCHITECTURE_REDESIGN new finding 8). NOT deployed yet: start it in the
# same change that sets NOVA_LOCAL_SQLITE=1 for the producer, e.g. as a
# session_watchdog-managed tmux session:
#     tmux new-session -d -s nova-replicate-local "bash /root/deploy/producer/replicate_local_loop.sh"
# It is off the producer's path: the producer only writes the local file;
# this process drains rows with replicated_at IS NULL to Hetzner, polling every
# 30 s when idle and back to back while there is a backlog. A second copy exits
# immediately (flock on <db>.replicate.lock), so a double start is harmless.
# Health: <db>.replication_status.json (ok, last_ok_at, dirty/quarantined counts).
set -u
DB="${NOVA_LOCAL_SQLITE_PATH:-/root/artifacts/nanobody_mutations.sqlite}"
PY="${NOVA_REPLICATE_PYTHON:-/root/.pixi/envs/bgen/bin/python}"
LOG="${NOVA_REPLICATE_LOG:-/root/artifacts/replicate_local.log}"
cd /root || exit 1
while true; do
    if [ -s "$DB" ]; then
        "$PY" -m nanobody_opt.replicate_local --local "$DB" --loop-interval 30 >> "$LOG" 2>&1
        echo "$(date -Is) replicate_local exited rc=$?" >> "$LOG"
    fi
    sleep 30
done

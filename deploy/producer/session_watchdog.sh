#!/bin/bash
# Keep every nanobody production session alive and reject frozen chain state.
set -u

LOG=/root/artifacts/session_watchdog.log
STATE=/root/artifacts/chain_state.json
MAX_ROSTER_AGE_BLOCKS=800
# The publisher rejects chain state after 180 seconds. Check the same wall-clock
# signal: comparing two block values inside a frozen file reports a small delta
# forever and missed the 25315-25317 producer outage.
MAX_STATE_AGE_SECONDS=150
INTERVAL=60
RESTART_GRACE=600
LAST_MINER_KILL=0
MINER_KILLS=0

log() { echo "$(date -Is) $*" >> "$LOG"; }

start_session() {
    local name=$1 script=$2
    log "session $name MISSING - restarting"
    tmux new-session -d -s "$name" "bash $script" \
        && log "session $name restarted" \
        || log "session $name FAILED to restart"
}

restart_miner() {
    local reason=$1
    local now; now=$(date +%s)
    if [ $((now - LAST_MINER_KILL)) -lt "$RESTART_GRACE" ]; then
        log "$reason but a restart is $((now - LAST_MINER_KILL))s old - waiting"
        return 0
    fi
    MINER_KILLS=$((MINER_KILLS + 1))
    log "$reason - restarting miner (restart #$MINER_KILLS)"
    if [ "$MINER_KILLS" -ge 3 ]; then
        log "ALERT: chain state stayed unhealthy across $MINER_KILLS miner restarts"
    fi
    LAST_MINER_KILL=$now
    tmux kill-session -t nova-nanobody-miner 2>/dev/null || true
}

check_state() {
    local facts age roster_age
    facts=$(python3 - "$STATE" <<'PY' 2>/dev/null
import json, os, sys, time
try:
    state = os.stat(sys.argv[1])
    body = json.load(open(sys.argv[1]))
    roster_age = max(0, int(body["block"]) - int(body["roster_block"]))
    wall_age = max(0, int(time.time() - state.st_mtime))
except Exception:
    print("1000000000 1000000000")
else:
    print(wall_age, roster_age)
PY
)
    read -r age roster_age <<<"${facts:-1000000000 1000000000}"
    if [ "$age" -gt "$MAX_STATE_AGE_SECONDS" ]; then
        restart_miner "chain state is ${age}s old (limit ${MAX_STATE_AGE_SECONDS}s)"
        return
    fi
    if [ "$roster_age" -gt "$MAX_ROSTER_AGE_BLOCKS" ]; then
        restart_miner "roster is ${roster_age} blocks stale (limit ${MAX_ROSTER_AGE_BLOCKS})"
    fi
}

log "watchdog started (state limit ${MAX_STATE_AGE_SECONDS}s, roster limit ${MAX_ROSTER_AGE_BLOCKS} blocks)"
while true; do
    tmux has-session -t nova-nanobody-miner 2>/dev/null \
        || start_session nova-nanobody-miner /root/miner_relaunch.sh
    tmux has-session -t nova-nanobody-producer 2>/dev/null \
        || start_session nova-nanobody-producer /root/producer_relaunch.sh
    tmux has-session -t nova-nanobody-scorerd 2>/dev/null \
        || start_session nova-nanobody-scorerd /root/scorer_daemon_supervisor.sh
    tmux has-session -t nova-rosterwatch 2>/dev/null \
        || start_session nova-rosterwatch /root/roster_watch.sh

    if ! tmux has-session -t nova-nanobody-publisher 2>/dev/null; then
        log "session nova-nanobody-publisher MISSING - restarting"
        python3 - >> "$LOG" 2>&1 <<'PY'
import subprocess
try:
    raw = open("/root/artifacts/publisher_argv.bin", "rb").read().decode()
except OSError as error:
    print(f"publisher argv unreadable ({error.strerror}); cannot restart")
    raise SystemExit(0)
argv = [value for value in raw.split("\0") if value]
if argv and argv[0] == "tmux" and "--commit-command" not in argv:
    print("publisher restart rc:", subprocess.run(argv).returncode)
else:
    print("publisher argv is unsafe or invalid; not restarting")
PY
    fi

    check_state
    sleep "$INTERVAL"
done

#!/bin/bash
# Relaunch the live producer exactly as it last ran.
#
# The argv is replayed verbatim from /root/artifacts/producer_argv.txt (one
# argument per line, captured from the running `pixi run ... live-producer`
# process), so a relaunch cannot drift from the hand-tuned flag set.
#
# Restored 2026-10-09 from deploy/producer_relaunch.sh (pipeline-overlap),
# adapted to what production actually runs today:
#   - NO NANOBODY_SCORER_DAEMON: the live producer runs the subprocess scorer.
#     The warm daemon's consumer is hf_field_refresh_loop.sh; pointing the
#     producer at it too would make the two contend for one daemon.
#   - NO nanobody.env: the running producer does not carry it, and it holds a
#     token the producer has no use for.
#   - output is appended to /root/live_producer.log, as the live session does.
# This is a launcher, not a supervisor: when the producer exits the session
# ends and session_watchdog.sh starts a fresh one on its next pass.
cd /root
ARGV_FILE=/root/artifacts/producer_argv.txt
if [ ! -s "$ARGV_FILE" ]; then
    echo "$(date -Is) $ARGV_FILE missing or empty; refusing to guess a producer argv" \
        | tee -a /root/live_producer.log
    sleep 300
    exit 1
fi
mapfile -t ARGV < "$ARGV_FILE"
# a trailing blank from the NUL-terminated cmdline would become an empty arg
while [ -n "${ARGV[*]}" ] && [ -z "${ARGV[-1]}" ]; do unset 'ARGV[-1]'; done
echo "$(date -Is) relaunching producer with ${#ARGV[@]} argv entries" | tee -a /root/live_producer.log
"${ARGV[@]}" 2>&1 | tee -a /root/live_producer.log

#!/bin/bash
# Keep one warm scorer daemon alive for the live producer.
#
# The producer's waves fall back to their own subprocess whenever the socket
# is absent or a round faults, so a daemon death costs speed, never a wave.
# Restart it anyway: a dead daemon silently returns the pipeline to the slow
# path, and nothing else would say so.
# Keep a record outside the tmux pane, which dies with the session.
exec >> /root/artifacts/scorer_daemon_supervisor.log 2>&1
CONFIG=/root/artifacts/nanobody_fast_scorer.json
SOCKET=/root/artifacts/nanobody_scorer.sock
PYTHON=$(python3 -c "import json;print(json.load(open('$CONFIG'))['python'])")
SCORER=$(python3 -c "import json;print(json.load(open('$CONFIG'))['scorer'])")
# Measured 32.8s -> 12.1s per warm round (2.71x), scores unchanged.
# Roll back with NANOBODY_CHUNK_REUSE=0 here; drop just the VRAM-holding
# piece with NANOBODY_CHUNK_REUSE_SKIP=warm_lightning.
# 2026-10-09: defaulted to 0 because the daemon production has actually run
# since the 2026-10-08 box move was started without it ("CHUNK_REUSE
# enabled=False" in /root/scorer_daemon.log).
# 2026-10-10: set 1 deliberately, not by restore -- verified live on an
# isolated 4-GPU box (13 real sequences, content-keyed stages + warm
# lightning + persistent analysis pool all installed, scores unchanged from
# the uncached path) before flipping this on the production daemon.
export NANOBODY_CHUNK_REUSE=${NANOBODY_CHUNK_REUSE:-1}
echo "supervisor: python=$PYTHON scorer=$SCORER socket=$SOCKET reuse=$NANOBODY_CHUNK_REUSE"
while true; do
    # Never start a second daemon on top of one that is already serving the
    # socket (one started by hand, or orphaned when this tmux session was
    # killed). Two daemons double the VRAM on every GPU and the rm below
    # would pull the socket out from under the live one's clients. Adopt it
    # instead: wait for it to exit, then take over.
    while OTHER=$(pgrep -o -f "scorer_daemon.py --scorer .*--socket $SOCKET"); do
        echo "=== $(date -Is) daemon pid $OTHER already serves $SOCKET; waiting for it to exit ==="
        while kill -0 "$OTHER" 2>/dev/null; do sleep 30; done
    done
    rm -f "$SOCKET"
    echo "=== $(date -Is) starting scorer daemon ==="
    "$PYTHON" /root/nanobody_opt/scorer_daemon.py \
        --scorer "$SCORER" --socket "$SOCKET"
    echo "=== $(date -Is) daemon exited ($?); restarting in 10s ==="
    sleep 10
done

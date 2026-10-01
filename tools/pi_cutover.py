#!/usr/bin/env python3
"""Hand the chain writer role to the Pi, once, inside the only safe window.

Production prepares its nanobody artifact at
    epoch_end - reserve_blocks - publication_prepare_blocks
and starts committing at epoch_end - reserve_blocks. With 55 and 20 that is a
20-block gap: before it the artifact holds LAST epoch's sequences, after it
production has already signed and a second write races.

The order below is the whole safety argument:

  1. wait for the artifact to carry THIS epoch
  2. pull it and verify it parses with the expected uids
  3. only then stop production's publisher
  4. commit from the Pi

If step 2 fails we never reach step 3, and production commits normally -- the
epoch is untouched. Stopping first and discovering a bad artifact afterwards
would lose every uid's nanobody as well as its molecules.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

PI = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "thomasg@100.64.243.65"]
PROD = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
        "-i", "/Users/tvgdeena/.ssh/id_ed25519", "-p", "15470", "root@ssh2.vast.ai"]


def say(**fields):
    print(json.dumps(dict(at=time.strftime("%H:%M:%S"), **fields)), flush=True)


def run(where, command, timeout=200):
    return subprocess.run(where + ["-n", command], capture_output=True,
                          text=True, timeout=timeout)


def producer_epoch() -> int | None:
    got = run(PROD, "python3 -c \"import json;print(json.load("
                    "open('/root/artifacts/nanobody_result.json'))['epoch'])\"")
    for line in got.stdout.strip().splitlines():
        if line.strip().isdigit():
            return int(line.strip())
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--reaction", type=int, required=True)
    parser.add_argument("--deadline-minutes", type=float, default=75.0)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="do everything except stop production and sign")
    arguments = parser.parse_args()

    say(event="waiting", epoch=arguments.epoch, reaction=arguments.reaction)
    deadline = time.time() + arguments.deadline_minutes * 60
    while time.time() < deadline:
        seen = producer_epoch()
        if seen is not None and seen >= arguments.epoch:
            say(event="producer_ready", artifact_epoch=seen)
            break
        say(event="poll", artifact_epoch=seen)
        time.sleep(arguments.poll_seconds)
    else:
        say(event="abandoned", why="producer never refreshed in the window")
        return 1

    # 2. pull and verify BEFORE touching production.
    #    Runs pi_submit as a PROGRAM. The first cutover exec()d part of it to
    #    reuse pull_producer and died on NameError: __file__ -- a path the dry
    #    run never reached, because it abandoned earlier on "producer never
    #    refreshed". Reuse by import boundary, not by exec.
    pull = run(PI, "cd /opt/nova-submit && ./.pixi/envs/default/bin/python "
                   f"pi_submit.py --epoch {arguments.epoch} "
                   f"--reaction {arguments.reaction} --pull-only",
               timeout=300)
    say(event="pull", rc=pull.returncode, out=(pull.stdout or pull.stderr).strip()[-160:])
    if pull.returncode:
        say(event="abandoned", why="pull failed; production untouched")
        return 1

    check = run(PI, "/opt/nova-submit/.pixi/envs/default/bin/python -c \""
                    "import json;d=json.load(open('/opt/nova-submit/nanobody_result.json'));"
                    "print(d['epoch'], len(d['submissions']))\"")
    parts = check.stdout.strip().split()
    if len(parts) != 2 or int(parts[0]) != arguments.epoch:
        say(event="abandoned", why=f"artifact check failed: {check.stdout.strip()[:80]}")
        return 1
    say(event="verified", artifact_epoch=int(parts[0]), entries=int(parts[1]))

    if arguments.dry_run:
        say(event="dry_run_stop", note="production NOT stopped, nothing signed")
        return 0

    # 3. production stops signing. From here the Pi owns the epoch.
    #    Its start command is read off the live tmux pane first: rollback has to
    #    replay what was ACTUALLY running, and there is no run script to fall
    #    back on -- it is a long inline command with the coldkey, the policies
    #    and the commit template baked in.
    captured = run(PROD, "tmux list-panes -t nova-nanobody-publisher "
                         "-F '#{pane_start_command}' 2>/dev/null | head -1")
    start_command = captured.stdout.strip()
    if not start_command:
        say(event="abandoned", why="could not capture the publisher start command; "
                                   "refusing to stop something I cannot restart")
        return 1
    say(event="captured_start_command", length=len(start_command))
    stop = run(PROD, "pkill -f 'publish_current_nano''body'; sleep 3; "
                     "ps -eo args | grep -c 'publish_current_nano''body'")
    say(event="production_stopped", rc=stop.returncode, remaining=stop.stdout.strip()[-10:])

    # 4. commit
    commit = run(PI, f"cd /opt/nova-submit && ./.pixi/envs/default/bin/python pi_submit.py "
                     f"--epoch {arguments.epoch} --reaction {arguments.reaction}",
                 timeout=900)
    for line in (commit.stdout or "").strip().splitlines():
        print(line, flush=True)

    # 5. if the Pi could not commit, put production back so the epoch is not
    #    simply lost. Nobody is watching this run: an unattended failure that
    #    leaves BOTH writers stopped costs all twelve uids their nanobody as
    #    well as their molecules, which is worse than never having tried.
    #    Production is idempotent per epoch -- it keeps a committed-uid ledger
    #    -- so restarting it is safe whether or not it had begun.
    if commit.returncode:
        say(event="rollback", why="pi_submit failed; restarting production")
        import shlex
        back = run(PROD, "tmux new-session -d -s nova-nanobody-publisher "
                         + shlex.quote(start_command)
                         + "; sleep 6; ps -eo args | grep -c 'publish_current_nano''body'",
                   timeout=180)
        say(event="rollback_result", rc=back.returncode,
            running=back.stdout.strip()[-10:])
    say(event="finished", rc=commit.returncode)
    return commit.returncode


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Submit every epoch from the Pi: molecules when we cover the reaction, the
nanobody always.

The reaction is drawn per epoch from the epoch's first block hash, so it cannot
be known in advance and a one-shot driver aimed at a named epoch cannot catch
it -- the first cutover attempt missed for exactly that reason. This reads the
reaction itself, each epoch, and acts inside the only safe window:

    epoch_end - 75   producer refreshes its artifact
    epoch_end - 55   producer starts committing        <- must be ours by here
    epoch_end - 16   validator blackout

Ordering is the safety argument. Pull and verify BEFORE stopping the producer;
if anything fails first, the producer commits the epoch as usual and nothing is
lost. Once the producer is stopped it stays stopped -- the Pi owns the fleet --
but a failed commit restarts it so an unattended failure is not an empty epoch.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PI = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "thomasg@100.64.243.65"]
PROD = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
        "-i", "/Users/tvgdeena/.ssh/id_ed25519", "-p", "15470", "root@ssh2.vast.ai"]
EPOCH_LENGTH = 361
# The validator stops reading at end-16.
BLACKOUT_BLOCKS = 16
# Measured on-chain at 25248: ten uids committed 5 blocks apart, 69 blocks left
# down to 26. Almost all of it is set_commitment waiting for the extrinsic.
BLOCKS_PER_UID = 5


def commit_reserve(uids: int) -> int:
    """How many blocks the commit run itself needs, plus the blackout.

    A fixed reserve was wrong in both directions: too small and the tail
    commits into the blackout, too large and we stop waiting for a producer we
    still had time for. It depends on how many uids there are, so compute it.
    """
    return BLACKOUT_BLOCKS + BLOCKS_PER_UID * max(1, uids)


def say(**f):
    print(json.dumps(dict(at=time.strftime("%H:%M:%S"), **f)), flush=True)


LOCAL = ["bash", "-lc"]


def stream(where, command, timeout=900):
    """Run a command and echo its stdout as it arrives.

    `run` captures, so a twelve-minute commit run showed nothing at all until
    it was over -- no way to tell a slow epoch from a wedged one while it still
    mattered. The commits are the one step worth watching live.
    """
    argv = where + ([command] if where is LOCAL else ["-n", command])
    child = subprocess.Popen(argv, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    try:
        for line in child.stdout:
            line = line.rstrip()
            if line:
                lines.append(line)
                print(line, flush=True)
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        child.kill()
        raise
    return child.returncode, lines


def run(where, command, timeout=240):
    """Run on a host. `where` of LOCAL means here -- no ssh, no -n.

    The loop is meant to live ON the submit host: a writer for twelve uids
    should not depend on a laptop staying awake, which is exactly how six
    hours of epochs went uncommitted.
    """
    if where is LOCAL:
        return subprocess.run(where + [command], capture_output=True,
                              text=True, timeout=timeout)
    return subprocess.run(where + ["-n", command], capture_output=True,
                          text=True, timeout=timeout)


def block_now(python: str) -> dict:
    """Just the block and epoch. Cheap: no metagraph.

    The first version built a full metagraph on every poll, which takes tens of
    seconds -- the loop spent all its time re-reading the chain and emitted
    nothing in a 280s dry run. The roster is only needed once, when the window
    opens.
    """
    script = ("import bittensor as bt, json;"
              "s=bt.Subtensor(network='finney');b=s.get_current_block();L=s.tempo(68)+1;"
              "e=b//L;st=e*L;"
              "print(json.dumps(dict(block=b,epoch=e,start=st,end=st+L)))")
    got = subprocess.run([python, "-c", script], capture_output=True, text=True, timeout=180)
    for line in got.stdout.strip().splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"block read failed: {got.stderr.strip()[-200:]}")


def chain(python: str) -> dict:
    """Block, epoch, this epoch's reaction and the roster. Expensive."""
    script = (
        "import bittensor as bt, json;"
        "s=bt.Subtensor(network='finney');b=s.get_current_block();L=s.tempo(68)+1;"
        "e=b//L;st=e*L;h=s.determine_block_hash(st);"
        "mg=s.metagraph(68, lite=True);"
        "import pathlib;"
        "ck=json.load(open(pathlib.Path.home()/'.bittensor/wallets/mining_ops/coldkeypub.txt'))['ss58Address'];"
        "crew=sorted((int(u),str(k)) for u,k,c in zip(mg.uids,mg.hotkeys,mg.coldkeys) if str(c)==ck);"
        "print(json.dumps(dict(block=b,epoch=e,start=st,end=st+L,"
        "reaction=int(h,16)%5+1,crew=crew)))")
    got = subprocess.run([python, "-c", script], capture_output=True, text=True, timeout=300)
    for line in got.stdout.strip().splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"chain read failed: {got.stderr.strip()[-200:]}")


PULL = ("cd /opt/nova-submit && ./.pixi/envs/default/bin/python pi_submit.py "
        "--epoch {epoch} --reaction {reaction} --pull-only")


def producer_epoch_here(a, epoch: int, reaction: int) -> int | None:
    """On the Pi, probe by pulling.

    The old probe ssh'd to the producer with a key path that only exists on the
    laptop, and with a hardcoded vast.ai port that rotates on every restart.
    pi_submit resolves the endpoint by LABEL, so let it do the pulling and read
    the epoch out of what landed. Same answer, no second credential path.
    """
    got = run(LOCAL, PULL.format(epoch=epoch, reaction=reaction), timeout=300)
    if got.returncode:
        return None
    artifact = Path("/opt/nova-submit/nanobody_result.json")
    try:
        return int(json.loads(artifact.read_text())["epoch"])
    except Exception:
        return None


def producer_epoch() -> int | None:
    got = run(PROD, "python3 -c \"import json;print(json.load("
                    "open('/root/artifacts/nanobody_result.json'))['epoch'])\"")
    for line in got.stdout.strip().splitlines():
        if line.strip().isdigit():
            return int(line.strip())
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reactions", default="3",
                        help="comma-separated reactions we hold a rescored bank for")
    parser.add_argument("--chain-python", required=True)
    parser.add_argument("--rdkit-python", required=True)
    parser.add_argument("--epochs", type=int, default=0, help="0 = run forever")
    parser.add_argument("--producer-commits", action="store_true",
                        help="the producer box still signs, so stop it before "
                             "we commit. Off by default: it now runs the "
                             "artifact-only publisher, and stopping that stops "
                             "the sequences we depend on")
    parser.add_argument("--on-submit-host", action="store_true",
                        help="the loop is running ON the Pi; talk to it directly")
    parser.add_argument("--root", default=None,
                        help="override the tree root (the submit host uses /opt/nova-submit)")
    parser.add_argument("--fleet-tree", default=None)
    parser.add_argument("--dry-run", action="store_true")
    a = parser.parse_args()
    global PI, ROOT
    if a.on_submit_host:
        PI = LOCAL
    if a.root:
        ROOT = Path(a.root)
    covered = {int(x) for x in a.reactions.split(",") if x.strip()}
    say(event="start", covered=sorted(covered), dry_run=a.dry_run)

    state = {"producer_stopped": False}

    # Which epochs are already done, on DISK.
    #
    # Restart=always makes restarts routine, and an in-memory set meant a
    # restart inside a commit window re-ran the whole epoch: ten uids
    # re-committed, each with a LATER block_submitted, which is the validator's
    # first tiebreak. Surviving a restart is the entire point of the unit.
    ledger = ROOT / "handled_epochs.json"
    try:
        handled = {int(e) for e in json.loads(ledger.read_text())}
    except Exception:
        handled = set()
    say(event="resumed", already_handled=sorted(handled)[-4:])

    def remember(epoch: int) -> None:
        handled.add(epoch)
        try:
            # Write via a temporary file: a torn ledger reads as empty, and an
            # empty ledger means re-committing every epoch it forgot.
            scratch = ledger.with_suffix(".tmp")
            scratch.write_text(json.dumps(sorted(handled)[-200:]))
            scratch.replace(ledger)
        except OSError as error:
            say(event="ledger_write_failed", err=str(error)[:120])

    while True:
        if a.epochs and len(handled) >= a.epochs:
            say(event="done", handled=sorted(handled)); return 0
        try:
            tick = block_now(a.chain_python)
        except Exception as error:
            say(event="chain_error", err=str(error)[:160]); time.sleep(60); continue
        epoch, block, end = tick["epoch"], tick["block"], tick["end"]
        if epoch in handled:
            time.sleep(30); continue
        refresh_at = end - 75
        if block < refresh_at:
            wait = max(20, min(300, (refresh_at - block) * 12 * 0.6))
            say(event="waiting", epoch=epoch, block=block,
                refresh_at=refresh_at, sleep=round(wait))
            try:
                warm_archive(a, epoch, state)
            except Exception as error:
                say(event="archive_warm_failed", err=str(error)[:140])
            time.sleep(wait)
            continue

        # window open: NOW pay for the metagraph, once
        try:
            view = chain(a.chain_python)
        except Exception as error:
            say(event="chain_error", err=str(error)[:160]); time.sleep(20); continue
        epoch, block, end = view["epoch"], view["block"], view["end"]
        if epoch in handled:
            continue
        say(event="window_open", epoch=epoch, reaction=view["reaction"],
            block=block, crew=len(view["crew"]))
        try:
            handle_epoch(a, view, epoch, block, end, covered, state)
        except Exception as error:
            # One bad epoch must not end the loop. It died on a full disk at
            # 25244 AFTER it had become the sole writer, so three epochs
            # committed nothing at all -- the rollback only covers a failed
            # commit, not the process going away between epochs.
            say(event="epoch_error", epoch=epoch, err=f"{type(error).__name__}: {error}"[:200])
        remember(epoch)
        continue

    return 0


def warm_archive(a, epoch: int, state: dict) -> None:
    """Refresh the submission archive and rebuild its fingerprints, early.

    Refreshing at release time would charge a grown archive's 54s rebuild to
    the commit window, where the producer slack is under two minutes. There is
    an hour of waiting before that window; spend it here instead.
    """
    if state.get("warmed") == epoch or not a.on_submit_host:
        return
    state["warmed"] = epoch
    code, lines = stream(PI, "cd /opt/nova-submit && ./.pixi/envs/default/bin/python -c "
                             "\"import sys; sys.path.insert(0,'tools'); "
                             "from pathlib import Path; import json; "
                             "import bank_release as b; "
                             "s=b.refresh_archive(Path('P40261_molecules.csv')); "
                             "n=len(b.historical_fingerprints(Path('P40261_molecules.csv'), "
                             "Path('P40261_molecules.csv.fps'))); "
                             "print(json.dumps(dict(event='archive_warm', fingerprints=n, **s)))\"",
                         timeout=600)
    if code:
        say(event="archive_warm_failed", rc=code, tail=(lines[-1][:140] if lines else ""))


def publish(a, names) -> None:
    """Put the epoch's inputs where the submit host reads them."""
    for name in names:
        src = ROOT / ("release" if a.on_submit_host else "data/release") / name
        if not src.is_file():
            say(event="publish_missing", name=name)
            continue
        if a.on_submit_host:
            subprocess.run(["cp", str(src), f"/opt/nova-submit/{name}"], timeout=120)
        else:
            subprocess.run(["rsync", "-t", "-e", "ssh -o BatchMode=yes", str(src),
                            f"thomasg@100.64.243.65:/opt/nova-submit/{name}"], timeout=300)


def handle_epoch(a, view, epoch, block, end, covered, state):
    if True:
        producer_stopped = state["producer_stopped"]
        crew_path = ROOT / ("release" if a.on_submit_host else "data/release") / "crew.json"
        crew_path.parent.mkdir(parents=True, exist_ok=True)
        crew_path.write_text(json.dumps(view["crew"]))

        # molecules only when we actually hold a rescored bank for this reaction
        have_molecules = False
        if view["reaction"] in covered:
            made = subprocess.run(
                [sys.executable, str(ROOT / "tools/make_fleet.py"),
                 "--epoch", str(epoch), "--reaction", str(view["reaction"]),
                 "--crew", str(crew_path), "--python", a.rdkit_python]
                + (["--bank", str(ROOT / "bank.sqlite"),
                    "--out", str(ROOT / "release"),
                    "--fleet-tree", str(ROOT)] if a.on_submit_host else []),
                capture_output=True, text=True, timeout=1200)
            for line in made.stdout.strip().splitlines():
                if line.startswith("{"):
                    say(**json.loads(line))
            have_molecules = made.returncode == 0
        else:
            say(event="reaction_not_covered", reaction=view["reaction"])

        # crew.json is the LIVE metagraph view and must be published EVERY
        # epoch. Publishing it only alongside a molecule release meant that on
        # the four-in-five epochs drawing an uncovered reaction, pi_submit
        # planned against whatever crew the last covered epoch left behind: a
        # slot registered since then would not submit until the next reaction-3
        # draw, and the slot manager is buying slots back continuously.
        publish(a, ["crew.json"] + (["compound_fleet.json"] if have_molecules else []))

        # Wait for the producer to publish THIS epoch's sequences.
        #
        # The old budget ran only to end-55 and then gave up entirely, which
        # made a producer running a few minutes late cost the whole epoch --
        # molecules included. Once the Pi became the sole writer that was five
        # empty epochs, 25243 through 25247. Wait to the last safe block
        # instead, and if the producer still has not moved, commit what we have
        # rather than nothing.
        reserve = commit_reserve(len(view["crew"]))
        deadline = time.time() + max(0, end - reserve - block) * 12
        say(event="wait_budget", uids=len(view["crew"]), reserve=reserve,
            blocks_to_wait=max(0, end - reserve - block))
        stale_ok = False
        while time.time() < deadline:
            seen = (producer_epoch_here(a, epoch, view["reaction"])
                    if a.on_submit_host else producer_epoch())
            if seen is not None and seen >= epoch:
                say(event="producer_ready", artifact_epoch=seen); break
            time.sleep(30)
        else:
            stale_ok = True
            say(event="producer_late", molecules=have_molecules,
                why="committing last epoch's sequences rather than nothing")

        pull = run(PI, "cd /opt/nova-submit && ./.pixi/envs/default/bin/python pi_submit.py "
                       f"--epoch {epoch} --reaction {view['reaction']} --pull-only", timeout=300)
        if pull.returncode:
            say(event="skip_epoch", why="pull failed; producer untouched")
            return
        say(event="pulled")

        if a.dry_run:
            say(event="dry_run_stop", epoch=epoch, molecules=have_molecules)
            return

        if a.producer_commits and not producer_stopped:
            captured = run(PROD, "tmux list-panes -t nova-nanobody-publisher "
                                 "-F '#{pane_start_command}' 2>/dev/null | head -1")
            start_command = captured.stdout.strip()
            if not start_command:
                say(event="skip_epoch", why="cannot capture producer start command")
                return
            run(PROD, "pkill -f 'publish_current_nano''body'; sleep 3")
            state["producer_stopped"] = True
            say(event="producer_stopped", epoch=epoch)
        else:
            start_command = None

        code, _ = stream(PI, "cd /opt/nova-submit && ./.pixi/envs/default/bin/python pi_submit.py "
                             f"--epoch {epoch} --reaction {view['reaction']}"
                             + (" --allow-stale-nanobody" if stale_ok else ""), timeout=900)
        commit = type("R", (), {"returncode": code})()
        say(event="committed", epoch=epoch, rc=code, molecules=have_molecules)

        if commit.returncode and start_command:
            import shlex
            say(event="rollback", why="commit failed on the epoch we took over")
            run(PROD, "tmux new-session -d -s nova-nanobody-publisher "
                      + shlex.quote(start_command) + "; sleep 6", timeout=180)
            state["producer_stopped"] = False
        return


if __name__ == "__main__":
    raise SystemExit(main())

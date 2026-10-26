#!/usr/bin/env python3
"""Submit every epoch from the Pi: molecules when we cover the reaction, the
nanobody always.

The reaction is drawn per epoch from the epoch's first block hash, so it cannot
be known in advance and a one-shot driver aimed at a named epoch cannot catch
it -- the first cutover attempt missed for exactly that reason. This reads the
reaction itself, each epoch, and acts inside the only safe window:

    epoch_end - 136  prepare crew and molecule inputs
    epoch_end - 41   final producer pull (parallel batches up to 20 UIDs)
    epoch_end - 36   commit/repair budget
    epoch_end - 16   validator blackout

Ordering is the safety argument. Pull and verify BEFORE stopping the producer;
if anything fails first, the producer commits the epoch as usual and nothing is
lost. Once the producer is stopped it stays stopped -- the Pi owns the fleet --
but a failed commit restarts it so an unattended failure is not an empty epoch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import shutil
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nanobody_opt import submission_timing
PI = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "thomasg@100.64.243.65"]
PROD = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
        "-i", "/Users/tvgdeena/.ssh/id_ed25519", "-p", "15470", "root@ssh2.vast.ai"]
EPOCH_LENGTH = 361
# The validator stops reading at end-16.
BLACKOUT_BLOCKS = submission_timing.BLACKOUT_BLOCKS
# Serial reference retained for opening the CPU-preparation window. The
# actual submission reserve now prices concurrent commitments and repairs.
BLOCKS_PER_UID = submission_timing.BLOCKS_PER_UID

# The window-open margin below (WINDOW_OPEN_BLOCKS) has to leave real slack
# for the producer-wait after commit_reserve() is subtracted, but crew size
# is not cheaply known until the cold, expensive chain() call that only runs
# AFTER the window opens (block_now() is deliberately cheap and crew-blind,
# to avoid paying for a full metagraph read every 30s while merely waiting).
# So the margin is sized against a generously assumed max crew instead of
# the crew this epoch actually has.
#
# This bit for real at crew=13: reserve=16+5*13=81 blocks, versus a window
# that opened at end-75 -- NEGATIVE slack, so the producer-wait deadline
# clamped to zero and every single epoch fell back to a stale nanobody
# artifact, silently, for 50+ epochs (last real producer_ready was 25292).
# The nanobody entry-win rate visibly collapsed over exactly that window.
ASSUMED_MAX_CREW = 20
MIN_PRODUCER_WAIT_BLOCKS = 20


# --- producer identity (ARCHITECTURE_REDESIGN P1 / R4) ----------------------
#
# For epochs 25613-25617 this loop committed the children-lane artifact
# (publication_stage=children, a 300-sequence frozen bank) on every uid and
# logged rc=0 each time: nothing here read WHO produced the artifact, and the
# only quality gate (>= 8 observations) was met exactly. The real publisher,
# tools/publish_current_nanobody.py, only ever writes publication_stage
# "provisional" or "qualified"; any other writer must use a different stage
# (children_submit.py writes "children").
#
# Phase A (always on): every final pull is checked; a mismatch logs
# UNEXPECTED_PRODUCER here and in producer_alerts.log, and the liveness
# checker alerts on it. The commit is NOT changed.
# Phase B (OFF by default): with the toggle file below present -- or
# --refuse-unexpected-producer -- an unexpected artifact is not committed; the
# last artifact that passed this check is committed instead (rebound to the
# fresh crew through the existing --allow-stale-nanobody path), or, if none
# exists, no nanobody is committed this epoch. Enable deliberately:
#     touch /opt/nova-submit/REFUSE_UNEXPECTED_PRODUCER     (off: rm it)
# It is read every epoch; no restart is needed either way.
EXPECTED_ARTIFACT_KIND = "nanobody-fleet"
EXPECTED_PUBLICATION_STAGES = frozenset({"provisional", "qualified"})
# Layer 1 of R4 (an explicit producer name in the artifact) is not deployed
# yet; once it is, any value outside this set is unexpected too.
EXPECTED_PRODUCER_NAMES = frozenset({"publish_current_nanobody"})
REFUSE_TOGGLE_NAME = "REFUSE_UNEXPECTED_PRODUCER"
LAST_EXPECTED_NAME = "nanobody_result.last_expected.json"
PI_ALERTS = Path("/opt/nova-submit/producer_alerts.log")


def artifact_provenance(path: Path) -> tuple[bool, dict]:
    """(expected, evidence) for the nanobody artifact at ``path``."""
    try:
        artifact = json.loads(Path(path).read_text())
    except Exception as error:
        return False, {"problems": [f"unreadable: {type(error).__name__}: {error}"[:160]]}
    if not isinstance(artifact, dict):
        return False, {"problems": ["artifact is not a JSON object"]}
    submissions = artifact.get("submissions") or []
    sub_stages = sorted({str(item.get("publication_stage"))
                         for item in submissions if isinstance(item, dict)})
    producer = artifact.get("producer")
    producer_name = producer.get("name") if isinstance(producer, dict) else None
    evidence = {
        "artifact_epoch": artifact.get("epoch"),
        "artifact_kind": artifact.get("artifact_kind"),
        "publication_stage": artifact.get("publication_stage"),
        "status": artifact.get("status"),
        "submission_stages": sub_stages,
        "sequence_pool_size": artifact.get("sequence_pool_size"),
        "producer_name": producer_name,
    }
    problems = []
    if artifact.get("artifact_kind") != EXPECTED_ARTIFACT_KIND:
        problems.append(f"artifact_kind={artifact.get('artifact_kind')!r}")
    if artifact.get("publication_stage") not in EXPECTED_PUBLICATION_STAGES:
        problems.append(f"publication_stage={artifact.get('publication_stage')!r}")
    if artifact.get("status") not in EXPECTED_PUBLICATION_STAGES:
        problems.append(f"status={artifact.get('status')!r}")
    odd = [stage for stage in sub_stages if stage not in EXPECTED_PUBLICATION_STAGES]
    if odd:
        problems.append(f"submission publication_stage={odd}")
    if producer is not None and producer_name not in EXPECTED_PRODUCER_NAMES:
        problems.append(f"producer.name={producer_name!r}")
    evidence["problems"] = problems
    return not problems, evidence


def refuse_unexpected_enabled(a) -> bool:
    if getattr(a, "refuse_unexpected_producer", False):
        return True
    return (ROOT / REFUSE_TOGGLE_NAME).exists()


def pi_alert(level: str, message: str) -> None:
    try:
        with PI_ALERTS.open("a") as handle:
            handle.write(time.strftime("%Y-%m-%d %H:%M:%S %z")
                         + f" [{level}] {message}\n")
    except OSError:
        pass


def check_producer_identity(a, epoch: int, nanobody: Path) -> bool:
    """Phase A always; Phase B only when enabled. Returns True when the
    artifact was replaced by the last expected one (commit must then pass
    --allow-stale-nanobody)."""
    ok, evidence = artifact_provenance(nanobody)
    if ok:
        try:
            shutil.copy2(nanobody, ROOT / LAST_EXPECTED_NAME)
        except OSError as error:
            say(event="last_expected_save_failed", err=str(error)[:120])
        say(event="producer_identity_ok", epoch=epoch,
            stage=evidence["publication_stage"],
            pool=evidence["sequence_pool_size"])
        return False
    refuse = refuse_unexpected_enabled(a)
    say(event="UNEXPECTED_PRODUCER", epoch=epoch, refuse_enabled=refuse, **evidence)
    pi_alert("ERROR", f"UNEXPECTED_PRODUCER epoch={epoch} "
                      f"problems={evidence['problems']} refuse_enabled={refuse}")
    if not refuse:
        return False
    # Phase B: keep the refused artifact for forensics, then substitute.
    try:
        refused_dir = ROOT / "archive" / f"epoch{epoch}"
        refused_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nanobody, refused_dir / "nanobody_result.refused.json")
    except OSError:
        pass
    last = ROOT / LAST_EXPECTED_NAME
    last_ok, last_evidence = artifact_provenance(last) if last.is_file() else (False, {})
    if last_ok:
        shutil.copy2(last, nanobody)
        say(event="unexpected_producer_refused", epoch=epoch, using="last_expected",
            using_epoch=last_evidence.get("artifact_epoch"))
        pi_alert("ERROR", f"UNEXPECTED_PRODUCER refused at epoch {epoch}; committing "
                          f"last expected artifact from epoch {last_evidence.get('artifact_epoch')}")
        return True
    nanobody.write_text(json.dumps({
        "schema_version": 4, "artifact_kind": EXPECTED_ARTIFACT_KIND,
        "status": "refused", "publication_stage": "refused", "epoch": epoch,
        "submissions": [], "sequence_pool": [], "sequence_pool_size": 0,
        "unassigned_uids": []}))
    say(event="unexpected_producer_refused", epoch=epoch, using="none",
        why="no previously expected artifact on disk; no nanobody this epoch")
    pi_alert("ERROR", f"UNEXPECTED_PRODUCER refused at epoch {epoch}; no expected "
                      "artifact available, committing no nanobody")
    return True


def commit_reserve(uids: int) -> int:
    """Parallel commit/repair allowance plus the validator blackout."""
    return submission_timing.commit_reserve(uids)


def say(**f):
    print(json.dumps(dict(at=time.strftime("%H:%M:%S"), **f)), flush=True)


LOCAL = ["bash", "-lc"]


def stream(where, command, timeout=900):
    """Run a command and echo its stdout as it arrives.

    `run` captures, so a twelve-minute commit run showed nothing at all until
    it was over -- no way to tell a slow epoch from a wedged one while it still
    mattered. The commits are the one step worth watching live.
    """
    if timeout <= 0:
        say(event="command_timeout", timeout_seconds=timeout)
        return 124, []
    argv = where + ([command] if where is LOCAL else ["-n", command])
    child = subprocess.Popen(argv, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                             start_new_session=True)
    lines = []
    pending = queue.Queue()
    def read_lines():
        try:
            for line in child.stdout:
                pending.put(line)
        finally:
            pending.put(None)
    threading.Thread(target=read_lines, daemon=True).start()
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                say(event="command_timeout", timeout_seconds=timeout)
                return 124, lines
            try:
                line = pending.get(timeout=min(1.0, remaining))
            except queue.Empty:
                continue
            if line is None:
                child.wait(timeout=max(.01, deadline - time.monotonic()))
                return child.returncode, lines
            line = line.rstrip()
            if line:
                lines.append(line)
                print(line, flush=True)
    except subprocess.TimeoutExpired:
        say(event="command_timeout", timeout_seconds=timeout)
        return 124, lines
    finally:
        from nanobody_opt.epoch_deadline import _stop_group
        _stop_group(child)


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
        "import json\n"
        "def rc(e):\n"
        "    try:\n"
        "        o=json.load(open('/opt/nova-submit/reaction_count.json'))\n"
        "        return int(o['count']) if e>=int(o['from_epoch']) else int(o.get('before',5))\n"
        "    except Exception:\n"
        "        return 44 if e>=25692 else 5\n"
        "import bittensor as bt, json;"
        "s=bt.Subtensor(network='finney');b=s.get_current_block();L=s.tempo(68)+1;"
        "e=b//L;st=e*L;h=s.determine_block_hash(st);"
        "mg=s.metagraph(68, lite=True);"
        "import pathlib;"
        "ck=json.load(open(pathlib.Path.home()/'.bittensor/wallets/mining_ops/coldkeypub.txt'))['ss58Address'];"
        "crew=sorted((int(u),str(k)) for u,k,c in zip(mg.uids,mg.hotkeys,mg.coldkeys) if str(c)==ck);"
        "print(json.dumps(dict(block=b,epoch=e,start=st,end=st+L,"
        "reaction=int(h,16)%rc(e)+1,crew=crew)))")
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


def artifact_confirmation(path: Path, epoch: int, minimum: int) -> tuple[int, int]:
    """(confirmed, total) submissions in this epoch's artifact.

    A submission is confirmed when it carries at least ``minimum`` exact
    observations.  The producer repeat-scores its finalists within the epoch;
    committing the first provisional artifact instead (1-5 minutes after the
    window opened) put 53% of our tickets on 1-5 observations, and those made
    the top 10 at 12.3% against ~20% for 6-40 observations (epochs
    25335-25394).
    """
    try:
        artifact = json.loads(Path(path).read_text())
    except Exception:
        return 0, 0
    if int(artifact.get("epoch", -1)) != epoch:
        return 0, 0
    submissions = artifact.get("submissions") or []
    confirmed = sum(1 for item in submissions
                    if int(item.get("observation_count") or 0) >= minimum)
    return confirmed, len(submissions)


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
    parser.add_argument("--minimum-compound-score", type=float, default=1.53,
                        help="fallback withholding threshold for a reaction not "
                             "named in --minimum-compound-score-map. A submitted "
                             "molecule becomes public immediately whether it wins "
                             "or not, so releasing below the field's current "
                             "parity permanently burns that chemistry for a "
                             "near-zero win chance. The nanobody half always "
                             "commits regardless.")
    parser.add_argument("--minimum-compound-score-map", default="",
                        help="per-reaction override as reaction:score,reaction:score "
                             "-- each reaction's competitive field differs (median "
                             "winning submission_score across 38-45 tracked epochs: "
                             "rxn1 1.62, rxn2 1.93, rxn3 1.56, rxn4 1.83, rxn5 1.93), "
                             "so one global threshold either wastes chemistry on "
                             "easy reactions or submits certain losses on hard ones. "
                             "A reaction not listed here falls back to "
                             "--minimum-compound-score.")
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
    parser.add_argument("--late-commit", action=argparse.BooleanOptionalAction, default=True,
                        help="hold the commit until just before the safe deadline so it carries the "
                             "producer's in-epoch repeat scoring (--no-late-commit: first artifact)")
    parser.add_argument("--late-commit-margin-seconds", type=int,
                        default=submission_timing.LATE_COMMIT_MARGIN_SECONDS,
                        help="stop holding this long before the commit deadline, leaving time for "
                             "the final pull and the crew refresh")
    parser.add_argument("--confirmation-min-observations", type=int, default=8,
                        help="observation depth reported as 'deep' in the late-commit log")
    parser.add_argument("--refuse-unexpected-producer", action="store_true",
                        help="Phase B of the producer-identity check: do not commit an "
                             "artifact the real publisher did not write. Off by default; "
                             f"the file <root>/{REFUSE_TOGGLE_NAME} enables it too")
    parser.add_argument("--dry-run", action="store_true")
    a = parser.parse_args()
    global PI, ROOT
    if a.on_submit_host:
        PI = LOCAL
    if a.root:
        ROOT = Path(a.root)
    covered = set()
    for part in a.reactions.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            covered |= set(range(int(lo), int(hi) + 1))
        elif part:
            covered.add(int(part))
    try:                      # cover everything the validator's database holds (reaction_count.json)
        covered |= set(range(1, int(json.loads((ROOT / "reaction_count.json").read_text())["count"]) + 1))
    except Exception:
        pass
    score_floor = {}
    for pair in a.minimum_compound_score_map.split(","):
        pair = pair.strip()
        if not pair:
            continue
        reaction_key, _, value = pair.partition(":")
        score_floor[int(reaction_key)] = float(value)
    a.score_floor = score_floor  # attached so handle_epoch() can read it via `a`
    say(event="start", covered=sorted(covered), dry_run=a.dry_run,
        score_floor={**{r: a.minimum_compound_score for r in covered - score_floor.keys()},
                    **score_floor})

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
        # Keep CPU preparation early; reducing the parallel commit reserve
        # must not move molecule/crew preparation into the last few minutes.
        refresh_at = end - submission_timing.commit_reserve(ASSUMED_MAX_CREW, serial=True) - MIN_PRODUCER_WAIT_BLOCKS
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


def refresh_crew_for_commit(a, view, epoch: int, crew_path: Path) -> dict:
    """Replace the pre-pull roster with the last possible chain snapshot.

    Pulling used to take long enough for a UID to turn over between the
    window-open metagraph and ``commit_batch``: the epoch-25296 rehearsal
    planned 13 but the ownership check could still send only 12. The artifact
    pool is UID-independent, so refresh the roster after the pull and let the
    Pi bind sequences to the actual final crew.
    """
    try:
        fresh = chain(a.chain_python)
    except Exception as error:
        say(event="crew_refresh_failed", err=str(error)[:160])
        return view
    if int(fresh["epoch"]) != int(epoch):
        say(event="crew_refresh_rolled", wanted=epoch, have=fresh["epoch"])
        return view
    before = len(view["crew"])
    view = dict(view, crew=fresh["crew"], block=fresh["block"])
    crew_path.write_text(json.dumps(view["crew"]))
    publish(a, ["crew.json"])
    say(event="crew_refreshed", epoch=epoch, before=before, after=len(view["crew"]))
    return view


def cached_nanobody_available(a) -> bool:
    """Whether the submit host still has a non-empty verified artifact.

    A producer outage must not turn into an empty epoch.  ``pi_submit`` already
    knows how to rebind a stale UID-independent sequence pool to the live crew;
    this probe only decides whether there is a local pool for that fallback.
    It deliberately does not parse or print the artifact here -- the normal
    submission preflight remains the authority on its contents.
    """
    if a.on_submit_host:
        return (ROOT / "nanobody_result.json").is_file() \
            and (ROOT / "nanobody_result.json").stat().st_size > 0
    cached = run(PI, "test -s /opt/nova-submit/nanobody_result.json", timeout=30)
    return cached.returncode == 0



# 2026-10-03 (user decision): each reaction's release floor is the median of
# that reaction's last FLOOR_WINDOW winning final_scores on the public
# leaderboard, not a static map. Winners are cached per epoch in
# archive/winner_cache.json (an epoch is cached only once it has a scored
# winner), so a warm cache costs ~1 fetch per epoch. Falls back to the static
# --minimum-compound-score-map value with fewer than FLOOR_MIN_SAMPLES winners
# or when the API is unreachable; the network walk is time-boxed so it can
# never eat the epoch's commit window.
LEADERBOARD_API = "https://compound-api.metanova-labs.ai/api/competitions/leaderboard/{epoch}/molecules?limit=10000"
FLOOR_WINDOW = 5
FLOOR_MIN_SAMPLES = 3
FLOOR_LOOKBACK_EPOCHS = 150
FLOOR_FETCH_BUDGET_SECONDS = 60
FLOOR_BURST_LOOKBACK = 5
FLOOR_BURST_MIN = 3
FLOOR_OPTIMISM_MARGIN = 0.10  # user 2026-10-03: our predictions run ~0.1 above realized
# User-pinned floors override the rolling rule (2026-10-03: rxn4 recent realized
# winners ~1.64, so submit at 1.64 + 0.10 = 1.74).
FLOOR_PINNED: dict = {r: 0.0 for r in range(1, 45)}  # 2026-10-13 SANQI: submit whatever the live search found, every reaction


def _epoch_winner(epoch: int, cache: dict, deadline: float):
    key = str(epoch)
    if key in cache:
        return cache[key]
    if time.time() > deadline:
        return None
    import urllib.request
    try:
        request = urllib.request.Request(LEADERBOARD_API.format(epoch=epoch),
                                         headers={"User-Agent": "nova-pi-loop/1"})
        with urllib.request.urlopen(request, timeout=10) as response:
            rows = json.load(response).get("data") or []
    except Exception:
        return None
    scored = [r for r in rows if (r.get("final_score") or -999) > -999]
    reactions = {}
    for row in rows:
        for molecule in row.get("molecules") or []:
            parts = str(molecule.get("name", "")).split(":")
            if len(parts) > 1 and parts[0] == "rxn" and parts[1].isdigit():
                reactions[int(parts[1])] = reactions.get(int(parts[1]), 0) + 1
    if not scored or not reactions:
        return None  # not scored yet -- do not cache, look again next time
    entry = {"reaction": max(reactions, key=reactions.get),
             "winner": max(float(r["final_score"]) for r in scored)}
    cache[key] = entry
    return entry


def _portfolio_digest(molecules) -> str:
    names = sorted(str(m.get("name") or "") for m in molecules)
    return hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]


def _epoch_rival_rows(epoch: int, cache: dict, deadline: float,
                      require_identity: bool = False):
    """Per-submission public rows, optionally with exact portfolio identity."""
    key = str(epoch)
    cached = cache.get(key)
    if cached is not None and (not require_identity or not cached or all(
            "uid" in row and "names_sha256" in row for row in cached)):
        return cached
    if time.time() > deadline:
        return None
    import urllib.request
    try:
        request = urllib.request.Request(LEADERBOARD_API.format(epoch=epoch),
                                         headers={"User-Agent": "nova-pi-loop/1"})
        with urllib.request.urlopen(request, timeout=10) as response:
            rows = json.load(response).get("data") or []
    except Exception:
        return None
    out = []
    for row in rows:
        coldkey = row.get("coldkey")
        score = row.get("final_score")
        if not coldkey or (score or -999) <= -999:
            continue
        for molecule in row.get("molecules") or []:
            parts = str(molecule.get("name", "")).split(":")
            if len(parts) > 1 and parts[0] == "rxn" and parts[1].isdigit():
                out.append({"coldkey": coldkey, "reaction": int(parts[1]),
                            "score": float(score), "uid": row.get("uid"),
                            "names_sha256": _portfolio_digest(row.get("molecules") or [])})
                break
    cache[key] = out
    return out


def _owned_score_evidence(reaction: int, cache: dict, deadline: float,
                          archive_root: Path | None = None) -> tuple[set[str], list[dict]]:
    """Match our archived exact portfolios to public results.

    This provides both the coldkey(s) to exclude from the rival model and a
    direct estimate of raw offline minus realised score. Matching uses epoch,
    reaction, UID, and the exact 20-name digest, so UID recycling cannot label
    somebody else's submission as ours.
    """
    archive_root = archive_root or ROOT / "archive"
    saved = cache.get("_self_calibration") or {}
    owned: set[str] = {str(value) for value in saved.get("coldkeys") or []}
    saved_evidence = saved.get("evidence") or []
    evidence_by_key = {
        (int(row["reaction"]), int(row["epoch"]), str(row["portfolio"])): row
        for row in saved_evidence
        if all(key in row for key in ("reaction", "epoch", "portfolio", "optimism"))
    }
    artifacts = sorted(
        archive_root.glob("epoch*/compound_fleet.json"),
        key=lambda path: int(path.parent.name.removeprefix("epoch")),
        reverse=True,
    )
    matched_epochs = 0
    for path in artifacts:
        if matched_epochs >= 12 or time.time() > deadline:
            break
        try:
            artifact = json.loads(path.read_text())
            artifact_reaction = int(artifact.get("reaction", -1))
            artifact_epoch = int(artifact.get("epoch", -1))
        except (OSError, ValueError, TypeError):
            continue
        if artifact_reaction != reaction:
            continue
        public = _epoch_rival_rows(artifact_epoch, cache, deadline,
                                   require_identity=True) or []
        index = {
            (int(row["reaction"]), int(row["uid"]), row["names_sha256"]): row
            for row in public if row.get("uid") is not None
        }
        found = False
        for submission in artifact.get("submissions") or []:
            key = (reaction, int(submission["uid"]),
                   _portfolio_digest(submission.get("molecules") or []))
            row = index.get(key)
            if row is None:
                continue
            found = True
            owned.add(str(row["coldkey"]))
            raw = float(submission.get("selection_score") or 0.0)
            evidence_by_key[(reaction, artifact_epoch, key[2])] = {
                "reaction": reaction,
                "epoch": artifact_epoch,
                "portfolio": key[2],
                "optimism": raw - float(row["score"]),
            }
        if found:
            matched_epochs += 1
    saved_evidence = sorted(
        evidence_by_key.values(), key=lambda row: (row["epoch"], row["portfolio"]))[-256:]
    cache["_self_calibration"] = {
        "coldkeys": sorted(owned),
        "evidence": saved_evidence,
    }
    evidence = [row for row in saved_evidence if int(row["reaction"]) == reaction]
    return owned, evidence


def _calibrated_optimism(evidence: list[dict], prior: float,
                         prior_epochs: int = 2) -> tuple[float, dict]:
    """Empirical-Bayes estimate of our raw-minus-realised portfolio gap."""
    by_epoch: dict[int, list[float]] = {}
    for row in evidence:
        by_epoch.setdefault(int(row["epoch"]), []).append(float(row["optimism"]))
    epoch_values = [statistics.median(by_epoch[epoch]) for epoch in sorted(by_epoch)][-8:]
    if not epoch_values:
        return prior, {"own_evidence_epochs": 0, "own_evidence_portfolios": 0}
    observed = statistics.median(epoch_values)
    n = len(epoch_values)
    estimate = (n * observed + prior_epochs * prior) / (n + prior_epochs)
    estimate = min(0.30, max(0.0, estimate))
    return estimate, {
        "own_evidence_epochs": n,
        "own_evidence_portfolios": len(evidence),
        "observed_optimism": round(observed, 4),
    }


def dynamic_floor(reaction: int, epoch: int, static: float) -> tuple[float, dict]:
    """Calibrated own score needed to beat the projected external winner."""
    if reaction in FLOOR_PINNED:
        return FLOOR_PINNED[reaction], {"source": "user_pinned", "static": static}
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "tools"))
    import rival_margin
    cache_path = ROOT / "archive" / "rival_cache.json"
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    deadline = time.time() + FLOOR_FETCH_BUDGET_SECONDS
    owned, own_evidence = _owned_score_evidence(reaction, cache, deadline)
    optimism, calibration = _calibrated_optimism(
        own_evidence, rival_margin.CURSE)
    optimism_by_epoch: dict[int, list[float]] = {}
    for row in own_evidence:
        optimism_by_epoch.setdefault(int(row["epoch"]), []).append(float(row["optimism"]))
    optimism_samples = [
        statistics.median(optimism_by_epoch[past])
        for past in sorted(optimism_by_epoch)
    ][-8:]
    rows = []
    for past in range(epoch - 1, max(0, epoch - 1 - FLOOR_LOOKBACK_EPOCHS), -1):
        r = _epoch_rival_rows(past, cache, deadline)
        if r:
            # rival_margin needs each row's epoch (KeyError 'epoch' killed 25501/25502)
            rows.extend({**x, "epoch": past} for x in r)
    external_rows = [row for row in rows if str(row.get("coldkey")) not in owned]
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache))
        tmp.replace(cache_path)
    except OSError:
        pass
    try:
        margin, info = rival_margin.required_margin(
            reaction, external_rows, epoch, optimism=optimism,
            optimism_samples=optimism_samples)
    except Exception as error:  # a floor bug must never block the commit
        return static, {"static": static, "source": "static_fallback_error",
                        "err": f"{type(error).__name__}: {error}"[:160]}
    if margin is None:
        return static, {**info, "static": static, "source": "static_fallback"}
    return margin, {**info, **calibration, "own_coldkeys_excluded": len(owned),
                    "static": static, "source": "rival_margin"}


def sanqi_epoch(epoch: int) -> bool:
    """True from the first epoch drawn over the SANQI database (reaction_count.json `from_epoch`).

    From then on the boxes race EVERY reaction 1-44, including 1-5, so their exports are pulled for
    every drawn reaction. (This was `reaction > 5`, which silently dropped the sprint results of the
    ~11% of epochs that draw reaction 1-5; the legacy rxn:1..5 names are the same molecules in the
    SANQI database, see tools/sanqi_pull.py.) Defaults to 25692 when the file is unreadable."""
    try:
        return int(epoch) >= int(json.loads((ROOT / "reaction_count.json").read_text())["from_epoch"])
    except Exception:
        return int(epoch) >= 25692


def sanqi_pull(reaction: int, timeout: float = 240) -> None:
    """Pull + merge the SANQI worker boxes' exported banks into bank.sqlite. Never raises."""
    try:
        got = subprocess.run([sys.executable, str(ROOT / "tools/sanqi_pull.py"), "--reaction", str(reaction)],
                             capture_output=True, text=True, timeout=timeout)
        for line in got.stdout.strip().splitlines():
            if line.startswith("{"):
                say(**json.loads(line))
        if got.returncode:
            say(event="sanqi_pull_failed", rc=got.returncode, err=(got.stderr or got.stdout)[-160:])
    except Exception as error:
        say(event="sanqi_pull_failed", err=f"{type(error).__name__}: {error}"[:160])


def run_preparation(command, *, cutoff_at, timeout, capture_output=True, text=True):
    """Bound a preparation process tree before its epoch's collection cutoff.

    Two seconds are reserved for the existing one-second TERM grace and KILL.
    Scientific gates and selection are unchanged; an expired result is refused.
    """
    import math
    from nanobody_opt.epoch_deadline import EpochDeadlineExceeded, run as bounded_run
    if not capture_output or not text:
        raise ValueError('preparation requires captured text output')
    if not math.isfinite(cutoff_at) or not math.isfinite(timeout):
        raise ValueError('preparation requires finite deadlines')

    class Cutoff:
        def remaining(self):
            remaining = cutoff_at - time.time() - 2.0
            if remaining <= 0:
                raise EpochDeadlineExceeded('preparation collection cutoff reached')
            return remaining

    try:
        Cutoff().remaining()  # Refuse to spawn after the cutoff.
        if timeout <= 0:
            raise EpochDeadlineExceeded('preparation time budget exhausted')
        return bounded_run(command, deadline=Cutoff(), timeout=timeout, check=False,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except EpochDeadlineExceeded:
        say(event='preparation_deadline_exceeded', cutoff_at=cutoff_at)
        pi_alert('ERROR', 'Epoch preparation exceeded its cutoff; process tree stopped and molecules withheld')
        return subprocess.CompletedProcess(command, 124, '', 'preparation_deadline_exceeded')


def handle_epoch(a, view, epoch, block, end, covered, state):
    # Anchor before any bank preparation. Adding the original block budget
    # to a later wall clock would silently extend the real chain deadline.
    epoch_end_at = time.time() + max(0, end - block) * submission_timing.BLOCK_SECONDS
    if True:
        producer_stopped = state["producer_stopped"]
        crew_path = ROOT / ("release" if a.on_submit_host else "data/release") / "crew.json"
        crew_path.parent.mkdir(parents=True, exist_ok=True)
        crew_path.write_text(json.dumps(view["crew"]))

        # molecules only when we actually hold a rescored bank for this reaction
        have_molecules = False
        if view["reaction"] in covered:
            static_floor = a.score_floor.get(view["reaction"], a.minimum_compound_score)
            try:
                floor, floor_info = dynamic_floor(view["reaction"], epoch, static_floor)
            except Exception as error:  # the floor must never cost the epoch (25501-25503)
                floor, floor_info = static_floor, {"source": "static_fallback_error",
                                                   "err": f"{type(error).__name__}: {error}"[:160]}
            # explicit fields win: rival_margin's info carries its own 'reaction' key,
            # which made say() raise TypeError at 25503
            say(**{**floor_info, "event": "dynamic_floor", "epoch": epoch,
                   "reaction": view["reaction"], "floor": floor})
            if sanqi_epoch(epoch) and a.on_submit_host:
                sanqi_pull(view["reaction"])        # live-search results from the racing boxes
            made = run_preparation(
                [sys.executable, str(ROOT / "tools/make_fleet.py"),
                 "--epoch", str(epoch), "--reaction", str(view["reaction"]),
                 "--crew", str(crew_path), "--python", a.rdkit_python,
                 "--minimum-score", str(floor),
                 "--prediction-offset", str(floor_info.get("optimism", 0.0))]
                + (["--bank", str(ROOT / "bank.sqlite"),
                    "--out", str(ROOT / "release"),
                    "--fleet-tree", str(ROOT)] if a.on_submit_host else []),
                capture_output=True, text=True, timeout=1200, cutoff_at=epoch_end_at - max(commit_reserve(len(view["crew"])), BLACKOUT_BLOCKS + 1) * submission_timing.BLOCK_SECONDS)
            for line in made.stdout.strip().splitlines():
                if line.startswith("{"):
                    say(**json.loads(line))
            have_molecules = made.returncode == 0
        else:
            say(event="reaction_not_covered", reaction=view["reaction"])

        # A one-shot check at window-open misses later improvement: the bank
        # keeps growing (nova-bank-pull.timer, every 5 minutes) during the ~9
        # minutes the Pi then spends only waiting on the nanobody producer,
        # and nothing ever looks again. Confirmed live: epoch 25474 missed its
        # floor here by 0.093 (1.4668 vs 1.56), then the box's own bank read
        # 1.565 just seven minutes later -- a result that would have cleared.
        # Re-run the same check once more in the BACKGROUND rather than
        # serially before commit: the final commit already sits inside a
        # ~120s margin before the chain deadline (late_commit_margin_seconds),
        # and make_fleet.py alone took ~2 minutes above, so doing this inline
        # would itself risk missing the deadline. Write to a per-epoch
        # directory, never the shared release/ path, so this can never race a
        # write against the primary check above (make_fleet.py's own output
        # write is a plain json.dump, not atomic) if the thread is still
        # running when the NEXT epoch starts its own first check.
        recheck_thread = None
        recheck_result: dict = {}
        recheck_out = ROOT / f"release-recheck-{epoch}"
        from compound_window import configured_minutes
        try:
            late_compound = bool(configured_minutes(json.loads((ROOT / "reaction_count.json").read_text()), epoch))
        except (OSError, ValueError, TypeError):
            late_compound = False
        if (late_compound or not have_molecules) and view["reaction"] in covered and a.on_submit_host:
            # A SINGLE fixed-delay recheck is not enough: confirmed live
            # 2026-10-02, epoch 25477 (reaction 3) -- one recheck at a fixed
            # 5-minute delay found the bank still below baseline, logged
            # nothing (no "cleared", but also no "still didn't clear" event
            # existed to say so -- fixed below too), and the bank crossed
            # the floor sometime in the ~3 minutes AFTER that single check,
            # unseen, right before commit. Retry on a cadence instead,
            # roughly matching nova-bank-pull.timer's 5-minute refresh, for
            # as long as the epoch's own deadline safely allows.
            def _recheck(reaction=view["reaction"], floor=floor):
                reserve = commit_reserve(len(view["crew"]))
                recheck_deadline = epoch_end_at - reserve * submission_timing.BLOCK_SECONDS
                safety_margin = 90  # stop attempting this long before commit
                attempt = 0
                # Every attempt must FINISH before the commit step collects this
                # thread (join(timeout=0) just before hold_until); an attempt
                # takes ~5 min, so the old loop was always mid-attempt at
                # collection and every result was discarded (25490-25500 all
                # logged release_recheck_still_running; 25497/25500 missed
                # their floors by 0.007).
                collect_at = recheck_deadline - getattr(a, "late_commit_margin_seconds", submission_timing.LATE_COMMIT_MARGIN_SECONDS) - 30
                if late_compound:
                    from compound_window import release_after, build_timeout
                    try:
                        ready_at = release_after(ROOT, epoch)
                    except (OSError, ValueError, KeyError, TypeError):
                        say(event="late_compound_payload_unavailable", epoch=epoch)
                        return
                    say(event="late_compound_rebuild_scheduled", epoch=epoch,
                        seconds_until_export=round(max(0, ready_at-time.time())),
                        seconds_until_collect=round(max(0, collect_at-time.time())))
                    while time.time() < min(ready_at, collect_at):
                        time.sleep(min(30, max(0, min(ready_at, collect_at)-time.time())))
                    if build_timeout(collect_at, time.time()) < 30:
                        say(event="late_compound_rebuild_skipped", epoch=epoch)
                        return
                    sanqi_pull(reaction, timeout=min(45, build_timeout(collect_at, time.time())))
                    timeout = build_timeout(collect_at, time.time())
                    if timeout < 20:
                        return
                    try:
                        result = run_preparation(
                            [sys.executable, str(ROOT / "tools/make_fleet.py"),
                             "--epoch", str(epoch), "--reaction", str(reaction),
                             "--crew", str(crew_path), "--python", a.rdkit_python,
                             "--minimum-score", str(floor),
                             "--prediction-offset", str(floor_info.get("optimism", 0.0)),
                             "--bank", str(ROOT / "bank.sqlite"), "--out", str(recheck_out),
                             "--fleet-tree", str(ROOT)],
                            capture_output=True, text=True, timeout=timeout, cutoff_at=collect_at)
                    except subprocess.TimeoutExpired:
                        say(event="late_compound_rebuild_timeout", epoch=epoch)
                        return
                    if time.time() < collect_at:
                        recheck_result.update(returncode=result.returncode,
                                              stdout=result.stdout, attempts=1)
                    say(event="late_compound_rebuild_finished", epoch=epoch,
                        rc=result.returncode, before_cutoff=time.time() < collect_at)
                    return
                last_duration = 330.0
                time.sleep(150)  # first look: give one bank-pull cycle a head start
                while time.time() + last_duration * 1.15 < min(collect_at, recheck_deadline - safety_margin):
                    attempt += 1
                    started = time.time()
                    if sanqi_epoch(epoch):
                        sanqi_pull(reaction)
                    r = run_preparation(
                        [sys.executable, str(ROOT / "tools/make_fleet.py"),
                         "--epoch", str(epoch), "--reaction", str(reaction),
                         "--crew", str(crew_path), "--python", a.rdkit_python,
                         "--minimum-score", str(floor),
                         "--prediction-offset", str(floor_info.get("optimism", 0.0)),
                         "--bank", str(ROOT / "bank.sqlite"),
                         "--out", str(recheck_out), "--fleet-tree", str(ROOT)],
                        capture_output=True, text=True, timeout=1200, cutoff_at=collect_at)
                    recheck_result["returncode"] = r.returncode
                    recheck_result["stdout"] = r.stdout
                    recheck_result["attempts"] = attempt
                    last_duration = time.time() - started
                    if r.returncode == 0:
                        return
                    time.sleep(60)
            recheck_thread = threading.Thread(target=_recheck, daemon=True)
            recheck_thread.start()

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
        deadline = epoch_end_at - reserve * submission_timing.BLOCK_SECONDS
        say(event="wait_budget", uids=len(view["crew"]), reserve=reserve,
            blocks_to_wait=max(0, int((deadline - time.time()) / 12)))
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

        # Then hold until just before the safe deadline, re-pulling so the
        # commit carries the producer's in-epoch repeat scoring.  The first
        # artifact for an epoch is provisional and built before the producer's
        # confirmation stage; committing it 1-5 minutes after the window opened
        # put 53% of our tickets on 1-5 exact observations, which made the top
        # 10 at 12.3% against ~20% for 6-40 observations (epochs 25335-25394).
        # Committing later only loses exact final-score ties to rivals that
        # commit in between, and every rival ticket in that window was
        # committed earlier than ours anyway (3 ties newly lost in 60 epochs).
        if not stale_ok and a.on_submit_host and a.late_commit:
            hold_until = deadline - a.late_commit_margin_seconds
            local = Path("/opt/nova-submit/nanobody_result.json")
            while time.time() < hold_until:
                confirmed, total = artifact_confirmation(local, epoch, a.confirmation_min_observations)
                say(event="late_commit_wait", seconds_left=round(hold_until - time.time()),
                    deep=confirmed, total=total, minimum=a.confirmation_min_observations)
                time.sleep(min(60, max(1, hold_until - time.time())))
                producer_epoch_here(a, epoch, view["reaction"])
            confirmed, total = artifact_confirmation(local, epoch, a.confirmation_min_observations)
            say(event="late_commit_ready", deep=confirmed, total=total,
                minimum=a.confirmation_min_observations)

        pull = run(PI, "cd /opt/nova-submit && ./.pixi/envs/default/bin/python pi_submit.py "
                       f"--epoch {epoch} --reaction {view['reaction']} --pull-only", timeout=300)
        if pull.returncode:
            if stale_ok and cached_nanobody_available(a):
                # This is exactly the last-resort path described above: the
                # producer is unavailable, but a previously validated ranked
                # pool is still much better than submitting nothing.  The
                # final pi_submit invocation receives --allow-stale-nanobody,
                # rebinds that pool to the fresh crew, and runs every ordinary
                # plaintext and validator-gate preflight before committing.
                say(event="cached_nanobody_fallback", epoch=epoch,
                    why="fresh pull failed; using last verified sequence pool")
            else:
                say(event="skip_epoch", why="pull failed and no verified cache exists")
                return
        else:
            say(event="pulled")
            receipt = submission_timing.artifact_receipt(
                Path("/opt/nova-submit/nanobody_result.json"), epoch)
            say(event="submission_handoff", epoch=epoch, **receipt)
            if receipt.get("handoff_policy") == submission_timing.POLICY and not receipt["confirmation_complete_at_start"]:
                pi_alert("ERROR", f"epoch {epoch}: final Pi input lacks completed confirmation; "
                         "committing the available validated fallback")

        # Who wrote what we are about to commit (P1 / R4). Must never cost the
        # epoch: any error in the check itself is logged and ignored.
        try:
            if a.on_submit_host and check_producer_identity(
                    a, epoch, Path("/opt/nova-submit/nanobody_result.json")):
                stale_ok = True
        except Exception as error:
            say(event="producer_identity_check_error",
                err=f"{type(error).__name__}: {error}"[:200])

        # The initial roster preceded a provider lookup and an artifact pull.
        # Refresh after both, immediately before planning, so a recycled UID is
        # replaced in the fleet rather than merely dropped by the later
        # ownership check. Slot purchases are frozen around this window too,
        # but other miners can still churn the subnet.
        view = refresh_crew_for_commit(a, view, epoch, crew_path)

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

        # Collect the background recheck started above, non-blocking: use it
        # only if it already finished by now (it has had the whole producer
        # wait, typically ~9 minutes, against its own 300s delay + ~2 minute
        # run -- comfortably done). If it is still running this close to
        # commit, leave it running and leave have_molecules as the original
        # decision; never delay the chain commit waiting on it.
        if recheck_thread is not None:
            recheck_thread.join(timeout=0)
            if recheck_thread.is_alive():
                # Still running its own subprocess inside recheck_out -- never
                # rmtree a directory a live process is writing into. Leave it;
                # it is daemon and per-epoch-named, so it cannot collide with
                # any later epoch's own recheck directory.
                say(event="release_recheck_still_running", epoch=epoch)
            else:
                if recheck_result.get("returncode") == 0:
                    for line in recheck_result.get("stdout", "").strip().splitlines():
                        if line.startswith("{"):
                            say(**json.loads(line))
                    recheck_fleet = recheck_out / "compound_fleet.json"
                    if recheck_fleet.is_file():
                        shutil.copy(recheck_fleet, ROOT / "release" / "compound_fleet.json")
                        have_molecules = True
                        publish(a, ["compound_fleet.json"])
                        say(event="release_recheck_cleared", epoch=epoch,
                            reaction=view["reaction"],
                            why="bank improved during the producer wait")
                elif "attempts" in recheck_result:
                    # Previously silent -- made this case undiagnosable
                    # without inferring it from an absence of log lines
                    # (confirmed live 2026-10-02, epoch 25477: ran, found
                    # nothing, logged nothing, looked identical to "never
                    # ran" from the log alone).
                    say(event="release_recheck_no_change", epoch=epoch,
                        reaction=view["reaction"],
                        attempts=recheck_result["attempts"])
                shutil.rmtree(recheck_out, ignore_errors=True)

        # Bound the whole local process group, including a silent RPC hang,
        # before the blackout. It can never occupy the following epoch.
        commit_seconds = max(0, epoch_end_at - (BLACKOUT_BLOCKS + 1) * 12 - time.time())
        code, _ = stream(PI, "cd /opt/nova-submit && ./.pixi/envs/default/bin/python pi_submit.py "
                             f"--epoch {epoch} --reaction {view['reaction']}"
                             + (" --allow-stale-nanobody" if stale_ok else ""), timeout=commit_seconds)
        commit = type("R", (), {"returncode": code})()
        say(event="committed", epoch=epoch, rc=code, molecules=have_molecules)
        # nanobody_result.json (per-uid ex-ante score_p90), crew.json and
        # compound_fleet.json are overwritten every epoch, so nothing recorded
        # what we actually believed about each uid at commit time -- only 1 of
        # 361 epochs could be backtested for fleet size (2026-10-02). Keep the
        # exact committed inputs per epoch.
        if a.on_submit_host:
            archived = ROOT / "archive" / f"epoch{epoch}"
            archived.mkdir(parents=True, exist_ok=True)
            names = ["nanobody_result.json", "crew.json"] + (
                ["compound_fleet.json"] if have_molecules else [])
            for name in names:
                source = Path("/opt/nova-submit") / name
                if source.is_file():
                    shutil.copy2(source, archived / name)
            say(event="archived", epoch=epoch, files=names)

        if commit.returncode and start_command:
            import shlex
            say(event="rollback", why="commit failed on the epoch we took over")
            run(PROD, "tmux new-session -d -s nova-nanobody-publisher "
                      + shlex.quote(start_command) + "; sleep 6", timeout=180)
            state["producer_stopped"] = False
        return


if __name__ == "__main__":
    raise SystemExit(main())

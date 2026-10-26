#!/usr/bin/env python3
"""Commit every uid from the host that holds the hotkeys.

Why this exists: the commitment is ONE message per uid, `molecules|sequence`,
so whoever writes it must have both halves. Until now that forced a master and
a slave -- one producer had to hand its plaintext to the other, or read it back
out of GitHub, which is impossible because the blob is drand timelock
ciphertext sealed until its reveal round.

Here the two producers stay independent and neither is the master:

    production (GPU box)  --  nanobody_result.json  --.
                                                       >--  this host  --> chain
    bank box              --  compound_fleet.json  ---'

Both inputs are plaintext, both are pulled, and the host that signs is the one
that already custodies the keys -- so no keyfile has to live on a rental at all.
It runs here rather than on a GPU box because `timelock_wasm_wrapper` publishes
only a cp310/x86_64 wheel; the aarch64 build in ./wheels is what makes this
possible, and it must stay pinned to 0.0.2 because the validator decrypts with
that version.

Commits are placed one per block inside the end-of-epoch reserve: the chain
accepts one extrinsic per block per key, and the reserve is the only part of
the epoch where the producers have stopped rewriting their artifacts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "utils"))

NULL = "~"

#: A nanobody artifact more than this many epochs old is guaranteed to already
#: be in the validator's HF submission archive; recommitting it just re-earns
#: the "not unique for target" rejection. Bound the age so a producer outage
#: cannot silently recommit the same pool forever.
STALE_AGE_LIMIT = 3

#: A writer stamps ``updated_at_unix`` when it writes the artifact. A missing,
#: non-finite, or very old stamp is stale evidence regardless of the epoch
#: override. Generous enough for two-host clock skew, far tighter than a day.
MAX_ARTIFACT_AGE_SECONDS = 3600


def producer_endpoints(instance: dict) -> list[tuple[str, str, int]]:
    """Direct SSH first, provider proxy second, with no written-down address."""
    endpoints: list[tuple[str, str, int]] = []
    public_ip = str(instance.get("public_ipaddr") or "").strip()
    ports = instance.get("ports") or {}
    direct_port = None
    if isinstance(ports, dict):
        for mapping in ports.get("22/tcp") or []:
            if not isinstance(mapping, dict):
                continue
            try:
                direct_port = int(mapping.get("HostPort"))
            except (TypeError, ValueError):
                continue
            if direct_port > 0:
                break
    if direct_port is None:
        try:
            direct_port = int(instance.get("direct_port_start"))
        except (TypeError, ValueError):
            direct_port = None
    if public_ip and direct_port and direct_port > 0:
        endpoints.append(("direct", public_ip, direct_port))
    proxy_host = str(instance.get("ssh_host") or "").strip()
    try:
        proxy_port = int(instance.get("ssh_port"))
    except (TypeError, ValueError):
        proxy_port = None
    if proxy_host and proxy_port and proxy_port > 0:
        proxy = ("proxy", proxy_host, proxy_port)
        if (proxy_host, proxy_port) not in {(host, port) for _, host, port in endpoints}:
            endpoints.append(proxy)
    return endpoints



def pull_producer(vastai: str, label: str, remote: str, destination: Path) -> None:
    """Fetch the nanobody half from whichever box currently carries the label.

    Deliberately not a stored host:port. The producer is moving to a bigger
    rental, and vast.ai rotates ports on restart; resolving the label each run
    means a rehomed producer needs no change here at all.
    """
    raw = subprocess.run([vastai, "show", "instances", "--raw"],
                         capture_output=True, text=True, timeout=120)
    if raw.returncode:
        raise SystemExit(f"vastai failed: {raw.stderr.strip()[:200]}")
    try:
        instances = json.loads(raw.stdout)
    except ValueError:
        raise SystemExit("vastai did not return JSON")
    if isinstance(instances, dict):
        instances = instances.get("instances", [])
    matches = [i for i in instances
               if str(i.get("label") or "") == label
               and str(i.get("actual_status") or "") == "running"]
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one running instance labelled {label!r}, found {len(matches)}")
    endpoints = producer_endpoints(matches[0])
    if not endpoints:
        raise SystemExit(f"instance {matches[0].get('id')} has no resolvable ssh endpoint")
    failures = []
    route = host = ""
    port = 0
    for route, host, port in endpoints:
        done = subprocess.run(
            ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
             "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
             "-P", str(port), f"root@{host}:{remote}", str(destination)],
            capture_output=True, text=True, timeout=300)
        if done.returncode == 0:
            break
        failures.append(f"{route}:rc={done.returncode}")
    else:
        raise SystemExit(
            f"pull from {label} failed on " + ", ".join(failures)
        )
    print(json.dumps(dict(event="pulled", label=label, port=port,
                          route=route,
                          bytes=destination.stat().st_size)), flush=True)


def _load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def attest_epoch(chain, epoch: int, netuid: int) -> dict:
    """Read the authoritative epoch dimensions independently of the producer."""
    from nanobody_opt import rules
    first_block = chain.get_current_block()
    tempo = chain.tempo(netuid)
    if type(tempo) is not int or tempo < 0:
        raise ValueError("invalid chain tempo")
    epoch_length = tempo + 1
    block_hash = chain.determine_block_hash(epoch * epoch_length)
    last_block = chain.get_current_block()
    if any(type(b) is not int or b // epoch_length != epoch
           for b in (first_block, last_block)):
        raise ValueError("Pi chain epoch changed during PR100 preflight")
    context = rules.challenge(rules.pinned_config(epoch), epoch, block_hash, epoch_length)
    return {**context, "chain_read_first_block": first_block,
            "chain_read_last_block": last_block, "source": "pi_rpc",
            "netuid": netuid, "epoch_length_source": "SubtensorModule.Tempo+1"}


def sequences_by_uid(artifact: dict, epoch: int,
                     allow_stale: bool = False,
                     crew=None, epoch_start_block_hash=None,
                     epoch_length=None) -> dict[int, str]:
    """The nanobody half. A stale artifact is refused, not silently reused.

    Its epoch is checked because the producer rewrites it every epoch and a
    commit carrying last epoch's sequence is a wasted slot, not a near miss.

    But refusing used to abort the WHOLE commit, molecules included, and once
    the Pi became the sole writer that turned a late producer into an empty
    epoch -- five of them between 25243 and 25247. So the caller may allow a
    stale artifact once it has waited as long as the window permits: last
    epoch's sequence is a worse submission than this epoch's, and a far better
    one than nothing.
    """
    from nanobody_opt import rules
    if epoch >= rules.ACTIVATION_EPOCH:
        artifact = rules.verified_fleet(artifact, epoch, epoch_start_block_hash, epoch_length)
        print(json.dumps(dict(event="pr100_fleet_verification", epoch=epoch,
                              **artifact["handoff_diagnostics"])), flush=True)
        allow_stale = False
        if not artifact["sequence_pool"]:
            return {}
    stale = int(artifact.get("epoch", -1)) != epoch
    artifact_epoch = int(artifact.get("epoch", -1))
    # Freshness (bug 4, format-robust): prefer the writer's updated_at_unix
    # stamp, but a producer that does not stamp it must NOT hard-fail the
    # commit. A missing/non-finite stamp falls back to the always-present
    # epoch field, and only a genuinely stale epoch (age > STALE_AGE_LIMIT)
    # is rejected. A VALID-but-out-of-window stamp is still rejected.
    updated = artifact.get("updated_at_unix")
    stamp_is_valid = (
        isinstance(updated, (int, float))
        and not isinstance(updated, bool)
        and updated == updated
    )
    if stamp_is_valid and abs(time.time() - updated) > MAX_ARTIFACT_AGE_SECONDS:
        raise ValueError("nanobody artifact updated_at_unix is stale")
    if stale and not allow_stale:
        raise ValueError(
            f"nanobody artifact is for epoch {artifact.get('epoch')}, not {epoch}")
    if stale and artifact_epoch >= 0 and epoch - artifact_epoch > STALE_AGE_LIMIT:
        # A producer outage must never silently recommit the same pool forever:
        # an artifact this old is guaranteed to already be in the HF archive,
        # so recommitting it just re-earns the "not unique" rejection.
        raise ValueError(
            f"nanobody artifact is {epoch - artifact_epoch} epochs stale "
            f"(epoch {artifact_epoch}, limit {STALE_AGE_LIMIT}); refusing to "
            "recommit a duplicate")
    if stale:
        print(json.dumps(dict(event="stale_nanobody", have=artifact.get("epoch"),
                              want=epoch)), flush=True)
    out = {}
    pool = []

    # New artifacts carry an explicitly UID-independent pool.  The GPU host
    # does not own registrations and therefore cannot authoritatively bind a
    # sequence to a uid: the Pi may have bought a replacement since the
    # producer's last chain snapshot.  Keep accepting the old per-uid shape so
    # a rolling deploy remains safe, but treat every distinct sequence in it as
    # a pool entry too.
    sequence_pool = artifact.get("sequence_pool")
    for item in sequence_pool or []:
        raw = item.get("sequence") if isinstance(item, dict) else item
        if raw is None:
            continue
        sequence = str(raw).strip().upper()
        if sequence and sequence not in pool:
            pool.append(sequence)
    # Bug 10: an explicit-but-empty pool is no evidence the artifact can fill
    # a freshly acquired uid. Legacy artifacts (no sequence_pool key) still
    # fall back to their per-uid submissions below.
    if sequence_pool is not None and not pool:
        raise ValueError("nanobody artifact has no usable sequence pool")
    for entry in artifact.get("submissions") or []:
        sequence = str(entry.get("sequence") or "").strip().upper()
        if sequence:
            out[int(entry["uid"])] = sequence
            if sequence not in pool:
                pool.append(sequence)

    if crew is None:
        return out

    # Preserve an assignment when that uid is still ours, then deterministically
    # fill every newly acquired/recycled uid from the unused ranked pool.  This
    # is the ownership boundary: only the submit host's fresh crew snapshot is
    # allowed to decide which uid carries which sequence.
    allocated = {}
    used = set()
    for uid, _hotkey in crew:
        sequence = out.get(int(uid))
        if sequence and sequence not in used:
            allocated[int(uid)] = sequence
            used.add(sequence)
    # Fill with the best pool entry that is NOT a validator duplicate of a
    # sequence already allocated: the validator keeps one of any pair at or
    # above 0.95 identity, so a near-copy scores nothing. Epoch 25400 filled
    # five uids in pool order with 0.976-0.984 copies. Only when no distinct
    # entry is left does a uid take a near-copy, so every uid still commits.
    for uid, _hotkey in crew:
        uid = int(uid)
        if uid in allocated:
            continue
        remaining = [sequence for sequence in pool if sequence not in used]
        if not remaining:
            break
        taken = list(allocated.values())
        sequence = next(
            (candidate for candidate in remaining
             if all(_identity(candidate, other) < DUPLICATE_IDENTITY
                    for other in taken)),
            remaining[0])
        allocated[uid] = sequence
        used.add(sequence)
    return allocated


#: The validator's pairwise duplicate line (nanobody_opt.fleet).
DUPLICATE_IDENTITY = 0.95


def _identity(left: str, right: str) -> float:
    """Position identity over the longer sequence, as the fleet dedupe uses."""
    if not left or not right:
        return 0.0
    return sum(1 for a, b in zip(left, right) if a == b) / max(len(left), len(right))


def commit_time_rules():
    """(validator_rules module, thresholds), or None when the checker cannot run.

    The last gate before an irreversible on-chain commit. One molecule that
    breaks a validator rule voids the uid's whole portfolio (epoch 25695:
    h-bond acceptors = 11, limit 10), and the fleet file may have been built
    hours earlier by older code, so every uid's molecules are re-checked here
    against the pinned upstream rules (tools/validator_rules.py). No network:
    the live-config comparison already ran in bank_release. None means "ship NO
    molecules" -- callers skip the compound half; the nanobody still commits.
    """
    try:
        tools = str(HERE / "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)
        import validator_rules
        thresholds, _info = validator_rules.load_thresholds(live=False)
        return validator_rules, thresholds
    except Exception as error:
        print(json.dumps(dict(event="molecule_rules_unavailable",
                              err=f"{type(error).__name__}: {error}"[:200],
                              action="no molecules committed")), flush=True)
        return None


def refuse_invalid_molecules(entry: dict, reaction, rules, label: str = "fleet") -> bool:
    """True when every molecule of this uid's submission passes (False -> do not commit it)."""
    if rules is None:
        return False
    module, thresholds = rules
    try:
        problems = module.check_submission(entry.get("molecules") or [], reaction, thresholds)
    except Exception as error:     # a check that raises is a failed check
        problems = [(None, "checker_error", f"{type(error).__name__}: {error}")]
    if problems:
        print(json.dumps(dict(event="molecules_refused", source=label, uid=int(entry["uid"]),
                              reaction=reaction, problem_count=len(problems),
                              first=[[n, r, str(d)[:90]] for n, r, d in problems[:3]])), flush=True)
        return False
    return True


def molecules_by_uid(artifact, epoch: int, reaction: int) -> dict[int, str]:
    """The compound half, already bound to uids by the fleet allocator.

    Absent or not matching this epoch's reaction is NORMAL, not an error: the
    validator draws the reaction uniformly from five and a bank covers one at a
    time, so four epochs in five there are legitimately no molecules to send.
    Exiting here would take the NANOBODY down with them -- the track we
    actually place in -- so the compound half is simply empty and every uid
    still commits its sequence.
    """
    if artifact is None:
        return {}
    if int(artifact.get("epoch", -1)) != epoch:
        print(json.dumps(dict(event="no_molecules", why="epoch",
                              have=artifact.get("epoch"), want=epoch)), flush=True)
        return {}
    if int(artifact.get("reaction", -1)) != reaction:
        print(json.dumps(dict(event="no_molecules", why="reaction",
                              have=artifact.get("reaction"), want=reaction)), flush=True)
        return {}
    out = {}
    rules = None
    for entry in artifact.get("submissions") or []:
        names = [str(m["name"]) for m in entry.get("molecules") or []]
        if not names:
            continue
        if rules is None:
            rules = commit_time_rules() or False
        # A uid whose molecules fail any validator rule gets none: committing an
        # invalid set loses the whole portfolio anyway, and burns the molecules.
        if refuse_invalid_molecules(entry, reaction, rules or None):
            out[int(entry["uid"])] = ",".join(names)
    return out


def add_hedge(molecules: dict, hedge_path: Path, epoch: int, crew) -> dict:
    """Optional second bet on a DIFFERENT reaction, carried by spare uids.

    During the SANQI transition the validators' reaction rule (draw mod 5 vs mod 44) is decided by the code and
    database they run when they score, which we cannot see in advance. ``release/hedge_fleet.json`` (same schema
    as compound_fleet.json, built for the other rule's reaction) puts that bet on uids that carry no molecules.
    A wrong-reaction uid just scores nothing on the compound track (like molecules:false); its nanobody is
    unaffected. Never overrides a uid that already has molecules, only uids that are in the crew, only this epoch.
    """
    if not hedge_path.is_file():
        return molecules
    try:
        artifact = _load(hedge_path)
        if int(artifact.get("epoch", -1)) != epoch:
            print(json.dumps(dict(event="hedge_ignored", why="epoch", have=artifact.get("epoch"), want=epoch)), flush=True)
            return molecules
        crew_uids = {int(u) for u, _ in crew}
        added = []
        rules = commit_time_rules() or None
        for entry in artifact.get("submissions") or []:
            uid = int(entry["uid"])
            names = [str(m["name"]) for m in entry.get("molecules") or []]
            if names and uid in crew_uids and uid not in molecules:
                # The hedge is a deliberate bet on ANOTHER reaction, so its names are checked
                # against the hedge's own declared reaction, and every property rule still applies.
                if not refuse_invalid_molecules(entry, int(artifact.get("reaction", -1)), rules, "hedge"):
                    continue
                molecules[uid] = ",".join(names)
                added.append(uid)
        print(json.dumps(dict(event="hedge_added", reaction=artifact.get("reaction"), uids=added)), flush=True)
    except Exception as error:                      # a bad hedge file must never cost the epoch
        print(json.dumps(dict(event="hedge_error", err=f"{type(error).__name__}: {error}"[:160])), flush=True)
    return molecules


def commit_batch(rows, arguments, environment) -> tuple[int, int, list]:
    """Encrypt all, push once, then set every commitment concurrently.

    The serial path spends ~41s per uid inside set_commitment waiting for the
    extrinsic, so a ten-uid fleet needed ~50 blocks and the tail fell past the
    validator's cutoff. The wait is the chain's, not ours: the uids hold
    DIFFERENT hotkeys, so their extrinsics are independent and can be in flight
    at the same time. Ten waits of 41s become one.

    Returns (committed, failed, dropped_uids).
    """
    import concurrent.futures as futures
    import os

    import bittensor as bt
    from bittensor.core.errors import MetadataError
    from btdr import QuicknetBittensorDrandTimelock
    from files import load_github_path, push_submissions
    from nanobody_opt import fleet
    from nanobody_opt.commit_submission import encoded_payload

    os.environ.update({k: v for k, v in environment.items() if k.startswith(
        ("GITHUB_", "NOVA_"))})

    subtensor = bt.Subtensor(network=arguments.network)
    metagraph = subtensor.metagraph(arguments.netuid)
    on_chain = {int(i): str(h) for i, h in enumerate(list(metagraph.hotkeys))}
    addresses = fleet.local_hotkey_addresses(arguments.wallet_path, arguments.wallet_name)

    # Resolve and verify BEFORE encrypting anything: uids are recycled, and
    # committing for a uid another operator now holds signs their slot.
    live = []
    stale_uids = []
    for row in rows:
        name = addresses.get(row["hotkey"], row["hotkey"])
        if on_chain.get(int(row["uid"])) != row["hotkey"]:
            print(json.dumps(dict(event="stale", uid=row["uid"])), flush=True)
            stale_uids.append(row["uid"])
            continue
        live.append(dict(row, name=name))
    if not live:
        # Zero verified commits from a nonempty intended fleet: report the
        # stale uids so the caller can fail loudly rather than read success.
        return 0, 0, stale_uids

    block = subtensor.get_current_block()
    timelock = QuicknetBittensorDrandTimelock()
    # ALWAYS load_github_path(). `arguments.github_path` is the bare
    # "data/results" that goes into GITHUB_REPO_PATH, and it is truthy, so
    # `args.github_path or load_github_path()` -- copied from the single-uid
    # driver, where that argument defaults to None -- silently skipped the
    # fallback and committed a pointer with no owner/repo/branch. The blobs
    # were pushed, the extrinsics landed in-window, and the validator dropped
    # every uid because it could not resolve the path. Absent from the
    # leaderboard entirely, which is how 25251 was lost.
    github_path = load_github_path()
    for row in live:
        filename, encoded = encoded_payload(row["plaintext"], int(row["uid"]),
                                            block, timelock)
        row["filename"], row["encoded"] = filename, encoded
        row["content"] = f"{github_path}/{filename}.txt"

    if arguments.rehearse:
        # Everything above this line is the new code: the metagraph read, the
        # hotkey resolution, the uid-ownership check and the encryption. Stop
        # before anything leaves the host.
        print(json.dumps(dict(event="rehearsal", uids=[r["uid"] for r in live],
                              names=[r["name"] for r in live], block=block,
                              distinct_blobs=len({r["filename"] for r in live}),
                              sample_content=live[0]["content"])), flush=True)
        return len(live), 0, []

    if not push_submissions([(r["filename"], r["encoded"]) for r in live]):
        print(json.dumps(dict(event="push_failed", uids=[r["uid"] for r in live])),
              file=sys.stderr, flush=True)
        # Nothing committed, so nothing is stranded pointing at a missing blob.
        return 0, len(live), []

    print(json.dumps(dict(event="pushed", blobs=len(live), block=block)), flush=True)

    left = blocks_left(arguments.epoch, arguments.network)
    if left is not None and left <= arguments.blackout_blocks:
        return 0, 0, [r["uid"] for r in live]

    def one(row):
        # A Subtensor per thread: the client is not documented as thread safe,
        # and sharing one would be a silent corruption rather than an error.
        try:
            local = bt.Subtensor(network=arguments.network)
            wallet_factory = getattr(bt, "Wallet", None) or bt.wallet
            wallet = wallet_factory(name=arguments.wallet_name, hotkey=row["name"],
                                    path=arguments.wallet_path)
            ok = local.set_commitment(wallet=wallet, netuid=arguments.netuid,
                                      data=row["content"])
            return row, bool(ok), "" if ok else "refused"
        except MetadataError:
            return row, False, "too soon"
        except Exception as error:
            return row, False, f"{type(error).__name__}: {error}"[:160]

    def report(row, ok, why, pass_name):
        print(json.dumps(dict(event="commit", uid=row["uid"], rc=0 if ok else 1,
                              molecules=row["has_molecules"],
                              sequence=row["has_sequence"], pass_name=pass_name,
                              last=f"{'COMMIT_OK' if ok else 'COMMIT_FAILED'} "
                                   f"uid={row['uid']} hotkey={row['name']} "
                                   f"content={row['content']} {why}".strip())),
              flush=True)

    with futures.ThreadPoolExecutor(max_workers=len(live)) as pool:
        results = list(pool.map(one, live))
    landed = [r for r, ok, _ in results if ok]
    for row, ok, why in results:
        report(row, ok, why, "parallel")

    # Whether the client tolerates this many threads is the one thing that
    # cannot be rehearsed without really committing, so it repairs itself:
    # anything the parallel pass did not land is retried one at a time, which
    # is the path that has always worked. The parallel pass costs ~41s of a
    # ~60-block budget, so there is room for the repair.
    stragglers = [r for r, ok, _ in results if not ok]
    repaired = []
    for row in stragglers:
        left = blocks_left(arguments.epoch, arguments.network)
        if left is not None and left <= arguments.blackout_blocks:
            print(json.dumps(dict(event="blackout_stop", blocks_left=left,
                                  dropped=[r["uid"] for r in stragglers
                                           if r not in repaired])), flush=True)
            break
        row, ok, why = one(row)
        report(row, ok, why, "repair")
        if ok:
            repaired.append(row)
    committed = len(landed) + len(repaired)
    if stragglers:
        print(json.dumps(dict(event="repair_summary", parallel=len(landed),
                              retried=len(stragglers), repaired=len(repaired))),
              flush=True)
    return committed, len(live) - committed, []


def blocks_left(epoch: int, network: str, epoch_length: int = 361):
    """Blocks until this epoch ends, or None if the chain cannot be read.

    Returning None rather than 0 matters: 0 would read as "blackout, stop", and
    a transient RPC hiccup must not silently cancel the remaining commits.
    """
    try:
        import bittensor
        block = bittensor.Subtensor(network=network).get_current_block()
        return (epoch + 1) * epoch_length - block
    except Exception:
        return None


def plan(crew, molecules, sequences, value=None):
    """One row per uid, with whichever halves are actually available.

    A uid with neither half is dropped rather than committed as "~|~", which
    scores nothing and still spends the commitment slot.
    """
    rows = []
    for uid, hotkey in crew:
        mols = molecules.get(uid, "")
        seq = sequences.get(uid, "")
        if not mols and not seq:
            continue
        rows.append(dict(uid=uid, hotkey=hotkey,
                         plaintext=f"{mols or NULL}|{seq or NULL}",
                         has_molecules=bool(mols), has_sequence=bool(seq)))
    # Best first. Commits are serial and the epoch can run out before the list
    # does, so crew order decides which uids get dropped -- and crew order is
    # arbitrary. If anything is going to be lost to the blackout, it must be
    # the least valuable portfolio, not whichever uid happens to sort last.
    value = value or {}
    rows.sort(key=lambda r: (-float(value.get(r["uid"], 0.0)),
                             -r["has_molecules"], r["uid"]))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--allow-stale-nanobody", action="store_true",
                        help="commit even if the producer has not refreshed for "
                             "this epoch. Only for the last safe block: prefer "
                             "fresh, but never commit nothing")
    parser.add_argument("--reaction", type=int, required=True, choices=list(range(1, 45)))
    parser.add_argument("--crew", default=str(HERE / "crew.json"))
    parser.add_argument("--nanobody", default=str(HERE / "nanobody_result.json"))
    # The producer is resolved by LABEL at run time, never by a written-down
    # endpoint: vast.ai rotates the ssh port on every restart, and the producer
    # itself moves between boxes (it is being rehomed to an 8-GPU rental). A
    # label survives both; an endpoint in a config file is a latent outage that
    # looks exactly like a dead box.
    parser.add_argument("--producer-label", default="nova-nanobody-pixi")
    parser.add_argument("--producer-path", default="/root/artifacts/nanobody_result.json")
    parser.add_argument("--vastai", default="/opt/nova-nanobody/deploy/slot_host/.pixi/envs/default/bin/vastai")
    parser.add_argument("--pull", action="store_true",
                        help="fetch the nanobody artifact from the producer first")
    parser.add_argument("--pull-only", action="store_true",
                        help="fetch the artifact and stop. The cutover driver "
                             "verifies the pull BEFORE it stops production, and "
                             "it must do that by running this program rather "
                             "than exec()ing part of it: module level here uses "
                             "__file__, which does not exist under exec, and "
                             "that failed on the first real cutover.")
    parser.add_argument("--fleet", default=str(HERE / "compound_fleet.json"))
    parser.add_argument("--wallet-name", default="mining_ops")
    parser.add_argument("--wallet-path", default=str(Path.home() / ".bittensor/wallets"))
    parser.add_argument("--netuid", type=int, default=68)
    parser.add_argument("--network", default="finney")
    parser.add_argument("--python", default=str(HERE / ".pixi/envs/default/bin/python"))
    parser.add_argument("--github-owner", default="thomasvangurp")
    parser.add_argument("--github-repo", default="nova-sn68-submissions")
    parser.add_argument("--github-branch", default="main")
    # Read off the live chain, not guessed: our existing commitments are
    # "<owner>/<repo>/main/data/results/<hash>.txt". Left unset, load_github_path
    # sees None rather than "" and builds ".../main/None/<hash>.txt" -- a chain
    # pointer to a blob that does not exist, which strands the uid for the whole
    # commitment window with no error at commit time.
    parser.add_argument("--github-path", default="data/results")
    parser.add_argument("--deploy-key", default=str(Path.home() / ".ssh/nova_deploy"))
    parser.add_argument("--rehearse", action="store_true",
                        help="run the parallel path up to the push and stop: "
                             "proves the new code without touching the repo or "
                             "the chain")
    parser.add_argument("--serial", action="store_true",
                        help="one commit_submission subprocess per uid, the "
                             "original path. ~41s each, so a ten-uid fleet runs "
                             "past the validator's cutoff; kept as the fallback")
    parser.add_argument("--blackout-blocks", type=int, default=16,
                        help="stop committing with this many blocks left: the "
                             "validator stops reading before the epoch ends")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.pull or arguments.pull_only:
        pull_producer(arguments.vastai, arguments.producer_label,
                      arguments.producer_path, Path(arguments.nanobody))
        if arguments.pull_only:
            return 0
    crew = [(int(u), str(h)) for u, h in _load(Path(arguments.crew))]
    from nanobody_opt import rules
    epoch_hash = None
    epoch_length = None
    if arguments.epoch >= rules.ACTIVATION_EPOCH:
        import bittensor as bt
        chain = bt.Subtensor(network=arguments.network)
        attestation = attest_epoch(chain, arguments.epoch, arguments.netuid)
        epoch_hash = attestation["epoch_start_block_hash"]
        epoch_length = attestation["epoch_length"]
        print(json.dumps(dict(event="pr100_chain_attestation", **attestation)), flush=True)
    try:
        sequences = sequences_by_uid(_load(Path(arguments.nanobody)), arguments.epoch,
                                     allow_stale=arguments.allow_stale_nanobody,
                                     crew=crew, epoch_start_block_hash=epoch_hash,
                                     epoch_length=epoch_length)
    except (ValueError, TypeError, KeyError, OSError) as error:
        if arguments.epoch < rules.ACTIVATION_EPOCH:
            raise
        sequences = {}
        print(json.dumps(dict(event="pr100_nanobody_rejected", error_type=type(error).__name__,
                              action="retain independently valid molecule submissions")), flush=True)
    fleet_path = Path(arguments.fleet)
    molecules = molecules_by_uid(
        _load(fleet_path) if fleet_path.is_file() else None,
        arguments.epoch, arguments.reaction)
    molecules = add_hedge(molecules, fleet_path.parent / "hedge_fleet.json", arguments.epoch, crew)

    fleet_artifact = _load(fleet_path) if fleet_path.is_file() else None
    value = {}
    if isinstance(fleet_artifact, dict):
        for entry in fleet_artifact.get("submissions") or []:
            try:
                value[int(entry["uid"])] = float(entry.get("selection_score") or 0.0)
            except (TypeError, ValueError, KeyError):
                pass
    rows = plan(crew, molecules, sequences, value)
    missing_sequence_uids = sorted(uid for uid, _ in crew if uid not in sequences)
    print(json.dumps(dict(event="plan", epoch=arguments.epoch,
                          reaction=arguments.reaction, crew=len(crew),
                          with_molecules=sum(r["has_molecules"] for r in rows),
                          with_sequence=sum(r["has_sequence"] for r in rows),
                          committing=len(rows),
                          missing_sequence_uids=missing_sequence_uids)), flush=True)

    # Validate every plaintext BEFORE committing any of them: a malformed
    # message discovered at uid 9 has already spent eight commitment slots.
    from nanobody_opt.commit_submission import validate_plaintext
    for row in rows:
        probe = row["plaintext"]
        try:
            validate_plaintext(probe)
        except Exception as error:
            print(json.dumps(dict(event="invalid", uid=row["uid"],
                                  error=str(error)[:160])), file=sys.stderr)
            return 2
    print(json.dumps(dict(event="validated", uids=[r["uid"] for r in rows])), flush=True)

    # The validator reads commitments up to `current_block - no_submission_blocks`,
    # so anything signed in the last ~16 blocks of the epoch is on-chain and
    # invisible. Commits take ~73s each (measured at 25242: 731s for ten uids),
    # which is six blocks apiece -- twelve uids cannot fit between the
    # producer's refresh at end-75 and that cutoff. Without this check the
    # overrun is completely silent: every extrinsic succeeds and the tail is
    # simply never scored.
    import os
    environment = dict(os.environ,
                       GITHUB_REPO_OWNER=arguments.github_owner,
                       GITHUB_REPO_NAME=arguments.github_repo,
                       GITHUB_REPO_BRANCH=arguments.github_branch,
                       GITHUB_REPO_PATH=arguments.github_path,
                       NOVA_DEPLOY_KEY=arguments.deploy_key,
                       NOVA_GIT_PUSH="1",
                       NOVA_GIT_WORKDIR=str(HERE / ".git_work"))

    if arguments.rehearse or (not arguments.serial and not arguments.dry_run):
        committed, failures, dropped = commit_batch(rows, arguments, environment)
        print(json.dumps(dict(event="done", committed=committed, failed=failures,
                              dropped_to_blackout=len(dropped),
                              dropped_uids=dropped)), flush=True)
        if not failures or arguments.rehearse:
            if not arguments.rehearse and committed == 0 and rows:
                # Zero verified commits from a nonempty intended fleet must not
                # read as success: "COMMIT_OK" has to mean something actually
                # landed on-chain for the validator to read.
                print(json.dumps(dict(event="zero_commits",
                                      intended=len(rows), committed=0)), flush=True)
                return 2
            return 1 if failures else 0
        # The parallel batch push failed (e.g. a racing push on the shared
        # submissions repo). Fall back to the serial per-uid path below so the
        # nanobody sequences still commit instead of the whole fleet scoring 0.
        # The serial path re-runs every uid; the chain refuses re-commits of
        # already-landed uids harmlessly.
        print(json.dumps(dict(event="parallel_fallback", committed=committed,
                              failed=failures)), flush=True)

    failures = 0
    dropped = []
    for index, row in enumerate(rows):
        left = blocks_left(arguments.epoch, arguments.network)
        if left is not None and left <= arguments.blackout_blocks:
            dropped = [r["uid"] for r in rows[index:]]
            print(json.dumps(dict(event="blackout_stop", blocks_left=left,
                                  committed=index, dropped=dropped)), flush=True)
            break
        command = [arguments.python, "-m", "nanobody_opt.commit_submission",
                   "--uid", str(row["uid"]), "--hotkey", row["hotkey"],
                   "--plaintext", row["plaintext"],
                   "--wallet-name", arguments.wallet_name,
                   "--wallet-path", arguments.wallet_path,
                   "--netuid", str(arguments.netuid),
                   "--network", arguments.network]
        if arguments.dry_run:
            command.append("--dry-run")
        import os
        environment = dict(os.environ,
                           GITHUB_REPO_OWNER=arguments.github_owner,
                           GITHUB_REPO_NAME=arguments.github_repo,
                           GITHUB_REPO_BRANCH=arguments.github_branch,
                           GITHUB_REPO_PATH=arguments.github_path,
                           NOVA_DEPLOY_KEY=arguments.deploy_key,
                           # Push the blob with the deploy key rather than the
                           # GitHub API. The API path needs GITHUB_TOKEN, which
                           # this host has no business holding; the deploy key
                           # was generated here, never left, and is already
                           # verified against the repo. Without this every
                           # commit dies on "Github environment variables not
                           # set" AFTER the timelock encryption succeeded --
                           # invisible to --dry-run, which returns before the
                           # upload.
                           NOVA_GIT_PUSH="1",
                           # The push clones into NOVA_GIT_WORKDIR, which
                           # defaults to /root/.nova_submissions. This host runs
                           # as an ordinary user and has no /root.
                           NOVA_GIT_WORKDIR=str(HERE / ".git_work"))
        done = subprocess.run(command, cwd=str(HERE), capture_output=True,
                              text=True, env=environment)
        tail = (done.stdout or done.stderr).strip().splitlines()
        print(json.dumps(dict(event="commit", uid=row["uid"], rc=done.returncode,
                              molecules=row["has_molecules"],
                              sequence=row["has_sequence"],
                              last=tail[-1][:160] if tail else "")), flush=True)
        if done.returncode:
            failures += 1
    # "committed" counts what the VALIDATOR can read, not what the chain
    # accepted. Reporting len(rows) here is how 25242 got recorded as ten
    # clean commits when most of the tail had already fallen past the cutoff.
    attempted = len(rows) - len(dropped)
    committed = attempted - failures
    print(json.dumps(dict(event="done", committed=committed,
                          failed=failures, dropped_to_blackout=len(dropped),
                          dropped_uids=dropped)), flush=True)
    if committed == 0 and rows:
        print(json.dumps(dict(event="zero_commits",
                              intended=len(rows), committed=0)), flush=True)
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

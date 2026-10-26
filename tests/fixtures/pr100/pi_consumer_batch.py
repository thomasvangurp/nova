# Exact functions from the installed Pi consumer ea9677c4 and signer 4065f061.
# Tests replace every external dependency before calling these functions.
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

def encoded_payload(message: str, uid: int, block: int, timelock) -> tuple[str, str]:
    """Timelock-encrypt one submission, returning (filename, base64 content).

    ``message`` must already be in the validator's wire format; build legacy
    sequence-only input with ``submission_message`` or combined input with
    ``validate_plaintext``.

    The encryption is keyed on the uid, which is why this cannot be done once for
    the whole fleet: each uid's payload is distinct even for an identical
    sequence.
    """
    encrypted = timelock.encrypt(uid, message, block)
    content = str(encrypted)
    encoded = base64.b64encode(content.encode()).decode()
    filename = hashlib.sha256(content.encode()).hexdigest()[:20]
    return filename, encoded

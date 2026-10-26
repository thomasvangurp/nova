# Exact main from installed consumer ea9677c4; imported only with all external operations stubbed.
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
    sequences = sequences_by_uid(_load(Path(arguments.nanobody)), arguments.epoch,
                                 allow_stale=arguments.allow_stale_nanobody,
                                 crew=crew)
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
    # validate_plaintext insists on a real sequence, but the VALIDATOR does not:
    # its own parse_decrypted_submission accepts "molecules|~" and returns 20
    # molecules with 0 sequences. A uid holding molecules but no qualifying
    # nanobody -- a freshly registered slot, say -- should still compete on the
    # compound track. So the molecule half is validated against the local guard
    # using a placeholder sequence, and the real "molecules|~" is what commits.
    PLACEHOLDER = "K" * 120
    for row in rows:
        probe = row["plaintext"]
        if not row["has_sequence"]:
            probe = f'{probe.split("|", 1)[0]}|{PLACEHOLDER}'
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

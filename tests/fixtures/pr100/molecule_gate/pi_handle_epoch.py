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
            made = subprocess.run(
                [sys.executable, str(ROOT / "tools/make_fleet.py"),
                 "--epoch", str(epoch), "--reaction", str(view["reaction"]),
                 "--crew", str(crew_path), "--python", a.rdkit_python,
                 "--minimum-score", str(floor),
                 "--prediction-offset", str(floor_info.get("optimism", 0.0))]
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
                        result = subprocess.run(
                            [sys.executable, str(ROOT / "tools/make_fleet.py"),
                             "--epoch", str(epoch), "--reaction", str(reaction),
                             "--crew", str(crew_path), "--python", a.rdkit_python,
                             "--minimum-score", str(floor),
                             "--prediction-offset", str(floor_info.get("optimism", 0.0)),
                             "--bank", str(ROOT / "bank.sqlite"), "--out", str(recheck_out),
                             "--fleet-tree", str(ROOT)],
                            capture_output=True, text=True, timeout=timeout)
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
                    r = subprocess.run(
                        [sys.executable, str(ROOT / "tools/make_fleet.py"),
                         "--epoch", str(epoch), "--reaction", str(reaction),
                         "--crew", str(crew_path), "--python", a.rdkit_python,
                         "--minimum-score", str(floor),
                         "--prediction-offset", str(floor_info.get("optimism", 0.0)),
                         "--bank", str(ROOT / "bank.sqlite"),
                         "--out", str(recheck_out), "--fleet-tree", str(ROOT)],
                        capture_output=True, text=True, timeout=1200)
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

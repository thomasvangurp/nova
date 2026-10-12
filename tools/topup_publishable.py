#!/usr/bin/env python3
"""Proactively top up the candidates we are about to publish to TARGET_OBS observations.

A candidate that looks good after 8 observations is mostly looking good by luck: on independent
replays (epochs 25662-25671) the ten best-looking candidates with 8-12 observations scored no better
than the average candidate (replay z +0.44 against local z -0.11), while the best-looking candidates
with 50+ observations held most of their advantage (-1.29 -> -0.99).  The producer's confirmation
deepens its own shortlist, but the publisher picks from all history, so tickets are regularly
published with 8-16 observations.

This worker reads the publisher's own latest ranking (already in publication order, chance floor
included), takes the first TOP_K candidates, and gives every one that has fewer than TARGET_OBS
observations the missing ones as extra scoring rounds on the warm scorer daemon -- the same scoring
call the race's confirmation rounds use, one round per extra observation, each round only containing
the candidates that still need it.  The results are imported with the existing exact-screen importer
(source local_exact_raw, the label every other race observation already carries), so the publisher's
next cycle sees the deeper counts and re-orders.  Candidates that regress drop out; newcomers are
topped up on the next pass.  It is a pro-active step, not a publishing requirement: nothing is ever
excluded for having few observations.

Cost: a round is ~9 s fixed + ~0.55 s per candidate on 8 GPUs; typically 3-5 of the final 13 are
short, so one pass is ~13-30 s of full-box time per round batch.

Usage:  topup_publishable.py once [--dry-run] | loop
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("NANOBODY_SCORER_DAEMON", "1")      # same warm-daemon path as the race
sys.path.insert(0, "/root")

TARGET = "P20809"
DB = Path("/root/artifacts/nanobody_mutations.sqlite")
LOG = Path("/root/artifacts/topup_publishable.log")
CHAIN = Path("/root/artifacts/chain_state.json")
TARGET_OBS = 20
TOP_K = 36                 # the 13 we publish plus the ones that can climb into them once strict qualification removes some
MAX_ROUNDS = 12            # 8 -> 20 observations
EPOCH_LENGTH = 361
FIRST_OFFSET = 200         # the strict cycle is fast now, so passes can start while the confirmation is still finishing
MIN_REMAINING_BLOCKS = 30  # leave the commit window alone
FINAL_CYCLE_OFFSET = 285   # the publisher's last strict cycle starts around here; every round must be imported before it
BLOCKS_PER_ROUND = 3       # ~34 s per round measured while sharing the GPUs with the confirmation
PREDICT_BATCH = 2          # the producer's --screen-predict-batch-size
SCORER = Path("/root/nanobody_opt/score_boltzgen_fast_adapter.py")   # the producer's --exact-scorer
PY = "/root/.pixi/envs/bgen/bin/python"


def log(**fields) -> None:
    fields["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with LOG.open("a") as handle:
        handle.write(json.dumps(fields, sort_keys=True) + "\n")


def chain() -> tuple[int, int]:
    state = json.loads(CHAIN.read_text())
    return int(state["epoch"]), int(state["block"]) % EPOCH_LENGTH


def latest_ranking(epoch: int) -> tuple[Path, float] | None:
    root = Path(f"/root/live/epoch{epoch}")
    paths = [p for p in (root / "qualified_rankings.json", root / "provisional_rankings.json") if p.is_file()]
    if not paths:
        return None
    best = max(paths, key=lambda p: p.stat().st_mtime)
    return best, best.stat().st_mtime


def fresh_ranking(epoch: int) -> Path | None:
    """The publisher's own provisional ranking, recomputed NOW from the database.

    The publisher writes its ranking about once an hour (plus a final cycle just before the commit), so
    the file on disk can be half an hour stale and would top up yesterday's order.  This runs the same
    rank_exact call the publisher's provisional pass makes (new chance-floor order included) so the
    candidates we deepen are the ones that would be published if it ran this minute.
    """
    from nanobody_opt import publication_gates

    root = Path(f"/root/live/epoch{epoch}")
    out = root / "topup" / "fresh_ranking.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    defaults = publication_gates.gate_defaults()
    command = [PY, "/root/nanobody_opt/rank_exact.py", "--observation-history", str(DB),
               "--archive", str(root / "archive.csv"), "--target", TARGET, "--output", str(out),
               "--exclude-file", str(root / "excluded_sequences.txt"), "--recent", "300", "--field-size", "49",
               "--bootstrap-fields", "64", "--limit", "1000", "--before-epoch", str(epoch),
               "--db-field-count", "10", "--minimum-api-fields", "3",
               "--minimum-observations", defaults["--minimum-observations"],
               "--competitive-score-threshold", str(defaults["--competitive-score-threshold"]),
               "--selection-order", "win-probability"]
    env = {**os.environ, "NOVA_FULL_ARCHIVE_FIELD": "0"}
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=600, env=env)
    except Exception as error:  # noqa: BLE001 -- fall back to the publisher's own file
        log(error=f"fresh ranking failed: {type(error).__name__}")
        return None
    return out if out.is_file() else None


def needs(epoch: int) -> tuple[dict[str, int], dict]:
    path = fresh_ranking(epoch)
    source = "fresh rank_exact"
    if path is None:
        found = latest_ranking(epoch)
        if found is None:
            return {}, {"reason": "no ranking yet"}
        path, source = found[0], f"publisher file {found[0].name}"
    mtime = path.stat().st_mtime
    candidates = json.loads(path.read_text())["candidates"]
    state_path = Path(f"/root/live/epoch{epoch}/topup/state.json")
    added = json.loads(state_path.read_text()) if state_path.is_file() else {}
    need: dict[str, int] = {}
    for item in candidates[:TOP_K]:
        sequence = str(item["sequence"]).strip().upper()
        if len(sequence) != 127:                 # the fast scorer only supports length 127
            continue
        have = int(item["observation_count"])
        # observations imported after this ranking was written are not in its counts yet
        have += sum(count for stamp, count in added.get(sequence, []) if stamp > mtime)
        if have < TARGET_OBS:
            need[sequence] = min(TARGET_OBS - have, MAX_ROUNDS)
    return need, {"ranking": source, "ranking_age_s": round(time.time() - mtime)}


def gpu_ids() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.split()
    return [x.strip() for x in out if x.strip()]


def run_once(dry_run: bool = False, force: bool = False) -> dict:
    epoch, offset = chain()
    result: dict = {"epoch": epoch, "offset": offset}
    if not force and not (FIRST_OFFSET <= offset <= EPOCH_LENGTH - MIN_REMAINING_BLOCKS):
        return {**result, "skipped": "outside the top-up window"}
    need, info = needs(epoch)
    result.update(info)
    epoch, offset = chain()                      # the ranking took ~95 s: judge the time left from now
    result["offset"] = offset
    result["short"] = len(need)
    if not need:
        return {**result, "skipped": "nothing short of the target"}
    if not force:
        affordable = max(0, (FINAL_CYCLE_OFFSET - offset) // BLOCKS_PER_ROUND)
        if affordable < 1:
            return {**result, "skipped": "no time left before the final publication cycle"}
        if max(need.values()) > affordable:     # give every candidate as many of its missing rounds as fit
            result["rounds_capped_to"] = affordable
            need = {s: min(n, affordable) for s, n in need.items()}
    result["observations_needed"] = sum(need.values())
    if dry_run:
        return {**result, "dry_run": True, "need": {s[:12]: n for s, n in need.items()}}
    from nanobody_opt.sequential_exact_race import _run_screen

    root = Path(f"/root/live/epoch{epoch}/topup")
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")
    config = Path(f"/root/live/epoch{epoch}/producer/config.yaml")
    if not config.is_file():
        config = Path(f"/root/live/epoch{epoch}/config.yaml")
    archive = Path(f"/root/live/epoch{epoch}/producer/archive.csv")
    gpus = gpu_ids()
    order = list(need)                                  # publication order
    started = time.monotonic()
    done_dirs, added = [], {}
    for round_number in range(1, max(need.values()) + 1):
        subset = [s for s in order if need[s] >= round_number]
        if not subset:
            break
        _, offset_now = chain()
        if offset_now + BLOCKS_PER_ROUND > FINAL_CYCLE_OFFSET + 5 and not force:
            result["stopped"] = "final publication cycle is near"
            break
        out_dir = root / f"round_{stamp}_{round_number:02d}"
        source = out_dir / "source.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps({"target": TARGET, "sequences": subset}))
        try:
            _run_screen(source=source, config=config, target=TARGET, output_dir=out_dir, gpus=gpus,
                        candidate_limit=len(subset), scorer=SCORER, predict_batch_size=PREDICT_BATCH)
        except Exception as error:  # noqa: BLE001 -- never hurt the live race
            result["round_failed"] = f"round {round_number}: {type(error).__name__}"
            break
        if (out_dir / "exact_all.json").is_file():
            done_dirs.append(out_dir)
            for s in subset:
                added.setdefault(s, 0)
                added[s] += 1
    result["rounds_scored"] = len(done_dirs)
    if done_dirs:
        command = [PY, "-m", "nanobody_opt.import_exact_screens", "--database", str(DB), "--target", TARGET,
                   "--archive", str(archive)]
        for d in done_dirs:
            command += ["--exact-root", str(d)]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
        result["import_rc"] = completed.returncode
        result["import_tail"] = (completed.stdout or completed.stderr)[-300:].strip().splitlines()[-2:]
        if completed.returncode == 0:
            state_path = root / "state.json"
            state = json.loads(state_path.read_text()) if state_path.is_file() else {}
            now = time.time()
            for s, count in added.items():
                state.setdefault(s, []).append([now, count])
            state_path.write_text(json.dumps(state))
            result["observations_added"] = sum(added.values())
    result["seconds"] = round(time.monotonic() - started, 1)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["once", "loop"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore the epoch-offset window (testing)")
    a = ap.parse_args()
    lock = open("/root/artifacts/topup_publishable.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another top-up is running")
        return 0
    if a.command == "once":
        result = run_once(a.dry_run, a.force)
        print(json.dumps(result, sort_keys=True))
        if not a.dry_run and "skipped" not in result:
            log(**result)
        return 0
    while True:
        try:
            result = run_once()
            if "skipped" not in result:
                log(**result)
        except Exception as error:  # noqa: BLE001
            log(error=f"{type(error).__name__}: {error}")
        time.sleep(120)


if __name__ == "__main__":
    raise SystemExit(main())

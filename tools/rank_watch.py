#!/usr/bin/env python3
"""Rank-regression watchdog for our SN68 nanobody fleet (target P20809).

Why this exists
---------------
On 2026-10-09 a submission process was silently swapped onto a frozen,
non-replenishing candidate bank. Our best rank went from a steady 1-6 to 10+
and most of the fleet sank into the bottom half, and nothing noticed for ~7 h.
This script turns the public leaderboard into an automated signal. It runs
every 15 min from launchd (``ai.nova.nanobody-rank-watch``).

What it does each run
---------------------
1. Fetches the public leaderboard for recent epochs
   (``/api/competitions/leaderboard/{epoch}/nanobodies``). Every response is
   saved to ``data/rank_watch/cache/<epoch>.json`` first and parsed from disk.
   A 404 means "not scored yet" (epochs lag real time by 1-2 h): we skip it and
   stop probing forward after 3 consecutive 404s. Older cached epochs are
   immutable and are not re-downloaded. The newest two are re-fetched each run
   in case they were still filling in.
2. Per epoch, takes the entries whose coldkey is ours and computes: fleet size
   ``n``, field size ``N``, best rank, mean rank, mean rank as a fraction of the
   field (``mean_pct``), and the number of top-10 slots.
3. Replays the last ``LOOKBACK`` epochs in order against a rolling baseline
   and raises alerts (see below). Each event goes to
   ``data/rank_watch/alerts.log`` and to a macOS notification. ``osascript``
   ``display notification`` stands in for a push notification, because a
   launchd job has no access to Claude's PushNotification tool.

Baseline: how it is built and why
---------------------------------
* The window is the median of the last ``BASELINE_WINDOW = 12`` *healthy*
  epochs before the epoch being judged. That is about 14 h at the ~72 min
  cadence. It is long enough that one lucky or unlucky epoch barely moves it,
  and short enough to follow real strategy changes. Our healthy best-rank
  median moved from ~6 (epochs 2557x) to ~1.5-3 (2560x) within a day.
* We use the median, not the mean. Best rank is heavy-tailed (one-off 17s and
  18s among healthy 1-4s), and a median ignores those.
* "Healthy" means the epoch did not itself trip a rank or fleet flag. Flagged
  epochs never enter the baseline, so a lasting regression cannot quietly
  become the new normal and silence the alarm. A permanent intentional change
  shows up as a persistent alert with reminders, and someone has to look at it.
* Seeding: the replay starts ``LOOKBACK = 48`` epochs back (~2.4 days). The
  first ``MIN_BASELINE = 8`` epochs with our fleet present go into the
  baseline unconditionally, and no judgement is made until there are 8. The
  replay is recomputed from cached data on every run, so the baseline never
  drifts as hidden state. Only the "what have I already alerted on" bookkeeping
  is persisted (``data/rank_watch/state.json``).

Alert rules and how they were chosen
------------------------------------
RANK_REGRESSION fires when at least ``BAD_NEEDED = 2`` of the last
``BAD_WINDOW = 3`` epochs with our fleet present have
``best_rank >= T``, where ``T = max(10, 2 * baseline_median_best + 2)``.
  * Rank 10 is the floor because our healthy best rank lives in 1-6. With a
    healthy baseline of ~3, T is 10.
  * The ``2*median+2`` term scales the threshold when the baseline is weaker.
    At a baseline of ~6, a best rank of 12 is ordinary noise.
  * Requiring 2 of 3 rather than 1 matters. Healthy history has isolated
    spikes (25580: 17, 25584: 12, 25588: 11, 25591: 10), and a single-epoch
    rule would page on every one of them. Requiring 2 *consecutive* epochs is
    worse: it would be blinded by a lucky middle epoch, such as 25615
    (best 5) between 25614 (31) and 25616 (10) during the incident.
  * Fleet mean rank is reported as context only, not as a trigger. On
    25604-25616 the 3-epoch mean_pct rose +0.12 over baseline already at
    25611, which is labelled healthy, so on its own it would false-alarm
    before the incident started.
FLEET_ABSENT fires when the field was scored (N > 0) but none of our hotkeys
  appear. It needs only one epoch, because no amount of noise explains that.
FLEET_COLLAPSED fires when n < 50% of the baseline median fleet size, for
  example 25614, where 1 entry appeared against a baseline of 13.
Each condition alerts once at onset, then reminds every ``REMIND_EVERY = 5``
epochs while it lasts, and logs a RECOVERED line when it clears.

Validation against 2026-10-09 labelled data (``--replay 25560 25616``)
----------------------------------------------------------------------
  healthy 25599-25611: best ranks 1,4,3,1,1,2,6,4,2,3,1,3 -> no rank alert.
  degraded 25612-25616: best 6,12,31,5,10. 25612 is indistinguishable from
    healthy on every metric. RANK_REGRESSION fires at 25614, the second
    distinguishable bad epoch after 25613, and FLEET_COLLAPSED also fires at
    25614. Detection comes ~1-2 epochs (~1.5-2.5 h) after onset, not ~7 h.
  The rule also fired across 25572-25574 (best 18, then 15, fleet 10 -> 7,
  field-wide empty epoch 25573) and FLEET_ABSENT fired at 25597/25598/25601
  (field scored, zero of our entries). Those are real outages, not noise.

A provisional epoch (the newest one, first seen < ``FINALIZE_MIN`` minutes ago
and with no newer epoch scored yet) is skipped. A half-written leaderboard
could otherwise look like FLEET_ABSENT.

Usage
-----
  tools/rank_watch.py                   # normal run (launchd)
  tools/rank_watch.py --dry-run         # evaluate, print, no log/notify/state
  tools/rank_watch.py --replay A B      # replay epochs A..B from the cache,
                                        # print per-epoch flags + events
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

COLDKEY = "5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ"
API = "https://compound-api.metanova-labs.ai/api/competitions/leaderboard/{epoch}/nanobodies?limit=10000"

ROOT = Path(__file__).resolve().parents[1]
WATCH_DIR = ROOT / "data" / "rank_watch"
CACHE = WATCH_DIR / "cache"
STATE = WATCH_DIR / "state.json"
ALERT_LOG = WATCH_DIR / "alerts.log"

SEED_EPOCH = 25616          # used only when the cache is empty
LOOKBACK = 48
BASELINE_WINDOW = 12
MIN_BASELINE = 8
BAD_WINDOW = 3
BAD_NEEDED = 2
RANK_FLOOR = 10
COLLAPSE_FRAC = 0.5
REMIND_EVERY = 5
MAX_404_STREAK = 3
REFETCH_NEWEST = 2
FINALIZE_MIN = 30


# ---------------------------------------------------------------- fetching
def fetch(epoch: int) -> int:
    """Download one epoch to the cache. Returns HTTP status (0 on network error)."""
    url = API.format(epoch=epoch)
    req = urllib.request.Request(url, headers={"User-Agent": "nova-rank-watch/1"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
            tmp = CACHE / f".{epoch}.part"
            tmp.write_bytes(body)
            try:
                json.loads(tmp.read_text())          # validate from disk
            except ValueError:
                tmp.unlink(missing_ok=True)
                return 0
            tmp.replace(CACHE / f"{epoch}.json")
            return 200
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return 404
            err = exc.code
        except Exception:
            err = 0
        time.sleep(2 * 2 ** attempt)
    return err


def cached_epochs() -> list[int]:
    return sorted(int(p.stem) for p in CACHE.glob("[0-9]*.json"))


def sync_cache(state: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    have = cached_epochs()
    latest = have[-1] if have else SEED_EPOCH - 1
    # backfill the lookback window (immutable once scored)
    lo = latest - LOOKBACK + 1
    for e in range(lo, latest + 1):
        if e not in have:
            fetch(e)
    # refresh the newest few, then probe forward
    for e in [x for x in have if x > latest - REFETCH_NEWEST]:
        fetch(e)
    e, streak = latest + 1, 0
    while streak < MAX_404_STREAK:
        code = fetch(e)
        if code == 200:
            streak = 0
        elif code == 404:
            streak += 1
        else:
            print(f"fetch {e}: status {code}; stopping probe this run", file=sys.stderr)
            break
        e += 1
    seen = state.setdefault("first_seen", {})
    now = time.time()
    for x in cached_epochs():
        seen.setdefault(str(x), now)
    # prune
    keep = set(map(str, cached_epochs()[-LOOKBACK * 2:]))
    state["first_seen"] = {k: v for k, v in seen.items() if k in keep}


# ---------------------------------------------------------------- metrics
def load_epoch(path: Path) -> dict | None:
    try:
        rows = json.loads(path.read_text()).get("data") or []
    except ValueError:
        return None
    N = len(rows)
    ours = sorted(int(r["nanobody_leaderboard_rank"]) for r in rows
                  if r.get("coldkey") == COLDKEY and r.get("nanobody_leaderboard_rank") is not None)
    return {
        "epoch": int(path.stem), "N": N, "n": len(ours), "ranks": ours,
        "best": ours[0] if ours else None,
        "mean": st.mean(ours) if ours else None,
        "mean_pct": st.mean(ours) / N if ours and N else None,
        "top10": sum(r <= 10 for r in ours),
        "bottom_half": sum(r > N / 2 for r in ours),
    }


def evaluate(epochs: list[dict]) -> list[dict]:
    """Replay in order; annotate each epoch with baseline + flags."""
    healthy: list[dict] = []
    present: list[dict] = []
    out = []
    for x in epochs:
        r = dict(x, flags=[], judged=False)
        out.append(r)
        if x["N"] == 0:
            r["note"] = "empty field (validator did not score)"
            continue
        base = healthy[-BASELINE_WINDOW:]
        if len(base) >= MIN_BASELINE:
            r["judged"] = True
            bb = st.median(b["best"] for b in base)
            bn = st.median(b["n"] for b in base)
            bmp = st.median(b["mean_pct"] for b in base)
            T = max(RANK_FLOOR, 2 * bb + 2)
            r.update(base_best=bb, base_n=bn, base_mean_pct=bmp, T=T,
                     base_epochs=[b["epoch"] for b in base])
        if x["n"] == 0:
            if r["judged"]:
                r["flags"].append("FLEET_ABSENT")
            continue
        present.append(r)
        if not r["judged"]:
            healthy.append(r)                           # seeding
            continue
        r["bad_best"] = r["best"] >= r["T"]
        win = present[-BAD_WINDOW:]
        r["window"] = [(w["epoch"], w["best"]) for w in win]
        nbad = sum(1 for w in win if w.get("bad_best"))
        if r["bad_best"] and nbad >= BAD_NEEDED:
            r["flags"].append("RANK_REGRESSION")
        elif nbad >= BAD_NEEDED and not r["bad_best"]:
            # still inside a bad window (lucky epoch); keep the incident open
            r["flags"].append("RANK_REGRESSION")
        if x["n"] < COLLAPSE_FRAC * r["base_n"]:
            r["flags"].append("FLEET_COLLAPSED")
        r["win_mean_pct"] = st.mean(w["mean_pct"] for w in win)
        if not r["flags"] and not r["bad_best"]:
            healthy.append(r)
    return out


# ---------------------------------------------------------------- messages
def describe(kind: str, r: dict, phase: str) -> str:
    e = r["epoch"]
    if phase == "RECOVERED":
        return (f"[RECOVERED] {kind} cleared at epoch {e}: best rank {r.get('best')}, "
                f"fleet {r['n']}/{r['N']}, mean rank {fmt(r.get('mean'))}.")
    head = f"[{phase}] {kind} at epoch {e} (P20809 nanobody)"
    ctx = (f"baseline (median of {len(r.get('base_epochs', []))} healthy epochs "
           f"{r['base_epochs'][0]}-{r['base_epochs'][-1]}): best rank {fmt(r['base_best'])}, "
           f"fleet size {fmt(r['base_n'])}, mean rank pct {fmt(r['base_mean_pct'])}")
    if kind == "FLEET_ABSENT":
        what = (f"Field of {r['N']} miners was scored but NONE of our coldkey's hotkeys appear. "
                "Submissions are not landing (submitter down, commits missed, or hotkeys deregistered).")
    elif kind == "FLEET_COLLAPSED":
        what = (f"Only {r['n']} of our entries scored vs baseline {fmt(r['base_n'])} "
                f"(field {r['N']}); ranks {r['ranks']}. Most of the fleet failed to submit or was rejected.")
    else:
        what = (f"Best rank {r['best']} (alert threshold >= {fmt(r['T'])}); last {BAD_WINDOW} epochs "
                f"(epoch, best) = {r['window']}. Fleet {r['n']}/{r['N']}, mean rank {fmt(r['mean'])} "
                f"(pct {fmt(r['mean_pct'])}, 3-epoch {fmt(r.get('win_mean_pct'))}), top-10 slots {r['top10']}, "
                f"bottom-half {r['bottom_half']}/{r['n']}.")
    if kind == "RANK_REGRESSION":
        act = ("Check: which candidate bank / source each submitter is reading and when it was last "
               "refreshed (2026-10-09 root cause: frozen bank); recent deploys or service swaps.")
    else:
        act = ("Check: submitter services alive, commit/reveal landed for this epoch, hotkeys "
               "still registered, per-hotkey submit logs.")
    return f"{head}\n  {what}\n  {ctx}\n  {act}"


def fmt(v):
    return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def notify(title: str, body: str) -> None:
    script = f'display notification {json.dumps(body[:240])} with title {json.dumps(title)} sound name "Basso"'
    subprocess.run(["/usr/bin/osascript", "-e", script], capture_output=True, timeout=20)


def events_for(rows: list[dict], after: int, state_open: dict) -> list[tuple[str, dict, str]]:
    """Walk rows; emit (kind, row, phase) for rows with epoch > after."""
    open_ = dict(state_open)     # kind -> {"onset": e, "last_alert": e}
    ev = []
    for r in rows:
        if not r["judged"]:
            continue
        e = r["epoch"]
        for kind in ("RANK_REGRESSION", "FLEET_COLLAPSED", "FLEET_ABSENT"):
            on = kind in r["flags"]
            cur = open_.get(kind)
            if on and cur is None:
                open_[kind] = {"onset": e, "last_alert": e}
                if e > after:
                    ev.append((kind, r, "ALERT"))
            elif on and e - cur["last_alert"] >= REMIND_EVERY:
                cur["last_alert"] = e
                if e > after:
                    ev.append((kind, r, f"STILL DEGRADED since {cur['onset']}"))
            elif not on and cur is not None and r["n"] > 0:
                del open_[kind]
                if e > after:
                    ev.append((kind, r, "RECOVERED"))
    return ev, open_


def table(rows):
    for r in rows:
        if r["N"] == 0:
            print(f"{r['epoch']}  (empty field)")
            continue
        print(f"{r['epoch']}  N={r['N']:3d} n={r['n']:2d} best={fmt(r['best']):>3} "
              f"mean_pct={fmt(r['mean_pct']):>5} top10={r['top10']}  "
              + (f"base_best={fmt(r['base_best']):>4} T={fmt(r['T']):>4} base_n={fmt(r['base_n']):>4}"
                 if r["judged"] else "seeding baseline")
              + ("  -> " + ",".join(r["flags"]) if r["flags"] else ""))


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replay", nargs=2, type=int, metavar=("FROM", "TO"))
    ap.add_argument("--cache", type=Path, help="alternate cache dir (replay only)")
    args = ap.parse_args()

    global CACHE
    if args.cache:
        CACHE = args.cache

    if args.replay:
        a, b = args.replay
        rows = [load_epoch(CACHE / f"{e}.json") for e in range(a, b + 1)
                if (CACHE / f"{e}.json").exists()]
        rows = evaluate([r for r in rows if r])
        table(rows)
        ev, _ = events_for(rows, after=0, state_open={})
        print("\nevents:")
        for kind, r, phase in ev:
            print(describe(kind, r, phase))
        return 0

    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    sync_cache(state)
    eps = cached_epochs()
    if not eps:
        print("no epochs cached yet", file=sys.stderr)
        return 1
    rows = [load_epoch(CACHE / f"{e}.json") for e in eps[-LOOKBACK:]]
    rows = [r for r in rows if r]
    # drop a provisional newest epoch
    newest = rows[-1]
    age_min = (time.time() - state["first_seen"].get(str(newest["epoch"]), 0)) / 60
    if age_min < FINALIZE_MIN:
        rows = rows[:-1]
    rows = evaluate(rows)
    if not rows:
        return 0
    last = rows[-1]["epoch"]
    first_run = "last_processed" not in state
    after = state.get("last_processed", last)
    ev, open_now = events_for(rows, after=after, state_open={})
    if first_run:   # surface anything already in progress at install time
        ev = [(k, rows[-1], f"ALERT (ongoing since {v['onset']})") for k, v in open_now.items()]

    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    r = rows[-1]
    print(f"{stamp} latest finalized epoch {last}: best={r.get('best')} n={r['n']}/{r['N']} "
          f"flags={r['flags'] or 'none'} T={fmt(r.get('T'))} events={len(ev)}")
    for kind, row, phase in ev:
        msg = describe(kind, row, phase)
        print(msg)
        if args.dry_run:
            continue
        with ALERT_LOG.open("a") as fh:
            fh.write(f"{stamp}  {msg}\n\n")
        title = f"SN68 nanobody {kind.replace('_', ' ').lower()} {'recovered' if phase == 'RECOVERED' else ''}".strip()
        notify(title, msg.split("\n")[1].strip())
    if not args.dry_run:
        state["last_processed"] = last
        state["last_run"] = stamp
        STATE.write_text(json.dumps(state, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

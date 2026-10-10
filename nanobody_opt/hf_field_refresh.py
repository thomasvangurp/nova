#!/usr/bin/env python3
"""Per-epoch refresh: extend the ``hf_field_20x_rank_sum`` pool with NEW sequences.

The parent pool is the flat top-50 by 20x HF-field rank-sum across three
sources.  This script grows that pool over time.  Each run it recomputes the
three source sets, finds the sequences that are NOT yet scored at 20x
(``hf_field_20x_rank_sum``), 20x-scores just those, and imports them.

Sources (all 20x, one shared 21k-58k rank-sum scale, lower = better):

  our_top50_20x           our top-50 by the mixed replay rank-sum read
  public_multipass_rivals every rival coldkey's submissions this epoch
                          (leaderboard API, coldkey != ours)
  hf_archive_top200       rank_binders top-200 over the raw HF archive

Idempotent and cheap to re-run: scoring only covers sequences missing from the
pool, so overlapping runs / restarts are safe.

What a row holds (2026-10-10 fix). One row per sequence under SCORE_KIND:

  score         20-pass MEAN sum-of-10 vs the HF archive + the scored batch
  metrics_json  the per-metric MEAN of the raw BoltzGen metrics over the 20
                passes (all ten SUM_OF_TEN_DIRS metrics, always)
  provenance    writer, formula, passes, cohort, field rows + sha256, and the
                RAW per-pass metrics (``per_pass_metrics``, pass order) plus
                ``per_pass_scores``, so any later formula / reference field
                can be recomputed per pass with no GPU

Before this fix every row had ``metrics_json='{}'``: the raw metrics were in
``score<k>/exact_all.json`` on disk but score_pass() dropped them and the
observation dict never carried a ``metrics`` key.

GPU / wall-clock guard (2026-10-10, after the GPU-contending 20x re-fold of
2026-10-09). The script itself -- not only its loop wrapper -- refuses to run
unless it gets a provably safe window, on ANY box:

  * no live-producer / GPU-scoring process on this host (PRODUCER_MARKERS);
  * every selected GPU idle (util, memory, compute apps) in EVERY sample of a
    sustained window, and nvidia-smi must answer (fail closed);
  * one instance at a time (flock), never through the shared scorer daemon;
  * a hard wall-clock cap per invocation (``--max-wall-hours``, ceiling
    HARD_MAX_WALL_HOURS) and a sequence-pass budget per invocation;
  * while scoring, a watchdog kills the scorer's process group the moment a
    producer process appears or the deadline passes.

A refusal or abort is LOUD: a ``HF_REFRESH_GUARD_REFUSED`` /
``HF_REFRESH_ABORTED`` marker on stderr, a ``pipeline_failures.jsonl`` entry,
and a distinct exit code (3 refused, 4 aborted). It never degrades to running
alongside the producer; there is no busy-but-stale override.

Run on a dedicated box (bgen env, over the tailscale MySQL relay)::

    SN68_MYSQL_HOST=127.0.0.1 SN68_MYSQL_PORT=3306 HF_FIELD_REFRESH_ENABLED=1 \
      /root/.pixi/envs/bgen/bin/python /root/hf_field_refresh.py
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable

import pandas as pd

TARGET = "P20809"
# A distinct label from the pre-sum-of-10 era: "hf_field_20x_rank_sum" rows
# already in the DB were written under algo2_score (minmax over weighted
# metric ranks) through epoch ~25610, and mixing sum-of-10 values into that
# same label would corrupt every comparison that reads it. New rows land
# under this label instead; the stale ones are untouched, not deleted.
SCORE_KIND = "hf_field_sum10_20x_rank_sum"
N_PASSES = 20
OUR_COLDKEY = "5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ"
API = "https://compound-api.metanova-labs.ai"

ARCHIVE = "/root/P20809_nanobodies.csv"
CONFIG = "/root/config/config.yaml"
SCORER = "/root/nanobody_opt/score_boltzgen_fast_adapter.py"
OUT_ROOT = Path("/root/hf_field_refresh")
#: provenance.writer (registered for SCORE_KIND in observation_labels).
WRITER = "hf_field_refresh"
FORMULA_ID = "sum10"
#: provenance.cohort per refresh source; a sequence in our top-50 is "ours".
SOURCE_COHORT = {"our_top50_20x": "ours", "public_multipass_rivals": "rival",
                 "hf_archive_top200": "hf_top200"}
COMPONENT = "hf_field_refresh"

# --- GPU / wall-clock guard ------------------------------------------------
#: Exit codes. 2 = disabled (P3 gate), 3 = no safe window, 4 = aborted mid-run.
EXIT_DISABLED, EXIT_REFUSED, EXIT_ABORTED = 2, 3, 4
#: Any process whose argv contains one of these means the live producer (or
#: another GPU scorer) owns this host's GPUs right now. Our own process tree
#: is excluded. Deliberately broad: a false refusal costs one 30-min tick, a
#: false start cost ~80 GPU-h and the producer's race on 2026-10-09.
PRODUCER_MARKERS = (
    "nanobody_opt.run_live_producer", "run_live_producer.py", "live-producer",
    "nova-nanobody-producer", "session_watchdog.sh", "producer_relaunch.sh",
    "sequential_exact_race", "qualify_screen_leader", "competitor_lane",
    "children_submit", "publish_current_nanobody", "publish-current-nanobody",
    "scorer_daemon", "multipass_all_rivals", "backfill_topup.py",
    "score_top50_32.py", "score_boltzgen",
)
#: A GPU counts as idle only below BOTH thresholds in EVERY sample.
GPU_IDLE_UTIL_PCT = 10
GPU_IDLE_MEM_MIB = 1024
#: Sustained idle window before starting (samples x interval seconds).
PREFLIGHT_SAMPLES = 6
PREFLIGHT_INTERVAL_S = 10.0
#: Watchdog poll period while the scorer runs.
MONITOR_POLL_S = 15.0
#: Wall-clock cap per invocation: default, and a ceiling no flag can exceed.
DEFAULT_MAX_WALL_HOURS = 2.0
HARD_MAX_WALL_HOURS = 6.0
#: Sequence-passes (sequences x N_PASSES) per invocation; the rest is deferred
#: to the next run (the refresh is idempotent). 2000 = 100 sequences, ~2.7
#: GPU-h at the 2026-10-09 rate (~730 sequence-passes per GPU-hour).
DEFAULT_MAX_SEQUENCE_PASSES = 2000
HARD_MAX_SEQUENCE_PASSES = 10000
#: Safety margin on the measured pass time when deciding if a pass still fits.
PASS_TIME_MARGIN = 1.25
LOCK_PATH = Path(os.environ.get("HF_REFRESH_LOCK", "/tmp/hf_field_refresh.py.lock"))

METRICS = ["design_iiptm", "design_ptm", "design_to_target_iptm",
           "min_design_to_target_pae", "interaction_pae",
           "plip_hbonds_refolded", "plip_saltbridge_refolded",
           "delta_sasa_refolded", "liability_score"]
WEIGHTS = {"design_iiptm": 1.0, "design_ptm": 2.0, "min_design_to_target_pae": 1.0,
           "plip_hbonds_refolded": 2.0, "plip_saltbridge_refolded": 2.0,
           "delta_sasa_refolded": 2.0}
DIRS = {"design_iiptm": "max", "design_ptm": "max", "min_design_to_target_pae": "min",
        "plip_hbonds_refolded": "max", "plip_saltbridge_refolded": "max",
        "delta_sasa_refolded": "max"}

# Validator's actual score is the SUM of the ten metric dense ranks
# (rho=1.000 vs final_score), not the Algorithm-2 minmax (~0.56).
SUM_OF_TEN_DIRS = {
    "design_iiptm": "max", "design_ptm": "max", "design_to_target_iptm": "max",
    "min_design_to_target_pae": "min", "interaction_pae": "min",
    "plip_hbonds_refolded": "max", "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max", "liability_score": "min",
    "liability_num_violations": "min",
}


def sum_of_ten(field: pd.DataFrame) -> pd.Series:
    """Sum of the ten metric dense ranks (lower = better)."""
    total = pd.Series(0.0, index=field.index)
    for metric, direction in SUM_OF_TEN_DIRS.items():
        if metric not in field.columns:
            continue
        total += field[metric].rank(method="dense", ascending=(direction == "min"))
    return total

# Our top-50: the exact mixed replay read `_live_top50_from_db` used (no
# fillers), so "our own new top-50" means the same stable quality read.
OUR_TOP50_SQL = """
    SELECT n.sequence, AVG(o.score) AS mean_score
    FROM observations o JOIN nodes n ON o.node_id = n.id
    WHERE n.target = %s
      AND o.score_kind IN ('local_validator_replay_rank_sum',
                           'local_exact_validator_replay_rank_sum')
    GROUP BY n.sequence
    HAVING COUNT(*) >= 8
    ORDER BY mean_score ASC
    LIMIT 50
"""


def algo2_score(field: pd.DataFrame) -> pd.Series:
    work = field[field["liability_num_violations"] <= 50].copy()
    rc = {}
    for metric, weight in WEIGHTS.items():
        if metric not in work.columns:
            continue
        rc[metric] = work[metric].rank(ascending=(DIRS[metric] == "min"),
                                       method="average") / weight
    return pd.DataFrame(rc, index=work.index).max(axis=1)


def latest_completed_epoch(start: int, lookback: int = 60) -> int | None:
    """Walk backward from ``start`` until the leaderboard API returns 200."""
    for epoch in range(start, start - lookback, -1):
        try:
            req = urllib.request.Request(
                f"{API}/api/competitions/leaderboard/{epoch}/nanobodies",
                headers={"User-Agent": "nova-hf-field-refresh/1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
            return epoch
        except Exception:
            continue
    return None


def fetch_rivals(epoch: int) -> set[str]:
    try:
        req = urllib.request.Request(
            f"{API}/api/competitions/leaderboard/{epoch}/nanobodies",
            headers={"User-Agent": "nova-hf-field-refresh/1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
    except Exception:
        return set()
    out: set[str] = set()
    for submission in payload.get("data") or []:
        if str(submission.get("coldkey") or "").strip() == OUR_COLDKEY:
            continue
        for nb in submission.get("nanobodies") or []:
            if nb.get("protein_name") != TARGET:
                continue
            seq = str(nb.get("sequence") or "").strip().upper()
            if seq:
                out.add(seq)
    return out


def fetch_hf_top200(archive: pd.DataFrame) -> set[str]:
    # One copy of this ranking: nanobody_opt.hf_topup (R1), which now feeds
    # the live parent pool through the dashboard pass scorer.
    from nanobody_opt.hf_topup import hf_archive_top
    return set(hf_archive_top(archive, 200))


def fetch_db_rivals(conn) -> set[str]:
    """Every rival sequence we have ever scored (cumulative, all epochs)."""
    rows = conn.execute(
        """
        SELECT DISTINCT n.sequence
        FROM observations o JOIN nodes n ON o.node_id = n.id
        WHERE o.source IN ('public_multipass_rivals', 'slave_public_multipass_rivals')
        """,
    ).fetchall()
    return {str(r[0]).strip().upper() for r in rows}


def fetch_our_top50(conn) -> set[str]:
    rows = conn.execute(OUR_TOP50_SQL, (TARGET,)).fetchall()
    return {str(r[0]).strip().upper() for r in rows}


def already_scored(conn) -> set[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT n.sequence
        FROM observations o JOIN nodes n ON o.node_id = n.id
        WHERE o.score_kind = %s
        """,
        (SCORE_KIND,),
    ).fetchall()
    return {str(r[0]).strip().upper() for r in rows}


# ===========================================================================
# GPU / wall-clock guard
# ===========================================================================
class GuardRefused(RuntimeError):
    """No provably safe window. Nothing was scored or written (exit 3)."""


class GuardAborted(RuntimeError):
    """The safe window closed mid-run; the scorer was killed (exit 4)."""


def _loud(marker: str, reason: str, **context) -> None:
    """Marker on stdout (refresh.log) AND stderr, plus pipeline_failures.jsonl."""
    line = f"{marker} {reason}"
    print(line, flush=True)
    print(line, file=sys.stderr, flush=True)
    try:
        from nanobody_opt.failure_log import record_failure
        record_failure(COMPONENT, marker, None, reason=reason, **context)
    except Exception as exc:  # noqa: BLE001 - the stdout/stderr marker already fired
        print(f"HF_REFRESH_FAILURE_LOG_UNAVAILABLE {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)


class SystemProbe:
    """Reads this host's process table and GPUs. Tests substitute a fake."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def process_table(self) -> list[tuple[int, int, str]]:
        """``[(pid, ppid, argv)]``; raises GuardRefused if ps cannot answer."""
        try:
            out = subprocess.run(["ps", "-eo", "pid=,ppid=,args="],
                                 capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GuardRefused(f"ps unavailable: {type(exc).__name__}: {exc}") from exc
        if out.returncode != 0:
            raise GuardRefused(f"ps failed rc={out.returncode}")
        rows = []
        for line in out.stdout.splitlines():
            parts = line.split(None, 2)
            if len(parts) < 2:
                continue
            try:
                pid, ppid = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            rows.append((pid, ppid, parts[2] if len(parts) > 2 else ""))
        if not rows:
            raise GuardRefused("ps returned no processes")
        return rows

    def _smi(self, query: str) -> list[list[str]]:
        try:
            out = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GuardRefused(f"nvidia-smi unavailable: {type(exc).__name__}: {exc}") from exc
        if out.returncode != 0:
            raise GuardRefused(f"nvidia-smi {query} rc={out.returncode}: "
                               f"{(out.stderr or '').strip()[:200]}")
        return [[c.strip() for c in line.split(",")]
                for line in out.stdout.splitlines() if line.strip()]

    def gpu_state(self) -> list[dict]:
        """``[{index, uuid, util, mem_mib}]``; any unreadable value refuses."""
        gpus = []
        for row in self._smi("--query-gpu=index,uuid,utilization.gpu,memory.used"):
            try:
                gpus.append({"index": int(row[0]), "uuid": row[1],
                             "util": float(row[2]), "mem_mib": float(row[3])})
            except (IndexError, ValueError) as exc:
                raise GuardRefused(f"unreadable nvidia-smi row {row}") from exc
        if not gpus:
            raise GuardRefused("nvidia-smi lists no GPUs")
        return gpus

    def gpu_compute_apps(self) -> list[dict]:
        """``[{uuid, pid}]`` of every compute process nvidia-smi can see."""
        apps = []
        for row in self._smi("--query-compute-apps=gpu_uuid,pid"):
            if not row or not row[0]:
                continue
            pid = row[1] if len(row) > 1 else ""
            apps.append({"uuid": row[0], "pid": int(pid) if pid.isdigit() else None})
        return apps


def own_tree(table: list[tuple[int, int, str]], self_pid: int) -> set[int]:
    """``self_pid`` and all its descendants (our scorer and its shard workers).

    Ancestors are NOT excluded: if this script were ever launched from inside
    a producer session, that producer must still count as a contender.
    """
    children: dict[int, list[int]] = {}
    for pid, ppid, _ in table:
        children.setdefault(ppid, []).append(pid)
    tree, stack = {self_pid}, [self_pid]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in tree:
                tree.add(child)
                stack.append(child)
    return tree


def foreign_contenders(table: list[tuple[int, int, str]], self_pid: int,
                       markers: tuple[str, ...] = PRODUCER_MARKERS) -> list[str]:
    """Processes outside our own tree whose argv names a producer/GPU scorer."""
    mine = own_tree(table, self_pid)
    hits = []
    for pid, _ppid, args in table:
        if pid in mine:
            continue
        for marker in markers:
            if marker in args:
                hits.append(f"pid={pid} [{marker}] {args[:160]}")
                break
    return hits


def assert_safe_window(gpus: list[int], probe, *, self_pid: int,
                       samples: int = PREFLIGHT_SAMPLES,
                       interval: float = PREFLIGHT_INTERVAL_S) -> None:
    """Refuse unless the host is provably free for the whole sample window.

    Every one of ``samples`` samples (``interval`` s apart) must show: no
    producer/GPU-scoring process, every selected GPU present, below
    GPU_IDLE_UTIL_PCT and GPU_IDLE_MEM_MIB, and no compute process on it. One
    busy sample refuses. A probe that cannot answer refuses (fail closed).
    Unlike the old loop gate there is no "stale enough, run anyway" override.
    """
    if not gpus:
        raise GuardRefused("no GPUs selected")
    for i in range(samples):
        if i:
            probe.sleep(interval)
        contenders = foreign_contenders(probe.process_table(), self_pid)
        if contenders:
            raise GuardRefused(
                f"live producer / GPU scorer running on this host "
                f"(sample {i + 1}/{samples}): " + " ; ".join(contenders[:5]))
        state = {g["index"]: g for g in probe.gpu_state()}
        missing = [g for g in gpus if g not in state]
        if missing:
            raise GuardRefused(f"selected GPUs {missing} not present (host has {sorted(state)})")
        for g in gpus:
            util, mem = state[g]["util"], state[g]["mem_mib"]
            if util >= GPU_IDLE_UTIL_PCT or mem >= GPU_IDLE_MEM_MIB:
                raise GuardRefused(
                    f"GPU {g} busy in sample {i + 1}/{samples}: util={util:g}% "
                    f"mem={mem:g}MiB (idle needs <{GPU_IDLE_UTIL_PCT}% and "
                    f"<{GPU_IDLE_MEM_MIB}MiB)")
        selected = {state[g]["uuid"] for g in gpus}
        known = {g["uuid"] for g in state.values()}
        apps = [a for a in probe.gpu_compute_apps()
                if a["uuid"] in selected or a["uuid"] not in known]
        if apps:
            raise GuardRefused(
                f"compute process(es) on the selected GPUs (sample {i + 1}/{samples}): "
                + ", ".join(f"pid={a['pid']} gpu={a['uuid']}" for a in apps[:8]))
    print(f"HF_REFRESH_GUARD_OK gpus={','.join(map(str, gpus))} "
          f"samples={samples} interval={interval:g}s", flush=True)


class Watchdog:
    """While the scorer runs: kill its process group the moment a producer /
    GPU scorer appears on the host, a visible foreign process takes one of our
    GPUs, a probe fails, or the wall-clock deadline passes."""

    def __init__(self, probe, *, deadline: float, gpus: list[int], self_pid: int,
                 poll_s: float = MONITOR_POLL_S, kill_grace_s: float = 30.0):
        self.probe, self.deadline, self.gpus = probe, deadline, list(gpus)
        self.self_pid, self.poll_s, self.kill_grace_s = self_pid, poll_s, kill_grace_s
        self.abort_reason: str | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hf-refresh-watchdog",
                                        daemon=True)

    def remaining(self) -> float:
        return self.deadline - self.probe.now()

    def check_once(self) -> str | None:
        if self.remaining() <= 0:
            return "wall-clock deadline reached"
        table = self.probe.process_table()
        contenders = foreign_contenders(table, self.self_pid)
        if contenders:
            return "live producer / GPU scorer appeared: " + " ; ".join(contenders[:5])
        # Container pid namespaces can hide or remap compute-app pids, so only a
        # pid we can SEE on this host and that is not ours counts mid-run (the
        # preflight already required zero compute processes before we started).
        visible = {pid for pid, _, _ in table}
        mine = own_tree(table, self.self_pid)
        selected = {g["uuid"] for g in self.probe.gpu_state() if g["index"] in self.gpus}
        foreign = [a for a in self.probe.gpu_compute_apps()
                   if a["uuid"] in selected and a["pid"] in visible and a["pid"] not in mine]
        if foreign:
            return ("foreign compute process on our GPUs: "
                    + ", ".join(f"pid={a['pid']}" for a in foreign[:8]))
        return None

    def trip(self, reason: str) -> None:
        with self._lock:
            if self.abort_reason is None:
                self.abort_reason = reason
            proc = self._proc
        if proc is not None and proc.poll() is None:
            _kill_group(proc, self.kill_grace_s)

    def attach(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._proc = proc
            reason = self.abort_reason
        if reason is not None:
            self.trip(reason)

    def detach(self) -> None:
        with self._lock:
            self._proc = None

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                reason = self.check_once()
            except Exception as exc:  # noqa: BLE001 - a blind watchdog must not keep running
                reason = f"watchdog probe failed (fail closed): {type(exc).__name__}: {exc}"
            if reason:
                self.trip(reason)
                return

    def start(self) -> "Watchdog":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def _kill_group(proc: subprocess.Popen, grace_s: float) -> None:
    """SIGTERM the scorer's whole process group (its shard workers too, so no
    orphaned GPU workers), then SIGKILL after ``grace_s``."""
    for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 10.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        except OSError:
            proc.send_signal(sig)
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def run_guarded(cmd: list[str], watchdog: Watchdog, *, log_path: Path,
                env: dict[str, str] | None = None) -> None:
    """Run one scorer pass under the watchdog and the hard deadline."""
    remaining = watchdog.remaining()
    if watchdog.abort_reason:
        raise GuardAborted(watchdog.abort_reason)
    if remaining <= 0:
        raise GuardAborted("wall-clock deadline reached before the scorer pass")
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
    watchdog.attach(proc)
    try:
        rc = proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        watchdog.trip("wall-clock deadline reached during a scorer pass")
        rc = proc.wait()
    finally:
        watchdog.detach()
    if watchdog.abort_reason:
        raise GuardAborted(watchdog.abort_reason)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def acquire_lock(path: Path = LOCK_PATH):
    """One refresh per host, however it was launched (loop, cron or by hand)."""
    handle = open(path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise GuardRefused(f"another hf_field_refresh holds {path}") from exc
    return handle


def plan_budget(batches: list[tuple[str, list[str]]], max_sequence_passes: int
                ) -> tuple[list[tuple[str, list[str]]], dict[str, int]]:
    """Cap this invocation at ``max_sequence_passes``; defer the rest, in order."""
    left = max_sequence_passes // N_PASSES
    planned, deferred = [], {}
    for source, seqs in batches:
        take = seqs[:max(left, 0)]
        left -= len(take)
        planned.append((source, take))
        deferred[source] = len(seqs) - len(take)
    return planned, deferred


def assert_label_open() -> None:
    """Before any GPU time: the registry must accept the rows we will write and
    ``sum_of_ten`` must still compute the registered formula. A closed label is
    refused here, not after a 40-minute scoring run."""
    from nanobody_opt import observation_labels
    observation_labels.assert_formula(SCORE_KIND, FORMULA_ID, sum_of_ten)
    observation_labels.check_observations([
        {"observation_id": "hf20x:preflight", "sequence": "A", "target": TARGET,
         "source": source, "score_kind": SCORE_KIND, "score": 1.0,
         "writer": WRITER, "formula": FORMULA_ID}
        for source in SOURCE_COHORT])


def gpu_ids(arg: str | None, probe) -> list[int]:
    if arg:
        return [int(x) for x in arg.split(",") if x.strip()]
    return sorted(g["index"] for g in probe.gpu_state())


# ===========================================================================
# Scoring and the observation rows
# ===========================================================================
def raw_metrics_from_payload(payload: dict, wanted: set[str] | None = None
                             ) -> dict[str, dict[str, float]]:
    """``{sequence: {metric: value}}``: every finite numeric metric of one pass
    (wave 0, TARGET), for the requested sequences only. Formula-independent."""
    wave = (payload.get("per_nanobody_components") or {}).get("0") or {}
    out: dict[str, dict[str, float]] = {}
    for seq, comp in wave.items():
        key = str(seq).strip().upper()
        if wanted is not None and key not in wanted:
            continue
        m = (comp or {}).get(TARGET, {}) or {}
        values = {k: float(v) for k, v in m.items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool)
                  and math.isfinite(float(v))}
        if values:
            out[key] = values
    return out


def rank_pass(raw: dict[str, dict[str, float]], archive: pd.DataFrame) -> dict[str, float]:
    """Sum-of-10 of each scored sequence vs the archive + this pass's batch.

    Unchanged from the pre-fix scorer (same field, same duplicate handling), so
    the stored score stays on the label's existing scale.
    """
    cols = METRICS + ["liability_num_violations"]
    rows = {s: {k: v for k, v in m.items() if k in cols} for s, m in raw.items()}
    rival = pd.DataFrame(rows).T
    arch_s = archive.set_index("sequence")[cols]
    field = pd.concat([arch_s, rival])
    scores = sum_of_ten(field)  # sum-of-10 (lower = better)
    out = {}
    for s in rival.index:
        mask = scores.index == s
        if mask.any():
            out[s] = float(scores[mask].iloc[-1])
    return out


def score_pass(sequences: list[str], run: int, archive: pd.DataFrame,
               out_root: Path, gpus: str, runner: Callable[[list[str], Path], None]
               ) -> tuple[dict[str, float], dict[str, dict[str, float]], str | None]:
    """One scorer pass -> ({seq: sum10}, {seq: raw metrics}, scorer_protocol)."""
    inp = out_root / f"input_{run}.json"
    out_dir = out_root / f"score{run}"
    inp.write_text(json.dumps({"sequences": [{"sequence": s} for s in sequences]}))
    runner(
        [sys.executable, "-m", "nanobody_opt.score_boltzgen_waves",
         "--sequences", str(inp), "--config", CONFIG, "--target", TARGET,
         "--output-dir", str(out_dir), "--gpus", gpus,
         "--wave-size", str(len(sequences)), "--predict-batch-size", "2",
         "--limit", str(len(sequences)), "--scorer", SCORER,
         "--capture-structures"],
        out_root / f"scorer_{run}.log",
    )
    payload = json.loads((out_dir / "exact_all.json").read_text())
    raw = raw_metrics_from_payload(payload, set(sequences))
    return rank_pass(raw, archive), raw, payload.get("scorer_protocol")


def field_fingerprint(path: str, archive: pd.DataFrame) -> dict:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"field": "hf_archive_full_plus_scored_batch", "field_path": str(path),
            "field_rows": int(len(archive)), "field_sha256": digest.hexdigest()}


def build_observations(source: str, per_pass_scores: list[dict[str, float]],
                       per_pass_raw: list[dict[str, dict[str, float]]], *,
                       field: dict, our: set[str], refresh_epoch: int | None,
                       scorer_protocol: str | None = None,
                       ) -> tuple[list[dict], list[str]]:
    """One row per sequence scored in EVERY pass with all ten metrics.

    Returns ``(observations, incomplete)``; ``incomplete`` are the sequences
    left out because some pass lacked a finite sum10 or one of the ten metrics
    (previously dropped silently as a NaN score).
    """
    passes = len(per_pass_scores)
    seen = set().union(*(set(p) for p in per_pass_raw)) if per_pass_raw else set()
    obs, incomplete = [], []
    for s in sorted(seen):
        raws = [p.get(s) for p in per_pass_raw]
        scores = [p.get(s) for p in per_pass_scores]
        if (any(r is None for r in raws)
                or any(sc is None or not math.isfinite(sc) for sc in scores)
                or any(m not in r for r in raws for m in SUM_OF_TEN_DIRS)):
            incomplete.append(s)
            continue
        shared = set.intersection(*(set(r) for r in raws))
        mean_metrics = {m: float(sum(r[m] for r in raws) / passes) for m in sorted(shared)}
        obs.append({
            "observation_id": f"hf20x:{s}",
            "sequence": s,
            "target": TARGET,
            "source": source,
            "score_kind": SCORE_KIND,
            "score": float(sum(scores) / passes),
            "observation_count": passes,
            # metrics_json: per-metric mean over the passes (flat {metric: v}).
            "metrics": mean_metrics,
            # Every key below lands in provenance_json.
            "writer": WRITER,
            "formula": FORMULA_ID,
            "cohort": "ours" if s in our else SOURCE_COHORT.get(source, source),
            "passes": passes,
            "metrics_aggregation": "mean_over_passes",
            "per_pass_metrics": raws,
            "per_pass_scores": [float(x) for x in scores],
            "refresh_epoch": refresh_epoch,
            "scorer_protocol": scorer_protocol,
            **field,
        })
    return obs, incomplete


def _retry_db(fn, *args, retries=6, delay=12, **kwargs):
    """Retry on the transient tailscale-relay 'Lost connection' errors."""
    import time as _time
    from MySQLdb import OperationalError

    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except OperationalError as exc:
            print(f"  DB retry {attempt}/{retries}: {exc}", flush=True)
            if attempt == retries:
                raise
            _time.sleep(delay)
    raise RuntimeError("unreachable")


def _default_writer(obs: list[dict]) -> int:
    from nanobody_opt.mutation_network import update_observation_store
    return _retry_db(update_observation_store, Path("/root/.sn68_mysql_creds"),
                     obs, commit_every=1000, commit_interval=5.0)


def score_and_import(source: str, seqs: list[str], archive: pd.DataFrame, gpus: str, *,
                     watchdog: Watchdog, field: dict, our: set[str],
                     refresh_epoch: int | None, env: dict[str, str],
                     writer: Callable[[list[dict]], int] = _default_writer,
                     out_root_base: Path | None = None) -> int:
    seqs = sorted(seqs)
    if not seqs:
        print(f"  {source}: nothing new", flush=True)
        return 0
    print(f"  {source}: scoring {len(seqs)} new sequences at {N_PASSES}x", flush=True)
    out_root = (out_root_base or OUT_ROOT) / source
    out_root.mkdir(parents=True, exist_ok=True)

    def runner(cmd: list[str], log_path: Path) -> None:
        run_guarded(cmd, watchdog, log_path=log_path, env=env)

    per_pass_scores: list[dict[str, float]] = []
    per_pass_raw: list[dict[str, dict[str, float]]] = []
    protocol = None
    pass_seconds: list[float] = []
    for run in range(1, N_PASSES + 1):
        if pass_seconds:
            need = max(pass_seconds) * PASS_TIME_MARGIN
            if watchdog.remaining() < need:
                raise GuardAborted(
                    f"{source}: pass {run}/{N_PASSES} needs ~{need:.0f}s but only "
                    f"{max(watchdog.remaining(), 0):.0f}s of the wall-clock cap remain; "
                    f"nothing written for {source} (lower --max-sequence-passes)")
        started = watchdog.probe.now()
        scores, raw, proto = score_pass(seqs, run, archive, out_root, gpus, runner)
        pass_seconds.append(watchdog.probe.now() - started)
        protocol = protocol or proto
        per_pass_scores.append(scores)
        per_pass_raw.append(raw)
        print(f"    pass {run}/{N_PASSES} done ({pass_seconds[-1]:.0f}s)", flush=True)
    obs, incomplete = build_observations(
        source, per_pass_scores, per_pass_raw, field=field, our=our,
        refresh_epoch=refresh_epoch, scorer_protocol=protocol)
    if incomplete:
        _loud("HF_REFRESH_INCOMPLETE_METRICS",
              f"{source}: {len(incomplete)} of {len(seqs)} sequences lack a finite "
              f"score or one of the ten metrics in some pass; not written",
              source=source, n=len(incomplete), example=incomplete[0])
    if not obs:
        return 0
    written = writer(obs)
    print(f"  {source}: imported {written}", flush=True)
    return written


def main(argv: list[str] | None = None, *, probe=None,
         writer: Callable[[list[dict]], int] = _default_writer) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default=None,
                    help="GPU ids to use; default = every GPU nvidia-smi lists")
    ap.add_argument("--epoch", type=int, default=None,
                    help="explicit epoch; default = latest completed")
    ap.add_argument("--allow-unconsumed", action="store_true",
                    help="run even though no production code reads this pool "
                         "(see the disable note below)")
    ap.add_argument("--max-wall-hours", type=float, default=DEFAULT_MAX_WALL_HOURS,
                    help=f"hard wall-clock cap for this invocation (ceiling "
                         f"{HARD_MAX_WALL_HOURS:g})")
    ap.add_argument("--max-sequence-passes", type=int, default=DEFAULT_MAX_SEQUENCE_PASSES,
                    help=f"sequences x {N_PASSES} passes per invocation; the rest "
                         f"is deferred (ceiling {HARD_MAX_SEQUENCE_PASSES})")
    args = ap.parse_args(argv)

    # DISABLED BY DEFAULT (2026-10-09, ARCHITECTURE_REDESIGN P3). Nothing in
    # production reads hf_field_20x_rank_sum or hf_field_sum10_20x_rank_sum
    # (re-checked: repo, master /root, dashboard /root, Pi /opt). The sum-of-10
    # label rename made already_scored() return 0 and started a ~10 h,
    # ~80 GPU-hour re-fold of 2,908 sequences, 99% of which the dashboard had
    # already 10x-scored with the identical scorer, competing with the
    # producer on all 8 GPUs; the operator killed it. Do not restart this on
    # the same terms. It should come back only as a consumer-backed,
    # read-only aggregator over stored raw per-pass metrics (R1/R2), never as
    # a second GPU scorer. --allow-unconsumed or HF_FIELD_REFRESH_ENABLED=1
    # overrides, deliberately.
    if not (args.allow_unconsumed or os.environ.get("HF_FIELD_REFRESH_ENABLED") == "1"):
        print("HF_FIELD_REFRESH_DISABLED: no production consumer reads this pool; "
              "see docs/ARCHITECTURE_REDESIGN.md P3. Pass --allow-unconsumed "
              "or set HF_FIELD_REFRESH_ENABLED=1 to override.", flush=True)
        return EXIT_DISABLED

    probe = probe or SystemProbe()
    started = probe.now()
    try:
        if not 0 < args.max_wall_hours <= HARD_MAX_WALL_HOURS:
            raise GuardRefused(f"--max-wall-hours {args.max_wall_hours:g} outside "
                               f"(0, {HARD_MAX_WALL_HOURS:g}]")
        if not N_PASSES <= args.max_sequence_passes <= HARD_MAX_SEQUENCE_PASSES:
            raise GuardRefused(f"--max-sequence-passes {args.max_sequence_passes} outside "
                               f"[{N_PASSES}, {HARD_MAX_SEQUENCE_PASSES}]")
        deadline = started + args.max_wall_hours * 3600.0
        lock = acquire_lock()  # held until the process exits
        gpus = gpu_ids(args.gpus, probe)
        assert_safe_window(gpus, probe, self_pid=os.getpid())
    except GuardRefused as exc:
        _loud("HF_REFRESH_GUARD_REFUSED", str(exc))
        return EXIT_REFUSED

    sys.path.insert(0, "/root")
    try:
        assert_label_open()
    except Exception as exc:  # LabelViolation, or a registry that cannot load
        _loud("HF_REFRESH_LABEL_REFUSED", f"{type(exc).__name__}: {exc}")
        return EXIT_REFUSED

    from nanobody_opt import db

    archive = pd.read_csv(ARCHIVE)
    archive["sequence"] = archive["sequence"].astype(str).str.strip().str.upper()
    field = field_fingerprint(ARCHIVE, archive)

    if args.epoch is None:
        # Walk back from the highest epoch the DB has ever seen (the producer's
        # current epoch) to the newest COMPLETED epoch on the leaderboard API.
        with db.connect() as conn:
            max_epoch = conn.execute(
                "SELECT COALESCE(MAX(epoch), 0) FROM observations"
            ).fetchone()[0]
        args.epoch = latest_completed_epoch(int(max_epoch) + 2)
        if args.epoch is None:
            print("no completed epoch found", flush=True)
            return 1
    print(f"epoch={args.epoch}", flush=True)

    with db.connect() as conn:
        scored = already_scored(conn)
        our = fetch_our_top50(conn)
        db_rivals = fetch_db_rivals(conn)

    leaderboard_rivals = fetch_rivals(args.epoch)
    # Union of the latest leaderboard rivals and every rival we have ever
    # scored, so a sequence skipped by an earlier backfill snapshot still gets
    # caught and 20x-scored here.
    rivals = leaderboard_rivals | db_rivals
    hf = fetch_hf_top200(archive)

    print(f"pool already scored={len(scored)}; new: ours={len(our - scored)} "
          f"rivals={len(rivals - scored)} hf={len(hf - scored)}", flush=True)
    planned, deferred = plan_budget(
        [("our_top50_20x", sorted(our - scored)),
         ("public_multipass_rivals", sorted(rivals - scored - our)),
         ("hf_archive_top200", sorted(hf - scored - our - rivals))],
        args.max_sequence_passes)
    print("HF_REFRESH_BUDGET max_sequence_passes=%d max_wall_hours=%g planned=%s deferred=%s"
          % (args.max_sequence_passes, args.max_wall_hours,
             {s: len(q) for s, q in planned}, deferred), flush=True)

    # Never route our folds through the producer's shared warm scorer daemon:
    # the work would run inside a process the watchdog cannot see or kill.
    env = dict(os.environ, NANOBODY_SCORER_DAEMON="0")
    watchdog = Watchdog(probe, deadline=deadline, gpus=gpus,
                        self_pid=os.getpid()).start()
    total = 0
    try:
        for source, seqs in planned:
            total += score_and_import(
                source, seqs, archive, ",".join(map(str, gpus)), watchdog=watchdog,
                field=field, our=our, refresh_epoch=args.epoch, env=env, writer=writer)
    except GuardAborted as exc:
        _loud("HF_REFRESH_ABORTED", f"{exc} (imported={total} before the abort)",
              imported=total)
        return EXIT_ABORTED
    finally:
        watchdog.stop()
        lock.close()
    print(f"REFRESH_DONE imported={total} deferred={sum(deferred.values())} "
          f"wall={probe.now() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

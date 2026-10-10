#!/usr/bin/env python3
"""Per-epoch 20x depth top-up of rival / HF-archive passes into ``multipass_sum10_pass``.

What it scores. Each run rebuilds three candidate sets:

  public_multipass_rivals  every rival sequence on this epoch's leaderboard,
                           plus every rival ever stored (cohort rival)
  hf_archive_top200        rank_binders top-200 over the raw HF archive
                           (nanobody_opt.hf_topup, the same copy the live
                           reader uses)
  our_top50_20x            our top-50 (only with --include-ours; see below)

For each sequence it counts the passes already stored under
``multipass_sum10_pass`` with raw metrics, from any writer
(``hf_topup.stored_pass_counts``). It scores only the passes MISSING to a
depth of N_PASSES. "20x" is a depth target, not a second pipeline
(ARCHITECTURE_REDESIGN R1), so no relabel can ever trigger a full re-fold
again.

What it writes (2026-10-10). One row per sequence per pass, under exactly
the contract that ``tools/multipass_all_rivals.py`` (dashboard) and
``tools/migrate_hf_field_20x.py`` already write
(``observation_labels.LABELS['multipass_sum10_pass']``, strict):

  score_kind    multipass_sum10_pass
  source        public_multipass_rivals   (the only registered source)
  score         ONE pass's sum-of-10 vs the full HF archive + that pass's batch
                (archive duplicates dropped, keep the fresh pass: same as
                multipass_all_rivals.rank_pass_payload)
  metrics_json  the RAW per-pass BoltzGen metrics (all ten metrics; this was
                '{}' on every row the old version wrote)
  provenance    writer=hf_field_refresh, formula=sum10, cohort, pass_index,
                passes, field / field_rows / field_sha256, epoch_basis,
                scorer_protocol, hf20x_source, refresh_epoch
  id            hf20x_pass:<source>:epoch<E>:<seq>:pass<k>  (the migration's
                scheme; k = the sequence's depth index, so ids never collide)
  epoch         the sequence's latest stored public_multipass_rivals epoch
                <= this run's epoch, else this run's epoch. This is the same
                rule the migration uses for re-scored sequences: an old rival
                must not be pulled into the live top-up reader's 16-epoch
                "current rival field" window just because we topped it up
                today.

Cohort (``ours`` / ``rival`` / ``hf_top200``) follows
``migrate_hf_field_20x.ours_set``: a sequence is ours if it is in our top-50,
carries cohort=ours on any stored pass row, or appears under our coldkey on
any leaderboard (or in the Pi's intended drop) since the first multipass
epoch. cohort=ours is NOT written by default, for the reason the migration
gives: the top-up reader drops it, and MutationNetwork still pools every
score_kind into a node's live score. ``--include-ours`` overrides this.

This store FEEDS THE LIVE PARENT POOL: ``build_top50_index`` ->
``hf_topup.ranked_topup`` ranks every non-ours pass under this label each
producer epoch. That is the point (one store), and it is why the guard below
exists.

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
    producer process appears or the deadline passes;
  * the label registry must accept the rows before any GPU time is spent.

A refusal or abort is LOUD: a ``HF_REFRESH_GUARD_REFUSED`` /
``HF_REFRESH_LABEL_REFUSED`` / ``HF_REFRESH_ABORTED`` marker on stdout and
stderr, a ``pipeline_failures.jsonl`` entry, and a distinct exit code (3
refused, 4 aborted). It never degrades to running alongside the producer;
there is no busy-but-stale override.

Writes go through ``mutation_network.update_observation_store``: one batched
call per source, with a constant number of round-trips per 1000 rows.

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
# 2026-10-10: per-pass rows under the ONE current multipass label. The old
# per-sequence mean labels (hf_field_20x_rank_sum, hf_field_sum10_20x_rank_sum)
# are closed; their rows were relabelled to legacy_* by migrate_hf_field_20x.
SCORE_KIND = "multipass_sum10_pass"
SOURCE = "public_multipass_rivals"
#: Target depth: stored passes (with raw metrics) per sequence.
N_PASSES = 20
OUR_COLDKEY = "5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ"
API = "https://compound-api.metanova-labs.ai"

ARCHIVE = "/root/P20809_nanobodies.csv"
CONFIG = "/root/config/config.yaml"
SCORER = "/root/nanobody_opt/score_boltzgen_fast_adapter.py"
OUT_ROOT = Path("/root/hf_field_refresh")
INTENDED_DIR = Path("/root/our_intended_sequences")
#: provenance.writer (registered for SCORE_KIND in observation_labels).
WRITER = "hf_field_refresh"
FORMULA_ID = "sum10"
#: provenance.cohort per refresh source when the sequence is not ours
#: (= migrate_hf_field_20x.SOURCE_COHORT).
SOURCE_COHORT = {"public_multipass_rivals": "rival", "hf_archive_top200": "hf_top200",
                 "our_top50_20x": "ours"}
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
#: MISSING sequence-passes topped up per invocation; the rest is deferred
#: to the next run (the refresh is idempotent). 2000 = e.g. 100 sequences from 0 to 20, ~2.7
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


def fetch_hf_top200(archive: pd.DataFrame) -> set[str]:
    # One copy of this ranking: nanobody_opt.hf_topup (R1), which now feeds
    # the live parent pool through the dashboard pass scorer.
    from nanobody_opt.hf_topup import hf_archive_top
    return set(hf_archive_top(archive, 200))


def fetch_our_top50(conn) -> set[str]:
    rows = conn.execute(OUR_TOP50_SQL, (TARGET,)).fetchall()
    return {str(r[0]).strip().upper() for r in rows}




def fetch_leaderboard(epoch: int) -> list[tuple[str, str]] | None:
    """``[(coldkey, sequence)]`` for every P20809 nanobody, deduped; ``None``
    when the epoch is not available (same semantics as
    ``multipass_all_rivals.fetch_leaderboard``: None != [])."""
    req = urllib.request.Request(
        f"{API}/api/competitions/leaderboard/{epoch}/nanobodies",
        headers={"User-Agent": "nova-hf-field-refresh/1"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
    except Exception:
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for submission in data:
        coldkey = str(submission.get("coldkey") or "").strip()
        if not coldkey:
            continue
        for nb in submission.get("nanobodies") or []:
            if nb.get("protein_name") != TARGET:
                continue
            seq = str(nb.get("sequence") or "").strip().upper()
            if seq and seq not in seen:
                seen.add(seq)
                pairs.append((coldkey, seq))
    return pairs


def fetch_rivals(epoch: int) -> set[str]:
    return {seq for ck, seq in fetch_leaderboard(epoch) or [] if ck != OUR_COLDKEY}


def fetch_our_intended(epoch: int) -> set[str]:
    """The Pi's per-epoch intended drop (= multipass_all_rivals.fetch_our_intended)."""
    path = INTENDED_DIR / f"{epoch}.json"
    try:
        rows = json.loads(path.read_text()) if path.is_file() else []
    except (OSError, ValueError):
        return set()
    return {str((r or {}).get("sequence") or "").strip().upper()
            for r in (rows if isinstance(rows, list) else [])} - {""}


def fetch_db_rivals(conn) -> set[str]:
    """Every sequence ever stored as a RIVAL pass: cohort rival, or no cohort
    (rows from before cohorts existed). Not hf_top200 / ours: this script
    writes this source too, and must not read its own output back as
    'rivals' (ARCHITECTURE_REDESIGN R3)."""
    rows = conn.execute(
        """
        SELECT DISTINCT n.sequence
        FROM observations o JOIN nodes n ON o.node_id = n.id
        WHERE n.target = %s
          AND o.source IN ('public_multipass_rivals', 'slave_public_multipass_rivals')
          AND COALESCE(JSON_UNQUOTE(JSON_EXTRACT(o.provenance_json, '$.cohort')),
                       'rival') = 'rival'
        """,
        (TARGET,),
    ).fetchall()
    return {str(r[0]).strip().upper() for r in rows}


def ours_set(conn, extra: set[str], *, cache_path: Path | None = None
             ) -> tuple[set[str], dict]:
    """Our sequences, by the rule ``migrate_hf_field_20x.ours_set`` uses:
    ``extra`` (our top-50), cohort=ours on any stored pass row, our coldkey on
    any published leaderboard since the first multipass epoch, and the Pi's
    intended drops. Published leaderboards are immutable, so the per-epoch
    "ours" lists are cached in ``cache_path``."""
    ours = set(extra)
    evidence: dict = {"our_top50": len(extra)}
    rows = conn.execute(
        "SELECT DISTINCT n.sequence FROM observations o JOIN nodes n ON n.id=o.node_id "
        "WHERE o.source=%s AND JSON_UNQUOTE(JSON_EXTRACT(o.provenance_json,'$.cohort'))='ours'",
        (SOURCE,)).fetchall()
    db_ours = {str(r[0]).strip().upper() for r in rows}
    evidence["db_cohort_ours"] = len(db_ours)
    ours |= db_ours
    lo, hi = conn.execute("SELECT MIN(epoch), MAX(epoch) FROM observations "
                          "WHERE source=%s", (SOURCE,)).fetchone()
    cache: dict[str, list[str]] = {}
    if cache_path is not None:
        try:
            cache = json.loads(cache_path.read_text())
        except (OSError, ValueError):
            cache = {}
    board_ours: set[str] = set()
    missing = 0
    if lo is not None and hi is not None:
        for epoch in range(int(lo), int(hi) + 1):
            if str(epoch) in cache:
                board_ours |= set(cache[str(epoch)])
            else:
                board = fetch_leaderboard(epoch)
                if board is None:
                    missing += 1
                else:
                    mine = sorted({seq for ck, seq in board if ck == OUR_COLDKEY})
                    cache[str(epoch)] = mine
                    board_ours |= set(mine)
            board_ours |= fetch_our_intended(epoch)
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, sort_keys=True))
        except OSError:
            pass
    evidence.update({"leaderboard_or_intended_ours": len(board_ours),
                     "leaderboard_epochs": f"{lo}-{hi}", "leaderboard_unavailable": missing})
    ours |= board_ours
    evidence["total"] = len(ours)
    return ours, evidence


def latest_epochs(conn, sequences, run_epoch: int) -> dict[str, tuple[int, str]]:
    """``{seq: (epoch, basis)}``: the sequence's latest stored
    public_multipass_rivals epoch <= ``run_epoch``, else ``run_epoch``
    (= migrate_hf_field_20x.rescore_epochs' DB rule)."""
    out = {s: (int(run_epoch), "run epoch (no prior epoch)") for s in sequences}
    ordered = sorted(out)
    for i in range(0, len(ordered), 300):
        chunk = ordered[i:i + 300]
        rows = conn.execute(
            "SELECT n.sequence, MAX(o.epoch) FROM observations o JOIN nodes n ON n.id=o.node_id "
            "WHERE o.source=%s AND o.epoch IS NOT NULL AND o.epoch<=%s "
            f"AND n.sequence IN ({','.join(['%s'] * len(chunk))}) GROUP BY n.sequence",
            (SOURCE, int(run_epoch), *chunk)).fetchall()
        for seq, epoch in rows:
            seq = str(seq).strip().upper()
            if seq in out and epoch is not None:
                out[seq] = (int(epoch), f"latest {SOURCE} row epoch")
    return out


def stored_counts(conn, sequences) -> dict[str, int]:
    """Stored passes WITH raw metrics per sequence: the live reader's own count."""
    from nanobody_opt.hf_topup import stored_pass_counts
    return stored_pass_counts(conn, sorted(sequences), target=TARGET)


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


def plan_budget(batches: list[tuple[str, list[str]]], counts: dict[str, int],
                max_sequence_passes: int
                ) -> tuple[list[tuple[str, dict[str, int]]], dict[str, int], dict[str, int]]:
    """``(planned, deferred, deep)``. ``planned`` is ``[(source, {seq: missing
    passes})]``: whole sequences, in source order, while the missing passes
    fit ``max_sequence_passes``. ``deferred`` counts the sequences left for a
    later run, and ``deep`` the ones already at N_PASSES (nothing to do)."""
    left = max_sequence_passes
    planned, deferred, deep = [], {}, {}
    for source, seqs in batches:
        plan: dict[str, int] = {}
        deferred[source] = deep[source] = 0
        for seq in seqs:
            missing = N_PASSES - int(counts.get(seq, 0))
            if missing <= 0:
                deep[source] += 1
            elif missing <= left:
                plan[seq] = missing
                left -= missing
            else:
                deferred[source] += 1
        planned.append((source, plan))
    return planned, deferred, deep


def assert_label_open() -> None:
    """Before any GPU time: the registry must accept the rows we will write and
    ``sum_of_ten`` must still compute the registered formula. A closed label or
    an unregistered writer is refused here, not after a 40-minute run."""
    from nanobody_opt import observation_labels
    observation_labels.assert_formula(SCORE_KIND, FORMULA_ID, sum_of_ten)
    observation_labels.check_observations([
        {"observation_id": "hf20x_pass:preflight", "sequence": "A", "target": TARGET,
         "source": SOURCE, "score_kind": SCORE_KIND, "score": 1.0,
         "writer": WRITER, "formula": FORMULA_ID, "cohort": cohort}
        for cohort in ("rival", "hf_top200")])


def gpu_ids(arg: str | None, probe) -> list[int]:
    if arg:
        return [int(x) for x in arg.split(",") if x.strip()]
    return sorted(g["index"] for g in probe.gpu_state())


# ===========================================================================
# Scoring and the observation rows
# ===========================================================================
METRICS_NEEDED = sorted(SUM_OF_TEN_DIRS)


def raw_metrics_from_payload(payload: dict, wanted: set[str] | None = None
                             ) -> dict[str, dict[str, float]]:
    """``{sequence: {metric: value}}``: every finite numeric metric of one pass
    (wave 0, TARGET), for the requested sequences only. Formula-independent
    (= multipass_all_rivals.raw_metrics_from_exact_all, restricted to the
    batch we asked for)."""
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
    """Sum-of-10 of each scored sequence vs the archive + this pass's batch,
    with archive duplicates dropped (the fresh pass wins) -- exactly
    ``multipass_all_rivals.rank_pass_payload``, so every row under the label
    is ranked the same way (tests/test_hf_field_refresh.py checks this)."""
    rows = {s: {k: v for k, v in m.items() if k in METRICS_NEEDED} for s, m in raw.items()}
    rival = pd.DataFrame(rows).T
    archive_s = archive.set_index("sequence")[METRICS_NEEDED]
    field = pd.concat([archive_s, rival])
    field = field[~field.index.duplicated(keep="last")]
    scores = sum_of_ten(field)
    return {seq: float(scores[seq]) for seq in rival.index if seq in scores.index}


def structures_from_payload(payload: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for seq, by_target in (payload.get("structures") or {}).items():
        rec = (by_target or {}).get(TARGET, {})
        if rec:
            out[str(seq).strip().upper()] = {
                "path": rec.get("path"), "writer_path": rec.get("writer_path"),
                "sample_selection": rec.get("sample_selection")}
    return out


def score_pass(sequences: list[str], run: int, archive: pd.DataFrame,
               out_root: Path, gpus: str, runner: Callable[[list[str], Path], None]
               ) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, dict],
                          str | None]:
    """One scorer pass -> ({seq: sum10}, {seq: raw metrics}, {seq: cif ref},
    scorer_protocol)."""
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
    return (rank_pass(raw, archive), raw, structures_from_payload(payload),
            payload.get("scorer_protocol"))


def field_fingerprint(path: str, archive: pd.DataFrame) -> dict:
    """= multipass_all_rivals.field_fingerprint (same keys, same values)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"field": "hf_archive_full", "field_path": str(path),
            "field_rows": int(len(archive)), "field_sha256": digest.hexdigest()}


def pass_observations(source: str, *, pass_of: dict[str, int], ranks: dict[str, float],
                      raw: dict[str, dict[str, float]], structs: dict[str, dict],
                      field: dict, ours: set[str], epochs: dict[str, tuple[int, str]],
                      refresh_epoch: int, scorer_protocol: str | None,
                      ) -> tuple[list[dict], list[str]]:
    """One ``multipass_sum10_pass`` row per sequence of one pass.

    ``pass_of[seq]`` is the sequence's depth index for this pass (stored + r
    + 1). Returns ``(rows, incomplete)``: a sequence without a finite score
    or with one of the ten metrics missing is reported, not written.
    """
    rows, incomplete = [], []
    for seq in sorted(pass_of):
        score, metrics = ranks.get(seq), raw.get(seq)
        if (score is None or not math.isfinite(score) or metrics is None
                or any(m not in metrics for m in SUM_OF_TEN_DIRS)):
            incomplete.append(seq)
            continue
        epoch, basis = epochs.get(seq, (int(refresh_epoch), "run epoch (no prior epoch)"))
        k = pass_of[seq]
        rows.append({
            "observation_id": f"hf20x_pass:{source}:epoch{refresh_epoch}:{seq}:pass{k}",
            "sequence": seq,
            "target": TARGET,
            "epoch": epoch,
            "source": SOURCE,
            "score_kind": SCORE_KIND,
            "score": float(score),
            "observation_count": 1,
            "metrics": dict(metrics),
            "structure": structs.get(seq),
            # Every key below lands in provenance_json.
            "writer": WRITER,
            "formula": FORMULA_ID,
            "cohort": "ours" if seq in ours else SOURCE_COHORT[source],
            "pass_index": k,
            "passes": N_PASSES,
            **field,
            "epoch_basis": basis,
            "scorer_protocol": scorer_protocol,
            "hf20x_source": source,
            "refresh_epoch": int(refresh_epoch),
        })
    return rows, incomplete


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
    """The batched store writer (constant round-trips per 1000-row chunk)."""
    from nanobody_opt.mutation_network import update_observation_store
    return _retry_db(update_observation_store, Path("/root/.sn68_mysql_creds"),
                     obs, commit_every=1000, commit_interval=5.0)


def score_and_import(source: str, plan: dict[str, int], archive: pd.DataFrame, gpus: str, *,
                     watchdog: Watchdog, field: dict, ours: set[str],
                     counts: dict[str, int], epochs: dict[str, tuple[int, str]],
                     refresh_epoch: int, env: dict[str, str],
                     writer: Callable[[list[dict]], int] = _default_writer,
                     out_root_base: Path | None = None) -> int:
    """Top ``plan`` ({seq: missing passes}) up to depth and write the passes.

    The rows of every COMPLETED pass are written in one batched call at the
    end -- also when the watchdog aborts the run (no GPU is involved in the
    write), so finished GPU work is never thrown away; the abort is then
    re-raised. Depth indices continue from the stored count, so the next run
    tops up exactly what is still missing.
    """
    if not plan:
        print(f"  {source}: nothing to top up", flush=True)
        return 0
    from nanobody_opt.hf_topup import pass_batches

    batches = pass_batches(plan)
    print(f"  {source}: topping up {len(plan)} sequences "
          f"({sum(plan.values())} sequence-passes, {len(batches)} scorer passes) "
          f"to depth {N_PASSES}", flush=True)
    out_root = (out_root_base or OUT_ROOT) / source / f"epoch{refresh_epoch}"
    out_root.mkdir(parents=True, exist_ok=True)

    def runner(cmd: list[str], log_path: Path) -> None:
        run_guarded(cmd, watchdog, log_path=log_path, env=env)

    rows: list[dict] = []
    incomplete: set[str] = set()
    pass_seconds: list[float] = []
    aborted: GuardAborted | None = None
    try:
        for r, batch in enumerate(batches):
            run = r + 1
            if pass_seconds:
                need = max(pass_seconds) * PASS_TIME_MARGIN
                if watchdog.remaining() < need:
                    raise GuardAborted(
                        f"{source}: scorer pass {run}/{len(batches)} needs ~{need:.0f}s but "
                        f"only {max(watchdog.remaining(), 0):.0f}s of the wall-clock cap "
                        f"remain (lower --max-sequence-passes)")
            started = watchdog.probe.now()
            ranks, raw, structs, proto = score_pass(batch, run, archive, out_root, gpus,
                                                    runner)
            pass_seconds.append(watchdog.probe.now() - started)
            new, bad = pass_observations(
                source, pass_of={s: int(counts.get(s, 0)) + run for s in batch},
                ranks=ranks, raw=raw, structs=structs, field=field, ours=ours,
                epochs=epochs, refresh_epoch=refresh_epoch, scorer_protocol=proto)
            rows += new
            incomplete |= set(bad)
            print(f"    pass {run}/{len(batches)} done: {len(batch)} sequences, "
                  f"{pass_seconds[-1]:.0f}s", flush=True)
    except GuardAborted as exc:
        aborted = exc
    if incomplete:
        _loud("HF_REFRESH_INCOMPLETE_METRICS",
              f"{source}: {len(incomplete)} of {len(plan)} sequences lack a finite score "
              f"or one of the ten metrics in some pass; those passes were not written",
              source=source, n=len(incomplete), example=sorted(incomplete)[0])
    written = writer(rows) if rows else 0
    print(f"  {source}: imported {written} pass rows"
          + (" (before the abort)" if aborted else ""), flush=True)
    if aborted is not None:
        raise GuardAborted(f"{aborted} [imported {written} completed pass rows]")
    return written


def main(argv: list[str] | None = None, *, probe=None,
         writer: Callable[[list[dict]], int] = _default_writer) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default=None,
                    help="GPU ids to use; default = every GPU nvidia-smi lists")
    ap.add_argument("--epoch", type=int, default=None,
                    help="explicit epoch; default = latest completed")
    ap.add_argument("--allow-unconsumed", action="store_true",
                    help="deliberate override of the P3 disable gate (see main())")
    ap.add_argument("--include-ours", action="store_true",
                    help="also top up and write cohort=ours passes (default: no; "
                         "MutationNetwork would pool them into live node scores)")
    ap.add_argument("--max-wall-hours", type=float, default=DEFAULT_MAX_WALL_HOURS,
                    help=f"hard wall-clock cap for this invocation (ceiling "
                         f"{HARD_MAX_WALL_HOURS:g})")
    ap.add_argument("--max-sequence-passes", type=int, default=DEFAULT_MAX_SEQUENCE_PASSES,
                    help=f"missing sequence-passes topped up per invocation; the rest "
                         f"is deferred (ceiling {HARD_MAX_SEQUENCE_PASSES})")
    args = ap.parse_args(argv)

    # DISABLED BY DEFAULT (2026-10-09, ARCHITECTURE_REDESIGN P3): the
    # 2026-10-09 run re-folded ~2,900 sequences (~80 GPU-h) on master's 8 GPUs
    # against the live producer. Re-enabling is a deliberate act:
    # --allow-unconsumed or HF_FIELD_REFRESH_ENABLED=1. Since 2026-10-10 the
    # rows land in multipass_sum10_pass, which the live top-up reader DOES
    # consume (see the module docstring).
    if not (args.allow_unconsumed or os.environ.get("HF_FIELD_REFRESH_ENABLED") == "1"):
        print("HF_FIELD_REFRESH_DISABLED: see docs/ARCHITECTURE_REDESIGN.md P3. Pass "
              "--allow-unconsumed or set HF_FIELD_REFRESH_ENABLED=1 to run.", flush=True)
        return EXIT_DISABLED

    probe = probe or SystemProbe()
    started = probe.now()
    try:
        if not 0 < args.max_wall_hours <= HARD_MAX_WALL_HOURS:
            raise GuardRefused(f"--max-wall-hours {args.max_wall_hours:g} outside "
                               f"(0, {HARD_MAX_WALL_HOURS:g}]")
        if not 1 <= args.max_sequence_passes <= HARD_MAX_SEQUENCE_PASSES:
            raise GuardRefused(f"--max-sequence-passes {args.max_sequence_passes} outside "
                               f"[1, {HARD_MAX_SEQUENCE_PASSES}]")
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
        our_top50 = fetch_our_top50(conn)
        db_rivals = fetch_db_rivals(conn)
        ours, ours_evidence = ours_set(conn, our_top50,
                                       cache_path=OUT_ROOT / "ours_by_epoch.json")
    leaderboard_rivals = fetch_rivals(args.epoch)
    hf = fetch_hf_top200(archive)
    rivals = (leaderboard_rivals | db_rivals) - ours
    hf_new = hf - ours - rivals
    candidates = [("public_multipass_rivals", sorted(rivals)),
                  ("hf_archive_top200", sorted(hf_new))]
    if args.include_ours:
        candidates.insert(0, ("our_top50_20x", sorted(our_top50)))
    with db.connect() as conn:
        counts = stored_counts(conn, {s for _, q in candidates for s in q})
    planned, deferred, deep = plan_budget(candidates, counts, args.max_sequence_passes)
    with db.connect() as conn:
        epochs = latest_epochs(conn, {s for _, p in planned for s in p}, args.epoch)
    print(f"ours={ours_evidence} candidates={ {s: len(q) for s, q in candidates} }",
          flush=True)
    print("HF_REFRESH_BUDGET max_sequence_passes=%d max_wall_hours=%g planned=%s "
          "sequence_passes=%d already_at_depth=%s deferred=%s"
          % (args.max_sequence_passes, args.max_wall_hours,
             {s: len(p) for s, p in planned},
             sum(sum(p.values()) for _, p in planned), deep, deferred), flush=True)

    # Never route our folds through the producer's shared warm scorer daemon:
    # the work would run inside a process the watchdog cannot see or kill.
    env = dict(os.environ, NANOBODY_SCORER_DAEMON="0")
    watchdog = Watchdog(probe, deadline=deadline, gpus=gpus,
                        self_pid=os.getpid()).start()
    total = 0
    try:
        for source, plan in planned:
            total += score_and_import(
                source, plan, archive, ",".join(map(str, gpus)), watchdog=watchdog,
                field=field, ours=ours,
                counts=counts, epochs=epochs, refresh_epoch=args.epoch, env=env,
                writer=writer)
    except GuardAborted as exc:
        _loud("HF_REFRESH_ABORTED", f"{exc} (imported={total} before this source)",
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

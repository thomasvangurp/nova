#!/usr/bin/env python3
"""Top up the sequences we are about to commit to 20 observations on the SLAVE box's idle GPUs.

This is the slave-side twin of ``topup_publishable.py``.  It runs on MASTER (the only writer of the
SQLite observation database), ships the work to the slave's four GPUs, pulls the raw ``exact_all.json``
rounds back and imports them with the normal importer, so every observation carries the label the
publisher counts (``local_exact_raw``).  The slave never opens the database.

Why a second worker.  The master top-up shares the race's eight GPUs, runs only inside offsets
200-285 (two or three passes), ranks fresh (~95 s) before every pass and looks at the first 36
candidates of the ranking.
On epochs 25694-25699 the under-confirmed sequences in the committed fleet were NOT in that top 36:
they sat at pool rank 106-110 (or 37), i.e. they were the sequences the publisher had already
published earlier in the epoch (the strict pool is the top 100 of a fresh ranking PLUS everything
already published) and they re-entered the qualified top ten with 8-13 observations.  So this worker
tops up, in priority order,

  1. every sequence of the current artifact (submissions and sequence_pool) and of this epoch's
     checkpoints                                    -- what is, or can be, committed;
  2. the whole qualified ranking                    -- the strict-gated candidates;
  3. the first POOL_K (100, the publisher's strict pool size) of the freshest provisional-mode ranking.

It needs no ranking of its own: counts come from a read-only query of the database (0.15 s for 150
candidates), the order from the newest ranking file the publisher or the master top-up already wrote.
It runs continuously from epoch offset 20 and repeats while anything in that set is short: topping
up the leaders lets their thinly-observed neighbours rise into the leaders' places (a top-up
regresses the topped-up candidates), so the next layer is picked up by the next pass.  Imports are
batched one per pass because each import makes the publisher run a whole strict cycle.  The last
useful import time depends on the crew size (see FINAL_IMPORT_OFFSET below).

The measurement is the master's measurement: the same ``score_boltzgen_waves`` ->
``score_boltzgen_fast_adapter`` -> warm-daemon path ``_run_screen`` uses, on the slave's copy of the
same pinned scorer, GPU pairs and batch size 2.  The only thing the slave's own scorer settings lack
is the master's ``fold_sampling_steps`` (the master runs 35, the replay on the slave still runs 50), so
the measurement keys are copied from the MASTER's settings file into a private settings file
(``NANOBODY_FAST_SCORER_CONFIG``) used by this worker only; the replay's settings are never touched.
Before scoring, the scoring code on both boxes is hashed; any difference skips the pass.

Interplay.
  * The slave's replay jobs hold the master-side flock ``slave_gpu.lock`` for whole jobs.  This worker
    does NOT take it: it talks to the same warm scorer daemon, whose per-GPU locks interleave a round
    between two replay waves (about one wave of delay, measured and logged), and it never pauses,
    kills or restarts anything on the slave.
  * It does not write the master's ``topup_running.json`` marker (that tells the race that ITS GPUs are
    shared; ours are not).  If the master top-up is mid-pass it waits for it and recounts, so the two
    do not score the same candidate twice.
  * Rounds are staged outside ``/root/live`` and moved into
    ``/root/live/epoch<N>/topup_slave/round_*`` with one atomic rename: the publisher imports every
    ``exact_all.json`` it finds under ``live/epoch*`` and must never see a half-copied file.
  * Any failure only skips the pass; nothing here can touch the producer, publisher, daemon or Pi.

Usage:  topup_slave.py once [--dry-run] [--force] | loop | preflight | rounds ...
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import queue
import random
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

TARGET = "P20809"
TARGET_OBS = 20
MAX_ROUNDS = 12                    # 8 -> 20 observations
SEQUENCE_LENGTH = 127              # the fast scorer only supports length 127
POOL_K = int(os.environ.get("TOPUP_SLAVE_POOL_K", "130"))   # publisher --strict-pool-size is 100; +30 for the churn a top-up itself causes
QUALIFIED_K = 60
MAX_OBS_PER_PASS = int(os.environ.get("TOPUP_SLAVE_MAX_OBS", "600"))
EPOCH_LENGTH = 361
BLOCK_SECONDS = 12.0
PREDICT_BATCH = 2                  # the producer's --screen-predict-batch-size
MIN_FREE_DISK_GB = 25.0            # the producer refuses GPU work below 21.5 GB

# Epoch offsets (blocks).  An import is useful only if the publisher can finish a whole strict cycle on
# it BEFORE the Pi's final pull of the artifact.  Measured (epochs 25693-25699): the Pi opens its window
# at offset 226-228 and pulls the artifact at 294-326 (loop.log: late_commit_ready; its wait budget is
# computed from the window-open block, then held to deadline-120 s, so the pull lands at about
# 335 - 5*crew + x blocks, x = 5-20 for the release step before it); the publisher's last strict cycle
# takes about 25 blocks (pool ranking written ~8 blocks after it starts, the qualified artifact ~17
# blocks later).  Hence the cut-off below, which depends on the crew size.
FIRST_OFFSET = int(os.environ.get("TOPUP_SLAVE_FIRST_OFFSET", "20"))
FINAL_IMPORT_OFFSET = int(os.environ.get("TOPUP_SLAVE_FINAL_OFFSET", "285"))   # upper bound of the cut-off
FINAL_IMPORT_FLOOR = 240           # never give up earlier than this, whatever the crew
PI_PULL_BASE = 335                 # pull offset = PI_PULL_BASE - PI_BLOCKS_PER_UID*crew + x
PI_BLOCKS_PER_UID = 5
PI_PULL_X_MIN = 5                  # the smallest x seen: be early, not late
CYCLE_BLOCKS = 30                  # publisher: pick up the import (<= 45 s) + one strict cycle (~25 blocks)
DEFAULT_CREW = 10                  # when no artifact says how many uids we commit
ROUND_BLOCKS = float(os.environ.get("TOPUP_SLAVE_ROUND_BLOCKS", "10"))   # round + pull + import, under replay contention
IDLE_SLEEP = 20.0
PASS_GAP = 30.0                    # after an import: let the publisher's watch loop notice it
PASS_MAX_SECONDS = float(os.environ.get("TOPUP_SLAVE_PASS_SECONDS", "480"))   # then import what we have
MASTER_WAIT = 240.0
CHAIN_MAX_AGE = 180.0
MAX_PASSES_PER_EPOCH = 24          # backstop: every import makes the publisher run a whole cycle
STALL_PAUSE = 1800.0               # an import that did not raise the counts: stop scoring the same thing

ARTIFACTS = Path("/root/artifacts")
LIVE_ROOT = Path("/root/live")
DB = ARTIFACTS / "nanobody_mutations.sqlite"
#: Rehearsal only (``once --rounds-root/--import-db``): where rounds land and which database the importer
#: writes, while ranking files and counts are still read from the live state.  None = the live ones.
ROUNDS_ROOT: Path | None = None
IMPORT_DB: Path | None = None
CHAIN = ARTIFACTS / "chain_state.json"
LOG = ARTIFACTS / "topup_slave.log"
LOCK = ARTIFACTS / "topup_slave.lock"
ARTIFACT = ARTIFACTS / "nanobody_result.json"
MASTER_MARKER = ARTIFACTS / "topup_running.json"
MASTER_SETTINGS = ARTIFACTS / "nanobody_fast_scorer.json"
NANOBODY_OPT = Path("/root/nanobody_opt")
IMPORT_PYTHON = "/root/.pixi/envs/bgen/bin/python"      # the interpreter topup_publishable.py imports with

SLAVE_HOST = os.environ.get("TOPUP_SLAVE_HOST", "root@ssh2.vast.ai")
SLAVE_PORT = int(os.environ.get("TOPUP_SLAVE_PORT", "28358"))
SLAVE_PYTHON = "/root/bgen/bin/python"
SLAVE_SETTINGS = "/root/topup_scorer_settings.json"
SLAVE_BASE_SETTINGS = "/root/artifacts/nanobody_fast_scorer.json"
SLAVE_SCORER = "/root/nanobody_opt/score_boltzgen_fast_adapter.py"
GROUP_SIZE = int(os.environ.get("TOPUP_SLAVE_GROUP_SIZE", "2"))   # GPUs per round; the master top-up uses pairs

#: the scoring code that has to be byte-identical on both boxes for the measurement to be the same
PARITY_FILES = (
    "score_boltzgen_waves.py", "score_boltzgen_fast_adapter.py", "scorer_daemon.py", "chunk_reuse.py",
    "resident_scorer.py", "diffusion_sample_selection.py",
)
PARITY_EXTRA = ("/root/fastscore/pinned-530f5dcbc54d/fast_score_batch.py",)
#: (master sha256, slave sha256) pairs whose divergence was READ and judged not to change the default-path
#: measurement this worker makes (it never passes --allow-replicates, and rounds are repeated calls, not
#: replicates).  2026-10-17 review of the rival-replay rollout:
#:   * score_boltzgen_fast_adapter.py: the slave adds the --allow-replicates flag and nothing else.
#:   * fast_score_batch.py: the slave has the finished replicate support (per-replicate seed, batch size 1,
#:     record-id keyed structures); the master has the original pinned scorer (sha 530f5dcbc54d, which is
#:     what scorer_sha256 pins).  Every difference sits behind allow_replicates or is equivalent without it.
#:   * scorer_daemon.py: the master adds worker-count shard widening (scorer_workers_per_gpu), the slave
#:     adds the orphan-worker parent watch and the replicate plumbing.  Sharding changes how a round is
#:     split across workers, not what a sequence scores.
#: The master's scorer files must keep matching ``scorer_sha256`` in its settings: the adapter refuses to
#: run otherwise, and an edit that skipped the pin stopped the whole race from 2026-10-16 18:58 EDT.
#: Only these exact pairs are accepted.  A change on either box produces a new pair, which blocks the
#: top-up again until it is reviewed and added here.
REVIEWED_DIVERGENCE = {
    "score_boltzgen_fast_adapter.py": (
        "efe27a946e26f6f76f6292c2cd2fa6f8d4638a64b31281016f5d743036367197",
        "09844a09324b710393d59ed49d41a71a6470b2edd711d2e338fcedf544780c64"),
    "scorer_daemon.py": (
        "1b959001e0c52320a96f402ae030da1be6edfdca2728c24478f83690832347b8",
        "5a0ced01655abc8bc7f2029d568b281f83e850f1452e2403f46971cbd0be7971"),
    "fast_score_batch.py": (
        "530f5dcbc54d4a012571f4f9a7ddb6c016552291606652766d02c91643a7d531",
        "0fccc5d630ef2ea98aad456e76ee65170383a6302f7eb92ac636565d2bfd98ef"),
}
#: settings keys that change the measurement: they follow the MASTER's settings file
MEASUREMENT_KEYS = ("fold_sampling_steps", "fold_diffusion_samples", "design_sampling_steps",
                    "fold_matmul_precision")
#: settings keys that must name the same thing on both boxes (paths may differ, content may not)
IDENTITY_KEYS = ("scorer_sha256",)

REQUIRED_METRICS = (
    "delta_sasa_refolded", "design_iiptm", "design_ptm", "design_to_target_iptm", "interaction_pae",
    "liability_num_violations", "liability_score", "min_design_to_target_pae", "plip_hbonds_refolded",
    "plip_saltbridge_refolded",
)                                  # mirrors nanobody_opt.simulate_round.METRICS (see the test)
SCORER_PROTOCOL = "validator-dual-sample-v1"


# --------------------------------------------------------------------------- logging and clock

def log(**fields: Any) -> None:
    fields["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as handle:
            handle.write(json.dumps(fields, sort_keys=True, default=str) + "\n")
    except OSError:
        pass


def chain_position() -> tuple[int, float, float]:
    """(epoch, offset in blocks, age of the chain file in seconds); the offset runs on between writes."""
    state = json.loads(CHAIN.read_text())
    age = max(0.0, time.time() - CHAIN.stat().st_mtime)
    block = int(state["block"]) + min(age, 40.0) / BLOCK_SECONDS
    return int(state["epoch"]), block % EPOCH_LENGTH, age


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- the slave

class Slave:
    """Everything this worker does on the slave, in one place so tests can replace it."""

    def __init__(self, host: str = SLAVE_HOST, port: int = SLAVE_PORT,
                 run: Callable[..., subprocess.CompletedProcess] | None = None):
        self.host, self.port = host, int(port)
        self._run = run or subprocess.run

    #: one multiplexed connection: every call otherwise pays the vast.ai proxy's handshake (~2-3 s)
    MUX = ["-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/topup-slave-%C", "-o", "ControlPersist=300"]

    def _ssh(self) -> list[str]:
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
                "-o", "ServerAliveCountMax=4", "-o", "StrictHostKeyChecking=accept-new", *self.MUX,
                "-p", str(self.port), self.host]

    def sh(self, script: str, *, data: bytes | None = None, timeout: float = 60.0
           ) -> subprocess.CompletedProcess:
        return self._run(self._ssh() + [script], input=data, capture_output=True, timeout=timeout)

    def write_file(self, remote: str, data: bytes, timeout: float = 60.0) -> None:
        """Atomic write: the file appears complete or not at all."""
        path = shlex.quote(remote)
        done = self.sh(f"mkdir -p {shlex.quote(os.path.dirname(remote))} && cat > {path}.part && "
                       f"mv -f {path}.part {path}", data=data, timeout=timeout)
        if done.returncode != 0:
            raise RuntimeError(f"slave write {remote} failed rc={done.returncode}")

    def read_file(self, remote: str, timeout: float = 30.0) -> bytes | None:
        done = self.sh(f"cat {shlex.quote(remote)}", timeout=timeout)
        return done.stdout if done.returncode == 0 else None

    def probe(self, local_names: tuple[str, ...], extra: tuple[str, ...]) -> dict[str, Any]:
        """One round trip: hashes of the scoring code, the daemon socket, GPU state, the base settings."""
        files = " ".join(shlex.quote(f"/root/nanobody_opt/{name}") for name in local_names)
        files += "".join(" " + shlex.quote(path) for path in extra)
        script = (
            f"echo '##hashes'; sha256sum {files} 2>&1; "
            "echo '##daemon'; (test -S /root/artifacts/nanobody_scorer.sock && echo 1 || echo 0); "
            "echo '##gpus'; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>&1; "
            f"echo '##settings'; cat {shlex.quote(SLAVE_BASE_SETTINGS)} 2>&1; "
            f"echo '##private'; cat {shlex.quote(SLAVE_SETTINGS)} 2>/dev/null; "
            "echo '##end'"
        )
        done = self.sh(script, timeout=60.0)
        if done.returncode != 0:
            raise RuntimeError(f"slave probe failed rc={done.returncode}")
        return parse_probe(done.stdout.decode(errors="replace"))

    def cleanup_old_rounds(self) -> None:
        self.sh("find /root/live -maxdepth 3 -path '*/topup_slave/round_*' -prune -mmin +720 "
                "-exec rm -rf {} + 2>/dev/null; true", timeout=60.0)

    def score(self, command: str, timeout: float) -> tuple[int, str]:
        done = self.sh(command, timeout=timeout)
        tail = (done.stdout + done.stderr).decode(errors="replace")[-400:]
        return done.returncode, tail

    def pull_dir(self, remote: str, local: Path, timeout: float = 300.0) -> None:
        local.mkdir(parents=True, exist_ok=True)
        ssh = ("ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 "
               + " ".join(self.MUX) + f" -p {self.port}")
        done = self._run(["rsync", "-a", "--exclude=work/", "-e", ssh,
                          f"{self.host}:{remote.rstrip('/')}/", f"{local}/"],
                         capture_output=True, timeout=timeout)
        if done.returncode != 0:
            raise RuntimeError(f"rsync pull {remote} failed rc={done.returncode}")

    def remove_dir(self, remote: str) -> None:
        if "/topup_slave/" in remote:                       # never anything outside our own namespace
            self.sh(f"rm -rf {shlex.quote(remote)}", timeout=60.0)


def parse_probe(text: str) -> dict[str, Any]:
    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("##"):
            current = line[2:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    hashes = {}
    for line in sections.get("hashes", []):
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            hashes[os.path.basename(parts[1])] = parts[0]
    return {
        "hashes": hashes,
        "daemon_socket": "".join(sections.get("daemon", [])).strip() == "1",
        "gpus": [line.strip() for line in sections.get("gpus", []) if line.strip()],
        "base_settings_text": "\n".join(sections.get("settings", [])),
        "private_settings_text": "\n".join(sections.get("private", [])),
    }


def build_score_command(*, source: str, config: str, output_dir: str, gpus: list[str], count: int,
                        timeout_s: int) -> str:
    """The exact argv ``_run_screen`` builds (``_screen_command``), run on the slave.

    ``--wave-size`` is the whole tranche and ``--predict-batch-size`` stays 2 whatever the tranche
    size, as ``_screen_execution_shape`` returns for a pinned batch size.
    """
    argv = [SLAVE_PYTHON, "-m", "nanobody_opt.score_boltzgen_waves",
            "--sequences", source, "--config", config, "--target", TARGET, "--output-dir", output_dir,
            "--gpus", ",".join(gpus), "--wave-size", str(count), "--predict-batch-size", str(PREDICT_BATCH),
            "--limit", str(count), "--scorer", SLAVE_SCORER, "--capture-structures"]
    environment = f"NANOBODY_SCORER_DAEMON=1 NANOBODY_FAST_SCORER_CONFIG={shlex.quote(SLAVE_SETTINGS)}"
    return (f"cd /root && env {environment} timeout {int(timeout_s)} "
            + " ".join(shlex.quote(part) for part in argv))


# --------------------------------------------------------------------------- parity and settings

def desired_slave_settings(master: dict, slave_base: dict) -> dict:
    """The slave's own paths with the master's measurement parameters."""
    merged = {key: value for key, value in slave_base.items() if key not in MEASUREMENT_KEYS}
    for key in MEASUREMENT_KEYS:
        if key in master:
            merged[key] = master[key]
    return merged


def settings_problem(master: dict, slave_base: dict) -> str | None:
    for key in IDENTITY_KEYS:
        pair = (master.get(key), slave_base.get(key))
        if pair[0] != pair[1] and not (key == "scorer_sha256" and pair == REVIEWED_DIVERGENCE.get("fast_score_batch.py")):
            return f"settings key {key} differs between master and slave"
    if os.path.basename(str(master.get("synth_reference"))) != os.path.basename(str(slave_base.get("synth_reference"))):
        return "synth_reference differs between master and slave"
    return None


def master_hashes() -> dict[str, str]:
    out = {}
    for name in PARITY_FILES:
        path = NANOBODY_OPT / name
        if path.is_file():
            out[name] = sha256_file(path)
    for extra in PARITY_EXTRA:
        if Path(extra).is_file():
            out[os.path.basename(extra)] = sha256_file(Path(extra))
    return out


def preflight(slave: Slave, *, write: bool = True) -> dict[str, Any]:
    """Is the slave a faithful second scorer?  Returns {"ok": bool, "reason": ..., ...}."""
    result: dict[str, Any] = {"ok": False}
    try:
        probe = slave.probe(PARITY_FILES, PARITY_EXTRA)
    except Exception as error:  # noqa: BLE001
        return {**result, "reason": f"slave unreachable: {type(error).__name__}"}
    mine = master_hashes()
    theirs = probe["hashes"]
    unequal = [name for name in mine if theirs.get(name) != mine[name]]
    reviewed = sorted(name for name in unequal if REVIEWED_DIVERGENCE.get(name) == (mine[name], theirs.get(name)))
    different = sorted(name for name in unequal if name not in reviewed)
    result["files_checked"] = len(mine)
    if reviewed:
        result["reviewed_divergence"] = reviewed
    if different or len(mine) != len(PARITY_FILES) + len(PARITY_EXTRA):
        return {**result, "reason": "scoring code differs from master", "different": different}
    master = read_json(MASTER_SETTINGS) or {}
    try:
        slave_base = json.loads(probe["base_settings_text"])
    except ValueError:
        return {**result, "reason": "slave scorer settings unreadable"}
    problem = settings_problem(master, slave_base)
    if problem:
        return {**result, "reason": problem}
    wanted = desired_slave_settings(master, slave_base)
    wanted_text = json.dumps(wanted, indent=2, sort_keys=True) + "\n"
    try:
        private_ok = json.loads(probe["private_settings_text"]) == wanted
    except ValueError:
        private_ok = False
    if not private_ok:
        if not write:
            result["settings_would_change"] = True
        else:
            try:
                slave.write_file(SLAVE_SETTINGS, wanted_text.encode())
            except Exception as error:  # noqa: BLE001
                return {**result, "reason": f"settings write failed: {type(error).__name__}"}
    gpus = [line.split(",")[0].strip() for line in probe["gpus"] if line.split(",")[0].strip().isdigit()]
    if not gpus:
        return {**result, "reason": "no GPUs reported by the slave"}
    result.update(ok=True, daemon_socket=probe["daemon_socket"], gpus=gpus,
                  fold_sampling_steps=wanted.get("fold_sampling_steps"),
                  gpu_state=probe["gpus"])
    if not probe["daemon_socket"]:
        result["note"] = "no warm daemon on the slave: rounds fall back to cold subprocesses (slower)"
    return result


def gpu_pairs(gpus: list[str], size: int | None = None) -> list[list[str]]:
    size = size or GROUP_SIZE
    groups = [gpus[i:i + size] for i in range(0, len(gpus), size)]
    return [group for group in groups if len(group) == size] or [gpus]


# --------------------------------------------------------------------------- what to top up

def _sequences_of(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict) and isinstance(value.get("sequence"), str):
        return [value["sequence"]]
    return []


def live_sequences(epoch: int) -> list[str]:
    """Everything already published this epoch: the sequences the strict pool must keep selectable."""
    found: list[str] = []

    def add(sequence: str) -> None:
        sequence = sequence.strip().upper()
        if sequence and sequence not in found:
            found.append(sequence)

    artifact = read_json(ARTIFACT)
    if isinstance(artifact, dict) and artifact.get("epoch") == epoch:
        for item in artifact.get("submissions") or []:
            for sequence in _sequences_of(item):
                add(sequence)
        for item in artifact.get("sequence_pool") or []:
            for sequence in _sequences_of(item):
                add(sequence)
    root = LIVE_ROOT / f"epoch{epoch}" / "checkpoints"
    if root.is_dir():
        for path in sorted(root.glob("*/uid*.json")):
            checkpoint = read_json(path)
            if isinstance(checkpoint, dict) and checkpoint.get("epoch") == epoch:
                for item in (checkpoint.get("sequences") or [])[:1]:
                    for sequence in _sequences_of(item):
                        add(sequence)
    return found


_RANKING_CACHE: dict[str, tuple[float, list[tuple[str, int]]]] = {}
_CLEANED: dict[str, int] = {}
_PASSES: dict[int, int] = {}
_STALLED_UNTIL = [0.0]


def load_ranking(path: Path) -> list[tuple[str, int]] | None:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    cached = _RANKING_CACHE.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]
    payload = read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        return None
    try:
        ranking = [(str(item["sequence"]).strip().upper(), int(item["observation_count"]))
                   for item in payload["candidates"]]
    except (KeyError, TypeError, ValueError):
        return None
    _RANKING_CACHE[str(path)] = (mtime, ranking)
    return ranking


def ranking_sources(epoch: int) -> dict[str, Any]:
    """The newest provisional-mode ranking (strict pool order) and the qualified ranking, if any."""
    root = LIVE_ROOT / f"epoch{epoch}"
    pool_files = [root / "strict_pool_rankings.json", root / "provisional_rankings.json",
                  root / "topup" / "fresh_ranking.json"]
    # Confirmation exists before publication. A blocked publisher must never
    # hide the producer's current finalists from the spare GPU worker.
    pool_files.extend(root.glob('producer/iteration_*/race/rounds/stage_ranking.json'))
    pool_files.extend(root.glob('producer/iteration_*/race/screen_ranking.json'))
    existing = sorted((p for p in pool_files if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)
    pool, pool_name, pool_age = [], None, None
    for path in existing:
        ranking = load_ranking(path)
        if ranking:
            pool, pool_name, pool_age = ranking, path.name if path.parent == root else f"topup/{path.name}", \
                round(time.time() - path.stat().st_mtime)
            break
    qualified_path = root / "qualified_rankings.json"
    qualified = load_ranking(qualified_path) or []
    return {"pool": pool, "pool_name": pool_name, "pool_age_s": pool_age,
            "qualified": qualified,
            "qualified_age_s": round(time.time() - qualified_path.stat().st_mtime) if qualified else None}


def db_counts(sequences: list[str]) -> dict[str, int] | None:
    """Current ``local_exact_raw`` observation counts, read-only (the publisher counts exactly these)."""
    if not sequences:
        return {}
    try:
        connection = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=20)
    except sqlite3.Error:
        return None
    try:
        connection.execute("PRAGMA query_only=1")
        counts: dict[str, int] = {}
        for start in range(0, len(sequences), 300):
            chunk = sequences[start:start + 300]
            marks = ",".join("?" * len(chunk))
            for sequence, count in connection.execute(
                    "SELECT n.sequence, COUNT(o.observation_id) FROM nodes n LEFT JOIN observations o "
                    "ON o.node_id = n.id AND o.source = 'local_exact_raw' "
                    f"WHERE n.target = ? AND n.sequence IN ({marks}) GROUP BY n.sequence",
                    [TARGET, *chunk]):
                counts[str(sequence)] = int(count)
        return counts
    except sqlite3.Error:
        return None
    finally:
        connection.close()


def epoch_state_path(epoch: int) -> Path:
    return (ROUNDS_ROOT or LIVE_ROOT) / f"epoch{epoch}" / "topup_slave" / "state.json"


def needs(epoch: int) -> tuple[dict[str, int], dict[str, Any]]:
    """{sequence: rounds still needed} in priority order, and what it was derived from."""
    live = live_sequences(epoch)
    sources = ranking_sources(epoch)
    pool = sources["pool"][:POOL_K]
    qualified = sources["qualified"][:QUALIFIED_K]
    order = list(dict.fromkeys(
        [*live, *(s for s, _ in qualified), *(s for s, _ in pool)]))
    info: dict[str, Any] = {"live": len(live), "qualified": len(qualified), "pool": len(pool),
                            "pool_ranking": sources["pool_name"], "pool_ranking_age_s": sources["pool_age_s"],
                            "qualified_age_s": sources["qualified_age_s"], "candidates": len(order)}
    counts = db_counts(order)
    if counts is not None:
        info["counts"] = "database"
    else:                                       # fall back to the ranking counts plus what we added since
        info["counts"] = "ranking files"
        known = {s: c for s, c in [*pool, *qualified]}
        state = read_json(epoch_state_path(epoch)) or {}
        counts = {}
        for sequence in order:
            counts[sequence] = known.get(sequence, 0) + sum(
                n for _stamp, n in (state.get("added", {}).get(sequence, [])))
    need: dict[str, int] = {}
    skipped_length = 0
    for sequence in order:
        if len(sequence) != SEQUENCE_LENGTH:
            skipped_length += 1
            continue
        have = counts.get(sequence, 0)
        if have < TARGET_OBS:
            need[sequence] = min(TARGET_OBS - have, MAX_ROUNDS)
    if skipped_length:
        info["skipped_length"] = skipped_length
    total, capped = 0, {}
    for sequence, rounds in need.items():            # priority order; drop the tail beyond the budget
        if total + rounds > MAX_OBS_PER_PASS:
            info["capped_by_budget"] = True
            continue
        capped[sequence] = rounds
        total += rounds
    info["short"] = len(capped)
    info["have"] = {s[:10]: counts.get(s, 0) for s in list(capped)[:12]}
    return capped, info


def crew_size() -> int:
    """How many uids we commit for (the Pi's crew): the newest artifact says."""
    artifact = read_json(ARTIFACT)
    if isinstance(artifact, dict):
        count = len(artifact.get("submissions") or []) or int(artifact.get("submission_count") or 0)
        if count > 0:
            return count
    return DEFAULT_CREW


def final_import_offset(crew: int | None = None) -> int:
    """The latest epoch offset by which an import must be done to reach the Pi's final pull."""
    crew = crew_size() if crew is None else crew
    pull = PI_PULL_BASE - PI_BLOCKS_PER_UID * crew + PI_PULL_X_MIN
    return max(FINAL_IMPORT_FLOOR, min(FINAL_IMPORT_OFFSET, pull - CYCLE_BLOCKS))


def master_topup_busy() -> bool:
    try:
        return time.time() - MASTER_MARKER.stat().st_mtime < 600
    except OSError:
        return False


# --------------------------------------------------------------------------- one round on the slave

def rounds_root() -> Path:
    return ROUNDS_ROOT or LIVE_ROOT


def stage_root() -> Path:
    """Where pulled rounds wait for validation: on the destination's own filesystem (so the final move
    is one atomic rename) but outside ``live/epoch*``, which the publisher globs for exact_all.json."""
    return rounds_root() / ".topup_slave_incoming"


def round_dir(epoch: int, stamp: str, number: int) -> Path:
    return rounds_root() / f"epoch{epoch}" / "topup_slave" / f"round_{stamp}_{number:02d}"


def validate_round(directory: Path, final_dir: Path, subset: list[str]) -> str | None:
    """Return a reason the round must not be imported, or None.  ``directory`` is where it sits now,
    ``final_dir`` where the paths inside it say it will live."""
    payload = read_json(directory / "exact_all.json")
    if not isinstance(payload, dict):
        return "exact_all.json missing or unreadable"
    if str(payload.get("target")) != TARGET:
        return "wrong target"
    if not str(payload.get("observation_id") or "").strip():
        return "no observation id"
    if payload.get("scorer_protocol") != SCORER_PROTOCOL:
        return "wrong scorer protocol"
    components = (payload.get("per_nanobody_components") or {}).get("0") or {}
    if set(components) != set(subset) or int(payload.get("n", -1)) != len(subset):
        return "sequences do not match the request"
    structures = payload.get("structures") or {}
    for sequence in subset:
        metrics = (components[sequence] or {}).get(TARGET)
        if not isinstance(metrics, dict):
            return "missing metrics"
        for name in REQUIRED_METRICS:
            value = metrics.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return f"metric {name} missing or not finite"
        record = (structures.get(sequence) or {}).get(TARGET) or {}
        for key, sha_key in (("path", "sha256"), ("writer_path", "writer_sha256")):
            recorded = str(record.get(key) or "")
            if not recorded.startswith(str(final_dir) + "/"):
                return "structure path is not under the round directory"
            staged = directory / recorded[len(str(final_dir)) + 1:]
            if not staged.is_file() or (record.get(sha_key) and sha256_file(staged) != record[sha_key]):
                return "structure file missing or corrupt"
    return None


def score_round(slave: Slave, *, epoch: int, stamp: str, number: int, subset: list[str],
                gpus: list[str], remote_config: str) -> dict[str, Any]:
    """Score one tranche on the slave, bring it home and put it in place atomically."""
    final = round_dir(epoch, stamp, number)
    result: dict[str, Any] = {"round": number, "candidates": len(subset), "gpus": ",".join(gpus)}
    started = time.monotonic()
    if final.exists():
        return {**result, "error": "round directory exists"}
    remote = str(final)
    # Batches of two share one predict call, and a candidate's partner moves its metrics by a few percent
    # of its own spread.  Repeating the same order every round repeats the same partners, so the repeats
    # are less independent than they look (measured: the same 12 sequences, same order x3 rounds vs the
    # reverse order x6 rounds, per-metric mean z -0.44..+0.19 vs -0.31..+0.12).  A seeded shuffle per
    # round gives every repeat fresh partners; the scoring call itself is unchanged.
    subset = random.Random(f"{stamp}-{number}").sample(subset, len(subset))
    try:
        slave.write_file(f"{remote}/source.json", json.dumps({"target": TARGET, "sequences": subset}).encode())
        command = build_score_command(source=f"{remote}/source.json", config=remote_config, output_dir=remote,
                                      gpus=gpus, count=len(subset), timeout_s=840)
        code, tail = slave.score(command, timeout=900)
        result["score_seconds"] = round(time.monotonic() - started, 1)
        if code != 0:
            return {**result, "error": f"scoring rc={code}", "tail": tail[-200:]}
        stage_root().mkdir(parents=True, exist_ok=True)
        staging = stage_root() / f"{stamp}_{number:02d}_{os.getpid()}"
        shutil.rmtree(staging, ignore_errors=True)
        slave.pull_dir(remote, staging)
        problem = validate_round(staging, final, subset)
        if problem:
            shutil.rmtree(staging, ignore_errors=True)
            return {**result, "error": f"invalid round: {problem}"}
        final.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, final)                 # one atomic step: the publisher never sees a partial round
        wave = read_json(final / "wave_000_exact.json") or {}
        if isinstance(wave.get("wall_seconds"), (int, float)):
            result["compute_seconds"] = round(float(wave["wall_seconds"]), 1)   # inside the daemon's GPU locks
        result["ok"] = True
        result["seconds"] = round(time.monotonic() - started, 1)
        result["dir"] = str(final)
        try:
            slave.remove_dir(remote)
        except Exception:  # noqa: BLE001 -- cleanup only
            pass
        return result
    except Exception as error:  # noqa: BLE001 -- never hurt anything else
        return {**result, "error": f"{type(error).__name__}: {str(error)[:120]}"}


# --------------------------------------------------------------------------- one pass

def import_rounds(epoch: int, directories: list[Path], archive: Path | None = None) -> tuple[int, list[str]]:
    if archive is None:
        archive = LIVE_ROOT / f"epoch{epoch}" / "producer" / "archive.csv"
        if not archive.is_file():
            archive = LIVE_ROOT / f"epoch{epoch}" / "archive.csv"
    command = ["nice", "-n", "5", IMPORT_PYTHON, "-m", "nanobody_opt.import_exact_screens",
               "--database", str(IMPORT_DB or DB), "--target", TARGET, "--archive", str(archive)]
    for directory in directories:
        command += ["--exact-root", str(directory)]
    environment = {**os.environ, "PYTHONPATH": "/root"}
    completed = subprocess.run(command, capture_output=True, text=True, timeout=900, cwd="/root", env=environment)
    tail = (completed.stdout or completed.stderr).strip().splitlines()[-2:]
    return completed.returncode, tail


def free_disk_gb() -> float:
    usage = shutil.disk_usage(str(LIVE_ROOT if LIVE_ROOT.exists() else ARTIFACTS))
    return usage.free / 1e9


def run_pass(slave: Slave, *, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    try:
        epoch, offset, age = chain_position()
    except (OSError, KeyError, ValueError):
        return {"skipped": "chain state unreadable"}
    result: dict[str, Any] = {"epoch": epoch, "offset": round(offset, 1)}
    if age > CHAIN_MAX_AGE and not force:
        return {**result, "skipped": "chain state is stale"}
    deadline_offset = final_import_offset()
    result["import_deadline_offset"] = deadline_offset
    if time.time() < _STALLED_UNTIL[0] and not dry_run:
        return {**result, "skipped": "paused: the last import did not raise the observation counts"}
    if _PASSES.get(epoch, 0) >= MAX_PASSES_PER_EPOCH and not dry_run:
        return {**result, "skipped": "pass limit for this epoch"}
    if not force and not (FIRST_OFFSET <= offset <= deadline_offset - ROUND_BLOCKS):
        return {**result, "skipped": "outside the top-up window"}
    if free_disk_gb() < MIN_FREE_DISK_GB and not dry_run:
        return {**result, "skipped": "master disk below the producer's reserve"}
    waited = 0.0
    while master_topup_busy() and waited < MASTER_WAIT and not dry_run:
        time.sleep(5.0)
        waited += 5.0
    if waited:
        result["waited_for_master_s"] = waited
        time.sleep(20.0)                           # let its import land before we recount
    need, info = needs(epoch)
    result.update(info)
    if not need:
        return {**result, "skipped": "nothing short of the target"}
    ready = preflight(slave, write=not dry_run)
    result["preflight"] = {k: v for k, v in ready.items() if k != "gpu_state"}
    if not ready["ok"]:
        return {**result, "skipped": f"preflight: {ready.get('reason')}"}
    if _CLEANED.get("epoch") != epoch:                 # rounds a failed pull left behind on the slave
        try:
            slave.cleanup_old_rounds()
            _CLEANED["epoch"] = epoch
        except Exception:  # noqa: BLE001 -- housekeeping only
            pass
    pairs = gpu_pairs(ready["gpus"])
    rounds = [(n, [s for s in need if need[s] >= n]) for n in range(1, max(need.values()) + 1)]
    rounds = [(n, subset) for n, subset in rounds if subset]
    result["observations_needed"] = sum(need.values())
    result["rounds_planned"] = len(rounds)
    if dry_run:
        return {**result, "dry_run": True,
                "round_sizes": [len(subset) for _n, subset in rounds],
                "need": {s[:12]: n for s, n in list(need.items())[:20]}}
    config = LIVE_ROOT / f"epoch{epoch}" / "producer" / "config.yaml"
    if not config.is_file():
        config = LIVE_ROOT / f"epoch{epoch}" / "config.yaml"
    remote_config = f"{round_dir(epoch, 'x', 0).parent}/config.yaml"
    try:
        slave.write_file(remote_config, config.read_bytes())
    except Exception as error:  # noqa: BLE001
        return {**result, "skipped": f"config upload failed: {type(error).__name__}"}
    _PASSES[epoch] = _PASSES.get(epoch, 0) + 1
    stamp = time.strftime("%H%M%S", time.gmtime()) + f"{_PASSES[epoch]:02d}"      # unique per pass
    started = time.monotonic()
    before = db_counts(list(need)) or {}
    if not ROUNDS_ROOT:                                  # a pass takes minutes: say it has begun
        log(event="pass_start", epoch=epoch, offset=result["offset"], short=len(need),
            observations_needed=result["observations_needed"], rounds_planned=len(rounds),
            import_deadline_offset=deadline_offset)
    available: "queue.Queue[list[str]]" = queue.Queue()
    for pair in pairs:
        available.put(pair)

    def work(item: tuple[int, list[str]]) -> dict[str, Any]:
        number, subset = item
        epoch_now, offset_now, _a = chain_position()
        if epoch_now != epoch:                                  # the epoch we were topping up is over
            return {"round": number, "candidates": len(subset), "skipped": "epoch rolled"}
        if not force and offset_now + ROUND_BLOCKS > deadline_offset:
            return {"round": number, "candidates": len(subset), "skipped": "final import deadline"}
        if time.monotonic() - started > PASS_MAX_SECONDS:      # import what is done; the rest is next pass
            return {"round": number, "candidates": len(subset), "skipped": "pass time limit"}
        pair = available.get()
        try:
            return score_round(slave, epoch=epoch, stamp=stamp, number=number, subset=subset,
                               gpus=pair, remote_config=remote_config)
        finally:
            available.put(pair)

    with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
        outcomes = list(pool.map(work, rounds))
    done = [o for o in outcomes if o.get("ok")]
    result["rounds_scored"] = len(done)
    failed = [o for o in outcomes if o.get("error")]
    if failed:
        result["rounds_failed"] = [f"{o['round']}: {o['error']}" for o in failed][:4]
    stopped = sorted({o["skipped"] for o in outcomes if o.get("skipped")})
    if stopped:
        result["stopped"] = ", ".join(stopped)
    result["round_seconds"] = [o.get("seconds") for o in done][:12]
    result["compute_seconds"] = [o.get("compute_seconds") for o in done][:12]
    if done:
        directories = [Path(o["dir"]) for o in done]
        rc, tail = import_rounds(epoch, directories)
        result["import_rc"] = rc
        result["import_tail"] = tail
        if rc == 0:
            added: dict[str, int] = {}
            for o in done:
                for sequence in next(subset for n, subset in rounds if n == o["round"]):
                    added[sequence] = added.get(sequence, 0) + 1
            state = read_json(epoch_state_path(epoch)) or {}
            now = time.time()
            for sequence, count in added.items():
                state.setdefault("added", {}).setdefault(sequence, []).append([now, count])
            epoch_state_path(epoch).write_text(json.dumps(state))
            result["observations_added"] = sum(added.values())
            after = db_counts(list(need))
            if after is not None and not IMPORT_DB:      # a rehearsal imports elsewhere: nothing to compare
                result["still_short_after"] = sum(1 for s in need if after.get(s, 0) < TARGET_OBS)
                gained = sum(max(0, after.get(s, 0) - before.get(s, 0)) for s in need)
                result["counts_gained"] = gained
                if before and gained < 0.5 * result["observations_added"]:
                    _STALLED_UNTIL[0] = time.time() + STALL_PAUSE     # e.g. an importer that drops rows
                    result["warning"] = "import did not raise the counts; pausing"
    result["seconds"] = round(time.monotonic() - started, 1)
    return result


# --------------------------------------------------------------------------- commands

def take_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return None
    return handle


def command_rounds(args: argparse.Namespace, slave: Slave) -> int:
    """Manual: score N rounds of the given sequences on the slave and bring them home (no import
    unless --import-db).  Used for the equivalence test; ``--round-root`` keeps it out of /root/live."""
    global ROUNDS_ROOT, IMPORT_DB
    if args.round_root:
        ROUNDS_ROOT = Path(args.round_root)
    sequences = [s.strip().upper() for s in Path(args.sequences).read_text().split() if s.strip()]
    ready = preflight(slave, write=True)
    print(json.dumps({"preflight": {k: v for k, v in ready.items() if k != "gpu_state"}}, sort_keys=True))
    if not ready["ok"]:
        return 2
    epoch = int(args.epoch)
    remote_config = f"{round_dir(epoch, 'x', 0).parent}/config.yaml"
    slave.write_file(remote_config, Path(args.config).read_bytes())
    pairs = gpu_pairs(ready["gpus"], args.group_size)
    stamp = time.strftime("%H%M%S", time.gmtime())
    available: "queue.Queue[list[str]]" = queue.Queue()
    for pair in pairs:
        available.put(pair)

    def work(number: int) -> dict[str, Any]:
        pair = available.get()
        try:
            return score_round(slave, epoch=epoch, stamp=stamp, number=number, subset=sequences,
                               gpus=pair, remote_config=remote_config)
        finally:
            available.put(pair)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
        outcomes = list(pool.map(work, range(1, int(args.rounds) + 1)))
    print(json.dumps({"outcomes": outcomes, "wall_seconds": round(time.monotonic() - started, 1)}, sort_keys=True))
    good = [Path(o["dir"]) for o in outcomes if o.get("ok")]
    if args.import_db and good:
        IMPORT_DB = Path(args.import_db)
        rc, tail = import_rounds(epoch, good, Path(args.archive) if args.archive else None)
        print(json.dumps({"import_rc": rc, "import_tail": tail}))
    return 0 if len(good) == len(outcomes) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    once = sub.add_parser("once")
    once.add_argument("--dry-run", action="store_true")
    once.add_argument("--force", action="store_true", help="ignore the epoch-offset window (testing)")
    once.add_argument("--rounds-root", help="rehearsal: put the rounds here instead of /root/live")
    once.add_argument("--import-db", help="rehearsal: import into THIS scratch database, never the live one")
    sub.add_parser("loop")
    sub.add_parser("preflight")
    rounds = sub.add_parser("rounds")
    rounds.add_argument("--sequences", required=True, help="file with whitespace-separated sequences")
    rounds.add_argument("--config", required=True)
    rounds.add_argument("--epoch", required=True, type=int)
    rounds.add_argument("--rounds", type=int, default=3)
    rounds.add_argument("--round-root", help="where the rounds live (default /root/live; use a scratch dir)")
    rounds.add_argument("--import-db", help="import into THIS database only (a scratch copy, never the live one)")
    rounds.add_argument("--archive", help="archive.csv for the importer (with --import-db)")
    rounds.add_argument("--group-size", type=int, help="GPUs per round (default 2, like the master top-up)")
    args = parser.parse_args()
    slave = Slave()
    if args.command == "preflight":
        print(json.dumps(preflight(slave, write=False), sort_keys=True, indent=1))
        return 0
    if args.command == "rounds":
        return command_rounds(args, slave)
    lock = take_lock(LOCK)
    if lock is None:
        print("another slave top-up is running")
        return 0
    if args.command == "once":
        global ROUNDS_ROOT, IMPORT_DB
        if args.rounds_root or args.import_db:
            if not (args.rounds_root and args.import_db):
                parser.error("a rehearsal needs both --rounds-root and --import-db")
            ROUNDS_ROOT, IMPORT_DB = Path(args.rounds_root), Path(args.import_db)
        result = run_pass(slave, dry_run=args.dry_run, force=args.force)
        if ROUNDS_ROOT:
            result["rehearsal"] = True
        print(json.dumps(result, sort_keys=True))
        if not args.dry_run and "skipped" not in result and not ROUNDS_ROOT:
            log(**result)
        return 0
    log(event="started", pool_k=POOL_K, first_offset=FIRST_OFFSET, final_import_offset_max=FINAL_IMPORT_OFFSET,
        import_deadline_now=final_import_offset())
    last_skip = None
    while True:
        pause = IDLE_SLEEP
        try:
            result = run_pass(slave)
            if "skipped" not in result:
                log(**result)
                pause = PASS_GAP
            else:                               # one line per change of reason, not one per poll
                reason = result["skipped"]
                if reason != last_skip and reason not in ("outside the top-up window", "nothing short of the target"):
                    log(**{k: v for k, v in result.items() if k != "have"})
                last_skip = reason
        except Exception as error:  # noqa: BLE001
            log(error=f"{type(error).__name__}: {str(error)[:200]}")
            pause = 60.0
        time.sleep(pause)


if __name__ == "__main__":
    raise SystemExit(main())

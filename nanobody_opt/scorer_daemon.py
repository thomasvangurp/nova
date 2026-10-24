"""A long-lived scorer the producer talks to, instead of a process per round.

WHY. `_run_screen` runs `subprocess.run(<scorer CLI>)` once per replication
round, and the scorer's `main()` then spawns one shard process per GPU. So a
round pays five process starts, five `import torch` + boltzgen + hydra + rdkit,
a model build, a rotamer TemplateLibrary and an analysis-pool spawn -- before
it scores anything.

Measured on the live producer over 168 replication rounds:

    seconds = 32.9 + 1.02 * candidates          (4 GPUs)

    candidates   rounds   round s   scoring s   GPU busy
             2       20      34.9         2.0         6%
            21        8      54.3        21.4        39%
            59        8      93.1        60.2        65%

    weighted over the real distribution: 9,150 s total, 3,623 s scoring
    -> 40% GPU busy, 60% of replication wall clock spent on setup

That 60% is not a sampling artifact: a 60 s window landing on 59-candidate
rounds measures 87% utilisation, a 150 s window landing on 2-candidate rounds
measures 2.5%, and the weighted truth is 40%.

WHAT DOES NOT FIX IT. Caching inside the scorer process (resident_scorer)
saves 3 s of 41 -- the model was never the cost -- and in production caches
nothing at all, because no process survives to make a second call. Batching
replicates into one call is rejected outright: the scorer deduplicates by
sequence at merge and returns "MERGE LOST SEQUENCES: submitted 12, merged 2".

WHAT THIS DOES. Starts the scorer once, keeps it alive, and feeds it rounds
over a unix socket. Each round is still its own `_worker` call with its own
output directory and predict composition, so shard isolation and the measured
metric are unchanged -- only the process boundary moves. Projected from the
same fit with setup at ~3 s: 9,150 s -> ~4,127 s (2.2x), GPU busy 40% -> 88%,
which is what the large-round sample already shows the hardware can sustain.

SAFETY. The producer keeps its subprocess path. If the socket is absent, the
daemon is unhealthy, or any round fails, `_run_screen` falls back to spawning
the scorer exactly as it does today, so the worst case is current behaviour.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time

#: Producer-side switch. Off means the current subprocess path, unchanged.
ENABLED = os.environ.get("NANOBODY_SCORER_DAEMON", "0") == "1"
SOCKET_PATH = os.environ.get("NANOBODY_SCORER_SOCKET",
                             "/root/artifacts/nanobody_scorer.sock")
#: Sized for the SCREEN path, not the replication path it was written for.
#: "A round of 59 candidates measured 93 s" was true when the daemon only saw
#: small replication rounds. It now also serves screen waves: a live wave of
#: 700 sequences took 730 s, 81% of the old 900 s budget.
#:
#: Timing out here is worse than waiting. The client gives up, the adapter
#: falls back, and the subprocess re-folds the same wave WHILE the daemon is
#: still folding it -- double the device memory and double the work, to save
#: a wait that would have ended. The cap exists only so a wedged worker
#: cannot block a producer forever, so it belongs well above any real round.
REQUEST_TIMEOUT = float(os.environ.get("NANOBODY_SCORER_TIMEOUT", "3600"))

#: Resident worker processes per GPU. One worker folds one shard at a time, so a GPU idles through
#: that worker's host-side stages (setup, featurisation, hand-off). With several workers per GPU
#: one worker's host-side stage overlaps another's GPU stage. 1 keeps the old behaviour exactly.
WORKERS_PER_GPU_FILE = os.environ.get("NANOBODY_SCORER_WORKERS_FILE",
                                      "/root/artifacts/scorer_workers_per_gpu")
#: A job only gets the extra workers when it has at least this many sequences per worker, so
#: small rounds (a shortlist of a few dozen) keep their single-worker layout unchanged.
MIN_SEQUENCES_PER_WORKER = 4


def workers_per_gpu() -> int:
    """Operator setting, re-read every round so it changes without a restart.

    The environment wins, then the one-line file, then 1. Anything unreadable or non-numeric
    means 1: the old single-worker behaviour.
    """
    raw = os.environ.get("NANOBODY_SCORER_WORKERS_PER_GPU")
    if raw is None:
        try:
            with open(WORKERS_PER_GPU_FILE) as handle:
                raw = handle.read().strip()
        except OSError:
            return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _worker_key(gpu, slot: int) -> str:
    """Registry key of the `slot`-th worker on `gpu`; slot 0 keeps the bare GPU id."""
    return str(gpu) if slot == 0 else "%s#%d" % (gpu, slot)


def effective_shards_per_gpu(requested: int, sequences: int, gpus: int) -> int:
    """Shards per GPU to use for a job that asked for `requested`.

    A caller that already asks for several shards per GPU keeps its choice (those shards now run
    on separate workers). A caller that asks for one -- the screening tranche -- is widened to the
    worker count, but only when the job is large enough to give every worker real work.
    """
    workers = workers_per_gpu()
    if requested != 1 or workers <= 1:
        return requested
    if sequences < MIN_SEQUENCES_PER_WORKER * workers * max(1, gpus):
        return requested
    return workers


def _send(sock, payload: dict) -> None:
    body = json.dumps(payload).encode()
    sock.sendall(struct.pack("!Q", len(body)) + body)


def _recv(sock) -> dict | None:
    header = b""
    while len(header) < 8:
        chunk = sock.recv(8 - len(header))
        if not chunk:
            return None
        header += chunk
    (length,) = struct.unpack("!Q", header)
    if length > (64 << 20):            # a scorer reply is small; refuse absurd
        raise ValueError("oversized daemon frame")
    body = b""
    while len(body) < length:
        chunk = sock.recv(min(65536, length - len(body)))
        if not chunk:
            return None
        body += chunk
    return json.loads(body)


# --- client -----------------------------------------------------------------

def available(path: str | None = None) -> bool:
    """True when a daemon is listening and answers a ping."""
    target = path or SOCKET_PATH
    if not os.path.exists(target):
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            sock.connect(target)
            _send(sock, {"op": "ping"})
            reply = _recv(sock)
        return bool(reply and reply.get("ok"))
    except OSError:
        return False


def score(job: dict, path: str | None = None,
          timeout: float | None = None) -> dict:
    """Run one round on the daemon. Raises on any failure so the caller falls
    back to the subprocess path rather than silently losing a round."""
    target = path or SOCKET_PATH
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout or REQUEST_TIMEOUT)
        sock.connect(target)
        _send(sock, {"op": "score", "job": job})
        reply = _recv(sock)
    if not reply:
        raise RuntimeError("scorer daemon closed the connection")
    if not reply.get("ok"):
        raise RuntimeError("scorer daemon error: %s" % reply.get("error"))
    return reply


# --- shard workers ----------------------------------------------------------

def _client_disconnected(conn) -> bool:
    import select
    try:
        ready, _, _ = select.select([conn], [], [], 0)
        return bool(ready) and not conn.recv(1, socket.MSG_PEEK)
    except OSError:
        return True


def _watch_parent(parent_pid: int, interval: float = 1.0) -> None:
    """Watch the owning daemon, including workers started through a launcher.

    The configured Python launcher may insert a process between daemon and
    worker. Comparing getppid() with the daemon PID would kill healthy workers.
    """
    def owner_alive():
        if parent_pid <= 1:
            return False
        try:
            os.kill(parent_pid, 0)
            # A zombie still answers kill(pid, 0); it cannot own a worker.
            with open(f"/proc/{parent_pid}/stat") as handle:
                return handle.read().rsplit(")", 1)[1].split()[0] != "Z"
        except FileNotFoundError:
            return os.path.isdir("/proc") is False
        except ProcessLookupError:
            return False

    def watch():
        while True:
            if not owner_alive():
                import signal
                if os.getpgrp() == os.getpid():
                    os.killpg(os.getpgrp(), signal.SIGKILL)
                os._exit(0)
            time.sleep(interval)
    threading.Thread(target=watch, name="scorer-parent-watch", daemon=True).start()


class ShardWorker:
    """One persistent process pinned to one GPU.

    MUST be a process, not a thread. The scorer's `_worker` begins with

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    which is process-global. Four shards in four threads race on that single
    variable and all land on whichever device was assigned last: a
    16-candidate round measured 62.9 s as four processes and 164 s as four
    threads -- 2.6x SLOWER, because the shards serialised onto one GPU.
    """

    def __init__(self, gpu: str, scorer_path: str, python: str | None = None,
                 slot: int = 0):
        import subprocess
        self.gpu = str(gpu)
        self.slot = int(slot)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = self.gpu   # pinned once, privately
        self.proc = subprocess.Popen(
            [python or __import__("sys").executable, os.path.abspath(__file__),
             "--shard-worker", "--parent-pid", str(os.getpid()), "--scorer", scorer_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            # To a FILE, never DEVNULL: a worker that dies takes its reason
            # with it and the parent sees only an empty read.
            stderr=open(os.path.join(
                os.environ.get("NANOBODY_WORKER_LOG_DIR", "/tmp"),
                "shard_worker_gpu%s.log" % self.gpu if self.slot == 0
                else "shard_worker_gpu%s_s%d.log" % (self.gpu, self.slot)), "w"),
            env=env, text=True, bufsize=1, start_new_session=True)
        #: Serialises request/reply on this worker's pipe.
        self._pipe = threading.Lock()
        try:
            line = self._read_reply(120)
        except Exception:
            self.close()
            raise
        try:
            hello = json.loads(line)
        except Exception:
            hello = {}
        if not hello.get("ready"):
            raise RuntimeError("shard worker on gpu %s failed to start: %r"
                               % (self.gpu, line[:200]))

    def alive(self) -> bool:
        return self.proc.poll() is None

    def _read_reply(self, timeout: float, cancelled=None) -> str:
        """Bound waits even when a dead worker's child retains its pipe.

        A blocking readline cannot notice worker death or client cancellation.
        Read bytes only when available, and check both between partial replies.
        """
        import select
        deadline = time.monotonic() + timeout
        data = bytearray()
        while True:
            if cancelled is not None and cancelled():
                raise RuntimeError("scoring client disconnected; cancelling shard")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("shard worker reply deadline exceeded")
            ready, _, _ = select.select([self.proc.stdout], [], [], min(0.25, remaining))
            if ready:
                chunk = os.read(self.proc.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("shard worker closed its reply pipe")
                data.extend(chunk)
                if b"\n" in data:
                    return bytes(data).split(b"\n", 1)[0].decode()
            if self.proc.poll() is not None:
                raise RuntimeError("shard worker exited before completing its reply")

    def run(self, args: list, target: str | None = None, cancelled=None) -> dict:
        """One request at a time down this worker's pipe.

        `shards_per_gpu > 1` puts two shards on one device, and dispatch
        submits them to the SAME ShardWorker from two pool threads. Without
        this lock both write a request and then race to read a reply: the
        replies are matched to the wrong requests, or a half-written line is
        parsed. It is the stdout/protocol corruption again, one level up.
        The worker process can only fold one shard at a time regardless, so
        serialising here costs nothing.
        """
        with self._pipe:
            if not self.alive():
                raise RuntimeError("shard worker on gpu %s is dead" % self.gpu)
            try:
                self.proc.stdin.write(
                    json.dumps({"args": args, "target": target}) + "\n")
                self.proc.stdin.flush()
                return json.loads(self._read_reply(900, cancelled))
            except Exception:
                self.close()
                raise

    def close(self) -> None:
        import signal
        try:
            if self.alive():
                self.proc.stdin.close()
                self.proc.wait(timeout=2)
        except Exception:
            pass  # Always clean the isolated process group, including orphans.
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc.wait(timeout=2)


def _install_chunk_reuse(module) -> dict:
    """Apply the off-by-default per-round reuses, never failing a worker.

    The pinned scorer is not edited; chunk_reuse wraps it. If the module is
    missing or an installer raises, the worker keeps today's behaviour.
    """
    try:
        import chunk_reuse
    except ImportError:
        return {}
    try:
        applied = chunk_reuse.install_all(module)
        # Say what took effect. Without this an A/B of the reuses cannot
        # tell "identical because it was never enabled" from "identical
        # because it is faithful", and the first passes every equality test.
        # STDERR, never stdout. In a shard worker this runs BEFORE stdout is
        # separated from the protocol, so a line on stdout is read by the
        # parent as the worker's JSON ready reply and the round dies.
        import sys as _sys
        print("CHUNK_REUSE applied=%r enabled=%r"
              % (applied, chunk_reuse.ENABLED), file=_sys.stderr, flush=True)
        return applied
    except Exception as error:                # noqa: BLE001
        return {"error": "%s: %s" % (type(error).__name__, error)}


def flattened_config(module, config, target: str):
    """What main() does at its line 531, before it builds any shard job.

    _worker takes the FLATTENED mapping BoltzgenWrapper reads, not a path to
    config.yaml. Handing it the path is not an error: the wrapper finds none
    of the keys it wants and every shard returns nothing, which surfaces
    three layers away as "MERGE LOST SEQUENCES: submitted 8, merged 0".
    """
    if not isinstance(config, str):
        return config                      # already flattened
    import yaml as _yaml
    with open(config) as handle:
        return module._flatten_config(_yaml.safe_load(handle), target)


def job_from_command(command, *, observation_id=None) -> dict:
    """Translate a fast_score_batch argv into a daemon job.

    THE COMMAND MUST BE THE INNERMOST SCORER CALL -- the one
    score_boltzgen_fast_adapter builds, whose flags are fast_score_batch's
    own. Two outer CLIs look similar and are not:

        run_live_producer
          -> score_boltzgen_waves      (no --out, --predict-batch-size)
            -> score_boltzgen_fast_adapter  (--work-root, --structure-dir)
              -> fast_score_batch           (--shard-root, --capture-structures)

    Feeding an outer one here raised "daemon job missing ['--out']" inside a
    quiet fallback, so the daemon reported itself enabled and never ran.

    Every default below is fast_score_batch's own argparse default. A guessed
    default would make the daemon compute a DIFFERENT metric than the
    subprocess while both looked healthy -- batch size in particular, whose
    scorer default is 24 and whose composition the scorer's own notes call
    load-bearing.
    """
    import glob as _glob
    import json as _json
    import uuid as _uuid

    flags, bare = {}, set()
    tokens = [str(t) for t in command]
    for index, token in enumerate(tokens):
        if not token.startswith("--"):
            continue
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if following is None or following.startswith("--"):
            bare.add(token)          # store_true, e.g. --require-structures
        else:
            flags[token] = following
    missing = [n for n in ("--sequences", "--config", "--out") if n not in flags]
    if missing:
        raise ValueError("daemon job missing %s" % missing)
    # An outer CLI must fail LOUDLY, not parse into a plausible-looking job.
    # The adapter's own argv also carries --out, so without this it would be
    # accepted and then run with the wrong shard root and no structure dir --
    # a silently different result, which is the worst outcome available.
    outer = sorted(set(flags) & {"--work-root", "--structure-dir",
                                 "--predict-batch-size", "--output-dir",
                                 "--wave-size", "--limit", "--scorer"})
    if outer:
        raise ValueError(
            "not a fast_score_batch argv: %s belong to an outer CLI" % outer)
    with open(flags["--sequences"]) as handle:
        raw_sequences = _json.load(handle)
    # fast_score_batch.py's own main() (--sequences loading, around line 520)
    # accepts {"sequences": [...]} / {"cases": [...]} / a bare list, and each
    # item may be a bare string OR a {"sequence": ...} object (the shape
    # score_boltzgen_waves._wave_command and the live race always write).
    # job_from_command used to hand the raw item list straight through when
    # items were dicts, so the worker's _deduplicate_nanobodies tried to hash
    # a dict and raised inside run_chunk, caught and logged as "continuing"
    # with the sequence silently dropped -- the daemon then answered "ok"
    # with nothing scored, which _merge correctly reports as
    # "MERGE LOST SEQUENCES: submitted N, merged 0". This mirrors
    # fast_score_batch.py's own extraction exactly so the two can never
    # disagree on what one sequence item means. See
    # tests/test_scorer_daemon.py::test_job_from_command_extracts_dict_items.
    _items = (
        raw_sequences if isinstance(raw_sequences, list)
        else (raw_sequences.get("cases") or raw_sequences.get("sequences") or [])
    )
    sequences_list = [
        item["sequence"] if isinstance(item, dict) else item for item in _items
    ]
    sequences_list = [s for s in sequences_list if isinstance(s, str) and s]
    if not sequences_list:
        raise ValueError(
            "no sequences parsed from %s" % flags["--sequences"])

    # main() derives the rotamer library by globbing beside the reference
    # when --synth-library is absent; the worker would otherwise iterate None.
    reference = flags.get("--synth-reference")
    library = None
    if reference:
        library = sorted(_glob.glob(os.path.join(
            os.path.dirname(os.path.abspath(reference)), "*.cif")))
        if not library:
            raise ValueError("no .cif files beside %s" % reference)

    def _int(flag, default):
        return int(flags[flag]) if flag in flags else default

    return {
        # Pass EVERY gpu: taking gpus[0] would run a whole round on one device.
        "gpus": flags.get("--gpus", "0,1"),
        "shards_per_gpu": _int("--shards-per-gpu", 1),
        "sequences": sequences_list,
        "target": flags.get("--target", "P20809"),
        "subnet_config": flags["--config"],
        "out_path": flags["--out"],
        "batch_size": _int("--batch-size", 24),
        "shard_root": flags.get("--shard-root", "/root/bgen_shards_fast"),
        "structure_dir": flags.get("--capture-structures"),
        "require_structures": "--require-structures" in bare,
        "keep_work_dir": "--keep-work-dir" in bare,
        "synth_reference": reference,
        "synth_library": library,
        "fold_matmul": flags.get("--fold-matmul-precision", "high"),
        "fold_steps": _int("--fold-sampling-steps", 50),
        "fold_samples": (int(flags["--fold-diffusion-samples"])
                         if "--fold-diffusion-samples" in flags else None),
        "design_steps": _int("--design-sampling-steps", 100),
        # main(): observation_id = a.observation_id or uuid4().hex
        "observation_id": (observation_id or flags.get("--observation-id")
                           or _uuid.uuid4().hex),
        "allow_replicates": "--allow-replicates" in bare,
        # the launcher the adapter chose, not whatever python holds the daemon
        "python": tokens[0] if tokens and not tokens[0].startswith("--") else None,
    }


def shard_worker_main(scorer_path: str) -> int:
    """Worker entry point: import once, then serve shards forever."""
    import importlib
    import sys as _sys

    directory = os.path.dirname(os.path.abspath(scorer_path))
    if directory not in _sys.path:
        _sys.path.insert(0, directory)
    import resident_scorer
    resident_scorer.bootstrap(scorer_path)
    name = os.path.splitext(os.path.basename(scorer_path))[0]
    module = importlib.import_module(name)
    resident_scorer.install_all(module)
    _install_chunk_reuse(module)
    worker = module._worker

    # The scorer prints progress to stdout. Sharing that channel with the
    # protocol corrupts it: replies interleave with progress bars and the
    # parent reads "JSONDecodeError: Expecting value: line 1 column 3".
    # Keep the ORIGINAL stdout for protocol only and send every other write
    # to stderr, which the parent discards.
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    protocol = os.fdopen(protocol_fd, "w")

    def reply(payload):
        protocol.write(json.dumps(payload) + "\n")
        protocol.flush()

    reply({"ready": True})
    while True:
        # readline(), not "for line in stdin": iteration read-aheads can
        # withhold a line until the buffer fills, which deadlocks a
        # request/response protocol.
        line = _sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            args = list(request["args"])
            args[3] = flattened_config(module, args[3], request.get("target"))
            worker(tuple(args))
            reply({"ok": True})
        except (Exception, SystemExit) as error:
            import traceback as _tb
            reply({"ok": False,
                   "error": "%s: %s" % (type(error).__name__, error),
                   "trace": _tb.format_exc()[-1200:]})
    return 0


# --- server -----------------------------------------------------------------

class Daemon:
    """Holds warm scorer state and serves rounds, serialised PER GPU.

    A single global lock is wrong here, and measurably so. The replication
    path (`_score_repeats`) launches one scorer per GPU AT THE SAME TIME,
    each with --gpus <one device>; under a global lock those four rounds
    queue behind each other and run one device at a time. Measured, that is
    4 x 12 s = 48 s where the subprocess path ran all four together in ~33 s
    -- a REGRESSION on the one path `--replication-rounds` multiplies.

    What actually must not overlap is two rounds on the SAME device: they
    would contend for it and share that device's single worker process. So
    a round takes the lock for each GPU it names, in sorted order so two
    jobs requesting overlapping sets cannot deadlock. Shard directories are
    per-job (--shard-root differs per output dir), so the scorer's warning
    that "two shards in one tree would silently read each other's results"
    is not in play between jobs.
    """

    def __init__(self, scorer_path: str, socket_path: str | None = None):
        self.scorer_path = scorer_path
        self.socket_path = socket_path or SOCKET_PATH
        #: One lock per device, created on demand under _registry_lock.
        self._gpu_locks: dict = {}
        #: Guards _gpu_locks and _workers, never held across a round.
        self._registry_lock = threading.Lock()
        self._worker = None
        self._module = None
        self._workers = {}
        self.served = 0
        self.failed = 0
        self.started = time.time()

    def warm(self) -> None:
        """Import the scorer once and start one resident worker per GPU."""
        import importlib
        import os
        import sys

        import resident_scorer
        resident_scorer.bootstrap(self.scorer_path)
        # Import by its REAL module name. The scorer spawns children with mp
        # "spawn", which pickles <module>._analysis_child BY NAME; a module
        # registered under a synthetic name does not exist in the child.
        directory = os.path.dirname(os.path.abspath(self.scorer_path))
        if directory not in sys.path:
            sys.path.insert(0, directory)
        self.module_name = os.path.splitext(os.path.basename(self.scorer_path))[0]
        module = importlib.import_module(self.module_name)
        resident_scorer.install_all(module)
        _install_chunk_reuse(module)
        self._module = module
        self._worker = module._worker
        self._workers = {}

    def start_workers(self, gpus, python=None, slots: int = 1) -> dict:
        """`slots` pinned processes per GPU (default one), each importing and building once.

        Guarded: rounds now run concurrently on disjoint devices, so two of
        them can reach this at the same moment. Unguarded, both would see no
        worker for a GPU and each spawn one, and the loser would be dropped
        from the dict still holding its share of device memory.
        """
        import concurrent.futures
        with self._registry_lock:
            missing = []
            for gpu in gpus:
                gpu = str(gpu)
                for slot in range(max(1, int(slots))):
                    existing = self._workers.get(_worker_key(gpu, slot))
                    if existing is not None and existing.alive():
                        continue
                    missing.append((gpu, slot))
            if missing:
                # Each worker blocks on its own import/build handshake; start them together so a
                # cold start costs one build, not N.
                failure = None
                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=len(missing)) as pool:
                    pending = [(gpu, slot, pool.submit(
                        ShardWorker, gpu, self.scorer_path, python, slot))
                        for gpu, slot in missing]
                    for gpu, slot, future in pending:
                        try:
                            # Register every worker that did start, so a failed sibling never
                            # leaves one running and untracked.
                            self._workers[_worker_key(gpu, slot)] = future.result()
                        except Exception as error:   # noqa: BLE001
                            failure = failure or error
                if failure is not None:
                    raise failure
            return {"workers": sorted(self._workers)}

    def _locks_for(self, gpus):
        """Locks for these devices, always in a fixed global order.

        Sorted, so a job wanting {0,1} and one wanting {1,0} cannot each
        hold one and wait for the other.
        """
        with self._registry_lock:
            for gpu in gpus:
                self._gpu_locks.setdefault(str(gpu), threading.Lock())
        return [self._gpu_locks[g] for g in sorted(set(str(g) for g in gpus))]

    def _shard(self, sequences, gpus, shards_per_gpu):
        """Round-robin sequences into shards, exactly as the scorer's main().

        Reproducing this rather than inventing a split matters: the merge
        below reassembles per-shard files and refuses a partial result, so a
        different split would surface as MERGE LOST SEQUENCES.
        """
        count = max(1, len(gpus) * max(1, int(shards_per_gpu)))
        buckets = [[] for _ in range(count)]
        for index, seq in enumerate(sequences):
            buckets[index % count].append(seq)
        return buckets

    def _merge(self, work, expected, out_path, job):
        """Merge at the SEQUENCE level and refuse a partial result.

        Every shard writes uid 0, so a plain dict.update() drops whole shards
        -- this mirrors the scorer's own merge, including its refusal. When the
        refusal fires it now names WHICH shard(s) came back short, so a lost
        sequence is a diagnosis instead of a bare count mismatch.
        """
        import json as _json
        merged: dict = {}
        structures: dict = {}
        short = []
        for index, gpu, bucket, out in work:
            if not os.path.exists(out):
                raise RuntimeError("shard did not write %s" % out)
            data = _json.load(open(out))
            got = sum(len(v) for v in data.get("components", {}).values())
            if got < len(bucket):
                short.append("shard %d produced %d/%d"
                             % (index, got, len(bucket)))
            for uid, by_seq in data["components"].items():
                merged.setdefault(uid, {}).update(by_seq)
            structures.update(data.get("structures") or {})
        produced = sum(len(v) for v in merged.values())
        if produced != expected:
            detail = "; ".join(short) if short else "no individual shard identified"
            raise RuntimeError(
                "MERGE LOST SEQUENCES: submitted %d, merged %d. Short shards: %s"
                % (expected, produced, detail))
        # The live producer always passes --require-structures. The scorer
        # refuses a result the pipeline would reject; without the same
        # refusal the daemon writes a wave that _validated_wave throws out
        # AFTER every shard has been paid for.
        if job.get("require_structures") and len(structures) != expected:
            raise RuntimeError(
                "structures captured for %d of %d sequences"
                % (len(structures), expected))
        # The scorer's own _worker does exactly this before importing the
        # package ("if '/root' not in sys.path"). The daemon merges outside
        # that function, so it must make the same guarantee or the round dies
        # after every shard has already been scored.
        import sys as _sys
        for candidate in (os.environ.get("NANOBODY_PACKAGE_ROOT"), "/root"):
            if candidate and candidate not in _sys.path:
                _sys.path.insert(0, candidate)
        from nanobody_opt.diffusion_sample_selection import SCORER_PROTOCOL
        with open(out_path, "w") as handle:
            # KEY NAMES ARE NOT INTERCHANGEABLE. The scorer writes
            # "components" in a PER-SHARD file and "per_nanobody_components"
            # in the merged --out file, and every consumer (_components in
            # score_boltzgen_waves, rank_exact, import_exact_screens) reads
            # only the latter. Writing the shard key here produced a document
            # those readers saw as EMPTY -- no error, just zero sequences.
            # default=str matters too: metric values arrive as numpy scalars.
            _json.dump({"target": job["target"],
                        "n": expected, "wall_seconds": job.get("_wall", 0.0),
                        "scorer_protocol": SCORER_PROTOCOL,
                        "observation_id": job["observation_id"],
                        "diagnostic_seed": None,
                        "predict_batch_size": (1 if job.get("allow_replicates") else job["batch_size"]),
                        "per_nanobody_components": merged,
                        "structures": structures},
                       handle, indent=2, default=str)

    @staticmethod
    def _shard_args(index, gpu, bucket, out, job, root):
        """The scorer _worker's positional args for one shard.

        Single source of truth for both the first dispatch and the retry, so
        the two can never hand the pinned worker different arguments.
        """
        args = [index, gpu, bucket, job["subnet_config"], out,
                job["batch_size"], job.get("keep_work_dir", False),
                root, job.get("fold_matmul", "high"),
                job.get("fold_steps", 50), job.get("fold_samples"),
                job.get("design_steps", 100),
                job.get("synth_reference"), job.get("synth_library"),
                job.get("structure_dir"), job["observation_id"]]
        if job.get("worker_argument_schema", "base-v1") == "replicates-v1":
            args.extend([bool(job.get("allow_replicates")),
                         list(range(index, len(job["sequences"]), job["_shard_count"]))])
        return args

    def _rerun_short_shards(self, work, job, root, gpus, slots, cancelled=None) -> None:
        """Re-run once every shard that came back short.

        A shard worker replies "ok" while its out file holds fewer components
        than its bucket: the pinned scorer's collection can return EMPTY
        WITHOUT raising, so the protocol sees success and only the file shows
        the loss. Re-running only the short shards (overwriting their out
        files) turns one transient GPU hiccup into ~one shard re-fold instead
        of a whole-epoch abort; _merge's count guard remains the last line of
        defense. Idempotent: same bucket, same paths, and _merge reads each
        file once, so nothing is double-counted.
        """
        import json as _json

        def produced(out):
            try:
                with open(out) as handle:
                    data = _json.load(handle)
                return sum(len(v) for v in data.get("components", {}).values())
            except Exception:
                return 0

        short = [(index, gpu, bucket, out)
                 for index, gpu, bucket, out in work
                 if produced(out) < len(bucket)]
        if not short:
            return
        import sys as _sys
        print("RETRY_SHORT_SHARDS " + "; ".join(
            "shard %d produced %d/%d" % (index, produced(out), len(bucket))
            for index, gpu, bucket, out in short), file=_sys.stderr, flush=True)
        for index, gpu, bucket, out in short:
            worker = self._workers.get(
                _worker_key(gpu, (index // len(gpus)) % slots))
            if worker is None:
                print("RETRY_SHARD_FAILED shard %d: no worker for gpu %s"
                      % (index, gpu), file=_sys.stderr, flush=True)
                continue
            try:
                reply = worker.run(
                    self._shard_args(index, gpu, bucket, out, job, root),
                    job["target"], cancelled)
                if not reply.get("ok"):
                    print("RETRY_SHARD_FAILED shard %d: %s"
                          % (index, reply.get("error")),
                          file=_sys.stderr, flush=True)
            except Exception as error:
                print("RETRY_SHARD_FAILED shard %d: %s: %s"
                      % (index, type(error).__name__, error),
                      file=_sys.stderr, flush=True)

    def handle(self, job: dict, cancelled=None) -> dict:
        # Dispatch needs pinned WORKERS, not an in-process _worker. Guarding
        # on the latter was left over from the thread design, which is the
        # thing that made shards serialise onto one GPU.
        if self._module is None and not self._workers:
            raise RuntimeError("daemon not warmed")
        # Fail here, not 40 s later inside the merge. A job missing "target"
        # surfaced as a bare KeyError after every shard had already run.
        missing = [k for k in ("target", "sequences", "gpus", "out_path",
                               "observation_id", "batch_size", "subnet_config")
                   if k not in job]
        if missing:
            raise RuntimeError("daemon job missing %s" % missing)
        import concurrent.futures

        gpus = [g for g in str(job["gpus"]).split(",") if g]
        schema = job.get("worker_argument_schema", "base-v1")
        if schema not in {"base-v1", "replicates-v1"}:
            raise RuntimeError("unknown scorer worker argument schema")
        if job.get("allow_replicates") and schema != "replicates-v1":
            raise RuntimeError("replicate job requires the extended worker schema")
        requested_shards = int(job.get("shards_per_gpu", 1))
        sequences = job["sequences"]
        shards_per_gpu = (requested_shards if schema == "replicates-v1" else
                          effective_shards_per_gpu(requested_shards, len(sequences), len(gpus)))
        if shards_per_gpu != requested_shards:
            import sys as _sys
            print("WORKERS_PER_GPU_APPLIED shards_per_gpu=%d->%d sequences=%d gpus=%d"
                  % (requested_shards, shards_per_gpu, len(sequences), len(gpus)),
                  file=_sys.stderr, flush=True)
        buckets = self._shard(sequences, gpus, shards_per_gpu)
        job["_shard_count"] = len(buckets)
        root = job["shard_root"]
        os.makedirs(root, exist_ok=True)
        # Start or replace any worker this round needs. A dead worker is
        # replaced rather than reused.
        # Workers per device are capped by what the round can use AND by the operator's setting;
        # with the default of 1 every shard of a device queues on its single worker, as before.
        slots = max(1, min(workers_per_gpu(), shards_per_gpu))
        self.start_workers(gpus, job.get("python"), slots)

        import contextlib
        with contextlib.ExitStack() as stack:
            for lock in self._locks_for(gpus):
                stack.enter_context(lock)
            if cancelled is not None and cancelled():
                raise RuntimeError("scoring client disconnected before dispatch")
            started = time.time()
            work = []
            for index, bucket in enumerate(buckets):
                if not bucket:
                    continue          # an empty shard writes nothing, as today
                out = os.path.join(root, "out%d.json" % index)
                work.append((index, gpus[index % len(gpus)], bucket, out))
            # Dispatch to the PINNED worker process for each GPU. Threads
            # cannot host _worker: CUDA_VISIBLE_DEVICES is process-global.
            errors = []
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max(1, len(work))) as pool:
                futures = {}
                for index, gpu, bucket, out in work:
                    args = self._shard_args(index, gpu, bucket, out, job, root)
                    # Shard `index` sits on gpus[index % len(gpus)]; its position among that
                    # device's shards picks the worker.
                    worker = self._workers.get(
                        _worker_key(gpu, (index // len(gpus)) % slots))
                    if worker is None:
                        errors.append("shard %d: no worker for gpu %s" % (index, gpu))
                        continue
                    # The thread only blocks on a pipe; the work happens in
                    # the pinned process on the other end.
                    futures[pool.submit(worker.run, args, job["target"], cancelled)] = (index, out)
                for future in concurrent.futures.as_completed(futures):
                    index, _out = futures[future]
                    try:
                        reply = future.result()
                        if not reply.get("ok"):
                            errors.append("shard %d: %s" % (index, reply.get("error")))
                    except Exception as error:
                        errors.append("shard %d: %s: %s"
                                      % (index, type(error).__name__, error))
            if errors:
                raise RuntimeError("; ".join(errors))
            # Re-run any shard that came back short before the merge guard is
            # allowed to abort: one transient GPU hiccup costs ~one shard
            # re-fold, not a whole epoch.
            self._rerun_short_shards(work, job, root, gpus, slots, cancelled)
            wall = time.time() - started
            job["_wall"] = round(wall, 2)
            self._merge(work, len(sequences), job["out_path"], job)
            return {"ok": True, "seconds": round(wall, 2),
                    "shards": len(work), "gpus": len(gpus),
                    "out_path": job["out_path"]}

    def stats(self) -> dict:
        return {"ok": True, "served": self.served, "failed": self.failed,
                "uptime_s": round(time.time() - self.started, 1),
                "warm": self._worker is not None}

    #: AF_UNIX paths are capped by the kernel (104 bytes on macOS, 108 on
    #: Linux). A long path fails inside bind() with a bare OSError that names
    #: neither the limit nor the path.
    MAX_SOCKET_PATH = 100

    def serve_forever(self) -> None:
        if len(self.socket_path.encode()) > self.MAX_SOCKET_PATH:
            raise ValueError(
                "socket path is %d bytes, over the %d-byte AF_UNIX limit: %s"
                % (len(self.socket_path.encode()), self.MAX_SOCKET_PATH,
                   self.socket_path))
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        server.listen(8)
        print("SCORER_DAEMON_READY socket=%s" % self.socket_path, flush=True)
        try:
            while True:
                conn, _ = server.accept()
                threading.Thread(target=self._session, args=(conn,),
                                 daemon=True).start()
        finally:
            server.close()
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

    def _session(self, conn) -> None:
        with conn:
            conn.settimeout(REQUEST_TIMEOUT)
            try:
                request = _recv(conn)
                if not request:
                    return
                op = request.get("op")
                if op == "ping":
                    _send(conn, {"ok": True})
                elif op == "stats":
                    _send(conn, self.stats())
                elif op == "score":
                    try:
                        reply = self.handle(request["job"], lambda: _client_disconnected(conn))
                        self.served += 1
                    except Exception as error:
                        self.failed += 1
                        # The message alone ("BrokenPipeError") names neither
                        # the frame nor the cause; a round that fails must say
                        # where, or every diagnosis is a guess.
                        import traceback as _tb
                        reply = {"ok": False,
                                 "error": "%s: %s" % (type(error).__name__, error),
                                 "trace": _tb.format_exc()[-1800:]}
                    _send(conn, reply)
                else:
                    _send(conn, {"ok": False, "error": "unknown op %r" % op})
            except Exception:
                pass          # a bad session must never take the daemon down


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--socket", default=SOCKET_PATH)
    parser.add_argument("--shard-worker", action="store_true",
                        help="internal: run as a GPU-pinned shard worker")
    parser.add_argument("--parent-pid", type=int, help="internal: owning daemon PID")
    args = parser.parse_args(argv)
    if args.shard_worker:
        _watch_parent(args.parent_pid if args.parent_pid is not None else os.getppid())
        return shard_worker_main(args.scorer)
    daemon = Daemon(args.scorer, args.socket)
    daemon.warm()
    daemon.serve_forever()
    return 0


if __name__ == "__main__":
    # The scorer spawns children with mp "spawn", which re-imports this file by
    # path; without the guard every child would start another daemon.
    raise SystemExit(main())

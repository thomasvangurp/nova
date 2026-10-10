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

class ShardWorker:
    """One persistent process pinned to one GPU.

    MUST be a process, not a thread. The scorer's `_worker` begins with

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    which is process-global. Four shards in four threads race on that single
    variable and all land on whichever device was assigned last: a
    16-candidate round measured 62.9 s as four processes and 164 s as four
    threads -- 2.6x SLOWER, because the shards serialised onto one GPU.
    """

    def __init__(self, gpu: str, scorer_path: str, python: str | None = None):
        import subprocess
        self.gpu = str(gpu)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = self.gpu   # pinned once, privately
        self.proc = subprocess.Popen(
            [python or __import__("sys").executable, os.path.abspath(__file__),
             "--shard-worker", "--scorer", scorer_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            # To a FILE, never DEVNULL: a worker that dies takes its reason
            # with it and the parent sees only an empty read.
            stderr=open(os.path.join(
                os.environ.get("NANOBODY_WORKER_LOG_DIR", "/tmp"),
                "shard_worker_gpu%s.log" % self.gpu), "w"),
            env=env, text=True, bufsize=1)
        #: Serialises request/reply on this worker's pipe.
        self._pipe = threading.Lock()
        line = self.proc.stdout.readline()
        try:
            hello = json.loads(line)
        except Exception:
            hello = {}
        if not hello.get("ready"):
            raise RuntimeError("shard worker on gpu %s failed to start: %r"
                               % (self.gpu, line[:200]))

    def alive(self) -> bool:
        return self.proc.poll() is None

    def run(self, args: list, target: str | None = None) -> dict:
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
            self.proc.stdin.write(
                json.dumps({"args": args, "target": target}) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("shard worker on gpu %s closed" % self.gpu)
            return json.loads(line)

    def close(self) -> None:
        try:
            if self.alive():
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
        except Exception:
            try: self.proc.kill()
            except Exception: pass


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
        except Exception as error:
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

    def start_workers(self, gpus, python=None) -> dict:
        """One pinned process per GPU, each importing and building once.

        Guarded: rounds now run concurrently on disjoint devices, so two of
        them can reach this at the same moment. Unguarded, both would see no
        worker for a GPU and each spawn one, and the loser would be dropped
        from the dict still holding its share of device memory.
        """
        with self._registry_lock:
            for gpu in gpus:
                gpu = str(gpu)
                existing = self._workers.get(gpu)
                if existing is not None and existing.alive():
                    continue
                self._workers[gpu] = ShardWorker(gpu, self.scorer_path, python)
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

    def _merge(self, shard_paths, expected, out_path, job):
        """Merge at the SEQUENCE level and refuse a partial result.

        Every shard writes uid 0, so a plain dict.update() drops whole shards
        -- this mirrors the scorer's own merge, including its refusal.
        """
        import json as _json
        merged: dict = {}
        structures: dict = {}
        for path in shard_paths:
            if not os.path.exists(path):
                raise RuntimeError("shard did not write %s" % path)
            data = _json.load(open(path))
            for uid, by_seq in data["components"].items():
                merged.setdefault(uid, {}).update(by_seq)
            structures.update(data.get("structures") or {})
        produced = sum(len(v) for v in merged.values())
        if produced != expected:
            raise RuntimeError(
                "MERGE LOST SEQUENCES: submitted %d, merged %d"
                % (expected, produced))
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
                        "predict_batch_size": job["batch_size"],
                        "per_nanobody_components": merged,
                        "structures": structures},
                       handle, indent=2, default=str)

    def handle(self, job: dict) -> dict:
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
        shards_per_gpu = int(job.get("shards_per_gpu", 1))
        sequences = job["sequences"]
        buckets = self._shard(sequences, gpus, shards_per_gpu)
        root = job["shard_root"]
        os.makedirs(root, exist_ok=True)
        # Start or replace any worker this round needs. A dead worker is
        # replaced rather than reused.
        self.start_workers(gpus, job.get("python"))

        import contextlib
        with contextlib.ExitStack() as stack:
            for lock in self._locks_for(gpus):
                stack.enter_context(lock)
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
                    args = [index, gpu, bucket, job["subnet_config"], out,
                            job["batch_size"], job.get("keep_work_dir", False),
                            root, job.get("fold_matmul", "high"),
                            job.get("fold_steps", 50), job.get("fold_samples"),
                            job.get("design_steps", 100),
                            job.get("synth_reference"), job.get("synth_library"),
                            job.get("structure_dir"), job["observation_id"]]
                    worker = self._workers.get(gpu)
                    if worker is None:
                        errors.append("shard %d: no worker for gpu %s" % (index, gpu))
                        continue
                    # The thread only blocks on a pipe; the work happens in
                    # the pinned process on the other end.
                    futures[pool.submit(worker.run, args, job["target"])] = (index, out)
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
            wall = time.time() - started
            job["_wall"] = round(wall, 2)
            self._merge([out for _i, _g, _b, out in work], len(sequences),
                        job["out_path"], job)
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
                        reply = self.handle(request["job"])
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
    args = parser.parse_args(argv)
    if args.shard_worker:
        return shard_worker_main(args.scorer)
    daemon = Daemon(args.scorer, args.socket)
    daemon.warm()
    daemon.serve_forever()
    return 0


if __name__ == "__main__":
    # The scorer spawns children with mp "spawn", which re-imports this file by
    # path; without the guard every child would start another daemon.
    raise SystemExit(main())

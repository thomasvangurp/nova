"""A long-lived TNP (developability) daemon the producer talks to instead of a
subprocess per sequence.

WHY.  ``filter_layer2._developability`` runs the metanano DevelopabilityService,
whose ``TNPWrapper.profile`` shells out to ``TNP --seq`` per sequence.  TNP then
shells out AGAIN to ``NanoBodyBuilder2 --heavy_sequence``, so one sequence pays
two process starts, one `import torch` + `import ImmuneBuilder` + `import
openmm`, a 700 MB 4-model weight load, a GPU forward pass (~0.34 s), an ANARCI
``hmmscan`` subprocess, an OpenMM refinement (~27 s at the ``--n_threads 1``
the validator uses) and the three theraprofnano metric stages.

Measured on the live box (metanano env, 8x NVIDIA, 224 cores):

    model load (once)            3.3 s
    predict (forward)            0.34 s / sequence
    refine  n_threads=1          27.7 s / sequence     <- dominant
    refine  n_threads=8           4.0 s / sequence     (different structure -> rejected)

    live producer: ~1.8-2.7 seq/s, GPU ~0%, CPU ~13%, subprocesses in D-state

The producer's subprocesses sit in D-state because ~15+ of them read the same
700 MB of weights off disk at once.  The GPU is irrelevant: TNP's cost is the
OpenMM refinement and the repeated model load, not the forward pass, so the
BoltzGen "batch the forward on 8 GPUs" shape does not transfer directly --
NanoBodyBuilder2 has no batch dimension and would not be the bottleneck if it
did.  The load-once half of the BoltzGen pattern transfers exactly.

WHAT THIS DOES.  Loads NanoBodyBuilder2 once PER WORKER PROCESS, keeps the
workers resident, and serves ``profile_batch`` over a unix socket.  Each
sequence is still the exact TNP pipeline -- ANARCI CDR numbering, the same
4-model predict, the same ``n_threads=1`` OpenMM refinement, hydrogen strip,
CDR3 compactness, PSH/PPC/PNC and the same six-flag red/amber/green
thresholds -- so the returned profile is metric-identical to the subprocess
path.  The only thing that moves is the process boundary, plus the refinement
runs concurrently across worker PROCESSES (OpenMM + torch in threads starve
each other under the GIL; processes do not).

SAFETY.  ``filter_layer2`` keeps its subprocess path.  If the socket is absent,
the daemon is unhealthy, or a batch fails, it falls back to the metanano
DevelopabilityService exactly as it does today, so the worst case is current
behaviour.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time

#: Producer-side switch.  Off means the current subprocess path, unchanged.
ENABLED = os.environ.get("NANOBODY_TNP_DAEMON", "0") == "1"
SOCKET_PATH = os.environ.get("NANOBODY_TNP_SOCKET",
                             "/root/artifacts/nanobody_tnp.sock")
#: Number of worker PROCESSES, one model copy each.  Each sequence's refinement
#: is a single-threaded OpenMM job, so this is the real throughput lever, not
#: GPU count.  64 x 700 MB = ~45 GB RAM; the box has 365 GB free.
WORKERS = int(os.environ.get("NANOBODY_TNP_WORKERS", "64"))
#: One sequence never takes this long; the cap exists so a wedged worker cannot
#: block a producer forever.  It sits well above any real batch.
REQUEST_TIMEOUT = float(os.environ.get("NANOBODY_TNP_TIMEOUT", "3600"))


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
    if length > (256 << 20):            # a profile batch is small; refuse absurd
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


def profile_batch(sequences: list[str], path: str | None = None,
                  timeout: float | None = None) -> list[dict]:
    """Profile ``sequences`` on the daemon, preserving order.

    Raises on any failure so the caller falls back to the subprocess path
    rather than silently losing a developability result.
    """
    target = path or SOCKET_PATH
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout or REQUEST_TIMEOUT)
        sock.connect(target)
        _send(sock, {"op": "profile", "sequences": list(sequences)})
        reply = _recv(sock)
    if not reply:
        raise RuntimeError("tnp daemon closed the connection")
    if not reply.get("ok"):
        raise RuntimeError("tnp daemon error: %s" % reply.get("error"))
    profiles = reply.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != len(sequences):
        raise RuntimeError("tnp daemon returned %r profiles for %d sequences"
                           % (type(profiles).__name__, len(sequences)))
    return profiles


# --- TNP core, in-process, metric-identical to the TNP CLI -------------------

def _assign_flag(metric: str, value) -> str:
    if metric == "L":
        if value < 20 or value > 39:
            return "red"
        return "amber" if (20 <= value <= 24 or 37 <= value <= 39) else "green"
    if metric == "L3":
        if value < 5 or value > 23:
            return "red"
        return "amber" if (5 <= value <= 8 or 21 <= value <= 23) else "green"
    if metric == "C":
        if value < 0.56 or value > 1.61:
            return "red"
        return "amber" if (0.56 <= value <= 0.81 or 1.57 <= value <= 1.61) else "green"
    if metric == "PSH":
        if value < 73.40 or value > 155.47:
            return "red"
        return "amber" if (73.40 <= value <= 79.59 or 126.83 <= value <= 155.47) else "green"
    if metric == "PPC":
        if value > 1.18:
            return "red"
        return "amber" if 0.39 <= value <= 1.18 else "green"
    if metric == "PNC":
        if value > 1.88:
            return "red"
        return "amber" if 1.47 <= value <= 1.88 else "green"
    raise ValueError("unknown TNP metric %r" % metric)


def _strip_hydrogens(input_pdb: str) -> None:
    """Remove hydrogens from a NanoBodyBuilder2 model in place (TNP's own
    pdb_remove_hydrogens): refinement writes a hydrogenated model, and the
    surface/compactness stages must read the de-hydrogenated one."""
    import re
    from Bio.PDB import PDBIO, PDBParser, Select

    _hydrogen = re.compile("[123 ]*H.*")

    class _RemoveHydrogen(Select):
        def accept_atom(self, atom):
            name = atom.get_id()
            return 0 if _hydrogen.match(name) else 1

    structure = PDBParser(QUIET=True).get_structure("nb", input_pdb)
    io = PDBIO()
    io.set_structure(structure)
    io.save(input_pdb, _RemoveHydrogen())


class _TNPProfiler:
    """Holds a loaded model + metric functions, profiles one sequence."""

    def __init__(self, tmp_root: str | None = None, *, structure_scorer=None, stage_observer=None):
        # Optional research hook runs while the exact refined heavy-atom model
        # still exists. The socket daemon does not enable it.
        self._structure_scorer = structure_scorer
        self._stage_observer = stage_observer
        import torch
        # The subprocess path runs NanoBodyBuilder2 --n_threads 1, which is
        # torch.set_num_threads(1).  Match it: the forward must be the same
        # single-threaded computation (and the metanano env already sets
        # OMP_NUM_THREADS=1, so this only pins the torch-side threads).
        torch.set_num_threads(1)
        from ImmuneBuilder.NanoBodyBuilder2 import NanoBodyBuilder2
        self.builder = NanoBodyBuilder2()
        from theraprofnano.CDR_Profiler.CDR_Assigner import main as _region
        from theraprofnano.CDR_Profiler.CDR3_Conf_Assigner import main_compactness as _compactness  # noqa: E501
        from theraprofnano.Hydrophobicity_and_Charge_Profiler.Hydrophobicity_and_Charge_Assigner import CreateAnnotation as _hydrophob  # noqa: E501
        self._region = _region
        self._compactness = _compactness
        self._hydrophob = _hydrophob
        self._tmp_root = tmp_root

    def profile_one(self, sequence: str, index: int) -> dict:
        import tempfile

        def stage(name):
            callback = getattr(self, '_stage_observer', None)
            if callback is not None:
                callback(name)

        name = "seq_%08d" % index
        aggregation = None
        with tempfile.TemporaryDirectory(dir=self._tmp_root) as tmpdir:
            # 1. ANARCI CDR numbering -> L, L3 (pure sequence, no structure).
            try:
                stage('anarci')
                _primary, length_dict = self._region(name, sequence, "H",
                                                     tmpdir, verbose=False)
            except Exception as error:
                return {"passed": False,
                        "reason": "Error: ANARCI numbering failed: %s" % error}
            total_cdr = (int(length_dict["imgt"]["cdrh1"])
                         + int(length_dict["imgt"]["cdrh2"])
                         + int(length_dict["imgt"]["cdrh3"]))
            cdr3_len = int(length_dict["imgt"]["cdrh3"])

            # 2. NanoBodyBuilder2 structure: predict + the SAME n_threads=1
            #    OpenMM refinement the TNP CLI runs.
            try:
                stage('predict')
                nanobody = self.builder.predict({"H": sequence})
                pdb_path = os.path.join(tmpdir, "model.pdb")
                stage('refine')
                nanobody.save(pdb_path, check_for_strained_bonds=True,
                              n_threads=1)
                stage('strip')
                _strip_hydrogens(pdb_path)
            except Exception as error:
                return {"passed": False,
                        "reason": "Error: structure failed: %s" % error}

            # 3. CDR3 compactness -> C = L3 / rho.
            try:
                stage('compactness')
                rho = self._compactness(pdb_path, "imgt", verbose=False)
                compactness = float(cdr3_len / rho) if rho is not None else float("nan")
            except Exception:
                compactness = float("nan")

            # 4. PSH / PPC / PNC from the refined, de-hydrogenated model.
            try:
                stage('hydrophob')
                hc = self._hydrophob(0, 7.4, pdb_path, "IG", "imgt",
                                     verbose=False)
                psh = float(hc[0]["Patch_Hydrophob_CDR"])
                ppc = float(hc[0]["Patch_Pos_Charge_CDR"])
                pnc = float(hc[0]["Patch_Neg_Charge_CDR"])
            except Exception as error:
                return {"passed": False,
                        "reason": "Error: surface patch calc failed: %s" % error}

            if self._structure_scorer is not None:
                stage('aggrescan')
                aggregation = self._structure_scorer(sequence, pdb_path)

        flags = {
            "L": _assign_flag("L", total_cdr),
            "L3": _assign_flag("L3", cdr3_len),
            "C": _assign_flag("C", compactness),
            "PSH": _assign_flag("PSH", psh),
            "PPC": _assign_flag("PPC", ppc),
            "PNC": _assign_flag("PNC", pnc),
        }
        passed = all(colour != "red" for colour in flags.values())
        red_flags = [metric for metric, colour in flags.items()
                     if colour == "red"]
        result = {
            "passed": passed,
            "total_cdr_length": total_cdr,
            "cdr3_length": cdr3_len,
            "cdr3_compactness": compactness,
            "surface_hydrophobic_patches": psh,
            "positive_charge_patches": ppc,
            "negative_charge_patches": pnc,
            "flags": flags,
        }
        if aggregation is not None:
            result["aggrescan3d"] = aggregation
        if red_flags:
            result["red_flags"] = red_flags
            result["reason"] = "; ".join(red_flags)
        return result


# --- process-pool workers ----------------------------------------------------

_WORKER_PROFILER: "_TNPProfiler | None" = None


def _init_worker(tmp_root: str) -> None:
    global _WORKER_PROFILER
    _WORKER_PROFILER = _TNPProfiler(tmp_root)


def _profile_worker(seq_index: tuple) -> dict:
    index, sequence = seq_index
    return _WORKER_PROFILER.profile_one(sequence, index)


# --- server -----------------------------------------------------------------

class Daemon:
    """Holds a pool of warm worker processes and serves batches in parallel."""

    def __init__(self, socket_path: str | None = None, tmp_root: str | None = None):
        self.socket_path = socket_path or SOCKET_PATH
        self.tmp_root = tmp_root or os.environ.get(
            "NANOBODY_TNP_TMP", "/tmp")
        self._pool = None
        self.served = 0
        self.failed = 0
        self.started = time.time()

    def warm(self) -> None:
        import multiprocessing as mp
        import concurrent.futures
        ctx = mp.get_context("spawn")
        self._pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=WORKERS,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(self.tmp_root,),
        )

    def handle(self, sequences: list[str]) -> list[dict]:
        if self._pool is None:
            raise RuntimeError("tnp daemon not warmed")
        # map preserves order; each item runs predict+refine+metrics in its own
        # worker process, so OpenMM never shares the GIL with another sequence.
        return list(self._pool.map(
            _profile_worker, list(enumerate(sequences))))

    def stats(self) -> dict:
        return {"ok": True, "served": self.served, "failed": self.failed,
                "uptime_s": round(time.time() - self.started, 1),
                "warm": self._pool is not None, "workers": WORKERS}

    MAX_SOCKET_PATH = 100

    def serve_forever(self) -> None:
        if len(self.socket_path.encode()) > self.MAX_SOCKET_PATH:
            raise ValueError("socket path is %d bytes, over the %d-byte "
                             "AF_UNIX limit: %s"
                             % (len(self.socket_path.encode()), self.socket_path))
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        server.listen(8)
        print("TNP_DAEMON_READY socket=%s workers=%d"
              % (self.socket_path, WORKERS), flush=True)
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
                elif op == "profile":
                    try:
                        profiles = self.handle(request["sequences"])
                        self.served += len(profiles)
                        reply = {"ok": True, "profiles": profiles}
                    except Exception as error:
                        self.failed += 1
                        import traceback as _tb
                        reply = {"ok": False,
                                 "error": "%s: %s"
                                          % (type(error).__name__, error),
                                 "trace": _tb.format_exc()[-1800:]}
                    _send(conn, reply)
                else:
                    _send(conn, {"ok": False, "error": "unknown op %r" % op})
            except Exception:
                pass          # a bad session must never take the daemon down


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=SOCKET_PATH)
    args = parser.parse_args(argv)
    daemon = Daemon(args.socket)
    print("TNP_DAEMON_WARMING workers=%d ..." % WORKERS, flush=True)
    daemon.warm()
    daemon.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

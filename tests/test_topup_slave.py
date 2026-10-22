"""Stdlib-only tests for the slave top-up worker: a fake slave, a fake chain, a fake database.

What must hold: the right candidates are chosen (published ones first, from the database counts), the
slave is asked for exactly the master's scoring call, a pulled round is validated and moved into
``live/epoch*`` in ONE rename (the publisher imports whatever it finds there), any failure only skips a
pass, and the measurement parameters follow the master's settings, never the replay's.
"""
import argparse
import ast
import hashlib
import json
import math
import os
import shlex
import shutil
import sqlite3
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tools import topup_slave as ts

ROOT = Path(__file__).resolve().parents[1]
LENGTH = 127


def seq(i: int) -> str:
    return ("ACDEFGHIKLMNPQRSTVWY"[i % 20] + "A" * (LENGTH - 2) + "ACDEFGHIKLMNPQRSTVWY"[(i // 20) % 20])[:LENGTH]


def metrics_for(i: int) -> dict:
    return {name: float(i % 7) + 0.5 for name in ts.REQUIRED_METRICS}


class Sandbox:
    """A fake master (the module's paths point into a temp dir) plus a fake slave file system."""

    def __init__(self, test: unittest.TestCase):
        self.tmp = Path(tempfile.mkdtemp())
        test.addCleanup(shutil.rmtree, self.tmp, True)
        self.artifacts = self.tmp / "artifacts"
        self.live = self.tmp / "live"
        self.remote = self.tmp / "slave_fs"
        self.artifacts.mkdir()
        self.live.mkdir()
        self.remote.mkdir()
        self.db = self.artifacts / "db.sqlite"
        patches = {
            "ARTIFACTS": self.artifacts, "LIVE_ROOT": self.live, "DB": self.db,
            "CHAIN": self.artifacts / "chain_state.json", "LOG": self.artifacts / "topup_slave.log",
            "LOCK": self.artifacts / "topup_slave.lock", "ARTIFACT": self.artifacts / "nanobody_result.json",
            "MASTER_MARKER": self.artifacts / "topup_running.json",
            "MASTER_SETTINGS": self.artifacts / "nanobody_fast_scorer.json",
            "NANOBODY_OPT": self.tmp / "nanobody_opt",
        }
        for name, value in patches.items():
            patcher = mock.patch.object(ts, name, value)
            patcher.start()
            test.addCleanup(patcher.stop)
        ts._RANKING_CACHE.clear()
        (self.tmp / "nanobody_opt").mkdir()
        self.set_chain(25700, 100)
        connection = sqlite3.connect(self.db)
        connection.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, target TEXT, sequence TEXT)")
        connection.execute("CREATE TABLE observations (observation_id TEXT PRIMARY KEY, node_id INTEGER, "
                           "source TEXT)")
        connection.commit()
        connection.close()

    def set_chain(self, epoch: int, offset: int) -> None:
        (self.artifacts / "chain_state.json").write_text(json.dumps(
            {"epoch": epoch, "block": epoch * ts.EPOCH_LENGTH + offset}))

    def add_observations(self, sequence: str, count: int, source: str = "local_exact_raw") -> None:
        connection = sqlite3.connect(self.db)
        row = connection.execute("SELECT id FROM nodes WHERE sequence=?", (sequence,)).fetchone()
        if row is None:
            node = connection.execute("INSERT INTO nodes (target, sequence) VALUES (?, ?)",
                                      (ts.TARGET, sequence)).lastrowid
        else:
            node = row[0]
        existing = connection.execute("SELECT count(*) FROM observations WHERE node_id=?", (node,)).fetchone()[0]
        for k in range(count):
            connection.execute("INSERT INTO observations VALUES (?,?,?)", (f"{sequence[:3]}{node}-{existing + k}-{source}", node, source))
        connection.commit()
        connection.close()

    def write_ranking(self, epoch: int, name: str, entries: list[tuple[str, int]]) -> Path:
        path = self.live / f"epoch{epoch}" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"candidates": [{"sequence": s, "observation_count": n} for s, n in entries]}))
        return path

    def write_artifact(self, epoch: int, sequences: list[str], pool: list[str] | None = None) -> None:
        (self.artifacts / "nanobody_result.json").write_text(json.dumps({
            "epoch": epoch, "submissions": [{"uid": str(i), "sequence": s} for i, s in enumerate(sequences)],
            "sequence_pool": pool if pool is not None else list(sequences)}))

    def write_checkpoint(self, epoch: int, stage: str, uid: int, sequence: str) -> None:
        path = self.live / f"epoch{epoch}" / "checkpoints" / stage / f"uid{uid}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"epoch": epoch, "sequences": [sequence]}))


class FakeSlave(ts.Slave):
    """Stands in for the slave: records every call, builds the files a real scoring run would."""

    def __init__(self, sandbox: Sandbox, *, hashes: dict | None = None, fail_rounds: set | None = None,
                 corrupt_rounds: set | None = None, unreachable: bool = False):
        super().__init__(host="fake", port=1)
        self.sandbox = sandbox
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple] = []
        self.hashes = hashes
        self.fail_rounds = fail_rounds or set()
        self.corrupt_rounds = corrupt_rounds or set()
        self.unreachable = unreachable
        self.removed: list[str] = []
        self.active = 0
        self.max_active = 0

    def probe(self, local_names, extra):
        if self.unreachable:
            raise RuntimeError("ssh down")
        hashes = self.hashes if self.hashes is not None else ts.master_hashes()
        return {"hashes": hashes, "daemon_socket": True, "gpus": ["0, 10 %, 100 MiB", "1, 0 %, 0 MiB",
                                                                  "2, 0 %, 0 MiB", "3, 0 %, 0 MiB"],
                "base_settings_text": json.dumps(BASE_SLAVE_SETTINGS),
                "private_settings_text": self.files.get(ts.SLAVE_SETTINGS, b"").decode()}

    def write_file(self, remote, data, timeout=60.0):
        self.calls.append(("write", remote))
        self.files[remote] = data

    def score(self, command, timeout):
        self.calls.append(("score", command))
        words = shlex.split(command)
        out = Path(words[words.index("--output-dir") + 1])
        number = int(out.name.rsplit("_", 1)[1])
        gpus = words[words.index("--gpus") + 1]
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if number in self.fail_rounds:
                return 1, "boom"
            sequences = json.loads(self.files[words[words.index("--sequences") + 1]].decode())["sequences"]
            self._build_round(out, number, sequences, corrupt=number in self.corrupt_rounds)
            return 0, ""
        finally:
            self.active -= 1

    def _build_round(self, out: Path, number: int, sequences: list[str], corrupt: bool) -> None:
        remote = self.sandbox.remote / str(out).lstrip("/")
        (remote / "structures").mkdir(parents=True, exist_ok=True)
        components, structures = {}, {}
        for index, sequence in enumerate(sequences):
            components[sequence] = {ts.TARGET: metrics_for(index)}
            if corrupt:
                components[sequence][ts.TARGET].pop("design_ptm")
            body = f"structure {sequence}".encode()
            for suffix in ("", "_writer"):
                (remote / "structures" / f"{sequence[:6]}{suffix}.cif").write_bytes(body)
            digest = hashlib.sha256(body).hexdigest()
            structures[sequence] = {ts.TARGET: {
                "path": f"{out}/structures/{sequence[:6]}.cif", "sha256": digest,
                "writer_path": f"{out}/structures/{sequence[:6]}_writer.cif", "writer_sha256": digest}}
        (remote / "exact_all.json").write_text(json.dumps({
            "target": ts.TARGET, "scorer_protocol": ts.SCORER_PROTOCOL, "observation_id": f"obs-{out.name}",
            "n": len(sequences), "per_nanobody_components": {"0": components}, "structures": structures}))
        (remote / "wave_000_exact.json").write_text(json.dumps({"wall_seconds": 12.5}))

    def pull_dir(self, remote, local, timeout=300.0):
        self.calls.append(("pull", remote))
        source = self.sandbox.remote / remote.lstrip("/")
        shutil.copytree(source, local, dirs_exist_ok=True)

    def remove_dir(self, remote):
        self.removed.append(remote)

    def cleanup_old_rounds(self):
        pass


BASE_SLAVE_SETTINGS = {
    "python": "/root/release_python_slave.sh", "scorer": "/root/fastscore/pinned/fast_score_batch.py",
    "scorer_sha256": "abc", "synth_reference": "/root/fastscore/pinned/refdesign/ref.cif",
}
MASTER_SETTINGS = {
    "python": "/root/fastscore/pinned/release_python.sh", "scorer": "/root/fastscore/pinned/fast_score_batch.py",
    "scorer_sha256": "abc", "synth_reference": "/elsewhere/refdesign/ref.cif", "fold_sampling_steps": 35,
}


def seed_master_scoring_files(box: Sandbox) -> None:
    for name in ts.PARITY_FILES:
        (box.tmp / "nanobody_opt" / name).write_text(f"# {name}\n")
    extra = box.tmp / "fast_score_batch.py"
    extra.write_text("# pinned scorer\n")
    patcher = mock.patch.object(ts, "PARITY_EXTRA", (str(extra),))
    patcher.start()
    box.cleanup = patcher.stop
    (box.artifacts / "nanobody_fast_scorer.json").write_text(json.dumps(MASTER_SETTINGS))


class PureHelperTests(unittest.TestCase):
    def test_required_metrics_mirror_the_canonical_list(self):
        tree = ast.parse((ROOT / "nanobody_opt" / "simulate_round.py").read_text())
        canonical = next(ast.literal_eval(node.value) for node in tree.body
                         if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "METRICS")
        self.assertEqual(set(canonical), set(ts.REQUIRED_METRICS))

    def test_scorer_protocol_mirrors_the_canonical_constant(self):
        text = (ROOT / "nanobody_opt" / "diffusion_sample_selection.py").read_text()
        self.assertIn(f'SCORER_PROTOCOL = "{ts.SCORER_PROTOCOL}"', text)

    def test_parse_probe(self):
        digest = "a" * 64
        text = (f"##hashes\n{digest}  /root/nanobody_opt/score_boltzgen_waves.py\n##daemon\n1\n"
                "##gpus\n0, 80 %, 3000 MiB\n1, 10 %, 200 MiB\n##settings\n{\"a\": 1}\n##private\n##end\n")
        probe = ts.parse_probe(text)
        self.assertEqual(probe["hashes"], {"score_boltzgen_waves.py": digest})
        self.assertTrue(probe["daemon_socket"])
        self.assertEqual(len(probe["gpus"]), 2)
        self.assertEqual(json.loads(probe["base_settings_text"]), {"a": 1})
        self.assertEqual(probe["private_settings_text"], "")

    def test_score_command_is_the_screen_command_on_the_slave(self):
        command = ts.build_score_command(source="/r/source.json", config="/r/config.yaml", output_dir="/r/round",
                                         gpus=["2", "3"], count=7, timeout_s=840)
        words = shlex.split(command)
        self.assertEqual(words[:2], ["cd", "/root"])
        self.assertIn("NANOBODY_SCORER_DAEMON=1", words)
        self.assertIn(f"NANOBODY_FAST_SCORER_CONFIG={ts.SLAVE_SETTINGS}", words)
        self.assertEqual(words[words.index("timeout") + 1], "840")
        flags = dict(zip(words[words.index("-m") + 2::2], words[words.index("-m") + 3::2]))
        self.assertEqual(words[words.index("-m") + 1], "nanobody_opt.score_boltzgen_waves")
        self.assertEqual(flags["--gpus"], "2,3")
        self.assertEqual(flags["--wave-size"], "7")            # the whole tranche
        self.assertEqual(flags["--predict-batch-size"], "2")   # pinned whatever the tranche size
        self.assertEqual(flags["--limit"], "7")
        self.assertEqual(flags["--target"], "P20809")
        self.assertEqual(flags["--scorer"], ts.SLAVE_SCORER)
        self.assertEqual(words[-1], "--capture-structures")

    def test_measurement_keys_follow_the_master_not_the_slave(self):
        slave = {**BASE_SLAVE_SETTINGS, "fold_sampling_steps": 50}
        wanted = ts.desired_slave_settings(MASTER_SETTINGS, slave)
        self.assertEqual(wanted["fold_sampling_steps"], 35)
        self.assertEqual(wanted["python"], "/root/release_python_slave.sh")   # the slave's own launcher
        master_default = {k: v for k, v in MASTER_SETTINGS.items() if k != "fold_sampling_steps"}
        self.assertNotIn("fold_sampling_steps", ts.desired_slave_settings(master_default, slave))

    def test_settings_must_name_the_same_scorer(self):
        self.assertIsNone(ts.settings_problem(MASTER_SETTINGS, BASE_SLAVE_SETTINGS))
        self.assertIn("scorer_sha256", ts.settings_problem(MASTER_SETTINGS, {**BASE_SLAVE_SETTINGS, "scorer_sha256": "x"}))
        self.assertIn("synth_reference", ts.settings_problem(
            MASTER_SETTINGS, {**BASE_SLAVE_SETTINGS, "synth_reference": "/x/other.cif"}))

    def test_the_scorer_pin_may_differ_only_as_the_reviewed_pair(self):
        reviewed = {"fast_score_batch.py": ("a" * 64, "b" * 64)}
        master = {**MASTER_SETTINGS, "scorer_sha256": "a" * 64}
        with mock.patch.object(ts, "REVIEWED_DIVERGENCE", reviewed):
            self.assertIsNone(ts.settings_problem(master, {**BASE_SLAVE_SETTINGS, "scorer_sha256": "b" * 64}))
            self.assertIn("scorer_sha256", ts.settings_problem(
                master, {**BASE_SLAVE_SETTINGS, "scorer_sha256": "c" * 64}))
            self.assertIn("scorer_sha256", ts.settings_problem(
                {**master, "scorer_sha256": "d" * 64}, {**BASE_SLAVE_SETTINGS, "scorer_sha256": "b" * 64}))

    def test_rounds_run_on_gpu_pairs_like_the_master_top_up(self):
        # Measured 2026-10-14: the same 12 sequences on one GPU per round had 1.3-1.7x the spread of the
        # master's observations on design_ptm / design_to_target_iptm / interaction_pae (sd of z 1.28
        # pooled, 1.73 on ptm); on pairs it was 1.04.  Do not change this without repeating that test.
        self.assertEqual(ts.GROUP_SIZE, 2)

    def test_gpu_pairs(self):
        self.assertEqual(ts.gpu_pairs(["0", "1", "2", "3"]), [["0", "1"], ["2", "3"]])
        self.assertEqual(ts.gpu_pairs(["0", "1", "2"]), [["0", "1"]])
        self.assertEqual(ts.gpu_pairs(["0"]), [["0"]])
        self.assertEqual(ts.gpu_pairs(["0", "1", "2", "3"], 1), [["0"], ["1"], ["2"], ["3"]])
        self.assertEqual(ts.gpu_pairs(["0", "1", "2", "3"], 4), [["0", "1", "2", "3"]])


class SlaveCommandTests(unittest.TestCase):
    def setUp(self):
        self.seen = []

        def run(argv, **kwargs):
            self.seen.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        self.slave = ts.Slave(host="root@h", port=2222, run=run)

    def test_ssh_goes_to_the_configured_slave_with_batchmode_and_timeout(self):
        self.slave.sh("true", timeout=7)
        argv, kwargs = self.seen[0]
        self.assertEqual(argv[0], "ssh")
        self.assertIn("BatchMode=yes", argv)
        self.assertEqual(argv[argv.index("-p") + 1], "2222")
        self.assertEqual(argv[-2:], ["root@h", "true"])
        self.assertEqual(kwargs["timeout"], 7)

    def test_write_is_atomic_on_the_slave(self):
        self.slave.write_file("/root/live/e/topup_slave/source.json", b"{}")
        script = self.seen[0][0][-1]
        self.assertIn(".part", script)
        self.assertIn("mv -f", script)
        self.assertEqual(self.seen[0][1]["input"], b"{}")

    def test_pull_excludes_the_work_tree(self):
        self.slave.pull_dir("/root/live/e/topup_slave/round_1", Path(tempfile.mkdtemp()))
        argv = self.seen[0][0]
        self.assertEqual(argv[0], "rsync")
        self.assertIn("--exclude=work/", argv)

    def test_remove_dir_refuses_anything_outside_our_namespace(self):
        self.slave.remove_dir("/root/live/epoch1/producer")
        self.slave.remove_dir("/root")
        self.assertEqual(self.seen, [])
        self.slave.remove_dir("/root/live/epoch1/topup_slave/round_1")
        self.assertEqual(len(self.seen), 1)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.box = Sandbox(self)

    def test_published_sequences_come_first_from_artifact_pool_and_checkpoints(self):
        self.box.write_artifact(25700, [seq(1), seq(2)], pool=[seq(2), seq(3)])
        self.box.write_checkpoint(25700, "provisional", 26, seq(4))
        self.box.write_checkpoint(25699, "provisional", 26, seq(9))     # another epoch: ignored
        self.assertEqual(ts.live_sequences(25700), [seq(1), seq(2), seq(3), seq(4)])

    def test_artifact_of_another_epoch_is_ignored(self):
        self.box.write_artifact(25699, [seq(1)])
        self.assertEqual(ts.live_sequences(25700), [])

    def test_needs_uses_database_counts_and_priority_order(self):
        self.box.write_artifact(25700, [seq(50)])                       # published, 8 observations
        self.box.write_ranking(25700, "strict_pool_rankings.json",
                               [(seq(1), 25), (seq(2), 8), (seq(3), 19), (seq(4), 3)])
        self.box.write_ranking(25700, "qualified_rankings.json", [(seq(2), 8), (seq(5), 12)])
        for i, n in {50: 8, 1: 25, 2: 10, 3: 19, 4: 14, 5: 12}.items():  # the database is the truth
            self.box.add_observations(seq(i), n)
        need, info = ts.needs(25700)
        self.assertEqual(info["counts"], "database")
        self.assertEqual(list(need), [seq(50), seq(2), seq(5), seq(3), seq(4)])   # live, qualified, pool
        self.assertEqual(need[seq(50)], 12)                                # 20 - 8
        self.assertEqual(need[seq(2)], 10)                                 # database says 10, ranking said 8
        self.assertEqual(need[seq(3)], 1)
        self.assertNotIn(seq(1), need)

    def test_only_local_exact_raw_counts(self):
        self.box.write_artifact(25700, [seq(1)])
        self.box.add_observations(seq(1), 5)
        self.box.add_observations(seq(1), 30, source="public_multipass_rivals")   # the replay label never counts
        need, _ = ts.needs(25700)
        self.assertEqual(need[seq(1)], 12)                                 # 20 - 5 = 15, capped at 12 rounds

    def test_wrong_length_sequences_are_skipped(self):
        self.box.write_artifact(25700, ["ACDE"])
        need, info = ts.needs(25700)
        self.assertEqual(need, {})
        self.assertEqual(info["skipped_length"], 1)

    def test_pool_is_limited_to_the_publishers_strict_pool_size(self):
        entries = [(seq(i), 8) for i in range(ts.POOL_K + 30)]
        self.box.write_ranking(25700, "strict_pool_rankings.json", entries)
        need, info = ts.needs(25700)
        self.assertEqual(info["pool"], ts.POOL_K)
        self.assertEqual(info["candidates"], ts.POOL_K)
        self.assertNotIn(seq(ts.POOL_K), need)             # the 101st candidate is the publisher's tail

    def test_budget_drops_the_lowest_priority_tail(self):
        entries = [(seq(i), 8) for i in range(60)]
        self.box.write_ranking(25700, "strict_pool_rankings.json", entries)
        with mock.patch.object(ts, "MAX_OBS_PER_PASS", 5 * 12):
            need, info = ts.needs(25700)
        self.assertTrue(info["capped_by_budget"])
        self.assertEqual(list(need), [s for s, _ in entries[:5]])

    def test_falls_back_to_ranking_counts_when_the_database_is_unreadable(self):
        self.box.write_ranking(25700, "strict_pool_rankings.json", [(seq(1), 18), (seq(2), 20)])
        self.box.db.unlink()
        need, info = ts.needs(25700)
        self.assertEqual(info["counts"], "ranking files")
        self.assertEqual(need, {seq(1): 2})

    def test_the_freshest_pool_ranking_wins(self):
        old = self.box.write_ranking(25700, "strict_pool_rankings.json", [(seq(1), 8)])
        os.utime(old, (time.time() - 600, time.time() - 600))
        self.box.write_ranking(25700, "provisional_rankings.json", [(seq(2), 8)])
        sources = ts.ranking_sources(25700)
        self.assertEqual(sources["pool_name"], "provisional_rankings.json")


class ValidateRoundTests(unittest.TestCase):
    def setUp(self):
        self.box = Sandbox(self)
        self.slave = FakeSlave(self.box)
        self.subset = [seq(1), seq(2), seq(3)]
        self.final = ts.round_dir(25700, "000001", 1)
        self.slave._build_round(self.final, 1, self.subset, corrupt=False)
        self.staged = self.box.remote / str(self.final).lstrip("/")

    def test_a_good_round_passes(self):
        self.assertIsNone(ts.validate_round(self.staged, self.final, self.subset))

    def test_rejections(self):
        cases = {
            "missing sequence": lambda: ts.validate_round(self.staged, self.final, self.subset + [seq(9)]),
            "wrong protocol": lambda: self._edit(lambda p: p.update(scorer_protocol="old")),
            "wrong target": lambda: self._edit(lambda p: p.update(target="P40261")),
            "no observation id": lambda: self._edit(lambda p: p.update(observation_id="")),
            "missing metric": lambda: self._edit(lambda p: next(iter(p["per_nanobody_components"]["0"].values()))[ts.TARGET].pop("design_ptm")),
            "not finite": lambda: self._edit(lambda p: next(iter(p["per_nanobody_components"]["0"].values()))[ts.TARGET].update(design_ptm=float("nan"))),
            "foreign structure path": lambda: self._edit(lambda p: next(iter(p["structures"].values()))[ts.TARGET].update(path="/elsewhere/x.cif")),
        }
        for name, build in cases.items():
            with self.subTest(name):
                self.assertIsNotNone(build())
                self.slave._build_round(self.final, 1, self.subset, corrupt=False)   # restore

    def test_corrupt_structure_is_rejected(self):
        next(iter((self.staged / "structures").glob("*.cif"))).write_bytes(b"truncated")
        self.assertEqual(ts.validate_round(self.staged, self.final, self.subset), "structure file missing or corrupt")

    def _edit(self, change):
        path = self.staged / "exact_all.json"
        payload = json.loads(path.read_text())
        change(payload)
        path.write_text(json.dumps(payload))
        return ts.validate_round(self.staged, self.final, self.subset)


class ScoreRoundTests(unittest.TestCase):
    def setUp(self):
        self.box = Sandbox(self)
        self.slave = FakeSlave(self.box)
        self.subset = [seq(1), seq(2)]

    def run_round(self, number=1):
        return ts.score_round(self.slave, epoch=25700, stamp="000001", number=number, subset=self.subset,
                              gpus=["0", "1"], remote_config="/remote/config.yaml")

    def test_success_puts_the_round_in_place_and_cleans_up(self):
        result = self.run_round()
        self.assertTrue(result["ok"], result)
        final = ts.round_dir(25700, "000001", 1)
        self.assertEqual(Path(result["dir"]), final)
        self.assertTrue((final / "exact_all.json").is_file())
        self.assertEqual(result["compute_seconds"], 12.5)
        self.assertEqual(self.slave.removed, [str(final)])
        self.assertEqual(list(ts.stage_root().iterdir()), [])          # nothing left waiting

    def test_partners_change_between_rounds_but_the_tranche_is_the_same(self):
        self.subset = [seq(i) for i in range(10)]
        orders = []
        for number in (1, 2, 3):
            self.run_round(number)
            source = json.loads(self.slave.files[f"{ts.round_dir(25700, '000001', number)}/source.json"].decode())
            orders.append(source["sequences"])
            self.assertEqual(sorted(source["sequences"]), sorted(self.subset))
        self.assertGreater(len({tuple(o) for o in orders}), 1)
        again = FakeSlave(self.box)
        for number in (1, 2, 3):                      # deterministic for a given stamp and round
            shutil.rmtree(ts.round_dir(25700, "000001", number))
            ts.score_round(again, epoch=25700, stamp="000001", number=number, subset=self.subset,
                           gpus=["0", "1"], remote_config="/c")
            source = json.loads(again.files[f"{ts.round_dir(25700, '000001', number)}/source.json"].decode())
            self.assertEqual(source["sequences"], orders[number - 1])

    def test_the_slave_writes_where_the_paths_inside_the_files_say(self):
        self.run_round()
        payload = json.loads((ts.round_dir(25700, "000001", 1) / "exact_all.json").read_text())
        record = payload["structures"][self.subset[0]][ts.TARGET]
        self.assertTrue(record["path"].startswith(str(ts.round_dir(25700, "000001", 1)) + "/"))
        self.assertTrue(Path(record["path"]).is_file())

    def test_staging_is_never_inside_an_epoch_directory(self):
        self.assertNotRegex(str(ts.stage_root().relative_to(ts.LIVE_ROOT)), r"^epoch\d+")
        self.assertEqual(ts.stage_root().parent, ts.LIVE_ROOT)         # same filesystem: rename is atomic

    def test_scoring_failure_leaves_nothing_behind(self):
        self.slave.fail_rounds = {1}
        result = self.run_round()
        self.assertIn("scoring rc=1", result["error"])
        self.assertFalse(ts.round_dir(25700, "000001", 1).exists())

    def test_invalid_round_is_not_moved_into_live(self):
        self.slave.corrupt_rounds = {1}
        result = self.run_round()
        self.assertIn("invalid round", result["error"])
        self.assertFalse(ts.round_dir(25700, "000001", 1).exists())
        self.assertEqual(list(ts.stage_root().iterdir()), [])

    def test_pull_failure_is_contained(self):
        self.slave.pull_dir = mock.Mock(side_effect=RuntimeError("rsync died"))
        result = self.run_round()
        self.assertIn("rsync died", result["error"])
        self.assertFalse(ts.round_dir(25700, "000001", 1).exists())

    def test_existing_round_directory_is_never_overwritten(self):
        ts.round_dir(25700, "000001", 1).mkdir(parents=True)
        self.assertEqual(self.run_round()["error"], "round directory exists")


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.box = Sandbox(self)
        seed_master_scoring_files(self.box)
        self.addCleanup(self.box.cleanup)

    def test_identical_code_is_ok_and_the_private_settings_follow_the_master(self):
        slave = FakeSlave(self.box)
        result = ts.preflight(slave)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["fold_sampling_steps"], 35)
        written = json.loads(slave.files[ts.SLAVE_SETTINGS])
        self.assertEqual(written["fold_sampling_steps"], 35)
        self.assertEqual(written["python"], "/root/release_python_slave.sh")
        self.assertEqual(result["gpus"], ["0", "1", "2", "3"])

    def test_dry_run_writes_nothing_to_the_slave(self):
        slave = FakeSlave(self.box)
        result = ts.preflight(slave, write=False)
        self.assertTrue(result["ok"])
        self.assertTrue(result["settings_would_change"])
        self.assertEqual(slave.files, {})

    def test_different_scoring_code_blocks_every_pass(self):
        hashes = ts.master_hashes()
        hashes["scorer_daemon.py"] = "0" * 64
        result = ts.preflight(FakeSlave(self.box, hashes=hashes))
        self.assertFalse(result["ok"])
        self.assertEqual(result["different"], ["scorer_daemon.py"])

    def test_a_reviewed_divergence_is_accepted_and_reported(self):
        master = ts.master_hashes()
        hashes = dict(master, **{"scorer_daemon.py": "1" * 64})
        reviewed = {"scorer_daemon.py": (master["scorer_daemon.py"], "1" * 64)}
        with mock.patch.object(ts, "REVIEWED_DIVERGENCE", reviewed):
            result = ts.preflight(FakeSlave(self.box, hashes=hashes), write=False)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["reviewed_divergence"], ["scorer_daemon.py"])

    def test_a_reviewed_divergence_does_not_cover_further_drift(self):
        master = ts.master_hashes()
        reviewed = {"scorer_daemon.py": (master["scorer_daemon.py"], "1" * 64)}
        with mock.patch.object(ts, "REVIEWED_DIVERGENCE", reviewed):
            slave_changed = ts.preflight(FakeSlave(self.box, hashes=dict(master, **{"scorer_daemon.py": "2" * 64})))
            other_file = ts.preflight(FakeSlave(self.box, hashes=dict(master, **{"chunk_reuse.py": "1" * 64})))
        self.assertEqual(slave_changed["different"], ["scorer_daemon.py"])
        self.assertEqual(other_file["different"], ["chunk_reuse.py"])
        # the master moving on invalidates the review as well
        moved = {"scorer_daemon.py": ("3" * 64, "1" * 64)}
        with mock.patch.object(ts, "REVIEWED_DIVERGENCE", moved):
            result = ts.preflight(FakeSlave(self.box, hashes=dict(master, **{"scorer_daemon.py": "1" * 64})))
        self.assertEqual(result["different"], ["scorer_daemon.py"])

    def test_the_reviewed_pairs_cover_exactly_the_three_files_reviewed(self):
        self.assertEqual(sorted(ts.REVIEWED_DIVERGENCE),
                         ["fast_score_batch.py", "score_boltzgen_fast_adapter.py", "scorer_daemon.py"])
        for master_sha, slave_sha in ts.REVIEWED_DIVERGENCE.values():
            self.assertEqual((len(master_sha), len(slave_sha)), (64, 64))
            self.assertNotEqual(master_sha, slave_sha)

    def test_missing_file_on_the_slave_blocks(self):
        hashes = ts.master_hashes()
        hashes.pop("chunk_reuse.py")
        self.assertFalse(ts.preflight(FakeSlave(self.box, hashes=hashes))["ok"])

    def test_unreachable_slave_is_a_skip_not_a_crash(self):
        result = ts.preflight(FakeSlave(self.box, unreachable=True))
        self.assertFalse(result["ok"])
        self.assertIn("unreachable", result["reason"])


class PassTests(unittest.TestCase):
    def setUp(self):
        self.box = Sandbox(self)
        seed_master_scoring_files(self.box)
        self.addCleanup(self.box.cleanup)
        config = self.box.live / "epoch25700" / "producer" / "config.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("nanobody: {}\n")
        self.imports = []
        ts._PASSES.clear()
        ts._STALLED_UNTIL[0] = 0.0
        self.import_adds = True

        def fake_import(epoch, dirs, archive=None):
            self.imports.append((epoch, list(dirs)))
            if self.import_adds:                              # what the real importer does to the database
                for directory in dirs:
                    source = json.loads((self.box.remote / str(directory).lstrip("/") / "source.json").read_text()) \
                        if (self.box.remote / str(directory).lstrip("/") / "source.json").exists() else None
                    payload = json.loads((directory / "exact_all.json").read_text())
                    for sequence in payload["per_nanobody_components"]["0"]:
                        self.box.add_observations(sequence, 1)
            return 0, ["ok"]
        patcher = mock.patch.object(ts, "import_rounds", side_effect=fake_import)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name, value in (("free_disk_gb", lambda: 500.0), ("sleep", lambda s: None)):
            if name == "sleep":
                p = mock.patch.object(ts.time, "sleep", value)
            else:
                p = mock.patch.object(ts, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.box.write_artifact(25700, [seq(1)])
        self.box.add_observations(seq(1), 16)                  # needs 4 rounds
        self.box.write_ranking(25700, "strict_pool_rankings.json", [(seq(2), 18), (seq(3), 8)])
        self.box.add_observations(seq(2), 18)                  # needs 2
        self.box.add_observations(seq(3), 8)                   # needs 12

    def test_outside_the_window_nothing_happens(self):
        self.box.set_chain(25700, 5)
        self.assertEqual(ts.run_pass(FakeSlave(self.box))["skipped"], "outside the top-up window")
        self.box.set_chain(25700, ts.final_import_offset())
        self.assertEqual(ts.run_pass(FakeSlave(self.box))["skipped"], "outside the top-up window")

    def test_stale_chain_state_is_a_skip(self):
        old = time.time() - 1000
        os.utime(self.box.artifacts / "chain_state.json", (old, old))
        self.assertEqual(ts.run_pass(FakeSlave(self.box))["skipped"], "chain state is stale")

    def test_nothing_short_is_a_cheap_skip_without_touching_the_slave(self):
        for s, n in ((seq(1), 4), (seq(2), 2), (seq(3), 12)):
            self.box.add_observations(s, n)
        slave = FakeSlave(self.box)
        self.assertEqual(ts.run_pass(slave)["skipped"], "nothing short of the target")
        self.assertEqual(slave.calls, [])

    def test_dry_run_plans_but_does_not_score_or_write(self):
        slave = FakeSlave(self.box)
        result = ts.run_pass(slave, dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["short"], 3)
        self.assertEqual(result["observations_needed"], 4 + 2 + 12)
        self.assertEqual(result["round_sizes"], [3, 3, 2, 2] + [1] * 8)   # round r holds those needing >= r
        self.assertEqual(slave.calls, [])
        self.assertEqual(self.imports, [])

    def test_full_pass_scores_every_missing_round_and_imports_once(self):
        slave = FakeSlave(self.box)
        result = ts.run_pass(slave)
        self.assertEqual(result["rounds_scored"], 12, result)
        self.assertEqual(result["observations_added"], 18)
        self.assertEqual(len(self.imports), 1)                  # one import, so one publisher cycle
        epoch, dirs = self.imports[0]
        self.assertEqual(epoch, 25700)
        self.assertEqual(len(dirs), 12)
        for d in dirs:
            self.assertTrue((d / "exact_all.json").is_file())
            self.assertEqual(d.parent, self.box.live / "epoch25700" / "topup_slave")
        self.assertLessEqual(slave.max_active, 2)               # two GPU pairs at a time
        sizes = sorted(len(json.loads(slave.files[f"{d}/source.json"].decode())["sequences"]) for d in dirs)
        self.assertEqual(sizes, sorted([3, 3, 2, 2] + [1] * 8))
        state = json.loads((self.box.live / "epoch25700" / "topup_slave" / "state.json").read_text())
        self.assertEqual(sum(n for _t, n in state["added"][seq(3)]), 12)
        self.assertEqual(slave.files[f"{self.box.live}/epoch25700/topup_slave/config.yaml"], b"nanobody: {}\n")

    def test_rehearsal_keeps_rounds_and_import_out_of_the_live_state(self):
        scratch = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, scratch, True)
        seen_db = []
        real = ts.import_rounds

        def spy(epoch, dirs, archive=None):
            seen_db.append(ts.IMPORT_DB)
            return 0, ["ok"]
        with mock.patch.object(ts, "ROUNDS_ROOT", scratch), mock.patch.object(ts, "IMPORT_DB", scratch / "x.sqlite"), \
                mock.patch.object(ts, "import_rounds", spy):
            result = ts.run_pass(FakeSlave(self.box))
        self.assertEqual(result["rounds_scored"], 12)
        self.assertEqual(seen_db, [scratch / "x.sqlite"])
        self.assertTrue(any(scratch.rglob("exact_all.json")))
        self.assertFalse((self.box.live / "epoch25700" / "topup_slave").exists())    # nothing under live/
        self.assertEqual(self.imports, [])

    def test_a_pass_announces_itself_before_the_minutes_it_takes(self):
        ts.run_pass(FakeSlave(self.box))
        events = [json.loads(line) for line in (self.box.artifacts / "topup_slave.log").read_text().splitlines()]
        start = [e for e in events if e.get("event") == "pass_start"]
        self.assertEqual(len(start), 1)
        self.assertEqual((start[0]["short"], start[0]["rounds_planned"]), (3, 12))

    def test_a_normal_import_raises_the_counts_and_the_next_pass_finds_nothing_short(self):
        first = ts.run_pass(FakeSlave(self.box))
        self.assertEqual(first["counts_gained"], 18)
        self.assertEqual(first["still_short_after"], 0)
        self.assertNotIn("warning", first)
        self.assertEqual(ts.run_pass(FakeSlave(self.box))["skipped"], "nothing short of the target")

    def test_an_import_that_changes_nothing_pauses_the_worker_instead_of_looping(self):
        self.import_adds = False
        first = ts.run_pass(FakeSlave(self.box))
        self.assertEqual(first["counts_gained"], 0)
        self.assertIn("pausing", first["warning"])
        self.assertIn("paused", ts.run_pass(FakeSlave(self.box))["skipped"])
        ts._STALLED_UNTIL[0] = 0.0
        self.assertEqual(ts.run_pass(FakeSlave(self.box))["rounds_scored"], 12)        # resumes after the pause

    def test_pass_limit_per_epoch(self):
        ts._PASSES[25700] = ts.MAX_PASSES_PER_EPOCH
        self.assertEqual(ts.run_pass(FakeSlave(self.box))["skipped"], "pass limit for this epoch")
        self.assertTrue(ts.run_pass(FakeSlave(self.box), dry_run=True)["dry_run"])      # planning is never blocked

    def test_a_failed_round_only_loses_that_round(self):
        slave = FakeSlave(self.box, fail_rounds={2})
        result = ts.run_pass(slave)
        self.assertEqual(result["rounds_scored"], 11)
        self.assertEqual(len(result["rounds_failed"]), 1)
        self.assertEqual(len(self.imports[0][1]), 11)

    def test_every_round_failing_imports_nothing(self):
        slave = FakeSlave(self.box, fail_rounds=set(range(1, 13)))
        result = ts.run_pass(slave)
        self.assertEqual(result["rounds_scored"], 0)
        self.assertEqual(self.imports, [])

    def test_deadline_stops_new_rounds_but_keeps_the_finished_ones(self):
        slave = FakeSlave(self.box)
        calls = {"n": 0}
        real = ts.chain_position

        def moving_clock():
            calls["n"] += 1
            epoch, offset, age = real()
            return epoch, (offset if calls["n"] <= 4 else ts.final_import_offset() - 1), age
        with mock.patch.object(ts, "chain_position", moving_clock):
            result = ts.run_pass(slave)
        self.assertIn("final import deadline", result["stopped"])
        self.assertGreater(result["rounds_scored"], 0)
        self.assertLess(result["rounds_scored"], 12)
        self.assertEqual(len(self.imports), 1)

    def test_a_pass_never_scores_into_the_next_epoch(self):
        slave = FakeSlave(self.box)
        real = ts.chain_position
        calls = {"n": 0}

        def rolled():
            calls["n"] += 1
            epoch, offset, age = real()
            return (epoch if calls["n"] <= 4 else epoch + 1), offset, age
        with mock.patch.object(ts, "chain_position", rolled):
            result = ts.run_pass(slave)
        self.assertIn("epoch rolled", result["stopped"])
        self.assertLess(result["rounds_scored"], 12)

    def test_pass_time_limit_imports_what_is_done(self):
        slave = FakeSlave(self.box)
        with mock.patch.object(ts, "PASS_MAX_SECONDS", -1.0):
            result = ts.run_pass(slave)
        self.assertEqual(result["stopped"], "pass time limit")
        self.assertEqual(result["rounds_scored"], 0)
        self.assertEqual(self.imports, [])

    def test_a_bad_slave_skips_the_pass(self):
        hashes = ts.master_hashes()
        hashes["scorer_daemon.py"] = "f" * 64
        result = ts.run_pass(FakeSlave(self.box, hashes=hashes))
        self.assertIn("preflight", result["skipped"])
        self.assertEqual(self.imports, [])

    def test_waits_for_the_master_top_up_then_recounts(self):
        (self.box.artifacts / "topup_running.json").write_text("{}")
        states = iter([True, True, False])
        waits = []
        with mock.patch.object(ts, "master_topup_busy", lambda: next(states)), \
                mock.patch.object(ts.time, "sleep", lambda s: waits.append(s)):
            result = ts.run_pass(FakeSlave(self.box))
        self.assertEqual(result["waited_for_master_s"], 10.0)
        self.assertEqual(result["rounds_scored"], 12)

    def test_old_rounds_on_the_slave_are_cleaned_once_per_epoch(self):
        ts._CLEANED.clear()
        slave = FakeSlave(self.box)
        slave.cleanup_old_rounds = mock.Mock()
        ts.run_pass(slave, dry_run=False)
        self.box.add_observations(seq(3), 4)       # short again for a second pass
        ts.run_pass(slave, dry_run=False)
        self.assertEqual(slave.cleanup_old_rounds.call_count, 1)

    def test_the_master_marker_is_never_written_by_this_worker(self):
        ts.run_pass(FakeSlave(self.box))
        self.assertFalse((self.box.artifacts / "topup_running.json").exists())

    def test_low_disk_is_a_skip(self):
        with mock.patch.object(ts, "free_disk_gb", lambda: 3.0):
            self.assertIn("disk", ts.run_pass(FakeSlave(self.box))["skipped"])


class DeploymentTests(unittest.TestCase):
    def test_the_watchdog_keeps_the_slave_top_up_alive_next_to_the_master_one(self):
        import re
        text = (ROOT / "deploy" / "producer" / "session_watchdog.sh").read_text()
        found = dict(re.findall(r"tmux has-session -t (\S+) 2>/dev/null \\\n\s+\|\| start_session \1 (\S+)", text))
        self.assertEqual(found["topup-slave"], "/root/tools/topup_slave_loop.sh")
        self.assertEqual(found["topup-publishable"], "/root/tools/topup_loop.sh")     # the fallback is untouched

    def test_the_loop_script_is_valid_bash_niced_and_records_its_own_death(self):
        path = ROOT / "tools" / "topup_slave_loop.sh"
        self.assertEqual(subprocess.run(["bash", "-n", str(path)]).returncode, 0)
        text = path.read_text()
        self.assertIn("nice -n 10", text)
        self.assertIn("topup_slave.py loop", text)
        self.assertIn("TOPUP_SLAVE_EXITED", text)
        self.assertNotIn("slave_gpu.lock", text)                 # does not queue behind the replay flock


class ManualRoundsTests(unittest.TestCase):
    def test_rounds_command_scores_pulls_and_optionally_imports_to_a_scratch_db(self):
        box = Sandbox(self)
        seed_master_scoring_files(box)
        self.addCleanup(box.cleanup)
        scratch = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, scratch, True)
        for name in ("ROUNDS_ROOT", "IMPORT_DB"):
            patcher = mock.patch.object(ts, name, None)
            patcher.start()
            self.addCleanup(patcher.stop)
        sequences = scratch / "seqs.txt"
        sequences.write_text("\n".join(seq(i) for i in range(3)))
        config = scratch / "config.yaml"
        config.write_text("x: 1\n")
        imported = []
        args = argparse.Namespace(sequences=str(sequences), config=str(config), epoch=25700, rounds=2, group_size=None,
                                  round_root=str(scratch / "rounds"), import_db=str(scratch / "scratch.sqlite"),
                                  archive=str(scratch / "archive.csv"))
        with mock.patch.object(ts, "import_rounds", lambda e, d, a=None: imported.append((ts.IMPORT_DB, a, len(d))) or (0, [])):
            rc = ts.command_rounds(args, FakeSlave(box))
        self.assertEqual(rc, 0)
        self.assertEqual(imported, [(scratch / "scratch.sqlite", scratch / "archive.csv", 2)])
        self.assertEqual(len(list((scratch / "rounds").rglob("exact_all.json"))), 2)
        self.assertFalse(any(box.live.rglob("exact_all.json")))             # never under the live tree


class DeadlineTests(unittest.TestCase):
    def setUp(self):
        self.box = Sandbox(self)

    def test_the_cut_off_follows_the_pis_pull_which_follows_the_crew_size(self):
        # crew 10 pulled at 294 (epoch 25693), crew 7 at 311-326: the import must leave a whole strict
        # cycle before the pull
        self.assertEqual(ts.final_import_offset(10), 260)
        self.assertEqual(ts.final_import_offset(7), 275)
        self.assertEqual(ts.final_import_offset(5), 285)          # capped
        self.assertEqual(ts.final_import_offset(30), ts.FINAL_IMPORT_FLOOR)   # floored
        for crew, observed_pull in ((10, 294), (10, 306), (9, 311), (8, 326), (7, 312), (7, 316), (7, 320)):
            self.assertLess(ts.final_import_offset(crew) + ts.CYCLE_BLOCKS, observed_pull + 1)

    def test_crew_comes_from_the_artifact(self):
        self.assertEqual(ts.crew_size(), ts.DEFAULT_CREW)
        self.box.write_artifact(25700, [seq(i) for i in range(7)])
        self.assertEqual(ts.crew_size(), 7)
        self.assertEqual(ts.final_import_offset(), 275)


class LockAndLogTests(unittest.TestCase):
    def test_only_one_instance_can_hold_the_lock(self):
        box = Sandbox(self)
        first = ts.take_lock(box.artifacts / "x.lock")
        self.assertIsNotNone(first)
        self.assertIsNone(ts.take_lock(box.artifacts / "x.lock"))
        first.close()
        self.assertIsNotNone(ts.take_lock(box.artifacts / "x.lock"))

    def test_log_lines_are_json_with_a_utc_stamp(self):
        box = Sandbox(self)
        ts.log(epoch=1, short=2)
        line = json.loads((box.artifacts / "topup_slave.log").read_text().splitlines()[0])
        self.assertEqual((line["epoch"], line["short"]), (1, 2))
        self.assertTrue(line["at"].endswith("Z"))


class ImportCommandTests(unittest.TestCase):
    def test_import_uses_the_masters_importer_with_one_root_per_round(self):
        box = Sandbox(self)
        (box.live / "epoch25700").mkdir()
        (box.live / "epoch25700" / "archive.csv").write_text("sequence\n")
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return subprocess.CompletedProcess(command, 0, "a\nb\n", "")
        with mock.patch.object(ts.subprocess, "run", fake_run):
            rc, tail = ts.import_rounds(25700, [Path("/r/1"), Path("/r/2")])
        command = captured["command"]
        self.assertEqual(rc, 0)
        self.assertEqual(command[:3], ["nice", "-n", "5"])
        self.assertIn("nanobody_opt.import_exact_screens", command)
        self.assertEqual(command[command.index("--database") + 1], str(box.db))
        self.assertEqual([command[i + 1] for i, w in enumerate(command) if w == "--exact-root"], ["/r/1", "/r/2"])
        self.assertEqual(command[command.index("--archive") + 1], str(box.live / "epoch25700" / "archive.csv"))
        self.assertEqual(captured["kwargs"]["env"]["PYTHONPATH"], "/root")


if __name__ == "__main__":
    unittest.main()

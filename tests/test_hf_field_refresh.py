"""hf_field_refresh: raw per-pass metrics under multipass_sum10_pass, plus the GPU/time guard.

Two bugs fixed on 2026-10-10, and one redirect:

1. Every row the refresh wrote had ``metrics_json='{}'``: score_pass() read
   the raw BoltzGen metrics from ``exact_all.json`` and threw them away.
2. The only contention guard was a single nvidia-smi sample in the loop
   wrapper, plus a "stale for 3h, run anyway" override, so a run started by
   hand had no guard at all.
3. The refresh now writes ONE ROW PER PASS under ``multipass_sum10_pass``,
   with the contract of the dashboard pass scorer
   (tools/multipass_all_rivals.py) and of tools/migrate_hf_field_20x.py.
   It only tops up the passes missing to depth 20. The old mean labels
   are closed.

The metrics tests use REAL scorer passes (tests/fixtures/hf_field_refresh/
real_passes.json). They compare against the dashboard writer's own
functions, and they send rows through the real
``mutation_network._prepare_observation`` (the code that builds the DB
columns) and the live top-up reader (``hf_topup``). The guard tests
simulate the live producer mid-race and check that the script refuses
before touching the DB or a GPU. One watchdog test uses real processes: it
kills a running scorer when a producer process appears.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:  # the real writer module imports MySQLdb; no DB is touched here
    import MySQLdb  # noqa: F401
except ImportError:
    import types
    _mysql = types.ModuleType("MySQLdb")
    _mysql.OperationalError = type("OperationalError", (Exception,), {})
    _mysql.cursors = types.ModuleType("MySQLdb.cursors")
    _mysql.cursors.SSCursor = object
    _mysql.connect = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no DB in tests"))
    sys.modules["MySQLdb"] = _mysql
    sys.modules["MySQLdb.cursors"] = _mysql.cursors

from nanobody_opt import hf_field_refresh as R
from nanobody_opt import hf_topup as H
from nanobody_opt import mutation_network as MN
from nanobody_opt import observation_labels as OL

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hf_field_refresh" / "real_passes.json"
TEN = list(R.SUM_OF_TEN_DIRS)

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def _isolated_failure_log(tmp_path, monkeypatch):
    monkeypatch.setenv("NOVA_FAILURE_LOG", str(tmp_path / "pipeline_failures.jsonl"))
    monkeypatch.setattr(R, "LOCK_PATH", tmp_path / "refresh.lock")
    OL._WARNED.clear()


@pytest.fixture
def real():
    data = json.loads(FIXTURE.read_text())
    archive = pd.DataFrame(data["archive"])
    archive["sequence"] = archive["sequence"].str.upper()
    return archive, data["passes"]


# ---------------------------------------------------------------------------
# Fake host
# ---------------------------------------------------------------------------
IDLE = [{"index": i, "uuid": f"GPU-{i}", "util": 0.0, "mem_mib": 3.0} for i in range(4)]
PRODUCER_ARGV = ("/root/.pixi/envs/metanano/bin/python -m nanobody_opt.run_live_producer "
                 "--target P20809 --exact-scorer /root/nanobody_opt/score_boltzgen_fast_adapter.py")


class FakeProbe:
    def __init__(self, *, table=None, gpus=None, apps=None, gpu_samples=None):
        self.table = table if table is not None else [(1, 0, "/sbin/init")]
        self.gpus = gpus if gpus is not None else IDLE
        self.apps = apps or []
        self.gpu_samples = list(gpu_samples or [])  # per-sample overrides
        self.t = 1000.0
        self.slept = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.slept += seconds
        self.t += seconds

    def process_table(self):
        return list(self.table)

    def gpu_state(self):
        if self.gpu_samples:
            return self.gpu_samples.pop(0)
        return [dict(g) for g in self.gpus]

    def gpu_compute_apps(self):
        return list(self.apps)


def _no_db(monkeypatch):
    """Make any DB/scoring access an immediate test failure."""
    def boom(*a, **k):
        raise AssertionError("guard let the run reach the DB / scorer")
    for name in ("fetch_our_top50", "fetch_db_rivals", "stored_counts", "score_pass",
                 "fetch_rivals", "ours_set"):
        monkeypatch.setattr(R, name, boom)


# ---------------------------------------------------------------------------
# 1. Per-pass rows under multipass_sum10_pass, carrying the real raw metrics
# ---------------------------------------------------------------------------
def _mp():
    """The dashboard pass scorer, whose row contract the refresh must match."""
    path = ROOT / "tools" / "multipass_all_rivals.py"
    spec = importlib.util.spec_from_file_location("multipass_all_rivals_for_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _seqs(passes):
    return sorted(passes[0]["per_nanobody_components"]["0"])


def _raw(passes, k, seq):
    return {m: float(v) for m, v in passes[k]["per_nanobody_components"]["0"][seq]["P20809"].items()}


def _install_fake_scorer(monkeypatch, passes, calls=None):
    """run_guarded stand-in: pass k's real exact_all.json, restricted to the batch."""
    def fake_run_guarded(cmd, watchdog, *, log_path, env=None):
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        batch = {s["sequence"] for s in json.loads(
            Path(cmd[cmd.index("--sequences") + 1]).read_text())["sequences"]}
        run = int(out_dir.name.replace("score", ""))
        payload = json.loads(json.dumps(passes[(run - 1) % len(passes)]))
        wave = payload["per_nanobody_components"]["0"]
        payload["per_nanobody_components"]["0"] = {s: v for s, v in wave.items() if s in batch}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "exact_all.json").write_text(json.dumps(payload))
        if calls is not None:
            calls.append((run, sorted(batch), (env or {}).get("NANOBODY_SCORER_DAEMON")))
    monkeypatch.setattr(R, "run_guarded", fake_run_guarded)


def _top_up(monkeypatch, tmp_path, archive, passes, *, plan, counts=None, ours=frozenset(),
            source="public_multipass_rivals", epochs=None):
    calls, written = [], []
    _install_fake_scorer(monkeypatch, passes, calls)
    watchdog = R.Watchdog(FakeProbe(), deadline=1e12, gpus=[0], self_pid=1)
    n = R.score_and_import(
        source, plan, archive, "0", watchdog=watchdog,
        field={"field": "hf_archive_full", "field_path": "/root/P20809_nanobodies.csv",
               "field_rows": len(archive), "field_sha256": "f" * 64},
        ours=set(ours), counts=counts or {}, epochs=epochs or {}, refresh_epoch=25640,
        env={"NANOBODY_SCORER_DAEMON": "0"},
        writer=lambda obs: written.extend(obs) or len(obs), out_root_base=tmp_path)
    return n, written, calls


def test_rows_carry_the_real_raw_metrics_per_pass(monkeypatch, tmp_path, real):
    archive, passes = real
    seqs = _seqs(passes)
    n, rows, calls = _top_up(monkeypatch, tmp_path, archive, passes,
                             plan={s: 4 for s in seqs})
    assert n == len(rows) == 6 * 4
    assert [c[0] for c in calls] == [1, 2, 3, 4] and all(c[2] == "0" for c in calls)
    for row in rows:
        k = row["pass_index"]
        # metrics_json = this pass's RAW metrics, all ten, exactly as scored
        assert row["metrics"] == _raw(passes, k - 1, row["sequence"])
        assert set(TEN) <= set(row["metrics"])
        assert row["score_kind"] == "multipass_sum10_pass"
        assert row["source"] == "public_multipass_rivals"
        assert row["writer"] == "hf_field_refresh" and row["formula"] == "sum10"
        assert row["cohort"] == "rival" and row["passes"] == R.N_PASSES
        assert row["observation_count"] == 1
        assert row["observation_id"] == (
            f"hf20x_pass:public_multipass_rivals:epoch25640:{row['sequence']}:pass{k}")
        assert row["epoch"] == 25640 and row["epoch_basis"].startswith("run epoch")
        assert row["scorer_protocol"] == "validator-dual-sample-v1"
    assert len({r["observation_id"] for r in rows}) == len(rows)


def test_score_matches_the_dashboard_pass_scorer(real):
    """Same label, same ranking: rank_pass == multipass_all_rivals.rank_pass_payload,
    and the raw metrics == its raw_metrics_from_exact_all, on real passes."""
    archive, passes = real
    mp = _mp()
    for payload in passes:
        want_ranks, _structs, want_raw = mp.rank_pass_payload(payload, archive)
        raw = R.raw_metrics_from_payload(payload)
        assert raw == want_raw
        assert R.rank_pass(raw, archive) == pytest.approx(want_ranks)
    # including a batch member that is also in the archive (dedup, keep the pass)
    dup = json.loads(json.dumps(passes[0]))
    arch_seq = archive["sequence"].iloc[0]
    first = _seqs(passes)[0]
    dup["per_nanobody_components"]["0"][arch_seq] = dup["per_nanobody_components"]["0"][first]
    want, _s, _r = mp.rank_pass_payload(dup, archive)
    assert R.rank_pass(R.raw_metrics_from_payload(dup), archive) == pytest.approx(want)


def test_row_shape_matches_the_multipass_contract(monkeypatch, tmp_path, real):
    """Every key the dashboard writer puts on a pass row is on ours too."""
    archive, passes = real
    mp = _mp()
    ranks, structs, raw = mp.rank_pass_payload(passes[0], archive)
    theirs = mp.pass_observations(epoch=25640, run=1, passes=10, ranks=ranks, raw=raw,
                                  structs=structs, cohort_of={}, field={
                                      "field": "hf_archive_full", "field_path": "x",
                                      "field_rows": 1, "field_sha256": "f"})[0]
    _, rows, _ = _top_up(monkeypatch, tmp_path, archive, passes,
                         plan={s: 1 for s in _seqs(passes)})
    assert set(theirs) <= set(rows[0])
    for key in ("source", "score_kind", "formula", "field"):
        assert rows[0][key] == theirs[key]


def test_db_serialization_is_not_empty(monkeypatch, tmp_path, real):
    """The exact column values update_observation_store would write (no DB)."""
    archive, passes = real
    _, rows, _ = _top_up(monkeypatch, tmp_path, archive, passes,
                         plan={s: 4 for s in _seqs(passes)})
    for row in rows:
        prepared = MN._prepare_observation(row)
        assert prepared is not None
        (oid, epoch, source, kind, score, count, metrics_json, _pose,
         provenance_json) = prepared["observation"]
        assert kind == "multipass_sum10_pass" and source == "public_multipass_rivals"
        assert epoch == 25640 and count == 1 and np.isfinite(score)
        assert metrics_json != "{}"
        assert json.loads(metrics_json) == _raw(passes, row["pass_index"] - 1, row["sequence"])
        prov = json.loads(provenance_json)
        assert {"writer", "formula", "cohort", "pass_index", "passes", "field",
                "field_rows", "field_sha256", "epoch_basis"} <= set(prov)


def test_the_live_reader_counts_and_ranks_these_rows(monkeypatch, tmp_path, real):
    """hf_topup (the live top-up reader, and the shape elite_breadth reads)
    gets every pass back and re-derives sum10 from metrics_json."""
    archive, passes = real
    seqs = _seqs(passes)
    _, rows, _ = _top_up(monkeypatch, tmp_path, archive, passes, plan={s: 4 for s in seqs},
                         source="hf_archive_top200")
    db_rows = []
    for row in rows:
        obs = MN._prepare_observation(row)["observation"]
        db_rows.append((row["sequence"], obs[1], obs[6], obs[8]))
    frame = H._passes_frame(db_rows)
    assert len(frame) == len(seqs) * 4 and frame[TEN].notna().all().all()
    assert set(frame["cohort"]) == {"hf_top200"}
    ranked = H.rank_topup_passes(frame, archive, min_passes=4)
    assert sorted(r["sequence"] for r in ranked) == seqs
    assert all(r["passes"] == 4 for r in ranked)


def test_depth_top_up_scores_only_the_missing_passes(monkeypatch, tmp_path, real):
    """Stored 18 -> 2 more; stored 0 -> 4 (plan); ids continue the depth."""
    archive, passes = real
    a, b = _seqs(passes)[:2]
    _, rows, calls = _top_up(monkeypatch, tmp_path, archive, passes,
                             plan={a: 2, b: 4}, counts={a: 18, b: 0})
    assert [c[1] for c in calls] == [sorted([a, b]), sorted([a, b]), [b], [b]]
    assert sorted(r["pass_index"] for r in rows if r["sequence"] == a) == [19, 20]
    assert sorted(r["pass_index"] for r in rows if r["sequence"] == b) == [1, 2, 3, 4]


def test_old_rival_keeps_its_own_epoch(monkeypatch, tmp_path, real):
    """A rival last seen at 25500 must not be pulled into the reader's 16-epoch
    'current rival field' window by today's top-up."""
    archive, passes = real
    a = _seqs(passes)[0]
    _, rows, _ = _top_up(monkeypatch, tmp_path, archive, passes, plan={a: 1},
                         epochs={a: (25500, "latest public_multipass_rivals row epoch")})
    assert rows[0]["epoch"] == 25500
    assert rows[0]["observation_id"].startswith("hf20x_pass:public_multipass_rivals:epoch25640:")


def test_incomplete_passes_are_reported_not_silently_dropped(monkeypatch, tmp_path, real, capsys):
    archive, passes = real
    seq = _seqs(passes)[2]
    del passes[1]["per_nanobody_components"]["0"][seq]["P20809"]["interaction_pae"]
    _, rows, _ = _top_up(monkeypatch, tmp_path, archive, passes,
                         plan={s: 4 for s in _seqs(passes)})
    assert len(rows) == 6 * 4 - 1
    assert (seq, 2) not in {(r["sequence"], r["pass_index"]) for r in rows}
    assert "HF_REFRESH_INCOMPLETE_METRICS" in capsys.readouterr().err


def test_cohort_ours_and_hf_top200(monkeypatch, tmp_path, real):
    archive, passes = real
    seqs = _seqs(passes)
    _, rows, _ = _top_up(monkeypatch, tmp_path, archive, passes, plan={s: 1 for s in seqs},
                         ours={seqs[0]}, source="hf_archive_top200")
    by = {r["sequence"]: r["cohort"] for r in rows}
    assert by[seqs[0]] == "ours" and {by[s] for s in seqs[1:]} == {"hf_top200"}


def test_abort_writes_the_completed_passes_then_raises(monkeypatch, tmp_path, real):
    """Finished GPU work is kept (one batched write), and the abort stays loud."""
    archive, passes = real
    seqs = _seqs(passes)
    probe = FakeProbe()
    _install_fake_scorer(monkeypatch, passes)
    real_score_pass = R.score_pass

    def slow(*a, **k):
        out = real_score_pass(*a, **k)
        probe.t += 600.0
        return out
    monkeypatch.setattr(R, "score_pass", slow)
    written = []
    watchdog = R.Watchdog(probe, deadline=probe.t + 1500.0, gpus=[0], self_pid=1)
    with pytest.raises(R.GuardAborted, match=r"pass 3/4.*imported 12 completed"):
        R.score_and_import("public_multipass_rivals", {s: 4 for s in seqs}, archive, "0",
                           watchdog=watchdog, field={}, ours=set(), counts={}, epochs={},
                           refresh_epoch=1, env={},
                           writer=lambda obs: written.extend(obs) or len(obs),
                           out_root_base=tmp_path)
    assert len(written) == 12 and {r["pass_index"] for r in written} == {1, 2}


# ---------------------------------------------------------------------------
# Label registry: accepted under the real registry, refused when closed
# ---------------------------------------------------------------------------
def _with_label(monkeypatch, **changes):
    labels = dict(OL.LABELS)
    labels[R.SCORE_KIND] = dataclasses.replace(OL.LABELS[R.SCORE_KIND], **changes)
    monkeypatch.setattr(OL, "LABELS", labels)


def test_registry_lists_the_writer():
    label = OL.LABELS["multipass_sum10_pass"]
    assert label.strict and "hf_field_refresh" in label.writers
    assert label.formula == R.FORMULA_ID and R.SOURCE in label.sources


def test_rows_pass_the_real_strict_registry(monkeypatch, tmp_path, real):
    archive, passes = real
    R.assert_label_open()
    _, rows, _ = _top_up(monkeypatch, tmp_path, archive, passes,
                         plan={s: 2 for s in _seqs(passes)})
    OL.check_observations(rows)  # raises on a wrong writer / formula / source


def test_an_unlisted_writer_or_closed_label_refuses_before_any_gpu_time(monkeypatch):
    _with_label(monkeypatch, writers=frozenset({"multipass_all_rivals"}))
    with pytest.raises(OL.LabelViolation, match="hf_field_refresh"):
        R.assert_label_open()
    _with_label(monkeypatch, writers=frozenset())
    with pytest.raises(OL.LabelViolation, match="closed"):
        R.assert_label_open()


# ---------------------------------------------------------------------------
# 2. Guard: refuses while the live producer needs the GPUs
# ---------------------------------------------------------------------------
def test_refuses_while_the_live_producer_is_mid_race(monkeypatch, capsys, tmp_path):
    """Today's incident: the producer is racing on all GPUs. The refresh
    must exit 3 and do nothing: no DB read, no scorer, no write."""
    _no_db(monkeypatch)
    busy = [{"index": i, "uuid": f"GPU-{i}", "util": 97.0, "mem_mib": 18000.0} for i in range(8)]
    probe = FakeProbe(table=[(1, 0, "/sbin/init"), (4242, 1, PRODUCER_ARGV),
                             (4300, 4242, "python -m nanobody_opt.score_boltzgen_waves --gpus 0,1")],
                      gpus=busy, apps=[{"uuid": "GPU-0", "pid": 4300}])
    rc = R.main(["--allow-unconsumed"], probe=probe,
                writer=lambda obs: pytest.fail("wrote while the producer raced"))
    assert rc == R.EXIT_REFUSED == 3
    err = capsys.readouterr().err
    assert "HF_REFRESH_GUARD_REFUSED" in err and "run_live_producer" in err
    entries = [json.loads(l) for l in (tmp_path / "pipeline_failures.jsonl").read_text().splitlines()]
    assert entries[-1]["marker"] == "HF_REFRESH_GUARD_REFUSED"


def test_refuses_busy_gpus_even_if_the_producer_process_is_invisible(monkeypatch, capsys):
    """E.g. the producer runs in another container: only the GPUs show it."""
    _no_db(monkeypatch)
    busy = [dict(g) for g in IDLE]
    busy[2]["util"] = 88.0
    probe = FakeProbe(gpus=busy)
    assert R.main(["--allow-unconsumed"], probe=probe) == R.EXIT_REFUSED
    assert "HF_REFRESH_GUARD_REFUSED GPU 2 busy" in capsys.readouterr().err


def test_one_busy_sample_in_the_window_refuses():
    """Not a single instantaneous sample: the GPUs must be idle in ALL samples."""
    busy = [dict(g) for g in IDLE]
    busy[0]["mem_mib"] = 9000.0
    probe = FakeProbe(gpu_samples=[IDLE, IDLE, IDLE, busy, IDLE, IDLE])
    with pytest.raises(R.GuardRefused, match="sample 4/6"):
        R.assert_safe_window([0, 1, 2, 3], probe, self_pid=999)
    assert probe.slept == pytest.approx(3 * R.PREFLIGHT_INTERVAL_S)


def test_resident_compute_process_refuses_even_at_low_util():
    """The idle-but-resident scorer daemon (~0.5 GB/GPU, 0% util) still owns the GPU."""
    probe = FakeProbe(apps=[{"uuid": "GPU-1", "pid": None}])
    with pytest.raises(R.GuardRefused, match="compute process"):
        R.assert_safe_window([0, 1, 2, 3], probe, self_pid=999)


def test_idle_dedicated_box_is_allowed():
    probe = FakeProbe()
    R.assert_safe_window([0, 1, 2, 3], probe, self_pid=999)
    assert probe.slept == pytest.approx((R.PREFLIGHT_SAMPLES - 1) * R.PREFLIGHT_INTERVAL_S)


def test_missing_gpu_and_unreadable_nvidia_smi_fail_closed(monkeypatch):
    with pytest.raises(R.GuardRefused, match="not present"):
        R.assert_safe_window([0, 7], FakeProbe(), self_pid=999)

    def no_smi(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(R.subprocess, "run", no_smi)
    with pytest.raises(R.GuardRefused, match="nvidia-smi unavailable"):
        R.SystemProbe().gpu_state()


def test_unparseable_gpu_value_fails_closed(monkeypatch):
    monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, stdout="0, GPU-0, [N/A], 10\n", stderr=""))
    with pytest.raises(R.GuardRefused, match="unreadable"):
        R.SystemProbe().gpu_state()


def test_own_descendants_are_not_contenders_but_ancestors_are():
    table = [(1, 0, "/sbin/init"),
             (50, 1, "bash /root/session_watchdog.sh"),           # ancestor producer session
             (100, 50, "python /root/hf_field_refresh.py"),       # us
             (101, 100, "python -m nanobody_opt.score_boltzgen_waves --gpus 0,1"),  # our scorer
             (102, 101, "python -c from multiprocessing.spawn import spawn_main")]
    hits = R.foreign_contenders(table, 100)
    assert len(hits) == 1 and "pid=50" in hits[0]


def test_wall_and_budget_flags_have_hard_ceilings(monkeypatch, capsys):
    _no_db(monkeypatch)
    assert R.main(["--allow-unconsumed", "--max-wall-hours", "10"], probe=FakeProbe()) == 3
    assert R.main(["--allow-unconsumed", "--max-sequence-passes", "999999"],
                  probe=FakeProbe()) == 3
    err = capsys.readouterr().err
    assert "GUARD_REFUSED --max-wall-hours 10" in err
    assert "GUARD_REFUSED --max-sequence-passes 999999" in err


def test_a_second_instance_is_refused(tmp_path):
    first = R.acquire_lock(tmp_path / "l")
    try:
        with pytest.raises(R.GuardRefused, match="another"):
            R.acquire_lock(tmp_path / "l")
    finally:
        first.close()


def test_budget_counts_missing_passes_and_defers_in_source_order():
    counts = {"a": 20, "b": 15, "c": 0, "d": 10}
    planned, deferred, deep = R.plan_budget(
        [("public_multipass_rivals", ["a", "b", "c", "d"]), ("hf_archive_top200", ["e"])],
        counts, max_sequence_passes=40)
    # a is at depth; b needs 5, c 20, d 10 -> 35; e needs 20 > 5 left -> deferred
    assert planned == [("public_multipass_rivals", {"b": 5, "c": 20, "d": 10}),
                       ("hf_archive_top200", {})]
    assert deep == {"public_multipass_rivals": 1, "hf_archive_top200": 0}
    assert deferred == {"public_multipass_rivals": 0, "hf_archive_top200": 1}


def test_disabled_by_default_still_exits_2(monkeypatch):
    monkeypatch.delenv("HF_FIELD_REFRESH_ENABLED", raising=False)
    _no_db(monkeypatch)
    assert R.main([], probe=FakeProbe()) == R.EXIT_DISABLED




# ---------------------------------------------------------------------------
# Real processes: ps sees a producer; the watchdog kills a running scorer
# ---------------------------------------------------------------------------
class _Detached:
    """A fake producer OUTSIDE our process tree (double fork, reparented to
    init), as the real producer is: our own descendants never count."""

    def __init__(self):
        out = subprocess.run(
            ["sh", "-c", f"'{sys.executable}' -c 'import time; time.sleep(60)' "
                         "nanobody_opt.run_live_producer >/dev/null 2>&1 & echo $!"],
            capture_output=True, text=True, check=True)
        self.pid = int(out.stdout.strip())

    def kill(self):
        import os
        import signal
        try:
            os.kill(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def wait(self):
        pass


def _fake_producer():
    return _Detached()


class RealPsIdleGpus(R.SystemProbe):
    def gpu_state(self):
        return [dict(g) for g in IDLE]

    def gpu_compute_apps(self):
        return []


def test_real_process_table_sees_a_producer():
    import os
    producer = _fake_producer()
    try:
        time.sleep(0.3)
        hits = R.foreign_contenders(R.SystemProbe().process_table(), os.getpid())
        assert any(f"pid={producer.pid}" in h for h in hits)
        with pytest.raises(R.GuardRefused, match="live producer"):
            R.assert_safe_window([0], RealPsIdleGpus(), self_pid=os.getpid(), samples=1)
    finally:
        producer.kill()
        producer.wait()


def test_watchdog_kills_the_scorer_when_the_producer_starts(tmp_path):
    import os
    probe = RealPsIdleGpus()
    watchdog = R.Watchdog(probe, deadline=probe.now() + 3600, gpus=[0],
                          self_pid=os.getpid(), poll_s=0.2, kill_grace_s=2.0).start()
    producers = []
    timer = threading.Timer(1.0, lambda: producers.append(_fake_producer()))
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(R.GuardAborted, match="live producer"):
            R.run_guarded(["sleep", "60"], watchdog, log_path=tmp_path / "scorer.log")
        assert time.monotonic() - started < 15
    finally:
        watchdog.stop()
        timer.cancel()
        for p in producers:
            p.kill()
            p.wait()


def test_hard_deadline_kills_a_running_scorer(tmp_path):
    import os
    probe = RealPsIdleGpus()
    watchdog = R.Watchdog(probe, deadline=probe.now() + 1.0, gpus=[0],
                          self_pid=os.getpid(), poll_s=30.0, kill_grace_s=2.0)
    started = time.monotonic()
    with pytest.raises(R.GuardAborted, match="deadline"):
        R.run_guarded(["sleep", "60"], watchdog, log_path=tmp_path / "scorer.log")
    assert time.monotonic() - started < 10


def test_main_end_to_end_on_an_idle_box(monkeypatch, tmp_path, real, capsys):
    """Full main() wiring on an idle box, under the REAL registry: guard OK ->
    label OK -> depth plan -> guarded passes -> one batched write per source.
    Ours are never scored or written by default."""
    import types

    import nanobody_opt
    archive, passes = real
    seqs = _seqs(passes)
    csv = tmp_path / "archive.csv"
    archive.to_csv(csv, index=False)
    monkeypatch.setattr(R, "ARCHIVE", str(csv))
    monkeypatch.setattr(R, "OUT_ROOT", tmp_path / "out")
    fake_db = types.ModuleType("nanobody_opt.db")

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False
    fake_db.connect = lambda: _Conn()
    monkeypatch.setitem(sys.modules, "nanobody_opt.db", fake_db)
    monkeypatch.setattr(nanobody_opt, "db", fake_db, raising=False)
    ours = {seqs[0]}
    monkeypatch.setattr(R, "fetch_our_top50", lambda conn: {seqs[0]})
    monkeypatch.setattr(R, "ours_set", lambda conn, extra, cache_path=None: (ours | extra, {}))
    monkeypatch.setattr(R, "fetch_db_rivals", lambda conn: set(seqs[:3]))  # includes ours
    monkeypatch.setattr(R, "fetch_rivals", lambda epoch: set(seqs[2:4]))
    monkeypatch.setattr(R, "fetch_hf_top200", lambda a: set(seqs[3:]))
    # seqs[1] already at depth; seqs[2] needs 1; seqs[3] needs 20 (budget: 18 left)
    counts = {seqs[1]: 20, seqs[2]: 19, seqs[3]: 0, seqs[4]: 18, seqs[5]: 19}
    monkeypatch.setattr(R, "stored_counts", lambda conn, s: {k: counts.get(k, 0) for k in s})
    monkeypatch.setattr(R, "latest_epochs", lambda conn, s, e: {k: (e, "run epoch") for k in s})
    calls = []
    _install_fake_scorer(monkeypatch, passes, calls)
    batches = []
    rc = R.main(["--allow-unconsumed", "--epoch", "25640", "--max-sequence-passes", "4"],
                probe=FakeProbe(), writer=lambda obs: batches.append(obs) or len(obs))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "HF_REFRESH_GUARD_OK gpus=0,1,2,3" in out
    written = [o for b in batches for o in b]
    # rivals: s2 (1 pass); s3 needs 20 > budget -> deferred. hf: s4 (2), s5 (1)
    assert sorted((o["sequence"], o["pass_index"]) for o in written) == sorted(
        [(seqs[2], 20), (seqs[4], 19), (seqs[4], 20), (seqs[5], 20)])
    assert seqs[0] not in {o["sequence"] for o in written}
    assert all(c[1] != [] and seqs[0] not in c[1] for c in calls)
    assert {o["hf20x_source"]: o["cohort"] for o in written} == {
        "public_multipass_rivals": "rival", "hf_archive_top200": "hf_top200"}
    assert len(batches) == 2  # one batched update_observation_store call per source
    OL.check_observations(written)
    assert "REFRESH_DONE imported=4 deferred=1" in out

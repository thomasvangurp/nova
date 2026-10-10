"""hf_field_refresh: raw metrics reach metrics_json, and the GPU/time guard.

Two bugs fixed on 2026-10-10:

1. Every row the refresh wrote had ``metrics_json='{}'``: score_pass() read
   the raw BoltzGen metrics from ``exact_all.json`` and threw them away.
   The metrics tests here use REAL scorer passes (tests/fixtures/
   hf_field_refresh/real_passes.json) and send the rows through the real
   ``mutation_network._prepare_observation``. That is the function that
   serializes ``metrics_json`` / ``provenance_json`` for the DB.
2. The only contention guard was a single nvidia-smi sample in the loop
   wrapper, plus a "stale for 3h, run anyway" override, so a run started
   by hand had no guard at all. The guard tests simulate the live
   producer mid-race and check that the script refuses before it touches
   the DB or a GPU. A watchdog test uses real processes: it kills a
   running scorer when a producer process appears.
"""
from __future__ import annotations

import dataclasses
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
    monkeypatch.setattr(R, "already_scored", boom)
    monkeypatch.setattr(R, "score_pass", boom)
    monkeypatch.setattr(R, "fetch_rivals", boom)


# ---------------------------------------------------------------------------
# 1. metrics_json is populated from the real raw metrics
# ---------------------------------------------------------------------------
def _run_real_passes(monkeypatch, tmp_path, archive, passes, *, our=frozenset()):
    """score_and_import over the fixture passes. A fake scorer writes pass k's
    exact_all.json where the real one would."""
    monkeypatch.setattr(R, "N_PASSES", len(passes))
    calls = []

    def fake_run_guarded(cmd, watchdog, *, log_path, env=None):
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        run = int(out_dir.name.replace("score", ""))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "exact_all.json").write_text(json.dumps(passes[run - 1]))
        calls.append((run, env.get("NANOBODY_SCORER_DAEMON")))

    monkeypatch.setattr(R, "run_guarded", fake_run_guarded)
    written = []
    seqs = sorted(passes[0]["per_nanobody_components"]["0"])
    watchdog = R.Watchdog(FakeProbe(), deadline=1e12, gpus=[0], self_pid=1)
    n = R.score_and_import(
        "public_multipass_rivals", seqs, archive, "0", watchdog=watchdog,
        field={"field": "hf_archive_full_plus_scored_batch", "field_rows": len(archive),
               "field_sha256": "f" * 64},
        our=set(our), refresh_epoch=25640, env={"NANOBODY_SCORER_DAEMON": "0"},
        writer=lambda obs: written.extend(obs) or len(obs), out_root_base=tmp_path)
    return n, written, seqs, calls


def _raw(passes, k, seq):
    return {m: float(v) for m, v in passes[k]["per_nanobody_components"]["0"][seq]["P20809"].items()}


def test_rows_carry_the_real_raw_metrics(monkeypatch, tmp_path, real):
    archive, passes = real
    n, obs, seqs, calls = _run_real_passes(monkeypatch, tmp_path, archive, passes)
    assert n == len(seqs) == len(obs) == 6
    assert [c[0] for c in calls] == [1, 2, 3, 4]
    assert all(c[1] == "0" for c in calls)  # never through the shared daemon
    for row in obs:
        s = row["sequence"]
        raws = [_raw(passes, k, s) for k in range(len(passes))]
        # metrics_json = per-metric mean over the passes, all ten metrics
        assert set(TEN) <= set(row["metrics"])
        for m in TEN:
            assert row["metrics"][m] == pytest.approx(np.mean([r[m] for r in raws]), rel=1e-12)
        # the raw per-pass values are preserved exactly, in pass order
        assert row["per_pass_metrics"] == raws
        # score = mean of the per-pass sum10 against archive + batch
        per_pass = [R.rank_pass({q: _raw(passes, k, q) for q in seqs}, archive)[s]
                    for k in range(len(passes))]
        assert row["per_pass_scores"] == pytest.approx(per_pass)
        assert row["score"] == pytest.approx(np.mean(per_pass))
        assert row["observation_count"] == 4 and row["passes"] == 4
        assert row["writer"] == "hf_field_refresh" and row["formula"] == "sum10"
        assert row["cohort"] == "rival" and row["refresh_epoch"] == 25640
        assert row["scorer_protocol"] == "validator-dual-sample-v1"
        # the per-pass metrics differ between passes, so a mean really is a mean
    assert any(len({json.dumps(p, sort_keys=True) for p in r["per_pass_metrics"]}) > 1 for r in obs)


def test_db_serialization_is_not_empty(monkeypatch, tmp_path, real):
    """The exact bytes update_observation_store would write (no DB needed)."""
    archive, passes = real
    _, obs, _, _ = _run_real_passes(monkeypatch, tmp_path, archive, passes)
    for row in obs:
        prepared = MN._prepare_observation(row)
        assert prepared is not None
        (oid, epoch, source, kind, score, count, metrics_json, _pose,
         provenance_json) = prepared["observation"]
        assert oid == f"hf20x:{row['sequence']}" and kind == R.SCORE_KIND
        assert epoch is None  # column unchanged; the run epoch is in provenance
        assert metrics_json != "{}"
        metrics = json.loads(metrics_json)
        assert set(TEN) <= set(metrics) and all(np.isfinite(list(metrics.values())))
        prov = json.loads(provenance_json)
        assert len(prov["per_pass_metrics"]) == 4
        assert {"writer", "formula", "cohort", "passes", "field_sha256", "field_rows",
                "per_pass_scores", "refresh_epoch"} <= set(prov)
        # scaled to 20 passes the provenance stays far below a TEXT column (64 KB)
        assert len(provenance_json) * 20 / 4 < 60_000


def test_stored_passes_are_usable_by_a_read_time_consumer(monkeypatch, tmp_path, real):
    """A consumer that re-derives sum10 per pass against its own reference field
    (hf_topup's reader, and the planned elite_breadth) gets every pass back."""
    archive, passes = real
    _, obs, seqs, _ = _run_real_passes(monkeypatch, tmp_path, archive, passes)
    rows = []
    for row in obs:
        prov = json.loads(MN._prepare_observation(row)["observation"][8])
        for pm in prov["per_pass_metrics"]:
            rows.append((row["sequence"], 25640, json.dumps(pm), json.dumps({"cohort": "rival"})))
    frame = H._passes_frame(rows)
    assert len(frame) == len(seqs) * 4
    assert frame[TEN].notna().all().all()
    derived = H.sum10_against_field(frame, archive)
    assert np.isfinite(derived).all()


def test_incomplete_sequences_are_reported_not_silently_dropped(monkeypatch, tmp_path, real, capsys):
    archive, passes = real
    seq = sorted(passes[0]["per_nanobody_components"]["0"])[2]
    del passes[1]["per_nanobody_components"]["0"][seq]["P20809"]["interaction_pae"]
    _, obs, _, _ = _run_real_passes(monkeypatch, tmp_path, archive, passes)
    assert len(obs) == 5 and seq not in {o["sequence"] for o in obs}
    assert "HF_REFRESH_INCOMPLETE_METRICS" in capsys.readouterr().err


def test_our_sequence_gets_cohort_ours(monkeypatch, tmp_path, real):
    archive, passes = real
    ours = sorted(passes[0]["per_nanobody_components"]["0"])[0]
    _, obs, _, _ = _run_real_passes(monkeypatch, tmp_path, archive, passes, our={ours})
    by = {o["sequence"]: o["cohort"] for o in obs}
    assert by[ours] == "ours" and sorted(set(by.values())) == ["ours", "rival"]


def test_rank_pass_matches_the_pre_fix_scoring(real):
    """The stored score keeps its scale: rank_pass is the old score_pass math."""
    archive, passes = real
    payload = passes[0]
    # Pre-fix score_pass body (git 48a23ce..a3ad1c9), verbatim minus the subprocess.
    wave = payload["per_nanobody_components"]["0"]
    rows = {}
    for seq, comp in wave.items():
        m = comp.get(R.TARGET, {})
        rows[seq] = {k: float(v) for k, v in m.items()
                     if k in R.METRICS + ["liability_num_violations"]
                     and isinstance(v, (int, float))}
    rival = pd.DataFrame(rows).T
    arch_s = archive.set_index("sequence")[R.METRICS + ["liability_num_violations"]]
    field = pd.concat([arch_s, rival])
    scores = R.sum_of_ten(field)
    old = {s: float(scores[scores.index == s].iloc[-1]) for s in rival.index}
    new = R.rank_pass(R.raw_metrics_from_payload(payload, set(wave)), archive)
    assert new == pytest.approx(old)


# ---------------------------------------------------------------------------
# Label preflight (before any GPU time)
# ---------------------------------------------------------------------------
def _with_label(monkeypatch, **changes):
    labels = dict(OL.LABELS)
    labels[R.SCORE_KIND] = dataclasses.replace(OL.LABELS[R.SCORE_KIND], **changes)
    monkeypatch.setattr(OL, "LABELS", labels)


def test_rows_pass_a_strict_open_label(monkeypatch, tmp_path, real):
    archive, passes = real
    _with_label(monkeypatch, strict=True, writers=frozenset({"hf_field_refresh"}),
                formula="sum10")
    R.assert_label_open()
    _, obs, _, _ = _run_real_passes(monkeypatch, tmp_path, archive, passes)
    OL.check_observations(obs)  # raises on a wrong writer / formula / source


def test_a_closed_label_refuses_before_any_gpu_time(monkeypatch):
    _with_label(monkeypatch, strict=True, writers=frozenset())
    with pytest.raises(OL.LabelViolation):
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


def test_budget_defers_in_source_order():
    planned, deferred = R.plan_budget(
        [("our_top50_20x", list("abc")), ("public_multipass_rivals", list("defgh")),
         ("hf_archive_top200", list("ij"))], max_sequence_passes=6 * R.N_PASSES)
    assert [len(s) for _, s in planned] == [3, 3, 0]
    assert deferred == {"our_top50_20x": 0, "public_multipass_rivals": 2, "hf_archive_top200": 2}


def test_disabled_by_default_still_exits_2(monkeypatch):
    monkeypatch.delenv("HF_FIELD_REFRESH_ENABLED", raising=False)
    _no_db(monkeypatch)
    assert R.main([], probe=FakeProbe()) == R.EXIT_DISABLED


def test_deadline_stops_the_next_pass_before_it_starts(monkeypatch, tmp_path, real):
    """A pass that would not fit in the remaining wall clock is never started;
    nothing is written for that source."""
    archive, passes = real
    monkeypatch.setattr(R, "N_PASSES", 4)
    probe = FakeProbe()

    def slow_pass(cmd, watchdog, *, log_path, env=None):
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "exact_all.json").write_text(json.dumps(passes[0]))
        probe.t += 600.0  # each pass takes 10 min
    monkeypatch.setattr(R, "run_guarded", slow_pass)
    watchdog = R.Watchdog(probe, deadline=probe.t + 1500.0, gpus=[0], self_pid=1)
    with pytest.raises(R.GuardAborted, match="pass 3/4"):
        R.score_and_import("public_multipass_rivals", ["X"], archive, "0", watchdog=watchdog,
                           field={}, our=set(), refresh_epoch=1, env={},
                           writer=lambda obs: pytest.fail("partial source written"),
                           out_root_base=tmp_path)


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
    """Full main() wiring on an idle box: guard OK -> label OK -> budget ->
    guarded passes -> one writer call per source with populated metrics."""
    import types

    import nanobody_opt
    archive, passes = real
    seqs = sorted(passes[0]["per_nanobody_components"]["0"])
    csv = tmp_path / "archive.csv"
    archive.to_csv(csv, index=False)
    monkeypatch.setattr(R, "ARCHIVE", str(csv))
    monkeypatch.setattr(R, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(R, "N_PASSES", 4)
    _with_label(monkeypatch, strict=True, writers=frozenset({"hf_field_refresh"}))
    fake_db = types.ModuleType("nanobody_opt.db")

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False
    fake_db.connect = lambda: _Conn()
    monkeypatch.setitem(sys.modules, "nanobody_opt.db", fake_db)
    monkeypatch.setattr(nanobody_opt, "db", fake_db, raising=False)
    monkeypatch.setattr(R, "already_scored", lambda conn: set())
    monkeypatch.setattr(R, "fetch_our_top50", lambda conn: {seqs[0]})
    monkeypatch.setattr(R, "fetch_db_rivals", lambda conn: set(seqs[:4]))  # overlaps ours
    monkeypatch.setattr(R, "fetch_rivals", lambda epoch: set(seqs[3:5]))
    monkeypatch.setattr(R, "fetch_hf_top200", lambda a: set(seqs[4:]))

    def fake_run_guarded(cmd, watchdog, *, log_path, env=None):
        assert env["NANOBODY_SCORER_DAEMON"] == "0"
        assert watchdog.remaining() > 0 and watchdog.abort_reason is None
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        run = int(out_dir.name.replace("score", ""))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "exact_all.json").write_text(json.dumps(passes[run - 1]))
    monkeypatch.setattr(R, "run_guarded", fake_run_guarded)
    batches = []
    rc = R.main(["--allow-unconsumed", "--epoch", "25640", "--max-sequence-passes", "16"],
                probe=FakeProbe(), writer=lambda obs: batches.append(obs) or len(obs))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "HF_REFRESH_GUARD_OK gpus=0,1,2,3" in out and "REFRESH_DONE imported=4" in out
    # budget 16 = 4 sequences x 4 passes; each sequence scored once, in source order
    assert [[o["source"] for o in b][0] for b in batches] == ["our_top50_20x", "public_multipass_rivals"]
    written = [o for b in batches for o in b]
    assert sorted(o["sequence"] for o in written) == seqs[:4]
    assert all(set(TEN) <= set(o["metrics"]) for o in written)
    assert "deferred={'our_top50_20x': 0, 'public_multipass_rivals': 1, 'hf_archive_top200': 1}" in out
    assert {o["sequence"]: o["cohort"] for o in written}[seqs[0]] == "ours"

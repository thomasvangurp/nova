import json
import shlex
import sys
import time

from nanobody_opt_v2.confirmation_eligibility import fill
from nanobody_opt_v2 import sequential_exact_race as race


def test_eighty_percent_attrition_does_not_become_eighty_percent_empty_seats():
    ordered = [f"c{i}" for i in range(1000)]
    calls = []
    def gate(batch, timeout, index, mode):
        calls.extend(batch)
        return [s for s in batch if int(s[1:]) >= 126]
    selected, receipt = fill(ordered=ordered, initial=ordered[:147], width=147,
        cached=lambda s: None, gate=gate, seconds_left=lambda: 1000)
    assert selected == ordered[126:273]
    assert len(calls) == len(set(calls)) == 273
    assert receipt["refilled_candidates"] == 126
    assert receipt["gated_candidates"] == 147


def test_deadline_reserves_repeats_and_refills_using_cached_passes_through_full_gate():
    ordered = [f"c{i}" for i in range(1000)]
    remaining = [100.0]
    modes = []
    def cached(s):
        i = int(s[1:])
        return False if i < 10 else True if i >= 200 else None
    def gate(batch, timeout, index, mode):
        assert timeout <= remaining[0]
        assert not any(int(s[1:]) < 10 for s in batch)
        modes.append(mode)
        if mode == "ranked":
            remaining[0] -= 65
            return []
        remaining[0] -= 1
        return batch
    selected, receipt = fill(ordered=ordered, initial=ordered[:147], width=147,
        cached=cached, gate=gate, seconds_left=lambda: remaining[0])
    assert selected == ordered[200:347]
    assert modes == ["ranked", "cached_refill"]
    assert receipt["known_measured_failures_skipped"] == 10
    assert receipt["deadline_limited"]
    assert remaining[0] == 34


def test_no_time_does_not_start_work_or_admit_unchecked_cache_entries():
    calls = []
    selected, receipt = fill(ordered=["c"], initial=["c"], width=1,
        cached=lambda s: True, gate=lambda *a: calls.append(a), seconds_left=lambda: 0)
    assert selected == calls == []
    assert receipt["deadline_limited"]


def test_partial_timeout_cache_checkpoint_still_requires_full_current_gate():
    state = {"c": None}
    remaining = [100.0]
    def gate(batch, timeout, index, mode):
        if mode == "ranked":
            state["c"] = True
            remaining[0] = 30
            return []
        assert mode == "cached_refill"
        return batch
    selected, receipt = fill(ordered=["c"], initial=["c"], width=1,
        cached=state.get, gate=gate, seconds_left=lambda: remaining[0])
    assert selected == ["c"]
    assert receipt["cached_refill_candidates"] == 1


def test_timed_out_gate_kills_descendants_and_cannot_read_old_output(tmp_path):
    marker = tmp_path / "orphan-finished"
    script = tmp_path / "gate.py"
    child = f"import time,pathlib;time.sleep(.7);pathlib.Path({str(marker)!r}).write_text('bad')"
    script.write_text(f"import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c',{child!r}])\ntime.sleep(20)\n")
    output = tmp_path / "layer2_gate/layer2.json"
    output.parent.mkdir()
    output.write_text(json.dumps({"candidates": [{"sequence": "c", "layer2_passed": True}]}))
    assert race._layer2_gate(["c"], command=[sys.executable, str(script)],
        root=tmp_path, target="TEST", timeout_seconds=.2) == []
    time.sleep(.8)
    assert not marker.exists()
    assert not output.exists()

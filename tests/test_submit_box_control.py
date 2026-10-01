"""The guards on the one box that signs.

Nothing here talks to the provider. What is tested is the part that decides
what to spend on, what to refuse to act on, and when the lease ends -- the
three places where a mistake is either money or a missed epoch.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def control():
    spec = importlib.util.spec_from_file_location(
        "_submit_box", ROOT / "tools" / "submit_box_control.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _offer(**overrides):
    base = dict(id=1, gpu_name="RTX 3090", num_gpus=4, gpu_ram=24576,
                disk_space=1414.0, dph_total=1.07, reliability2=0.9994,
                direct_port_count=98, cpu_cores_effective=32.0,
                geolocation="Finland, FI")
    base.update(overrides)
    return base


def test_the_offer_we_rented_is_acceptable(control):
    assert control._acceptable(_offer())


def test_turing_and_volta_are_refused_however_cheap(control):
    """Boltz's fused kernels need Ampere and bf16 does not exist below it, so
    the four-card Q RTX 6000 at a quarter of the price is not an option."""
    assert not control._acceptable(_offer(gpu_name="Q RTX 6000", dph_total=0.53))
    assert not control._acceptable(_offer(gpu_name="Tesla V100", dph_total=0.86))


def test_a_mining_card_is_refused(control):
    """CMP 170HX shows four cards and 64 GB for $1.34 and cannot run Boltz."""
    assert not control._acceptable(_offer(gpu_name="CMP 170HX", gpu_ram=65536))


def test_an_unreliable_host_is_refused_at_any_price(control):
    """This box misses an epoch when it is down, so reliability outranks price:
    0.99 is about ninety minutes a week, and an epoch is seventy-two minutes."""
    assert not control._acceptable(_offer(reliability2=0.99, dph_total=0.40))


def test_a_starved_host_is_refused(control):
    """Eight effective cores per GPU; below that the scorer contends."""
    assert not control._acceptable(_offer(cpu_cores_effective=16.0))


def test_the_region_that_could_not_reach_docker_hub_is_refused(control):
    assert not control._acceptable(_offer(geolocation="Taiwan, TW"))


def test_an_over_budget_offer_is_refused(control):
    assert not control._acceptable(_offer(dph_total=control.MAXIMUM_DPH + 0.01))


def test_the_guard_refuses_to_act_on_a_box_that_is_not_ours(control):
    """`stop` and `destroy` resolve through this. The forbidden set names
    production and both bank builders, so a manifest that has been edited,
    swapped or stale cannot aim a destroy at the box holding the bank."""
    saved = dict(id=51598530, label=control.LABEL, num_gpus=4)
    with pytest.raises(RuntimeError, match="instance_guard_failed"):
        control.guarded(dict(id=51598530, label=control.LABEL, num_gpus=4), saved)


def test_the_guard_refuses_a_mismatched_label_or_size(control):
    saved = dict(id=51676703, label=control.LABEL, num_gpus=4)
    for host in (dict(id=51676704, label=control.LABEL, num_gpus=4),
                 dict(id=51676703, label="something-else", num_gpus=4),
                 dict(id=51676703, label=control.LABEL, num_gpus=8)):
        with pytest.raises(RuntimeError, match="instance_guard_failed"):
            control.guarded(host, saved)


def test_the_guard_accepts_the_box_we_rented(control):
    saved = dict(id=51676703, label=control.LABEL, num_gpus=4)
    host = dict(id=51676703, label=control.LABEL, num_gpus=4)
    assert control.guarded(host, saved) is host


def test_every_forbidden_id_is_a_box_we_must_not_touch(control):
    """Production, the NIM epoch box, and both bank builders."""
    assert control.FORBIDDEN == {51125470, 51335750, 51598530, 51620065}


def test_extending_without_an_horizon_is_refused(control, capsys):
    """A lease that can be extended by an argumentless call is a lease that
    gets extended by a copied command line."""
    assert control.extend(None) == 2


def test_extending_moves_the_deadline_to_the_hours_given(control, tmp_path,
                                                         monkeypatch):
    manifest = tmp_path / "instance.json"
    created = time.time() - 3600           # rented an hour ago
    manifest.write_text(json.dumps(dict(id=51676703, created_unix=created,
                                        lease_safety_seconds=86400)))
    monkeypatch.setattr(control, "MANIFEST", manifest)
    assert control.extend(6) == 0
    saved = json.loads(manifest.read_text())
    remaining = saved["created_unix"] + saved["lease_safety_seconds"] - time.time()
    assert abs(remaining - 6 * 3600) < 5


def test_extending_is_from_now_not_from_the_old_deadline(control, tmp_path,
                                                         monkeypatch):
    """Compounding onto an existing deadline is how a box quietly outlives the
    work it was rented for."""
    manifest = tmp_path / "instance.json"
    manifest.write_text(json.dumps(dict(id=51676703, created_unix=time.time(),
                                        lease_safety_seconds=86400)))
    monkeypatch.setattr(control, "MANIFEST", manifest)
    control.extend(2)
    saved = json.loads(manifest.read_text())
    assert abs(saved["lease_safety_seconds"] - 2 * 3600) < 5


def test_the_default_lease_is_a_day_and_is_written_down(control):
    assert control.DEFAULT_LEASE_SECONDS == 86400


def test_help_needs_no_credential(control):
    """The operator runs --help before anything is configured."""
    import subprocess
    done = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "submit_box_control.py"), "--help"],
        capture_output=True, text=True, timeout=60)
    assert done.returncode == 0
    assert "survey" in done.stdout and "extend" in done.stdout

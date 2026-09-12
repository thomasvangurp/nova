"""Offline tests for the vast.ai dynamic inventory.

The failure these guard against is specific and has already happened: on
2026-09-12 the instance's SSH port moved from 21300 to 20868 on a restart, every
command aimed at the recorded port timed out, and the conclusion drawn was that
the box was gone. The inventory must therefore report whatever the endpoint is
*now*, must select on the durable label rather than on the endpoint, and must
never leak a credential field out of the instance document into host variables
that ``-vvv`` prints.

These run with no vast.ai account and no host: ``NOVA_VAST_INSTANCES_FILE``
feeds the script a saved ``vastai show instances --raw`` document.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "ansible" / "inventory" / "vast_nanobody.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("vast_nanobody", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vast_nanobody = _load_module()


def _instance(**overrides):
    """A trimmed instance document with the fields the API actually returns."""
    document = {
        "id": 50420868,
        "label": "nova-nanobody-pixi",
        "actual_status": "running",
        "ssh_host": "ssh1.vast.ai",
        "ssh_port": 21300,
        "num_gpus": 10,
        "gpu_name": "RTX 4090",
        "machine_id": 12345,
        # Credential-adjacent fields the real payload carries. They must not
        # reach host variables.
        "jupyter_token": "not-a-real-token-but-shaped-like-one",
        "ssh_key": "ssh-ed25519 AAAA...",
    }
    document.update(overrides)
    return document


def _inventory(tmp_path, monkeypatch, instances, **environment):
    path = tmp_path / "instances.json"
    path.write_text(json.dumps(instances), encoding="utf-8")
    monkeypatch.setenv("NOVA_VAST_INSTANCES_FILE", str(path))
    monkeypatch.delenv("NOVA_VAST_LABEL", raising=False)
    monkeypatch.delenv("NOVA_VAST_ANY_STATUS", raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    return vast_nanobody.build_inventory()


def test_rotated_port_is_reported_not_remembered(tmp_path, monkeypatch):
    """The same labelled instance on a new port is the same inventory host."""
    before = _inventory(tmp_path, monkeypatch, [_instance(ssh_port=21300)])
    after = _inventory(tmp_path, monkeypatch, [_instance(ssh_port=20868)])

    assert before["nanobody_producer"]["hosts"] == ["vast50420868"]
    assert after["nanobody_producer"]["hosts"] == ["vast50420868"]
    assert before["_meta"]["hostvars"]["vast50420868"]["ansible_port"] == 21300
    assert after["_meta"]["hostvars"]["vast50420868"]["ansible_port"] == 20868


def test_host_variables_carry_no_credential_fields(tmp_path, monkeypatch):
    inventory = _inventory(tmp_path, monkeypatch, [_instance()])
    variables = inventory["_meta"]["hostvars"]["vast50420868"]
    assert "jupyter_token" not in variables
    assert "ssh_key" not in variables
    serialized = json.dumps(variables)
    assert "not-a-real-token" not in serialized
    assert "ssh-ed25519" not in serialized


def test_gpu_count_comes_from_the_rental(tmp_path, monkeypatch):
    """The publisher fails closed on an unexpected GPU count, so it is derived."""
    inventory = _inventory(tmp_path, monkeypatch, [_instance(num_gpus=4)])
    assert inventory["_meta"]["hostvars"]["vast50420868"]["gpu_count"] == 4


def test_other_labels_are_not_ours(tmp_path, monkeypatch):
    inventory = _inventory(
        tmp_path,
        monkeypatch,
        [_instance(id=1, label="nova-compound"), _instance(id=2)],
    )
    assert inventory["nanobody_producer"]["hosts"] == ["vast2"]


def test_label_is_overridable(tmp_path, monkeypatch):
    inventory = _inventory(
        tmp_path,
        monkeypatch,
        [_instance(id=7, label="nova-nanobody-spot")],
        NOVA_VAST_LABEL="nova-nanobody-spot",
    )
    assert inventory["nanobody_producer"]["hosts"] == ["vast7"]


def test_non_running_instances_are_excluded_by_default(tmp_path, monkeypatch):
    inventory = _inventory(
        tmp_path, monkeypatch, [_instance(actual_status="loading")]
    )
    assert inventory["nanobody_producer"]["hosts"] == []

    included = _inventory(
        tmp_path,
        monkeypatch,
        [_instance(actual_status="loading")],
        NOVA_VAST_ANY_STATUS="1",
    )
    assert included["nanobody_producer"]["hosts"] == ["vast50420868"]


def test_direct_port_mapping_is_the_documented_fallback(tmp_path, monkeypatch):
    instance = _instance(ssh_host=None, ssh_port=None)
    instance["public_ipaddr"] = "203.0.113.7"
    instance["ports"] = {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "40060"}]}
    inventory = _inventory(tmp_path, monkeypatch, [instance])
    variables = inventory["_meta"]["hostvars"]["vast50420868"]
    assert variables["ansible_host"] == "203.0.113.7"
    assert variables["ansible_port"] == 40060


def test_endpointless_instance_is_dropped_with_a_message(tmp_path, monkeypatch, capsys):
    inventory = _inventory(
        tmp_path, monkeypatch, [_instance(ssh_host=None, ssh_port=None)]
    )
    assert inventory["nanobody_producer"]["hosts"] == []
    assert "no resolvable SSH endpoint" in capsys.readouterr().err


def test_slot_manager_exists_without_any_gpu_box(tmp_path, monkeypatch):
    """Registrations must keep being bought while no box is rented."""
    monkeypatch.delenv("NOVA_SLOT_HOST", raising=False)
    inventory = _inventory(tmp_path, monkeypatch, [])
    # The default purchase host is the always-on node, not this machine: the
    # recycle price halves every 360 blocks, so a host that sleeps buys at the
    # top of the decay curve.
    assert inventory["slot_manager"]["hosts"] == ["raspberrypi"]
    assert "ansible_connection" not in inventory["_meta"]["hostvars"]["raspberrypi"]


def test_the_purchase_host_can_be_moved_back_to_this_machine(tmp_path, monkeypatch):
    """While the always-on node is down, local is better than not buying."""
    monkeypatch.setenv("NOVA_SLOT_HOST", "operator")
    inventory = _inventory(tmp_path, monkeypatch, [])
    assert inventory["slot_manager"]["hosts"] == ["operator"]
    assert inventory["_meta"]["hostvars"]["operator"]["ansible_connection"] == "local"


def test_the_purchase_host_address_and_user_are_overridable(tmp_path, monkeypatch):
    """A different always-on node should need no code change."""
    monkeypatch.setenv("NOVA_SLOT_HOST", "buyer")
    monkeypatch.setenv("NOVA_SLOT_ADDRESS", "100.64.0.9")
    monkeypatch.setenv("NOVA_SLOT_USER", "nova")
    inventory = _inventory(tmp_path, monkeypatch, [])
    host = inventory["_meta"]["hostvars"]["buyer"]
    assert inventory["slot_manager"]["hosts"] == ["buyer"]
    assert host["ansible_host"] == "100.64.0.9"
    assert host["ansible_user"] == "nova"


def test_many_hosts_are_all_returned(tmp_path, monkeypatch):
    """Nothing here assumes a single box; N labelled instances give N hosts."""
    inventory = _inventory(
        tmp_path,
        monkeypatch,
        [_instance(id=identifier, ssh_port=20000 + identifier) for identifier in (1, 2, 3)],
    )
    assert inventory["nanobody_producer"]["hosts"] == ["vast1", "vast2", "vast3"]
    assert inventory["nanobody_publisher"]["hosts"] == ["vast1", "vast2", "vast3"]


def test_a_non_json_payload_is_an_error_not_an_empty_inventory(tmp_path, monkeypatch):
    path = tmp_path / "instances.json"
    path.write_text("<html>rate limited</html>", encoding="utf-8")
    monkeypatch.setenv("NOVA_VAST_INSTANCES_FILE", str(path))
    with pytest.raises(SystemExit, match="did not return JSON"):
        vast_nanobody.build_inventory()


def test_instances_key_wrapper_is_accepted(tmp_path, monkeypatch):
    inventory = _inventory(tmp_path, monkeypatch, [_instance()])
    assert inventory["nanobody_producer"]["hosts"] == ["vast50420868"]

    path = tmp_path / "wrapped.json"
    path.write_text(json.dumps({"instances": [_instance()]}), encoding="utf-8")
    monkeypatch.setenv("NOVA_VAST_INSTANCES_FILE", str(path))
    wrapped = vast_nanobody.build_inventory()
    assert wrapped["nanobody_producer"]["hosts"] == ["vast50420868"]

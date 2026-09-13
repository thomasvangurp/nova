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


# Every environment variable the script reads. All of them are cleared before
# each build and only the ones a test names are set, because an operator's own
# shell exports NOVA_SLOT_ADDRESS and NOVA_SLOT_USER -- so a test that read the
# ambient environment would assert about that operator's Pi rather than about the
# script, and would pass or fail depending on who ran it.
_OVERRIDES = (
    "NOVA_VAST_LABEL",
    "NOVA_VAST_ANY_STATUS",
    "NOVA_SLOT_HOST",
    "NOVA_SLOT_ADDRESS",
    "NOVA_SLOT_USER",
    "NOVA_SLOT_PYTHON",
)


def _inventory(tmp_path, monkeypatch, instances, **environment):
    path = tmp_path / "instances.json"
    path.write_text(json.dumps(instances), encoding="utf-8")
    monkeypatch.setenv("NOVA_VAST_INSTANCES_FILE", str(path))
    for key in _OVERRIDES:
        monkeypatch.delenv(key, raising=False)
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
    inventory = _inventory(tmp_path, monkeypatch, [])
    # The default purchase host is the always-on node, not this machine: the
    # recycle price halves every 360 blocks, so a host that sleeps buys at the
    # top of the decay curve.
    assert inventory["slot_manager"]["hosts"] == ["raspberrypi"]
    assert "ansible_connection" not in inventory["_meta"]["hostvars"]["raspberrypi"]


def test_the_purchase_host_can_be_moved_back_to_this_machine(tmp_path, monkeypatch):
    """While the always-on node is down, local is better than not buying."""
    inventory = _inventory(tmp_path, monkeypatch, [], NOVA_SLOT_HOST="operator")
    assert inventory["slot_manager"]["hosts"] == ["operator"]
    assert inventory["_meta"]["hostvars"]["operator"]["ansible_connection"] == "local"


def test_the_purchase_host_address_and_user_are_overridable(tmp_path, monkeypatch):
    """A different always-on node should need no code change."""
    inventory = _inventory(
        tmp_path,
        monkeypatch,
        [],
        NOVA_SLOT_HOST="buyer",
        NOVA_SLOT_ADDRESS="100.64.0.9",
        NOVA_SLOT_USER="nova",
    )
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


# --- Host-variable allowlisting -------------------------------------------
# The instance document is the whole vast.ai API payload. Naming two known
# credential fields is not enough, because the leak that matters is the field
# nobody thought of: host variables are printed by ``ansible-inventory``, echoed
# at -vvv and templated into task arguments, so the mechanism has to be an
# allowlist and the test has to assert the allowlist, not a denylist.
ALLOWED_PRODUCER_KEYS = frozenset(
    {"ansible_host", "ansible_port", "ansible_user", "gpu_count"}
    | set(vast_nanobody.SAFE_FIELDS.values())
)


def test_no_host_variable_escapes_the_allowlist(tmp_path, monkeypatch):
    instance = _instance()
    # Fields a future API version might add, of exactly the shape that must not
    # be published. None of them are named anywhere in the script.
    instance.update(
        {
            "ssh_idx": "ssh5",
            "api_key": "vast-api-key-shaped-value",
            "jupyter_url": "https://jupyter.vast.ai/?token=another-token-shape",
            "extra_env": {"WANDB_API_KEY": "secret-shaped"},
            "client_secret": "client-secret-shaped",
        }
    )
    variables = _inventory(tmp_path, monkeypatch, [instance])["_meta"]["hostvars"][
        "vast50420868"
    ]
    assert set(variables) <= ALLOWED_PRODUCER_KEYS, sorted(
        set(variables) - ALLOWED_PRODUCER_KEYS
    )
    serialized = json.dumps(variables)
    for secret in (
        "vast-api-key-shaped-value",
        "another-token-shape",
        "secret-shaped",
        "client-secret-shaped",
    ):
        assert secret not in serialized


# --- gpu_count ------------------------------------------------------------
# nanobody_services.yml asserts `gpu_count is defined` and `| int > 0` with no
# fallback, on purpose: the GPU count is a fact about the rental and a default
# here would substitute a guess. That contract only holds if this script either
# reports a usable integer or reports nothing at all.


def test_absent_num_gpus_omits_gpu_count_for_the_play_to_catch(tmp_path, monkeypatch):
    instance = _instance()
    del instance["num_gpus"]
    variables = _inventory(tmp_path, monkeypatch, [instance])["_meta"]["hostvars"][
        "vast50420868"
    ]
    assert "gpu_count" not in variables
    assert "vast_num_gpus" not in variables


def test_a_string_num_gpus_is_still_an_int(tmp_path, monkeypatch):
    """The play does `gpu_count | int > 0`; the publisher wants a real number."""
    variables = _inventory(tmp_path, monkeypatch, [_instance(num_gpus="2")])["_meta"][
        "hostvars"
    ]["vast50420868"]
    assert variables["gpu_count"] == 2
    assert isinstance(variables["gpu_count"], int)


def test_an_unparseable_num_gpus_is_reported_not_raised(tmp_path, monkeypatch, capsys):
    """A traceback out of an inventory script takes every host down at once."""
    inventory = _inventory(tmp_path, monkeypatch, [_instance(num_gpus="")])
    variables = inventory["_meta"]["hostvars"]["vast50420868"]
    # The host is still usable, and gpu_count is absent so the play's own
    # assertion reports it by name.
    assert inventory["nanobody_producer"]["hosts"] == ["vast50420868"]
    assert "gpu_count" not in variables
    assert "is not a number" in capsys.readouterr().err


# --- The purchase host ----------------------------------------------------


def test_the_purchase_host_address_defaults_to_its_name(tmp_path, monkeypatch):
    """No NOVA_SLOT_ADDRESS means resolve the tailscale name, not an empty host."""
    host = _inventory(tmp_path, monkeypatch, [], NOVA_SLOT_HOST="buyer")["_meta"][
        "hostvars"
    ]["buyer"]
    assert host["ansible_host"] == "buyer"
    assert host["ansible_user"] == "root"
    assert host["ansible_python_interpreter"] == "/usr/bin/python3"


def test_the_purchase_host_overrides_tolerate_shell_whitespace(tmp_path, monkeypatch):
    """`export NOVA_SLOT_USER='thomasg '` must not produce the user 'thomasg '."""
    inventory = _inventory(
        tmp_path,
        monkeypatch,
        [],
        NOVA_SLOT_HOST=" buyer ",
        NOVA_SLOT_ADDRESS=" 100.64.243.65\t",
        NOVA_SLOT_USER="thomasg ",
        NOVA_SLOT_PYTHON=" /usr/bin/python3 ",
    )
    # The group member and the hostvars key have to be the same stripped name, or
    # `hosts: slot_manager` resolves a host with no variables at all.
    assert inventory["slot_manager"]["hosts"] == ["buyer"]
    host = inventory["_meta"]["hostvars"]["buyer"]
    assert host["ansible_host"] == "100.64.243.65"
    assert host["ansible_user"] == "thomasg"
    assert host["ansible_python_interpreter"] == "/usr/bin/python3"


def test_the_purchase_host_is_the_only_member_of_its_group(tmp_path, monkeypatch):
    """slots.yml runs on slot_manager; a second member would buy twice."""
    inventory = _inventory(
        tmp_path, monkeypatch, [_instance()], NOVA_SLOT_HOST="buyer"
    )
    assert inventory["slot_manager"]["hosts"] == ["buyer"]
    # And the GPU box is never in it: the coldkey must not reach rented hardware.
    assert "vast50420868" not in inventory["slot_manager"]["hosts"]


# --- Zero instances -------------------------------------------------------


def test_zero_instances_names_the_label_it_looked_for(tmp_path, monkeypatch, capsys):
    """An empty producer group makes the GPU plays say "no hosts matched" and
    exit 0, so the inventory has to say what it searched for on stderr."""
    inventory = _inventory(tmp_path, monkeypatch, [])
    assert inventory["nanobody_producer"]["hosts"] == []
    assert inventory["nanobody_publisher"]["hosts"] == []
    message = capsys.readouterr().err
    assert "nova-nanobody-pixi" in message
    assert "vastai label instance" in message


def test_a_label_nothing_matches_is_not_a_silent_empty_inventory(
    tmp_path, monkeypatch, capsys
):
    inventory = _inventory(
        tmp_path, monkeypatch, [_instance()], NOVA_VAST_LABEL="nova-nanobody-spot"
    )
    assert inventory["nanobody_producer"]["hosts"] == []
    assert "nova-nanobody-spot" in capsys.readouterr().err


def test_a_skipped_non_running_instance_says_which_and_why(
    tmp_path, monkeypatch, capsys
):
    _inventory(tmp_path, monkeypatch, [_instance(actual_status="exited")])
    message = capsys.readouterr().err
    assert "'exited'" in message
    assert "NOVA_VAST_ANY_STATUS=1" in message

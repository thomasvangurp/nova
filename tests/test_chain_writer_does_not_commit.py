"""The chain writer must publish state and never commit candidates.

neurons/miner/miner.py is the upstream reference miner.  We run it for one
reason: it is the only writer of chain_state.json, which the publisher reads for
the block, the epoch and the fleet it discovers from the metagraph.

Its candidates, though, come from the example implementation -- a random 1-30
mutation of a hardcoded base sequence, which its own docstring calls "therefore
not competitive".  And resolve_chain_writer.py must hand it a REGISTERED hotkey
or it dies on metagraph.hotkeys.index(...), so the hotkey it signs with is
always one of our fleet uids.  Left submitting, it overwrites that uid's real
submission with noise and races the publisher's own set_commitment for the same
uid.  Observed live on uid 19: "Commitment set successfully".
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MINER = ROOT / "neurons/miner/miner.py"
SERVICES = ROOT / "ansible/nanobody_services.yml"
PUBLISHER_CAPTURE = ROOT / "ansible/tasks/capture_artifact_publisher.yml"


def _services_text() -> str:
    """The playbook, or a skip.

    The deploy copies the source tree to the box WITHOUT ansible/ and runs the
    suite there, so an assertion about the playbook has nothing to read on the
    host.  Reading it unguarded failed a deploy outright.
    """
    if not SERVICES.is_file():
        pytest.skip("ansible/ is not part of the deployed tree")
    return SERVICES.read_text()



def test_the_submit_path_is_gated_on_chain_state_only():
    source = MINER.read_text()
    match = re.search(
        r"async def _submit_current_candidates\(.*?\n(.*?)\n\s*candidate_small_molecules =",
        source,
        re.S,
    )
    assert match, "_submit_current_candidates changed shape; re-check the gate"
    assert 'state.get("chain_state_only")' in match.group(1), (
        "_submit_current_candidates must refuse before doing anything when "
        "chain_state_only is set"
    )


def test_the_flag_exists_and_defaults_to_submitting():
    source = MINER.read_text()
    assert "'--chain-state-only'" in source, "the flag must exist"
    # Default off: this file is upstream's, and a miner run WITHOUT another
    # process owning submission should still submit.
    assert "os.getenv('NOVA_CHAIN_STATE_ONLY') == '1'" in source
    assert "'chain_state_only': bool(getattr(config, 'chain_state_only', False))" in source, (
        "the parsed flag must reach state, or the gate reads None and submits"
    )


def test_the_deployed_chain_writer_passes_the_flag():
    """The gate is worthless if the deploy does not turn it on."""
    services = _services_text()
    assert "neurons/miner/miner.py" in services
    assert "--chain-state-only" in services, (
        "ansible/nanobody_services.yml launches the chain writer; without this "
        "flag it commits example candidates on one of our registered uids"
    )


def test_gpu_publisher_is_artifact_only_and_prepares_churn_headroom():
    services = _services_text()
    assert "\n          - --commit-command\n" not in services, (
        "the Pi is the sole commitment writer; a GPU publisher commit can "
        "overwrite Pi near the validator blackout"
    )
    assert "\n          - --minimum-sequence-pool-size\n          - \"16\"" in services
    assert "tasks/capture_artifact_publisher.yml" in services
    capture = PUBLISHER_CAPTURE.read_text()
    assert 'if "--commit-command" in argv' in capture
    assert 'if int(pool) < 16' in capture


def test_the_deploy_restarts_a_writer_whose_command_line_drifted():
    """A running session is not a CORRECT one.

    --chain-state-only was added, the deploy ran green, and the flag reached
    nothing: the start task is gated on the tmux session being absent, so the
    old process kept committing.  The deploy must compare the whole command
    line against what it would launch, and restart on any difference.
    """
    services = _services_text()
    assert "chain_writer_miner_argv" in services, (
        "the launch argv must be one var, or the drift check compares against "
        "a copy that can disagree with what is started"
    )
    assert "chain_writer_running_args" in services, (
        "the deploy must read the running writer's actual command line"
    )
    assert (
        "(chain_writer_running_args.stdout | trim)\n"
        "          != (chain_writer_miner_argv | map('string') | join(' '))"
    ) in services, "the drift check must compare running argv to the launch argv"


def test_the_launch_argv_carries_the_flag():
    import yaml

    doc = yaml.safe_load(_services_text())
    play = next(p for p in doc if isinstance(p, dict) and "vars" in p)
    argv = play["vars"]["chain_writer_miner_argv"]
    assert "--chain-state-only" in argv
    assert argv[:2] == [
        "/root/.pixi/envs/chain310/bin/python",
        "neurons/miner/miner.py",
    ], "the argv must start where `ps -eo args` starts, or drift never matches"

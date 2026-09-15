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

ROOT = Path(__file__).resolve().parents[1]
MINER = ROOT / "neurons/miner/miner.py"
SERVICES = ROOT / "ansible/nanobody_services.yml"


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
    services = SERVICES.read_text()
    assert "neurons/miner/miner.py" in services
    assert "--chain-state-only" in services, (
        "ansible/nanobody_services.yml launches the chain writer; without this "
        "flag it commits example candidates on one of our registered uids"
    )

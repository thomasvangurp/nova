"""Committing one uid's submission, and refusing to commit someone else's."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanobody_opt import commit_submission as cs


class _Timelock:
    """Records what it was asked to encrypt; the uid is part of the key."""

    def __init__(self):
        self.calls = []

    def encrypt(self, uid, message, block):
        self.calls.append((uid, message, block))
        return f"cipher(uid={uid},msg={message},blk={block})"


ROSTER = [
    {"uid": 40, "hotkey": "5EsjOURS"},
    {"uid": 46, "hotkey": "5HEHOURS"},
    {"uid": 51, "hotkey": "5RIVALxx"},
]


def test_a_hotkey_that_holds_its_uid_is_accepted():
    assert cs.holds_uid(ROSTER, "5EsjOURS", 40) is True


def test_a_recycled_uid_is_refused():
    """uids are recycled, so yesterday's uid can be another operator's today.

    Committing for it would sign someone else's slot. The uid is therefore checked
    against the chain roster rather than trusted from the command line.
    """
    assert cs.holds_uid(ROSTER, "5EsjOURS", 51) is False


def test_a_uid_absent_from_the_roster_is_refused():
    """A deregistered uid vanishes from the metagraph entirely."""
    assert cs.holds_uid(ROSTER, "5EsjOURS", 999) is False


def test_the_payload_is_keyed_on_the_uid():
    """Identical sequences must not collide across uids.

    Encryption keyed on the uid is what makes each uid's file distinct, which is
    why a single repository can hold the whole fleet's submissions: the filename
    is a content hash, and the content differs per uid.
    """
    timelock = _Timelock()
    first, _ = cs.encoded_payload("SEQ", 40, 100, timelock)
    second, _ = cs.encoded_payload("SEQ", 46, 100, timelock)
    assert first != second
    assert timelock.calls[0][0] == 40 and timelock.calls[1][0] == 46


def test_the_payload_is_deterministic_for_one_uid():
    timelock = _Timelock()
    first, first_body = cs.encoded_payload("SEQ", 40, 100, timelock)
    second, second_body = cs.encoded_payload("SEQ", 40, 100, timelock)
    assert (first, first_body) == (second, second_body)


def test_the_content_is_base64_of_the_ciphertext():
    import base64

    timelock = _Timelock()
    _, encoded = cs.encoded_payload("SEQ", 40, 100, timelock)
    assert base64.b64decode(encoded).decode() == "cipher(uid=40,msg=SEQ,blk=100)"


def test_skip_results_are_distinguishable_from_a_commit():
    """A caller iterating the fleet must tell a skip from a submission without
    parsing prose, and a stale hotkey must not look like success."""
    assert cs.RESULT_COMMITTED != cs.RESULT_STALE != cs.RESULT_TOO_SOON
    assert cs.RESULT_COMMITTED == "COMMIT_OK"
    assert "STALE" in cs.RESULT_STALE
    assert "RATE_LIMITED" in cs.RESULT_TOO_SOON


def test_help_works_without_the_chain_stack():
    """The deploy gate runs --help where bittensor may be absent, so nothing
    chain-related may be imported at module or argparse time."""
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [sys.executable, "-m", "nanobody_opt.commit_submission", "--help"],
        cwd=repo, capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0
    assert "--uid" in done.stdout
    assert "bittensor" not in done.stderr.lower()


def test_an_ss58_is_distinguished_from_a_key_file_name():
    """The publisher substitutes {hotkey} from the CHAIN, so it arrives as an
    address, while a wallet is built from a key-file name. Getting this wrong
    means a "keyfile not found" at the last step of an epoch."""
    assert cs._looks_like_address("5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ")
    assert not cs._looks_like_address("nova68-slot-1")


def test_a_truncated_address_is_not_mistaken_for_either():
    """A truncated address is what appears in a log line or a copied summary.
    It matches no neuron row, so treating it as a key-file name would look up a
    file that cannot exist."""
    assert not cs._looks_like_address("5EkCwcNTfxeN")


def test_the_commit_driver_puts_the_repository_and_utils_on_sys_path():
    """btdr and files live in utils/, not site-packages.

    neurons/miner/miner.py appends both paths before importing them; this
    program did not, so every uid in the fleet died on "No module named
    'btdr'" at the last step of an epoch while the miner beside it committed.
    """
    import sys

    root = Path(cs.__file__).resolve().parent.parent
    before = list(sys.path)
    try:
        cs._add_repository_to_path()
        assert str(root) in sys.path
        assert str(root / "utils") in sys.path
    finally:
        sys.path[:] = before


def test_self_test_needs_no_submission_arguments():
    """The deploy gate runs it with no uid, hotkey or sequence to commit."""
    parsed = cs._parser().parse_args(["--self-test"])
    assert parsed.self_test is True
    assert parsed.uid is None


def test_a_real_invocation_still_requires_the_submission_arguments(capsys):
    with pytest.raises(SystemExit):
        cs.main(["--uid", "1"])
    assert "--hotkey" in capsys.readouterr().err


def test_load_github_path_has_exactly_one_definition():
    """The value it returns is what goes on-chain.

    It lived in neurons/miner/miner.py while the commit driver imported it
    from `files`, where it did not exist -- every uid failed on ImportError
    after the fleet was published. One definition, in the module both import.
    """
    root = Path(cs.__file__).resolve().parent.parent
    marker = "def load_github_path"
    assert marker in (root / "utils" / "files.py").read_text()
    assert marker not in (root / "neurons" / "miner" / "miner.py").read_text()

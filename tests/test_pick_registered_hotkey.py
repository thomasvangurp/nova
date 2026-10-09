"""Signer selection derives from the coldkey, not from a remembered slot name."""
import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "deploy/producer/pick_registered_hotkey.py"
_spec = importlib.util.spec_from_file_location("pick_registered_hotkey", _PATH)
pick = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pick)

OURS, THEIRS = "COLD_OURS", "COLD_THEIRS"
LOCAL = {"slot-330": "hk330", "slot-335": "hk335", "slot-400": "hk400", "slot-9": "hk9"}
CHAIN = {
    # slot-330 deregistered: its address is gone; uid 10 now belongs to a rival.
    "rival10": pick.Neuron(10, THEIRS, 9.0),
    "hk335": pick.Neuron(13, OURS, 0.5),
    "hk400": pick.Neuron(6, OURS, 2.0),
    # A local keyfile whose address is registered under another coldkey.
    "hk9": pick.Neuron(9, THEIRS, 5.0),
}


def test_candidates_are_ours_and_registered_only():
    usable = pick.candidates(LOCAL, CHAIN, OURS)
    assert set(usable) == {"slot-335", "slot-400"}


def test_deregistered_prefer_and_sticky_are_ignored():
    usable = pick.candidates(LOCAL, CHAIN, OURS)
    name, _ = pick.choose(usable, prefer="slot-330", sticky="slot-330")
    assert name == "slot-400"  # highest emission among ours


def test_valid_sticky_keeps_signer_and_prefer_overrides_it():
    usable = pick.candidates(LOCAL, CHAIN, OURS)
    assert pick.choose(usable, prefer="", sticky="slot-335")[0] == "slot-335"
    assert pick.choose(usable, prefer="slot-400", sticky="slot-335")[0] == "slot-400"


def test_emission_tie_breaks_to_lowest_uid():
    usable = {"a": pick.Neuron(20, OURS, 1.0), "b": pick.Neuron(4, OURS, 1.0)}
    assert pick.choose(usable, "", "")[0] == "b"

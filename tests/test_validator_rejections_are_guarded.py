"""Every way the validator can reject us must have a guard on our side.

The validator's per-UID rejections live in neurons/validator/nanobody_validity.py.
Each one costs a registration: the UID's whole submission is dropped, scored
nowhere, and the tau we burned to hold that slot buys nothing.

This test reads that file and fails if it grows a rejection we have not
accounted for.  It is deliberately a source-parsing test rather than a list of
strings: upstream adds checks, and a list would silently go stale exactly when
it mattered.
"""
import re
from pathlib import Path

import pytest

VALIDATOR = Path(__file__).resolve().parents[1] / "neurons/validator/nanobody_validity.py"

#: Every rejecting branch, mapped to where WE stop it first.  A rejection with
#: no guard belongs in UNGUARDED, with the reason it cannot be guarded.
GUARDED: dict[str, str] = {
    "expected": "n/a: num_sequences is 1 and we commit exactly one per uid",
    "duplicate sequences": "n/a: one sequence per uid, so a within-submission duplicate is impossible",
    "valid length range": "gates.check_sequence, on the validator's own nanobody_requirements",
    "invalid amino acids": "gates.check_sequence",
    "homopolymer runs": "gates.check_sequence",
    "di-repeat pairs": "gates.check_sequence",
    "too few cysteines": "gates.check_sequence",
    "plausible cysteine pairs": "gates.check_sequence",
    "signal peptide-like": "gates.check_sequence",
    "not VHH": "filter_layer2, enforce_vhh_hallmarks from the same config",
    "low in nativeness score": "filter_layer2, min_nativeness_score from the same config",
    "low in human framework score": "filter_layer2, min_human_framework_score from the same config",
    "missing IgBLAST results": "filter_layer2 rejects a candidate with no IgBLAST result",
    "not developable": "filter_layer2 full TNP; the publisher commits only layer2_passed entries",
    "too similar to a top sequence": (
        "check_validator_novelty replicates utils.nanobodies.is_duplicate "
        "(Algorithm-2 alignment identity >= 0.95 + CDR3 clonotype >= 0.85) "
        "against the live top-50 with the coarse k-mer prefilter disabled, run "
        "before commit"
    ),
}

#: Rejections we cannot prevent, each with why.  Keeping them here rather than
#: in GUARDED means a reader can tell "handled" from "hopeless" at a glance.
UNGUARDED: dict[str, str] = {
    "failed the similarity check": (
        "the validator's own search raised; nothing about our molecule causes it "
        "and nothing we do prevents it"
    ),
}


def _rejecting_reasons() -> set[str]:
    """Warnings whose branch actually drops the submission."""
    lines = VALIDATOR.read_text().splitlines()
    found = set()
    for index, line in enumerate(lines):
        if "bt.logging.warning" not in line or "UID {uid}" not in line:
            continue
        tail = "\n".join(lines[index + 1 : index + 3])
        if "continue" not in tail and "uid_invalid = True" not in tail:
            continue                      # warns but proceeds
        message = re.search(r'f"UID \{uid\}: ([^"]*)"', line)
        if message:
            found.add(message.group(1))
    return found


def test_every_validator_rejection_is_accounted_for():
    reasons = _rejecting_reasons()
    assert reasons, "parsed no rejections; the validator source moved or changed shape"
    known = {**GUARDED, **UNGUARDED}
    unaccounted = [
        reason for reason in sorted(reasons)
        if not any(key in reason for key in known)
    ]
    assert not unaccounted, (
        "the validator gained rejection(s) with no guard and no recorded reason:\n  "
        + "\n  ".join(unaccounted)
        + "\n\nEvery one of these costs a registration. Add a guard, or add it to "
          "UNGUARDED with why it cannot be guarded."
    )


def test_the_unguarded_set_stays_small_and_explained():
    """A rejection may be unguardable, but never unexplained."""
    for reason, why in UNGUARDED.items():
        assert len(why) > 40, f"{reason!r} needs a real explanation, not {why!r}"
    assert len(UNGUARDED) <= 2, (
        "more than two unguardable rejections means we are losing registrations "
        "for reasons nobody has worked on"
    )


def test_our_gates_read_the_validators_own_thresholds():
    """Thresholds must come from the config, never be copied into our source.

    A duplicated constant is how a gate silently stops matching the validator.
    """
    gates = (VALIDATOR.parents[2] / "nanobody_opt/gates.py").read_text()
    for key in (
        "min_sequence_length", "max_sequence_length", "max_homopolymer_run",
        "max_di_repeat_pairs", "min_cysteines",
    ):
        assert f'cfg["{key}"]' in gates or f'cfg.get("{key}"' in gates, (
            f"gates.check_sequence must read {key} from the config it is given"
        )
    layer2 = (VALIDATOR.parents[2] / "nanobody_opt/filter_layer2.py").read_text()
    for key in ("min_nativeness_score", "min_human_framework_score"):
        assert f'requirements["{key}"]' in layer2, (
            f"filter_layer2 must read {key} from the validator's requirements"
        )

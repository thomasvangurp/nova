import pytest

from nanobody_opt.summarize_teacher_contacts import summarize_contacts


def _contact(residues, interface_size=3):
    return {
        "target_chain": "A",
        "cutoff_angstrom": 5.0,
        "interface_residue_count": interface_size,
        "interface_residues": [
            {
                "target_contacts": [
                    {"residue_number": residue} for residue in residues
                ]
            }
        ],
    }


def test_summarizes_frequency_and_compresses_ranges():
    got = summarize_contacts(
        [_contact([11, 14, 15, 94]), _contact([11, 14, 15]), _contact([11, 14])],
        minimum_frequency=2 / 3,
    )

    assert got["minimum_teacher_count"] == 2
    assert got["recommended_residues"] == [11, 14, 15]
    assert got["recommended_binding_site"] == "11,14..15"
    assert got["residue_teacher_counts"] == {"11": 3, "14": 3, "15": 2, "94": 1}


def test_rejects_inconsistent_chains():
    contacts = [_contact([11]), _contact([11])]
    contacts[1]["target_chain"] = "B"
    with pytest.raises(ValueError, match="chains disagree"):
        summarize_contacts(contacts, minimum_frequency=0.5)

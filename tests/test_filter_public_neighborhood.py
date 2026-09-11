import csv
import subprocess
import sys


from nanobody_opt import filter_public_neighborhood as pn


def _sequence(base, edits):
    chars = list(base)
    for index, residue in edits.items():
        chars[index] = residue
    return "".join(chars)


BASE = "A" * 127


def test_distance_is_measured_to_the_nearest_public_sequence():
    archive = [BASE, _sequence(BASE, {10: "C", 20: "D"})]
    queries = [
        BASE,                                   # identical
        _sequence(BASE, {5: "E"}),            # one substitution
        _sequence(BASE, {5: "E", 6: "F"}),    # two
        _sequence(BASE, {10: "C", 20: "D", 30: "G"}),  # one from the second
    ]

    distances = pn.minimum_distances(queries, archive)

    assert list(distances) == [0, 1, 2, 1]


def test_an_absent_archive_leaves_the_bank_alone():
    # Discarding every proposal would be worse than screening unsupported ones.
    assert list(pn.minimum_distances([BASE], [])) == [0]


def _write(path, sequences, types=None):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "proposal_type"])
        writer.writeheader()
        for index, sequence in enumerate(sequences):
            writer.writerow({
                "sequence": sequence,
                "proposal_type": (types or ["lane"] * len(sequences))[index],
            })


def _run(tmp_path, sequences, archive, **flags):
    proposals = tmp_path / "candidates.csv"
    _write(proposals, sequences)
    archive_path = tmp_path / "archive.csv"
    _write(archive_path, archive)
    out = tmp_path / "filtered.csv"
    args = [
        sys.executable, "-m", "nanobody_opt.filter_public_neighborhood",
        "--input", str(proposals), "--archive", str(archive_path),
        "--output", str(out),
    ]
    for key, value in flags.items():
        args.extend([f"--{key.replace('_', '-')}", str(value)])
    completed = subprocess.run(args, capture_output=True, text=True, cwd="/Users/tvgdeena/nova-nanobody")
    assert completed.returncode == 0, completed.stderr
    with out.open() as handle:
        return list(csv.DictReader(handle)), completed.stdout


def test_distant_proposals_are_discarded(tmp_path):
    near = _sequence(BASE, {5: "E"})
    far = _sequence(BASE, {5: "E", 6: "F", 7: "G"})

    kept, stdout = _run(tmp_path, [near, far], [BASE], minimum_keep=1)

    assert [row["sequence"] for row in kept] == [near]
    assert kept[0]["public_distance"] == "1"
    assert "discarded 1 proposals" in stdout


def test_an_aggressive_filter_still_hands_the_screen_something(tmp_path):
    # Screening the nearest available beats screening nothing at all.
    far = [_sequence(BASE, {i: "E", i + 1: "F", i + 2: "G"}) for i in (5, 20, 40)]

    kept, _ = _run(tmp_path, far, [BASE], minimum_keep=2)

    assert len(kept) == 2


def test_the_closest_proposals_are_kept_first(tmp_path):
    one = _sequence(BASE, {5: "E"})
    two = _sequence(BASE, {5: "E", 6: "F"})

    kept, _ = _run(tmp_path, [two, one], [BASE], maximum_distance=2, minimum_keep=1)

    assert [row["sequence"] for row in kept] == [one, two]

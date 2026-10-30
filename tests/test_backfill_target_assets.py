"""Backfill synthesis uses measured assets for the actual target and length."""
import hashlib
import json
from pathlib import Path

import pytest

from scoring_backfill import score_worker as worker

REFERENCES = Path(__file__).resolve().parents[1] / 'external_tools/fast_nanobody_scorer/refdesign'


@pytest.mark.parametrize('target,length', list(worker.SYNTH_REFERENCES))
def test_registered_reference_matches_bundled_bytes_and_designed_chain(target, length):
    path = Path(worker.synth_reference_for(target, length, reference_dir=REFERENCES))
    assert path.name.endswith(f'_{target}_input.cif')
    sequences = [line.split()[-1] for line in path.read_text().splitlines()
                 if line.startswith('2 polypeptide(L)')]
    assert [len(sequence) for sequence in sequences] == [length]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == worker.SYNTH_REFERENCES[target, length][1]


def test_unknown_or_uninstalled_reference_uses_ordinary_design(tmp_path):
    assert worker.synth_reference_for('Q9NZQ7', 127, reference_dir=REFERENCES) is None
    assert worker.synth_reference_for('Q9NZQ7', 114, reference_dir=tmp_path) is None


def test_explicit_wrong_target_or_length_reference_is_refused():
    wrong = worker.synth_reference_for('P20809', 127, reference_dir=REFERENCES)
    with pytest.raises(ValueError, match='qualified target/length'):
        worker.synth_reference_for('Q9NZQ7', 114, override=wrong)
    with pytest.raises(ValueError, match='unqualified'):
        worker.synth_reference_for('Q9NZQ7', 127, override=wrong)


def test_batches_preserve_every_candidate_without_mixing_lengths():
    rows = [{'sequence': 'A' * length, 'id': index}
            for index, length in enumerate([114, 120, 114, 127, 120, 114])]
    batches = list(worker.batches_by_length(rows, 2))
    assert sorted(row['id'] for _, batch in batches for row in batch) == list(range(6))
    assert all(len(batch) <= 2 and {len(r['sequence']) for r in batch} == {length}
               for length, batch in batches)


@pytest.mark.parametrize('cached_target', [None, 'P20809'])
def test_cache_without_matching_target_provenance_is_not_reused(tmp_path, monkeypatch, cached_target):
    shard, results = tmp_path / 'shard.json', tmp_path / 'results.json'
    shard.write_text(json.dumps([{'sequence': 'A' * 114, 'epoch': 26018}]))
    results.write_text(json.dumps({'hash': {'sequence': 'A' * 114, 'target': cached_target, 'p1': {}}}))
    monkeypatch.setattr(worker.sys, 'argv', ['worker', '--shard', str(shard),
        '--results', str(results), '--target', 'Q9NZQ7'])
    monkeypatch.setattr(worker.subprocess, 'run', lambda *a, **k: pytest.fail('must not launch scorer'))
    before = results.read_bytes()
    with pytest.raises(ValueError, match='target provenance'):
        worker.main()
    assert results.read_bytes() == before

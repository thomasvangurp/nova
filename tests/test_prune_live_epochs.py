"""prune_live_epochs: an epoch without archive.csv is skipped unless forced."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import prune_live_epochs as prune  # noqa: E402


def make_args(root, **overrides):
    values = dict(live_root=root, database=root / 'db.sqlite', target='T', python=sys.executable,
                  min_idle_minutes=0, execute=False, force_archive_missing=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def make_epoch(root):
    epoch = root / 'epoch1'
    (epoch / 'producer' / 'round').mkdir(parents=True)
    (epoch / 'producer' / 'round' / 'candidates.csv').write_text('x' * 100)
    return epoch


def test_missing_archive_is_skipped_by_default(tmp_path, capsys):
    epoch = make_epoch(tmp_path)
    assert prune.prune_epoch(make_args(tmp_path), epoch)['status'] == 'skipped_archive_missing_unimportable'
    assert not (epoch / 'PRUNED.json').exists()  # dry run writes nothing
    result = prune.prune_epoch(make_args(tmp_path, execute=True), epoch)
    assert result['status'] == 'skipped_archive_missing_unimportable'
    assert (epoch / 'producer' / 'round' / 'candidates.csv').exists()
    assert json.loads((epoch / 'PRUNED.json').read_text())['skipped'] == 'archive_missing_unimportable'
    assert 'WARNING' in capsys.readouterr().err


def test_missing_archive_pruned_only_when_forced(tmp_path):
    epoch = make_epoch(tmp_path)
    result = prune.prune_epoch(make_args(tmp_path, execute=True, force_archive_missing=True), epoch)
    assert result['status'] == 'pruned'
    assert json.loads((epoch / 'PRUNED.json').read_text())['archive_csv'] == 'missing_forced'

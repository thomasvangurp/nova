"""Cheap sequence-level hydrophobicity screen: pure, no fold, conservative."""
import json
import sys
import types

import pytest

from nanobody_opt import hydrophobicity
from nanobody_opt.regions import find_regions


def vhh_with_cdr3(cdr3: str) -> str:
    """Build a minimal VHH-shaped sequence whose find_regions CDR3 is ``cdr3``.

    Anchors are placed so ``find_regions`` returns the supplied CDR3 verbatim:
    C at index 22 (FR1), the FR2 ``WVRQ`` motif, the FR3 ``YYC`` cysteine, then
    the CDR3 loop followed by a ``WGQG`` FR4 start. Framework padding is
    glycine, which is not in the hydrophobic set, so the reported CDR3 fraction
    is exactly the fraction of hydrophobic residues in ``cdr3``.
    """
    return 'G' * 22 + 'C' + 'GGG' + 'WVRQ' + 'G' * 16 + 'YYC' + cdr3 + 'WGQG' + 'GGGG'


def test_builder_exposes_the_requested_cdr3_verbatim():
    cdr3 = 'DRQKNTSSDDWYA'
    seq = vhh_with_cdr3(cdr3)
    regions = find_regions(seq)
    assert seq[regions.cdr3[0]:regions.cdr3[1]] == cdr3


def test_score_uses_the_calibrated_hydrophobic_set():
    # 6/13 = 0.4615 (reject-like) and 4/13 = 0.3077 (top-miner-like).
    high = hydrophobicity.score(vhh_with_cdr3('FLIVMWDRQKNTS'))
    low = hydrophobicity.score(vhh_with_cdr3('FDLVRQKNTWSSD'))
    assert high['cdr3'] == pytest.approx(6 / 13)
    assert low['cdr3'] == pytest.approx(4 / 13)
    # The hydrophobic set is exactly F, L, I, V, M, W, Y, A.
    assert hydrophobicity.HYDROPHOBIC == frozenset('FLIVMWYA')


def test_top_miner_like_sequence_passes():
    score, passed = hydrophobicity.check(vhh_with_cdr3('FDLVRQKNTWSSD'))
    assert score == pytest.approx(4 / 13)
    assert passed is True
    assert hydrophobicity.reason(vhh_with_cdr3('FDLVRQKNTWSSD')) == ''


def test_hydrophobic_heavy_sequence_is_rejected():
    score, passed = hydrophobicity.check(vhh_with_cdr3('FLIVMWYFLIVMW'))
    assert score == pytest.approx(1.0)
    assert passed is False
    assert hydrophobicity.reason(vhh_with_cdr3('FLIVMWYFLIVMW')) == \
        hydrophobicity.REJECT_REASON


def test_short_regime_rafdy_head_passes():
    # The short-CDR3 winning scaffold RAFDY (R-A-F-D-Y) is 3/5 = 0.60 over the
    # F/L/I/V/M/W/Y/A set; the real PSH gate accepts it because the fold buries
    # the aromatics, so the coarse screen must not false-positive on it.
    score, passed = hydrophobicity.check(vhh_with_cdr3('RAFDY'))
    assert score == pytest.approx(3 / 5)
    assert passed is True


def test_non_vhh_shaped_sequence_passes_conservatively():
    # find_regions cannot locate the VHH anchors, so the screen must not
    # reject: annotation/native gates are the authority for what it cannot
    # cheaply assess.
    score, passed = hydrophobicity.check('F' * 120)
    assert score is None and passed is True
    assert hydrophobicity.reason('F' * 120) == ''


def test_threshold_is_tunable_via_keyword_argument():
    cdr3 = 'FDLVRQKNTWSSD'  # 4/13 = 0.3077
    assert hydrophobicity.check(vhh_with_cdr3(cdr3), cdr3_max=0.30)[1] is False
    assert hydrophobicity.check(vhh_with_cdr3(cdr3), cdr3_max=0.50)[1] is True


def test_sequence_phase_rejects_hydrophobic_candidate_with_distinct_reason(
        monkeypatch, tmp_path):
    """The screen is wired before the fold and emits its own reason code."""
    from nanobody_opt import gates, rules
    from nanobody_opt import pr100_live_worker as worker
    from nanobody_opt import pr100_proposal_pilot as proposals

    hydrophobic_seq = vhh_with_cdr3('FLIVMWYFLIVMW')

    monkeypatch.setitem(sys.modules, 'abnumber', types.SimpleNamespace(
        Chain=lambda sequence, **kw: types.SimpleNamespace(cdr3_seq='')))
    monkeypatch.setattr(worker.rules, 'pinned_config',
                        lambda epoch: {'nanobody_requirements': {}})
    monkeypatch.setattr(worker.pr100_filter, 'validate_annotation_runtime', lambda: None)
    monkeypatch.setattr(worker.pr100_filter, 'helper', lambda name:
                        types.SimpleNamespace(_anarci_panel=lambda species: ({}, None)))
    monkeypatch.setattr(worker.pr100_filter, '_annotate_uncached',
                        lambda sequence: {'fixture_annotation': True})
    monkeypatch.setattr(worker.pr100_filter, 'annotation_identity', lambda: 'a' * 64)
    monkeypatch.setattr(worker.rules, 'annotation_reason', lambda *a, **kw: '')
    monkeypatch.setattr(gates, 'check_sequence', lambda *a: False)
    monkeypatch.setattr(proposals, 'propose', lambda *a, **kw: {
        'sequence': hydrophobic_seq, 'reference_stratum': 'unseeded_control'})

    bank_path = tmp_path / 'bank.json'
    bank_path.write_text(json.dumps({'rows': []}))
    directory = tmp_path / 'run'
    directory.mkdir()
    (directory / 'archive.csv').write_text('sequence\nAAA\n')

    job = {'challenge': {'epoch': 25969, 'target': rules.TARGET,
                         'allowed_v': 'IGHV3-23', 'allowed_cdr3_length': 11},
           'references': str(bank_path),
           'archive_sha256': worker.file_digest(directory / 'archive.csv'),
           'archive_fetched_at': 1, 'proposals': 1}
    result = worker.sequence_phase(job, directory)
    assert result['annotation_reasons'] == {'surface_hydrophobicity_screen': 1}
    assert result['candidates'] == [] and result['annotated_total'] == 0

"""Real worker/export/consumer contract replaying two recorded epoch-25956 rows.

The annotation, native measurement, novelty verdict and scoring metrics below are
frozen observations from the successful five-entry incident. Their source artifact
hashes identify the local audit originals. We replay I/O boundaries only: native
compute, archive search and chain/network access. Actual selection, evidence gates,
export, checkpoint validation, publication and Pi allocation remain unmocked.

This is an interface regression, not a fresh scientific or novelty evaluation.
The tiny CSV exists only to exercise archive hashing/transport; its search result
is replayed, and it is never presented as a fresh full-archive scientific proof.
"""
import copy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from nanobody_opt import pr100_filter, rules
from nanobody_opt import pr100_live_controller as controller
from nanobody_opt import pr100_live_trial_export as exporter
from nanobody_opt import pr100_live_worker as worker
from tests.test_pr100_live_consumer import load_candidate


RECORDED = {'source_sha256': {'validity': '10237b03633b5b8837e32c1e187514fd3310f3a66f14ae8ee2229cd22751d6c8',
                   'scores': 'ae764274be9b2c59c04a7b26b8d99af61481c5d59e42fb845b65685814a0dad3',
                   'settings': 'b5e41e0dd15256f1bcbce316e0f9037ac471290e6b4ed9b7299427ece084dba0'},
 'challenge': {'epoch': 25956,
               'target': 'Q9NZQ7',
               'epoch_length': 361,
               'epoch_start_block': 9370116,
               'epoch_start_block_hash': '0x7a869cd0297c740f5c80a8d97fe43917daafe272dcbf34645d050b4c6cf4374d',
               'ruleset_sha256': '4fd0e5cd97993ca854205d7288364577a2f7cca9f3c48ef9d8dc1e2ca76b268c',
               'allowed_v': 'IGHV3-23',
               'allowed_cdr3_length': 18},
 'settings': {'scorer': '/root/fastscore/pinned-b982b2438b54/fast_score_batch.py',
              'scorer_sha256': 'b982b2438b549b09d4a1dab169d2aedb531702ce1dfa7be586c0fb64adc5b288',
              'python': '/root/fastscore/pinned-530f5dcbc54d/release_python.sh',
              'worker_argument_schema': 'base-v1',
              'fold_sampling_steps': 35,
              'use_daemon': False,
              'fold_diffusion_samples': 5},
 'rows': [{'sequence': 'EVQLVESGGGLVQPGGSLRLSCAASGFTFTNYNLGWFRQAPGKEREFVAAISSSDGSTYYADSVKGRFTISRDNSKNTVYLQMNSLRAEDTAVYYCATGRAGWSGYKYESYFDYWGQGTQVTVSS',
           'sequence_hash': '0883046d04a0e9aea97ceec0ac2f3b460aa9090669f0816fb2f3566dfbdc3512',
           'annotation': {'profile': 'nanobody-pr100-v1',
                          'sequence_hash': '0883046d04a0e9aea97ceec0ac2f3b460aa9090669f0816fb2f3566dfbdc3512',
                          'status': 'ok',
                          'germline': {'method': 'anarci_hmm_reference_identity',
                                       'species': 'human',
                                       'reference_sha256': '91813d37e96b23606652694c22f93051cd3e396169d23a7678fb9b96ad1146ef',
                                       'hmm_sha256': '60051f5a838704eb97fe39488f23894aa335ddadd24ad72b241f6e1cea6829ee',
                                       'alignment_species_filter': None,
                                       'alignment_model': 'alpaca_H',
                                       'V': {'identity_pct': 85.71428571428571,
                                             'identity_fraction': '6/7',
                                             'alleles': ['IGHV3-23*04'],
                                             'genes': ['IGHV3-23'],
                                             'hits': [{'allele': 'IGHV3-23*04',
                                                       'matches': 84,
                                                       'reference_length': 98,
                                                       'query_covered': 98,
                                                       'coverage_pct': 100.0}],
                                             'reference_count': 249},
                                       'J': {'identity_pct': 92.85714285714286,
                                             'identity_fraction': '13/14',
                                             'alleles': ['IGHJ4*01', 'IGHJ4*02', 'IGHJ4*03'],
                                             'genes': ['IGHJ4'],
                                             'hits': [{'allele': 'IGHJ4*01',
                                                       'matches': 13,
                                                       'reference_length': 14,
                                                       'query_covered': 14,
                                                       'coverage_pct': 100.0},
                                                      {'allele': 'IGHJ4*02',
                                                       'matches': 13,
                                                       'reference_length': 14,
                                                       'query_covered': 14,
                                                       'coverage_pct': 100.0},
                                                      {'allele': 'IGHJ4*03',
                                                       'matches': 13,
                                                       'reference_length': 14,
                                                       'query_covered': 14,
                                                       'coverage_pct': 100.0}],
                                             'reference_count': 14}},
                          'single_heavy_domain': True,
                          'scheme': 'imgt',
                          'vhh_hallmarks': True,
                          'canonical_cysteines': True,
                          'cdr3_length': 18,
                          'annotation_identity': 'e153985994c071b7ed383372199c7e4af3d2f8fada6d001cfe6227bdad99af4f'},
           'developability': {'passed': True,
                              'total_cdr_length': 34,
                              'cdr3_length': 18,
                              'cdr3_compactness': 1.4102144616428585,
                              'surface_hydrophobic_patches': 81.7595,
                              'positive_charge_patches': 0.0993,
                              'negative_charge_patches': 0.0548,
                              'aggrescan3d': {'passed': True,
                                              'sequence_hash': '0883046d04a0e9aea97ceec0ac2f3b460aa9090669f0816fb2f3566dfbdc3512',
                                              'citation_doi': '10.1093/nar/gkv359',
                                              'status': 'ok',
                                              'input_structure_sha256': '4090a2315064cc5e935493ebd0522ccfa2bca3d3111f826bd30468a9337a0a99',
                                              'normalized_structure_sha256': 'e5aae33e3c68fc5a8de69ce19ae20ff07179b3db6f7a247bff0a98f8aa83ba5f',
                                              'residue_count': 125,
                                              'positive_residue_count': 8,
                                              'radius_angstrom': 10.0,
                                              'ph_scale': 'default-scale',
                                              'surface_method': 'FreeSASA-2.0.1-NACCESS-LR100',
                                              'mode': 'static-no-minimization',
                                              'a3d_total': -95.45241587100229,
                                              'a3d_average': -0.7636193269680183,
                                              'a3d_maximum': 1.0992434190485132,
                                              'a3d_minimum': -2.9930524040922295,
                                              'a3d_positive_auc': 2.586536500200665,
                                              'checks': {'a3d_average': {'value': '-0.7636193269680183',
                                                                         'operator': '<=',
                                                                         'threshold': '-0.68002724866200',
                                                                         'passed': True},
                                                         'a3d_maximum': {'value': '1.0992434190485132',
                                                                         'operator': '<=',
                                                                         'threshold': '1.58023066762925',
                                                                         'passed': True}},
                                              'failed_metrics': []},
                              'flags': {'L': 'green',
                                        'L3': 'green',
                                        'C': 'green',
                                        'PSH': 'green',
                                        'PPC': 'green',
                                        'PNC': 'green'}},
           'novelty': {'validator_novel': True,
                       'exact_historical': False,
                       'coarse_candidates': 500,
                       'maximum_identity': 0.9},
           'metrics': {'Q9NZQ7': {'design_iiptm': 0.18513,
                                  'design_ptm': 0.73155,
                                  'design_to_target_iptm': 0.15853,
                                  'min_design_to_target_pae': 15.92401,
                                  'interaction_pae': 22.69204,
                                  'plip_hbonds_refolded': 2.0,
                                  'plip_saltbridge_refolded': 1.0,
                                  'delta_sasa_refolded': 728.43555,
                                  'liability_score': 195.0,
                                  'liability_num_violations': 21.0}}},
          {'sequence': 'EVQLVESGGGLVQPGGSLRLQCAASGFTFTNYNLGWFRQAPGKEREFVAAISSSDGSTYYADSVKGRFTISRDNSKNTVYLQMNSLRAEDTAVYYCATGRAGWSGLKYESYFDYWGQGTQVTVSS',
           'sequence_hash': '1a012d0c0577fd1be19b6039f395b6a6196950b0f9eed3ce09d13d51f77c020e',
           'annotation': {'profile': 'nanobody-pr100-v1',
                          'sequence_hash': '1a012d0c0577fd1be19b6039f395b6a6196950b0f9eed3ce09d13d51f77c020e',
                          'status': 'ok',
                          'germline': {'method': 'anarci_hmm_reference_identity',
                                       'species': 'human',
                                       'reference_sha256': '91813d37e96b23606652694c22f93051cd3e396169d23a7678fb9b96ad1146ef',
                                       'hmm_sha256': '60051f5a838704eb97fe39488f23894aa335ddadd24ad72b241f6e1cea6829ee',
                                       'alignment_species_filter': None,
                                       'alignment_model': 'alpaca_H',
                                       'V': {'identity_pct': 84.6938775510204,
                                             'identity_fraction': '83/98',
                                             'alleles': ['IGHV3-23*04'],
                                             'genes': ['IGHV3-23'],
                                             'hits': [{'allele': 'IGHV3-23*04',
                                                       'matches': 83,
                                                       'reference_length': 98,
                                                       'query_covered': 98,
                                                       'coverage_pct': 100.0}],
                                             'reference_count': 249},
                                       'J': {'identity_pct': 92.85714285714286,
                                             'identity_fraction': '13/14',
                                             'alleles': ['IGHJ4*01', 'IGHJ4*02', 'IGHJ4*03'],
                                             'genes': ['IGHJ4'],
                                             'hits': [{'allele': 'IGHJ4*01',
                                                       'matches': 13,
                                                       'reference_length': 14,
                                                       'query_covered': 14,
                                                       'coverage_pct': 100.0},
                                                      {'allele': 'IGHJ4*02',
                                                       'matches': 13,
                                                       'reference_length': 14,
                                                       'query_covered': 14,
                                                       'coverage_pct': 100.0},
                                                      {'allele': 'IGHJ4*03',
                                                       'matches': 13,
                                                       'reference_length': 14,
                                                       'query_covered': 14,
                                                       'coverage_pct': 100.0}],
                                             'reference_count': 14}},
                          'single_heavy_domain': True,
                          'scheme': 'imgt',
                          'vhh_hallmarks': True,
                          'canonical_cysteines': True,
                          'cdr3_length': 18,
                          'annotation_identity': 'e153985994c071b7ed383372199c7e4af3d2f8fada6d001cfe6227bdad99af4f'},
           'developability': {'passed': True,
                              'total_cdr_length': 34,
                              'cdr3_length': 18,
                              'cdr3_compactness': 1.4223099395199685,
                              'surface_hydrophobic_patches': 88.4324,
                              'positive_charge_patches': 0.0511,
                              'negative_charge_patches': 0.0721,
                              'aggrescan3d': {'passed': True,
                                              'sequence_hash': '1a012d0c0577fd1be19b6039f395b6a6196950b0f9eed3ce09d13d51f77c020e',
                                              'citation_doi': '10.1093/nar/gkv359',
                                              'status': 'ok',
                                              'input_structure_sha256': '03bfe7b249b0544b63fd7c37371a430463a90cfbbca835535cb5bc438f9c02a7',
                                              'normalized_structure_sha256': 'e7f88bb1f0fc8df2409c1737ab380d97f69fec28fdb8f6872925c1ca19fddc64',
                                              'residue_count': 125,
                                              'positive_residue_count': 8,
                                              'radius_angstrom': 10.0,
                                              'ph_scale': 'default-scale',
                                              'surface_method': 'FreeSASA-2.0.1-NACCESS-LR100',
                                              'mode': 'static-no-minimization',
                                              'a3d_total': -97.5851433867703,
                                              'a3d_average': -0.7806811470941624,
                                              'a3d_maximum': 1.0954497785052777,
                                              'a3d_minimum': -3.0418794190247773,
                                              'a3d_positive_auc': 2.5098821788460817,
                                              'checks': {'a3d_average': {'value': '-0.7806811470941624',
                                                                         'operator': '<=',
                                                                         'threshold': '-0.68002724866200',
                                                                         'passed': True},
                                                         'a3d_maximum': {'value': '1.0954497785052777',
                                                                         'operator': '<=',
                                                                         'threshold': '1.58023066762925',
                                                                         'passed': True}},
                                              'failed_metrics': []},
                              'flags': {'L': 'green',
                                        'L3': 'green',
                                        'C': 'green',
                                        'PSH': 'green',
                                        'PPC': 'green',
                                        'PNC': 'green'}},
           'novelty': {'validator_novel': True,
                       'exact_historical': False,
                       'coarse_candidates': 500,
                       'maximum_identity': 0.8833333333333333},
           'metrics': {'Q9NZQ7': {'design_iiptm': 0.2087,
                                  'design_ptm': 0.71569,
                                  'design_to_target_iptm': 0.17592,
                                  'min_design_to_target_pae': 13.97178,
                                  'interaction_pae': 21.66021,
                                  'plip_hbonds_refolded': 2.0,
                                  'plip_saltbridge_refolded': 3.0,
                                  'delta_sasa_refolded': 817.69434,
                                  'liability_score': 195.0,
                                  'liability_num_violations': 21.0}}}]}

@pytest.fixture
def replay(monkeypatch, tmp_path):
    data = copy.deepcopy(RECORDED)
    challenge = data['challenge']
    rows = data['rows']
    archive = b'sequence\n'  # Search boundary returns recorded observations below.
    (tmp_path / 'archive.csv').write_bytes(archive)
    archive_sha = hashlib.sha256(archive).hexdigest()
    incoming = {
        'challenge': challenge, 'target': challenge['target'], 'epoch': challenge['epoch'],
        'archive_sha256': archive_sha, 'archive_checked_at_unix': 1793046772.2367644,
        'candidates': [{'sequence': r['sequence'], 'sequence_hash': r['sequence_hash'],
                        'pr100_annotation': r['annotation'], 'source': 'recorded_contract_fixture'}
                       for r in rows],
    }
    (tmp_path / 'input.json').write_text(json.dumps(incoming))
    # Tool setup and expensive measurements are replay boundaries, not validity checks.
    monkeypatch.setattr(pr100_filter, 'helper', lambda name: SimpleNamespace(tool_paths=None))
    by_sequence = {r['sequence']: r for r in rows}
    measured = []

    def measure(sequences, workers, *, clock=None):
        measured.extend(sequences)
        return [copy.deepcopy(by_sequence[s]['developability']) for s in sequences], 0.0, 1.0

    class NoveltyReplay:
        searched = []

        def _init_worker(self, frame, sequences, all_archive=False):
            assert all_archive is True
            assert list(frame.columns) == ['sequence']

        def _search_one(self, row):
            self.searched.append(row['sequence'])
            return copy.deepcopy(by_sequence[row['sequence']]['novelty'])

    novelty = NoveltyReplay()
    # Exactly one recorded observation per row; do not invent repeated measurements.
    job = {'phase': 'native', 'challenge': challenge, 'native_tools': '/recorded-tools',
           'workers': 2, 'max_workers': 2, 'repeats': 1, 'allow_marginal': True,
           'archive_sha256': archive_sha}
    validity = worker.native_phase(job, tmp_path, measure=measure, novelty=novelty)
    assert sorted(measured) == sorted(by_sequence)
    assert sorted(novelty.searched) == sorted(by_sequence)
    assert validity['status'] == 'complete' and validity['novelty_checked'] is True
    assert validity['tier_counts'] == {'A': 0, 'B': 2}
    scores = {'target': challenge['target'], 'per_nanobody_components': {'0': {
        r['sequence']: r['metrics'] for r in rows}}}
    return SimpleNamespace(data=data, validity=validity, scores=scores, archive=archive,
                           directory=tmp_path, challenge=challenge)


def build(replay):
    return exporter.build(replay.validity, replay.scores, replay.data['settings'],
                          scorer_sha256=replay.data['settings']['scorer_sha256'],
                          settings_file_sha256=replay.data['source_sha256']['settings'],
                          now=time.time())


def verify(fleet, challenge):
    return rules.verified_fleet(fleet, challenge['epoch'],
                                challenge['epoch_start_block_hash'], challenge['epoch_length'])


def test_recorded_worker_result_survives_real_export_and_pi_allocation(replay):
    fleet = build(replay)
    accepted = verify(fleet, replay.challenge)
    assert accepted['sequence_pool'] == fleet['sequence_pool']
    assert accepted['handoff_diagnostics']['accepted_count'] == 2
    assert accepted['handoff_diagnostics']['rejected_count'] == 0
    for sequence, checkpoint in accepted['sequence_pool_checkpoints'].items():
        assert checkpoint['exact']['observation_count'] == 1
        assert checkpoint['validation']['boltzgen']['validator_matched'] is False
        assert checkpoint['validation']['layer_1']['novelty']['sequence_hash'] == rules.sequence_hash(sequence)
        assert checkpoint['validation']['layer_1']['novelty']['archive_sha256'] == fleet['archive_sha256']
    consumer = load_candidate()
    allocated = consumer.sequences_by_uid(
        fleet, replay.challenge['epoch'], crew=[(1, 'fixture-owner-1'), (2, 'fixture-owner-2')],
        epoch_start_block_hash=replay.challenge['epoch_start_block_hash'],
        epoch_length=replay.challenge['epoch_length'])
    assert set(allocated) == {1, 2}
    assert set(allocated.values()) == set(fleet['sequence_pool'])


def test_real_publish_validates_and_writes_the_actual_exporter_result(replay, monkeypatch):
    ch = replay.challenge

    class Chain:
        def get_current_block(self):
            return ch['epoch_start_block'] + 10

        def tempo(self, netuid):
            assert netuid == 68
            return ch['epoch_length'] - 1

        def determine_block_hash(self, block):
            assert block == ch['epoch_start_block']
            return ch['epoch_start_block_hash']

    monkeypatch.setattr(controller, 'fetch_archive', lambda target: replay.archive)
    destination = replay.directory / 'published.json'
    outcome = controller.publish(
        {'publication_path': str(destination)}, Chain(), ch, replay.validity, replay.scores,
        replay.data['settings'], replay.data['settings']['scorer_sha256'],
        replay.data['source_sha256']['settings'], replay.directory)
    assert outcome['status'] == 'published' and outcome['actual_pi_validation'] is True
    written = json.loads(destination.read_text())
    assert len(written['sequence_pool']) == outcome['count'] == 2
    assert verify(written, ch)['handoff_diagnostics']['rejected_count'] == 0


def test_unscored_recorded_passer_does_not_block_the_completed_partial_fleet(replay):
    omitted = replay.data['rows'][1]['sequence']
    replay.scores['per_nanobody_components']['0'].pop(omitted)
    fleet = build(replay)
    assert omitted not in fleet['sequence_pool']
    assert verify(fleet, replay.challenge)['handoff_diagnostics']['accepted_count'] == 1


@pytest.mark.parametrize('fault', ['novelty', 'native', 'annotation', 'metrics'])
def test_real_export_refuses_corrupted_recorded_gate_or_score(replay, fault):
    row = replay.validity['candidates'][0]
    if fault == 'novelty':
        row['novelty']['validator_novel'] = False
    elif fault == 'native':
        row['pr100_evidence']['developability']['aggrescan3d']['sequence_hash'] = '0' * 64
    elif fault == 'annotation':
        row['pr100_evidence']['annotation']['cdr3_length'] += 1
    else:
        replay.scores['per_nanobody_components']['0'][row['sequence']][rules.TARGET].pop('design_iiptm')
    with pytest.raises(ValueError):
        build(replay)


def test_real_consumer_rejects_rebinding_to_another_epoch_hash(replay):
    fleet = build(replay)
    changed = dict(replay.challenge, epoch_start_block_hash='0x' + '0' * 64)
    rejected = verify(fleet, changed)
    assert rejected['sequence_pool'] == []
    assert rejected['handoff_diagnostics']['rejected_count'] == 2


def test_real_consumer_rejects_a_changed_archive_after_export(replay):
    fleet = build(replay)
    fleet['archive_sha256'] = '0' * 64
    rejected = verify(fleet, replay.challenge)
    assert rejected['sequence_pool'] == []
    assert rejected['handoff_diagnostics']['reason_counts'] == {'novelty_archive_mismatch': 2}


@pytest.mark.parametrize('markers', [
    {'evidence_backend': 'cuda_refine', 'authoritative': False},
    {'evidence_backend': 'resident_cpu', 'authoritative': False},
    {'evidence_backend': 'cuda_refine', 'authoritative': True},
    {'authoritative': False},
])
@pytest.mark.parametrize('tier', ['A', 'B'])
def test_export_and_pi_reject_screen_evidence_despite_forged_tier(replay, markers, tier):
    fleet = build(replay)
    row = replay.validity['candidates'][0]
    row['tier'] = tier
    row['pr100_evidence']['developability'].update(markers)
    with pytest.raises(ValueError, match='non_authoritative_measurement'):
        build(replay)
    checkpoint = fleet['sequence_pool_checkpoints'][row['sequence']]
    checkpoint['pr100_evidence']['developability'].update(markers)
    rejected = verify(fleet, replay.challenge)
    assert row['sequence'] not in rejected['sequence_pool']
    assert rejected['handoff_diagnostics']['reason_counts'] == {'non_authoritative_measurement': 1}


@pytest.mark.parametrize('fault', ['missing_repeat', 'screen_repeat', 'failed_repeat'])
def test_export_and_pi_recheck_assisted_cpu_confirmation(replay, fault):
    row = replay.validity['candidates'][0]
    evidence = row['pr100_evidence']
    evidence['native_screen_confirmation'] = {
        'minimum': 3, 'measurements': [copy.deepcopy(evidence['developability']) for _ in range(3)]}
    fleet = build(replay)
    assert row['sequence'] in verify(fleet, replay.challenge)['sequence_pool']
    measurement = evidence['native_screen_confirmation']['measurements']
    if fault == 'missing_repeat':
        measurement.pop(); reason = 'cpu_confirmation_required'
    elif fault == 'screen_repeat':
        measurement[-1].update(evidence_backend='resident_cpu', authoritative=False)
        reason = 'non_authoritative_measurement'
    else:
        measurement[-1]['surface_hydrophobic_patches'] = 90.
        reason = 'psh_threshold'
    with pytest.raises(ValueError, match=reason):
        build(replay)
    fleet['sequence_pool_checkpoints'][row['sequence']]['pr100_evidence'] = evidence
    rejected = verify(fleet, replay.challenge)
    assert row['sequence'] not in rejected['sequence_pool']
    assert rejected['handoff_diagnostics']['reason_counts'] == {reason: 1}

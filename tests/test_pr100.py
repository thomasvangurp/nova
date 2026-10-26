"""Activation, validator parity, evidence isolation, and Pi boundary regressions."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from nanobody_opt import rules, pr100_filter, fetch_hf_archive
from nanobody_opt.prepare_pr100_bank import memberships, select
from nanobody_opt.commit_submission import validate_plaintext

ROOT = Path(__file__).resolve().parents[1]
EPOCH = 25955
HASH = '0x' + 'a' * 64
SEQ = 'ACDEFGHIKLMNPQRSTVWY' * 6


def evidence(sequence=SEQ, block_hash=HASH):
    config = rules.pinned_config(EPOCH)
    challenge = rules.challenge(config, EPOCH, block_hash)
    annotation = {'status': 'ok', 'profile': rules.PROFILE,
        'sequence_hash': rules.sequence_hash(sequence), 'single_heavy_domain': True,
        'scheme': 'imgt', 'vhh_hallmarks': True, 'canonical_cysteines': True,
        'cdr3_length': challenge['allowed_cdr3_length'],
        'germline': {'reference_sha256': '1'*64, 'hmm_sha256': '2'*64,
                     'V': {'identity_pct': 82., 'genes': ['OTHER', challenge['allowed_v']]},
                     'J': {'identity_pct': 86.}}}
    develop = {'passed': True, 'red_flags': [], 'surface_hydrophobic_patches': 89.,
        'total_cdr_length': 30, 'cdr3_length': 16, 'cdr3_compactness': 1.,
        'positive_charge_patches': .5, 'negative_charge_patches': .5,
        'aggrescan3d': {'status': 'ok', 'passed': True, 'sequence_hash': rules.sequence_hash(sequence),
            'a3d_average': str(rules.MAX_AVERAGE), 'a3d_maximum': str(rules.MAX_PEAK),
            'residue_count': len(sequence), 'input_structure_sha256': '3'*64,
            'normalized_structure_sha256': '4'*64, 'ph_scale': 'default-scale',
            'surface_method': 'FreeSASA-2.0.1-NACCESS-LR100',
            'mode': 'static-no-minimization', 'radius_angstrom': 10}}
    return {'challenge': challenge, 'filter_sha': rules.FILTER_SHA,
            'annotation': annotation, 'developability': develop}


def test_epoch_transition_preserves_legacy_and_unrelated_compound_settings():
    old = rules.pinned_config(EPOCH-1)
    old['protein_selection']['small_molecule']['target'] = 'UNCHANGED'
    assert rules.for_epoch(old, EPOCH-1) == old
    new = rules.for_epoch(old, EPOCH)
    assert new['protein_selection']['small_molecule']['target'] == 'UNCHANGED'
    assert new['protein_selection']['nanobody']['target'] == rules.TARGET
    assert rules.for_epoch(new, EPOCH-1)['protein_selection']['nanobody']['target'] == 'P20809'
    with pytest.raises(ValueError, match='not been confirmed'):
        rules.for_epoch(old, EPOCH+1)
    changed = copy.deepcopy(new)
    changed['nanobody_requirements']['min_human_v_identity'] += 1
    with pytest.raises(ValueError, match='differ'):
        rules.for_epoch(changed, EPOCH+1)


def test_ten_thousand_hashes_match_the_pinned_validator_function():
    # Execute the actual upstream function, without importing its network stack.
    tree = ast.parse((ROOT/'tests/fixtures/pr100/upstream_challenge.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'get_challenge_params_from_blockhash')
    namespace = {'random': random}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<pinned-upstream>', 'exec'), namespace)
    config = rules.pinned_config(EPOCH)
    settings = config['germline_filtering']
    flat = {'small_molecule_target': 'irrelevant', 'nanobody_target': rules.TARGET,
            'random_valid_reaction': False,
            'v_neighborhood_categories': settings['v_neighborhood_categories'],
            'cdr3_length_range': range(5,24)}
    pairs = set()
    for i in range(10000):
        block_hash = '0x'+hashlib.sha256(str(i).encode()).hexdigest()
        official = namespace[function.name](flat, block_hash)
        ours = rules.challenge(config, EPOCH, block_hash)
        for key in ('allowed_v','allowed_cdr3_length'):
            assert ours[key] == official[key]
        pairs.add((ours['allowed_v'],ours['allowed_cdr3_length']))
    assert len(pairs) == 171


def test_tied_gene_and_inclusive_thresholds_pass_but_wrong_epoch_does_not():
    e = evidence(); c = rules.pinned_config(EPOCH)
    rules.validate_evidence(e, SEQ, e['challenge'], c)
    with pytest.raises(ValueError, match='mismatched'):
        rules.validate_evidence(e, SEQ, rules.challenge(c,EPOCH+1,HASH), c)
    with pytest.raises(ValueError, match='mismatched'):
        rules.validate_evidence(e, SEQ, rules.challenge(c,EPOCH,'0x'+'b'*64), c)
    with pytest.raises(ValueError, match='sequence'):
        rules.validate_evidence(e, SEQ[:-1]+'A', e['challenge'], c)


@pytest.mark.parametrize('key,value', [('a3d_average','-0.68002724866199'),
    ('a3d_maximum','1.58023066762926'), ('a3d_average','NaN'),
    ('a3d_maximum','-1'), ('mode','dynamic'), ('radius_angstrom',9),
    ('residue_count',True), ('input_structure_sha256','invalid'), ('status','error')])
def test_missing_failed_or_incompatible_measurement_cannot_pass(key,value):
    e=evidence(); e['developability']['aggrescan3d'][key]=value
    with pytest.raises(ValueError, match='aggrescan'):
        rules.validate_evidence(e,SEQ,e['challenge'],rules.pinned_config(EPOCH))


def test_legacy_developability_cache_is_never_pr100_evidence(tmp_path):
    (tmp_path/(rules.sequence_hash(SEQ)+'.json')).write_text(json.dumps({'sequence':SEQ,
        'developability':{'passed':True,'flags':{'PSH':'green'}}}))
    assert pr100_filter.cached_result(SEQ,tmp_path) is None


def test_error_measurement_retried_instead_of_cached(tmp_path,monkeypatch):
    path=tmp_path/'pr100/developability'/rules.FILTER_SHA/(rules.sequence_hash(SEQ)+'.json')
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'sequence_hash':rules.sequence_hash(SEQ),'filter_sha':rules.FILTER_SHA,
                               'developability':{'passed':False,'aggrescan3d':{'status':'error'}}}))
    measured=evidence()['developability']; calls=[]
    async def calculate(sequences, concurrency):
        calls.append(sequences); return [measured]
    monkeypatch.setattr(pr100_filter,'develop',calculate)
    assert pr100_filter.cached_develop([{'sequence':SEQ}],tmp_path,1)==[measured]
    assert calls==[[SEQ]]


def test_bucket_membership_keeps_ties_but_does_not_reuse_other_challenges():
    e=evidence(); c=rules.pinned_config(EPOCH)
    a=e['annotation']; gene=e['challenge']['allowed_v']; length=a['cdr3_length']
    assert memberships(a,c)==[f'{gene}:{length}']
    bank={'target':rules.TARGET,'ruleset_sha256':rules.fingerprint(c),'target_scores_transferred':0,
          'buckets':{f'{gene}:{length}':[{'sequence':SEQ,'pr100_annotation':a}]}}
    assert len(select(bank,e['challenge'],8000))==1
    assert not select(bank,{**e['challenge'],'allowed_cdr3_length':length+1},8000)


class Response(io.BytesIO):
    headers={}


def test_new_target_missing_archive_requires_public_inventory_confirmation(tmp_path,monkeypatch):
    def request(url,**kwargs):
        if url==fetch_hf_archive.INVENTORY:
            return Response(json.dumps([{'path':'P20809_nanobodies.csv'}]).encode())
        assert url.endswith('/Q9NZQ7_nanobodies.csv')
        raise HTTPError(url,404,'absent',{},None)
    monkeypatch.setattr(fetch_hf_archive.urllib.request,'urlopen',request)
    result=fetch_hf_archive.fetch(tmp_path/'archive.csv',rules.TARGET)
    assert result['confirmed_absent'] and result['rows']==0


@pytest.mark.parametrize('status',[401,403,500])
def test_archive_service_failure_cannot_be_treated_as_empty(tmp_path,monkeypatch,status):
    def request(url,**kwargs):
        raise HTTPError(url,status,'unavailable',{},None)
    monkeypatch.setattr(fetch_hf_archive.urllib.request,'urlopen',request)
    with pytest.raises(ValueError,match='unavailable'):
        fetch_hf_archive.fetch(tmp_path/'archive.csv',rules.TARGET)
    assert not (tmp_path/'archive.csv').exists()


def test_molecule_half_survives_nanobody_refusal_without_committing_empty_payload():
    spec=importlib.util.spec_from_file_location('_pr100_pi',ROOT/'deploy/submit_host/pi_submit.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    molecules=','.join('mol'+str(i) for i in range(20))
    rows=module.plan([(1,'public-key'),(2,'other-public-key')],{1:molecules},{})
    assert len(rows)==1 and rows[0]['plaintext']==molecules+'|~'
    assert validate_plaintext(rows[0]['plaintext'])==rows[0]['plaintext']
    with pytest.raises(ValueError): validate_plaintext('~|~')
    with pytest.raises(ValueError): validate_plaintext('mol0|~')


def checkpoint(sequence=SEQ):
    from compound_miner.nanobody_provider import validator_boltzgen_metrics
    return {'schema_version':3,'publication_stage':'qualified','status':'qualified',
        'epoch':EPOCH,'target':rules.TARGET,'sequences':[sequence],
        'pr100_evidence':evidence(sequence),
        'validation':{'layer_1':{'passed':True},'layer_2':{'passed':True},
            'boltzgen':{'exact':True,'runtime_matched':True,
                       'metrics':{k:1. for k in validator_boltzgen_metrics()}}},
        'exact':{'observation_count':20,'pose_stability':{'structure_observation_count':20,
            'contact_jaccard_median':1.,'cdr3_rmsd_max_angstrom':0.,'epitope_occupancy_min':1.}},
        'selection':{'competitive_rate':1.,'observation_competitive_rate':1.,'score_p90':1.,
            'p90_category_rank_sums':{'confidence':1.,'physical_interaction':1.,'developability':1.}}}


def test_pi_checks_spare_pool_evidence_and_actual_chain_hash():
    cp=checkpoint()
    payload={'epoch':EPOCH,'target':rules.TARGET,'sequence_pool':[SEQ,'A'*120],
             'submissions':[], 'sequence_pool_checkpoints':{SEQ:cp}}
    assert rules.verified_fleet(payload,EPOCH,HASH)['sequence_pool']==[SEQ]
    assert rules.verified_fleet(payload,EPOCH,'0x'+'b'*64)['sequence_pool']==[]
    with pytest.raises(ValueError): rules.verified_fleet(payload,EPOCH+1,HASH)
    cp['pr100_evidence'].pop('developability')
    assert rules.verified_fleet(payload,EPOCH,HASH)['sequence_pool']==[]


def test_new_target_reference_uses_only_measured_folds_and_expires_after_startup():
    import pandas as pd
    from nanobody_opt.rank_exact import cold_start_reference
    from nanobody_opt.simulate_round import METRICS
    empty = pd.DataFrame(columns=['sequence'])
    folds = {SEQ: [{m: 1. for m in METRICS}, {m: 3. for m in METRICS}],
             SEQ[:-1]+'A': [{m: 4. for m in METRICS}]}
    field = cold_start_reference(empty,folds,target=rules.TARGET,epoch=EPOCH,minimum_rows=49)
    assert len(field)==2 and field.iloc[0][next(iter(METRICS))]==2.
    assert cold_start_reference(empty,folds,target='P20809',epoch=EPOCH,minimum_rows=49) is None
    assert cold_start_reference(empty,folds,target=rules.TARGET,epoch=EPOCH+3,minimum_rows=49) is None
    with pytest.raises(ValueError,match='two complete'):
        cold_start_reference(empty,{SEQ:[{}]},target=rules.TARGET,epoch=EPOCH,minimum_rows=49)


def test_config_loader_accepts_both_rule_generations(tmp_path):
    import yaml
    from config.config_loader import load_config
    for epoch in [EPOCH-1,EPOCH]:
        path=tmp_path/(str(epoch)+'.yaml')
        path.write_text(yaml.safe_dump(rules.pinned_config(epoch),sort_keys=False))
        loaded=load_config(str(path))
        assert loaded['nanobody_target']==[rules.pinned_config(epoch)['protein_selection']['nanobody']['target']]
        if epoch==EPOCH:
            assert loaded['min_human_v_identity']==82 and list(loaded['cdr3_length_range'])==list(range(5,24))


def test_preparation_never_guesses_a_future_block_hash(tmp_path):
    from nanobody_opt_v2.pr100_epoch import prepare
    c=rules.pinned_config(EPOCH)
    path=tmp_path/'artifacts/pr100_candidate_bank.json';path.parent.mkdir()
    path.write_text(json.dumps({'target':rules.TARGET,'ruleset_sha256':rules.fingerprint(c)}))
    args=SimpleNamespace(runtime_root=tmp_path)
    assert prepare(args,{'epoch':EPOCH-1},EPOCH,'live')==path
    assert not (tmp_path/f'live/epoch{EPOCH}/producer/epoch_candidate_pool.json').exists()


@pytest.mark.parametrize('key',['total_cdr_length','cdr3_compactness','positive_charge_patches'])
def test_pass_flag_cannot_replace_missing_developability_measurements(key):
    e=evidence();e['developability'].pop(key)
    with pytest.raises(ValueError,match='developability_unavailable'):
        rules.validate_evidence(e,SEQ,e['challenge'],rules.pinned_config(EPOCH))


def test_new_target_structure_is_packaged_exactly():
    assert hashlib.sha256((ROOT/'data/structures/8AOK.cif').read_bytes()).hexdigest()==(
        'b0633f8925379a9c73b20478b88c7f0a4c81e3f5359b553cb19b4a7ae55a62f9')
    manifest=ROOT/'build_stamp.json'
    if manifest.is_file():
        package=json.loads(manifest.read_text())
        assert package['submodules']['NOVA-nanobody-filter']==rules.FILTER_SHA
        assert 'data/structures/8AOK.cif' in package['runtime_files']
        assert 'NOVA-nanobody-filter/metanano/filters/aggrescan3d.py' in package['runtime_files']

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


def evidence(sequence=SEQ, block_hash=HASH, epoch_length=361):
    config = rules.pinned_config(EPOCH)
    challenge = rules.challenge(config, EPOCH, block_hash, epoch_length)
    annotation = {'status': 'ok', 'profile': rules.PROFILE,
        'annotation_identity': pr100_filter.annotation_identity(),
        'sequence_hash': rules.sequence_hash(sequence), 'single_heavy_domain': True,
        'scheme': 'imgt', 'vhh_hallmarks': True, 'canonical_cysteines': True,
        'cdr3_length': challenge['allowed_cdr3_length'],
        'germline': {'reference_sha256': pr100_filter.ANNOTATION_RUNTIME['panel_sha256'],
                     'hmm_sha256': pr100_filter.ANNOTATION_RUNTIME['hmm_sha256'],
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
    new = rules.pinned_config(EPOCH)
    new['protein_selection']['small_molecule']['target'] = 'UNCHANGED'
    new = rules.for_epoch(new, EPOCH)
    assert new['protein_selection']['small_molecule']['target'] == 'UNCHANGED'
    assert new['protein_selection']['nanobody']['target'] == rules.TARGET
    with pytest.raises(ValueError, match='before the announced'):
        rules.for_epoch(new, EPOCH-1)
    for epoch in [EPOCH, EPOCH+1]:
        with pytest.raises(ValueError, match='not been confirmed'):
            rules.for_epoch(old, epoch)
    changed = copy.deepcopy(new)
    changed['nanobody_requirements']['min_human_v_identity'] += 1
    with pytest.raises(ValueError, match='differ'):
        rules.for_epoch(changed, EPOCH+1)


def test_ten_thousand_hashes_match_the_pinned_validator_function(monkeypatch):
    # Execute the actual upstream function, without importing its network stack.
    oracle=ROOT/'tests/fixtures/pr100/validator'
    manifest=json.loads((oracle/'manifest.json').read_text())
    for name in ['utils/challenge.py','neurons/validator/validator.py','config/config_loader.py']:
        assert hashlib.sha256((oracle/name).read_bytes()).hexdigest()==manifest['files'][name]
    tree = ast.parse((oracle/'utils/challenge.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'get_challenge_params_from_blockhash')
    # The unrelated reaction count does not consume rng; use an explicit
    # stand-in for its database, while preserving the validator's call shape.
    import sys,types
    reaction_module=types.ModuleType('_oracle_reactions.reactions')
    reaction_module.get_total_reactions=lambda: 100
    monkeypatch.setitem(sys.modules,'_oracle_reactions.reactions',reaction_module)
    namespace = {'random': random,'__package__':'_oracle_reactions'}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<pinned-upstream>', 'exec'), namespace)
    config = rules.pinned_config(EPOCH)
    spec=importlib.util.spec_from_file_location('_oracle_config',oracle/'config/config_loader.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    flat=loader.load_config(str(oracle/'config/config.yaml'))
    caller=ast.parse((oracle/'neurons/validator/validator.py').read_text())
    calls=[n for n in ast.walk(caller) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
           and n.func.id=='get_challenge_params_from_blockhash']
    assert len(calls)==1 and {k.arg for k in calls[0].keywords}=={'config','block_hash'}
    call=compile(ast.Expression(body=calls[0]),'<pinned-validator-call>','eval')
    namespace['config']=flat
    pairs = set()
    for i in range(10000):
        block_hash = '0x'+hashlib.sha256(str(i).encode()).hexdigest()
        namespace['start_block_hash']=block_hash
        official = eval(call,namespace)
        ours = rules.challenge(config, EPOCH, block_hash, 361)
        for key in ('allowed_v','allowed_cdr3_length'):
            assert ours[key] == official[key]
        pairs.add((ours['allowed_v'],ours['allowed_cdr3_length']))
    assert len(pairs) == 171


def test_tied_gene_and_inclusive_thresholds_pass_but_wrong_epoch_does_not():
    e = evidence(); c = rules.pinned_config(EPOCH)
    rules.validate_evidence(e, SEQ, e['challenge'], c)
    with pytest.raises(ValueError, match='mismatched'):
        rules.validate_evidence(e, SEQ, rules.challenge(c,EPOCH+1,HASH,361), c)
    with pytest.raises(ValueError, match='mismatched'):
        rules.validate_evidence(e, SEQ, rules.challenge(c,EPOCH,'0x'+'b'*64,361), c)
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


def test_legacy_developability_cache_is_never_pr100_evidence(tmp_path,monkeypatch):
    path=tmp_path/'pr100/developability'/rules.FILTER_SHA/(rules.sequence_hash(SEQ)+'.json')
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'sequence_hash':rules.sequence_hash(SEQ),'filter_sha':rules.FILTER_SHA,
        'developability':{'passed':True,'flags':{'PSH':'green'}}}))
    assert pr100_filter.cached_result(SEQ,tmp_path) is None
    measured=evidence()['developability']; calls=[]
    async def calculate(sequences,concurrency):
        calls.append(sequences);return [measured]
    monkeypatch.setattr(pr100_filter,'develop',calculate)
    assert pr100_filter.cached_develop([{'sequence':SEQ}],tmp_path,1)==[measured]
    assert calls==[[SEQ]]


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
          'annotation_identity':pr100_filter.annotation_identity(),
          'buckets':{f'{gene}:{length}':[{'sequence':SEQ,'pr100_annotation':a}]}}
    assert len(select(bank,e['challenge'],8000))==1
    assert not select(bank,{**e['challenge'],'allowed_cdr3_length':length+1},8000)


@pytest.mark.parametrize('changed', ['bank_identity', 'row_identity', 'panel', 'hmm'])
def test_current_challenge_cannot_reuse_stale_annotation(changed):
    e = evidence(); config = rules.pinned_config(EPOCH)
    annotation = e['annotation']
    key = f"{e['challenge']['allowed_v']}:{annotation['cdr3_length']}"
    bank = {'target': rules.TARGET, 'ruleset_sha256': rules.fingerprint(config),
            'target_scores_transferred': 0, 'annotation_identity': pr100_filter.annotation_identity(),
            'buckets': {key: [{'sequence': SEQ, 'pr100_annotation': annotation}]}}
    if changed == 'bank_identity':
        bank['annotation_identity'] = 'old-runtime'
        with pytest.raises(ValueError, match='annotation identity'):
            select(bank, e['challenge'], 8000)
        return
    if changed == 'row_identity': annotation['annotation_identity'] = 'old-runtime'
    if changed == 'panel': annotation['germline']['reference_sha256'] = '1'*64
    if changed == 'hmm': annotation['germline']['hmm_sha256'] = '2'*64
    assert select(bank, e['challenge'], 8000) == []
    with pytest.raises(ValueError, match='annotation_'):
        rules.validate_evidence(e, SEQ, e['challenge'], config)


def test_wrong_annotation_runtime_is_refused_even_with_a_current_cache(tmp_path, monkeypatch):
    identity = pr100_filter.annotation_identity()
    path = tmp_path/'pr100/annotation'/identity/(rules.sequence_hash(SEQ)+'.json')
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(evidence()['annotation']))
    pr100_filter.validate_annotation_runtime.cache_clear()
    monkeypatch.setattr(pr100_filter.importlib.metadata, 'version', lambda name: '0.2.7')
    try:
        with pytest.raises(ValueError, match='dependency version mismatch'):
            pr100_filter.annotate(SEQ, tmp_path)
    finally:
        pr100_filter.validate_annotation_runtime.cache_clear()


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
    assert rules.verified_fleet(payload,EPOCH,HASH,361)['sequence_pool']==[SEQ]
    assert rules.verified_fleet(payload,EPOCH,'0x'+'b'*64,361)['sequence_pool']==[]
    with pytest.raises(ValueError): rules.verified_fleet(payload,EPOCH+1,HASH,361)
    cp['pr100_evidence'].pop('developability')
    assert rules.verified_fleet(payload,EPOCH,HASH,361)['sequence_pool']==[]


@pytest.mark.parametrize('count,expected', [(1, True), (0, False), (True, False)])
def test_pi_uses_provisional_checkpoint_evidence_floor(count, expected):
    cp = checkpoint()
    cp.update(publication_stage='provisional', status='provisional')
    cp['exact']['observation_count'] = count
    cp['exact'].pop('pose_stability')
    cp['selection'] = {'score': 12., 'score_p90': 12.}
    payload = {'epoch': EPOCH, 'target': rules.TARGET, 'sequence_pool': [SEQ],
               'submissions': [], 'sequence_pool_checkpoints': {SEQ: cp}}
    assert bool(rules.verified_fleet(payload, EPOCH, HASH, 361)['sequence_pool']) is expected


def test_pi_attestation_uses_rpc_hash_and_nondefault_epoch_length():
    spec=importlib.util.spec_from_file_location('_pr100_pi_attest',ROOT/'deploy/submit_host/pi_submit.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    calls=[]
    chain=SimpleNamespace(get_current_block=lambda:EPOCH*362+10,
        tempo=lambda netuid:361,
        determine_block_hash=lambda block:(calls.append(block) or HASH))
    context=module.attest_epoch(chain,EPOCH,68)
    assert context['epoch_length']==362 and calls==[EPOCH*362]
    assert context['source']=='pi_rpc'
    cp=checkpoint();cp['pr100_evidence']=evidence(epoch_length=362)
    payload={'epoch':EPOCH,'target':rules.TARGET,'sequence_pool':[SEQ],
        'submissions':[], 'sequence_pool_checkpoints':{SEQ:cp}}
    options={'epoch_start_block_hash':context['epoch_start_block_hash'],
             'epoch_length':context['epoch_length'], 'crew':[(7,'public-key')]}
    assert module.sequences_by_uid(payload,EPOCH,**options)=={7:SEQ}
    # Internally consistent fabricated evidence is still refused against RPC.
    cp['pr100_evidence']=evidence(block_hash='0x'+'b'*64,epoch_length=362)
    assert module.sequences_by_uid(payload,EPOCH,**options)=={}
    cp['pr100_evidence']=evidence(epoch_length=361)
    assert module.sequences_by_uid(payload,EPOCH,**options)=={}
    with pytest.raises(ValueError,match='epoch length'):
        module.sequences_by_uid(payload,EPOCH,epoch_start_block_hash=HASH)


def test_pi_attestation_refuses_epoch_boundary_crossing():
    spec=importlib.util.spec_from_file_location('_pr100_pi_boundary',ROOT/'deploy/submit_host/pi_submit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    blocks=iter([(EPOCH+1)*361-1,(EPOCH+1)*361])
    chain=SimpleNamespace(get_current_block=lambda:next(blocks),tempo=lambda _:360,
                          determine_block_hash=lambda _:HASH)
    with pytest.raises(ValueError,match='epoch changed'):
        module.attest_epoch(chain,EPOCH,68)


@pytest.mark.parametrize('length', [361, 362])
@pytest.mark.parametrize('offset', [0, 1, 17, 360])
def test_actual_validator_boundary_selection_and_tempo_source(length, offset):
    import asyncio
    oracle=ROOT/'tests/fixtures/pr100/validator'
    manifest=json.loads((oracle/'manifest.json').read_text())
    for name in ['neurons/validator/validator.py','neurons/validator/setup.py']:
        assert hashlib.sha256((oracle/name).read_bytes()).hexdigest()==manifest['files'][name]
    setup=ast.parse((oracle/'neurons/validator/setup.py').read_text())
    assignment=next(n for n in ast.walk(setup) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Attribute) and t.attr=='epoch_length' for t in n.targets))
    calls=[];config=SimpleNamespace(netuid=68)
    node=SimpleNamespace(query=lambda *args:(calls.append(args) or SimpleNamespace(value=length-1)))
    exec(compile(ast.Module(body=[assignment],type_ignores=[]),'<pinned-setup>','exec'),
         {'config':config,'node':node})
    assert config.epoch_length==length and calls==[('SubtensorModule','Tempo',[68])]
    tree=ast.parse((oracle/'neurons/validator/validator.py').read_text())
    wait=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and ast.unparse(n.test)=='not local_input')
    # The actual branch includes a continue for failed waits. Keep it inside a
    # loop and make the fake chain report success; no source expression changes.
    wrapper=ast.parse('async def boundary(current_block, config, chain):\n while True:\n  pass\n  return current_block\n')
    wrapper.body[0].body[0].body[0]=copy.deepcopy(wait)
    async def call(f, **kwargs):
        return f(SimpleNamespace(wait_for_block=lambda _:True))
    namespace={'local_input':False,'bt':SimpleNamespace(logging=SimpleNamespace(
        info=lambda *a:None,warning=lambda *a:None))}
    exec(compile(ast.fix_missing_locations(wrapper),'<pinned-loop>','exec'),namespace)
    read_block=EPOCH*length+offset
    boundary=asyncio.run(namespace['boundary'](read_block,config,SimpleNamespace(call=call)))
    assert boundary==(EPOCH if offset==0 else EPOCH+1)*length
    process=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='process_epoch')
    assignments=[next(n for n in ast.walk(process) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        for name in ['start_block','current_epoch']]
    scope={'config':config,'current_block':boundary}
    exec(compile(ast.Module(body=assignments,type_ignores=[]),'<pinned-epoch>','exec'),scope)
    # At an exact boundary the validator processes the just-CLOSED epoch.
    # At other offsets it waits for the end of the current submission epoch.
    closed_epoch=EPOCH-1 if offset==0 else EPOCH
    assert scope['current_epoch']==closed_epoch
    assert scope['start_block']==closed_epoch*length
    context=rules.challenge(rules.pinned_config(EPOCH),EPOCH,HASH,length)
    assert context['epoch_start_block']==EPOCH*length


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
    path.write_text(json.dumps({'target':rules.TARGET,'ruleset_sha256':rules.fingerprint(c),
                               'annotation_identity':pr100_filter.annotation_identity()}))
    args=SimpleNamespace(runtime_root=tmp_path)
    assert prepare(args,{'epoch':EPOCH-1},EPOCH,'live')==path
    assert not (tmp_path/f'live/epoch{EPOCH}/producer/epoch_candidate_pool.json').exists()


@pytest.mark.parametrize('has_cached', [False, True])
def test_empty_fresh_pool_preserves_cached_precheck_and_next_epoch(tmp_path, monkeypatch, has_cached):
    from nanobody_opt_v2 import pr100_epoch, sequential_exact_race
    from nanobody_opt.prepare_pr100_bank import bucket_key
    args = SimpleNamespace(runtime_root=tmp_path, epoch_candidate_budget=4800,
                           epoch_proposal_budget=8000, mutation_database=tmp_path/'observations.sqlite')
    annotation = evidence()['annotation']
    expected = evidence()['challenge']
    rows = [{'sequence': SEQ, 'pr100_annotation': annotation}] if has_cached else []
    bank = {'target': rules.TARGET, 'ruleset_sha256': rules.fingerprint(rules.pinned_config(EPOCH)),
            'annotation_identity': pr100_filter.annotation_identity(),
            'target_scores_transferred': 0, 'buckets': {
                bucket_key(expected['allowed_v'], expected['allowed_cdr3_length']): rows}}
    path = tmp_path/'artifacts/pr100_candidate_bank.json';path.parent.mkdir()
    path.write_text(json.dumps(bank))
    lookups = []
    def prior(database, target, *, sequences):
        lookups.append((target, list(sequences)))
        return set(sequences)
    monkeypatch.setattr(sequential_exact_race, '_prior_evaluation_sequences', prior)
    outputs = []
    for epoch in [EPOCH, EPOCH+1]:
        state = {'epoch': epoch, 'epoch_length': 361, 'nanobody_target': rules.TARGET,
                 'epoch_challenge': rules.challenge(rules.pinned_config(epoch), epoch, HASH, 361)}
        output = pr100_epoch.prepare(args, state, epoch, 'live')
        pool = json.loads(output.read_text())
        assert pool['screen_budget_policy'] == 'unseen-candidates-v2'
        assert pool['candidates'] == [] and pool['sequences'] == []
        assert pool['challenge'] == state['epoch_challenge']
        coverage = json.loads((output.parent/'pr100_coverage.json').read_text())
        assert coverage['shortfall_before_novelty'] == 4800
        assert coverage['previously_evaluated'] == len(rows)
        assert pr100_epoch.prepare(args, state, epoch, 'live') == output
        outputs.append(output)
    assert outputs[0] != outputs[1]
    assert lookups == [(rules.TARGET, [r['sequence'] for r in rows])]*2
    from nanobody_opt_v2 import run_live_producer as producer, cached_confirmation
    starts = []
    args.complete_screen_pool = outputs[0]
    args.complete_screen_until_epoch = EPOCH+1
    args.chain_state = tmp_path/'chain_state.json'
    state = {'epoch': EPOCH, 'nanobody_target': rules.TARGET}
    monkeypatch.setattr(producer, '_producer_subtree', lambda a: 'live')
    monkeypatch.setattr(producer, '_prepare_epoch', lambda **k: (
        tmp_path/'config.yaml', tmp_path/'archive.json', rules.TARGET))
    monkeypatch.setattr(cached_confirmation, 'start', lambda **k: starts.append(k))
    monkeypatch.setattr(producer, '_validator_novel_sequences', lambda **k: set())
    monkeypatch.setattr(producer, '_complete_pool_coverage', lambda **k: {'complete': True})
    monkeypatch.setattr(producer, '_stamp_artifact_policy', lambda *a: None)
    (tmp_path/'config.yaml').write_text(__import__('yaml').safe_dump(
        rules.bind(rules.pinned_config(EPOCH), EPOCH, HASH, 361)))
    assert producer._run_complete_screen_pool(args, state) == producer.PRODUCER_WAIT
    assert len(starts) == 1 and starts[0]['state']['epoch'] == EPOCH


@pytest.mark.parametrize('key',['total_cdr_length','cdr3_compactness','positive_charge_patches'])
def test_pass_flag_cannot_replace_missing_developability_measurements(key):
    e=evidence();e['developability'].pop(key)
    with pytest.raises(ValueError,match='developability_unavailable'):
        rules.validate_evidence(e,SEQ,e['challenge'],rules.pinned_config(EPOCH))


@pytest.mark.parametrize('new_count', [0, 1, 8])
def test_cached_lane_starts_independently_of_nonzero_new_pool(tmp_path, monkeypatch, new_count):
    from nanobody_opt_v2 import run_live_producer as producer, cached_confirmation, sequential_exact_race
    import yaml
    seqs=[SEQ[:-1]+'ACDEFGHI'[i] for i in range(new_count)]
    rows=[{'sequence':s,'pr100_annotation':evidence(s)['annotation']} for s in seqs]
    source={'origin_epoch':EPOCH,'target':rules.TARGET,'screen_budget_policy':'unseen-candidates-v2',
            'candidates':rows,'sequences':seqs,'exact_admission_limit':4800}
    pool=tmp_path/'input.json';pool.write_text(json.dumps(source))
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(
        rules.bind(rules.pinned_config(EPOCH),EPOCH,HASH,361)))
    args=SimpleNamespace(runtime_root=tmp_path,complete_screen_pool=pool,
                         complete_screen_until_epoch=EPOCH+1,chain_state=tmp_path/'state.json',
                         mutation_database=tmp_path/'unused.sqlite')
    state={'epoch':EPOCH,'nanobody_target':rules.TARGET}
    starts=[]
    monkeypatch.setattr(producer,'_producer_subtree',lambda _: 'live')
    monkeypatch.setattr(producer,'_prepare_epoch',lambda **_: (config,tmp_path/'archive.csv',rules.TARGET))
    monkeypatch.setattr(cached_confirmation,'start',lambda **k:starts.append(k))
    monkeypatch.setattr(producer,'_validator_novel_sequences',lambda **_:set(seqs))
    monkeypatch.setattr(sequential_exact_race,'_prior_evaluation_sequences',lambda *a,**k:set())
    monkeypatch.setattr(producer,'_stamp_artifact_policy',lambda *a:None)
    monkeypatch.setattr(producer,'_complete_pool_coverage',lambda **_: {'complete':True})
    assert producer._run_complete_screen_pool(args,state)==producer.PRODUCER_WAIT
    assert len(starts)==1 and starts[0]['state']['epoch']==EPOCH
    actual=json.loads((tmp_path/f'live/epoch{EPOCH}/producer/full_scan_pool.json').read_text())
    assert len(actual['candidates'])==new_count


def test_new_target_structure_is_packaged_exactly():
    assert hashlib.sha256((ROOT/'data/structures/8AOK.cif').read_bytes()).hexdigest()==(
        'b0633f8925379a9c73b20478b88c7f0a4c81e3f5359b553cb19b4a7ae55a62f9')
    manifest=ROOT/'build_stamp.json'
    import os
    if os.environ.get('SN68_DEPLOY_ARTIFACT_ROOT'):
        assert manifest.is_file(), 'shipping CI must execute this against its staged package'
    if manifest.is_file():
        package=json.loads(manifest.read_text())
        assert package['submodules']['NOVA-nanobody-filter']==rules.FILTER_SHA
        required={'data/structures/8AOK.cif','NOVA-nanobody-filter/metanano/filters/aggrescan3d.py'}
        assert required <= set(package['files'])
        if package.get('deployment_profile') == 'release-controls':
            # Control-only qualification tests the full source package, but
            # its promotion must not install application code or target data.
            assert required.isdisjoint(package['runtime_files'])
        else:
            assert required <= set(package['runtime_files'])

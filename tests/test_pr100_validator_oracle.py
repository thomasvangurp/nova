"""Pinned validator control flow; dependency fixtures are synthetic, not GPU proof.

Execute unchanged upstream function bodies and cheap sequence helpers. Expensive
annotation/archive/developability dependencies are injected boundary fixtures.
These tests cannot qualify real biological acceptance or throughput.
"""
import ast
import asyncio
import copy
from contextlib import nullcontext
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from nanobody_opt import rules

ORACLE = Path(__file__).parent/'fixtures/pr100/validator'
SEQ = 'ACDEFGHIKLMNPQRSTVWY'*6


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name in names]
    assert {n.name for n in nodes} == set(names)
    # Future annotations avoid importing the validator's full runtime stack.
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)


def test_oracle_bytes_match_the_recorded_upstream_pin():
    manifest = json.loads((ORACLE/'manifest.json').read_text())
    assert manifest['revision'] == rules.UPSTREAM_SHA
    assert {'utils/constants.py', 'utils/minmax_weighted_rank.py',
            'utils/stage_timer.py', 'utils/__init__.py'} <= manifest['files'].keys()
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((ORACLE/name).read_bytes()).hexdigest() == digest
    assert yaml.safe_load((ORACLE/'config/config.yaml').read_text()) == rules.pinned_config(25955)


def oracle_constants():
    source = ORACLE/'utils/constants.py'
    manifest = json.loads((ORACLE/'manifest.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['files']['utils/constants.py']
    namespace = {'__file__': str(source)}
    exec(compile(source.read_text(), str(source), 'exec'), namespace)
    return {name: namespace[name] for name in ('ALLOWED_AAS', 'HYDROPHOBIC')}


def test_rank_source_mutation_is_rejected_before_execution(tmp_path):
    from nanobody_opt.check_validator_novelty import _verified_rank_source
    source = ORACLE/'utils/minmax_weighted_rank.py'
    assert _verified_rank_source(source) == source.read_bytes()
    changed = tmp_path/'minmax_weighted_rank.py'
    changed.write_bytes(source.read_bytes() + b'\nraise RuntimeError("must not execute")\n')
    with pytest.raises(ValueError, match='upstream pin'):
        _verified_rank_source(changed)


def oracle_case(monkeypatch, change):
    monkeypatch.delenv('NOVA_SKIP_HISTORICAL_CHECKS', raising=False)
    config = yaml.safe_load((ORACLE/'config/config.yaml').read_text())
    req = config['nanobody_requirements']
    flat = {**req, 'nanobody_target': [rules.TARGET], 'num_sequences': 1,
            'num_top_sequences': None, 'min_human_v_identity': 82,
            'min_human_j_identity': 86}
    germline = {'V': {'genes': ['OTHER', 'IGHV3-23'], 'identity_pct': 82.},
                'J': {'identity_pct': 86.}}
    dev = {'red_flags': [], 'surface_hydrophobic_patches': 89.,
           'aggrescan3d': {'status': 'ok', 'passed': True}}
    class Chain(dict):
        cdr3_seq = 'A'*16
        chain_type = 'H'
        scheme = 'imgt'
        name = 'fixture'
    chain = Chain(H23='C', H104='C', H49='Q', H50='R')
    if change == 'v_low': germline['V']['identity_pct'] = 81.999
    if change == 'j_low': germline['J']['identity_pct'] = 85.999
    if change == 'wrong_v': germline['V']['genes'] = ['IGHV3-66']
    if change == 'wrong_cdr3': chain.cdr3_seq = 'A'*15
    if change == 'wrong_hallmark': chain['H50'] = 'A'
    if change == 'missing_cysteine': del chain['H104']
    if change == 'light_chain': chain.chain_type = 'L'
    if change == 'red_flag': dev['red_flags'] = ['recorded boundary fixture']
    if change == 'tnp_error': dev['error'] = True
    if change == 'psh_over': dev['surface_hydrophobic_patches'] = 89.00001
    if change == 'a3d_failed': dev['aggrescan3d']['passed'] = False
    if change == 'a3d_missing': del dev['aggrescan3d']
    async def develop(sequences):
        if change == 'develop_exception': raise RuntimeError('fixture')
        if change == 'truncated_developability': return []
        return [copy.deepcopy(dev) for _ in sequences]
    def identify(sequence):
        if change == 'annotation_exception': raise ValueError('fixture')
        return germline
    def search(*args, **kwargs):
        if change == 'similarity_exception': raise ValueError('fixture')
        return SimpleNamespace(matches=[SimpleNamespace(duplicate=change == 'similar')])
    log = SimpleNamespace(debug=lambda *a: None, warning=lambda *a: None)
    namespace = {'os': os, 'bt': SimpleNamespace(logging=log),
        **oracle_constants(),
        'seq_hash': lambda s: hashlib.sha256(s.encode()).hexdigest(),
        'index_top_sequences': lambda *a: SimpleNamespace(search=search),
        'entry_unique_for_protein_hf': lambda *a: change != 'archived',
        'is_duplicate': lambda m: (m.duplicate, ''),
        'human_germline_identity_anarci': identify,
        'extract_chains_batch': lambda sequences: {int(key.split('-')[0]): [chain]
                                                  for key in sequences},
        'extract_regions_from_chains': lambda chains: {},
        'analyze_developability': develop, 'stage': lambda *a, **k: nullcontext()}
    load_functions(ORACLE/'utils/nanobodies.py', ['normalize_seq', 'max_run_length',
        'max_di_repeat_pairs', 'looks_like_signal_peptide', 'classify_chain'], namespace)
    load_functions(ORACLE/'utils/aggrescan3d.py', ['aggrescan_rejection'], namespace)
    load_functions(ORACLE/'neurons/validator/nanobody_validity.py', ['validate_nanobodies'], namespace)
    sequences = [SEQ]
    if change == 'invalid_aa': sequences = [SEQ[:-1]+'X']
    if change == 'too_short': sequences = [SEQ[:20]]
    if change == 'empty_submission': sequences = []
    accepted = asyncio.run(namespace['validate_nanobodies'](
        {7: {'sequences': sequences}}, {}, flat, 'IGHV3-23', 16))
    return 7 in accepted


@pytest.mark.parametrize('case', ['v_low', 'j_low', 'wrong_v', 'wrong_cdr3',
    'wrong_hallmark', 'missing_cysteine', 'light_chain', 'red_flag', 'tnp_error',
    'psh_over', 'a3d_failed', 'a3d_missing', 'develop_exception',
    'annotation_exception', 'similarity_exception', 'archived', 'similar',
    'invalid_aa', 'too_short', 'empty_submission'])
def test_actual_validator_rejects_boundary_failures(monkeypatch, case):
    assert not oracle_case(monkeypatch, case)


def test_actual_validator_accepts_inclusive_boundaries_and_tied_v(monkeypatch):
    assert oracle_case(monkeypatch, 'boundary_pass')


def test_upstream_truncated_developability_gap_is_not_copied_locally(monkeypatch):
    # Document the exact pinned behavior; our complete-evidence rule deliberately
    # fails closed rather than copying an upstream zip/slice cardinality gap.
    assert oracle_case(monkeypatch, 'truncated_developability')
    assert rules.developability_reason({}, rules.pinned_config(25955)['nanobody_requirements'],
                                      sequence=SEQ)


@pytest.mark.parametrize('cpu_count', [None, 2, 8, 80])
@pytest.mark.parametrize('count', [0, 1, 512, 513, 1025, 4097])
def test_archive_batch_inputs_and_failure_merges_match_exact_validator(monkeypatch, cpu_count, count):
    """Synthetic backend failures; this is not real annotation-runtime proof."""
    from nanobody_opt.check_validator_novelty import _pr100_archive_cdrs
    calls=[]
    class Chain:
        @staticmethod
        def batch(sequences, **kwargs):
            calls.append((dict(sequences),kwargs))
            if 'RAISE' in sequences.values():raise ValueError('batch fixture failure')
            chains={key:SimpleNamespace(cdr1_seq=seq, cdr2_seq='', cdr3_seq=seq[-3:])
                    for key,seq in sequences.items() if seq!='UNPARSEABLE'}
            return chains,{key:'fixture' for key in sequences if key not in chains}
    monkeypatch.setitem(sys.modules,'abnumber',SimpleNamespace(Chain=Chain))
    monkeypatch.setattr(os,'cpu_count',lambda:cpu_count)
    class SerialExecutor:
        def __init__(self, **kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def map(self,func,batches):return map(func,batches)
    namespace={'os':os,'math':math,'_CDR_BATCH_SIZE':512,'_MAX_CDR_WORKERS':8,
        'mp':SimpleNamespace(get_context=lambda method:None),
        'ProcessPoolExecutor':SerialExecutor,'_stop_worker_log_listener':lambda:None,
        'bt':SimpleNamespace(logging=SimpleNamespace(warning=lambda *args:None))}
    load_functions(ORACLE/'utils/nanobodies.py', ['_extract_cdrs_for_records',
        '_extract_cdrs_worker','extract_cdrs_batch','_chain_to_cdrs'],namespace)
    records=[('SEQ'+str(i),str(i)) for i in range(count)]
    if count>=513:
        records[0]=('EARLY','reused-id');records[-1]=('LATE','reused-id')
        records[11]=('UNPARSEABLE','11');records[300]=('RAISE','300')
    expected=namespace['_extract_cdrs_for_records'](records);expected_calls=calls[:];calls.clear()
    assert _pr100_archive_cdrs(records)==expected
    assert calls==expected_calls


def test_exact_validator_molecule_path_runs_without_any_nanobody_half():
    """Execute upstream routing with synthetic chemistry and inference boundaries."""
    log=SimpleNamespace(info=lambda *a:None,warning=lambda *a:None,error=lambda *a:None)
    config={'filters':{},'small_molecule_target':['fixture'],'random_valid_reaction':True,
            'num_molecules':20,'max_similarity_to_historical':0.9,'min_entropy':0}
    namespace={'bt':SimpleNamespace(logging=log),'os':SimpleNamespace(environ={}),
        'validate_thresholds':lambda value:value,'get_historical_submissions':lambda *a:None,
        'rdFingerprintGenerator':SimpleNamespace(GetMorganGenerator=lambda **kw:
            SimpleNamespace(GetFingerprint=lambda mol:mol)),
        'Chem':SimpleNamespace(MolFromSmiles=lambda value:value),
        'is_reaction_allowed':lambda *a:True,'get_smiles':lambda value:value,
        'check_filters':lambda *a:[],'is_boltz_safe_smiles':lambda *a:(True,''),
        'entry_unique_for_protein_hf':lambda *a:True,'find_chemically_identical':lambda *a:{},
        'compute_fingerprint_entropy':lambda *a:1.0}
    load_functions(ORACLE/'neurons/validator/molecule_validity.py',
                   ['validate_molecules_and_calculate_entropy'],namespace)
    molecules=[f'rxn:1:{i}:2' for i in range(20)];scores={7:{}}
    # No sequences key at all: an accidental dependency fails immediately.
    valid=namespace['validate_molecules_and_calculate_entropy'](
        {7:{'molecules':molecules}},scores,config,allowed_reaction=1)
    assert valid[7]['names']==molecules
    calls=[]
    def workers(ctx,specs,**kwargs):
        calls.extend(specs)
        return [{'shard_id':0,'gpu':0,'per_molecule_components':{},'final_boltz_scores':{7:1}}]
    def assemble(*args):scores[7]['molecule_path_executed']=True
    namespace.update(torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:1)),
        mp=SimpleNamespace(get_context=lambda *a:None),_inference_deadline_s=lambda c:60,
        _max_retries=lambda:0,_run_workers=workers,_merge_boltz_shards=lambda *a:({}, {}, {7:1}),
        assemble_molecule_scores=assemble,_log_inference_summary=lambda *a:None,
        BoltzResult=lambda *a:SimpleNamespace(),BoltzgenResult=lambda *a:SimpleNamespace(),
        InferenceResult=SimpleNamespace)
    load_functions(ORACLE/'utils/inference.py',['main'],namespace)
    result=namespace['main'](valid,{},scores,SimpleNamespace(to_dict=lambda:config))
    assert len(calls)==1 and calls[0][2]=='boltz'
    assert result.boltz is not None and result.boltzgen is None
    assert scores[7]['molecule_path_executed']

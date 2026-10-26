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
import os
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
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((ORACLE/name).read_bytes()).hexdigest() == digest
    assert yaml.safe_load((ORACLE/'config/config.yaml').read_text()) == rules.pinned_config(25955)


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
        'ALLOWED_AAS': set('ACDEFGHIKLMNPQRSTVWY'), 'HYDROPHOBIC': set('AILMFWV'),
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

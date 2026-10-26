"""Metanano-only parent gate; no historical exact-match ban for seed parents.

Use the child novelty gate's engine builder and the validator's own function
bodies without importing utils/__init__.py (which requires rdkit).
"""
from __future__ import annotations

import os

# Silence dependency, HF, and native-library diagnostics before importing them.
# The caller receives only an exit status and an allowlisted JSON result.
_null = os.open(os.devnull, os.O_WRONLY)
os.dup2(_null, 1)
os.dup2(_null, 2)
os.close(_null)

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, '/root')


def _source_function(path: Path, name: str, namespace: dict):
    """Execute the unmodified live function AST, avoiding unrelated imports."""
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fetch-archive', action='store_true')
    args = parser.parse_args()
    request = json.loads(args.input.read_text())
    target, sequences = request['target'], request['sequences']
    if not isinstance(target, str) or not target or not isinstance(sequences, list):
        raise ValueError('invalid request')
    if any(not isinstance(seq, str) or not seq or seq != seq.strip().upper()
           or set(seq) - set('ACDEFGHIKLMNPQRSTVWY') for seq in sequences):
        raise ValueError('invalid sequence')

    silent_log = SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None)
    namespace = dict(os=os, pd=pd, bt=SimpleNamespace(logging=silent_log))
    from nanobody_opt import rules
    all_archive = target == rules.TARGET
    if args.fetch_archive:
        if all_archive:
            from nanobody_opt.fetch_hf_archive import fetch
            fetch(args.output, target)
            return
        # huggingface_hub is available in bgen; metanano carries the search stack.
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        namespace.update(hf_hub_download=hf_hub_download,
                         EntryNotFoundError=EntryNotFoundError)
        historical = _source_function(Path('/root/utils/challenge.py'),
                                      'get_historical_submissions', namespace)
        archive = historical(target, 'nanobodies')
        if archive is None or archive.empty:
            raise ValueError('archive unavailable')
        # Use pandas' round-trip float reader below to preserve ranking inputs.
        # CSV also avoids pandas-version-dependent pickle internals across envs.
        archive.to_csv(args.output, index=False)
        pd.testing.assert_frame_equal(
            archive, pd.read_csv(args.output, float_precision='round_trip'),
            check_exact=True,
        )
        return

    from metanano.utils.cdr_utils import extract_cdrs
    from nanobody_opt.check_validator_novelty import _rank_binders, _build_search_engine

    is_duplicate = _source_function(Path('/root/utils/nanobodies.py'),
                                    'is_duplicate', namespace)
    with tempfile.TemporaryDirectory(prefix='validator-hf-archive-') as directory:
        archive_path = Path(directory) / 'archive.csv'
        subprocess.run(
            ['/root/.pixi/envs/bgen/bin/python', str(Path(__file__).resolve()),
             '--fetch-archive', '--input', str(args.input), '--output', str(archive_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=180, check=True,
        )
        archive = pd.read_csv(archive_path, float_precision='round_trip')
    if archive is None or (archive.empty and not all_archive):
        raise ValueError('archive unavailable')
    top = _rank_binders(archive, 0 if all_archive else 50, 50)[['sequence', 'sequence_hash']]
    if not all_archive and len(top) != 50:
        raise ValueError('incomplete top 50')
    # The shared builder normalizes strings and skips missing annotations. Require
    # canonical archive rows and successful AbNumber annotation so neither behavior
    # can silently diverge from index_top_sequences (including its IgBLAST fallback).
    if any(not isinstance(seq, str) or seq != seq.strip().upper()
           or (not all_archive and extract_cdrs(seq) is None) for seq in top.sequence):
        raise ValueError('top 50 cannot be indexed exactly')
    if any(not isinstance(seq_id, str) or not seq_id for seq_id in top.sequence_hash):
        raise ValueError('invalid top 50 IDs')
    engine = (_build_search_engine(top, all_archive=True) if all_archive else _build_search_engine(top))
    results = {}
    for sequence in sequences:
        search = engine.search(
            sequence,
            include_alignment=True,
            exclude_ids=None,
            coarse_min_shared=None,
            coarse_jaccard=None,
        )
        if search.total_indexed != len(set(top.sequence_hash)):
            raise ValueError('incomplete index')
        cdrs = extract_cdrs(sequence)
        flags = [is_duplicate(m)[0] for m in search.matches]
        identities = [
            float(m.identity) for m in search.matches
            if getattr(m, 'identity', None) is not None
        ]
        cdr3_similarities = [
            float(m.cdr_similarity.get('CDR3', 0.0)) for m in search.matches
            if getattr(m, 'cdr_similarity', None) is not None
        ]
        max_identity = max(identities) if identities else 0.0
        max_cdr3 = max(cdr3_similarities) if cdr3_similarities else 0.0
        # The duplicate decision is the exact is_duplicate predicate above.
        # identity/cdr3_similarity are diagnostics ONLY: no scalar parent margin
        # is asserted, because child-level safety cannot be certified by a flat
        # threshold on a parent whose local-alignment denominator (and backend:
        # parasail vs the BioPython fallback) varies per child. Child safety is
        # established by the unchanged commit-time gate (check_validator_novelty).
        results[sequence] = {
            'duplicate': any(flags),
            'cdr3': (cdrs.get('CDR3') or cdrs.get('cdr3') or '') if cdrs is not None else '',
            'identity': max_identity,
            'cdr3_similarity': max_cdr3,
        }
    args.output.write_text(json.dumps({
        'target': target, 'algorithm2_top_count': len(top), 'results': results,
    }))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        os._exit(1)

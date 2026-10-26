#!/usr/bin/env python3
"""Prepare only the pinned Pi loop's process-tree/cutoff correction."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PI_LOOP_SHA256 = '4294746ad30644ecd872d4f2a6c3fec86e8415864ee7ca6bb853abb7e90486d0'
HELPER_SHA256 = '2412cdee1291e68a7b6709655472c15c710bd0d2626f3867d40fe008ecd6155f'
HANDLE_SHA256 = 'f66058039f928542f8b3f1e27de207eb6dbbccf6a6d527b8dc6454cfd35486e4'
CALL_SHA256 = (
    '4c88c6eca2eeea427631ddf3472dc4b30734c18209a1d740021f7960b60ab790',
    '3657143ac60779b3144e8ecd2cae05b825b1ff5435a4246b4fe5b062900273f7',
    '4b6cb2a4e508bc36972fd51486022668e6ea08891285582ea779b04aa302fa52',
)
PRIMARY_CUTOFF = 'epoch_end_at - max(commit_reserve(len(view["crew"])), BLACKOUT_BLOCKS + 1) * submission_timing.BLOCK_SECONDS'


def helper_source():
    source = (ROOT/'tools/pi_epoch_loop.py').read_text()
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'run_preparation']
    if len(nodes) != 1:
        raise ValueError('preparation helper missing or ambiguous')
    result = ast.get_source_segment(source, nodes[0]).encode()
    if hashlib.sha256(result).hexdigest() != HELPER_SHA256:
        raise ValueError('reviewed preparation helper changed')
    return result


def span(source, node):
    lines = source.splitlines(keepends=True)
    return (sum(map(len, lines[:node.lineno-1])) + node.col_offset,
            sum(map(len, lines[:node.end_lineno-1])) + node.end_col_offset)


def patch_pi_preparation(source, *, expected_sha256=PI_LOOP_SHA256):
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError('Pi loop source changed')
    text = source.decode(); tree = ast.parse(text)
    if any(isinstance(n, ast.FunctionDef) and n.name == 'run_preparation' for n in tree.body):
        raise ValueError('already patched')
    handles = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'handle_epoch']
    if len(handles) != 1 or hashlib.sha256(ast.get_source_segment(text, handles[0]).encode()).hexdigest() != HANDLE_SHA256:
        raise ValueError('Pi epoch handler changed')
    handle = handles[0]
    calls = sorted([n for n in ast.walk(handle) if isinstance(n, ast.Call)
        and ast.unparse(n.func) == 'subprocess.run'
        and 'tools/make_fleet.py' in ast.get_source_segment(text, n)], key=lambda n:n.lineno)
    segments = [ast.get_source_segment(text, n) for n in calls]
    if tuple(hashlib.sha256(s.encode()).hexdigest() for s in segments) != CALL_SHA256:
        raise ValueError('Pi preparation call sites changed')
    edits = []
    for i, (node, old) in enumerate(zip(calls, segments)):
        cutoff = PRIMARY_CUTOFF if i == 0 else 'collect_at'
        replacement = old.replace('subprocess.run(', 'run_preparation(', 1)[:-1] + ', cutoff_at=' + cutoff + ')'
        start, end = span(source, node); edits.append((start, end, replacement.encode()))
    start, _ = span(source, handle); edits.append((start, start, helper_source()+b'\n\n\n'))
    result = source
    for start, end, replacement in sorted(edits, reverse=True):
        result = result[:start]+replacement+result[end:]
    ast.parse(result.decode())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:
        result=patch_pi_preparation(a.source.read_bytes())
        fd=os.open(a.output,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,'wb') as f:f.write(result)
        print(json.dumps({'status':'prepared_not_installed','old_sha256':PI_LOOP_SHA256,'candidate_sha256':hashlib.sha256(result).hexdigest(),'helper_sha256':hashlib.sha256(helper_source()).hexdigest(),'call_sites':3}))
        return 0
    except Exception as e:
        print(json.dumps({'status':'refused','error_class':type(e).__name__}));return 1


if __name__ == '__main__':raise SystemExit(main())

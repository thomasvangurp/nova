"""Measured baseline phases for the Pi-owned PR100 epoch controller."""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
from nanobody_opt import rules, pr100_filter


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sequence_phase(job, directory):
    from abnumber import Chain
    import pandas as pd
    from nanobody_opt import gates, pr100_proposal_pilot as proposals
    from nanobody_opt import check_validator_novelty as novelty
    expected = job['challenge']
    config = rules.pinned_config(expected['epoch'])
    pr100_filter.validate_annotation_runtime()
    archive = directory / 'archive.csv'
    if file_digest(archive) != job['archive_sha256']:
        raise ValueError('archive snapshot changed')
    bank = json.loads(Path(job['references']).read_text())['rows']
    cell = f"{expected['allowed_v']}:{expected['allowed_cdr3_length']}"
    refs = sorted((r for r in bank if cell in r['cells']),
                  key=lambda r: rules.digest([expected['epoch'], r['sequence_hash']]))[:16]
    panels, _ = pr100_filter.helper('germline_identity')._anarci_panel('human')
    rows = [{**r, 'source': 'existing_native_reference'} for r in refs]
    for ordinal in range(64):
        ref = refs[ordinal % len(refs)] if refs and ordinal < 32 else None
        cdr3 = (Chain(ref['sequence'], scheme='imgt', cdr_definition='imgt',
                      use_anarcii=False).cdr3_seq if ref else None)
        row = proposals.propose(panels, expected['allowed_v'], expected['allowed_cdr3_length'],
                                seed=expected['epoch'], ordinal=ordinal, reference_cdr3=cdr3)
        rows.append({**row, 'source': 'fresh_baseline_proposal',
                     'reference_hash': ref['sequence_hash'] if ref else None})
    unique = {r['sequence']: r for r in rows}
    eligible, reasons = [], {}
    for sequence, row in unique.items():
        annotation = {**pr100_filter._annotate_uncached(sequence),
                      'annotation_identity': pr100_filter.annotation_identity()}
        reason = ('basic_sequence' if gates.check_sequence(sequence, config['nanobody_requirements'])
                  else rules.annotation_reason(annotation, config['nanobody_requirements'],
                                               sequence=sequence, epoch_challenge=expected))
        reasons[reason or 'passed'] = reasons.get(reason or 'passed', 0) + 1
        if not reason:
            eligible.append({**row, 'sequence_hash': rules.sequence_hash(sequence),
                             'pr100_annotation': annotation})
    frame = pd.read_csv(archive)
    novelty._init_worker(frame, set(frame.sequence.dropna().astype(str).str.strip().str.upper()), all_archive=True)
    result = {'challenge': expected, 'target': expected['target'], 'epoch': expected['epoch'],
              'archive_sha256': job['archive_sha256'], 'archive_checked_at_unix': job['archive_fetched_at'],
              'archive_rows': len(frame), 'existing_references': len(refs), 'proposals': 64,
              'annotation_reasons': reasons, 'search_total': len(eligible), 'search_completed': 0,
              'candidates': [], 'status': 'running'}
    for row in eligible:
        found = novelty._search_one(row)
        result['search_completed'] += 1
        if found['validator_novel']:
            row['novelty'] = {k: found[k] for k in ('validator_novel', 'exact_historical',
                                                  'coarse_candidates', 'maximum_identity')}
            result['candidates'].append(row)
        pr100_filter.atomic(directory / 'result.json', result)
    result['status'] = 'complete'
    pr100_filter.atomic(directory / 'result.json', result)
    return result


def native_phase(job, directory):
    incoming = json.loads((directory / 'input.json').read_text())
    if incoming['challenge'] != job['challenge'] or len(incoming['candidates']) > 80:
        raise ValueError('native input challenge or size mismatch')
    tools = Path(job['native_tools'])
    pr100_filter.helper('aggrescan3d').tool_paths = lambda: (tools / 'aggrescan3d-rs', tools / 'freesasa')
    result = {k: incoming[k] for k in ('challenge', 'target', 'epoch', 'archive_sha256', 'archive_checked_at_unix')}
    result.update(status='running', input_count=len(incoming['candidates']), completed=0,
                  candidates=[], rejection_counts={}, cache_hits=0)
    config = rules.pinned_config(job['challenge']['epoch'])
    # Queued profiles must not spend their per-profile timeout awaiting a slot.
    for offset in range(0, len(incoming['candidates']), 4):
        batch = incoming['candidates'][offset:offset + 4]
        values = asyncio.run(pr100_filter.develop([r['sequence'] for r in batch], 4))
        for row, value in zip(batch, values, strict=True):
            reason = rules.developability_reason(value, config['nanobody_requirements'], sequence=row['sequence'])
            result['completed'] += 1
            result['rejection_counts'][reason or 'passed'] = result['rejection_counts'].get(reason or 'passed', 0) + 1
            if not reason:
                evidence = {'challenge': job['challenge'], 'filter_sha': rules.FILTER_SHA,
                            'annotation': row['pr100_annotation'], 'developability': value}
                rules.validate_evidence(evidence, row['sequence'], job['challenge'], config)
                result['candidates'].append({**row, 'layer2_passed': True, 'pr100_evidence': evidence})
        pr100_filter.atomic(directory / 'result.json', result)
    result['status'] = 'complete'
    pr100_filter.atomic(directory / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    directory = args.job.parent
    expected = job['challenge']
    config = rules.pinned_config(expected['epoch'])
    if expected != rules.challenge(config, expected['epoch'], expected['epoch_start_block_hash'], expected['epoch_length']):
        raise ValueError('wrong current challenge')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:job['cpu_slots']])
    started = time.monotonic()
    try:
        result = {'sequence': sequence_phase, 'native': native_phase}[job['phase']](job, directory)
        receipt = {'status': result['status'], 'candidate_count': len(result['candidates'])}
    except Exception as error:
        receipt = {'status': 'failed', 'error_type': type(error).__name__}
    receipt.update(epoch=expected['epoch'], phase=job['phase'], wall_seconds=time.monotonic() - started)
    pr100_filter.atomic(directory / 'worker-receipt.json', receipt)
    return 0 if receipt['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Cold-start PR100 pools, bound to the actual epoch-start challenge."""
from __future__ import annotations
import json
from pathlib import Path

from nanobody_opt import rules
from nanobody_opt.prepare_pr100_bank import select
from nanobody_opt.pr100_filter import atomic


def prepare(args, state, epoch, subtree):
    bank_path = args.runtime_root / 'artifacts/pr100_candidate_bank.json'
    bank = json.loads(bank_path.read_text())
    config = rules.pinned_config(epoch)
    if bank.get('ruleset_sha256') != rules.fingerprint(config):
        raise ValueError('PR100 bank has not been prepared for the qualified rules')
    if epoch > int(state['epoch']):
        # The future block hash does not exist. Only the all-combinations bank
        # can be prepared now; no speculative challenge becomes scoring input.
        return bank_path
    expected = rules.challenge(config, epoch,
        (state.get('epoch_challenge') or {}).get('epoch_start_block_hash'), state['epoch_length'])
    if state.get('epoch_challenge') != expected or state['nanobody_target'] != expected['target']:
        raise ValueError('PR100 chain challenge is incomplete or inconsistent')
    root = args.runtime_root / subtree / f'epoch{epoch}/producer'
    output = root / 'epoch_candidate_pool.json'
    if output.is_file():
        previous = json.loads(output.read_text())
        if (previous.get('challenge') != expected or
                previous.get('candidate_budget') != args.epoch_candidate_budget or
                previous.get('proposal_budget') != args.epoch_proposal_budget):
            raise ValueError('prepared PR100 pool does not match the current challenge')
        return output
    rows = select(bank, expected, None)
    from nanobody_opt_v2.sequential_exact_race import _prior_evaluation_sequences
    evaluated = _prior_evaluation_sequences(args.mutation_database, rules.TARGET,
                                            sequences=[row['sequence'] for row in rows])
    unseen = [r for r in rows if r['sequence'] not in evaluated]
    candidates = unseen[:args.epoch_proposal_budget]
    atomic(root / 'pr100_coverage.json', {'epoch': epoch, 'challenge': expected,
           'eligible_bank': len(rows), 'previously_evaluated': len(rows)-len(unseen),
           'unseen': len(unseen), 'proposals': len(candidates),
           'fresh_evaluation_budget': args.epoch_candidate_budget,
           'shortfall_before_novelty': max(0, args.epoch_candidate_budget-len(candidates))})
    # An exhausted (or empty) challenge bank is a fresh-work shortfall, not a
    # reason to suppress the separate cached-candidate eligibility precheck.
    # Emit the empty pool so the normal epoch path can still start that work.
    atomic(output, {'target': rules.TARGET, 'challenge': expected,
           'origin_epoch': epoch, 'target_epoch': epoch, 'prepared_during_epoch': state['epoch'],
           'candidate_budget': args.epoch_candidate_budget, 'proposal_budget': args.epoch_proposal_budget,
           'exact_admission_limit': args.epoch_candidate_budget,
           'screen_budget_policy': 'unseen-candidates-v2',
           'admission_policy': 'current_challenge_before_target_specific_fresh_budget',
           'candidates': candidates, 'sequences': [r['sequence'] for r in candidates]})
    return output

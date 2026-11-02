"""Exact-cell leaderboard CDR3 graft lane for PR100 proposal generation.

The validator restricts each epoch to one germline and one exact CDR3 length.
A winning CDR3 can be re-used on a different same-V framework with a few
neutral substitutions; this lane emits exactly those children. The framework
(fr1/cdr1/fr2/cdr2/fr3/fr4) comes from one exact-cell donor and the CDR3 comes
from either another exact-cell donor (verbatim + bounded neutral substitutions)
or a deterministic de-novo recipe sampled from the validated winner motifs.
Both modes share the same framework donor, divergence gate and downstream
gates.

Every emitted residue has an exact donor interval plus an explicit edit list
(for the verbatim mode) or a recorded de-novo CDR3, so history ingestion and
lineage analytics can reconstruct the child from the recorded provenance.
Children still require ordinary live gates.
"""
from __future__ import annotations

from collections import Counter
import random
import re

from nanobody_opt import rules
from nanobody_opt import pr100_reference_donors as reference_donors
from nanobody_opt import pr100_segment_generation as segments
from nanobody_opt.pr100_reference_donors import merit

SOURCE = 'leaderboard_graft'
PROTOCOL = 'imgt-exact-cell-cdr3-graft-v1'
PROPOSAL_LANE = 'graft_cdr3'
DE_NOVO_PART_SOURCE = 'de_novo_recipe'

#: Bounded neutral residue swaps. Keys deliberately exclude charged residues
#: (K/R/D/E), histidine, aromatic residues (F/Y/W), cysteine and proline. None
#: of those are ever touched by the neutral substitution pass.
NEUTRAL_SWAPS = {'S': 'T', 'T': 'S', 'N': 'Q', 'Q': 'N', 'M': 'V',
                 'A': 'S', 'G': 'A', 'I': 'V', 'L': 'I', 'V': 'L'}

_DEFAULTS = {'enabled': False, 'limit': 300, 'cdr3_donors': 8,
             'framework_donors': 8, 'donor_slots': 64}
_ALLOWED_KEYS = frozenset(_DEFAULTS)
_BOUNDS = (('limit', 1, 4800), ('cdr3_donors', 1, 64),
           ('framework_donors', 1, 64), ('donor_slots', 2, 128))
_PROVENANCE_KEYS = ('source_hash', 'source_start', 'source_end', 'region',
                    'sequence', 'imgt_start', 'imgt_end')
_NEUTRAL_SUBSTITUTIONS = 3  # floor; see ``required_substitutions``
_REGION_ORDER = ('fr1', 'cdr1', 'fr2', 'cdr2', 'fr3', 'cdr3', 'fr4')
#: Validator thresholds the child must stay under against every donor.
_WHOLE_IDENTITY_MAX = 0.95
_CDR3_CLONOTYPE_MAX = 0.85
_CONSERVED_CDR3, _CONSERVED_CDR1, _CONSERVED_CDR2 = 0.80, 0.90, 0.90
#: N-x-S/T glycosylation (x any residue, deliberately stricter than x != P),
#: asparagine deamidation (NG) and aspartate isomerisation/cleavage (DG, DP).
_LIABILITY = re.compile(r'(?=(N[A-Z][ST]|NG|DG|DP))')
_DE_NOVO_MIN_LENGTH = 5
_DE_NOVO_MAX_ATTEMPTS = 4000
_MODES = ('verbatim', 'de_novo')

#: Measured de-novo winner recipe components (121-epoch winner analysis). The
#: recipe is germline-independent: the exact-cell framework donor already
#: supplies the germline, so only CDR3 length conditions the template.
_ANCHORS = ('AR', 'AK', 'SK', 'SR')
_TAILS = ('FDY', 'FDV', 'FDP', 'MDV')
_SHORT_AA = ('T', 'S', 'A', 'G', 'Q', 'N')
_MEDIUM_CORES = ('YPYHF', 'HPYHFK', 'YPYR', 'PYHF')
_LONG_CORES = ('FQPR', 'FQPRY', 'YPYR', 'FPYR', 'PYRF')
_LINKERS = ('QGSDSG', 'QGSGD')
_POSITIVE = ('K', 'R')
_NEGATIVE = ('D', 'E')
_AROMATIC_AA = ('F', 'Y')
_HYDRO_AA = ('M', 'V', 'I', 'L')


def settings(job):
    """Validate and normalize ``job['graft_generation']`` (OFF by default)."""
    supplied = job.get('graft_generation', {})
    if (not isinstance(supplied, dict)
            or type(supplied.get('enabled', False)) is not bool
            or set(supplied) - _ALLOWED_KEYS):
        raise ValueError('invalid graft_generation settings')
    result = {**_DEFAULTS, **supplied}
    for key, low, high in _BOUNDS:
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError('invalid graft_generation bound')
    return result


def select_exact_donors(bank, challenge, slots):
    """The exact-cell parent rows in merit order, never a missing-cell fallback."""
    exact, _chosen, _evidence = reference_donors.select(bank, challenge, slots,
                                                        fallback=False)
    return exact


def prepare_framework(row, numbered):
    """Thin wrapper over the shared exact-numbering preparation."""
    return segments.prepare(row, numbered)


def extract_cdr3(row, challenge):
    """The donor's IMGT CDR3, requiring the challenge's exact length."""
    from abnumber import Chain
    cdr3 = Chain(row['sequence'], scheme='imgt', cdr_definition='imgt',
                 use_anarcii=False).cdr3_seq
    if len(cdr3) != challenge['allowed_cdr3_length']:
        raise ValueError('cdr3 length mismatch')
    return cdr3


def whole_identity(left, right):
    """Positional identity over the longer length (zip over the shared prefix)."""
    size = max(len(left), len(right))
    return (sum(a == b for a, b in zip(left, right)) / size) if size else 1.0


def required_substitutions(length):
    """Smallest k with k > 0.15 * length (and at least 3).

    Identity (L-k)/L must fall below the 0.85 CDR3 clonotype threshold; three
    substitutions are not enough from L=20 upward. Integer arithmetic avoids
    float edge cases at exact multiples.
    """
    return max(_NEUTRAL_SUBSTITUTIONS, (15 * length) // 100 + 1)


def liability_count(sequence):
    """Number of (overlapping) N-x-S/T, NG, DG and DP motifs in ``sequence``."""
    return len(_LIABILITY.findall(sequence))


def neutral_substitutions(cdr3, n=None):
    """Deterministic, seedless neutral substitutions.

    The scan starts at index 2 (the AR/AK/SK/SR-style anchor A is never
    mutated) and never touches the last 3 residues. A swap that would add a
    liability motif is skipped. ``n`` defaults to ``required_substitutions``.
    Fewer than ``n`` edits are returned when the CDR3 has too few swappable
    residues; callers must treat that as a shortfall and skip the donor.

    Returns the edited CDR3 and one edit record per applied substitution. Each
    edit's ``child_offset`` is relative to the CDR3 start and ``from`` is the
    verbatim donor residue.
    """
    n = required_substitutions(len(cdr3)) if n is None else n
    edited, edits = list(cdr3), []
    for i in range(2, len(cdr3) - 3):
        if len(edits) == n:
            break
        if cdr3[i] not in NEUTRAL_SWAPS:
            continue
        trial = edited[:]
        trial[i] = NEUTRAL_SWAPS[cdr3[i]]
        if liability_count(''.join(trial)) > liability_count(''.join(edited)):
            continue
        edited = trial
        edits.append({'child_offset': i, 'from': cdr3[i], 'to': NEUTRAL_SWAPS[cdr3[i]],
                      'kind': 'neutral_cdr3_substitution'})
    return ''.join(edited), edits


def _charge(sequence):
    """Counts of positively (K/R) and negatively (D/E) charged residues."""
    return (sum(1 for c in sequence if c in _POSITIVE),
            sum(1 for c in sequence if c in _NEGATIVE))


def de_novo_cdr3(length, seed):
    """Deterministic, length-conditioned de-novo CDR3 from the winner recipe.

    The invariant anchor (AR/AK/SK/SR) and J-tail (FDY/FDV/FDP/MDV) frame the
    CDR3; the middle is filled from a length-bucket template: minimal neutral
    filler (5-10), a PYR/PYHF core (11-14), an FQPR/PYR core plus an
    alternating K/R-D/E charge network (15-18), or an FQPR/PYR core plus a
    QGSDSG/QGSGD linker padding and charge network (19-23). The charge network
    is tuned so the total (K+R) and (D+E) counts stay within one of each other,
    which bounds the CDR3 charge content that the native developability
    charge-patch gates re-measure.
    """
    if type(length) is not int or length < 5:
        raise ValueError('de-novo CDR3 recipe requires length >= 5')
    rng = random.Random(seed)
    anchor = rng.choice(_ANCHORS)
    tail = rng.choice(_TAILS)
    middle_len = length - 5
    if length <= 10:
        middle = _short_middle(middle_len, rng)
    elif length <= 14:
        middle = _medium_middle(middle_len, rng)
    elif length <= 18:
        middle = _long_middle(middle_len, rng)
    else:
        middle = _xlong_middle(middle_len, rng)
    return anchor + middle + tail


def _pick(pool, built, rng):
    """Choose from ``pool`` without extending a run of one residue past two."""
    allowed = [r for r in pool if built[-2:] != [r, r]]
    return rng.choice(allowed or list(pool))


def _fill_middle(prefix, middle_len, rng):
    """Fill to ``middle_len`` with a winner-proportioned residue mix.

    The middle is not a monotone K/R-D/E run: roughly 40% charged (sign
    alternating to keep K+R within one of D+E), 30% polar/small, 20% aromatic
    and 10% hydrophobic, matching the 121-epoch winner middle composition.
    Consecutive charged runs are capped at two so the native developability
    charge-patch gates are not provoked, and no residue of any class (charged,
    polar, aromatic, hydrophobic) repeats more than twice in a row, counting
    the prefix.
    """
    prefix_pos, prefix_neg = _charge(prefix)
    net_len = middle_len - len(prefix)
    next_positive = (prefix_pos - prefix_neg) < 0
    built = list(prefix)
    charge_run = 0
    for _ in range(net_len):
        if charge_run >= 2 or rng.random() >= 0.40:
            roll = rng.random()
            if roll < 0.50:
                built.append(_pick(_SHORT_AA, built, rng))
            elif roll < 0.83:
                built.append(_pick(_AROMATIC_AA, built, rng))
            else:
                built.append(_pick(_HYDRO_AA, built, rng))
            charge_run = 0
        else:
            built.append(_pick(_POSITIVE if next_positive else _NEGATIVE, built, rng))
            next_positive = not next_positive
            charge_run += 1
    return ''.join(built)


def _filler(prefix, count, rng):
    """``count`` neutral residues continuing ``prefix`` under the max-run-2 cap."""
    built = list(prefix)
    for _ in range(count):
        built.append(_pick(_SHORT_AA, built, rng))
    return ''.join(built[len(prefix):])


def _short_middle(n, rng):
    """Minimal neutral filler; no aromatic/His core."""
    return _filler([], n, rng)


def _medium_middle(n, rng):
    """One PYR/PYHF core motif, extended with neutral filler to the exact length."""
    core = rng.choice(_MEDIUM_CORES)[:n]
    return core + _filler(list(core), n - len(core), rng)


def _long_middle(n, rng):
    """An FQPR/PYR core followed by an alternating charge network."""
    core = rng.choice(_LONG_CORES)[:n]
    return _fill_middle(core, n, rng)


def _xlong_middle(n, rng):
    """An FQPR/PYR core plus a QGSDSG/QGSGD linker, then a charge network."""
    core = rng.choice(_LONG_CORES)
    linker = rng.choice(_LINKERS)
    prefix = (core + linker)[:n]
    return _fill_middle(prefix, n, rng)


def reconstruct_with_edits(row, references):
    """Exact residue-level provenance check for graft rows, edits included.

    Validates legitimacy, not only internal consistency: exactly the seven IMGT
    regions in order; the six framework/CDR1/CDR2 parts verbatim from a single
    donor interval; only the cdr3 part may be de-novo (byte-identical recorded
    sequence) or carry edits; edits are bounded by the substitution count for
    that CDR3 length, unique by offset, and exactly the neutral swap of the
    donor residue. Any violation, including malformed parts, raises ValueError.
    """
    if not isinstance(row, dict) or not isinstance(row.get('sequence'), str):
        raise ValueError('invalid graft row')
    parts = row.get('segment_provenance')
    if not isinstance(parts, list) or len(parts) != len(_REGION_ORDER):
        raise ValueError('invalid graft provenance')
    if not isinstance(references, dict):
        raise ValueError('invalid graft references')
    assembled, cursor, framework_sources, cdr3_source = [], 0, set(), None
    for index, (part, region) in enumerate(zip(parts, _REGION_ORDER)):
        if not isinstance(part, dict) or part.get('region') != region:
            raise ValueError('graft provenance region order mismatch')
        c, d = part.get('child_start'), part.get('child_end')
        edits = part.get('edits', [])
        if not isinstance(edits, list) or (edits and region != 'cdr3'):
            raise ValueError('invalid graft edits')
        if part.get('source') == DE_NOVO_PART_SOURCE:
            piece = part.get('sequence')
            if (region != 'cdr3' or edits or not isinstance(piece, str)
                    or type(c) is not int or type(d) is not int
                    or c != cursor or d - c != len(piece)
                    or 'source_hash' in part):
                raise ValueError('graft de-novo cdr3 mismatch')
        else:
            if 'source' in part:
                raise ValueError('graft provenance source mismatch')
            sha, a, b = part.get('source_hash'), part.get('source_start'), part.get('source_end')
            source = references.get(sha) if isinstance(sha, str) else None
            if (not isinstance(source, str) or rules.sequence_hash(source) != sha
                    or type(a) is not int or type(b) is not int or not 0 <= a < b <= len(source)
                    or type(c) is not int or type(d) is not int or c != cursor
                    or d - c != b - a):
                raise ValueError('graft provenance interval mismatch')
            piece = source[a:b]
            if region == 'cdr3':
                cdr3_source = sha
            else:
                framework_sources.add(sha)
        assembled.append(piece)
        cursor = d
    if len(framework_sources) != 1 or cdr3_source in framework_sources:
        raise ValueError('graft provenance source mismatch')
    result = ''.join(assembled)
    cdr3_part, cdr3_piece = parts[5], assembled[5]
    edits = cdr3_part.get('edits', [])
    if len(edits) > required_substitutions(len(cdr3_piece)):
        raise ValueError('graft edit count exceeds substitution cap')
    seen = set()
    for edit in edits:
        if not isinstance(edit, dict):
            raise ValueError('graft edit mismatch')
        offset = edit.get('child_offset')
        source_residue = edit.get('from')
        if (edit.get('kind') != 'neutral_cdr3_substitution'
                or type(offset) is not int or not 0 <= offset < len(cdr3_piece)
                or offset in seen
                or not isinstance(source_residue, str)
                or source_residue != cdr3_piece[offset]
                or edit.get('to') != NEUTRAL_SWAPS.get(source_residue)):
            raise ValueError('graft edit mismatch')
        seen.add(offset)
        position = cdr3_part['child_start'] + offset
        result = result[:position] + edit['to'] + result[position + 1:]
    if result != row['sequence']:
        raise ValueError('graft provenance reconstruction mismatch')
    return result


def _emit_row(child, sha, framework, cdr3_part, cdr3_source_hash, neutral_subs,
              shortfall, graft_mode, challenge):
    """Build the shared emitted row shape for both graft modes."""
    provenance, cursor = [], 0
    for name in ('fr1', 'cdr1', 'fr2', 'cdr2', 'fr3'):
        segment = framework['segments'][name]
        end = cursor + len(segment['sequence'])
        provenance.append({k: segment[k] for k in _PROVENANCE_KEYS}
                          | {'child_start': cursor, 'child_end': end})
        cursor = end
    end = cursor + len(cdr3_part['sequence'])
    provenance.append({**cdr3_part, 'child_start': cursor, 'child_end': end})
    cursor = end
    fr4 = framework['segments']['fr4']
    fr4_end = cursor + len(fr4['sequence'])
    provenance.append({k: fr4[k] for k in _PROVENANCE_KEYS}
                      | {'child_start': cursor, 'child_end': fr4_end})
    return {'sequence': child, 'sequence_hash': sha, 'source': SOURCE,
            'proposal_type': PROTOCOL, 'proposal_lane': PROPOSAL_LANE,
            'segment_layout': 'cdr3_graft', 'segment_provenance': provenance,
            'graft_cdr3_source_hash': cdr3_source_hash,
            'graft_framework_source_hash': framework['sequence_hash'],
            'graft_neutral_subs': neutral_subs, 'graft_neutral_subs_shortfall': shortfall,
            'graft_mode': graft_mode, 'requested_v': challenge['allowed_v'],
            'requested_cdr3_length': challenge['allowed_cdr3_length'],
            'acceptance_qualified': False, 'target_scored': False}


def _child_cdrs(framework, cdr3):
    return (framework['segments']['cdr1']['sequence'],
            framework['segments']['cdr2']['sequence'], cdr3)


def _clears_donor(child, child_cdrs, donor):
    """True when the CHILD (not the donor pair) is far enough from ``donor``."""
    cdr1, cdr2, cdr3 = child_cdrs
    donor_cdr3 = donor['segments']['cdr3']['sequence']
    cdr3_similarity = segments.cdr_similarity(cdr3, donor_cdr3)
    if (whole_identity(child, donor['sequence']) >= _WHOLE_IDENTITY_MAX
            or cdr3_similarity >= _CDR3_CLONOTYPE_MAX):
        return False
    return not (cdr3_similarity >= _CONSERVED_CDR3
                and segments.cdr_similarity(cdr1, donor['segments']['cdr1']['sequence'])
                >= _CONSERVED_CDR1
                and segments.cdr_similarity(cdr2, donor['segments']['cdr2']['sequence'])
                >= _CONSERVED_CDR2)


def _assemble(framework, cdr3):
    s = framework['segments']
    return (s['fr1']['sequence'] + s['cdr1']['sequence'] + s['fr2']['sequence']
            + s['cdr2']['sequence'] + s['fr3']['sequence'] + cdr3 + s['fr4']['sequence'])


class _Clonotypes:
    """Emitted CDR3s; a new CDR3 at or above the clonotype threshold is a dup."""

    def __init__(self):
        self.exact, self.items = set(), []

    def duplicate(self, cdr3):
        return cdr3 in self.exact or any(
            segments.cdr_similarity(cdr3, other) >= _CDR3_CLONOTYPE_MAX
            for other in self.items)

    def add(self, cdr3):
        self.exact.add(cdr3)
        self.items.append(cdr3)


def generate(cdr3_donors, framework_donors, challenge, *, limit=300,
             excluded=(), stats=None, modes=_MODES, proposal_offset=0):
    """Emit exact-cell CDR3 grafts, in deterministic merit order, up to ``limit``.

    ``cdr3_donors`` are raw exact-cell bank rows (prepared here). ``framework_donors``
    are already-prepared exact-cell rows from ``prepare_framework``.

    ``verbatim``: each (CDR3 donor, framework donor) pair emits the donor CDR3
    with ``required_substitutions`` neutral swaps. A donor whose CDR3 has fewer
    swappable residues is skipped. ``de_novo``: recipe-sampled CDR3s on each
    framework donor in round-robin, seeded from epoch, ``proposal_offset`` and
    the attempt number so continuation yields fresh candidates.

    Every emitted CHILD must clear the validator-style gates against the donors
    it was built from (whole identity < 0.95, CDR3 similarity < 0.85, conserved
    CDR rule), carry no liability motif in the generated region, and be unique
    by sequence and by CDR3 clonotype. A graft needs at least two distinct
    exact-cell donors, so fewer than two emit nothing.
    """
    from abnumber import Chain
    if (type(limit) is not int or not 1 <= limit <= 4800
            or not isinstance(cdr3_donors, list) or not isinstance(framework_donors, list)
            or type(proposal_offset) is not int or proposal_offset < 0):
        raise ValueError('invalid graft generation input bound')
    modes = tuple(modes)
    if not modes or any(mode not in _MODES for mode in modes):
        raise ValueError('invalid graft mode')
    counters = Counter()
    distinct = {row['sequence_hash'] for row in [*cdr3_donors, *framework_donors]}
    if len(distinct) < 2:
        counters['stop_reason'] = 'insufficient_exact_cell_donors'
        counters['emitted'] = 0
        counters['filled'] = False
        if stats is not None:
            stats.update(counters)
        return []

    def key(row):
        rank, _kind = merit(row, challenge)
        return (*rank, row['sequence_hash'])

    cdr3_donors = sorted(cdr3_donors, key=key)
    framework_donors = sorted(framework_donors, key=key)
    references = {fd['sequence_hash']: fd['sequence'] for fd in framework_donors}
    length = challenge['allowed_cdr3_length']
    prepared_cds = []
    for cd in cdr3_donors:
        try:
            chain = Chain(cd['sequence'], scheme='imgt', cdr_definition='imgt',
                          use_anarcii=False)
            numbered = [{'number': pos.number, 'letter': pos.letter, 'aa': aa}
                        for pos, aa in chain]
            prepared = prepare_framework(cd, numbered)
        except Exception:
            counters['donor_numbering_failed'] += 1
            continue
        cdr3 = prepared['segments']['cdr3']['sequence']
        if len(cdr3) != length:
            counters['cdr3_length_mismatch'] += 1
            continue
        references[cd['sequence_hash']] = cd['sequence']
        prepared_cds.append((prepared, cdr3))

    excluded = set(excluded)
    result, seen_sequences, clonotypes = [], set(), _Clonotypes()
    stop_reason = 'donor_pairs_exhausted'

    def finish(child, sha, fd, cdr3_part, cdr3_hash, subs, shortfall, mode, cdr3):
        row = _emit_row(child, sha, fd, cdr3_part, cdr3_hash, subs, shortfall,
                        mode, challenge)
        if reconstruct_with_edits(row, references) != child:
            raise ValueError('graft reconstruction mismatch')
        result.append(row)
        seen_sequences.add(child)
        clonotypes.add(cdr3)
        return len(result) >= limit

    if 'verbatim' in modes:
        needed = required_substitutions(length)
        for prepared, cdr3_verbatim in prepared_cds:
            edited_cdr3, edits = neutral_substitutions(cdr3_verbatim, needed)
            if len(edits) < needed:
                counters['cdr3_insufficient_swappable_residues'] += 1
                continue
            if liability_count(edited_cdr3) > liability_count(cdr3_verbatim):
                counters['liability_drops'] += 1
                continue
            for fd in framework_donors:
                if fd['sequence_hash'] == prepared['sequence_hash']:
                    continue
                child = _assemble(fd, edited_cdr3)
                cdrs = _child_cdrs(fd, edited_cdr3)
                if not (_clears_donor(child, cdrs, prepared)
                        and _clears_donor(child, cdrs, fd)):
                    counters['divergence_drops'] += 1
                    continue
                sha = rules.sequence_hash(child)
                if sha in excluded:
                    counters['exact_or_resume_collisions'] += 1
                    continue
                if child in seen_sequences or clonotypes.duplicate(edited_cdr3):
                    counters['duplicate_drops'] += 1
                    continue
                cdr3_part = {k: prepared['segments']['cdr3'][k] for k in _PROVENANCE_KEYS}
                cdr3_part['edits'] = edits
                if finish(child, sha, fd, cdr3_part, prepared['sequence_hash'],
                          len(edits), 0, 'verbatim', edited_cdr3):
                    stop_reason = 'limit_filled'
                    break
            if stop_reason == 'limit_filled':
                break

    if 'de_novo' in modes and stop_reason != 'limit_filled':
        if length < _DE_NOVO_MIN_LENGTH:
            counters['de_novo_length_unsupported'] += 1
        elif framework_donors:
            attempts = min(limit * 8, _DE_NOVO_MAX_ATTEMPTS)
            base = challenge['epoch'] * 1_000_000_000 + proposal_offset * 1_000_000
            for attempt in range(attempts):
                fd = framework_donors[attempt % len(framework_donors)]
                generated = de_novo_cdr3(length, base + attempt)
                if liability_count(generated):
                    counters['liability_drops'] += 1
                    continue
                child = _assemble(fd, generated)
                if not _clears_donor(child, _child_cdrs(fd, generated), fd):
                    counters['divergence_drops'] += 1
                    continue
                sha = rules.sequence_hash(child)
                if sha in excluded:
                    counters['exact_or_resume_collisions'] += 1
                    continue
                if child in seen_sequences or clonotypes.duplicate(generated):
                    counters['duplicate_drops'] += 1
                    continue
                cdr3_part = {'source': DE_NOVO_PART_SOURCE, 'region': 'cdr3',
                             'sequence': generated}
                if finish(child, sha, fd, cdr3_part, None, 0, 0, 'de_novo', generated):
                    stop_reason = 'limit_filled'
                    break
    counters['emitted'] = len(result)
    counters['filled'] = stop_reason == 'limit_filled'
    counters['stop_reason'] = stop_reason
    if stats is not None:
        stats.update(counters)
    return result

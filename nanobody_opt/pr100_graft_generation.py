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
_VALIDATOR_WHOLE_IDENTITY = 0.95
_VALIDATOR_CDR3_CLONOTYPE = 0.85
#: Safety margin below the validator thresholds. The fleet/validator side applies
#: alignment slack (it gates at 0.93 whole identity), so borderline children that
#: only just clear 0.95 / 0.85 are not emitted. Gates use the margined values.
_WHOLE_IDENTITY_MARGIN = 0.02
_CDR3_CLONOTYPE_MARGIN = 0.02
_WHOLE_IDENTITY_MAX = round(_VALIDATOR_WHOLE_IDENTITY - _WHOLE_IDENTITY_MARGIN, 2)  # 0.93
_CDR3_CLONOTYPE_MAX = round(_VALIDATOR_CDR3_CLONOTYPE - _CDR3_CLONOTYPE_MARGIN, 2)  # 0.83
#: Conserved-CDR rule (unchanged): a child is a derivative when CDR3 >= 0.80 AND
#: CDR1 >= 0.90 AND CDR2 >= 0.90 against a donor.
_CONSERVED_CDR3, _CONSERVED_CDR1, _CONSERVED_CDR2 = 0.80, 0.90, 0.90
_AMINO_ACIDS = frozenset('ACDEFGHIKLMNPQRSTVWY')
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
#: FDP is deliberately absent: its DP is a liability motif, so every FDP-tailed
#: candidate would be generated only to be rejected.
_TAILS = ('FDY', 'FDV', 'MDV')
_SHORT_AA = ('T', 'S', 'A', 'G', 'Q', 'N')
#: Weighted core pools, ``(core, weight)``. The dominant cores are the
#: 131-epoch Q9NZQ7-era winner backbone FQPR + PRY/YPR (score-weighted over
#: epochs 25959-26096): FQPRYPRT at medium length, FQPRYPRTQN at long length.
#: The remaining entries are the secondary winner families from the 26092-26097
#: submission sweep (FQYKFR x4 26096, HYGKFR x5 26094, FQHFKR x4 26092, GFHPTQH
#: x2 26096; the leader HYGDFRH 26096 is the HYGKFR family), kept so every
#: length bucket retains a winner-family core while the FQPR+PRY/YPR backbone
#: dominates. The older Y-P-Y-R/P-Y-H-F cores (YPYHF/HPYHFK/YPYR/PYHF) never
#: appear in the winner set, so they are removed rather than kept.
_MEDIUM_CORES = (
    ('FQPRYPRT', 7.0),                      # dominant F-[QE]-P-R-Y-P-[RK]-T backbone
    ('FQYKFRT', 4.0),                       # 26096 winner (FQYKFR family)
    ('HYGKFREFQD', 2.0), ('GFHPTQH', 2.0),  # 26094 / 26096 winners
    ('HYGDFRH', 1.0),                       # HYGKFR-family leader variant
)
_LONG_CORES = (
    ('FQPRYPRTQN', 8.0), ('FEPRYPKTEHHY', 7.0),  # dominant backbone (26097 x6, 26093 x5)
    ('HYGKFREFQD', 3.0),                          # 26094 x5
    ('FQHFKRSNESD', 2.0),                         # 26092 x4 (highest liability, de-weighted)
    ('FQYKFRT', 1.0),                             # 26096 (short, charge-network padded)
)
#: Probability that a sampled winner core is emitted with 1-2 sign-preserving
#: conservative swaps (needed so the clonotype de-duplication still yields many
#: distinct CDR3s rather than one near-identical copy per core).
_CORE_VARIATION = 0.5
_CONSERVATIVE = {'F': 'Y', 'Y': 'F', 'Q': 'N', 'N': 'Q', 'K': 'R', 'R': 'K',
                 'D': 'E', 'E': 'D', 'T': 'S', 'S': 'T', 'I': 'V', 'V': 'I',
                 'L': 'M', 'M': 'L'}
_LINKERS = ('QGSDSG', 'QGSGD')
_POSITIVE = ('K', 'R')
_NEGATIVE = ('D', 'E')
#: 131-epoch winner charge-network motifs (score-weighted k-mers, epochs
#: 25959-26096). Each bundles charged K/R/D/E with P/T/F context (KPR/FKP/DKT/
#: KTR) instead of a bare alternating K/R-D/E run. The sampler uses each motif's
#: net charge (K+R minus D+E) so the whole CDR3 stays within one of balance.
_CHARGE_MOTIFS = (('KPR', 0.5), ('FKP', 0.5), ('DKT', 0.4), ('KTR', 0.4))
_AROMATIC_AA = ('F', 'Y')
_HYDRO_AA = ('M', 'V', 'I', 'L')
_HIS_AA = ('H',)
#: Fraction of the padding that is charged (K/R/D/E via the winner motifs),
#: reduced from 0.40 to match the passing set's ~32% charged middle.
_CHARGED_FRACTION = 0.35
#: Non-charged fill mix, calibrated to the measured passing developability set
#: (1256 profiles in ``q9-developability.sqlite``): polar/small, aromatic,
#: histidine, then strong-hydrophobic M/V/I/L. The passing set carries ~5%
#: M/V/I/L overall, while the old 10%-hydrophobic padding over-emitted them by
#: ~40%; M/V/I/L are the dominant PSH drivers, so they are halved here and the
#: freed weight goes to polar/small plus the histidine the winners use.
_NEUTRAL_FILL = (('polar', 0.55), ('aromatic', 0.30), ('histidine', 0.10),
                 ('hydrophobic', 0.05))


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
    """Smallest k with k > 0.20 * length (and at least 3).

    The verbatim CDR3 must clear the conserved-CDR rule as well as the clonotype
    bound: CDR3 identity (L-k)/L has to fall below 0.80 (the conserved-CDR3
    threshold), which also keeps it below the margined clonotype bound. At
    L=15-23 a 0.15-based floor leaves CDR3 identity at 0.80-0.84, so every
    verbatim pair would be dropped. Integer arithmetic avoids float edge cases
    at exact multiples.
    """
    return max(_NEUTRAL_SUBSTITUTIONS, (20 * length) // 100 + 1)


def _substitution_seed(cdr3_donor_hash, framework_donor_hash):
    """Seed that varies the neutral-substitution subset per framework donor."""
    return f'{cdr3_donor_hash}:{framework_donor_hash}'


def liability_count(sequence):
    """Number of (overlapping) N-x-S/T, NG, DG and DP motifs in ``sequence``."""
    return len(_LIABILITY.findall(sequence))


def neutral_substitutions(cdr3, n=None, seed=None):
    """Deterministic neutral substitutions.

    The scan starts at index 2 (the AR/AK/SK/SR-style anchor A is never
    mutated) and never touches the last 3 residues. A swap that would add a
    liability motif is skipped. ``n`` defaults to ``required_substitutions``.
    With ``seed=None`` the first ``n`` swappable positions are edited (the
    original, seedless behaviour). With a ``seed`` the candidate positions are
    visited in a seed-determined order, so different seeds select different
    position subsets of the same CDR3 (still deterministic per seed); edits are
    returned in position order either way.
    Fewer than ``n`` edits are returned when the CDR3 has too few swappable
    residues; callers must treat that as a shortfall and skip the donor.

    Returns the edited CDR3 and one edit record per applied substitution. Each
    edit's ``child_offset`` is relative to the CDR3 start and ``from`` is the
    verbatim donor residue.
    """
    n = required_substitutions(len(cdr3)) if n is None else n
    edited, edits = list(cdr3), []
    order = list(range(2, len(cdr3) - 3))
    if seed is not None:
        random.Random(str(seed)).shuffle(order)
    for i in order:
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
    edits.sort(key=lambda edit: edit['child_offset'])
    return ''.join(edited), edits


def _charge(sequence):
    """Counts of positively (K/R) and negatively (D/E) charged residues."""
    return (sum(1 for c in sequence if c in _POSITIVE),
            sum(1 for c in sequence if c in _NEGATIVE))


#: Length buckets shared with ``pr100_recipe_prior`` (inclusive upper bounds).
RECIPE_BUCKETS = (('short', 5, 10), ('medium', 11, 14), ('long', 15, 18),
                  ('xlong', 19, 23))
_PRIOR_ANCHOR_LEN, _PRIOR_TAIL_LEN, _PRIOR_MAX_CORE_LEN = 2, 3, 12


def length_bucket(length):
    """Recipe bucket name for a CDR3 length (>= 24 is treated as ``xlong``)."""
    if length <= RECIPE_BUCKETS[0][2]:
        return RECIPE_BUCKETS[0][0]
    for name, _low, high in RECIPE_BUCKETS[1:-1]:
        if length <= high:
            return name
    return RECIPE_BUCKETS[-1][0]


def _bucket_motifs(prior, bucket):
    """Well-formed ``anchors``/``cores``/``tails`` tuples from ``prior[bucket]``.

    Missing, malformed or wrong-sized entries are dropped (never raised on) so a
    partly bad prior degrades to the static pool for that motif kind.
    """
    entry = ((prior or {}).get('by_length') or {}).get(bucket)
    entry = entry if isinstance(entry, dict) else {}

    def clean(kind, low, high):
        values = entry.get(kind)
        if not isinstance(values, (list, tuple)):
            return ()
        return tuple(dict.fromkeys(
            v for v in values
            if isinstance(v, str) and low <= len(v) <= high and set(v) <= _AMINO_ACIDS))

    return {'anchors': clean('anchors', _PRIOR_ANCHOR_LEN, _PRIOR_ANCHOR_LEN),
            'tails': clean('tails', _PRIOR_TAIL_LEN, _PRIOR_TAIL_LEN),
            'cores': clean('cores', 1, _PRIOR_MAX_CORE_LEN)}


def _core_pool(prior, bucket, static_pool):
    """Weighted ``(core, weight)`` pool: the prior's cores if any, else static.

    Prior cores arrive best-first, so they get linearly decreasing rank weights
    (``len`` down to 1); the pool is then sampled by ``_pick_core`` exactly like
    the static weighted pools (variation, charge-safety, truncation included).
    """
    cores = _bucket_motifs(prior, bucket)['cores']
    if not cores:
        return static_pool
    return tuple((core, float(len(cores) - i)) for i, core in enumerate(cores))


def de_novo_cdr3(length, seed, prior=None):
    """Deterministic, length-conditioned de-novo CDR3 from the winner recipe.

    The invariant anchor (AR/AK/SK/SR) and J-tail (FDY/FDV/MDV) frame the
    CDR3; the middle is filled from a length-bucket template: minimal neutral
    filler (5-10), a weighted winner core such as FQYKFRT/FQPRYPRT (11-14), a
    weighted winner core such as FQPRYPRTQN/FEPRYPKTEHHY plus an alternating
    K/R-D/E charge network (15-18), or a winner core plus a QGSDSG/QGSGD linker
    padding and charge network (19-23). Cores are sampled by observed winner
    frequency (see ``_LONG_CORES``). The charge network
    is tuned so the total (K+R) and (D+E) counts stay within one of each other,
    which bounds the CDR3 charge content that the native developability
    charge-patch gates re-measure.

    ``prior`` is an optional ``recipe-prior-v1`` document (see
    ``pr100_recipe_prior``). For the CDR3's length bucket, every non-empty
    prior anchor/core/tail list replaces the matching static pool; an empty or
    absent list falls back to the static constants, so ``prior=None`` is
    byte-identical to the static recipe.
    """
    if type(length) is not int or length < 5:
        raise ValueError('de-novo CDR3 recipe requires length >= 5')
    rng = random.Random(seed)
    bucket = length_bucket(length)
    motifs = _bucket_motifs(prior, bucket)
    anchor = rng.choice(motifs['anchors'] or _ANCHORS)
    tail = rng.choice(motifs['tails'] or _TAILS)
    middle_len = length - 5
    if length <= 10:
        middle = _short_middle(middle_len, rng, prior)
    elif length <= 14:
        middle = _medium_middle(middle_len, rng, prior)
    elif length <= 18:
        middle = _long_middle(middle_len, rng, prior)
    else:
        middle = _xlong_middle(middle_len, rng, prior)
    return anchor + middle + tail


def _pick(pool, built, rng):
    """Choose from ``pool`` without extending a run of one residue past two."""
    allowed = [r for r in pool if built[-2:] != [r, r]]
    return rng.choice(allowed or list(pool))


def _motif_fits(built, motif):
    """True when appending ``motif`` never extends any residue run to three."""
    return not (len(built) >= 2 and built[-2] == built[-1] == motif[0])


def _fill_middle(prefix, middle_len, rng):
    """Fill to ``middle_len`` with a winner-proportioned residue mix.

    The middle is not a monotone K/R-D/E run: roughly 35% charged, ~36%
    polar/small, ~20% aromatic, ~6% histidine and ~3% strong-hydrophobic
    M/V/I/L, calibrated to the measured passing developability composition
    (see ``_NEUTRAL_FILL``). The charged part is sampled from the observed
    winner charge motifs (KPR/FKP/DKT/KTR: charged K/R/D/E bundled with P/T/F)
    plus bare K/R/D/E, and each unit is accepted only when it keeps the total
    (K+R) and (D+E) counts within one of each other (the anchor already
    contributes one positive and the tail one negative). Consecutive charged
    runs are capped at two so the native developability charge-patch gates are
    not provoked, and no residue of any class (charged, polar, aromatic,
    histidine, hydrophobic) repeats more than twice in a row, counting the
    prefix.
    """
    built = list(prefix)
    pos, neg = _charge(built)
    charge_run = 0
    while len(built) < middle_len:
        room = middle_len - len(built)
        if charge_run >= 2 or rng.random() >= _CHARGED_FRACTION:
            kind = rng.choices([k for k, _w in _NEUTRAL_FILL],
                               weights=[w for _k, w in _NEUTRAL_FILL])[0]
            if kind == 'polar':
                built.append(_pick(_SHORT_AA, built, rng))
            elif kind == 'aromatic':
                built.append(_pick(_AROMATIC_AA, built, rng))
            elif kind == 'histidine':
                built.append(_pick(_HIS_AA, built, rng))
            else:
                built.append(_pick(_HYDRO_AA, built, rng))
            charge_run = 0
            continue
        net = pos - neg
        # Bare K/R/D/E always land within one of balance; the observed motifs
        # join only when their net charge would not push the CDR3 past that.
        candidates, weights = [], []
        if net < 1:
            candidates.append('pos')
            weights.append(0.5)
        if net > -1:
            candidates.append('neg')
            weights.append(0.5)
        if room >= 3:
            for motif, weight in _CHARGE_MOTIFS:
                motif_pos, motif_neg = _charge(motif)
                if (abs(net + motif_pos - motif_neg) <= 1
                        and _motif_fits(built, motif)):
                    candidates.append(motif)
                    weights.append(weight)
        if not candidates:  # defensive: never reached, but keeps the loop moving
            built.append(_pick(_SHORT_AA, built, rng))
            charge_run = 0
            continue
        choice = rng.choices(candidates, weights=weights)[0]
        if choice == 'pos':
            built.append(_pick(_POSITIVE, built, rng))
            pos += 1
            charge_run += 1
        elif choice == 'neg':
            built.append(_pick(_NEGATIVE, built, rng))
            neg += 1
            charge_run += 1
        else:
            motif_pos, motif_neg = _charge(choice)
            built.extend(choice)
            pos += motif_pos
            neg += motif_neg
            charge_run += motif_pos + motif_neg
    return ''.join(built)


def _filler(prefix, count, rng):
    """``count`` neutral residues continuing ``prefix`` under the max-run-2 cap."""
    built = list(prefix)
    for _ in range(count):
        built.append(_pick(_SHORT_AA, built, rng))
    return ''.join(built[len(prefix):])


def _short_middle(n, rng, prior=None):
    """Minimal neutral filler; no aromatic/His core unless the prior has cores."""
    pool = _core_pool(prior, 'short', ())
    if not pool:
        return _filler([], n, rng)
    core = _pick_core(pool, n, rng)
    return core + _filler(list(core), n - len(core), rng)


def _charge_safe(prefix, n):
    """``prefix`` adjusted so K+R and D+E differ by at most one, or ``None``.

    Cationic winner cores (e.g. FQPRYPRTQN, net +2) cannot sit in a CDR3 whose
    anchor/tail already balance (AR + FD*) without breaking the one-residue
    charge-balance bound. Preference order: keep as is; append a compensating
    E if the middle has room; otherwise swap the first neutral polar residue
    (Q/N/S/T) for E (an exchange seen in the winners: F[QE]PRYP[RK]T).
    """
    built = prefix
    while True:
        pos, neg = _charge(built)
        if abs(pos - neg) <= 1:
            return built
        if pos < neg:
            return None
        if pos - neg - 1 <= min(2, n - len(built)):
            return built + 'E' * (pos - neg - 1)
        for polar in 'QNST':
            index = built.find(polar)
            if index >= 0:
                built = built[:index] + 'E' + built[index + 1:]
                break
        else:
            return None


def _pick_core(pool, n, rng, suffix=''):
    """Weighted winner-core choice for a middle of length ``n``.

    Deterministic under ``rng``. The core is optionally varied by sign-preserving
    conservative swaps, truncated to leave room for ``suffix`` (the linker is
    never truncated away), then made charge-safe; a core that cannot be made
    charge-safe is dropped and the draw repeated. Returns ``''`` only if no pool
    entry is feasible.
    """
    remaining = list(pool)
    while remaining:
        entry = rng.choices(remaining, weights=[w for _, w in remaining])[0]
        core = entry[0]
        if rng.random() < _CORE_VARIATION:
            built = list(core)
            for i in rng.sample(range(len(core)), min(rng.randint(1, 2), len(core))):
                built[i] = _CONSERVATIVE.get(built[i], built[i])
            core = ''.join(built)
        head = core[:n - len(suffix)] if suffix else core[:n]
        safe = _charge_safe((head + suffix)[:n], n)
        if safe is not None:
            return safe
        remaining.remove(entry)
    return ''


def _medium_middle(n, rng, prior=None):
    """One winner core (truncated to ``n``), extended with neutral filler."""
    core = _pick_core(_core_pool(prior, 'medium', _MEDIUM_CORES), n, rng)
    return core + _filler(list(core), n - len(core), rng)


def _long_middle(n, rng, prior=None):
    """A winner core followed by an alternating charge network."""
    pool = _core_pool(prior, 'long', _LONG_CORES)
    return _fill_middle(_pick_core(pool, n, rng), n, rng)


def _xlong_middle(n, rng, prior=None):
    """A winner core plus a QGSDSG/QGSGD linker, then a charge network."""
    linker = rng.choice(_LINKERS)
    pool = _core_pool(prior, 'xlong', _LONG_CORES)
    return _fill_middle(_pick_core(pool, n, rng, suffix=linker), n, rng)


def reconstruct_with_edits(row, references):
    """Exact residue-level provenance check for graft rows, edits included.

    Validates legitimacy, not only internal consistency: exactly the seven IMGT
    regions in order; the six framework/CDR1/CDR2 parts verbatim from a single
    donor interval; only the cdr3 part may be de-novo (non-empty, valid amino
    acids, byte-identical recorded sequence) or carry edits. A verbatim CDR3
    carries 1..``required_substitutions`` edits that are exactly the
    deterministic ``neutral_substitutions`` output (offsets only inside the
    scanned window, never arbitrary). ``graft_mode`` and the recorded CDR3 /
    framework source hashes must agree with the parts. Any violation, including
    malformed parts, raises ValueError.
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
                    or not piece or not set(piece) <= _AMINO_ACIDS
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
    de_novo = cdr3_part.get('source') == DE_NOVO_PART_SOURCE
    # graft_mode is journaled with every row; the two source-hash fields are not part
    # of the history journal's GENERATION_FIELDS, so they are cross-checked whenever
    # the row carries them (always true for freshly emitted rows).
    if row.get('graft_mode') != ('de_novo' if de_novo else 'verbatim'):
        raise ValueError('graft provenance metadata mismatch')
    for field, expected in (('graft_cdr3_source_hash', cdr3_source),
                            ('graft_framework_source_hash', next(iter(framework_sources)))):
        if field in row and row[field] != expected:
            raise ValueError('graft provenance metadata mismatch')
    if de_novo:
        if edits:
            raise ValueError('graft de-novo cdr3 mismatch')
    elif not 1 <= len(edits) <= required_substitutions(len(cdr3_piece)):
        raise ValueError('graft edit count outside substitution bounds')
    if 'graft_neutral_subs' in row and row['graft_neutral_subs'] != len(edits):
        raise ValueError('graft edit count metadata mismatch')
    seen = set()
    for edit in edits:
        if not isinstance(edit, dict):
            raise ValueError('graft edit mismatch')
        offset = edit.get('child_offset')
        source_residue = edit.get('from')
        if (edit.get('kind') != 'neutral_cdr3_substitution'
                or type(offset) is not int
                or not 2 <= offset < len(cdr3_piece) - 3  # the neutral-substitution scan window
                or offset in seen
                or not isinstance(source_residue, str)
                or source_residue != cdr3_piece[offset]
                or edit.get('to') != NEUTRAL_SWAPS.get(source_residue)):
            raise ValueError('graft edit mismatch')
        seen.add(offset)
        position = cdr3_part['child_start'] + offset
        result = result[:position] + edit['to'] + result[position + 1:]
    if edits:  # edits must be the deterministic pass output, not arbitrary swaps
        def shape(items):
            return [(e['child_offset'], e['from'], e['to'], e['kind']) for e in items]

        # Current rows use the per-(CDR3 donor, framework donor) seeded selection;
        # rows emitted before the seed was introduced used the seedless scan.
        accepted = [shape(neutral_substitutions(
            cdr3_piece, len(edits), seed=_substitution_seed(cdr3_source,
                                                           next(iter(framework_sources))))[1]),
                    shape(neutral_substitutions(cdr3_piece, len(edits))[1])]
        if shape(edits) not in accepted:
            raise ValueError('graft edit mismatch')
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
    """Emitted CDR3s; only a byte-identical CDR3 is a duplicate.

    The validator rejects >= 0.85 CDR3 similarity only against HISTORICAL
    sequences, not against same-batch siblings, so near-identical variants of
    one donor are all allowed to emit (each is still gated against its donors
    in ``_clears_donor``). Byte-identical full sequences are deduplicated
    separately via ``seen_sequences``.
    """

    def __init__(self):
        self.exact = set()

    def duplicate(self, cdr3):
        return cdr3 in self.exact

    def add(self, cdr3):
        self.exact.add(cdr3)


def generate(cdr3_donors, framework_donors, challenge, *, limit=300,
             excluded=(), stats=None, modes=_MODES, proposal_offset=0, prior=None):
    """Emit exact-cell CDR3 grafts, in deterministic merit order, up to ``limit``.

    ``cdr3_donors`` are raw exact-cell bank rows (prepared here). ``framework_donors``
    are already-prepared exact-cell rows from ``prepare_framework``.

    ``verbatim``: each (CDR3 donor, framework donor) pair emits the donor CDR3
    with ``required_substitutions`` neutral swaps. A donor whose CDR3 has fewer
    swappable residues is skipped. ``de_novo``: recipe-sampled CDR3s on each
    framework donor in round-robin, seeded from epoch, ``proposal_offset`` and
    the attempt number so continuation yields fresh candidates.

    Every emitted CHILD must clear the validator-style gates against the donors
    it was built from (whole identity < 0.93, CDR3 similarity < 0.83, conserved
    CDR rule; the margins sit below the 0.95 / 0.85 validator thresholds), carry no liability motif in the generated region, and be unique
    by sequence and by exact CDR3 (near-identical variants of one donor may all emit). A graft needs at least two distinct
    exact-cell donors, so fewer than two emit nothing.

    ``prior`` is an optional ``recipe-prior-v1`` dict (``pr100_recipe_prior``)
    that replaces the static de-novo anchors/cores/tails per length bucket; with
    ``None`` the static recipe is used. ``stats['recipe_prior_used']`` is set
    when the prior actually has motifs for this CDR3 length.
    """
    from abnumber import Chain
    if (type(limit) is not int or not 1 <= limit <= 4800
            or not isinstance(cdr3_donors, list) or not isinstance(framework_donors, list)
            or type(proposal_offset) is not int or proposal_offset < 0
            or (prior is not None and not isinstance(prior, dict))):
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
            _, probe = neutral_substitutions(cdr3_verbatim, needed)
            if len(probe) < needed:
                counters['cdr3_insufficient_swappable_residues'] += 1
                continue
            for fd in framework_donors:
                if fd['sequence_hash'] == prepared['sequence_hash']:
                    continue
                # Vary the substitution subset per (CDR3 donor, framework donor)
                # so each framework emits a distinct variant of the same donor.
                edited_cdr3, edits = neutral_substitutions(
                    cdr3_verbatim, needed,
                    seed=_substitution_seed(prepared['sequence_hash'], fd['sequence_hash']))
                if (len(edits) < needed
                        or liability_count(edited_cdr3) > liability_count(cdr3_verbatim)):
                    counters['liability_drops'] += 1
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
            if any(_bucket_motifs(prior, length_bucket(length)).values()):
                counters['recipe_prior_used'] = True
            base = challenge['epoch'] * 1_000_000_000 + proposal_offset * 1_000_000
            for attempt in range(attempts):
                fd = framework_donors[attempt % len(framework_donors)]
                generated = de_novo_cdr3(length, base + attempt, prior)
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

"""Registry of ``observations.score_kind`` labels (ARCHITECTURE_REDESIGN R3 / P4).

Rule: **a ``score_kind`` names one scale.** Every value stored under a label
must be comparable, by ``<``, with every other value under that label. When a
formula, its field or its pass count changes, the writer must move to a NEW
label with its own entry here. It must never keep writing under the old one.

Why this exists. Until 2026-10-09 the dashboard multipass writer
(``tools/multipass_all_rivals.py``) wrote under the producer's
``local_validator_replay_rank_sum`` label. Over one day it switched formula
twice (algo2 -> category3 -> sum10) without changing the label, so the
``public_multipass_rivals`` rows under that one name held three scales
(~37k, ~53k, ~120k), on top of the producer's own ~200-scale rows. See
``docs/OBSERVATION_LABELS.md``, which is generated from this file
(``python -m nanobody_opt.observation_labels --markdown``).

Enforcement, at write time (``mutation_network.update_observation_store``
calls :func:`check_observations`):

* Unknown label, or a ``source`` not listed for the label: a warning on
  stderr (once per process). It raises only when ``NOVA_LABEL_ENFORCE=1``.
* ``strict=True`` labels (the multipass labels and every closed legacy label)
  always raise (and log ``OBSERVATION_LABEL_REFUSED`` to
  ``pipeline_failures.jsonl``) :class:`LabelViolation` when a write would mix scales:
  - a legacy/closed label is written by anyone other than its relabel tool;
  - ``provenance.writer`` is not one of the label's ``writers``;
  - ``provenance.formula`` disagrees with the label's ``formula``.

The writer also proves that the *code* still computes the registered formula.
:func:`assert_formula` runs the writer's scoring function on a fixed field
(:data:`GOLDEN_FIELD`) and compares it with the pinned outputs in
:data:`GOLDEN_OUTPUTS`. Suppose someone edits ``sum_of_ten`` to compute
something else but keeps ``formula="sum10"``. The writer then refuses to
start, before it spends any GPU time.

Adding a label: add an entry (with formula, scale, field and passes), then
regenerate the markdown. Changing an existing label's formula/scale is not
allowed; add a new label and close the old one (``writers=frozenset()``).
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# ---------------------------------------------------------------------------
# Formula ids (shared with tools/validate_scoring_change.py ``builtin:<id>``)
# ---------------------------------------------------------------------------
FORMULAS = {
    "sum10": "sum of the ten per-metric dense ranks within the field "
             "(the validator's published final_score; rho=1.000 on our submissions)",
    "category3": "dense-rank each of the 3 metric categories' rank-sums, sum the 3 "
                 "category ranks (briefly live 2026-10-09; rho=0.918, WRONG)",
    "algo2": "Algorithm-2 minmax: drop liability_num_violations>50, then max over "
             "6 weighted metrics of (average rank / weight) (novelty reference only)",
    "validator_published": "the validator's own published final_score / rank-sum",
}


@dataclass(frozen=True)
class Label:
    score_kind: str
    klass: str                    # raw_pass | derived | legacy | validator
    formula: str | None           # key of FORMULAS, or None if mixed/unknown
    scale: str                    # human description incl. typical range
    field: str                    # what the rank is taken against
    passes: int | None            # passes per stored value (1 = per-pass row)
    sources: frozenset[str]       # sources allowed to carry this label
    writers: frozenset[str]       # provenance.writer ids allowed; empty = closed
    strict: bool = False          # violations raise even without NOVA_LABEL_ENFORCE
    typical_range: tuple[float, float] | None = None  # for the scale trip-wire
    introduced: str = ""
    epochs: str = ""              # epochs this label covers, where meaningful
    notes: str = ""
    known_foreign: dict[str, str] = field(default_factory=dict)
    # ^ (source -> explanation) rows that sit under this label today but are
    #   NOT on its scale. Documented, not silently tolerated: readers must
    #   filter by source (see build_top50_index.REPLAY_SCALE_SOURCES).


PRODUCER_REPLAY_SOURCES = frozenset({
    "live_exact_screen", "live_exact_repeat", "local_exact_raw",
    "children_8x", "children_20x", "children_40x", "backfill_8x", "exhaustive_8x",
})

RELABEL_WRITER = "relabel_multipass_legacy"
#: nanobody_opt/hf_field_refresh.py: tops rival / HF top-200 sequences up to a
#: depth of 20 stored passes, one multipass_sum10_pass row per pass.
HF_REFRESH_WRITER = "hf_field_refresh"
#: tools/migrate_hf_field_20x.py: imports master 20x disk passes into
#: multipass_sum10_pass and relabels the hf_field_* means to legacy labels.
HF20X_MIGRATE_WRITER = "migrate_hf_field_20x"
HF20X_SOURCES = frozenset({"public_multipass_rivals", "hf_archive_top200",
                           "motif_variants_20x", "our_top50_20x"})

LABELS: dict[str, Label] = {label.score_kind: label for label in (
    Label(
        score_kind="validator_final_rank_sum",
        klass="validator", formula="validator_published",
        scale="validator's published rank-sum for the epoch; ~13-340; lower=better",
        field="that epoch's real competition field", passes=1,
        sources=frozenset({"public_validator_api", "heldout_prior_public_field"}),
        writers=frozenset({"seed_mutation_database", "select_elite_parents",
                           "audit_live_generator_recall"}),
        typical_range=(1, 600), introduced="<2026-09",
    ),
    Label(
        score_kind="local_validator_replay_rank_sum",
        klass="derived", formula="sum10",
        scale="producer replay rank-sum vs sampled ~49-sequence replay fields "
              "from recent public epochs; ~20-370; lower=better",
        field="replayed epoch fields (~49 sequences)", passes=None,
        sources=PRODUCER_REPLAY_SOURCES,
        writers=frozenset(),  # producer rows predate provenance.writer
        typical_range=(1, 600), introduced="<2026-09",
        notes="Producer scale ONLY. Readers must also filter by source "
              "(build_top50_index.REPLAY_SCALE_SOURCES).",
        known_foreign={
            "hf_archive_top200": "800 rows, ~5.8e4, epoch NULL, no writer in the "
                                 "repo (audit §3a); not on this scale",
            "public_multipass_rivals": "2 rows, epoch 25586 pass1 (49367.5 and "
                                       "53891): a partial multipass run whose "
                                       "formula (algo2 vs category3) cannot be "
                                       "pinned; the relabel refused them, so "
                                       "they stay here, off-scale",
        },
    ),
    Label(
        score_kind="local_exact_validator_replay_rank_sum",
        klass="derived", formula="sum10",
        scale="exact-screen replay rank-sum; ~20-322; lower=better",
        field="replayed epoch fields", passes=None,
        sources=frozenset({"local_exact_raw"}), writers=frozenset(),
        typical_range=(1, 600), introduced="<2026-09",
    ),
    # --- dashboard multipass (source=public_multipass_rivals) ---------------
    Label(
        score_kind="multipass_sum10_pass",
        klass="raw_pass", formula="sum10",
        scale="ONE fast-scorer pass, sum-of-10 dense rank-sum vs the full HF "
              "archive field (~61k rows on 2026-10-09); per pass ~8.5e4-2.1e5, "
              "epoch mean ~1.2e5; lower=better. The CURRENT multipass label.",
        field="hf_archive_full (field_rows + field_sha256 in provenance)", passes=1,
        sources=frozenset({"public_multipass_rivals"}),
        writers=frozenset({"multipass_all_rivals", "backfill_multipass_metrics",
                           RELABEL_WRITER, HF20X_MIGRATE_WRITER, HF_REFRESH_WRITER}),
        strict=True, typical_range=(5.0e4, 2.6e5), introduced="2026-10-09",
        epochs="25533-25534, 25593-25603 (backfilled from disk), 25613-25618 "
               "(relabelled, no metrics), 25619+ (native); 25598-25616 also "
               "hold migrated master 20x passes (ids hf20x_pass:*)",
        notes="Rows written natively or by backfill_multipass_metrics carry raw "
              "metrics_json + field fingerprint. Rows relabelled by "
              "relabel_multipass_legacy (25613-25618) have no metrics and an "
              "unrecorded hf_archive snapshot; same formula and field type. "
              "Rows from migrate_hf_field_20x are the raw per-pass metrics of "
              "master's killed 20x hf_field_refresh runs (same scorer, "
              "validator-dual-sample-v1, read back from master disk); epoch = "
              "the refresh run's epoch, cohort derived per sequence.",
    ),
    Label(
        score_kind="slave_multipass_sum10_pass",
        klass="raw_pass", formula="sum10",
        scale="ONE fast-scorer pass, run on the slave box's idle GPUs "
              "(ssh2.vast.ai), sum-of-10 dense rank-sum vs the full HF archive "
              "field; same scale as multipass_sum10_pass (~8.5e4-2.1e5 per "
              "pass); lower=better.",
        field="hf_archive_full (field_rows + field_sha256 in provenance)", passes=1,
        sources=frozenset({"slave_public_multipass_rivals"}),
        writers=frozenset({"slave_multipass_writeback"}),
        strict=True, typical_range=(5.0e4, 2.6e5), introduced="2026-10-11",
        notes="Scored on the slave box with hf_field_refresh.py's unchanged "
              "score_pass/sum_of_ten harness, over real sequences that reached "
              "an epoch's qualified-for-publication shortlist "
              "(checkpoints/qualified/uid*.json). Slave never opens "
              "nanobody_mutations.sqlite: results are shipped to master as "
              "NDJSON (tools/run_slave_multipass_epoch.sh) and inserted ONLY "
              "by tools/ingest_slave_multipass.py running on master, via the "
              "same update_observation_store() the live producer uses "
              "(NOVA_LOCAL_SQLITE=1). A new label, not a reuse of "
              "multipass_sum10_pass, because the registry keys a label to its "
              "exact writer set (ARCHITECTURE_REDESIGN R3/P4) even when the "
              "formula/field/scale match. `slave_public_multipass_rivals` was "
              "a source placeholder referenced by hf_field_refresh.py "
              "(0 rows, no writer) until this.",
    ),
    Label(
        score_kind="legacy_multipass_category3_pass",
        klass="legacy", formula="category3",
        scale="ONE fast-scorer pass, category3 rank vs the full HF archive; "
              "epoch mean ~5.2e4-5.4e4; lower=better. NOT comparable to sum10.",
        field="/root/hf_archive.csv snapshot at write time (unrecorded)", passes=1,
        sources=frozenset({"public_multipass_rivals"}),
        writers=frozenset({RELABEL_WRITER}),
        strict=True, typical_range=(2.0e4, 1.0e5), introduced="2026-10-09 (relabel)",
        epochs="25536-25538, 25611-25612",
        notes="Written 2026-10-09 14:01Z-17:19Z, while the category3 build "
              "was deployed on the dashboard box. Verified: stored == "
              "category3 recomputed from raw metrics, rho=1.000 (25536-25538 "
              "from metrics_json, 25611-25612 from disk passes).",
    ),
    Label(
        score_kind="legacy_multipass_algo2_pass",
        klass="legacy", formula="algo2",
        scale="ONE fast-scorer pass, Algorithm-2 minmax vs the archive field; "
              "epoch mean ~3.6e4-3.9e4; fractional values; lower=better. "
              "NOT comparable to sum10.",
        field="hf_archive.csv (dashboard box) or the epoch's producer "
              "archive.csv (master); unrecorded", passes=1,
        sources=frozenset({"public_multipass_rivals"}),
        writers=frozenset({RELABEL_WRITER}),
        strict=True, typical_range=(1.0e4, 7.0e4), introduced="2026-10-09 (relabel)",
        epochs="25559-25610 (25593-25602 and part of 25603 since back-filled to sum10)",
        notes="Every multipass write before the category3 deploy at "
              "2026-10-09 14:01:40Z. Verified: stored == algo2 recomputed from "
              "disk passes, rho=1.000 (25594, 25599, 25604, 25610).",
    ),
    # --- master 20x HF-field refresh (off; P3; relabelled 2026-10-10) ------
    Label(
        score_kind="legacy_hf_field_algo2_20x_mean",
        klass="legacy", formula="algo2",
        scale="ONE row per sequence: 20-pass MEAN algo2 vs the full HF archive "
              "+ the scored batch; ~2.65e4-6.05e4; lower=better. NOT comparable "
              "to sum10 or to per-pass rows.",
        field="/root/P20809_nanobodies.csv snapshot at write time (unrecorded)",
        passes=20, sources=HF20X_SOURCES,
        writers=frozenset({HF20X_MIGRATE_WRITER}),
        strict=True, typical_range=(1.0e4, 7.0e4), introduced="2026-10-10 (relabel)",
        epochs="epoch NULL (refresh runs 25598-25615 and earlier imports)",
        notes="Was hf_field_20x_rank_sum: 3,772 rows (2,853 rival, 642 "
              "motif_variants_20x, 275 hf_archive_top200, 2 ours), no "
              "metrics_json. Verified: stored == algo2 recomputed from the 20 "
              "disk passes still on master for 2,114 rival rows (rho 0.999, "
              "median |diff| 1.1%) and 75 HF rows (algo2 rho 0.76, |diff| "
              "2.5%; sum10 |diff| >190%). The other 1,583 rows have no raw "
              "passes anywhere: legacy only. The raw passes that survived "
              "were migrated to multipass_sum10_pass (hf20x_pass:* ids); "
              "provenance.raw_passes_migrated_to says so per row.",
    ),
    Label(
        score_kind="legacy_hf_field_sum10_20x_mean",
        klass="legacy", formula="sum10",
        scale="ONE row per sequence: 20-pass MEAN sum-of-10 vs the full HF "
              "archive + the scored batch; ~1.06e5-1.31e5; lower=better",
        field="/root/P20809_nanobodies.csv snapshot at write time (unrecorded)",
        passes=20, sources=frozenset({"our_top50_20x"}),
        writers=frozenset({HF20X_MIGRATE_WRITER}),
        strict=True, typical_range=(5.0e4, 2.6e5), introduced="2026-10-10 (relabel)",
        epochs="epoch NULL (refresh run for 25616)",
        notes="Was hf_field_sum10_20x_rank_sum: 50 rows, our top-50 only (the "
              "rival batch was killed in pass 4 of 20, before any import). "
              "Verified: stored == sum10 recomputed from the 20 disk passes "
              "(rho 1.000). All 20 raw passes migrated to multipass_sum10_pass "
              "(cohort ours).",
    ),
    Label(
        score_kind="hf_field_20x_rank_sum",
        klass="legacy", formula="algo2",
        scale="CLOSED, 0 rows: relabelled to legacy_hf_field_algo2_20x_mean",
        field="hf_archive_full", passes=20, sources=HF20X_SOURCES,
        writers=frozenset(), strict=True,
        typical_range=(1.0e4, 7.0e4), introduced="2026-10-08",
        notes="Closed 2026-10-10. Never write it again.",
    ),
    Label(
        score_kind="hf_field_sum10_20x_rank_sum",
        klass="legacy", formula="sum10",
        scale="CLOSED, 0 rows: relabelled to legacy_hf_field_sum10_20x_mean",
        field="hf_archive_full", passes=20,
        sources=frozenset({"public_multipass_rivals", "hf_archive_top200",
                           "our_top50_20x"}),
        writers=frozenset(), strict=True,
        typical_range=(5.0e4, 2.6e5), introduced="2026-10-09",
        notes="Closed 2026-10-10. hf_field_refresh (off, P3) wrote it; a "
              "re-enabled refresh is now refused at write time. Its job is "
              "done by multipass_all_rivals' bounded top-up (R1).",
    ),
)}


# ---------------------------------------------------------------------------
# Formula golden test: the code must still compute the registered formula
# ---------------------------------------------------------------------------
GOLDEN_COLUMNS = (
    "design_iiptm", "design_ptm", "design_to_target_iptm",
    "min_design_to_target_pae", "interaction_pae", "plip_hbonds_refolded",
    "plip_saltbridge_refolded", "delta_sasa_refolded", "liability_score",
    "liability_num_violations",
)
GOLDEN_FIELD = (
    (0.80, 0.88, 0.70, 2.2, 10.5, 3, 1, 678.4, 270, 30),
    (0.78, 0.89, 0.69, 2.3, 10.6, 4, 0, 705.5, 255, 28),
    (0.80, 0.85, 0.72, 2.1, 9.9, 3, 2, 650.0, 300, 55),
    (0.75, 0.90, 0.66, 2.6, 11.2, 5, 1, 720.1, 240, 25),
    (0.82, 0.87, 0.71, 2.2, 10.1, 2, 0, 690.0, 270, 31),
    (0.70, 0.80, 0.60, 3.0, 12.0, 6, 3, 600.0, 200, 20),
    (0.79, 0.88, 0.70, 2.4, 10.5, 4, 1, 710.0, 260, 29),
    (0.81, 0.86, 0.73, 2.0, 9.8, 3, 1, 665.0, 310, 60),
)
#: Pinned from tools/validate_scoring_change.py f_sum10/f_category3/f_algo2
#: (2026-10-09). NaN = excluded by the formula (algo2's liability filter).
GOLDEN_OUTPUTS = {
    "sum10": (39.0, 37.0, 41.0, 35.0, 38.0, 47.0, 36.0, 38.0),
    "category3": (13.0, 12.0, 15.0, 10.0, 13.0, 12.0, 11.0, 14.0),
    "algo2": (2.5, 4.0, math.nan, 5.0, 3.0, 6.0, 4.0, math.nan),
}


class LabelViolation(RuntimeError):
    """A write would put a value on the wrong scale under a label."""


def golden_field():
    import pandas as pd  # local: the registry itself must import without pandas
    return pd.DataFrame(list(GOLDEN_FIELD), columns=list(GOLDEN_COLUMNS),
                        index=[f"G{i}" for i in range(len(GOLDEN_FIELD))])


def assert_formula(score_kind: str, formula_id: str, fn: Callable[[Any], Any]) -> None:
    """Refuse to write ``score_kind`` unless ``fn`` really computes its formula.

    ``fn(field_dataframe)`` may return a Series (aligned to the field index,
    missing = excluded) or an array. Raises :class:`LabelViolation` on any
    mismatch: unknown label, label formula != ``formula_id``, or outputs that
    differ from :data:`GOLDEN_OUTPUTS`.
    """
    label = LABELS.get(score_kind)
    if label is None:
        raise LabelViolation(f"score_kind {score_kind!r} is not registered in "
                             "nanobody_opt/observation_labels.py; add a Label first")
    if label.formula != formula_id:
        raise LabelViolation(
            f"score_kind {score_kind!r} is registered as formula "
            f"{label.formula!r}, but the writer computes {formula_id!r}. A new "
            "formula needs a NEW score_kind (and registry entry), never a reuse.")
    expected = GOLDEN_OUTPUTS.get(formula_id)
    if expected is None:
        raise LabelViolation(f"no golden outputs for formula {formula_id!r}")
    frame = golden_field()
    got = fn(frame)
    if hasattr(got, "reindex"):
        got = got.reindex(frame.index)
    got = [float(x) for x in list(got)]
    for want, have in zip(expected, got):
        if math.isnan(want) != math.isnan(have) or (
                not math.isnan(want) and abs(want - have) > 1e-9):
            raise LabelViolation(
                f"writer for {score_kind!r} claims formula {formula_id!r} but its "
                f"scoring function returns {got} on the golden field, expected "
                f"{list(expected)}. The formula changed: register a new label.")


# ---------------------------------------------------------------------------
# Write-time check (called by mutation_network.update_observation_store)
# ---------------------------------------------------------------------------
_WARNED: set[tuple] = set()


def _enforce_all() -> bool:
    return os.environ.get("NOVA_LABEL_ENFORCE", "") == "1"


def _report(kind: str, message: str, context: dict) -> None:
    key = (kind, message)
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"OBSERVATION_LABEL_{kind}: {message}", file=sys.stderr, flush=True)
    if kind != "REFUSED":
        # Warnings stay on stderr: pipeline_failures.jsonl is alerted on by
        # check_mining_liveness, and a warn-mode label must not page anyone.
        return
    try:
        from nanobody_opt.failure_log import record_failure
        record_failure("observation_labels", f"OBSERVATION_LABEL_{kind}",
                       LabelViolation(message), **context)
    except Exception:  # noqa: BLE001 - reporting must never break a write
        pass


def violations(observation: dict[str, Any]) -> list[tuple[bool, str]]:
    """``[(hard, message)]`` for one observation dict (as passed to the store)."""
    kind = str(observation.get("score_kind") or "")
    source = str(observation.get("source") or "")
    label = LABELS.get(kind)
    if label is None:
        return [(_enforce_all(), f"unregistered score_kind {kind!r} "
                                 f"(source={source!r})")]
    out: list[tuple[bool, str]] = []
    hard = label.strict or _enforce_all()
    if source not in label.sources:
        out.append((hard, f"source {source!r} is not registered for "
                          f"score_kind {kind!r} (allowed: {sorted(label.sources)})"))
    writer = observation.get("writer")
    if label.strict:
        if not label.writers:
            out.append((True, f"score_kind {kind!r} is closed (legacy); no new writes"))
        elif writer not in label.writers:
            out.append((True, f"writer {writer!r} may not write {kind!r} "
                              f"(allowed: {sorted(label.writers)})"))
        formula = observation.get("formula")
        if formula != label.formula:
            out.append((True, f"provenance formula {formula!r} != registered "
                              f"{label.formula!r} for {kind!r}"))
    if label.klass == "raw_pass" and writer != RELABEL_WRITER:
        # User, 2026-10-11: "Ensure we always store all raw data so this
        # cannot reoccur." Root cause of the whole 2026-10-11 dashboard
        # rescale effort: pre-2026-10-09 multipass_sum10_pass rows were
        # written with metrics={} (no raw per-pass BoltzGen metrics), so the
        # formula/field could never be recomputed later -- only the original
        # score survived, on whatever scale it was written at. raw_pass is
        # the class that exists SPECIFICALLY so a later formula or field
        # change needs no GPU work (see tools/rescale_multipass_history.py's
        # docstring); an empty metrics dict defeats its entire purpose
        # silently, the same way a dropped NOVA_LOCAL_SQLITE write did.
        # Same severity as the writer/formula checks above (hard under
        # strict/_enforce_all, same as everything else here) -- except for
        # RELABEL_WRITER, which legitimately, documentedly writes metrics-less
        # rows for a handful of already-closed historical epochs (see
        # multipass_sum10_pass's own notes) and must stay able to.
        metrics = observation.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            out.append((hard, f"score_kind {kind!r} is class=raw_pass but "
                              "metrics is empty/missing -- raw_pass rows must "
                              "carry their raw per-pass metrics or the "
                              "formula/field can never be recomputed later"))
    return out


def check_observations(observations: Iterable[dict[str, Any]]) -> None:
    """Warn on every violation; raise :class:`LabelViolation` on hard ones.

    Called once per ``update_observation_store`` batch before anything is
    written, so a refused batch writes nothing.
    """
    hard_messages: list[str] = []
    for observation in observations:
        for hard, message in violations(observation):
            context = {"score_kind": observation.get("score_kind"),
                       "source": observation.get("source")}
            if hard:
                hard_messages.append(message)
            else:
                _report("WARN", message, context)
    if hard_messages:
        unique = sorted(set(hard_messages))
        _report("REFUSED", "; ".join(unique)[:2000], {})
        raise LabelViolation("observation batch refused: " + "; ".join(unique)[:2000])


# ---------------------------------------------------------------------------
# Generated documentation
# ---------------------------------------------------------------------------
def markdown() -> str:
    lines = [
        "# Observation labels (`observations.score_kind`)",
        "",
        "<!-- GENERATED by `python3 -m nanobody_opt.observation_labels --markdown "
        "> docs/OBSERVATION_LABELS.md`. Edit nanobody_opt/observation_labels.py, "
        "not this file. -->",
        "",
        "**Rule: a `score_kind` names one scale.** A new formula, field or pass "
        "count gets a new label; an old label is closed, never re-used. "
        "`source` says who/what population wrote a row; it never changes what "
        "`score` means.",
        "",
        "Enforced at write time by `mutation_network.update_observation_store` "
        "(`check_observations`): strict labels refuse wrong writer / formula / "
        "source (logged as `OBSERVATION_LABEL_REFUSED` in "
        "`pipeline_failures.jsonl`); other labels warn on stderr and refuse with `NOVA_LABEL_ENFORCE=1`. "
        "Writers of strict labels also run `assert_formula` against a pinned "
        "golden field before scoring. `tools/check_observation_labels.py` is the "
        "read-only DB trip-wire (unregistered labels, off-range or jumping "
        "per-epoch medians).",
        "",
        "## Formulas",
        "",
    ]
    for key, text in FORMULAS.items():
        lines.append(f"- `{key}`: {text}")
    lines += ["", "## Labels", ""]
    for label in LABELS.values():
        lines += [
            f"### `{label.score_kind}`",
            "",
            f"- class: {label.klass}{' (strict)' if label.strict else ''}; "
            f"formula: `{label.formula}`; passes per value: {label.passes}",
            f"- scale: {label.scale}",
            f"- field: {label.field}",
            f"- sources: {', '.join(f'`{s}`' for s in sorted(label.sources))}",
            f"- writers: {', '.join(f'`{w}`' for w in sorted(label.writers)) or 'closed / unrecorded'}",
        ]
        if label.typical_range:
            lines.append(f"- trip-wire range: {label.typical_range[0]:g} to "
                         f"{label.typical_range[1]:g}")
        if label.epochs:
            lines.append(f"- epochs: {label.epochs}")
        if label.introduced:
            lines.append(f"- introduced: {label.introduced}")
        if label.notes:
            lines.append(f"- notes: {label.notes}")
        for source, why in label.known_foreign.items():
            lines.append(f"- KNOWN OFF-SCALE rows from `{source}`: {why}")
        lines.append("")
    lines += [
        "## `public_multipass_rivals` history (2026-10-09 relabel)",
        "",
        "Until 2026-10-09 every dashboard multipass pass row was written as "
        "`local_validator_replay_rank_sum`. The formula behind a row depends "
        "on when the row was last written (an upsert on "
        "`public_multipass:epoch<E>:<seq>:pass<k>`), not on the epoch number. "
        "Box deploy times are taken from the `multipass_all_rivals.py.bak.*` "
        "backups on the dashboard box. Each regime was verified by recomputing "
        "all three formulas from raw metrics.",
        "",
        "| Epochs (DB state after relabel) | Formula | Written (UTC, 2026-10-09) | Now labelled |",
        "|---|---|---|---|",
        "| 25559-25610 (except back-filled) | algo2 | before 14:01:40Z (old build; 25610 run started 13:37Z) | `legacy_multipass_algo2_pass` |",
        "| 25536-25538, 25611-25612 | category3 | 14:01:40Z-17:19:41Z (category3 build live) | `legacy_multipass_category3_pass` |",
        "| 25613-25618 | sum10 | after 17:25:15Z (sum10 build) | `multipass_sum10_pass` (no metrics) |",
        "| 25533-25534, 25593-25603 | sum10 (re-derived from disk passes) | backfill 2026-10-09/10 | `multipass_sum10_pass` (with metrics; original score/kind in provenance) |",
        "| 25619+ | sum10 | native | `multipass_sum10_pass` |",
        "| 25586 (2 rows) | algo2 or category3, undeterminable | unknown | left as `local_validator_replay_rank_sum` (refused by the fingerprint guard) |",
        "",
        "The relabel (`tools/relabel_multipass_legacy.py`, applied 2026-10-10 "
        "00:5xZ, 24,398 rows) only changes `score_kind` and adds provenance "
        "(`relabeled_from`, `formula`, `formula_evidence`); scores are untouched.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    if "--markdown" in sys.argv:
        sys.stdout.write(markdown())
    else:
        print(json.dumps({k: {"formula": v.formula, "scale": v.scale,
                              "sources": sorted(v.sources)}
                          for k, v in LABELS.items()}, indent=1))

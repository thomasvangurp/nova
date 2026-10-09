#!/usr/bin/env python3
"""Install production-exact compound filtering into an existing live tree."""
from __future__ import annotations

import argparse
import errno
import os
import shutil
import time
from pathlib import Path


PORTFOLIO_FINALIZE = r'''def _calibrate_reduced_to_full(args, reduced_rows):
    """Robust ridge calibration from historical paired reduced/full scores."""
    reduced_history = {}
    if args.accumulator and args.accumulator.is_file():
        try:
            reduced_history = {
                str(row["name"]): float(row["score"])
                for row in json.loads(args.accumulator.read_text())
                if row.get("name") is not None and row.get("score") is not None
            }
        except (OSError, ValueError, TypeError):
            reduced_history = {}
    with sqlite3.connect(f"file:{args.bank.resolve()}?mode=ro", uri=True, timeout=120) as connection:
        full = {
            str(row[0]): (float(row[1]), int(row[2]))
            for row in connection.execute(
                "SELECT ss.name,ss.mean_score,ss.observation_count FROM score_summaries ss "
                "JOIN candidates ca ON ca.name=ss.name WHERE ca.reaction=?",
                (args.reaction,),
            )
        }
    paired = [
        (value, full[name][0], min(3, max(1, full[name][1])))
        for name, value in reduced_history.items() if name in full
        and math.isfinite(value) and math.isfinite(full[name][0])
    ]
    if paired:
        x = np.asarray([row[0] for row in paired], dtype=np.float64)
        y = np.asarray([row[1] for row in paired], dtype=np.float64)
        weights = np.asarray([row[2] for row in paired], dtype=np.float64)
    else:
        x = y = weights = np.asarray([], dtype=np.float64)
    if len(paired) >= 8 and float(np.std(x)) > 1e-6:
        design = np.column_stack([np.ones(len(x)), x])
        root_weight = np.sqrt(weights)
        weighted_design = design * root_weight[:, None]
        weighted_y = y * root_weight
        ridge = np.diag([0.05, 0.50])
        prior = np.asarray([0.0, 1.0])
        beta = np.linalg.solve(
            weighted_design.T @ weighted_design + ridge,
            weighted_design.T @ weighted_y + ridge @ prior,
        )
        intercept = float(beta[0])
        slope = float(np.clip(beta[1], 0.10, 2.50))
    else:
        slope = 1.0
        intercept = float(np.median(y - x)) if len(paired) else 0.0
    residuals = y - (intercept + slope * x) if len(paired) else np.asarray([])
    if len(residuals):
        # Admission operates in the extreme reduced-score tail, where the
        # global fit is optimistic (selection/winner's curse) and noisier.
        # Calibrate on that tail, and use RMSE rather than MAD: MAD hid rare
        # but decisive 0.02-0.05 reduced->full misses in the first replay.
        frontier = x >= float(np.quantile(x, 0.90)) if len(x) >= 20 else np.ones(len(x), dtype=bool)
        tail = residuals[frontier]
        centre = float(np.mean(tail)) if len(tail) else float(np.mean(residuals))
        intercept += centre
        centred = tail - centre if len(tail) else residuals - centre
        rmse = float(np.sqrt(np.mean(centred * centred))) if len(centred) else 0.020
        posterior_sd = float(np.clip(max(0.010, rmse), 0.010, 0.035))
    else:
        posterior_sd = 0.020
    predictions = {
        str(row["name"]): float(np.clip(
            intercept + slope * float(row["score"]), -0.20, 0.25,
        ))
        for row in reduced_rows
    }
    return predictions, posterior_sd, {
        "paired": len(paired), "intercept": round(intercept, 6),
        "slope": round(slope, 6), "posterior_sd": round(posterior_sd, 6),
        "frontier_bias": round(centre, 6) if len(residuals) else None,
    }


def _production_incumbents(args):
    with sqlite3.connect(f"file:{args.bank.resolve()}?mode=ro", uri=True, timeout=120) as connection:
        rows = [dict(
            name=row[0], smiles=row[1], reaction=row[2], components=row[3],
            inchikey=row[4], heavy_atoms=row[5], rotatable_bonds=row[6],
            proxy_mean=row[7], proxy_std=row[8], score=float(row[9]),
            passes=int(row[10]),
        ) for row in connection.execute(
            "SELECT ca.name,ca.smiles,ca.reaction,ca.components,ca.inchikey,"
            "ca.heavy_atoms,ca.rotatable_bonds,ca.proxy_mean,ca.proxy_std,"
            "ss.mean_score,ss.observation_count FROM score_summaries ss "
            "JOIN candidates ca ON ca.name=ss.name WHERE ca.reaction=? "
            "AND ss.observation_count>=3 ORDER BY ss.mean_score DESC",
            (args.reaction,),
        )]
    eligibility = ReleaseEligibility(
        args.historical_molecules, args.public_history, args.submitted_archive,
    )
    eligible, stats = eligibility.filter_rows(rows)
    return eligible, eligibility, stats


def _resolve_portfolio_context(args, source):
    """Keep v11 compatible with finalizers launched by an in-flight v10 shell."""
    # portfolio_context_v11: these paths are explicit in every new invocation,
    # but an already-running shell may reach finalize after the live file swap.
    art = args.acquisition.resolve().parent
    if args.bank is None:
        args.bank = art / "rxn3-bank-live.sqlite"
    if args.reaction is None:
        args.reaction = int(source.get("reaction"))
    if args.historical_molecules is None:
        args.historical_molecules = art / "P40261_molecules.csv"
    if args.public_history is None:
        args.public_history = art / "public-history.csv"
    if args.submitted_archive is None:
        args.submitted_archive = art.parent / "submitted_archive"
    required = (args.bank, args.historical_molecules, args.public_history)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("portfolio context is incomplete: " + ", ".join(missing))
    if not args.submitted_archive.is_dir():
        raise FileNotFoundError(
            f"submitted archive is unavailable: {args.submitted_archive}"
        )


def finalize_ranked(args, state, source, source_rows, by_name, reduced_rows) -> int:
    _resolve_portfolio_context(args, source)
    # Persist every reduced observation first: calibration learns across rounds,
    # including cohorts whose current portfolio value is zero.
    reduced_score = {row["name"]: float(row["score"]) for row in reduced_rows}
    update_accumulator(args.accumulator, reduced_rows, POLICY_VERSION)
    predictions, posterior_sd, calibration = _calibrate_reduced_to_full(
        args, reduced_rows,
    )
    incumbents, eligibility, incumbent_filter = _production_incumbents(args)

    candidate_rows = []
    for row in reduced_rows:
        name = str(row["name"])
        if name not in by_name:
            continue
        value = dict(by_name[name])
        value["predicted_full_score"] = predictions[name]
        value["posterior_sd"] = posterior_sd
        candidate_rows.append(value)
    candidate_rows, candidate_filter = eligibility.filter_rows(candidate_rows)
    simulation = portfolio_thompson_acquisition(
        incumbents, candidate_rows, draws=args.portfolio_draws,
        seed=args.portfolio_seed, minimum_entropy=args.minimum_entropy,
    )
    metrics = {row["name"]: row for row in simulation["metrics"]}
    ranked = [
        row for row in simulation["metrics"]
        if row["portfolio_entry_probability"] >= args.portfolio_min_entry_probability
        and row["portfolio_expected_delta"] > 0
    ]
    full_cap = min(max(1, int(args.full_cap)), POLICY_FULL_CAP)
    selected_metrics = ranked[:full_cap]
    selected_names = {row["name"] for row in selected_metrics}

    control_budget = min(
        args.portfolio_max_controls,
        max(0, math.ceil(len(selected_metrics) * args.portfolio_control_fraction)),
        max(0, full_cap - len(selected_metrics)),
    )
    controls = sorted(
        (row for row in simulation["metrics"] if row["name"] not in selected_names),
        key=lambda row: (
            row["portfolio_selection_probability"] * row["posterior_sd"],
            row["portfolio_entry_probability"], row["predicted_full_score"],
        ), reverse=True,
    )[:control_budget]
    control_names = {row["name"] for row in controls}
    chosen_metrics = selected_metrics + controls
    chosen_names = {row["name"] for row in chosen_metrics}
    gate = {
        "passed": bool(selected_metrics),
        "admission_policy": "production_portfolio_thompson",
        "portfolio_selected": len(selected_metrics),
        "calibration_controls": len(controls),
        "candidate_count": len(candidate_rows),
        "legacy_threshold": args.gate_threshold,
        "above_legacy_threshold": sum(
            float(row["score"]) >= args.gate_threshold for row in reduced_rows
        ) if args.gate_threshold is not None else None,
        "minimum_entry_probability": args.portfolio_min_entry_probability,
    }
    if not selected_metrics:
        source["candidates"] = []
        source["expected_count"] = 0
        source["reduced_primary"] = {
            "policy_version": POLICY_VERSION,
            "source_count": len(source_rows), "gate": gate,
            "full_fidelity_count": 0, "calibration": calibration,
            "incumbent_filter": incumbent_filter,
            "candidate_filter": candidate_filter,
            "portfolio_thompson": {key: value for key, value in simulation.items()
                                   if key != "metrics"},
        }
        atomic_json(args.acquisition, source)
        print(json.dumps({
            "event": "portfolio_thompson_rejected", "policy_version": POLICY_VERSION,
            **gate, "baseline_score": simulation["baseline_score"],
            "improved_draw_probability": simulation["improved_draw_probability"],
        }))
        return GATE_REJECTED

    final = []
    for metric in chosen_metrics:
        name = metric["name"]
        reason = ("portfolio_thompson" if name in selected_names
                  else "portfolio_calibration")
        value = annotate(
            by_name[name], reason, state["cpu_rank"][name], reduced_score[name],
        )
        value.update({
            "portfolio_entry_probability": metric["portfolio_entry_probability"],
            "portfolio_selection_probability": metric["portfolio_selection_probability"],
            "portfolio_expected_delta": metric["portfolio_expected_delta"],
            "portfolio_predicted_full_score": metric["predicted_full_score"],
            "portfolio_posterior_sd": metric["posterior_sd"],
        })
        final.append(value)

    lane_selected_counts = {}
    field_arm_selected_counts = {}
    for row in final:
        lane = str(row.get("lane") or "unassigned")
        lane_selected_counts[lane] = lane_selected_counts.get(lane, 0) + 1
        if lane.split(":", 1)[0] == "block_bandit":
            arm = str(row.get("arm") or "unassigned")
            field_arm_selected_counts[arm] = field_arm_selected_counts.get(arm, 0) + 1
    source["candidates"] = final
    source["expected_count"] = len(final)
    source["reduced_primary"] = {
        "policy_version": POLICY_VERSION,
        "source_count": len(source_rows), "gate": gate,
        "full_fidelity_count": len(final),
        "portfolio_selected": len(selected_metrics),
        "calibration_controls": len(controls),
        "calibration": calibration,
        "incumbent_filter": incumbent_filter,
        "candidate_filter": candidate_filter,
        "portfolio_thompson": {key: value for key, value in simulation.items()
                               if key != "metrics"},
        "selected_metrics": chosen_metrics,
        "lane_selected_counts": lane_selected_counts,
        "field_arm_selected_counts": field_arm_selected_counts,
        "reduced_score_range": [min(reduced_score.values()), max(reduced_score.values())],
    }
    atomic_json(args.acquisition, source)
    print(json.dumps({"event": "portfolio_thompson_ranked", **source["reduced_primary"]}))
    return 0
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def write(path: Path, text: str, *, backup: bool = True) -> None:
    if path.is_file() and path.read_text() == text:
        return
    if path.is_file() and backup:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        shutil.copy2(path, path.with_name(f"{path.name}.pre-release-aware-{stamp}.bak"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.release-aware.tmp")
    temporary.write_text(text)
    if path.is_file():
        os.chmod(temporary, path.stat().st_mode)
    try:
        os.replace(temporary, path)
    except OSError as error:
        if error.errno != errno.ESTALE:
            raise
        with path.open("w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.unlink(missing_ok=True)


def patch_reduced(path: Path) -> None:
    text = path.read_text()
    if 'RELEASE_AWARE_POLICY_VERSION = "reduced-primary-v10-release-aware"' not in text:
        text = replace_once(
            text,
            'TRUE_ADMISSION_POLICY_VERSION = "reduced-primary-v8-true-admission"\n'
            'POLICY_VERSION = "reduced-primary-v9-proportional-controls"',
            'TRUE_ADMISSION_POLICY_VERSION = "reduced-primary-v8-true-admission"\n'
            'PROPORTIONAL_POLICY_VERSION = "reduced-primary-v9-proportional-controls"\n'
            'RELEASE_AWARE_POLICY_VERSION = "reduced-primary-v10-release-aware"\n'
            'POLICY_VERSION = RELEASE_AWARE_POLICY_VERSION',
            "v10 policy version",
        )
        text = replace_once(
            text,
            '        TRUE_ADMISSION_POLICY_VERSION,\n        POLICY_VERSION,',
            '        TRUE_ADMISSION_POLICY_VERSION, PROPORTIONAL_POLICY_VERSION,\n'
            '        RELEASE_AWARE_POLICY_VERSION, POLICY_VERSION,',
            "v9 state compatibility",
        )
    if 'PORTFOLIO_POLICY_VERSION = "reduced-primary-v11-portfolio-thompson"' not in text:
        text = replace_once(
            text,
            'RELEASE_AWARE_POLICY_VERSION = "reduced-primary-v10-release-aware"\n'
            'POLICY_VERSION = RELEASE_AWARE_POLICY_VERSION',
            'RELEASE_AWARE_POLICY_VERSION = "reduced-primary-v10-release-aware"\n'
            'PORTFOLIO_POLICY_VERSION = "reduced-primary-v11-portfolio-thompson"\n'
            'POLICY_VERSION = PORTFOLIO_POLICY_VERSION',
            "v11 policy version",
        )
        text = replace_once(
            text,
            '        RELEASE_AWARE_POLICY_VERSION, POLICY_VERSION,',
            '        RELEASE_AWARE_POLICY_VERSION, PORTFOLIO_POLICY_VERSION, POLICY_VERSION,',
            "v10 state compatibility",
        )
    if "import numpy as np" not in text:
        text = replace_once(
            text,
            "import time\nfrom pathlib import Path\n",
            "import time\nimport sqlite3\nfrom pathlib import Path\n\nimport numpy as np\n",
            "portfolio dependencies",
        )
    if "portfolio_thompson_acquisition" not in text:
        text = replace_once(
            text,
            "from compound_miner.types import Candidate, ScoredCandidate\n",
            "from compound_miner.types import Candidate, ScoredCandidate\n"
            "from tools.release_eligibility import (\n"
            "    ReleaseEligibility, portfolio_thompson_acquisition,\n"
            ")\n",
            "portfolio imports",
        )
    start = text.index("def finalize_ranked(")
    end = text.index("\n\ndef finalize(", start)
    if ("production_portfolio_thompson" not in text[start:end]
            or "portfolio_context_v11" not in text[start:end]):
        text = text[:start] + PORTFOLIO_FINALIZE.rstrip() + text[end:]
    if "--portfolio-draws" not in text:
        marker = '    finish.add_argument("--allow-partial-reduced", action="store_true")\n'
        additions = '''    finish.add_argument("--bank", type=Path)
    finish.add_argument("--reaction", type=int)
    finish.add_argument("--historical-molecules", type=Path)
    finish.add_argument("--public-history", type=Path)
    finish.add_argument("--submitted-archive", type=Path)
    finish.add_argument("--portfolio-draws", type=int, default=128)
    finish.add_argument("--portfolio-seed", type=int, default=0)
    finish.add_argument("--portfolio-min-entry-probability", type=float, default=0.05)
    finish.add_argument("--portfolio-control-fraction", type=float, default=0.05)
    finish.add_argument("--portfolio-max-controls", type=int, default=4)
'''
        text = replace_once(text, marker, marker + additions, "portfolio arguments")
    # v11 was first installed while some v10 rank shells were in flight.  Make
    # the contextual arguments optional so those processes can infer the same
    # canonical paths and complete instead of discarding their reduced cohort.
    text = text.replace(
        'finish.add_argument("--bank", type=Path, required=True)',
        'finish.add_argument("--bank", type=Path)',
    ).replace(
        'finish.add_argument("--reaction", type=int, required=True)',
        'finish.add_argument("--reaction", type=int)',
    ).replace(
        'finish.add_argument("--historical-molecules", type=Path, required=True)',
        'finish.add_argument("--historical-molecules", type=Path)',
    ).replace(
        'finish.add_argument("--public-history", type=Path, required=True)',
        'finish.add_argument("--public-history", type=Path)',
    ).replace(
        'finish.add_argument("--submitted-archive", type=Path, required=True)',
        'finish.add_argument("--submitted-archive", type=Path)',
    )
    text = text.replace(
        "default=POLICY_DISAGREEMENT_SEATS",
        "default=POLICY_MAX_CONTROL_SEATS",
    ).replace(
        "default=POLICY_UNIFORM_INFORMATION_SEATS",
        "default=POLICY_MAX_UNIFORM_INFORMATION_SEATS",
    )
    write(path, text)


def patch_rank(path: Path) -> None:
    text = path.read_text()
    if "production_exact_prefilter" not in text:
        marker = '''if ! "${run[@]}" tools/reduced_rank_acquisition.py prune \\
    --acquisition "$acquisition" --accumulator "$accumulator" >/dev/null; then
'''
        insert = '''# production_exact_prefilter: refresh and enforce every cheap release rule
# before reduced Boltz sees the molecule.  The scorer repeats this fail-closed
# check immediately before each NIM request because the archive can grow while
# a reduced cohort is running.
if ! "${run[@]}" tools/refresh_public_data.py \\
    --history-output "$public_history" --novelty-output "$historical" >/dev/null; then
  echo '{"event":"reduced_primary_retry","reason":"release_archive_refresh_failed"}'
  exit 75
fi
if ! "${run[@]}" tools/release_eligibility.py filter-manifest \\
    --acquisition "$acquisition" --historical-molecules "$historical" \\
    --public-history "$public_history" \\
    --submitted-archive "$(dirname "$art")/submitted_archive" >/dev/null; then
  echo '{"event":"reduced_primary_retry","reason":"release_eligibility_empty_or_failed"}'
  exit 75
fi

'''
        text = replace_once(text, marker, insert + marker, "rank prefilter")
    if "--portfolio-draws" not in text:
        text = replace_once(
            text,
            '''  --uniform-information-seats 2 "${partial_args[@]}" \\
  --minimum-entropy 0.25 --accumulator "$accumulator" 2>&1 | tail -1
''',
            '''  --uniform-information-seats 2 "${partial_args[@]}" \\
  --minimum-entropy 0.25 --accumulator "$accumulator" \\
  --bank "$bank" --reaction "$reaction" \\
  --historical-molecules "$historical" --public-history "$public_history" \\
  --submitted-archive "$(dirname "$art")/submitted_archive" \\
  --portfolio-draws 128 --portfolio-seed 0 \\
  --portfolio-min-entry-probability 0.05 \\
  --portfolio-control-fraction 0.05 --portfolio-max-controls 4 2>&1 | tail -1
''',
            "portfolio finalize invocation",
        )
    write(path, text)


def patch_scorer(path: Path) -> None:
    text = path.read_text()
    if "production-exact-v1" not in text:
        start = text.index("def _novelty_gate(")
        end = text.index("\n\ndef score_manifest", start)
        replacement = '''def _novelty_gate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fail-closed production gate immediately before any NIM request."""
    archive = args.historical_molecules or args.acquisition.resolve().parent / "P40261_molecules.csv"
    if args.no_novelty_gate:
        raise RuntimeError("production scoring cannot disable release eligibility")
    try:
        from tools.release_eligibility import ReleaseEligibility
        eligibility = ReleaseEligibility(
            archive,
            archive.parent / "public-history.csv",
            archive.parent.parent / "submitted_archive",
        )
        kept, stats = eligibility.filter_rows(rows)
    except Exception as error:
        raise RuntimeError(
            f"release eligibility prefilter failed closed: {type(error).__name__}: {error}"
        ) from error
    if not kept:
        raise SystemExit(json.dumps({
            "event": "no_release_eligible_candidates",
            "release_eligibility": stats,
        }))
    return kept, {
        "applied": True,
        "policy": "production-exact-v1",
        "archive": str(archive),
        **stats,
    }
'''
        text = text[:start] + replacement + text[end:]
    if '"portfolio_entry_probability",' not in text:
        marker = '''                    "reduced_primary_cpu_rank",
'''
        fields = '''                    "portfolio_entry_probability",
                    "portfolio_selection_probability",
                    "portfolio_expected_delta",
                    "portfolio_predicted_full_score",
                    "portfolio_posterior_sd",
'''
        text = replace_once(text, marker, marker + fields, "portfolio scorer provenance")
    write(path, text)


def patch_confirm(path: Path) -> None:
    text = path.read_text()
    if "ReleaseEligibility" not in text:
        text = replace_once(
            text,
            "from compound_miner.types import Candidate  # noqa: E402\n",
            "from compound_miner.types import Candidate  # noqa: E402\n"
            "from tools.release_eligibility import ReleaseEligibility  # noqa: E402\n",
            "confirm release import",
        )
    text = text.replace(
        "from tools.release_eligibility import ReleaseEligibility  # noqa: E402",
        "from tools.release_eligibility import (  # noqa: E402\n"
        "    ReleaseEligibility, select_entropy_constrained,\n"
        ")",
        1,
    )
    old = '''    history = HistoricalIndex(args.historical_molecules)
    gate = ValidityGate(history)
    cache = default_cache(history)
'''
    new = '''    eligibility = ReleaseEligibility(
        args.historical_molecules,
        args.historical_molecules.parent / "public-history.csv",
        args.historical_molecules.parent.parent / "submitted_archive",
    )
'''
    if old in text:
        text = replace_once(text, old, new, "confirm eligibility context")
    old = '''            candidates = [_candidate(row) for row in rows]
            _, kept, _ = prefilter(candidates, history, cache)
            stop_chunk = False
'''
    new = '''            candidates = [_candidate(row) for row in rows]
            _, kept, _ = eligibility.filter_candidates(candidates)
            stop_chunk = False
'''
    if old in text:
        text = replace_once(text, old, new, "confirm early release filter")
    text = text.replace(
        '''                if not gate.accepts(candidate, check_similarity=False):
                    continue
''',
        "",
        1,
    )
    if "production_entries" not in text:
        text = replace_once(
            text,
            '''            still = threshold is None or mean_after >= threshold
            survived += passes_after >= GATE_PASSES and still
            molecules.append({
''',
            '''            still = threshold is None or mean_after >= threshold
            molecules.append({
''',
            "remove raw survived counter",
        )
        text = replace_once(
            text,
            '''                "entry_probability": round(float(item["entry_probability"]), 4),
                "survived": bool(still),
            })
    return {
''',
            '''                "entry_probability": round(float(item["entry_probability"]), 4),
                "above_raw_threshold": bool(still),
            })

        confirmed_rows = [dict(
            name=row[0], smiles=row[1], reaction=row[2], components=row[3],
            inchikey=row[4], heavy_atoms=row[5], rotatable_bonds=row[6],
            proxy_mean=row[7], proxy_std=row[8], score=float(row[9]), passes=int(row[10]),
        ) for row in connection.execute(
            "SELECT ca.name,ca.smiles,ca.reaction,ca.components,ca.inchikey,"
            "ca.heavy_atoms,ca.rotatable_bonds,ca.proxy_mean,ca.proxy_std,"
            "ss.mean_score,ss.observation_count FROM score_summaries ss "
            "JOIN candidates ca ON ca.name=ss.name WHERE ca.reaction=? "
            "AND ss.observation_count>=3 ORDER BY ss.mean_score DESC",
            (selection["reaction"],),
        )]
    eligibility = ReleaseEligibility(
        args.bank.parent / "P40261_molecules.csv",
        args.bank.parent / "public-history.csv",
        args.bank.parent.parent / "submitted_archive",
    )
    eligible_rows, release_stats = eligibility.filter_rows(confirmed_rows)
    try:
        production_portfolio, production_entropy = select_entropy_constrained(eligible_rows)
    except ValueError:
        production_portfolio, production_entropy = [], None
    production_names = {row["name"] for row in production_portfolio}
    for molecule in molecules:
        molecule["production_entry"] = molecule["name"] in production_names
        molecule["survived"] = molecule["production_entry"]
    survived = sum(molecule["production_entry"] for molecule in molecules)
    return {
''',
            "production confirmation accounting",
        )
        text = replace_once(
            text,
            '''        "survived_threshold": survived,
        "molecules": molecules,
''',
            '''        "survived_threshold": sum(
            molecule["above_raw_threshold"] for molecule in molecules
        ),
        "production_entries": survived,
        "production_selected_score": (
            round(sum(float(row["score"]) for row in production_portfolio), 6)
            if production_portfolio else None
        ),
        "production_entropy": (
            round(production_entropy, 6) if production_entropy is not None else None
        ),
        "production_eligible_pool": len(eligible_rows),
        "release_filter": release_stats,
        "molecules": molecules,
''',
            "production confirmation summary",
        )
    write(path, text)


def patch_lane_performance(path: Path) -> None:
    text = path.read_text()
    if "production_entry_names" in text:
        return
    text = replace_once(
        text,
        "from pathlib import Path\n",
        "from pathlib import Path\n\n"
        "from release_eligibility import ReleaseEligibility, select_entropy_constrained\n",
        "lane release import",
    )
    old = '''    confirmed = [float(row[0]) for row in connection.execute(
        "SELECT ss.mean_score FROM score_summaries ss JOIN candidates ca ON ca.name=ss.name "
        "WHERE ca.reaction=? AND ss.observation_count>=3 ORDER BY ss.mean_score DESC",
        (args.reaction,),
    )]
entry = confirmed[19] if len(confirmed) >= 20 else None
'''
    new = '''    confirmed_rows = [dict(
        name=row[0], smiles=row[1], reaction=row[2], components=row[3],
        inchikey=row[4], heavy_atoms=row[5], rotatable_bonds=row[6],
        proxy_mean=row[7], proxy_std=row[8], score=float(row[9]), passes=int(row[10]),
    ) for row in connection.execute(
        "SELECT ca.name,ca.smiles,ca.reaction,ca.components,ca.inchikey,"
        "ca.heavy_atoms,ca.rotatable_bonds,ca.proxy_mean,ca.proxy_std,"
        "ss.mean_score,ss.observation_count FROM score_summaries ss "
        "JOIN candidates ca ON ca.name=ss.name WHERE ca.reaction=? "
        "AND ss.observation_count>=3 ORDER BY ss.mean_score DESC",
        (args.reaction,),
    )]
eligibility = ReleaseEligibility(
    args.art / "P40261_molecules.csv",
    args.art / "public-history.csv",
    args.art.parent / "submitted_archive",
)
eligible_rows, release_stats = eligibility.filter_rows(confirmed_rows)
try:
    production_portfolio, production_entropy = select_entropy_constrained(eligible_rows)
except ValueError:
    production_portfolio, production_entropy = [], None
production_entry_names = {row["name"] for row in production_portfolio}
entry = min((float(row["score"]) for row in production_portfolio), default=None)
'''
    text = replace_once(text, old, new, "lane production portfolio")
    text = text.replace(
        '"ge_entry": sum(value >= entry for value in values) if entry is not None else None,',
        '"ge_entry": sum(name in production_entry_names for name in names),',
    )
    old_rate = '''            round(100 * sum(value >= entry for value in values) / len(values), 3)
            if values and entry is not None else None
'''
    new_rate = '''            round(100 * sum(name in production_entry_names for name in names) / len(values), 3)
            if values else None
'''
    text = replace_once(text, old_rate, new_rate, "lane entry rate")
    text = replace_once(
        text,
        '    "entry_score": round(entry, 6) if entry is not None else None,\n',
        '    "entry_score": round(entry, 6) if entry is not None else None,\n'
        '    "production_selected_score": round(sum(float(row["score"]) for row in production_portfolio), 6) if production_portfolio else None,\n'
        '    "production_entropy": round(production_entropy, 6) if production_entropy is not None else None,\n'
        '    "production_eligible_pool": len(eligible_rows),\n'
        '    "release_filter": release_stats,\n',
        "lane output metrics",
    )
    write(path, text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--module-source", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    write(root / "tools/release_eligibility.py", args.module_source.read_text(), backup=False)
    patch_reduced(root / "tools/reduced_rank_acquisition.py")
    patch_rank(root / "tools/rank_primary_acquisition.sh")
    patch_scorer(root / "tools/score_rxn3_acquisition.py")
    patch_confirm(root / "tools/confirm_hits.py")
    patch_lane_performance(root / "tools/lane_performance.py")
    print({"event": "release_aware_policy_installed", "policy": "production-exact-v1"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

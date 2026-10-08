#!/usr/bin/env python3
"""Apply the information-efficient promotion policy to one live search tree.

The live reaction boxes intentionally carry reaction-specific loop scripts, so
deploying one box's whole file to every sibling would erase their tuned search
geometry.  This migrator changes only the common promotion/confirmation
functions and refuses if the expected boundaries are absent.
"""
from __future__ import annotations

import argparse
import errno
import os
import shutil
import time
from pathlib import Path


LOOP = "tools/run_continuous_multi_reaction_event_motif_bandit_boxnative.sh"
REDUCED = "tools/reduced_rank_acquisition.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def backup_and_write(path: Path, text: str) -> None:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = path.with_name(f"{path.name}.pre-infoefficient-{stamp}.bak")
    shutil.copy2(path, backup)
    temporary = path.with_name(f".{path.name}.infoefficient.tmp")
    temporary.write_text(text)
    os.chmod(temporary, path.stat().st_mode)
    try:
        os.replace(temporary, path)
    except OSError as exc:
        if exc.errno != errno.ESTALE:
            raise
        # Some Vast volumes briefly reject rename(2) with ESTALE.  The loop is
        # stopped before using this fallback, so an fsynced in-place update is
        # preferable to leaving RXN3 on the old policy indefinitely.
        with path.open("w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, temporary.stat().st_mode)
        temporary.unlink(missing_ok=True)


def patch_reduced(path: Path) -> bool:
    text = path.read_text()
    if "reduced-primary-v5-information" in text:
        return False
    text = replace_once(
        text,
        'POLICY_VERSION = "reduced-primary-v4-breadth"',
        'BREADTH_POLICY_VERSION = "reduced-primary-v4-breadth"\n'
        'POLICY_VERSION = "reduced-primary-v5-information"',
        "policy version",
    )
    text = replace_once(
        text,
        "        LEGACY_POLICY_VERSION, GATED_POLICY_VERSION, RANKED_POLICY_VERSION,\n"
        "        POLICY_VERSION,\n",
        "        LEGACY_POLICY_VERSION, GATED_POLICY_VERSION, RANKED_POLICY_VERSION,\n"
        "        BREADTH_POLICY_VERSION, POLICY_VERSION,\n",
        "policy compatibility",
    )
    rejection = '''    if not gate["passed"]:
        source["candidates"] = []
        source["expected_count"] = 0
        source["reduced_primary"] = {
            "policy_version": POLICY_VERSION,
            "source_count": len(source_rows),
            "gate": gate,
            "full_fidelity_count": 0,
        }
        atomic_json(args.acquisition, source)
        print(json.dumps({
            "event": "reduced_primary_gate_rejected", "policy_version": POLICY_VERSION,
            **gate,
        }))
        return GATE_REJECTED

'''
    text = replace_once(
        text,
        rejection,
        '    # Reduced Boltz ranks the cohort but never vetoes chemistry.  Its\n'
        '    # measured frontier correlation is too weak for a hard admission\n'
        '    # boundary; the threshold remains as an online diagnostic.\n'
        '    gate["admission_policy"] = "rank_only_no_veto"\n'
        '    gate["would_have_rejected"] = not gate["passed"]\n\n',
        "hard cohort veto",
    )
    marker = '''    block_bandit = [
        item for item in ranked
        if by_name[item.candidate.name].get("lane") == "block_bandit"
    ]
'''
    informative = '''    # Protected falsification seats.  Positive disagreement means the
    # proxy ranked a molecule much higher than reduced Boltz did: exactly the
    # false-negative region a rank-only reduced model would otherwise hide.
    cpu_rank = state["cpu_rank"]
    reduced_rank = {item.candidate.name: index for index, item in enumerate(ranked)}
    disagreement = sorted(
        ranked,
        key=lambda item: (
            reduced_rank[item.candidate.name] - cpu_rank[item.candidate.name],
            item.score,
        ),
        reverse=True,
    )
    uniform = [
        item for item in ranked
        if by_name[item.candidate.name].get("arm") == "random"
    ]
    disagreement_take = min(24, target)
    extend(disagreement, disagreement_take)
    disagreement_names = set(chosen_names)
    uniform_take = min(16, target - len(chosen))
    extend(uniform, uniform_take)
    uniform_names = set(chosen_names) - disagreement_names

    block_bandit = [
        item for item in ranked
        if by_name[item.candidate.name].get("lane") == "block_bandit"
    ]
'''
    text = replace_once(text, marker, informative, "informative full-score seats")
    text = replace_once(
        text,
        '''    extend(block_random, random_take)
    extend(block_bandit, block_take - len(chosen))
''',
        '''    before_block = set(chosen_names)
    extend(block_random, random_take)
    block_selected = sum(
        by_name[item.candidate.name].get("lane") == "block_bandit"
        and item.candidate.name not in before_block
        for item in chosen
    )
    extend(block_bandit, max(0, block_take - block_selected))
''',
        "block-bandit reserve after information seats",
    )
    text = replace_once(
        text,
        '''        if name in eligible_names:
            reason = "reduced_qualified"
        elif name in reserved_names:
            reason = "reduced_lane_reserved"
        else:
            reason = "reduced_ranked"
''',
        '''        if name in disagreement_names:
            reason = "proxy_reduced_disagreement"
        elif name in uniform_names:
            reason = "uniform_information"
        elif name in eligible_names:
            reason = "reduced_qualified"
        elif name in reserved_names:
            reason = "reduced_lane_reserved"
        else:
            reason = "reduced_ranked"
''',
        "promotion reasons",
    )
    text = replace_once(
        text,
        '''        "promoted_below_threshold": sum(
            item.candidate.name not in eligible_names for item in chosen
        ),
''',
        '''        "promoted_below_threshold": sum(
            item.candidate.name not in eligible_names for item in chosen
        ),
        "disagreement_seats": len(disagreement_names),
        "uniform_information_seats": len(uniform_names),
''',
        "promotion metrics",
    )
    text = replace_once(
        text,
        'print(json.dumps({"event": "reduced_primary_gate_passed", **source["reduced_primary"]}))',
        'print(json.dumps({"event": "reduced_primary_ranked", **source["reduced_primary"]}))',
        "promotion event",
    )
    backup_and_write(path, text)
    return True


ADAPTIVE_CONFIRM = r'''confirm_hits() {   # sequential VOI: ingest before deciding the next pass
  local prefix stage stage_prefix selected acq out cap report
  prefix="$ART/rxn$HC_REACTION-adaptive-confirm-$1"
  rm -f "$prefix"-stage*.json "$prefix"-stage*-scores.json "$prefix"-stage*.selection.json
  for stage in 1 2; do
    stage_prefix="$prefix-stage$stage"
    if [ "$stage" -eq 1 ]; then cap="${HIT_CONFIRM_FIRST_MAX:-40}"; else cap="${HIT_CONFIRM_FINAL_MAX:-24}"; fi
    selected="$("${RUN_PROXY[@]}" tools/confirm_hits.py select \
      --bank "$BANK" --reaction "$HC_REACTION" --historical-molecules "$HMOL" \
      --max "$cap" --minimum-probability "${HIT_CONFIRM_MIN_PROB:-0.10}" \
      --admission-margin "${HIT_CONFIRM_MARGIN:-0.010}" \
      --prefix "$stage_prefix" 2>&1 | tail -1)"
    acq="$stage_prefix-next.json"
    if [ ! -f "$stage_prefix.selection.json" ] || [ ! -f "$acq" ]; then
      echo "$selected"
      continue
    fi
    out="$stage_prefix-scores.json"
    "${RUN[@]}" tools/score_rxn3_acquisition.py \
      --acquisition "$acq" --output "$out" \
      --target P40261 --nim-url "$NIM" --no-api-key \
      --backend-scorer native-nim19-seed68+nim19-oss-mw-f64-v1 \
      --concurrency "$(current_concurrency)" --timeout 180 --attempts 3 --checkpoint-every 8 >/dev/null 2>&1 || true
    if is_complete "$out"; then
      "${RUN[@]}" tools/build_rxn3_bank.py \
        --bank "$BANK" --reaction "$HC_REACTION" --scores "$out" --historical-molecules "$HMOL" \
        --portfolio-output "$stage_prefix-portfolio.json" \
        --portfolio-count 1 --reserve-count 0 \
        --minimum-score "$(hc_minimum_score)" --minimum-entropy 0.25 --maximum-historical-similarity 0.5 >/dev/null 2>&1 || true
    fi
    report="$("${RUN[@]}" tools/confirm_hits.py report --bank "$BANK" --prefix "$stage_prefix" 2>&1 | tail -1)"
    echo "$report"
  done
}
'''


def patch_loop(path: Path) -> bool:
    text = path.read_text()
    changed = False
    if "sequential VOI: ingest before deciding the next pass" not in text:
        start = text.index("confirm_hits() {")
        end_marker = "\n}\n\n# Proxy-guided search"
        end = text.index(end_marker, start) + 2
        text = text[:start] + ADAPTIVE_CONFIRM.rstrip() + text[end:]
        changed = True
    old = '''rescore_top() {
  rescore_export "$1"
  rescore_score "$1"
  rescore_ingest "$1"
}
'''
    new = '''rescore_top() {
  if [ "${INFORMATION_EFFICIENT_CONFIRM:-1}" = 1 ]; then
    echo '{"event":"rescore_top_skipped","reason":"adaptive_confirmation_owns_passes_2_and_3"}'
    return 0
  fi
  rescore_export "$1"
  rescore_score "$1"
  rescore_ingest "$1"
}
'''
    if "adaptive_confirmation_owns_passes_2_and_3" not in text:
        if old in text:
            text = replace_once(text, old, new, "legacy rescore guard")
        else:
            # RXN3 predates the split rescore helpers and keeps the complete
            # export/score/ingest body inline.  Replace that bounded function
            # without touching its reaction-specific loop geometry.
            start = text.index("rescore_top() {")
            end_marker = "\n}\n\nHC_REACTION="
            end = text.index(end_marker, start) + 2
            text = text[:start] + new.rstrip() + text[end:]
        changed = True
    if changed:
        backup_and_write(path, text)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--loop", default=LOOP,
                        help="loop path relative to --root")
    args = parser.parse_args()
    root = args.root.resolve()
    reduced = root / REDUCED
    loop = root / args.loop
    if not reduced.is_file() or not loop.is_file():
        raise SystemExit("expected live reduced-ranker and loop are absent")
    changed = {
        "reduced": patch_reduced(reduced),
        "loop": patch_loop(loop),
    }
    print({"event": "information_efficient_policy", "changed": changed})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

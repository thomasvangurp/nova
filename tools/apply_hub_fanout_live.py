#!/usr/bin/env python3
"""Install reaction-specific reduced-hit hub fan-out on live compound loops."""
from __future__ import annotations

import argparse
import errno
import os
import shutil
import time
from pathlib import Path


MARKER = "reduced_hub_fanout_v1"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def write(path: Path, text: str) -> None:
    if path.read_text() == text:
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    shutil.copy2(path, path.with_name(f"{path.name}.pre-hub-fanout-{stamp}.bak"))
    temporary = path.with_name(f".{path.name}.hub-fanout.tmp")
    temporary.write_text(text)
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


def patch_anchor_builder(path: Path) -> None:
    text = path.read_text()
    if "reduced_accumulator_anchor_v1" in text:
        return
    text = replace_once(
        text,
        '''        payload = json.loads(path.read_text())
        values = payload.get("scores")
        if not isinstance(values, list):
            raise ValueError(f"private score artifact has no scores list: {path}")
''',
        '''        payload = json.loads(path.read_text())
        # reduced_accumulator_anchor_v1: the cross-round reduced ledger is a
        # top-level list; ordinary exact-score artifacts wrap rows in scores.
        values = payload if isinstance(payload, list) else payload.get("scores")
        if not isinstance(values, list):
            raise ValueError(f"private score artifact has no score rows: {path}")
''',
        "reduced accumulator anchor input",
    )
    write(path, text)


def patch_proxy_ranker(path: Path) -> None:
    text = path.read_text()
    if "reduced_hub_score_floor_v1" in text:
        return
    old = '''    strategy = _read_flag(art / f".competitor_strategy_rxn{reaction}")
    if strategy not in ("aggressive", "field_recombination"):
        return dict(quotas), {"enabled": False}

    def requested(path):
        try:
            return max(0, min(500, int(_read_flag(path) or 0)))
        except ValueError:
            return 0

    groups = [
        ("competitor_motif", requested(art / f".competitor_motif_share_rxn{reaction}")),
        ("block_bandit", requested(art / f".block_bandit_share_rxn{reaction}")),
    ]
    out = dict(quotas)
    protected = set()
    report = {"enabled": True, "strategy": strategy, "batch_reference": 500, "groups": {}}
'''
    new = '''    strategy = _read_flag(art / f".competitor_strategy_rxn{reaction}")
    # reduced_hub_score_floor_v1: public winners show reaction-specific family
    # concentration (especially rxn3/5). Reserve enough proxy-arm observations
    # to judge descendants of top reduced hits even when the global proxy has
    # never seen their new hub component. A flag containing "off" is the
    # reversible kill switch; absent means live.
    hub_flag = _read_flag(art / f".reduced_hub_strategy_rxn{reaction}")
    hub_enabled = hub_flag != "off"
    competitor_enabled = strategy in ("aggressive", "field_recombination")
    if not hub_enabled and not competitor_enabled:
        return dict(quotas), {"enabled": False}

    def requested(path, default=0):
        raw = _read_flag(path)
        try:
            return max(0, min(500, int(raw if raw is not None else default)))
        except ValueError:
            return max(0, min(500, int(default)))

    # Final-500 seat targets: broad rxn1/4 remain exploratory, rxn2 is
    # intermediate, and rxn3/5 receive the concentrated family harvesting
    # observed in the field's winning portfolios.
    hub_default = {1: 60, 2: 120, 3: 200, 4: 60, 5: 180}.get(int(reaction), 60)
    groups = []
    if hub_enabled:
        groups.append((
            "reduced_hub",
            requested(art / f".reduced_hub_score_share_rxn{reaction}", hub_default),
        ))
    if competitor_enabled:
        groups.extend([
            ("competitor_motif", requested(art / f".competitor_motif_share_rxn{reaction}")),
            ("block_bandit", requested(art / f".block_bandit_share_rxn{reaction}")),
        ])
    out = dict(quotas)
    protected = set()
    report = {
        "enabled": True, "strategy": strategy, "hub_enabled": hub_enabled,
        "batch_reference": 500, "groups": {},
    }
'''
    text = replace_once(text, old, new, "reduced hub score floor")
    write(path, text)


HUB_FUNCTIONS = r'''# reduced_hub_fanout_v1: convert the strongest reduced-Boltz observations
# into families of novel single-role neighbours. Field winners concentrate a
# hub in roughly 49% of rxn3 and 37% of rxn5 portfolios, but remain broad on
# rxn1/rxn4; keep that geometry reaction-specific.
hub_max_component_reuse() {
  case "$REACTION" in
    1) echo 5 ;;
    2) echo 6 ;;
    3) echo 12 ;;
    4) echo 5 ;;
    5) echo 10 ;;
  esac
}
hub_pool_share() {
  local default raw
  case "$REACTION" in
    1) default=180 ;;
    2) default=350 ;;
    3) default=600 ;;
    4) default=180 ;;
    5) default=550 ;;
  esac
  raw="$(cat "$ART/.reduced_hub_pool_share_rxn$REACTION" 2>/dev/null || echo "$default")"
  case "$raw" in ''|*[!0-9]*) raw="$default" ;; esac
  echo "$raw"
}
hub_role_weights() {
  case "$REACTION" in
    1|2) echo "1,1.2" ;;
    3) echo "1,1.8,1.5" ;;
    4) echo "1.3,1" ;;
    5) echo "1,4.5,4.5" ;;
  esac
}
generate_reduced_hub_round() {
  local r="$1" acq anchor_count result weights
  [ -s "$REDUCED_ACCUMULATOR" ] || return 1
  acq="$(acq_path "$r")"
  weights="$(hub_role_weights)"
  for anchor_count in 30 24 16 12 8; do
    rm -f "$acq"
    result="$("${RUN[@]}" tools/build_rxn3_anchor_neighborhood.py \
      --history "$HIST" --anchor-scores "$REDUCED_ACCUMULATOR" \
      --historical-molecules "$HMOL" --component-database "$DB" \
      --bank "$BANK" --reaction "$REACTION" --output "$acq" \
      --anchor-count "$anchor_count" --neighbours-per-role 16 \
      --role-weights "$weights" --minimum-anchor-score 0.0 \
      --minimum-component-similarity 0.3 --maximum-historical-similarity 0.5 \
      --max-component-reuse "$(hub_max_component_reuse)" \
      --minimum-candidates 20 2>&1 | tail -1)"
    if [ -f "$acq" ]; then
      log "round $r: $result (anchor-count=$anchor_count, reduced-hub fanout)"
      return 0
    fi
    log "round $r: reduced-hub anchor-count=$anchor_count failed: $result"
  done
  return 1
}
'''


def patch_loop(path: Path) -> None:
    text = path.read_text()
    if MARKER in text:
        return
    accumulator = 'REDUCED_ACCUMULATOR="$ART/.reduced-primary-scores-rxn$REACTION.json"\n'
    text = replace_once(
        text, accumulator, accumulator + HUB_FUNCTIONS,
        "hub functions",
    )
    if "--max-component-reuse 5" in text:
        text = text.replace(
            "--max-component-reuse 5",
            '--max-component-reuse "$(hub_max_component_reuse)"',
        )
    if "--max-component-reuse 3" in text:
        text = text.replace(
            "--max-component-reuse 3",
            '--max-component-reuse "$(hub_max_component_reuse)"',
        )
    text = replace_once(
        text,
        '''  case "$tier" in
    refinement) generate_refinement_round "$r" || true ;;
''',
        '''  case "$tier" in
    reduced_hub) generate_reduced_hub_round "$r" || true
                 cap="$(hub_pool_share)" ;;
    refinement) generate_refinement_round "$r" || true ;;
''',
        "reduced hub tier handler",
    )
    text = replace_once(
        text,
        '  local r="$1" order="refinement"',
        '  local r="$1" order="reduced_hub refinement"',
        "reduced hub tier order",
    )
    write(path, text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--loop-script", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    patch_anchor_builder(root / "tools/build_rxn3_anchor_neighborhood.py")
    patch_proxy_ranker(root / "tools/proxy_rank_acquisition.py")
    patch_loop(root / args.loop_script)
    print({"event": "reduced_hub_fanout_installed", "policy": MARKER})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

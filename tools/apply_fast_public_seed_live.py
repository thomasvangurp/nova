#!/usr/bin/env python3
"""Install the fast reduced cascade and public-best fanout on live loops."""
from __future__ import annotations

import argparse
import errno
import os
import shutil
import time
from pathlib import Path


MARKER = "fast_public_seed_v1"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def write(path: Path, text: str) -> None:
    if path.read_text() == text:
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    shutil.copy2(path, path.with_name(f"{path.name}.pre-fast-public-{stamp}.bak"))
    temporary = path.with_name(f".{path.name}.fast-public.tmp")
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


def patch_rank_shell(path: Path) -> None:
    text = path.read_text()
    if "fast_reduced_cascade_v1" in text:
        return
    text = replace_once(
        text,
        "# Permanent reduced-primary cascade:\n",
        "# fast_reduced_cascade_v1: score a proxy-ranked 256 rather than waiting\n"
        "# for 800 reduced structures before every full-fidelity decision.\n"
        "# Permanent reduced-primary cascade:\n",
        "fast cascade marker",
    )
    text = replace_once(text, "proxy_cap 950", "proxy_cap 512", "proxy cap")
    text = replace_once(
        text,
        "--state \"$state\" --reduced-proxy-count 720 --reduced-random-count 80 \\",
        "--state \"$state\" --reduced-proxy-count 224 --reduced-random-count 32 \\",
        "reduced batch size",
    )
    text = replace_once(
        text,
        '  if [ "$completed" -lt 256 ]; then',
        '  if [ "$completed" -lt 160 ]; then',
        "partial salvage floor",
    )
    write(path, text)


def patch_proxy_ranker(path: Path) -> None:
    text = path.read_text()
    if "public_best_score_floor_v1" in text:
        return
    text = replace_once(
        text,
        '''    hub_flag = _read_flag(art / f".reduced_hub_strategy_rxn{reaction}")
    hub_enabled = hub_flag != "off"
    competitor_enabled = strategy in ("aggressive", "field_recombination")
    if not hub_enabled and not competitor_enabled:
        return dict(quotas), {"enabled": False}
''',
        '''    hub_flag = _read_flag(art / f".reduced_hub_strategy_rxn{reaction}")
    hub_enabled = hub_flag != "off"
    # public_best_score_floor_v1: descendants of the live field's strongest
    # molecules must receive real reduced-score observations even when the
    # proxy has not learned their component combination yet.
    public_flag = _read_flag(art / f".public_best_strategy_rxn{reaction}")
    public_enabled = public_flag != "off"
    competitor_enabled = strategy in ("aggressive", "field_recombination")
    if not hub_enabled and not public_enabled and not competitor_enabled:
        return dict(quotas), {"enabled": False}
''',
        "public strategy enablement",
    )
    text = replace_once(
        text,
        '''    hub_default = {1: 60, 2: 120, 3: 200, 4: 60, 5: 180}.get(int(reaction), 60)
    groups = []
    if hub_enabled:
''',
        '''    hub_default = {1: 60, 2: 120, 3: 200, 4: 60, 5: 180}.get(int(reaction), 60)
    public_default = {1: 50, 2: 80, 3: 120, 4: 50, 5: 100}.get(int(reaction), 50)
    groups = []
    if public_enabled:
        groups.append((
            "public_best",
            requested(art / f".public_best_score_share_rxn{reaction}", public_default),
        ))
    if hub_enabled:
''',
        "public score floor",
    )
    text = replace_once(
        text,
        '''        "enabled": True, "strategy": strategy, "hub_enabled": hub_enabled,
        "batch_reference": 500, "groups": {},
''',
        '''        "enabled": True, "strategy": strategy, "hub_enabled": hub_enabled,
        "public_best_enabled": public_enabled,
        "batch_reference": 500, "groups": {},
''',
        "public floor report",
    )
    write(path, text)


PUBLIC_BEST_FUNCTIONS = r'''# fast_public_seed_v1: fan out from the strongest
# public molecules before broad generic enumeration. The descendants still
# pass the normal validity, public-history novelty and submitted-space gates.
public_best_max_component_reuse() {
  case "$REACTION" in
    1) echo 5 ;;
    2) echo 8 ;;
    3) echo 14 ;;
    4) echo 5 ;;
    5) echo 12 ;;
  esac
}
public_best_pool_share() {
  local default raw
  case "$REACTION" in
    1) default=220 ;;
    2) default=300 ;;
    3) default=450 ;;
    4) default=400 ;;
    5) default=450 ;;
  esac
  raw="$(cat "$ART/.public_best_pool_share_rxn$REACTION" 2>/dev/null || echo "$default")"
  case "$raw" in ''|*[!0-9]*) raw="$default" ;; esac
  echo "$raw"
}
generate_public_best_round() {
  local r="$1" acq anchor_count result weights public_anchors cache sig_file sig old_sig
  acq="$(acq_path "$r")"
  weights="$(hub_role_weights)"
  # public_best_cache_v3: the generator is deterministic for a fixed public
  # history/novelty archive. Reuse its full shortlist across rounds and let
  # pool_merge remove already-reduced rows, rather than spending 1-2 minutes
  # reconstructing the same molecules each iteration.
  cache="$ART/public-best-shortlist-rxn$REACTION.json"
  sig_file="$ART/.public-best-shortlist-rxn$REACTION.sha256"
  sig="$(sha256sum "$HIST" "$HMOL" 2>/dev/null | sha256sum | cut -d' ' -f1)"
  old_sig="$(cat "$sig_file" 2>/dev/null || true)"
  if [ -n "$sig" ] && [ "$sig" = "$old_sig" ] && [ -f "$cache" ]; then
    cp "$cache" "$acq"
    log "round $r: reused public-best shortlist for unchanged public evidence"
    return 0
  fi
  # public_best_rxn3_multisub_v2: a one-component neighbour of a public
  # RXN3 molecule cannot normally cross the <=0.5 whole-molecule novelty
  # boundary. Build two-position children from only the public top 20 instead.
  # This yielded 100 release-novel candidates in the live exhaustion state,
  # where the one-position version yielded zero.
  if [ "$REACTION" = 3 ]; then
    public_anchors="$ART/.public-best-anchors-rxn3-$(round_tag "$r").json"
    "${RUN[@]}" tools/merge_top_anchors.py \
      --bank "$BANK" --history "$HIST" --component-database "$DB" \
      --reaction 3 --bank-limit 0 --public-limit 20 \
      --output "$public_anchors" >/dev/null 2>&1 || return 1
    rm -f "$acq"
    result="$("${RUN[@]}" tools/build_novel_multisub.py \
      --reaction 3 --anchor-scores "$public_anchors" \
      --historical-molecules "$HMOL" --component-database "$DB" \
      --bank "$BANK" --output "$acq" --anchor-count 20 \
      --neighbours-per-role 16 --maximum-similarity 0.5 \
      --minimum-component-similarity 0.25 --minimum-candidates 20 \
      2>&1 | tail -1)"
    if [ -f "$acq" ]; then
      cp "$acq" "$cache"
      echo "$sig" > "$sig_file"
      log "round $r: $result (public top-20, two-position fanout)"
      return 0
    fi
    log "round $r: public top-20 two-position fanout failed: $result"
    return 1
  fi
  # Twenty public parents already produce roughly a full lane on measured
  # two-role chemistry; larger sets added CPU latency without adding scored
  # seats because pool_merge immediately capped the lane.
  for anchor_count in 20 16 12 8; do
    rm -f "$acq"
    result="$("${RUN[@]}" tools/build_rxn3_anchor_neighborhood.py \
      --history "$HIST" --historical-molecules "$HMOL" \
      --component-database "$DB" --bank "$BANK" --reaction "$REACTION" \
      --output "$acq" --anchor-count "$anchor_count" \
      --neighbours-per-role 16 --role-weights "$weights" \
      --minimum-anchor-score 0.0 --minimum-component-similarity 0.25 \
      --maximum-historical-similarity 0.5 \
      --max-component-reuse "$(public_best_max_component_reuse)" \
      --minimum-candidates 20 2>&1 | tail -1)"
    if [ -f "$acq" ]; then
      cp "$acq" "$cache"
      echo "$sig" > "$sig_file"
      log "round $r: $result (anchor-count=$anchor_count, public-best fanout)"
      return 0
    fi
    log "round $r: public-best anchor-count=$anchor_count failed: $result"
  done
  return 1
}
'''


def add_hub_cache(text: str) -> str:
    if "reduced_hub_cache_v2" in text:
        return text
    text = replace_once(
        text,
        '''  local r="$1" acq anchor_count result weights
  [ -s "$REDUCED_ACCUMULATOR" ] || return 1
  acq="$(acq_path "$r")"
  weights="$(hub_role_weights)"
''',
        '''  local r="$1" acq anchor_count result weights cache sig_file anchor_sig evidence_sig sig old_sig
  [ -s "$REDUCED_ACCUMULATOR" ] || return 1
  acq="$(acq_path "$r")"
  weights="$(hub_role_weights)"
  # reduced_hub_cache_v2: regeneration is only useful when the actual top-30
  # reduced anchors or public evidence changes. Reuse the deterministic full
  # shortlist otherwise; pool_merge removes observations already consumed.
  cache="$ART/reduced-hub-shortlist-rxn$REACTION.json"
  sig_file="$ART/.reduced-hub-shortlist-rxn$REACTION.sha256"
  anchor_sig="$(python3 -c 'import hashlib,json,sys; rows=json.load(open(sys.argv[1])); rows=sorted(((float(x.get("score")),str(x.get("name") or x.get("smiles") or "")) for x in rows if isinstance(x.get("score"),(int,float))),key=lambda z:(-z[0],z[1]))[:30]; print(hashlib.sha256(json.dumps(rows,separators=(",",":")).encode()).hexdigest())' "$REDUCED_ACCUMULATOR" 2>/dev/null || true)"
  evidence_sig="$(sha256sum "$HIST" "$HMOL" 2>/dev/null | sha256sum | cut -d' ' -f1)"
  sig="$(printf '%s\\n%s\\n' "$anchor_sig" "$evidence_sig" | sha256sum | cut -d' ' -f1)"
  old_sig="$(cat "$sig_file" 2>/dev/null || true)"
  if [ -n "$anchor_sig" ] && [ "$sig" = "$old_sig" ] && [ -f "$cache" ]; then
    cp "$cache" "$acq"
    log "round $r: reused reduced-hub shortlist for unchanged top anchors"
    return 0
  fi
''',
        "reduced hub cache prelude",
    )
    text = replace_once(
        text,
        '''    if [ -f "$acq" ]; then
      log "round $r: $result (anchor-count=$anchor_count, reduced-hub fanout)"
''',
        '''    if [ -f "$acq" ]; then
      cp "$acq" "$cache"
      echo "$sig" > "$sig_file"
      log "round $r: $result (anchor-count=$anchor_count, reduced-hub fanout)"
''',
        "save reduced hub cache",
    )
    return text


def patch_loop(path: Path) -> None:
    text = path.read_text()
    if MARKER in text:
        if "public_best_rxn3_multisub_v2" not in text:
            old = '''  local r="$1" acq anchor_count result weights
  acq="$(acq_path "$r")"
  weights="$(hub_role_weights)"
'''
            new = PUBLIC_BEST_FUNCTIONS.split("generate_public_best_round() {\n", 1)[1].split(
                "  # Twenty public parents", 1
            )[0]
            text = replace_once(text, old, new, "RXN3 public multisub upgrade")
        if "efficient_tier_skip_v2" not in text:
            text = replace_once(
                text,
                '  local r="$1" order="public_best reduced_hub refinement"',
                '  local r="$1" order="public_best reduced_hub"',
                "defer exhausted refinement",
            )
            text = replace_once(
                text,
                '  sibling_scan_enabled && order="$order sibling"\n',
                '  order="$order refinement"\n'
                '  sibling_scan_enabled && order="$order sibling"\n',
                "late refinement tier",
            )
            text = replace_once(
                text,
                '''    count="$(pool_count "$pool")"
    [ "$count" -ge "$POOL_TARGET" ] && { log "round $r: pool target reached ($count >= $POOL_TARGET), skipping $tier and later tiers"; break; }
''',
                '''    count="$(pool_count "$pool")"
    # efficient_tier_skip_v2: refinement became mostly reduced-history
    # duplicates, and ladder was still running for tiny residual gaps despite
    # being documented as a below-POOL_MIN_BATCH fallback. A 500+ pool is
    # already sufficient for the 512 proxy cap / 256 reduced cohort.
    if { [ "$tier" = refinement ] || [ "$tier" = ladder ]; } \
        && [ "$count" -ge "$POOL_MIN_BATCH" ]; then
      log "round $r: pool $count >= $POOL_MIN_BATCH, skipping fallback tier $tier"
      continue
    fi
    [ "$count" -ge "$POOL_TARGET" ] && { log "round $r: pool target reached ($count >= $POOL_TARGET), skipping $tier and later tiers"; break; }
''',
                "skip exhausted fallback tiers",
            )
        if "public_best_cache_v3" not in text:
            text = replace_once(
                text,
                '''  local r="$1" acq anchor_count result weights public_anchors
  acq="$(acq_path "$r")"
  weights="$(hub_role_weights)"
''',
                '''  local r="$1" acq anchor_count result weights public_anchors cache sig_file sig old_sig
  acq="$(acq_path "$r")"
  weights="$(hub_role_weights)"
  # public_best_cache_v3: reuse the deterministic public-seed shortlist until
  # the public history or novelty archive actually changes.
  cache="$ART/public-best-shortlist-rxn$REACTION.json"
  sig_file="$ART/.public-best-shortlist-rxn$REACTION.sha256"
  sig="$(sha256sum "$HIST" "$HMOL" 2>/dev/null | sha256sum | cut -d' ' -f1)"
  old_sig="$(cat "$sig_file" 2>/dev/null || true)"
  if [ -n "$sig" ] && [ "$sig" = "$old_sig" ] && [ -f "$cache" ]; then
    cp "$cache" "$acq"
    log "round $r: reused public-best shortlist for unchanged public evidence"
    return 0
  fi
''',
                "public shortlist cache prelude",
            )
            text = text.replace(
                '''    if [ -f "$acq" ]; then
      log "round $r: $result (public top-20, two-position fanout)"
''',
                '''    if [ -f "$acq" ]; then
      cp "$acq" "$cache"
      echo "$sig" > "$sig_file"
      log "round $r: $result (public top-20, two-position fanout)"
''',
                1,
            )
            text = text.replace(
                '''    if [ -f "$acq" ]; then
      log "round $r: $result (anchor-count=$anchor_count, public-best fanout)"
''',
                '''    if [ -f "$acq" ]; then
      cp "$acq" "$cache"
      echo "$sig" > "$sig_file"
      log "round $r: $result (anchor-count=$anchor_count, public-best fanout)"
''',
                1,
            )
        text = add_hub_cache(text)
        write(path, text)
        return
    general_anchor = '}\nNIM="${NIM_URL:-http://127.0.0.1:8000/biology/mit/boltz2/predict}"'
    rxn3_anchor = '''log() { printf '[%s] %s\\n' "$(date +%H:%M:%S)" "$*"; }\n'''
    if text.count(general_anchor) == 1:
        text = text.replace(
            general_anchor,
            '}\n' + PUBLIC_BEST_FUNCTIONS
            + 'NIM="${NIM_URL:-http://127.0.0.1:8000/biology/mit/boltz2/predict}"',
            1,
        )
    elif text.count(rxn3_anchor) == 1:
        text = text.replace(
            rxn3_anchor,
            rxn3_anchor + '\n' + PUBLIC_BEST_FUNCTIONS,
            1,
        )
    else:
        raise RuntimeError("public best functions: no supported insertion point")
    text = replace_once(
        text,
        'POOL_TARGET="${POOL_TARGET:-1500}"',
        'POOL_TARGET="${POOL_TARGET:-900}"',
        "pool target",
    )
    text = replace_once(
        text,
        '''  case "$tier" in
    reduced_hub) generate_reduced_hub_round "$r" || true
''',
        '''  case "$tier" in
    public_best) generate_public_best_round "$r" || true
                 cap="$(public_best_pool_share)" ;;
    reduced_hub) generate_reduced_hub_round "$r" || true
''',
        "public best tier handler",
    )
    text = replace_once(
        text,
        '  local r="$1" order="reduced_hub refinement"',
        '  local r="$1" order="public_best reduced_hub"',
        "public best tier order",
    )
    text = replace_once(
        text,
        '  sibling_scan_enabled && order="$order sibling"\n',
        '  order="$order refinement"\n',
        "defer refinement and sibling scan",
    )
    text = replace_once(
        text,
        '  echo "$order ladder wide"',
        '  sibling_scan_enabled && order="$order sibling"\n  echo "$order ladder wide"',
        "late sibling scan",
    )
    text = replace_once(
        text,
        '''    count="$(pool_count "$pool")"
    [ "$count" -ge "$POOL_TARGET" ] && { log "round $r: pool target reached ($count >= $POOL_TARGET), skipping $tier and later tiers"; break; }
''',
        '''    count="$(pool_count "$pool")"
    # efficient_tier_skip_v2: true fallback tiers only below 500.
    if { [ "$tier" = refinement ] || [ "$tier" = ladder ]; } \
        && [ "$count" -ge "$POOL_MIN_BATCH" ]; then
      log "round $r: pool $count >= $POOL_MIN_BATCH, skipping fallback tier $tier"
      continue
    fi
    [ "$count" -ge "$POOL_TARGET" ] && { log "round $r: pool target reached ($count >= $POOL_TARGET), skipping $tier and later tiers"; break; }
''',
        "skip exhausted fallback tiers",
    )
    text = add_hub_cache(text)
    write(path, text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--loop-script", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    patch_rank_shell(root / "tools/rank_primary_acquisition.sh")
    patch_proxy_ranker(root / "tools/proxy_rank_acquisition.py")
    patch_loop(root / args.loop_script)
    print({"event": "fast_public_seed_installed", "policy": MARKER})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

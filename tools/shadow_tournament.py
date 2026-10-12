#!/usr/bin/env python3
"""Shadow tournament: which ticket-ordering rule would have won the most?

Every closed epoch we record the 13 tickets EACH candidate ordering rule would
have published from that epoch's final qualified_rankings.json (the information
the publisher actually had), have the replay queue give every one of those
sequences 10 independent validator-equivalent passes, and then play the epoch
out: every submission (ours or a rival's) gets one random draw from its own
replays, the validator's rank-sum decides the winner, repeated T times.  No
rule's tickets are ever published; real submissions are untouched.

Why this exists: the 32-epoch retrospective (simulate against the replays we
happened to have) could only score tickets somebody had already replayed, so
rules were compared on a pre-selected pool.  Here every rule's exact top-13 is
replayed, so the comparison needs no "fill" and has no coverage bias.

Not modelled: the publisher's Layer-2 / novelty / CDR gates and uid assignment.

Subcommands (all idempotent; state under /root/artifacts/shadow_tournament):
  pick   --epoch E     write epoch E's picks.json
  eval   --epoch E     simulate epoch E once every pick has >= PASSES passes
  report               aggregate results.jsonl into REPORT.md and print it
  run-once             pick + eval for every closed epoch in the lookback, then report
  pending              print the sequences the replay queue should replay first
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SHADOW_ROOT", "/root/artifacts/shadow_tournament"))
DB = "/root/artifacts/nanobody_mutations.sqlite"
TARGET = "P20809"
API = "https://compound-api.metanova-labs.ai/api/competitions/leaderboard/{}/nanobodies"
OURS = "5EkCwcNTfx"
LENGTH = 127                    # the fast scorer only supports this length
PASSES = 10
TRIALS = 4000
LOOKBACK = 40
RIVAL_MIN_COVERAGE = 0.9        # share of rivals needing >= 5 passes before an epoch is evaluated
KEYS = ["design_iiptm", "design_ptm", "design_to_target_iptm", "min_design_to_target_pae",
        "interaction_pae", "plip_hbonds_refolded", "plip_saltbridge_refolded",
        "delta_sasa_refolded", "liability_score", "liability_num_violations"]
SIGN = np.array([-1, -1, -1, 1, 1, -1, -1, -1, 1, 1], float)   # lower is better after the sign
RULES = ["current", "eps1e-4", "eps1e-3", "off", "p90", "median", "random"]


def chain_epoch() -> int:
    return int(json.load(open("/root/artifacts/chain_state.json"))["epoch"])


def conn() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=120)


def picks_path(epoch: int) -> Path:
    return ROOT / f"epoch{epoch}" / "picks.json"


# ---------------------------------------------------------------------- pick
def rule_picks(epoch: int) -> dict[str, list[str]] | None:
    path = f"/root/live/epoch{epoch}/qualified_rankings.json"
    if not os.path.exists(path):
        return None
    cand = json.load(open(path))["candidates"]
    wp = lambda x: x.get("win_probability") or 0.0
    tier = lambda x: x.get("settled_deviations", 0)
    seq = lambda x: x["sequence"].strip().upper()
    ok = [x for x in cand if len(seq(x)) == LENGTH]
    eligible = [x for x in ok if x["observation_count"] >= 8]

    def top(key, pool=ok):
        return [seq(x) for x in sorted(pool, key=key)[:13]]

    picks = {
        "current": top(lambda x: (0 if wp(x) > 0 else 1, tier(x), -wp(x))),
        "eps1e-4": top(lambda x: (0 if wp(x) >= 1e-4 else 1, tier(x), -wp(x))),
        "eps1e-3": top(lambda x: (0 if wp(x) >= 1e-3 else 1, tier(x), -wp(x))),
        "off": top(lambda x: -wp(x)),
        "p90": top(lambda x: x["selection_score"], eligible),
        "median": top(lambda x: x["score_median"], eligible),
    }
    rng = np.random.default_rng(epoch)
    picks["random"] = [seq(eligible[i]) for i in rng.choice(len(eligible), 13, replace=False)] \
        if len(eligible) >= 13 else []
    return picks


def cmd_pick(epoch: int) -> bool:
    path = picks_path(epoch)
    if path.exists():
        return False
    picks = rule_picks(epoch)
    if not picks:
        return False
    if len(glob.glob(f"/root/live/epoch{epoch}/checkpoints/qualified/uid*.json")) < 5:
        return False                      # too few of our tickets that epoch to evaluate anything
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"epoch": epoch, "made_at": time.time(), "rules": picks}, indent=1))
    os.replace(tmp, path)
    return True


# ------------------------------------------------------------------- replays
def passes(seqs) -> dict[str, list[list[float]]]:
    out: dict[str, list[list[float]]] = collections.defaultdict(list)
    seqs = sorted(set(seqs))
    c = conn()
    for i in range(0, len(seqs), 300):
        ch = seqs[i:i + 300]
        q = ("SELECT n.sequence, o.metrics_json FROM observations o JOIN nodes n ON n.id=o.node_id "
             "WHERE n.target=? AND o.source='public_multipass_rivals' AND o.score_kind='multipass_sum10_pass' "
             f"AND o.metrics_json NOT IN ('','{{}}') AND n.sequence IN ({','.join('?' * len(ch))})")
        for s, mj in c.execute(q, (TARGET, *ch)):
            try:
                m = json.loads(mj)
                m = m.get("metrics", m) if isinstance(m, dict) else {}
                if set(KEYS) <= set(m):
                    out[s.strip().upper()].append([float(m[k]) for k in KEYS])
            except ValueError:
                continue
    c.close()
    return out


def pending_sequences() -> list[str]:
    """Pick sequences still short of PASSES, newest epoch first (replay-queue tier 0)."""
    files = sorted(glob.glob(str(ROOT / "epoch*" / "picks.json")),
                   key=lambda p: int(Path(p).parent.name[5:]), reverse=True)
    ordered: list[str] = []
    for f in files:
        done = (Path(f).parent / "evaluated").exists()
        if done:
            continue
        rules = json.load(open(f))["rules"]
        for name in RULES:
            ordered += rules.get(name, [])
    ordered = list(dict.fromkeys(s for s in ordered if len(s) == LENGTH))
    have = passes(ordered)
    return [s for s in ordered if len(have.get(s, [])) < PASSES]


# ---------------------------------------------------------------------- eval
def dense_rank_sum(X: np.ndarray) -> np.ndarray:
    order = np.argsort(X, axis=1, kind="stable")
    s = np.take_along_axis(X, order, 1)
    d = np.ones_like(s, dtype=np.int32)
    d[:, 1:, :] = (s[:, 1:, :] != s[:, :-1, :])
    c = np.cumsum(d, axis=1)
    r = np.empty_like(c)
    np.put_along_axis(r, order, c, 1)
    return r.sum(2)


def fetch_board(epoch: int):
    req = urllib.request.Request(API.format(epoch), headers={"User-Agent": "nova-shadow/1"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=30))["data"]
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def cmd_eval(epoch: int, trials: int = TRIALS) -> str:
    if (ROOT / f"epoch{epoch}" / "evaluated").exists():
        return "already"
    path = picks_path(epoch)
    if not path.exists():
        return "no-picks"
    picks = json.load(open(path))["rules"]
    board = fetch_board(epoch)
    if board is None:
        return "no-leaderboard"
    ent = [(r, n) for r in board for n in r["nanobodies"] if n.get("protein_name") == TARGET]
    mine = [i for i, (r, n) in enumerate(ent) if str(r["coldkey"]).startswith(OURS)]
    if len(mine) < 5:
        return "too-few-ours"
    seq = lambda i: ent[i][1]["sequence"].strip().upper()
    mine_set = set(mine)
    rival_idx = [i for i in range(len(ent)) if i not in mine_set]
    rivals = list(dict.fromkeys(seq(i) for i in rival_idx))
    actual = list(dict.fromkeys(seq(i) for i in mine))
    all_arm = {name: [s for s in dict.fromkeys(v)] for name, v in picks.items() if v}
    need = set(s for v in all_arm.values() for s in v)
    have = passes(set(rivals) | set(actual) | need)

    short = [s for s in need if len(have.get(s, [])) < PASSES]
    if short:
        return f"waiting-picks:{len(short)}"
    rival_ok = sum(len(have.get(s, [])) >= 5 for s in rivals) / max(len(rivals), 1)
    if rival_ok < RIVAL_MIN_COVERAGE:
        return f"waiting-rivals:{rival_ok:.2f}"

    actmat = {seq(i): [float(ent[i][1][k]) for k in KEYS] for i in range(len(ent))}
    pool = {}
    for s in set(rivals) | set(actual) | need:
        rows = list(have.get(s, [])) + ([actmat[s]] if s in actmat else [])
        if rows:
            pool[s] = np.array(rows) * SIGN
    rng = np.random.default_rng(epoch)
    U = {s: rng.random(trials) for s in pool}
    draw = lambda s: pool[s][(U[s] * len(pool[s])).astype(int)]
    rivals = [s for s in rivals if s in pool]
    Rm = np.stack([draw(s) for s in rivals], 1)
    nr = Rm.shape[1]
    arms = dict(all_arm)
    arms["actual"] = actual
    out = []
    real_best = min((ent[i][0].get("nanobody_leaderboard_rank") or 99) for i in mine)
    rs = set(rivals)
    for name, seqs in arms.items():
        have_seqs = [s for s in seqs if s in pool and s not in rs]
        if not have_seqs:
            continue
        Om = np.stack([draw(s) for s in have_seqs], 1)
        F = dense_rank_sum(np.concatenate([Rm, Om], 1))
        Fo, Fr = F[:, nr:], F[:, :nr]
        best, rb = Fo.min(1), Fr.min(1)
        win = (best < rb) + 0.5 * (best == rb)
        rank_best = 1 + (F < best[:, None]).sum(1)
        top10 = ((1 + (F[:, None, :] < Fo[:, :, None]).sum(2)) <= 10).sum(1)
        out.append({"epoch": epoch, "rule": name, "p_win": float(win.mean()),
                    "mean_best_rank": float(rank_best.mean()), "top10": float(top10.mean()),
                    "n_tickets": len(have_seqs), "trials": trials,
                    "min_passes": int(min(len(have.get(s, [])) for s in have_seqs)),
                    "real_best_rank": int(real_best), "rivals": nr,
                    "evaluated_at": time.time()})
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "results.jsonl", "a") as fh:
        for row in out:
            fh.write(json.dumps(row) + "\n")
    (ROOT / f"epoch{epoch}" / "evaluated").write_text(str(time.time()))
    return "evaluated"


# -------------------------------------------------------------------- report
def load_results() -> dict[str, dict[int, dict]]:
    res: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    p = ROOT / "results.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                res[row["rule"]][row["epoch"]] = row      # last write wins
    return res


def cmd_report() -> str:
    res = load_results()
    if "current" not in res:
        return "No evaluated epochs yet."
    epochs = sorted(res["current"])
    lines = [f"# Shadow tournament\n", f"Epochs evaluated: {len(epochs)} ({epochs[0]}..{epochs[-1]}), "
             f"{TRIALS} simulated draws each, every rule's exact top-13 replayed with {PASSES} independent passes.\n",
             "Gates (Layer-2 / novelty / CDR) are not modelled. Differences are what matter, not absolute levels.\n"]
    real_wins = sum(res["actual"][e]["real_best_rank"] == 1 for e in epochs if e in res.get("actual", {}))
    if "actual" in res:
        sim = sum(res["actual"][e]["p_win"] for e in epochs if e in res["actual"])
        lines.append(f"Calibration: simulator expects {sim:.2f} wins for our real tickets, real wins {real_wins}.\n")
    lines.append("| rule | expected wins | per epoch | vs current (wins) | 95% CI per epoch | better/worse epochs | top10/epoch | mean best rank |")
    lines.append("|---|---:|---:|---:|---|---|---:|---:|")
    rng = np.random.default_rng(1)
    for name in ["actual", "current", "eps1e-4", "eps1e-3", "off", "p90", "median", "random"]:
        if name not in res:
            continue
        ee = [e for e in epochs if e in res[name]]
        tot = sum(res[name][e]["p_win"] for e in ee)
        d = np.array([res[name][e]["p_win"] - res["current"][e]["p_win"] for e in ee])
        if name == "current" or len(d) < 3:
            ci, bw = "-", "-"
        else:
            bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
            ci = f"{np.percentile(bs, 2.5):+.3f}..{np.percentile(bs, 97.5):+.3f}"
            bw = f"{int((d > 0.005).sum())}/{int((d < -0.005).sum())}"
        lines.append(f"| {name} | {tot:.2f} | {tot / len(ee):.3f} | {d.sum():+.2f} | {ci} | {bw} | "
                     f"{np.mean([res[name][e]['top10'] for e in ee]):.2f} | "
                     f"{np.mean([res[name][e]['mean_best_rank'] for e in ee]):.1f} |")
    text = "\n".join(lines) + "\n"
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "REPORT.md").write_text(text)
    return text


def cmd_run_once() -> None:
    latest = chain_epoch()
    changed = False
    for epoch in range(latest - 1, latest - 1 - LOOKBACK, -1):
        made = cmd_pick(epoch)
        status = cmd_eval(epoch) if picks_path(epoch).exists() else "no-picks"
        if made or status == "evaluated":
            changed = True
        if made or status not in ("already", "no-picks") and not status.startswith("waiting"):
            print(f"SHADOW epoch={epoch} picked={made} eval={status}", flush=True)
    waiting = len(pending_sequences())
    print(f"SHADOW_STATUS latest_closed={latest - 1} sequences_awaiting_replay={waiting}", flush=True)
    if changed:
        print(cmd_report(), flush=True)


def main() -> int:
    global PASSES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["pick", "eval", "report", "run-once", "pending"])
    ap.add_argument("--epoch", type=int)
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--passes", type=int, default=PASSES, help="plumbing tests only (SHADOW_ROOT set)")
    a = ap.parse_args()
    PASSES = a.passes
    if a.command == "pick":
        print("picked" if cmd_pick(a.epoch) else "skipped")
    elif a.command == "eval":
        print(cmd_eval(a.epoch, a.trials))
    elif a.command == "report":
        print(cmd_report())
    elif a.command == "run-once":
        cmd_run_once()
    else:
        print("\n".join(pending_sequences()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

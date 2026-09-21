#!/usr/bin/env python3
"""Produce one epoch's compound fleet artifact, locally, from the mirrored bank.

Runs on the controller so no rented box has to be awake to submit. Two things
this must get right, both of which have already bitten:

*The entropy metric.* The allocator calls `maccs_entropy`, which in THIS tree
computes AtomPair -- the metric the network gates on. A sibling tree still
ships a MACCS implementation under the same name, and a fleet selected against
it satisfies a constraint nobody measures: 0.28 MACCS can be 0.187 AtomPair
against a 0.25 gate. Always import the allocator from the compound fleet tree.

*The crew.* Read from the chain every epoch, never cached. uid numbers are
recycled and a remembered one silently follows whoever holds it next.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLEET_TREE = ROOT.parent / "nova-compound-fleet"   # overridden by --fleet-tree


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--reaction", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--bank", default=str(ROOT / "data/release/bank.sqlite"))
    parser.add_argument("--crew", required=True)
    parser.add_argument("--out", default=str(ROOT / "data/release"))
    parser.add_argument("--python", required=True,
                        help="interpreter with rdkit AND the compound fleet tree importable")
    parser.add_argument("--fleet-tree", default=None,
                        help="directory holding compound_miner; on the submit "
                             "host it sits beside the tools, not in a sibling "
                             "worktree")
    parser.add_argument("--archive", default="/opt/nova-submit/P40261_molecules.csv",
                        help="Metanova/Submission-Archive CSV. bank_release "
                             "defaults this to /root/bank, which is the BANK "
                             "box path and does not exist on the submit host; "
                             "without it every molecule we send is refused.")
    parser.add_argument("--core", type=int, default=16,
                        help="shared core size; 16 measured best for P(win) at this pool size")
    parser.add_argument("--entropy", type=float, default=0.25,
                        help="the real gate. Do not pad it against MACCS numbers: "
                             "on AtomPair these sets read 0.36+, never near the floor")
    arguments = parser.parse_args()

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    release = out / f"result_rxn{arguments.reaction}.json"
    marker = out / f"epoch{arguments.epoch}_rxn{arguments.reaction}_complete.json"
    fleet = out / "compound_fleet.json"

    crew = json.loads(Path(arguments.crew).read_text())
    made = subprocess.run(
        [arguments.python, str(ROOT / "tools/bank_release.py"),
         "--bank", arguments.bank, "--reaction", str(arguments.reaction),
         "--epoch", str(arguments.epoch), "--fleet", str(len(crew)),
         "--archive", arguments.archive,
         "--output", str(out / "result_rxn{reaction}.json"),
         "--complete", str(out / f"epoch{arguments.epoch}_rxn{{reaction}}_complete.json")],
        capture_output=True, text=True, timeout=900)
    if made.returncode:
        print(json.dumps(dict(event="release_failed",
                              err=(made.stdout or made.stderr).strip()[-200:])))
        return 1
    print(json.dumps(dict(event="release", line=made.stdout.strip().splitlines()[-1][:180])))

    script = f'''
import json, sys
sys.path.insert(0, {str(Path(arguments.fleet_tree) if arguments.fleet_tree else FLEET_TREE)!r})
from compound_miner.fleet import allocate_compound_fleet_from_release, FleetMember
from compound_miner.validity import maccs_entropy
crew = [FleetMember(uid=int(u), hotkey=h) for u, h in json.load(open({arguments.crew!r}))]
payload = json.load(open({str(release)!r}))
art = allocate_compound_fleet_from_release(
    payload, crew, shared_core_size={arguments.core}, spare_count=4,
    minimum_entropy={arguments.entropy})
json.dump(art, open({str(fleet)!r}, "w"), indent=1)
subs = sorted(art["submissions"], key=lambda s: -s["selection_score"])
print(json.dumps(dict(event="allocated", uids=len(subs), epoch=art["epoch"],
                      reaction=art["reaction"],
                      best=round(subs[0]["selection_score"], 4),
                      worst=round(subs[-1]["selection_score"], 4),
                      entropy_min=round(min(s["entropy"] for s in subs), 4),
                      distinct=len({{m["name"] for s in subs for m in s["molecules"]}}))))
'''
    done = subprocess.run([arguments.python, "-c", script],
                          capture_output=True, text=True, timeout=900)
    if done.returncode:
        print(json.dumps(dict(event="allocate_failed",
                              err=(done.stdout or done.stderr).strip()[-200:])))
        return 1
    for line in done.stdout.strip().splitlines():
        if line.startswith("{"):
            print(line)
    if not marker.is_file():
        print(json.dumps(dict(event="no_marker", path=str(marker))))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

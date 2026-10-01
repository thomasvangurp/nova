import json, os
from pathlib import Path

q = json.load(open("public_queue.json"))
single_pass_only = [e for e in q if not e["has_replay"]]
single_pass_only.sort(key=lambda x: (x["best_single_pass"] is not None,
                                     x["best_single_pass"] if x["best_single_pass"] is not None else 1e9,
                                     x["sequence"]))
top50 = single_pass_only[: len(single_pass_only) // 2]
print("single_pass_only:", len(single_pass_only))
print("top50 (work queue):", len(top50))
print("best single-pass score in queue:", top50[0]["best_single_pass"])
print("worst single-pass score in queue:", top50[-1]["best_single_pass"])
print("first excluded single-pass score:", single_pass_only[len(top50)]["best_single_pass"])

# sanity: all length 127
lens = {len(e["sequence"]) for e in top50}
print("lengths in queue:", lens)

work = [{"sequence": e["sequence"], "epoch": e["first_epoch"], "node_id": e["id"]} for e in top50]
json.dump(work, open("work_queue.json", "w"))
print("wrote work_queue.json:", len(work))

# shard across 10 boxes (round-robin for near-uniform cost)
BOXES = [f"box{i}" for i in range(10)]
shards = {b: [] for b in BOXES}
for i, w in enumerate(work):
    shards[BOXES[i % 10]].append(w)
os.makedirs("shards", exist_ok=True)
for b, items in shards.items():
    json.dump(items, open(f"shards/{b}.json", "w"))
    print(f"shard {b}: {len(items)}")

"""Save the public compound leaderboard for every P40261 epoch, one file each.

Saved to disk first and analysed separately -- never parsed straight out of the
download -- so a partial or error response is inspectable rather than silently
becoming a number in a table.
"""
import json, sys, time, urllib.request
from pathlib import Path

ROOT = "https://compound-api.metanova-labs.ai/api"
OUT = Path("/private/tmp/claude-502/-Users-tvgdeena/6a0daf20-2ab0-4cb4-bf0b-7bcb06048b44/scratchpad/epochs")
OUT.mkdir(exist_ok=True)

def get(url, attempts=4):
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nova-compound-miner/1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as exc:
            if attempt + 1 == attempts:
                print(f"  FAILED {url.rsplit('/',2)[-2]}: {type(exc).__name__}", flush=True)
                return None
            time.sleep(0.5 * 2 ** attempt)

index = OUT / "competitions.json"
if not index.exists():
    body = get(f"{ROOT}/competitions/molecules?limit=10000")
    if body is None:
        sys.exit("could not list competitions")
    index.write_bytes(body)
payload = json.loads(index.read_text())
epochs = sorted({int(i["epoch_number"]) for i in payload.get("data", [])
                 if "P40261" in (i.get("target_proteins") or [])})
print(f"P40261 epochs available: {len(epochs)}  range {epochs[0]}-{epochs[-1]}")

fetched = 0
for epoch in epochs:
    path = OUT / f"{epoch}.json"
    if path.exists() and path.stat().st_size > 200:
        continue
    body = get(f"{ROOT}/competitions/leaderboard/{epoch}/molecules?limit=10000&molecule_detail=full")
    if body is None:
        continue
    path.write_bytes(body)
    fetched += 1
    if fetched % 10 == 0:
        print(f"  fetched {fetched}", flush=True)
    time.sleep(0.25)
print(f"saved {fetched} new, {len(list(OUT.glob('[0-9]*.json')))} total on disk")

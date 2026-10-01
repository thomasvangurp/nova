import json, os, sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
import numpy as np, statistics

API = "https://compound-api.metanova-labs.ai/api/competitions/leaderboard"
DB = '/Users/tvgdeena/nova-nanobody/data/nanobody_dash.sqlite'
TARGET = '5GhREcsKiqXyafTNd7HsVDm3y22bp5HMnZYMjtLccQWmnLpK'
METRICS = ['design_iiptm','design_ptm','design_to_target_iptm','min_design_to_target_pae',
           'interaction_pae','plip_hbonds_refolded','plip_saltbridge_refolded',
           'delta_sasa_refolded','liability_score','liability_num_violations']
EPOCHS = list(range(25330, 25420))  # 90 epochs

os.makedirs('lb', exist_ok=True)
def fetch(ep):
    p = f'lb/{ep}.json'
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return ep, json.load(open(p))
    try:
        req = Request(f'{API}/{ep}/nanobodies?limit=10000', headers={'User-Agent':'nova-parent-analysis/1'})
        with urlopen(req, timeout=30) as r:
            d = json.load(r)
        json.dump(d, open(p,'w'))
        return ep, d
    except Exception as e:
        return ep, None

print('fetching', len(EPOCHS), 'epochs...')
payloads = {}
with ThreadPoolExecutor(max_workers=12) as ex:
    for fut in as_completed([ex.submit(fetch, ep) for ep in EPOCHS]):
        ep, d = fut.result()
        if d and d.get('success'):
            payloads[ep] = d
print('got', len(payloads), 'epochs')

# Build per-epoch nanobody list: (sequence, rank, coldkey, metrics)
# rank = 1..N by final_score desc within epoch
epoch_rows = {}   # epoch -> list of {seq, rank, coldkey, score, metrics}
for ep in sorted(payloads):
    data = payloads[ep]['data']
    rows = []
    for m in data:
        nb = (m.get('nanobodies') or [{}])[0]
        seq = str(nb.get('sequence','')).upper()
        if not seq or nb.get('protein_name') != 'P20809': continue
        try:
            met = {k: float(nb[k]) for k in METRICS}
            score = float(nb['final_score'])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(dict(seq=seq, coldkey=str(m.get('coldkey','')), score=score, metrics=met))
    rows.sort(key=lambda r: -r['score'])
    for i, r in enumerate(rows, start=1):
        r['rank'] = i
    epoch_rows[ep] = rows

# sequence -> its metrics at the epoch it was public (first/most-recent)
# Build a per-sequence best observation: when it was public, its rank+metrics
seq_obs = {}   # seq -> dict(epoch, rank, score, metrics, coldkey)
for ep in sorted(epoch_rows):
    for r in epoch_rows[ep]:
        s = r['seq']
        if s not in seq_obs:
            seq_obs[s] = dict(epoch=ep, rank=r['rank'], score=r['score'], metrics=r['metrics'], coldkey=r['coldkey'])

# Reconstruct 5GhREcsK parents: for each target submission, most-recent distance-1 public parent
all_seq = [(ep, r['seq']) for ep in sorted(epoch_rows) for r in epoch_rows[ep]]
seqlist = [s for _, s in all_seq]
arr = np.frombuffer(''.join(seqlist).encode(), dtype=np.uint8).reshape(len(seqlist), len(seqlist[0]))
epoch_of = np.array([ep for ep, s in all_seq])

parent_metrics = []   # parent metric vector
field_metrics = []    # ALL public metric vectors (for baseline)
field_ranks = []

# collect field distribution
for ep in sorted(epoch_rows):
    for r in epoch_rows[ep]:
        field_metrics.append(list(r['metrics'].values()))
        field_ranks.append(r['rank'])

field_metrics = np.array(field_metrics)

# target submissions
c = sqlite3.connect(DB); cur = c.cursor()
cur.execute("SELECT epoch, rank, sequence FROM entries WHERE coldkey=? ORDER BY epoch ASC", (TARGET,))
tgt = cur.fetchall()
print(f"\ntarget submissions from db: {len(tgt)}")

parent_rows = []
for ep, rank, seq in tgt:
    # find most-recent prior public sequence at hamming 1
    prior = np.where(epoch_of < ep)[0]
    if len(prior)==0: continue
    t = arr[np.where((epoch_of==ep) & (np.array(seqlist)==seq))[0]]
    if len(t)==0: continue
    t = t[0]
    d = (arr[prior] != t).sum(axis=1)
    d1 = prior[d==1]
    if len(d1)==0: continue
    best = d1[np.argmax(epoch_of[d1])]
    pseq = seqlist[best]; pep = int(epoch_of[best])
    # get parent metrics from seq_obs (its public observation)
    obs = seq_obs.get(pseq)
    if obs is None: continue
    parent_rows.append(dict(
        epoch=ep, rank=rank, pseq=pseq, pep=pep,
        parent_rank=obs['rank'], parent_score=obs['score'],
        parent_coldkey=obs['coldkey'], parent_metrics=list(obs['metrics'].values()),
    ))

print(f"parents with full metrics: {len(parent_rows)}")

# Compare parent metrics vs field
pm = np.array([r['parent_metrics'] for r in parent_rows])
print(f"\n=== PARENT METRIC PROFILE vs FIELD (field n={len(field_metrics)}) ===")
print(f"{'metric':28s} {'parent mean':>11s} {'field mean':>10s} {'field med':>10s} {'z of parent mean':>15s}")
fmean = field_metrics.mean(axis=0); fstd = field_metrics.std(axis=0)
fmed = np.median(field_metrics, axis=0)
for i, name in enumerate(METRICS):
    pmean = pm[:,i].mean()
    z = (pmean - fmean[i]) / (fstd[i] or 1)
    print(f"{name:28s} {pmean:11.4f} {fmean[i]:10.4f} {fmed[i]:10.4f} {z:15.2f}")

# Key question: within the SAME rank band, do parents differ from non-parents?
print(f"\n=== parent rank distribution ===")
pranks = [r['parent_rank'] for r in parent_rows]
print(f"median={statistics.median(pranks)} mean={statistics.mean(pranks):.1f}")
for lo,hi in [(1,2),(3,5),(6,10),(11,20),(21,53)]:
    n = sum(1 for x in pranks if lo<=x<=hi)
    print(f"  rank {lo}-{hi}: {n} ({100*n/len(pranks):.0f}%)")

# Save
json.dump(dict(parent_rows=parent_rows), open('parent_metric_rows.json','w'))
print('\nsaved parent_metric_rows.json')

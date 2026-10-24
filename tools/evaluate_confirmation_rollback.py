"""Allocation-only rollback comparison. No model calls or live writes."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nanobody_opt_v2 import confirmation_policy as policy
from nanobody_opt_v2.compute_budget import TimingModel
CONTROL=dict(candidates=64,rounds=20,rounds_per_stage=5,keep_fraction=.65,minimum_keep=20)
plan=policy.plan(**CONTROL,gpus=8,model=TimingModel(9.43,5.52,0))
variants={'old_allocation':(plan['control_schedule'],False),'current_allocation':(plan['schedule'],True),'broad_without_random_sample':(plan['schedule'],False)}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--trials-per-seed',type=int,default=1000)
args=parser.parse_args()
if args.trials_per_seed<2:parser.error('at least two trials per seed required')
results={}
for path in sorted((ROOT/'tests/fixtures/confirmation_replay').glob('*.json')):
    x=np.asarray(json.loads(path.read_text())['folds'],dtype=float)
    scores=[]
    for seed in [701289,901233,503711]:
        rng=np.random.default_rng(seed)
        for trial in range(args.trials_per_seed):
            y=np.take_along_axis(x,np.argsort(rng.random(x.shape),axis=1),axis=1)
            cumulative=y[:,:21].cumsum(axis=1);heldout=y[:,21:].mean(axis=1);order=np.argsort(y[:,0])
            row=[]
            for name,(schedule,rescue) in variants.items():
                selected,_=policy.shortlist([str(i) for i in order],width=schedule[0][0],metadata={},observations={},epoch=seed*args.trials_per_seed+trial,broadened=rescue)
                alive=np.asarray(selected,dtype=int);depth=1
                for stage,(width,rounds) in enumerate(schedule):
                    if stage:alive=alive[np.argsort(cumulative[alive,depth-1])[:width]]
                    depth+=rounds
                final=alive[np.argsort(cumulative[alive,depth-1])[:13]]
                row.append(float(heldout[final].mean()))
            scores.append(row)
    a=np.asarray(scores);report={'candidates':len(x),'training_folds':21,'heldout_folds':x.shape[1]-21,'trials':len(a),'variants':{}}
    for i,name in enumerate(variants):
        delta=a[:,i]-a[:,0];se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
        report['variants'][name]={'heldout_mean':float(a[:,i].mean()),'paired_delta_vs_old':float(delta.mean()),'monte_carlo_se':se,'conditional_mc_interval_95':[float(delta.mean()-1.96*se),float(delta.mean()+1.96*se)],'better_trial_fraction':float(np.mean(delta<0)),'repeat_evaluations':policy.evaluation_count(variants[name][0])}
    results[path.stem]=report
args.output.write_text(json.dumps(results,indent=2))
print(json.dumps(results))

"""Theory and same-fit checks of SI Sections A, E.5 and F.5, and the data of Figure 2b-c.

Algorithms: 06_methods/code/{concordance,validate_methods}.py.
Seeds are 20260923 and 20261000 + training draw.
Outputs go to the directory given by --output.
"""
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT','1')
from pathlib import Path
import argparse,hashlib,itertools,json,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'06_methods/code'))
import validate_methods as vm
from concordance import pair_bound,comparable_pairs

def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def export_expanded(out,result):
    h=lambda x:float(x>0)+.5*float(x==0)
    scalar=[]
    for d,e in itertools.product([-2.,-1.,0.,1.,2.],repeat=2):
        r=pair_bound([d],[e]);actual=abs(h(d+e)-h(d));sharp=max(abs(h(d+abs(e))-h(d)),abs(h(d-abs(e))-h(d)))
        assert actual<=r['tie_aware_bound'] and sharp==r['tie_aware_bound']
        scalar.append(dict(D=d,E=e,actual_change=actual,sharp_sign_choice=sharp,**r))
    dump(out/'scalar_25_cases.json',{'cases':scalar,'sharpness_example':{'D':-1,'E':2,'change':1}})
    rng=np.random.default_rng(20260923);survival=[]
    for k in range(200):
        t=rng.integers(1,20,40);events=rng.binomial(1,.65,40);base=rng.integers(-4,5,40).astype(float);added=rng.integers(-2,3,40).astype(float);weight=rng.uniform(.1,3,40)
        i,j=comparable_pairs(t,events);r=pair_bound(base[i]-base[j],added[i]-added[j],weight[i])
        pairs=[(a,b)for a in range(40)for b in range(40)if events[a] and t[a]<t[b]];den=sum(weight[a]for a,b in pairs)
        C=lambda s:sum(weight[a]*h(s[a]-s[b])for a,b in pairs)/den
        old,new=C(base),C(base+added)
        assert abs(new-old-r['signed_change'])<2e-14
        survival.append({'case':k,'time':t.tolist(),'event':events.tolist(),'base':base.tolist(),'added':added.tolist(),'weight':weight.tolist(),'C_base':old,'C_full':new,'direct_change':new-old,**r})
    dump(out/'weighted_survival_200_cases.json',{'seed':20260923,'cases':survival})
    # Section 2.1: independently check the five-pair decomposition.
    D=np.array([1,-1,1,3,0.]);E=np.array([-2,2,2,1,1.])
    change=np.array([h(d+e)-h(d) for d,e in zip(D,E)])
    example=pair_bound(D,E)
    example.update(P=float(np.maximum(change,0).mean()),N=float(np.maximum(-change,0).mean()))
    example['cancellation']=example['absolute_pair_change']-abs(example['signed_change'])
    example['slack']=example['tie_aware_bound']-example['absolute_pair_change']
    for key,expected in [('signed_change',.1),('absolute_pair_change',.5),('tie_aware_bound',.7),('inclusive_bound',.8),('P',.3),('N',.2),('cancellation',.4),('slack',.2)]:
        assert abs(example[key]-expected)<1e-14,key
    dump(out/'five_pair_example.json',example)
    construction={**result['construction'],'seed_formula':'20261000 + draw','draws':[{**r,'permutation_seed':20261000+r['seed']}for r in result['construction']['draws']]}
    dump(out/'label_blind_construction.json',construction)
    pd.DataFrame(result['application']['ranking_checks']).to_csv(out/'complete_rank_diagnostics.csv',index=False)
    rows=[]
    for record in result['application']['bounds']:
        for point in record['budget_curve']:
            rows.append({'domain':record['domain'],'draw':record['seed'],'design':record['design'],**point})
    df=pd.DataFrame(rows);df.to_csv(out/'kernel_budget_points.csv',index=False)
    selected=df[df.domain.isin(['PHH','external'])&df.design.eq('matched_centred')]
    assert len(selected)==140
    selected.to_csv(out/'figure2bc_bound_points.csv',index=False)
    return {'scalar_cases':len(scalar),'weighted_survival_cases':len(survival),
        'construction_draws':len(construction['draws']),'maximum_coordinate_difference':max(r['target_coordinate_error']for r in construction['draws']),
        'complete_rank_rows':len(result['application']['ranking_checks']),'figure2bc_points':len(selected),'kernel_budget_points':len(df)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    result=vm.run();saved=json.loads((ROOT/'06_methods/results/methods_validation.json').read_text());vm.compare_result(result,saved)
    frozen=json.loads((ROOT/'06_methods/provenance/frozen_experiment_hashes.json').read_text())
    for name,sha in frozen['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,'Changed frozen input: '+name
    dump(out/'methods_validation_reproduced.json',result)
    summary=export_expanded(out,result)
    summary.update({'status':'PASS','saved_result_comparison':'Counts/identities exact; floats rtol=1e-10, atol=1e-12',
        'hash_checked_files':len(frozen['files']),'scope':'Theory and same-fit checks recomputed from the saved component scores; no model is refitted.',
        'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}})
    dump(out/'REPLAY_SUMMARY.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

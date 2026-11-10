"""Theory and same-fit checks of SI Sections A, C, D, G.5 and H.5, and the data of Figure 2B-C.

Algorithms: 06_methods/code/{concordance,validate_methods}.py.
Seeds are 20260923 and 20261000 + source draw.
Outputs go to the directory given by --output.
"""
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT','1')
from pathlib import Path
import argparse,hashlib,itertools,json,sys
import numpy as np,pandas as pd
from scipy.stats import levy_stable
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
    # Store the actual common random inputs as well as the seed and algorithm.
    rng=np.random.default_rng(20260923);arrays={};max_error=0.
    for alpha in [.7,1.,1.5,2.]:
        e1=levy_stable.rvs(alpha,0,size=200000,random_state=rng);e2=levy_stable.rvs(alpha,0,size=200000,random_state=rng)
        tag=str(alpha).replace('.','p');arrays['alpha_'+tag+'_e1']=e1;arrays['alpha_'+tag+'_e2']=e2
        for row in result['dilution']['cells']:
            if row['alpha']==alpha:
                auc=float(np.mean(row['mu']+e1+row['tau']*e2>0));assert auc==row['simulated_auc']
                max_error=max(max_error,abs(auc-row['exact_auc']))
    np.savez_compressed(out/'stable_mc_inputs.npz',**arrays)
    dilution={**result['dilution'],'max_absolute_error':max_error,
        'design_note':'For each alpha, one independent pair of length-200000 noise vectors is reused across the 15 location/scale settings. Sixty cells do not imply sixty independent Monte Carlo experiments.'}
    dump(out/'stable_dilution.json',dilution);pd.DataFrame(dilution['cells']).to_csv(out/'stable_dilution_60_cells.csv',index=False)
    ranks=[]
    for values in [[0,1,2,3],[0,0,1,1]]:
        for permutation in itertools.permutations(values):
            p=(1+sum(x>=permutation[0]for x in permutation[1:]))/4
            ranks.append({'values':values,'permutation':permutation,'p':p})
    dump(out/'exchangeability_enumeration.json',{'summary':result['exchangeability'],'permutations':ranks,
        'tie_note':'Enumerate all 24 permutations of labelled positions; tied value arrangements therefore occur with their correct multiplicities.'})
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
    selected.to_csv(out/'figure2BC_budget_points.csv',index=False)
    dense=[]
    for alpha in [.7,1.,1.5,2.]:
        for mu in [.8,-.8]:
            for tau in np.linspace(0,4,121):
                dense.append({'alpha':alpha,'mu':mu,'sigma':1.,'tau':float(tau),'auc':float(levy_stable.cdf(mu/(1+tau**alpha)**(1/alpha),alpha,0))})
    pd.DataFrame(dense).to_csv(out/'figureS1A_analytic_curves.csv',index=False)
    pd.DataFrame(result['dilution']['counterexamples']).to_csv(out/'figureS1B_counterexamples.csv',index=False)
    return {'scalar_cases':len(scalar),'weighted_survival_cases':len(survival),'stable_cells':len(dilution['cells']),
        'maximum_stable_error':max_error,'rank_permutations':len(ranks),'rank_summary_rows':len(result['exchangeability']['exact_rank_checks']),
        'construction_draws':len(construction['draws']),'maximum_coordinate_difference':max(r['target_coordinate_error']for r in construction['draws']),
        'complete_rank_rows':len(result['application']['ranking_checks']),'figure2BC_points':len(selected),'kernel_budget_points':len(df),'figureS1A_points':len(dense)}

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

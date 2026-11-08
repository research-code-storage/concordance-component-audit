"""ADEMP checks of cancellation, ties, and bounded reference perturbations."""
import json,itertools,sys,argparse
from pathlib import Path
import numpy as np,pandas as pd
from reference import BASE,ROOT
sys.path.insert(0,str(ROOT/'06_methods/code'))
from concordance import pair_bound

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def H(x):return (x>0).astype(float)+.5*(x==0)
def main(output=None):
    out=Path(output)if output else BASE/'results/simulation';out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(20260927)
    rows=[];summaries=[];violations=0
    for n,strength,kind,dist in itertools.product([50,200],[.01,.1,.5,1.],['aligned','independent','opposed'],['normal','t3','rounded_normal']):
        cell=[];y=np.r_[np.zeros(n//2),np.ones(n//2)];pos=y==1;neg=y==0
        for rep in range(200):
            seed=int(rng.integers(0,2**32-1));g=np.random.default_rng(seed)
            error=g.standard_t(3,n)/np.sqrt(3)if dist=='t3'else g.normal(size=n)
            base=.6*y+error
            if dist=='rounded_normal':base=np.round(base)
            add=strength*(g.normal(size=n)+(.6*y if kind=='aligned'else-.6*y if kind=='opposed'else 0))
            D=(base[pos,None]-base[None,neg]).ravel();E=(add[pos,None]-add[None,neg]).ravel();r=pair_bound(D,E)
            # The uncertainty radius is specified and truly contains the generated perturbation.
            radius=.05*(1+abs(g.normal(size=n)));delta=g.uniform(-1,1,n)*radius
            R=(radius[pos,None]+radius[None,neg]).ravel();dp=(delta[pos,None]-delta[None,neg]).ravel()
            robust=float(np.mean(abs(D)<=abs(E)+R));actual_perturbed=abs(float((H(D+dp+E)-H(D+dp)).mean()))
            v=int(actual_perturbed>robust+1e-14);violations+=v
            maximum_multiplier_error=0.
            for t in [0,.25,.5,1,2,4,8]:
                bound=float(np.mean(abs(D)<=t*abs(E)))
                actual=abs(float((H(D+t*E)-H(D)).mean()))
                maximum_multiplier_error=max(maximum_multiplier_error,actual-bound)
            assert maximum_multiplier_error<=1e-14 and v==0
            row={'n':n,'strength':strength,'component':kind,'distribution':dist,'replicate':rep,'seed':seed,
                **r,'cancellation':r['absolute_pair_change']-abs(r['signed_change']),
                'robust_bound':robust,'actual_under_reference_perturbation':actual_perturbed,
                'robust_violation':v,'maximum_multiplier_violation':maximum_multiplier_error}
            rows.append(row);cell.append(row)
        summary={k:cell[0][k]for k in ['n','strength','component','distribution']};summary['replicates']=200
        for key in ['signed_change','absolute_pair_change','tie_aware_bound','inclusive_bound','cancellation','robust_bound','actual_under_reference_perturbation']:
            v=np.array([r[key]for r in cell]);summary[key]={'mean':float(v.mean()),'mcse':float(v.std(ddof=1)/np.sqrt(200))}
        summaries.append(summary)
    pd.DataFrame(rows).to_csv(out/'all_14400_replicates.csv',index=False)
    dump(out/'simulation.json',{'status':'PASS','seed':20260927,'cells':len(summaries),'replicates':len(rows),
        'bound_violations':violations,'summaries':summaries,
        'reference_bound':'If |delta_i|<=r_i, pair order cannot change upon adding t E when |D| > |t| |E| + r_i+r_j. The envelope fixes pairs/weights and is not a population CI.',
        'scope':'Independent generated datasets per replicate; stored replicate seeds reproduce all sample inputs.'})
    print('PASS:',len(rows),'replicates in',len(summaries),'cells; no bound violations.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();main(a.output)

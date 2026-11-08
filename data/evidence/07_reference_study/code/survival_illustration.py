"""Single held-out Cox illustration on the public NCCTG lung dataset."""
import json,hashlib,sys,argparse
from pathlib import Path
import numpy as np,pandas as pd
from statsmodels.duration.hazard_regression import PHReg
from reference import BASE,ROOT
sys.path.insert(0,str(ROOT/'06_methods/code'))
from concordance import comparable_pairs,pair_bound

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def H(x):return (x>0).astype(float)+.5*(x==0)
def main(output=None):
    path=BASE/'data/lung.csv';source=json.loads(path.with_suffix('.source.json').read_text());raw=pd.read_csv(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
    assert len(raw)==228 and set(raw.status.unique())=={0,1} and raw.status.sum()==165
    assert not raw.duplicated().any()
    # The snapshot codes status as 0 = censored, 1 = death (survival::lung status minus one).
    cols=['age','sex','ph.ecog'];f=raw.dropna(subset=['time','status']+cols).copy();f['subject_id']=f.index+1
    assert (f.time>0).all();rng=np.random.default_rng(20260928);order=rng.permutation(len(f));cut=int(.7*len(f));tr,te=order[:cut],order[cut:]
    X=f[cols].to_numpy();mean=X[tr].mean(0);scale=X[tr].std(0);Z=(X-mean)/scale
    model=PHReg(f.iloc[tr].time.to_numpy(),Z[tr],status=f.iloc[tr].status.to_numpy(),ties='efron')
    fit=model.fit(disp=False,maxiter=1000);gradient=model.score(fit.params);assert abs(gradient).max()<1e-6
    base=Z[te,:2]@fit.params[:2];added=Z[te,2]*fit.params[2];t=f.iloc[te].time.to_numpy();e=f.iloc[te].status.to_numpy()
    i,j=comparable_pairs(t,e);D=base[i]-base[j];E=added[i]-added[j];r=pair_bound(D,E)
    c0=float(H(D).mean());c1=float(H(D+E).mean());assert abs(c1-c0)<=r['tie_aware_bound']+1e-14
    # Independent Python enumeration includes each comparable ordered pair once.
    pairs=[(a,b)for a in range(len(te))for b in range(len(te))if e[a]==1 and t[a]<t[b]]
    oracle=lambda v:sum(float(v[a]>v[b])+.5*float(v[a]==v[b])for a,b in pairs)/len(pairs)
    assert abs(oracle(base)-c0)<1e-14 and abs(oracle(base+added)-c1)<1e-14
    out=Path(output)if output else BASE/'results/survival';out.mkdir(parents=True,exist_ok=True)
    result={'status':'PASS','data_source':source,'data_rows':len(raw),'complete_cases':len(f),
        'excluded_subject_ids':(raw.index[~raw.index.isin(f.index)]+1).tolist(),'seed':20260928,'train':len(tr),'test':len(te),
        'train_subject_ids':f.iloc[tr].subject_id.tolist(),'test_subject_ids':f.iloc[te].subject_id.tolist(),
        'event_encoding':'0 = censored, 1 = death (survival::lung status minus one).',
        'training_means':mean.tolist(),'training_scales':scale.tolist(),'coefficients':fit.params.tolist(),'features':cols,
        'maximum_score_equation_residual':float(abs(gradient).max()),'base_C':c0,'full_C':c1,'pair_bound':r,
        'interpretation':'Fixed-pair concordance on the held-out patients, conditional on the fitted Cox model.'}
    dump(out/'survival_illustration.json',result)
    table=f.iloc[te][['subject_id','time','status']+cols].copy();table['age_sex_component']=base;table['ecog_component']=added;table['full_score']=base+added
    table.to_csv(out/'heldout_scores.csv',index=False);f.to_csv(out/'complete_case_inputs.csv',index=False)
    pd.DataFrame({'earlier_subject_id':f.iloc[te].subject_id.to_numpy()[i],'later_subject_id':f.iloc[te].subject_id.to_numpy()[j],'base_difference':D,'added_difference':E}).to_csv(out/'comparable_pairs.csv',index=False)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();main(a.output)

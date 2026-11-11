"""Reproduce the theory checks and same-fit audits; never refit outcome models."""
from pathlib import Path
import argparse, hashlib, itertools, json, sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from concordance import pair_bound, comparable_pairs

ROOT = Path(__file__).resolve().parents[2]
SAVED = ROOT / '05_reconstruction/results/experiments'
sys.path.insert(0, str(ROOT / '05_reconstruction/code'))
from reconstructed_features import load, origins, fit_coordinates, transform


def check_bounds():
    # Independent scalar oracle includes exact ties and sharpness constructions.
    def h(x): return 1. if x > 0 else (0. if x < 0 else .5)
    grid = [-2., -1., 0., 1., 2.]
    count = 0
    for d, e in itertools.product(grid, repeat=2):
        r = pair_bound([d], [e]); exact = abs(h(d+e)-h(d))
        assert exact <= r['tie_aware_bound'] <= r['inclusive_bound']
        # Maximise over the two possible signs of the known absolute perturbation.
        sharp = max(abs(h(d+abs(e))-h(d)), abs(h(d-abs(e))-h(d)))
        assert sharp == r['tie_aware_bound']; count += 1
    assert pair_bound([-1], [2])['signed_change'] == 1
    assert pair_bound([0], [0])['tie_aware_bound'] == 0
    rng = np.random.default_rng(20260923); random_cases = 200
    for _ in range(random_cases):
        t = rng.integers(1, 20, 40); event = rng.binomial(1, .65, 40)
        base = rng.integers(-4, 5, 40).astype(float); add = rng.integers(-2, 3, 40).astype(float)
        weight = rng.uniform(.1, 3, 40)  # Fixed event-subject weights, not refitted with scores.
        i, j = comparable_pairs(t, event)
        r = pair_bound(base[i]-base[j], add[i]-add[j], weight[i])
        pairs = [(a,b) for a in range(40) for b in range(40) if event[a] and t[a]<t[b]]
        denominator = sum(weight[a] for a,b in pairs)
        def c(s): return sum(weight[a]*h(s[a]-s[b]) for a,b in pairs)/denominator
        exact = c(base+add)-c(base)
        assert abs(exact-r['signed_change']) < 2e-14
        assert abs(exact) <= r['tie_aware_bound']+2e-14
        assert abs((2*c(base+add)-1)-(2*c(base)-1)) <= 2*r['tie_aware_bound']+4e-14
    rejected = 0
    for d,e,w in [([1],[2],[-1]),([1],[2],[0]),([np.nan],[1],[1])]:
        try: pair_bound(d,e,w)
        except ValueError: rejected += 1
    assert rejected == 3
    return {'scalar_tie_cases': count, 'weighted_survival_cases': random_cases,
            'invalid_inputs_rejected': rejected, 'unit_coefficient_sharp': True, 'status': 'PASS'}


def aggregate(d, key):
    a=pd.DataFrame(d[key]);a['compound']=d['compound']
    return a.groupby('compound',sort=True).mean().to_numpy()


def application():
    records=[]; diagnostics=[]; hashes={}
    for domain in ['PHH','external','source']:
        for seed in range(10):
            path=SAVED/f'n60_seed{seed}_{domain}_scores.npz'
            hashes[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path,allow_pickle=False) as a:
                meta=pd.DataFrame({'compound':a['compound'],'label':a['label']}).groupby('compound',sort=True).label.first()
                labels=meta.to_numpy();pos=np.flatnonzero(labels==1);neg=np.flatnonzero(labels==0)
                full,lin,kernel,fixed=[aggregate(a,key) for key in ['full','linear','kernel','diagnostic']]
                names=a['arm_names'].tolist();dn=a['diagnostic_names'].tolist()
                for design in dn:
                    idx=names.index(design+'__softplus_original');l=lin[:,idx];g=kernel[:,idx]
                    if domain=='source' and design!='matched_centred':continue
                    D=(l[pos,None]-l[None,neg]).ravel();E=(g[pos,None]-g[None,neg]).ravel()
                    r=pair_bound(D,E)
                    # Direct full score agrees even though its saved intercept is omitted in D+E.
                    actual=(full[pos,None,idx]-full[None,neg,idx]).ravel()
                    assert np.array_equal(np.sign(actual),np.sign(D+E))
                    curves=[dict(t=t,**pair_bound(D,t*E))for t in [0.,.25,.5,1.,2.,4.,8.]]
                    if domain!='source':records.append(dict(domain=domain,seed=seed,design=design,**r,budget_curve=curves))
                    if design=='matched_centred':
                        f=full[:,idx];s=fixed[:,dn.index(design)]
                        all_i,all_j=np.triu_indices(len(f),1)
                        sign=lambda v,i,j:np.sign(v[i]-v[j])
                        h=lambda v:np.mean((v>0)+.5*(v==0))
                        fd=(f[pos,None]-f[None,neg]).ravel()
                        diagnostics.append(dict(domain=domain,seed=seed,
                            fixed_sum_all_pair_disagreements=int(np.count_nonzero(sign(f,all_i,all_j)!=sign(s,all_i,all_j))),
                            original_auc=float(h(fd))))
    return dict(bounds=records,ranking_checks=diagnostics,input_hashes=hashes,
                conditioning='Saved n60 fits; no outcome model is retrained; moments are descriptive dependent-pair summaries.',status='PASS')


def construction():
    domains,source_comp,held=load(with_embeddings=False);source=domains['source'];O,_=origins(source,True);records=[]
    for seed in range(10):
        run=json.loads((SAVED/f'n60_seed{seed}.json').read_text());compounds=run['train_compounds']
        base=fit_coordinates(source,O,compounds,'matched_centred');altered=source.copy()
        # Change annotations at compound level; selected IDs/references are frozen.
        labs=source.groupby('ck').labels.first();rng=np.random.default_rng(20261000+seed)
        mapping=dict(zip(labs.index,rng.permutation(labs.to_numpy())))
        altered['labels']=altered.ck.map(mapping)
        other=fit_coordinates(altered,O,compounds,'matched_centred')
        numeric=['centre','scale','centroids','q15','q85']
        error=max(float(np.max(np.abs(np.asarray(base[k])-np.asarray(other[k]))))for k in numeric)
        saved_error=max(float(np.max(np.abs(np.asarray(base[k])-np.asarray(run['coordinate_assets']['matched_centred'][k]))))for k in numeric)
        assert error<1e-12 and saved_error<1e-10
        assert base['retained_ids']==other['retained_ids']==run['coordinate_assets']['matched_centred']['retained_ids']
        assert not (set(compounds)&set(held))
        coordinate_error=0.
        for domain in ['PHH','external']:
            frame=domains[domain];origin,_=origins(frame,True)
            valid=np.isfinite(origin).all(1)
            a=transform(frame,origin,base)[valid];b=transform(frame,origin,other)[valid]
            coordinate_error=max(coordinate_error,float(np.max(np.abs(a-b))))
        assert coordinate_error<1e-12
        records.append(dict(seed=seed,changed_label_profiles=int((altered.labels!=source.labels).sum()),
                            asset_error=error,saved_asset_error=saved_error,target_coordinate_error=coordinate_error))
    return dict(draws=records,scope='Coordinates conditional on selected source compounds, fixed reference IDs, keyword rule and random seeds.',status='PASS')


def run():
    out={}
    with threadpool_limits(limits=1):
        for name,func in [('bounds',check_bounds),('application',application),('construction',construction)]:
            out[name]=func();print(name,'PASS',flush=True)
    out['status']='PASS'
    return out


def compare_result(actual, expected, path='root'):
    """Count/identity checks are exact; floating calculations allow roundoff."""
    if isinstance(expected, dict):
        assert actual.keys()==expected.keys(),path
        for k in expected: compare_result(actual[k],expected[k],path+'/'+k)
    elif isinstance(expected, list):
        assert len(actual)==len(expected),path
        for j,(a,b) in enumerate(zip(actual,expected)):compare_result(a,b,path+'/'+str(j))
    elif isinstance(expected, float):
        assert np.isclose(actual,expected,rtol=1e-10,atol=1e-12),path
    else:
        assert actual==expected,path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--verify',action='store_true');a=p.parse_args()
    result=run();canonical=ROOT/'06_methods/results/methods_validation.json'
    if a.verify or a.output is None:
        saved=json.loads(canonical.read_text());compare_result(result,saved)
        frozen=json.loads((ROOT/'06_methods/provenance/frozen_experiment_hashes.json').read_text())
        for name,sha in frozen['files'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,'Changed frozen experiment input: '+name
        print('Frozen scientific files verified:',len(frozen['files']),flush=True)
        print('All new method results reproduced within stated floating-point tolerances.',flush=True)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
